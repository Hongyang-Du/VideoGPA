"""
Wan2.2 TI2V DPO - Step 2: Encode videos to latents + encode text/image conditions

Input:  scored metadata JSON (with consistency_score & motion_norm)
Output: metadata JSON with latent_path & condition_path added

Each group gets:
  - condition: {encoder_hidden_states: [L, 4096], image_latent: [z_dim, 1, H_z, W_z]}
  - per-video latent: [z_dim, F_z, H_z, W_z]
"""

import os
import sys
import json
import torch
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import time
from decord import VideoReader, cpu

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

wan_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'Wan2.2'))
sys.path.insert(0, wan_path)

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON {path}: {e}")
        return None


def safe_save_json(path, data):
    temp_path = str(path) + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_path, path)
    except Exception as e:
        logging.error(f"Save failed: {e}")


def load_video_frames(video_path: str, num_frames: int, device: torch.device) -> torch.Tensor:
    """Load video and return tensor [3, F, H, W] normalized to [-1, 1]."""
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()  # [F, H, W, 3], uint8
    frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0
    frames = frames.permute(3, 0, 1, 2).to(device)  # [3, F, H, W]
    return frames


def load_image_tensor(image_path: str, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    """Load image and return tensor [3, 1, H, W] normalized to [-1, 1]."""
    img = Image.open(image_path).convert("RGB")
    scale = max(target_w / img.width, target_h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1 = (img.width - target_w) // 2
    y1 = (img.height - target_h) // 2
    img = img.crop((x1, y1, x1 + target_w, y1 + target_h))
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)  # [3, H, W] in [-1, 1]
    return img_tensor.unsqueeze(1).to(device)  # [3, 1, H, W]


def gpu_worker(gpu_id, groups_chunk, worker_idx, base_path, model_path, latent_root, num_frames):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    time.sleep(worker_idx * 2)

    config = WAN_CONFIGS['ti2v-5B']
    output_base = Path(base_path) / latent_root
    output_base.mkdir(parents=True, exist_ok=True)

    try:
        t5 = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=device,
            checkpoint_path=os.path.join(model_path, config.t5_checkpoint),
            tokenizer_path=os.path.join(model_path, config.t5_tokenizer),
        )
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(model_path, config.vae_checkpoint),
            device=device,
        )
        logging.info(f"Worker-{worker_idx} (GPU {gpu_id}) ready.")
    except Exception as e:
        logging.error(f"Worker-{worker_idx} init failed: {e}")
        return []

    processed_groups = []

    for group in tqdm(groups_chunk, desc=f"GPU-{gpu_id}", position=worker_idx, leave=False):
        group_id = group.get("group_id")
        text_prompt = group.get("text_prompt", group.get("prompt", ""))
        image_path = group.get("image_path", group.get("input_image_path", ""))

        if not text_prompt or not image_path:
            continue

        # Resolve image path
        if not os.path.isabs(image_path):
            image_path = os.path.join(base_path, image_path)

        try:
            # Encode condition (text + image) - shared per group
            cond_path = output_base / f"cond_{group_id}.pt"

            if cond_path.exists():
                cond_rel = str(cond_path.relative_to(base_path))
            else:
                condition_data = {}

                # Text embedding via T5
                context = t5([text_prompt], device)
                condition_data["encoder_hidden_states"] = context[0].cpu()  # [L, 4096]

                # Image -> VAE latent
                first_video = group.get('videos', [{}])[0]
                first_video_path = Path(base_path) / first_video.get('video_path', '')
                if first_video_path.exists():
                    vr = VideoReader(str(first_video_path), ctx=cpu(0))
                    h, w = vr[0].shape[:2]
                else:
                    h, w = 704, 1280  # fallback

                img_tensor = load_image_tensor(image_path, h, w, device)
                with torch.no_grad():
                    img_latent = vae.encode([img_tensor])
                condition_data["image_latent"] = img_latent[0].cpu()

                torch.save(condition_data, cond_path)
                cond_rel = str(cond_path.relative_to(base_path))

            # Encode each video latent
            video_entries = []
            for entry in group.get('videos', []):
                rel_v_path = entry.get("video_path")
                video_full = Path(base_path) / rel_v_path
                video_name = Path(rel_v_path).stem
                lat_path = output_base / f"latent_{group_id}_{video_name}.pt"

                if lat_path.exists():
                    lat_rel = str(lat_path.relative_to(base_path))
                else:
                    try:
                        video_tensor = load_video_frames(str(video_full), num_frames, device)
                        with torch.no_grad():
                            latent = vae.encode([video_tensor])
                        torch.save(latent[0].cpu(), lat_path)
                        lat_rel = str(lat_path.relative_to(base_path))
                    except Exception as e:
                        logging.error(f"Encode error {rel_v_path}: {e}")
                        continue

                entry["latent_path"] = lat_rel
                entry["condition_path"] = cond_rel
                video_entries.append(entry)

            if video_entries:
                group['videos'] = video_entries
                processed_groups.append(group)

        except Exception as e:
            logging.error(f"Group error {group_id}: {e}")

    del t5, vae
    torch.cuda.empty_cache()
    return processed_groups


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V: Encode videos to latents")
    parser.add_argument("--base_path", type=str, required=True, help="Base dataset path")
    parser.add_argument("--model_path", type=str, required=True, help="Wan2.2-TI2V-5B model path")
    parser.add_argument("--input_json", type=str, required=True, help="Scored metadata JSON")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON with latent paths")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1], help="GPU IDs")
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--latent_root", type=str, default="wan_dpo_latents",
                        help="Subdirectory under base_path for latent storage")
    parser.add_argument("--num_frames", type=int, default=81, help="4n+1 for Wan2.2 VAE")
    args = parser.parse_args()

    # Load input data
    input_data = safe_load_json(args.input_json)
    if not input_data:
        logging.error(f"Cannot load {args.input_json}")
        return

    if isinstance(input_data, dict):
        all_groups = input_data.get('groups', [])
    elif isinstance(input_data, list):
        all_groups = input_data
    else:
        logging.error("Unsupported JSON format")
        return

    logging.info(f"Loaded {len(all_groups)} groups")

    num_gpus = len(args.gpus)
    total_workers = num_gpus * args.workers_per_gpu
    chunks = [all_groups[i::total_workers] for i in range(total_workers)]

    logging.info(f"Encoding across {num_gpus} GPUs ({args.workers_per_gpu} workers/GPU)")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=total_workers) as pool:
        results = pool.starmap(gpu_worker, [
            (args.gpus[i // args.workers_per_gpu], chunks[i], i,
             args.base_path, args.model_path, args.latent_root, args.num_frames)
            for i in range(total_workers)
        ])

    final_groups = [g for sub in results for g in sub]
    if final_groups:
        safe_save_json(args.output_json, {'groups': final_groups})
        logging.info(f"Done! Encoded {len(final_groups)} groups -> {args.output_json}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {(time.time() - start) / 3600:.2f} hours")
