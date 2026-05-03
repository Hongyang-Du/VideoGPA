import os
import json
import torch
import torch.multiprocessing as mp
from pathlib import Path
from peft import PeftModel
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

# ================= Configuration =================
def parse_int_list_env(name, default):
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


ENV_MODE = os.getenv('RUN_MODE', 'dpo')
ENV_LORA_PATH = os.getenv('RUN_LORA_PATH', '/workspace/workspace/outputs/cogvideo_dpo_v3/final_lora')
ENV_OUTPUT_DIR = os.getenv('RUN_OUTPUT_DIR', '/workspace/VideoGPA/benchmark_videos/dpo_v3')
ENV_PROMPT_JSON = os.getenv('PROMPT_JSON', '/workspace/VideoGPA/dl3dv_video_captions/captions_1K.json')
ENV_DL3DV_BASE_DIR = os.getenv('DL3DV_BASE_DIR', '/datasets/DL3DV-10K')
ENV_DEVICES = parse_int_list_env('RUN_DEVICES', [0])
ENV_NUM_PROMPTS = int(os.getenv('RUN_NUM_PROMPTS', '100'))
ENV_SEEDS = parse_int_list_env('RUN_SEEDS', [456])

CONFIG = {
    'devices': ENV_DEVICES,
    'mode': ENV_MODE,
    'weight_list': [1.0],
    'base_model': 'THUDM/CogVideoX-5B-I2V',
    'lora_path': ENV_LORA_PATH,
    'prompt_json': ENV_PROMPT_JSON,
    'dl3dv_base_dir': ENV_DL3DV_BASE_DIR,
    'output_dir': ENV_OUTPUT_DIR,
    'num_prompts': ENV_NUM_PROMPTS,
    'seeds_per_prompt': ENV_SEEDS,
    'num_inference_steps': 50,
    'guidance_scale': 6.0,
    'fps': 8,
}


def extract_pure_hash_from_json_key(json_key):
    """
    Extract the pure hash string from a JSON key (format: 1K/hash/images_8 → returns only the hash).
    """
    try:
        json_key = json_key.strip()
        key_parts = json_key.split("/")

        if len(key_parts) == 3:
            pure_hash = key_parts[1]
        else:
            pure_hash = json_key.replace("/", "_").replace("\\", "_")

        if not pure_hash:
            raise ValueError("Extracted hash string is empty, cannot use as storage folder")

        return pure_hash
    except Exception as e:
        raise RuntimeError(f"Failed to extract pure hash string: {str(e)}") from e


def find_dl3dv_first_frame(json_key, dl3dv_base_dir):
    """
    Find the first frame in a DL3DV frame directory, and also return the pure hash string.
    """
    try:
        json_key = json_key.strip()
        if not json_key:
            raise ValueError("JSON Key is empty, cannot construct path")

        dl3dv_root = Path(dl3dv_base_dir)
        target_folder = dl3dv_root / json_key

        if not target_folder.exists() or not target_folder.is_dir():
            raise FileNotFoundError(f"Frame folder not found under DL3DV directory: {target_folder}")

        print(f"🔍 Found complete frame folder: {target_folder}")

        standard_first_frame_name = "frame_00001.png"
        standard_first_frame_path = target_folder / standard_first_frame_name

        if not standard_first_frame_path.exists():
            raise FileNotFoundError(f"Standard first frame not found in frame folder: {standard_first_frame_name}")

        pure_hash = extract_pure_hash_from_json_key(json_key)
        print(f"✅ Found standard first frame, extracted pure hash: {pure_hash}")

        return standard_first_frame_path, pure_hash, target_folder
    except Exception as e:
        print(f"⚠️  Failed to find frame file: {e}")
        raise


def main():
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    dl3dv_base_dir = Path(CONFIG['dl3dv_base_dir'])
    if not dl3dv_base_dir.exists():
        print(f"❌ DL3DV root directory does not exist: {dl3dv_base_dir}")
        return

    try:
        with open(CONFIG['prompt_json'], 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        json_items = list(prompt_dict.items())
        print(f"📄 Successfully loaded JSON file with {len(json_items)} entries")
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        return

    selected_items = json_items[:CONFIG['num_prompts']]
    print(f"📋 Selected {len(selected_items)} entries for generation")

    num_gpus = len(CONFIG['devices'])
    chunks = [selected_items[i::num_gpus] for i in range(num_gpus)]

    if num_gpus == 1:
        worker(0, CONFIG['devices'][0], chunks[0], dl3dv_base_dir)
    else:
        mp.set_start_method('spawn', force=True)
        processes = []
        for rank, gpu_id in enumerate(CONFIG['devices']):
            p = mp.Process(
                target=worker,
                args=(rank, gpu_id, chunks[rank], dl3dv_base_dir)
            )
            p.start()
            processes.append(p)

        failed = False
        for p in processes:
            p.join()
            if p.exitcode != 0:
                failed = True
                print(f"❌ Subprocess exited abnormally, PID={p.pid}, exitcode={p.exitcode}")

        if failed:
            raise RuntimeError("At least one I2V generation subprocess failed, please check the logs.")

    print(f"\n🎉 All generation tasks for mode {CONFIG['mode']} completed!")
    print(f"📁 Video output directory: {CONFIG['output_dir']}/<pure_hash>/seed_xxx.mp4")


def worker(rank, gpu_id, json_items, dl3dv_base_dir):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mode = CONFIG['mode']
    weights_to_run = CONFIG['weight_list'] if mode in ['dpo', 'dpo_epipolar', 'sft'] else [0.0]

    print(f"🚀 GPU {gpu_id} | Starting generation | Mode: {mode} | Weights: {weights_to_run}")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        CONFIG['base_model'], torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if mode in ["dpo", "dpo_epipolar", "sft"] and CONFIG['lora_path']:
        print(f"🔧 GPU {gpu_id} loading LoRA weights: {CONFIG['lora_path']}")
        try:
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                CONFIG['lora_path'],
                adapter_name=f"{mode}_video",
                torch_device=str(device),
            )
            pipe.transformer.eval()
            pipe.transformer = pipe.transformer.to(device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"❌ GPU {gpu_id} LoRA loading failed: {e}")
            del pipe
            torch.cuda.empty_cache()
            return

    for json_key, text_prompt in json_items:
        text_prompt = text_prompt.strip()
        if not text_prompt:
            print(f"⚠️ GPU {gpu_id} skipping invalid entry (empty prompt): {json_key}")
            continue

        try:
            first_frame_path, pure_hash, _ = find_dl3dv_first_frame(json_key, dl3dv_base_dir)
        except Exception as e:
            print(f"❌ GPU {gpu_id} skipping entry: {e}")
            continue

        output_hash_dir = Path(CONFIG['output_dir']) / pure_hash
        output_hash_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Ensured CogVideo output directory exists: {output_hash_dir}")

        try:
            image = load_image(str(first_frame_path))
            image = image.resize((1080, 720))
        except Exception as e:
            print(f"❌ GPU {gpu_id} image loading failed: {e}")
            continue

        for lora_weight in weights_to_run:
            if mode in ["dpo", "dpo_epipolar", "sft"] and CONFIG['lora_path']:
                for _, module in pipe.transformer.named_modules():
                    if hasattr(module, "scaling") and f"{mode}_video" in module.scaling:
                        try:
                            module.scaling[f"{mode}_video"] = lora_weight * (
                                module.lora_alpha[f"{mode}_video"] / module.r[f"{mode}_video"]
                            )
                        except Exception as e:
                            print(f"⚠️ GPU {gpu_id} LoRA weight adjustment failed: {e}")
                            continue

            for seed in CONFIG['seeds_per_prompt']:
                if mode in ['dpo', 'dpo_epipolar', 'sft']:
                    video_filename = f"seed_{seed}_{mode}_w{lora_weight}.mp4"
                else:
                    video_filename = f"seed_{seed}_original.mp4"

                video_output_path = output_hash_dir / video_filename

                if video_output_path.exists():
                    print(f"ℹ️ GPU {gpu_id} video already exists, skipping: {video_filename}")
                    continue

                generator = torch.Generator(device=device).manual_seed(seed)
                try:
                    with torch.inference_mode():
                        print(f"⚙️ GPU {gpu_id} generating {pure_hash} - Seed {seed}...")
                        video = pipe(
                            prompt=text_prompt,
                            image=image,
                            num_inference_steps=CONFIG['num_inference_steps'],
                            guidance_scale=CONFIG['guidance_scale'],
                            generator=generator,
                        ).frames[0]

                    export_to_video(video, str(video_output_path), fps=CONFIG['fps'])
                    print(f"✅ GPU {gpu_id} saved: {video_filename}")
                except Exception as e:
                    print(f"❌ GPU {gpu_id} generation failed {pure_hash} - Seed {seed}: {e}")
                    continue

    del pipe
    torch.cuda.empty_cache()
    print(f"🔚 GPU {gpu_id} tasks completed, VRAM released")


if __name__ == "__main__":
    main()
