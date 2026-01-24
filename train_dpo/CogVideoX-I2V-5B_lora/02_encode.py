import os
import sys
import json
import torch
import logging
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from decord import VideoReader, cpu
from diffusers import CogVideoXImageToVideoPipeline
import numpy as np
from PIL import Image
import time

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================================
# ⭐ Configuration Area
# ============================================================================
GPUS = [0, 1]  # List of GPUs to use
MODEL_ID = "THUDM/CogVideoX-5B-I2V"

# [ANONYMIZED]: Replace with your actual paths
BASE_PATH = "/path/to/your/dataset/root" 

# your Input and Output paths
INPUT_JSON_PATH = f"{BASE_PATH}/path/to/your/input_file.json"
OUTPUT_JSON_PATH = f"{BASE_PATH}/meta_data.json"

# I2V Encoding Results Root Directory 
LATENT_ROOT =  f"{BASE_PATH}/i2v_latent"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Helper Functions
# ============================================================================

def safe_load_json(path):
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON {path}: {e}")
        return None

def safe_save_json(path, data):
    temp_path = str(path) + ".tmp"
    try:
        with open(temp_path, "w") as f: json.dump(data, f, indent=4)
        os.replace(temp_path, path)
    except Exception as e:
        logging.error(f"Save failed: {e}")

def load_video_frames_tensor(video_path: str, num_frames: int = 49, device: torch.device = None) -> torch.Tensor:
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    return frames.permute(3, 0, 1, 2).to(device)

def load_input_image_tensor(image_path: Path, device: torch.device = None) -> torch.Tensor:
    if not image_path.exists(): return None
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1).to(device)

# ============================================================================
# ⭐ Core Functionality: Encoding
# ============================================================================

def encode_group_condition(pipeline, prompt, image_path, group_id, sub_folder, output_base):
    """Generate common Condition for a Group and save it"""
    condition_data = {}
    
    # 1. Text Embedding
    if pipeline.text_encoder and pipeline.tokenizer:
        with torch.no_grad():
            text_input = pipeline.tokenizer(
                prompt, padding="max_length", max_length=226, truncation=True, return_tensors="pt"
            ).input_ids.to(pipeline.device)
            condition_data["encoder_hidden_states"] = pipeline.text_encoder(text_input)[0].squeeze(0).cpu()

    # 2. Image Embedding
    if image_path:
        img_full = Path(BASE_PATH) / image_path
        image_tensor = load_input_image_tensor(img_full, pipeline.device)
        if image_tensor is not None:
            condition_data["image_embeds"] = image_tensor.cpu()

    # Save path: BASE_PATH / LATENT_ROOT / sub_folder / cond_xxx.pt
    save_dir = output_base / sub_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cond_p = save_dir / f"cond_{group_id}.pt"
    torch.save(condition_data, cond_p)
    return str(cond_p)

def encode_video_latent(pipeline, video_rel_path, sub_folder, output_base, group_id):
    """Encode video Latent"""
    video_path = Path(BASE_PATH) / video_rel_path
    if not video_path.exists(): 
        raise FileNotFoundError(f"Missing video at: {video_path}")
    
    video_tensor = load_video_frames_tensor(str(video_path), 49, pipeline.device).unsqueeze(0)
    with torch.no_grad():
        latent = pipeline.vae.encode(video_tensor.to(pipeline.vae.dtype)).latent_dist.sample().squeeze(0).cpu()

    video_name = Path(video_rel_path).stem
    unique_id = f"{group_id}_{video_name}"
    
    # Save path: BASE_PATH / LATENT_ROOT / sub_folder / latent_xxx.pt
    save_dir = output_base / sub_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    lat_p = save_dir / f"latent_{unique_id}.pt"
    torch.save(latent, lat_p)
    return str(lat_p)

# ============================================================================
# GPU Worker Process
# ============================================================================

def gpu_worker(gpu_id, groups_chunk, worker_idx, sub_folder_name):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device) 
    time.sleep(worker_idx * 2)

    output_base = LATENT_ROOT

    try:
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        pipeline.vae.eval()
        logging.info(f"Worker-{worker_idx} (GPU {gpu_id}) Ready.")
    except Exception as e:
        logging.error(f"Worker-{worker_idx} Init Failed: {e}")
        return []

    processed_groups = []

    for group in tqdm(groups_chunk, desc=f"GPU-{gpu_id}", position=worker_idx, leave=False):
        group_prompt = group.get("text_prompt")
        group_image_path = group.get("image_path") 
        group_id = group.get("group_id")
        
        if not group_prompt or not group_image_path: continue
        
        try:
            # Encode Text + Image Condition
            cond_path = encode_group_condition(pipeline, group_prompt, group_image_path, group_id, sub_folder_name, output_base)
            
            video_entries = []
            for entry in group.get('videos', []):
                rel_v_path = entry.get("video_path")

                try:
                    # Encode Video Latent
                    lat_path = encode_video_latent(pipeline, rel_v_path, sub_folder_name, output_base, group_id)
                    
                    # Update paths to be relative to BASE_PATH for portability
                    entry["latent_path"] = str(Path(lat_path).relative_to(BASE_PATH))
                    entry["condition_path"] = str(Path(cond_path).relative_to(BASE_PATH))
                    entry["video_path"] = rel_v_path
                    video_entries.append(entry)
                except Exception as e:
                    logging.error(f"Latent Error on {rel_v_path}: {e}")
                    continue
            
            if video_entries:
                group['videos'] = video_entries
                processed_groups.append(group)
                
        except Exception as e:
            logging.error(f"Group Condition Error on {group_id}: {e}")
            continue
            
    del pipeline
    torch.cuda.empty_cache()
    return processed_groups

def process_single_input():
    logging.info(f"🚀 Starting Latent encoding for file: {INPUT_JSON_PATH}")
    
    data = safe_load_json(INPUT_JSON_PATH)
    if not data:
        logging.error(f"❌ Could not load input file: {INPUT_JSON_PATH}")
        return

    # Extract groups (support both list format and dict with 'groups' key)
    if isinstance(data, dict) and 'groups' in data:
        all_groups = data['groups']
    elif isinstance(data, list):
        all_groups = data
    else:
        logging.error("❌ Invalid JSON format. Expected list or dict with 'groups' key.")
        return

    logging.info(f"✅ Loaded {len(all_groups)} groups.")

    num_gpus = len(GPUS)
    chunks = [all_groups[i::num_gpus] for i in range(num_gpus)]
    
    # Define a sub-folder name for these latents to keep them organized
    sub_folder_name = "processed" 

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        # Pass the sub_folder_name to workers
        results = pool.starmap(gpu_worker, [(GPUS[i], chunks[i], i, sub_folder_name) for i in range(num_gpus)])

    final_groups = [g for sub in results for g in sub]
    
    if final_groups:
        # Save as a dict with 'groups' key, or just a list depending on your preference
        safe_save_json(OUTPUT_JSON_PATH, {'groups': final_groups})
        logging.info(f"✅ Encoding complete. Processed {len(final_groups)} groups.")
        logging.info(f"💾 Results saved to: {OUTPUT_JSON_PATH}")
    else:
        logging.warning("⚠️ No groups were successfully processed.")

if __name__ == "__main__":
    start_total = time.time()
    
    process_single_input()
        
    print(f"🎉 Task completed! Total time: {(time.time() - start_total)/3600:.2f} hours")