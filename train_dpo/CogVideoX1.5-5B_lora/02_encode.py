import os
import sys
import json
import torch
import logging
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from diffusers import CogVideoXPipeline
from decord import VideoReader, cpu

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================================
# ⭐ Configurations (Specific to CogVideoX 1.5 T2V)
# ============================================================================
GPUS = [0, 1, 2, 3, 4, 5, 6, 7] 
MODEL_ID = "THUDM/CogVideoX1.5-5B"  

# Replace with your actual environment paths
BASE_PATH = os.environ.get("DATASET_BASE_PATH", "/path/to/dataset/root")

# Input/Output JSON configurations
INPUT_JSON = f"{BASE_PATH}/your_meta_temp_t2v.json"
OUTPUT_JSON = f"{BASE_PATH}/meta_data_t2v15.json"

T2V_LATENT_ROOT = "root/to/save/latents" 
TARGET_NUM_FRAMES = 81 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Helper Functions
# ============================================================================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return None

def safe_save_json(path, data):
    temp_path = str(path) + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, path)
        logging.info(f"JSON saved successfully to {path}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")

def load_video_frames_tensor(video_path: str, num_frames: int, device: torch.device = None) -> torch.Tensor:
    """Reads video, samples fixed number of frames, and converts to tensor for T2V models."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0 # [F, H, W, C]
    return frames.permute(3, 0, 1, 2).to(device) # [C, F, H, W]

# ============================================================================
# Core Logic
# ============================================================================
def encode_text_condition(pipeline, prompt, group_id, output_base):
    """
    Encodes text prompts into embeddings for T2V conditioning.
    """
    condition_data = {}
    
    if hasattr(pipeline, "text_encoder") and hasattr(pipeline, "tokenizer"):
        with torch.no_grad():
            text_input = pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=226,  # CogVideoX 1.5 standard length
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(pipeline.device)
            
            text_embeds = pipeline.text_encoder(text_input)[0].squeeze(0).cpu()
            condition_data["encoder_hidden_states"] = text_embeds

    cond_filename = f"cond_{group_id}.pt"
    cond_path = output_base / cond_filename
    torch.save(condition_data, cond_path)
    
    return str(cond_path)

def encode_video_latent(pipeline, video_rel_path, output_base, group_id):
    """
    Encodes video frames into VAE latents.
    """
    video_full_path = Path(BASE_PATH) / video_rel_path
    if not video_full_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_full_path}")
    
    device = pipeline.device
    # Use pre-configured target frames (e.g., 81)
    video_tensor = load_video_frames_tensor(str(video_full_path), TARGET_NUM_FRAMES, device).unsqueeze(0)
    
    with torch.no_grad():
        vae = pipeline.vae
        video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
        
        # NOTE: Handling scaling factor is crucial for CogVideoX 1.5 latent distribution
        latent_dist = vae.encode(video_tensor).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        latent = latent.squeeze(0).cpu()

    video_stem = Path(video_rel_path).stem
    latent_filename = f"latent_{group_id}_{video_stem}.pt"
    latent_path = output_base / latent_filename
    
    torch.save(latent, latent_path)
    
    return str(latent_path)

# ============================================================================
# Multiprocessing GPU Worker
# ============================================================================
def gpu_worker(gpu_id, groups_chunk, worker_idx):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    # Staggered start to prevent simultaneous VRAM spikes during loading
    time.sleep(worker_idx * 2)

    output_base = Path(BASE_PATH) / T2V_LATENT_ROOT
    output_base.mkdir(parents=True, exist_ok=True)

    try:
        logging.info(f"Worker-{worker_idx} loading model: {MODEL_ID}")
        pipeline = CogVideoXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        pipeline.vae.eval()
        
    except Exception as e:
        logging.error(f"Worker-{worker_idx} initialization failed: {e}")
        return []

    processed_groups = []

    for group in tqdm(groups_chunk, desc=f"GPU-{gpu_id}", position=worker_idx, leave=False):
        group_prompt = group.get("text_prompt")
        group_id = group.get("group_id")
        video_entries = group.get("videos", [])
        
        if not group_prompt or not video_entries:
            continue
        
        try:
            # 1. Encode Text Condition
            cond_full_path = encode_text_condition(pipeline, group_prompt, group_id, output_base)
            
            # 2. Encode Video Latents
            processed_videos = []
            for entry in video_entries:
                rel_video_path = entry.get("video_path")
                if not rel_video_path:
                    continue
                
                try:
                    latent_full_path = encode_video_latent(pipeline, rel_video_path, output_base, group_id)
                    
                    processed_entry = entry.copy()
                    processed_entry["condition_path"] = str(Path(cond_full_path).relative_to(BASE_PATH))
                    processed_entry["latent_path"] = str(Path(latent_full_path).relative_to(BASE_PATH))
                    processed_videos.append(processed_entry)
                    
                except Exception as e:
                    logging.error(f"Error processing video {rel_video_path}: {e}")
                    continue
            
            if processed_videos:
                processed_group = {
                    "group_id": group_id,
                    "text_prompt": group_prompt,
                    "videos": processed_videos
                }
                processed_groups.append(processed_group)
                
        except Exception as e:
            logging.error(f"Error processing group {group_id}: {e}")
            continue
            
    del pipeline
    torch.cuda.empty_cache()
    return processed_groups

# ============================================================================
# Main Execution Flow
# ============================================================================
def process_t2v_encoding():
    logging.info(f"Starting T2V encoding pipeline (Target Frames={TARGET_NUM_FRAMES})...")
    
    data = safe_load_json(INPUT_JSON)
    if not data:
        return

    if isinstance(data, list):
        all_groups = data
    else:
        all_groups = data.get("t2v_groups", []) or data.get("groups", [])
        
    if not all_groups:
        logging.warning("Metadata is empty. Nothing to process.")
        return
    
    num_gpus = len(GPUS)
    chunks = [all_groups[i::num_gpus] for i in range(num_gpus)]
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        results = pool.starmap(gpu_worker, [(GPUS[i], chunks[i], i) for i in range(num_gpus)])

    final_groups = [g for sub_result in results for g in sub_result]
    
    if final_groups:
        safe_save_json(OUTPUT_JSON, {"groups": final_groups})
        logging.info(f"Encoding complete. Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    start_total = time.time()
    process_t2v_encoding()
    logging.info(f"Total execution time: {(time.time() - start_total) / 3600:.2f} hours")