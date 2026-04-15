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

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================================
# ⭐ Configuration Area
# ============================================================================
GPUS = [0, 1]  # Your GPUs Configuration
MODEL_ID = "THUDM/CogVideoX-5B"  # T2V Model (No I2V suffix)

BASE_PATH = "/path/to/your/root" # Replace with your actual dataset root path

# Input/Output JSON Configuration
INPUT_JSON = f"{BASE_PATH}/your_meta_temp_t2v.json"  # Output from video Scorer
OUTPUT_JSON = f"{BASE_PATH}/meta_data_t2v.json"

# T2V Encoding Results Root Directory 
T2V_LATENT_ROOT =  f"{BASE_PATH}/t2v_latent"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Helper Functions (T2V Exclusive, No Image Dependency, No Subset)
# ============================================================================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Load JSON failed: {e}")
        return None

def safe_save_json(path, data):
    temp_path = str(path) + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, path)
        logging.info(f"JSON saved successfully to {path}")
    except Exception as e:
        logging.error(f"Save JSON failed: {e}")

def load_video_frames_tensor(video_path: str, num_frames: int = 49, device: torch.device = None) -> torch.Tensor:
    """Read video, sample fixed frames, convert to tensor for T2V model."""
    vr = VideoReader(video_path, ctx=cpu(0))
    if len(vr) < num_frames:
        indices = np.arange(0, len(vr)).astype(int)
    else:
        indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    return frames.permute(3, 0, 1, 2).to(device)

# ============================================================================
# ⭐ Core Functionality (T2V Exclusive, Text Prompt & Video Only, No Image/Subset)
# ============================================================================
def encode_text_condition(pipeline, prompt, group_id, output_base):
    """
    Encode T2V text condition (Text Embedding only), save to t2v_latent directory (No Subset).
    """
    condition_data = {}
    
    # Extract T2V Text Embedding (using model's own text_encoder and tokenizer)
    if hasattr(pipeline, "text_encoder") and hasattr(pipeline, "tokenizer"):
        with torch.no_grad():
            text_input = pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=226,  # CogVideoX recommended max text length
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(pipeline.device)
            
            # Extract text encoding (encoder_hidden_states)
            text_embeds = pipeline.text_encoder(text_input)[0].squeeze(0).cpu()
            condition_data["encoder_hidden_states"] = text_embeds

    # Save text condition file (No Subset, save directly to t2v_latent root)
    cond_filename = f"cond_{group_id}.pt"
    cond_path = output_base / cond_filename
    torch.save(condition_data, cond_path)
    
    return str(cond_path)

def encode_video_latent(pipeline, video_rel_path, output_base, group_id):
    """
    Encode T2V Video Latent (VAE latent variables), save to t2v_latent directory (No Subset).
    """
    # Build full video path
    video_full_path = Path(BASE_PATH) / video_rel_path
    if not video_full_path.exists():
        raise FileNotFoundError(f"Video not found: {video_full_path}")
    
    # Load video tensor and extract Latent
    device = pipeline.device
    video_tensor = load_video_frames_tensor(str(video_full_path), 49, device).unsqueeze(0)
    
    with torch.no_grad():
        # Use T2V model's VAE to encode video into latents
        vae = pipeline.vae
        video_tensor = video_tensor.to(vae.dtype)
        latent_dist = vae.encode(video_tensor).latent_dist
        latent = latent_dist.sample().squeeze(0).cpu()

    # Build video Latent filename (Unique ID: group_id + video name)
    video_stem = Path(video_rel_path).stem
    latent_filename = f"latent_{group_id}_{video_stem}.pt"
    latent_path = output_base / latent_filename
    
    # Save video Latent (No Subset, save directly to t2v_latent root)
    torch.save(latent, latent_path)
    
    return str(latent_path)

# ============================================================================
# GPU Worker (T2V Exclusive, No Image Processing/No Subset)
# ============================================================================
def gpu_worker(gpu_id, groups_chunk, worker_idx):
    """
    Single GPU worker process, processing assigned group data (T2V Exclusive, No Subset).
    """
    # Device initialization
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    time.sleep(worker_idx * 2)  # Stagger model load times to avoid VRAM contention

    # Build T2V encoding result output directory (No Subset, create t2v_latent directly)
    output_base =T2V_LATENT_ROOT
    output_base.mkdir(parents=True, exist_ok=True)

    # Load CogVideoX T2V model (No I2V suffix)
    try:
        pipeline = CogVideoXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        # VAE optimization config (reduce VRAM usage, support large batch processing)
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        pipeline.vae.eval()
        
        logging.info(f"Worker-{worker_idx} (GPU {gpu_id}) Ready for T2V Encoding.")
    except Exception as e:
        logging.error(f"Worker-{worker_idx} Init Failed: {e}")
        return []

    processed_groups = []

    # Batch process group data (No Subset, directly process assigned chunk)
    for group in tqdm(groups_chunk, desc=f"GPU-{gpu_id}", position=worker_idx, leave=False):
        # Extract required T2V fields (No image_path)
        group_prompt = group.get("text_prompt")
        group_id = group.get("group_id")
        video_entries = group.get("videos", [])
        
        # Field validation (No image field, only validate text and video list)
        if not group_prompt or not video_entries:
            logging.warning(f"Group {group_id} missing prompt or videos, skipped.")
            continue
        
        try:
            # Step 1: Encode text condition (Text Embedding only, No Subset)
            cond_full_path = encode_text_condition(pipeline, group_prompt, group_id, output_base)
            
            # Step 2: Iterate through video entries, encode Latent for each video
            processed_videos = []
            for entry in video_entries:
                rel_video_path = entry.get("video_path")
                if not rel_video_path:
                    continue
                
                try:
                    # Encode video Latent (No Subset, store directly)
                    latent_full_path = encode_video_latent(pipeline, rel_video_path, output_base, group_id)
                    
                    # Update video entry: Add encoding paths (Relative to BASE_PATH)
                    processed_entry = entry.copy()
                    processed_entry["condition_path"] = str(Path(cond_full_path).relative_to(BASE_PATH))
                    processed_entry["latent_path"] = str(Path(latent_full_path).relative_to(BASE_PATH))
                    processed_videos.append(processed_entry)
                    
                except Exception as e:
                    logging.error(f"Video {rel_video_path} Encoding Error: {e}")
                    continue
            
            # Step 3: Build processing results (T2V Exclusive, No Subset)
            if processed_videos:
                processed_group = {
                    "group_id": group_id,
                    "text_prompt": group_prompt,
                    "videos": processed_videos
                }
                processed_groups.append(processed_group)
                
        except Exception as e:
            logging.error(f"Group {group_id} Encoding Error: {e}")
            continue
            
    # Clean up resources, release VRAM
    del pipeline
    torch.cuda.empty_cache()
    return processed_groups

# ============================================================================
# Main Processing Workflow (T2V Exclusive, No Subset, Single File Batch Encoding)
# ============================================================================
def process_t2v_encoding():
    """
    Process T2V data encoding (No Subset).
    Read T2V scoring result JSON, Multi-GPU parallel encoding, output final meta data.
    """
    logging.info(f"🚀 Starting T2V data encoding (Text + Video, No Subset)...")
    logging.info(f"Encoding results will be saved to: {Path(BASE_PATH) / T2V_LATENT_ROOT}")
    
    # 1. Load Input JSON (T2V Scoring Results)
    data = safe_load_json(INPUT_JSON)
    if not data:
        logging.error(f"Input JSON not found or invalid: {INPUT_JSON}")
        return
    

    # Adapt to two input formats: 1. Pure List 2. Object with groups/t2v_groups keys
    if isinstance(data, list):
        all_groups = data  # Pure list format, assign directly
    else:
        all_groups = data.get("t2v_groups", []) or data.get("groups", [])  # Object format, extract group list
        
    if not all_groups:
        logging.warning("No valid T2V groups found in input JSON.")
        return
    
    # 3. Task Dispatch (Multi-GPU Parallel, No Subset Splitting)
    num_gpus = len(GPUS)
    chunks = [all_groups[i::num_gpus] for i in range(num_gpus)]
    
    # 4. Start Multiprocessing (Spawn mode, supports GPU isolation)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        # No Subset, pass only GPU ID, data chunk, worker index
        results = pool.starmap(gpu_worker, [(GPUS[i], chunks[i], i) for i in range(num_gpus)])

    # 5. Aggregate results (No Subset, merge all GPU results directly)
    final_groups = [g for sub_result in results for g in sub_result]
    if not final_groups:
        logging.warning("No groups processed successfully.")
        return
    
    # 6. Save Final Output JSON (Contains encoding paths, for DPO training)
    safe_save_json(OUTPUT_JSON, {"groups": final_groups})
    logging.info(f"✅ T2V Encoding Complete, processed {len(final_groups)} groups, results saved to {OUTPUT_JSON}")

# ============================================================================
# Main Function (No Subset Traversal, Execute T2V Encoding Directly)
# ============================================================================
if __name__ == "__main__":
    start_total = time.time()
    process_t2v_encoding()
    total_hours = (time.time() - start_total) / 3600
    logging.info(f"🎉 All T2V encoding tasks completed! Total time: {total_hours:.2f} hours")