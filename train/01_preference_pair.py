import os
import sys
import json
import torch
import lpips
import time
from tqdm import tqdm
from pathlib import Path
import logging
import multiprocessing as mp

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project components
from pipelines.process_video import VideoProcessor
from metrics.consistency_score import Consistency_Score

# ============================================================================
# ⭐ Configuration Area 
# ============================================================================
GPUS = [0,1]  # Your configuration

# Replace with your actual dataset and file paths
BASE_PATH = "/path/to/your/root" 

# Input JSON (List of groups, containing absolute video paths)
INPUT_JSON = "/path/to/your/input/json"

# Output JSON (List of groups, with scoring results)
OUTPUT_JSON = os.path.join(BASE_PATH, "meta_temp_t2v.json")

CONF_THRES = 0
NUM_FRAMES = 10 # Number of frames to sample for scoring

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# JSON Helper Functions (No Subset, Safe Read/Write, Supports List Format)
# ============================================================================
def safe_load_json(path):
    """
    Safely load JSON file. Supports both list and object formats.
    Returns None if loading fails.
    """
    try:
        if not os.path.exists(path):
            logging.warning(f"JSON file not found: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {path}: {str(e)}")
        return None

def safe_save_json(path, data):
    """
    Safely save JSON file. Writes to a temporary file first, then replaces 
    the original to prevent corruption.
    """
    temp_path = str(path) + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # Atomic replacement to avoid file corruption on interrupt
        os.replace(temp_path, path)
        logging.info(f"Successfully saved JSON to {path} (total items: {len(data) if isinstance(data, list) else 'object'})")
    except Exception as e:
        logging.error(f"Failed to save JSON file {path}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ============================================================================
# Worker Function (Multi-GPU, Absolute Paths, No Image Dependency)
# ============================================================================
def gpu_worker(gpu_id, groups_chunk, worker_idx, scored_video_map):
    """
    Independent process running on a specific GPU to process assigned video groups.
    
    Args:
        gpu_id: Physical GPU ID.
        groups_chunk: List of groups assigned to this process.
        worker_idx: Process index (for progress bar positioning).
        scored_video_map: Map of already scored videos (for resuming).
        
    Returns:
        List of processed groups with scoring results.
    """
    # 1. Configure GPU environment for current process (Isolate GPU usage)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logging.error(f"Worker-{worker_idx} (GPU {gpu_id}): CUDA is not available, using CPU (slow)")

    # 2. Initialize in-process deduplication set
    processed_video_paths = set()

    # 3. Initialize scoring models and video processor
    try:
        # Load LPIPS model (for consistency calculation)
        lpips_net = lpips.LPIPS(net='vgg').to(device)
        lpips_net.eval()  # Eval mode, disable gradients
        # Build metrics dictionary
        metrics_dict = {"Consistency_Score": Consistency_Score(lpips_net, device=device)}
        # Initialize Video Processor
        vp = VideoProcessor(metrics=metrics_dict, model_name="facebook/VGGT-1B", device=device)
    except Exception as e:
        logging.error(f"Worker-{worker_idx} (GPU {gpu_id}) failed to initialize models: {str(e)}")
        return []

    # 4. Process assigned data chunk
    processed_chunk = []
    pbar = tqdm(groups_chunk, desc=f"GPU-{gpu_id}", position=worker_idx, leave=False)

    for group in pbar:
        # Deep copy original group to avoid modifying shared data structures unpredictably
        new_group = group.copy()
        # Ensure 'videos' field exists
        original_videos = new_group.get("videos", [])
        if not isinstance(original_videos, list):
            original_videos = []

        # 5. Process all videos in the current group
        scored_videos = []
        for video_entry in original_videos:
            current_entry = video_entry.copy()
            video_abs_path = current_entry.get("video_path")

            # Skip entries without a video path
            if not video_abs_path:
                scored_videos.append(current_entry)
                continue

            # 6. In-process deduplication
            if video_abs_path in processed_video_paths:
                scored_videos.append(current_entry)
                continue
            processed_video_paths.add(video_abs_path)

            # 7. Resume: Reuse existing scores if available
            existing_entry = scored_video_map.get(video_abs_path)
            if existing_entry and all(k in existing_entry for k in ["consistency_score", "motion_norm"]):
                current_entry.update({
                    "consistency_score": existing_entry["consistency_score"],
                    "motion_norm": existing_entry["motion_norm"]
                })
                scored_videos.append(current_entry)
                continue

            # 8. Validate video file
            video_path_obj = Path(video_abs_path)
            if not video_path_obj.exists():
                logging.warning(f"Worker-{worker_idx} (GPU {gpu_id}): Video not found - {video_abs_path}")
                scored_videos.append(current_entry)
                continue
            if not os.access(video_path_obj, os.R_OK):
                logging.warning(f"Worker-{worker_idx} (GPU {gpu_id}): No read permission - {video_abs_path}")
                scored_videos.append(current_entry)
                continue
            if video_path_obj.stat().st_size <= 0:
                logging.warning(f"Worker-{worker_idx} (GPU {gpu_id}): Empty video file - {video_abs_path}")
                scored_videos.append(current_entry)
                continue

            # 9. Execute Video Scoring
            try:
                with torch.no_grad():  # Disable gradients to save VRAM
                    results = vp.process(
                        video_path=str(video_path_obj),
                        thresholds=[CONF_THRES],
                        num_frames=NUM_FRAMES,
                        save_visuals=False,
                        out_dir=None
                    )
                # Extract results
                metric_results = results.get(CONF_THRES, {})
                consistency_score = metric_results.get("Consistency_Score")
                motion_norm = metric_results.get("motion_norm")

                # 10. Save valid scores (convert to float to avoid JSON serialization issues)
                if consistency_score is not None and motion_norm is not None:
                    current_entry["consistency_score"] = float(consistency_score)
                    current_entry["motion_norm"] = float(motion_norm)
                else:
                    logging.warning(f"Worker-{worker_idx} (GPU {gpu_id}): No valid scores for - {video_abs_path}")
            except Exception as e:
                logging.warning(f"Worker-{worker_idx} (GPU {gpu_id}): Failed to process video {video_abs_path}: {str(e)}")

            # 11. Add processed entry to list
            scored_videos.append(current_entry)

        # 12. Update group's video list, only keeping valid/processed entries
        new_group["videos"] = scored_videos
        if scored_videos:  # Only add groups that have valid videos
            processed_chunk.append(new_group)

    # 13. Return results processed by this worker
    return processed_chunk

# ============================================================================
# Main Control Function (List Format, Multi-GPU, Resume Support)
# ============================================================================
def process_video_scoring():
    """Main workflow: Load data, Schedule GPUs, Merge results, Save output."""
    logging.info("🚀 Starting Batch Video Scoring (List Format, Multi-GPU)")
    logging.info(f"Input File: {INPUT_JSON}")
    logging.info(f"Output File: {OUTPUT_JSON}")
    logging.info(f"GPUs: {GPUS} | Frames: {NUM_FRAMES} | Conf Thres: {CONF_THRES}")

    # 1. Load Original Input Data
    original_data = safe_load_json(INPUT_JSON)
    if not original_data:
        logging.error("❌ Task Terminated: Failed to load valid Input JSON.")
        return

    # Extract Group List: Support both {"groups": [...]} and pure list [...] formats
    all_groups = []
    if isinstance(original_data, dict):
        if "groups" in original_data and isinstance(original_data["groups"], list):
            all_groups = original_data["groups"]
            logging.info(f"📌 Identified Input JSON Format: {{'groups': [...]}}, extracted {len(all_groups)} groups.")
        else:
            logging.error("❌ Task Terminated: Input JSON is an object but missing 'groups' key or value is not a list.")
            return
    elif isinstance(original_data, list):
        all_groups = original_data
        logging.info(f"📌 Identified Input JSON Format: Pure List, containing {len(all_groups)} groups.")
    else:
        logging.error("❌ Task Terminated: Input JSON format not supported.")
        return

    if len(all_groups) == 0:
        logging.warning("⚠️  Extracted group list is empty, nothing to process.")
        return
    logging.info(f"✅ Successfully loaded input data with {len(all_groups)} valid groups.")
    

    # 2. Build Scored Video Map (Resume Capability)
    scored_video_map = {}
    existing_output = safe_load_json(OUTPUT_JSON)
    if existing_output and isinstance(existing_output, list):
        logging.info(f"✅ Existing output file detected. Loading {len(existing_output)} groups for resuming.")
        for group in existing_output:
            for video_entry in group.get("videos", []):
                video_path = video_entry.get("video_path")
                if video_path and all(k in video_entry for k in ["consistency_score", "motion_norm"]):
                    scored_video_map[video_path] = video_entry
    logging.info(f"✅ Loaded {len(scored_video_map)} previously scored videos.")

    # 3. Split Data: Divide groups among available GPUs
    num_gpus = len(GPUS)
    group_chunks = [all_groups[i::num_gpus] for i in range(num_gpus)]
    logging.info(f"✅ Data split into {num_gpus} chunks for parallel processing.")

    # 4. Start Multiprocessing (Using 'spawn' mode for CUDA compatibility)
    ctx = mp.get_context("spawn")
    processed_results = []
    try:
        with ctx.Pool(processes=num_gpus) as pool:
            # Dispatch tasks
            results = pool.starmap(
                gpu_worker,
                [(GPUS[i], group_chunks[i], i, scored_video_map) for i in range(num_gpus)]
            )
            # Merge results
            for res in results:
                if res and isinstance(res, list):
                    processed_results.extend(res)
    except Exception as e:
        logging.error(f"❌ Multiprocessing execution failed: {str(e)}")
        return

    logging.info(f"✅ Multiprocessing complete. Processed {len(processed_results)} valid groups.")

    # 6. Save Final Results
    safe_save_json(OUTPUT_JSON, processed_results)

    # 7. Output Summary
    logging.info("🎉 Video Batch Scoring Task Completed!")
    logging.info(f"📊 Stats: Input {len(original_data)} groups, Output {len(processed_results)} valid groups.")
    logging.info(f"📊 Reused {len(scored_video_map)} previously scored videos.")

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    # Record total time
    start_time = time.time()

    # Execute scoring process
    process_video_scoring()

    # Log duration
    total_hours = (time.time() - start_time) / 3600
    logging.info(f"⏱️  Total Duration: {total_hours:.2f} hours ({total_hours*60:.2f} minutes)")