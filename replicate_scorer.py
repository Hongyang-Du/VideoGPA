import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import lpips
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.consistency_score import Consistency_Score
from metrics.epipolar import EpipolarMetric
from metrics.lpips import LPIPSMetric
from metrics.mse import MSEMetric, PSNRMetric, SSIMMetric
from metrics.mvcs import MVCSMetric
from pipelines.process_video import DEFAULT_DA3_MODEL, DEFAULT_VGGT_MODEL, VideoProcessor

def parse_int_list_env(name, default):
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_score_config():
    backbone = os.getenv("SCORE_BACKBONE", "da3").strip().lower()
    default_model = DEFAULT_DA3_MODEL if backbone == "da3" else DEFAULT_VGGT_MODEL
    return {
        "devices": parse_int_list_env("SCORE_DEVICES", [0]),
        "base_dir": os.getenv("SCORE_BASE_DIR", "output/replicate"),
        "output_csv": os.getenv("SCORE_OUTPUT_CSV", "output/replicate/scores.csv"),
        "output_json": os.getenv("SCORE_OUTPUT_JSON", ""),
        "num_frames": int(os.getenv("SCORE_NUM_FRAMES", "10")),
        "conf_thres": int(os.getenv("SCORE_CONF_THRES", "0")),
        "ignore_seed": parse_bool_env("SCORE_IGNORE_SEED", True),
        "descriptor_type": os.getenv("SCORE_DESCRIPTOR_TYPE", "lightglue"),
        "backbone": backbone,
        "model_name": os.getenv("SCORE_MODEL_NAME", default_model),
        "resume": parse_bool_env("SCORE_RESUME", False),
        "max_videos": int(os.getenv("SCORE_MAX_VIDEOS", "0")),
        "seed_filter": os.getenv("SCORE_SEED_FILTER", ""),  # e.g. "456" to only score seed_456 videos
    }


SCORE_CONFIG = build_score_config()
METRIC_COLS = ["psnr", "ssim", "lpips", "mvcs", "consistency_score", "epipolar"]


def build_relative_path(base_path, video_path):
    return str(video_path.relative_to(base_path))


def build_metrics(device):
    lpips_net = lpips.LPIPS(net="vgg").to(device)
    lpips_net.eval()
    return {
        "MSE": MSEMetric(),
        "Consistency_Score": Consistency_Score(lpips_net, device=device),
        "MVCS": MVCSMetric(device=device),
        "PSNR": PSNRMetric(device=device),
        "SSIM": SSIMMetric(device=device),
        "LPIPS": LPIPSMetric(device=device, lpips_net=lpips_net),
        "Epipolar": EpipolarMetric(descriptor_type=SCORE_CONFIG["descriptor_type"], device=device),
    }


def score_worker(rank, gpu_id, tasks):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        metrics_dict = build_metrics(device)
        processor = VideoProcessor(
            metrics=metrics_dict,
            model_name=SCORE_CONFIG["model_name"],
            device=device,
            backbone=SCORE_CONFIG["backbone"],
        )
    except Exception as exc:
        print(f"GPU-{gpu_id} init failed: {exc}")
        return []

    scored_data = []
    pbar = tqdm(tasks, desc=f"GPU-{gpu_id} scoring", position=rank, leave=False)
    for task in pbar:
        v_path = Path(task["path"])
        scored_item = {
            "prompt_id": task["prompt_id"],
            "video_name": v_path.name,
            "video_path": str(v_path),
            "relative_path": task["relative_path"],
            "backbone": SCORE_CONFIG["backbone"],
        }

        try:
            results = processor.process(
                video_path=str(v_path),
                thresholds=[SCORE_CONFIG["conf_thres"]],
                num_frames=SCORE_CONFIG["num_frames"],
                save_visuals=False,
            )
            res = results.get(SCORE_CONFIG["conf_thres"], {})
            scored_item.update(
                {
                    "mse": float(res.get("MSE", )),
                    "consistency_score": float(res.get("Consistency_Score", 0.0)),
                    "motion_score": float(res.get("motion_norm", 0.0)),
                    "psnr": float(res.get("PSNR", 0.0)),
                    "ssim": float(res.get("SSIM", 0.0)),
                    "lpips": float(res.get("LPIPS", 0.0)),
                    "mvcs": float(res.get("MVCS", 0.0)),
                    "epipolar": float(res.get("Epipolar", 0.0)),
                }
            )
        except Exception as exc:
            print(f"\nWarning GPU-{gpu_id} failed on {v_path.name}: {exc}")
            scored_item["error"] = str(exc)
            for metric_name in METRIC_COLS:
                scored_item.setdefault(metric_name, None)

        scored_data.append(scored_item)

    return scored_data


def collect_all_video_tasks():
    """
    Scan base_dir for VideoGPA flat structure: base_dir/<prompt_id>/*.mp4
    (single level of subdirectory, no group level)
    """
    base_path = Path(SCORE_CONFIG["base_dir"])
    all_tasks = []

    if not base_path.exists():
        print(f"Base dir does not exist: {base_path}")
        return all_tasks

    print(f"Scanning benchmark root: {base_path}")
    for prompt_dir in sorted(base_path.iterdir()):
        if not prompt_dir.is_dir():
            continue
        video_files = sorted(prompt_dir.glob("*.mp4"))
        if not video_files:
            continue
        for v_file in video_files:
            seed_filter = SCORE_CONFIG.get("seed_filter", "")
            if seed_filter and f"seed_{seed_filter}" not in v_file.name:
                continue
            all_tasks.append(
                {
                    "path": v_file,
                    "prompt_id": prompt_dir.name,
                    "relative_path": build_relative_path(base_path, v_file),
                }
            )

    if SCORE_CONFIG["max_videos"] > 0:
        all_tasks = all_tasks[: SCORE_CONFIG["max_videos"]]
    return all_tasks


def load_existing_items():
    output_json = SCORE_CONFIG.get("output_json")
    if not SCORE_CONFIG["resume"] or not output_json:
        return {}

    out_path = Path(output_json)
    if not out_path.exists():
        return {}

    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {item["relative_path"]: item for item in payload.get("items", [])}


def build_summary(df):
    summary = {"overall": {"video_count": int(len(df))}}
    if df.empty:
        return summary

    for metric in METRIC_COLS:
        if metric not in df.columns:
            df[metric] = None

    summary["overall"].update(
        {
            metric: (None if pd.isna(df[metric].mean()) else float(df[metric].mean()))
            for metric in METRIC_COLS
        }
    )

    return summary


def save_json_report(flat_results, df):
    output_json = SCORE_CONFIG.get("output_json")
    if not output_json:
        return

    payload = {
        "config": SCORE_CONFIG,
        "items": flat_results,
        "summary": build_summary(df),
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON report to: {out_path}")


def main():
    all_tasks = collect_all_video_tasks()
    if not all_tasks:
        print("No videos found for scoring.")
        return

    existing_items = load_existing_items()
    pending_tasks = [task for task in all_tasks if task["relative_path"] not in existing_items]

    print("\nScoring config")
    print(f"  Backbone    : {SCORE_CONFIG['backbone']}")
    print(f"  Model       : {SCORE_CONFIG['model_name']}")
    print(f"  Devices     : {SCORE_CONFIG['devices']}")
    print(f"  Total videos: {len(all_tasks)}")
    print(f"  Pending     : {len(pending_tasks)}")

    if pending_tasks:
        num_gpus = len(SCORE_CONFIG["devices"])
        chunk_size = (len(pending_tasks) + num_gpus - 1) // num_gpus
        task_chunks = [
            pending_tasks[i:i + chunk_size] for i in range(0, len(pending_tasks), chunk_size)
        ]
        while len(task_chunks) < num_gpus:
            task_chunks.append([])

        if num_gpus == 1:
            new_results = score_worker(0, SCORE_CONFIG["devices"][0], task_chunks[0])
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_gpus) as pool:
                process_results = pool.starmap(
                    score_worker,
                    [(i, SCORE_CONFIG["devices"][i], task_chunks[i]) for i in range(num_gpus)],
                )
            new_results = [item for sublist in process_results for item in sublist]
    else:
        new_results = []

    merged = dict(existing_items)
    for item in new_results:
        merged[item["relative_path"]] = item

    flat_results = sorted(
        merged.values(),
        key=lambda item: (item["prompt_id"], item["video_name"]),
    )
    if not flat_results:
        print("No valid scoring results were produced.")
        return

    df = pd.DataFrame(flat_results)

    output_csv = SCORE_CONFIG.get("output_csv")
    if output_csv:
        out_file = Path(output_csv)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"\nSaved CSV report to: {out_file}")

    save_json_report(flat_results, df)

    print("\n========================================")
    print("Overall Mean Metrics")
    print("========================================")
    if not df.empty:
        for metric in METRIC_COLS:
            if metric not in df.columns:
                df[metric] = None
        overall_mean = df[METRIC_COLS].mean()
        overall_mean_df = pd.DataFrame(overall_mean).T.rename(index={0: "overall"})
        print(overall_mean_df.round(4))
        print(f"\nTotal videos scored: {len(df)}")

    print("\nAll scoring tasks completed.")


if __name__ == "__main__":
    main()
