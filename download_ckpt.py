import os
import requests
import argparse
from tqdm import tqdm

def download_file(url, save_path):
    """Downloads a file with a real-time progress bar."""
    if os.path.exists(save_path):
        print(f"✅ File already exists, skipping: {save_path}")
        return

    print(f"🚀 Downloading: {url}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path) 
        print(f"❌ Failed to download {url}: {e}")

def main():
    # GitHub Release 配置
    RELEASE_TAG = "v1.0.0-weights"
    BASE_URL = f"https://github.com/Hongyang-Du/VideoGPA/releases/download/{RELEASE_TAG}"
    
    # 映射表
    ckpt_map = {
        "i2v": {
            "url": f"{BASE_URL}/i2v_adapter_model.safetensors",
            "save_path": "checkpoints/VideoGPA-I2V-lora/adapter_model.safetensors"
        },
        "t2v": {
            "url": f"{BASE_URL}/t2v_adapter_model.safetensors",
            "save_path": "checkpoints/VideoGPA-T2V-lora/adapter_model.safetensors"
        },
        "t2v15": {
            "url": f"{BASE_URL}/t2v15_adapter_model.safetensors",
            "save_path": "checkpoints/VideoGPA-T2V1.5-lora/adapter_model.safetensors"
        }
    }

    parser = argparse.ArgumentParser(description="VideoGPA Checkpoint Downloader")
    parser.add_argument(
        "type", 
        choices=["i2v", "t2v", "t2v15", "all"], 
        help="Select which checkpoint to download (i2v, t2v, t2v15, or all)"
    )
    args = parser.parse_args()

    tasks = []
    if args.type == "all":
        tasks = list(ckpt_map.values())
    else:
        tasks = [ckpt_map[args.type]]

    print(f"🔍 Checking for {args.type} checkpoints...")
    for task in tasks:
        download_file(task["url"], task["save_path"])

    print("\n✨ Done!")

if __name__ == "__main__":
    main()