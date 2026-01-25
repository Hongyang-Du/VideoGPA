import os
import requests
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
        response.raise_for_status() # Check for HTTP errors
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
            os.remove(save_path) # Clean up partial downloads
        print(f"❌ Failed to download {url}: {e}")

def main():
    # Replace with your actual GitHub Release URL
    RELEASE_TAG = "v1.0.0-weights"
    BASE_URL = f"https://github.com/Hongyang-Du/VideoGPA/releases/download/{RELEASE_TAG}"
    # Mapping remote files to local paths
    weights_info = [
        {
            "url": f"{BASE_URL}/i2v_adapter_model.safetensors",
            "save_path": "checkpoints/VideoGPA-I2V-lora/adapter_model.safetensors"
        },
        {
            "url": f"{BASE_URL}/t2v_adapter_model.safetensors",
            "save_path": "checkpoints/VideoGPA-T2V-lora/adapter_model.safetensors"
        }
    ]

    print("🔍 Checking checkpoints...")
    for weight in weights_info:
        download_file(weight["url"], weight["save_path"])

    print("\n✨ All checkpoints are ready!")

if __name__ == "__main__":
    main()