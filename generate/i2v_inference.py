import os
import argparse
import torch
from pathlib import Path
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from peft import PeftModel
from huggingface_hub import snapshot_download # Added for model downloading

# ================= Configuration =================
# Default settings (can be overridden by command line arguments)
DEFAULT_BASE_MODEL = 'THUDM/CogVideoX-5B-I2V'
DEFAULT_OUTPUT_DIR = './outputs'
DEFAULT_LORA_PATH = None 

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
SEEDS = [42] 

def check_and_download_model(model_path):
    """
    Check if the model path exists locally. 
    If not, and it's not the default Repo ID, download the official model to that path.
    """
    # If the path is the official Repo ID, diffusers handles the cache download automatically.
    if model_path == DEFAULT_BASE_MODEL:
        return

    # If it's a custom path and doesn't exist, download the official weights there.
    if not os.path.exists(model_path):
        print(f"⚠️  Base model not found at local path: {model_path}")
        print(f"⬇️  Downloading official weights ({DEFAULT_BASE_MODEL}) to {model_path}...")
        try:
            snapshot_download(
                repo_id=DEFAULT_BASE_MODEL,
                local_dir=model_path,
                local_dir_use_symlinks=False # Ensure real files are downloaded
            )
            print(f"✅ Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise e
    else:
        print(f"✅ Found local base model at: {model_path}")

def generate_video(args):
    """
    Main generation function.
    """
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    
    print(f"🚀 Initializing generation on GPU {args.gpu_id}...")
    print(f"📝 Prompt: {args.prompt}")
    print(f"🖼️  Image Path: {args.image_path}")

    # 0. Check and download base model if necessary
    check_and_download_model(args.base_model)

    # 1. Load Input Image
    try:
        image = load_image(args.image_path)
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    # 2. Load Base Model
    print(f"⬇️  Loading base model pipeline from: {args.base_model}")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Enable memory optimizations
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # 3. Mount LoRA Adapter (if provided)
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"🛠️  Mounting LoRA adapter from: {args.lora_path}")
        # Using PeftModel to wrap the transformer as per the original logic
        # We assign a generic adapter name 'default'
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, 
            args.lora_path, 
            adapter_name="default"
        )
        # Merge weights to simplify inference (optional, but good for performance)
        pipe.transformer.merge_and_unload()
        print(f"✅ LoRA adapter mounted and merged.")
    elif args.lora_path:
        print(f"⚠️  Warning: LoRA path provided but not found: {args.lora_path}. Proceeding with base model.")

    pipe.transformer.eval()

    # 4. Prepare Output Directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 5. Inference Loop
    for seed in SEEDS:
        # Create a safe filename
        image_name = Path(args.image_path).stem
        file_name = f"{image_name}_seed{seed}.mp4"
        out_file = output_path / file_name
        
        if out_file.exists():
            print(f"⏭️  Skipping existing file: {file_name}")
            continue

        print(f"🎨 Generating seed {seed}...")
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        with torch.inference_mode():
            video = pipe(
                prompt=args.prompt,
                image=image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).frames[0]
        
        export_to_video(video, str(out_file), fps=8)
        print(f"✅ Saved video to: {out_file}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="CogVideoX Image-to-Video Generator (Single Input)")
    
    # Required arguments
    parser.add_argument("prompt", type=str, help="The text prompt for generation")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save output videos")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="HuggingFace path OR Local Path for CogVideoX I2V")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH, help="Path to LoRA weights (optional)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found at {args.image_path}")
        return

    generate_video(args)

if __name__ == "__main__":
    main()