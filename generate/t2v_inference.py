import os
import argparse
import torch
from pathlib import Path
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from peft import PeftModel  # Must import to support CogVideoX Transformer LoRA
from huggingface_hub import snapshot_download # Added for model downloading

# ================= Configuration =================
# Default settings (can be overridden by command line arguments)
DEFAULT_BASE_MODEL = 'THUDM/CogVideoX-5B'
DEFAULT_OUTPUT_DIR = './outputs'
DEFAULT_LORA_PATH = None  # Set path here or via CLI if needed

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
SEEDS = 42  # List of seeds to generate

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

def generate_video(prompt, args):
    """
    Main generation function for a single GPU.
    """
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    
    print(f"🚀 GPU {args.gpu_id} | Initializing generation...")
    print(f"📝 Prompt: {prompt}")

    # 0. Check and download base model if necessary
    check_and_download_model(args.base_model)

    # 1. Load base T2V Pipeline
    print(f"⬇️  Loading base model pipeline from: {args.base_model}")
    pipe = CogVideoXPipeline.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Enable memory optimizations
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # 2. Mount LoRA adapter (if path is provided)
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"🛠️  Mounting LoRA adapter from: {args.lora_path}")
        # Directly wrap the transformer module using PeftModel
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, 
            args.lora_path, 
            adapter_name="default_adapter"
        )
        # Merge LoRA weights into the base model
        pipe.transformer.merge_and_unload()
        print(f"✅ LoRA adapter mounted and merged successfully.")
    elif args.lora_path:
        print(f"⚠️  Warning: LoRA path provided but not found: {args.lora_path}. Proceeding with base model.")

    pipe.transformer.eval()

    # 3. Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 4. Inference Loop

    # Create a safe filename from the prompt (first 30 chars)
    safe_prompt = "".join([c for c in prompt if c.isalnum() or c in (' ', '_')]).rstrip()[:30].replace(" ", "_")
    file_name = f"{safe_prompt}_seed{SEED}.mp4"
    out_file = output_path / file_name
        
    if out_file.exists():
        print(f"⏭️  Skipping existing file: {file_name}")
        continue

        print(f"🎨 Generating seed {SEED}...")
        
    generator = torch.Generator(device=device).manual_seed(SEED)
        
    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
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
    parser = argparse.ArgumentParser(description="CogVideoX Text-to-Video Generator (Single Prompt)")
    
    # Required argument
    parser.add_argument("prompt", type=str, help="The text prompt to generate video from")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save generated videos")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="HuggingFace path OR Local Path for CogVideoX")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH, help="Path to LoRA weights (optional)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()

    generate_video(args.prompt, args)

if __name__ == "__main__":
    main()