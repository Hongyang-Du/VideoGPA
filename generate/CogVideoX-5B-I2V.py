import os
import json
import argparse
import torch
from pathlib import Path
from PIL import Image
from peft import PeftModel
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video


def generate(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # Load pipeline
    print(f"Loading base model: {args.base_model}")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Mount LoRA (optional)
    if args.lora_path:
        if not os.path.exists(args.lora_path):
            print(f"LoRA path not found: {args.lora_path}, using base model")
        else:
            print(f"Mounting LoRA: {args.lora_path}")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path, adapter_name="default")
            pipe.transformer.merge_and_unload()
            print("LoRA merged.")

    pipe.transformer.eval()

    # Load prompts
    with open(args.prompt_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        tasks = list(raw_data.items())  # [(key, {text_prompt, image_prompt, ...}), ...]
    elif isinstance(raw_data, list):
        tasks = [(item.get('group_id', i), item) for i, item in enumerate(raw_data)]
    else:
        print("Unsupported JSON format"); return

    if args.num_prompts:
        tasks = tasks[:args.num_prompts]

    print(f"Generating {len(tasks)} prompts, seed={args.seed}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, (group_id, item) in enumerate(tasks):
        group_id = str(group_id).replace('/', '_')
        text_prompt = item.get('text_prompt', item.get('prompt', '')).strip()
        image_path = item.get('image_prompt', item.get('image_path', item.get('input_image_path', '')))

        if not text_prompt or not image_path:
            continue

        # Resolve image path
        if not Path(image_path).exists() and args.base_dir:
            image_path = str(Path(args.base_dir) / image_path)
        if not Path(image_path).exists():
            print(f"[{idx+1}/{len(tasks)}] Image not found: {image_path}, skipping")
            continue

        out_dir = output_root / group_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"seed_{args.seed}.mp4"

        if video_path.exists():
            print(f"[{idx+1}/{len(tasks)}] Skip existing: {group_id}")
            continue

        print(f"[{idx+1}/{len(tasks)}] Generating: {group_id}")
        try:
            image = Image.open(image_path).convert("RGB")
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.inference_mode():
                output = pipe(
                    prompt=text_prompt,
                    image=image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
            export_to_video(output.frames[0], str(video_path), fps=args.fps)
        except Exception as e:
            print(f"  Failed: {e}")
        torch.cuda.empty_cache()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="CogVideoX-5B I2V generation")
    parser.add_argument("--base_model", type=str, default="THUDM/CogVideoX-5B-I2V")
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="e.g. checkpoints/VideoGPA-I2V-lora")
    parser.add_argument("--base_dir", type=str, default=None, help="Base dir for relative image paths")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
