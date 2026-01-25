# VideoGPA

VideoGPA is a video generation quality assessment and optimization framework with DPO (Direct Preference Optimization) training capabilities.

# Quick Inference Scripts 🚀

This directory contains simplified command-line scripts for generating videos using **CogVideoX** models. These scripts are designed for quick testing and allow you to run inference directly from the terminal without preparing complex JSON configuration files.

Both scripts support loading **LoRA adapters** for customized generation.

## 📋 Requirements

Make sure you have the required Python libraries installed:

```bash
pip install torch diffusers transformers peft accelerate huggingface_hub
```

## 📝 Available Scripts

### 1. Text-to-Video Generation ([t2v_inference.py](generate/t2v_inference.py))

Generate videos from text prompts using CogVideoX-5B.

**Basic Usage:**
```bash
cd generate
python t2v_inference.py "A cat playing with a ball in a garden"
```

**Advanced Usage:**
```bash
python t2v_inference.py "A flying drone over a city skyline at sunset" \
    --output_dir ./my_videos \
    --lora_path ./checkpoints/my_lora_adapter \
    --gpu_id 0
```

**Arguments:**
- `prompt` (required): Text prompt for video generation
- `--output_dir`: Directory to save generated videos (default: `./outputs`)
- `--lora_path`: Path to LoRA adapter weights (optional)
- `--gpu_id`: GPU device ID (default: 0)

**Output:** Videos saved as `{prompt}_seed{seed}.mp4`

---

### 2. Image-to-Video Generation ([i2v_inference.py](generate/i2v_inference.py))

Generate videos from a static image with text guidance using CogVideoX-5B-I2V.

**Basic Usage:**
```bash
cd generate
python i2v_inference.py "The camera slowly zooms in" ./path/to/image.jpg
```

**Advanced Usage:**
```bash
python i2v_inference.py "Camera pans from left to right" ./input_image.png \
    --output_dir ./i2v_outputs \
    --lora_path ./checkpoints/i2v_lora \
    --gpu_id 1
```

**Arguments:**
- `prompt` (required): Text prompt describing motion/scene
- `image_path` (required): Path to input image file
- `--output_dir`: Directory to save generated videos (default: `./outputs`)
- `--lora_path`: Path to LoRA adapter weights (optional)
- `--gpu_id`: GPU device ID (default: 0)

**Output:** Videos saved as `{image_name}_seed{seed}.mp4`

---

## ⚙️ Configuration

Both scripts include configurable generation parameters:

```python
NUM_INFERENCE_STEPS = 50  # Number of diffusion steps
GUIDANCE_SCALE = 6.0      # Classifier-free guidance scale
SEEDS = [42]              # Random seeds for generation
```

Edit these values in the scripts to adjust generation quality and diversity.

## 🎯 Key Features

- **Automatic Model Download**: Models automatically download from HuggingFace if not found locally
- **LoRA Support**: Load and merge LoRA adapters for customized generation
- **Memory Optimization**: VAE tiling and slicing enabled for efficient GPU memory usage
- **Smart File Handling**: Automatically skips existing output files
- **Multi-seed Generation**: Generate multiple variations by configuring the `SEEDS` list

## 💾 GPU Memory Requirements

- **Minimum VRAM**: ~16GB for base models
- **Recommended VRAM**: 24GB+ for smooth generation
- Memory optimizations (VAE tiling/slicing) are automatically enabled

## 🎬 Visual Comparisons



<video src="https://github.com/user-attachments/assets/40bfebaf-365c-48f0-90dc-ee574228024a" width="100%" controls preload="metadata"></video>

<details>
  <summary><b>Prompt:</b> Pirate-themed amusement rides in a serene outdoor park....</summary>
  <br>
  <blockquote>
     The video features a series of pirate-themed amusement rides in an outdoor park setting, with each ride having unique names like 'Pirate Ship,' 'Pirate's Bay,' 'Pirate's Plunder,' 'Pirate's Cove,' 'Pirate's Revenge,' 'Pirate's Castle,' 'Pirate's Bay,' 'Pirate's Plunder,' 'Pirate's Castle,' and 'Pirate's Plunder.' The rides are adorned with vibrant colors, decorative elements, and safety signs, including a 'No Entry' sign and a 'Safety' sign. The surrounding area is lush with trees, and the atmosphere is serene, with no people present. The video captures the tranquil and still ambiance of the park.
  </blockquote>
</details>








## 🚀 Features

- **Video Quality Assessment**: Comprehensive metrics for evaluating video generation quality
- **DPO Training**: Direct Preference Optimization for video generation models
- **Multi-Model Support**: Compatible with CogVideoX and other video generation models
- **Flexible Pipeline**: Easy-to-use inference and training pipelines

## 📁 Project Structure

```
VideoGPA/
├── data_prep/      # Data preparation scripts
├── train_dpo/      # DPO training scripts
├── pipelines/      # Inference pipelines
├── metrics/        # Quality assessment metrics
├── vggt/           # Video generation model architecture
└── utils/          # Utility functions
```

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd VideoGPA

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage

### Training

```bash
# Run DPO training
python train_dpo/CogVideoX-T2V-5B_lora/03_train.py
```

### Inference

```bash
# Generate videos with trained model
python pipelines/inference.py --model_path <path-to-checkpoint>
```

## 📊 Metrics

VideoGPA provides comprehensive video quality metrics including:
- Visual quality assessment
- Temporal consistency
- Motion smoothness
- Prompt alignment

## 🙏 Acknowledgements

Built on top of CogVideoX and other state-of-the-art video generation models.

## 📄 License

[Add your license here]
