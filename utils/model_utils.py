import torch
import numpy as np
import os, glob
from PIL import Image
from torchvision import transforms as TF
from typing import Dict, Any, List

from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ----------------------------------------------------
# 1. I/O FREE Preprocessing
# Based on VGGT load_and_preprocess_images
# ----------------------------------------------------

def preprocess_images_from_numpy(frames_np_array: np.ndarray, mode: str = "crop") -> torch.Tensor:
    """
    I/O Free version of load_and_preprocess_images. 
    Processes a numpy array of frames ([T, H, W, 3], RGB) for VGGT model input.
    """
    if frames_np_array.ndim != 4 or frames_np_array.shape[-1] != 3:
        raise ValueError("Input frames_np_array must be [T, H, W, 3] (RGB).")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images: List[torch.Tensor] = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # Loop all Frames
    for frame_np in frames_np_array:
        # 1. Numpy -> PIL Image 
        img = Image.fromarray(frame_np, 'RGB')
        width, height = img.size
        # 2. Resize while maintaining aspect ratio
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img_tensor = to_tensor(img)  # Convert to tensor (0, 1)
        # 3. Crop or Pad to target_size x target_size
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        
        shapes.add((img_tensor.shape[1], img_tensor.shape[2]))
        images.append(img_tensor)

    if len(shapes) > 1:
        pass 

    images_stacked = torch.stack(images)  # [T, C, H, W]

    # [B, T, C, H, W]
    if images_stacked.dim() == 4:
        images_stacked = images_stacked.unsqueeze(0)
        
    return images_stacked # return [1, T, C, H, W] or [B, T, C, H, W]



def run_model_gpu(frames_np_array: np.ndarray, model, save_path=None) -> Dict[str, Any]:
    """
    [Final Optimized] Run VGGT model directly on input frames_np_array (Numpy) and keep everything on CUDA.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device).eval()
    
    images = preprocess_images_from_numpy(frames_np_array).to(device)
    
    if images.shape[1] == 0:
        raise ValueError("No frames processed from input array.")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

    # Pose decoding (Pose encoding to Extrinsic/Intrinsic)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"], predictions["intrinsic"] = extrinsic, intrinsic

    # Squeeze batch (1, ...) -> (...)
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1:
             predictions[k] = v.squeeze(0)

    if "world_points" in predictions:
        predictions["world_points_from_depth"] = predictions["world_points"]

    if save_path:
        torch.save(predictions, save_path)
        
    return predictions
