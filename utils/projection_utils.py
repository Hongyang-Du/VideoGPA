import os
import sys
import time
import cv2 
import torch
import numpy as np

# ==============================================================================
# GPU Rendering Core (project_points)
# ==============================================================================

def project_points(pc: torch.Tensor, colors: torch.Tensor, K: torch.Tensor, E: torch.Tensor, H: int, W: int, bg=(0, 0, 0)) -> torch.Tensor:
    """
    PyTorch PointCloud Rendering (GPU Version)。
    """
    R = E[:3, :3]
    t = E[:3, 3]

    pc_cam = torch.matmul(pc, R.T) + t
    pc_proj = torch.matmul(pc_cam, K.T)

    z = pc_proj[:, 2]
    u = (pc_proj[:, 0] / (z + 1e-8)).round().long()
    v = (pc_proj[:, 1] / (z + 1e-8)).round().long()

    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    c = colors[valid_mask]

    if len(u) == 0:
        return torch.tensor(bg, device=pc.device, dtype=torch.uint8).view(1, 1, 3).repeat(H, W, 1)

    sort_idx = torch.argsort(z, descending=True)
    
    u = u[sort_idx]
    v = v[sort_idx]
    c = c[sort_idx]

    # [H, W, 3]
    canvas = torch.tensor(bg, device=pc.device, dtype=torch.uint8).view(1, 1, 3).repeat(H, W, 1)
    
    if c.max() <= 1.0:
        c = (c * 255).clamp(0, 255).to(torch.uint8)
    else:
        c = c.clamp(0, 255).to(torch.uint8)

    canvas[v, u] = c
    return canvas

# ==============================================================================
# GPU Batch Reprojection (batch_reproject)
# ==============================================================================

def batch_reproject(pc, colors, intrinsics, extrinsics, H, W, save_path=None) -> torch.Tensor:
    """
    Batch reproject (GPU Tensor)。
    Input: [T, 3, H, W] Tensor range [-1, 1]
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
    images_list = [] 

    # --- CUDA Float32 ---
    def to_cuda_float(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device='cuda', dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            return x.to(device='cuda', dtype=torch.float32)
        return x
    
    # 1. 清洗数据类型
    # 注意: 如果 pc, colors 已经是 GPU Tensor，可以跳过这步，但在函数内部做保证最安全
    pc = to_cuda_float(pc)
    colors = to_cuda_float(colors)
    intrinsics = to_cuda_float(intrinsics)
    extrinsics = to_cuda_float(extrinsics)

    # 2. Frame Rendering (output [H, W, 3] uint8 Tensor)
    for i in range(len(extrinsics)):
        canvas_gpu_uint8 = project_points(
            pc, colors, intrinsics[i], extrinsics[i], H, W
        )
        
        if save_path is not None:
            # I/O
            img_np = canvas_gpu_uint8.cpu().numpy()
            cv2.imwrite(os.path.join(save_path, f"{i:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
        images_list.append(canvas_gpu_uint8)

    # Stack: [T, H, W, 3] -> Permute: [T, 3, H, W] -> Normalize: [-1, 1] float32
    if not images_list:
        return torch.zeros((0, 3, H, W), device='cuda', dtype=torch.float32)

    stack = torch.stack(images_list).permute(0, 3, 1, 2).float() 
    return (stack / 255.0) * 2.0 - 1.0