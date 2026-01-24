import numpy as np
from plyfile import PlyData, PlyElement



import torch
import numpy as np
from plyfile import PlyData, PlyElement 

def get_colored_pointcloud(predictions, mode="pointmap", conf_thres=50):
    """
    [GPU Optimized] Generate colored point cloud from VGGT predictions.
    Keeps everything on CUDA. Returns Tensors, not Numpy arrays.
    """
    # 1. Determine points and confidence
    if "pointmap" in mode.lower() and "world_points" in predictions:
        points = predictions["world_points"]
        # Ensure confidence is a tensor
        conf = predictions.get("world_points_conf", torch.ones_like(points[..., 0]))
    else:
        points = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf", torch.ones_like(points[..., 0]))

    # Ensure inputs are tensors
    if isinstance(points, np.ndarray): points = torch.from_numpy(points).cuda()
    if isinstance(conf, np.ndarray): conf = torch.from_numpy(conf).cuda()

    # 2. Reshape to List of Points [N, 3]
    vertices = points.reshape(-1, 3)
    
    # 3. Handle Colors
    images = predictions["images"]
    if isinstance(images, np.ndarray): images = torch.from_numpy(images).cuda()
    
    # Check shape: [B, C, H, W] -> [B, H, W, C]
    if images.ndim == 4 and images.shape[1] == 3:
        colors = images.permute(0, 2, 3, 1)
    else:
        colors = images
        
    # Scale 0-1 to 0-255 (Keep as float for now, render step handles int casting)
    colors = (colors.reshape(-1, 3) * 255)

    # 4. Filter by Confidence (GPU Top-K)
    vals = conf.reshape(-1)
    # Check valid: finite and > 1e-5
    valid_mask = torch.isfinite(vals) & (vals > 1e-5)
    
    # If using threshold 0, keep all valid
    if conf_thres <= 0:
        mask = valid_mask
        thr = -float('inf')
    else:
        # Calculate Top-K logic
        N = valid_mask.sum().item()
        if N == 0:
            mask = valid_mask # All false
            thr = -float('inf')
        else:
            keep_frac = max(0.0, min(1.0, 1.0 - conf_thres / 100.0))
            k = max(1, int(np.ceil(N * keep_frac)))
            
            # Use PyTorch's optimized topk (faster than sorting on GPU)
            # We only look at valid values
            valid_vals = vals[valid_mask]
            
            # Get the threshold value from the k-th element
            # torch.topk returns (values, indices)
            top_k_vals, _ = torch.topk(valid_vals, k)
            thr = top_k_vals[-1] # The smallest value in the top k
            
            # Create final mask: must be valid AND greater/equal to threshold
            mask = valid_mask & (vals >= thr)

    # Filter vertices and colors using the boolean mask
    kept_vertices = vertices[mask]
    kept_colors = colors[mask]


    return kept_vertices, kept_colors

def save_as_ply(vertices_3d, colors_rgb, filename):
    """
    Save colored point cloud to binary PLY file.
    This function involves Disk I/O, so using CPU/Numpy here is correct and unavoidable.
    """
    # Move to CPU and convert to Numpy for file writing
    if isinstance(vertices_3d, torch.Tensor):
        P = vertices_3d.detach().cpu().numpy()
    else:
        P = np.asarray(vertices_3d)
        
    if isinstance(colors_rgb, torch.Tensor):
        C = colors_rgb.detach().cpu().numpy()
    else:
        C = np.asarray(colors_rgb)
        
    # Ensure colors are uint8 for PLY standard
    C = C.astype(np.uint8)

    verts = np.empty(P.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),
                                        ('red','u1'),('green','u1'),('blue','u1')])
    verts['x'], verts['y'], verts['z'] = P[:,0], P[:,1], P[:,2]
    verts['red'], verts['green'], verts['blue'] = C[:,0], C[:,1], C[:,2]
    
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(filename)
