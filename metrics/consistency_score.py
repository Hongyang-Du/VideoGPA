
import torch
import numpy as np
from metrics.base import Metric
from metrics.mse import MSEMetric
from metrics.lpips import LPIPSMetric


def compute_motion_score_vectorized(extrinsics, device='cuda'):
    """
    Computes motion score entirely on GPU using vectorized operations.
    
    Args:
        extrinsics: [T, 4, 4] Tensor (or list/numpy)
        device: target device
    Returns:
        torch.Tensor (scalar on GPU)
    """
    # 1. Ensure input is a Tensor on the correct device
    if isinstance(extrinsics, torch.Tensor):
        E = extrinsics.to(device).float()
    else:
        E = torch.tensor(extrinsics, dtype=torch.float32, device=device)

    # E shape: [T, 4, 4]
    Rs = E[:, :3, :3]
    ts = E[:, :3, 3]

    # 2. Translation Score (Vectorized)
    trans_diff = torch.norm(ts[1:] - ts[:-1], dim=1)
    mean_trans = torch.mean(trans_diff)

    # 3. Rotation Score (Vectorized Batch Matmul)
    dR = torch.matmul(Rs[1:], Rs[:-1].transpose(-1, -2)) 
    traces = dR.diagonal(dim1=-2, dim2=-1).sum(-1)

    trace_val = torch.clamp((traces - 1) / 2, -1.0, 1.0)
    angles = torch.acos(trace_val)
    mean_rot = torch.mean(angles)

    # 4. Final Score (Returns GPU Tensor)
    score = mean_trans + 0.1 * mean_rot
    
    # Check for NaN (keep as tensor for now)
    if torch.isnan(score):
        return torch.tensor(0.0, device=device)
        
    return score
# -----------------------------------------------------------------------------
# Main Metric Class
# -----------------------------------------------------------------------------
class Consistency_Score(Metric):
    """
    Combined Motion-normalized Metric (GPU Optimized).
    Formula: (MSE + LPIPS) / Motion_Score
    """

    def __init__(self, lpips_net=None, device="cuda"):
        # Name used in JSON output
        super().__init__("Consistency_Score")
        
        self.device = device
        
        # Initialize child metrics
        # (They are now optimized to handle batch tensors)
        self.mse_metric = MSEMetric()
        self.lpips_metric = LPIPSMetric(lpips_net=lpips_net, device=device)

    def compute(self, *, gt, rep, extrinsics, **kwargs):
        """
        Args:
            gt: [B, C, H, W] Tensor on GPU (ideally)
            rep: [B, C, H, W] Tensor on GPU (ideally)
            extrinsics: [T, 4, 4] Tensor or List
        
        Returns:
            float: Single scalar score
        """
        # 1. Compute Base Metrics (Batch Inference on GPU)
        # Note: The updated MSE/LPIPS classes automatically handle 
        # tensor conversion and resizing if needed.
        val_mse = self.mse_metric.compute(gt=gt, rep=rep)
        val_lpips = self.lpips_metric.compute(gt=gt, rep=rep)

        # 2. Compute Motion Score (Vectorized on GPU)
        motion_score = compute_motion_score_vectorized(extrinsics, device=self.device)
        
        # 3. Normalize and Return
        # Add epsilon to avoid division by zero if motion is 0
        # final_score = (val_mse + val_lpips) / (motion_score + 1e-8)
        final_score = (val_mse + val_lpips)
        
        return float(final_score), float(motion_score)