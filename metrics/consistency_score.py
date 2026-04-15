
import torch
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
    if isinstance(extrinsics, torch.Tensor):
        E = extrinsics.to(device).float()
    else:
        E = torch.tensor(extrinsics, dtype=torch.float32, device=device)

    Rs = E[:, :3, :3]
    ts = E[:, :3, 3]

    trans_diff = torch.norm(ts[1:] - ts[:-1], dim=1)
    mean_trans = torch.mean(trans_diff)

    dR = torch.matmul(Rs[1:], Rs[:-1].transpose(-1, -2))
    traces = dR.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_val = torch.clamp((traces - 1) / 2, -1.0, 1.0)
    angles = torch.acos(trace_val)
    mean_rot = torch.mean(angles)

    score = mean_trans + 0.1 * mean_rot

    if torch.isnan(score):
        return torch.tensor(0.0, device=device)

    return score


class Consistency_Score(Metric):
    """
    Reprojection-based consistency metric.
    Supports both VGGT and DA3 backbones — backbone selection happens in
    VideoProcessor; this class only receives reprojected frames and extrinsics.
    Formula: MSE + ratio * LPIPS  (motion score returned separately)
    """

    def __init__(self, lpips_net=None, device="cuda"):
        super().__init__("Consistency_Score")
        self.device = device
        self.mse_metric = MSEMetric()
        self.lpips_metric = LPIPSMetric(lpips_net=lpips_net, device=device)

    def compute(self, *, gt, rep, extrinsics, ratio=0.1, **kwargs):
        """
        Args:
            gt:         [B, C, H, W] Tensor
            rep:        [B, C, H, W] Tensor  (reprojected frames)
            extrinsics: [T, 4, 4] Tensor or list
            ratio:      weight for LPIPS term (default 0.1)

        Returns:
            (float, float): (consistency_score, motion_score)
        """
        val_mse = self.mse_metric.compute(gt=gt, rep=rep)
        val_lpips = self.lpips_metric.compute(gt=gt, rep=rep)
        motion_score = compute_motion_score_vectorized(extrinsics, device=self.device)
        final_score = val_mse + ratio * val_lpips
        return float(final_score), float(motion_score)
