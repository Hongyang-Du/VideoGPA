import torch
import lpips
import numpy as np
import torch.nn.functional as F
from metrics.base import Metric

class LPIPSMetric(Metric):
    """
    Video-level LPIPS Metric.
    Optimized for Batch GPU Inference (No for-loops).
    """
    def __init__(self, device=None, lpips_net=None):
        super().__init__(name="lpips")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if lpips_net is not None:
            self.lpips = lpips_net
        else:
            self.lpips = lpips.LPIPS(net='vgg').to(self.device).eval()

    def compute(self, *, gt, rep, **kwargs):
        """
        Args:
            gt: [T, C, H, W] Tensor (-1~1) OR [T, H, W, C] Numpy
            rep: same as gt
        """
        gt_t = self._to_tensor_neg1_pos1(gt)
        rep_t = self._to_tensor_neg1_pos1(rep)

        if gt_t.shape[-2:] != rep_t.shape[-2:]:
            rep_t = F.interpolate(rep_t, size=gt_t.shape[-2:], mode='bilinear', align_corners=False)

        with torch.no_grad():
            lp_dist = self.lpips(gt_t, rep_t)
            
        return float(lp_dist.mean().item())

    def _to_tensor_neg1_pos1(self, x):
        """Helper: Ensure input is [B, C, H, W] Tensor on GPU in range [-1, 1]"""
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.ndim == 3: t = t.unsqueeze(0)
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
            
            if t.device != torch.device(self.device):
                t = t.to(self.device)

            if t.min() >= 0:
                if t.max() > 1.0:
                    t = t / 255.0
                t = t * 2.0 - 1.0
            
            return t
            
        # Fallback for Numpy
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float().to(self.device)
            if t.ndim == 3: t = t.unsqueeze(0)
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
            # 0-255 -> -1~1
            return (t / 255.0) * 2.0 - 1.0
            
        return x