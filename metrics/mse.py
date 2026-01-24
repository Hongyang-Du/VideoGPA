import torch
import numpy as np
import torch.nn.functional as F
from metrics.base import Metric
from piq import ssim 
class MSEMetric(Metric):
    """
    Video-level Mean Squared Error (MSE).
    Optimized for GPU Tensors.
    """
    def __init__(self):
        super().__init__(name="mse")

    def compute(self, *, gt, rep, **kwargs) -> float:
        """
        Args:
            gt: [T, C, H, W] Tensor (-1~1 or 0~1) OR [T, H, W, C] Numpy
            rep: same as gt
        """

        gt_t = self._to_tensor_01(gt)
        rep_t = self._to_tensor_01(rep)

        if gt_t.shape[-2:] != rep_t.shape[-2:]:
            rep_t = F.interpolate(rep_t, size=gt_t.shape[-2:], mode='bilinear', align_corners=False)

        mse_val = (gt_t - rep_t).pow(2).mean()
        
        return float(mse_val.item())

    def _to_tensor_01(self, x):
        """Helper: Ensure input is [B, C, H, W] Tensor in range [0, 1]"""
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.ndim == 3: t = t.unsqueeze(0) # CHW -> BCHW
            
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)

            if t.min() < 0:
                t = (t + 1.0) / 2.0
            elif t.max() > 1.0:
                t = t / 255.0
            
            return t
            
        # Fallback for Numpy
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
            if t.ndim == 3: t = t.unsqueeze(0)
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
            if t.max() > 1.0: t = t / 255.0
            return t.cuda() if torch.cuda.is_available() else t
            
        return x
    
class PSNRMetric(Metric):
    def __init__(self, device="cuda"):
        super().__init__(name="psnr")
        self.device = device

    def compute(self, *, gt, rep, **kwargs) -> float:
        gt_t = self._to_tensor_01(gt).to(self.device)
        rep_t = self._to_tensor_01(rep).to(self.device)

        if gt_t.shape[-2:] != rep_t.shape[-2:]:
            rep_t = F.interpolate(rep_t, size=gt_t.shape[-2:], mode='bilinear', align_corners=False)

        mse = F.mse_loss(gt_t, rep_t)
        if mse == 0:
            return 100.0  
        
        # 10 * log10(MAX^2 / MSE)
        psnr = 10 * torch.log10(1.0 / mse)
        return float(psnr.item())

    def _to_tensor_01(self, x):
        """Helper: Ensure input is [B, C, H, W] Tensor in range [0, 1]"""
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.ndim == 3: t = t.unsqueeze(0) # CHW -> BCHW
            
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)

            if t.min() < 0:
                t = (t + 1.0) / 2.0
            elif t.max() > 1.0:
                t = t / 255.0
            
            return t
            
        # Fallback for Numpy
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
            if t.ndim == 3: t = t.unsqueeze(0)
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
            if t.max() > 1.0: t = t / 255.0
            return t.cuda() if torch.cuda.is_available() else t
            
        return x
    
class SSIMMetric(Metric):
    def __init__(self, device="cuda"):
        super().__init__(name="ssim")
        self.device = device

    def compute(self, *, gt, rep, **kwargs) -> float:

        gt_t = self._to_tensor_01(gt).to(self.device)
        rep_t = self._to_tensor_01(rep).to(self.device)
        return float(ssim(gt_t, rep_t, data_range=1.0).item())
    
    def _to_tensor_01(self, x):
        """Helper: Ensure input is [B, C, H, W] Tensor in range [0, 1]"""
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.ndim == 3: t = t.unsqueeze(0) # CHW -> BCHW
            
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)

            if t.min() < 0:
                t = (t + 1.0) / 2.0
            elif t.max() > 1.0:
                t = t / 255.0
            
            return t
            
        # Fallback for Numpy
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
            if t.ndim == 3: t = t.unsqueeze(0)
            if t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
            if t.max() > 1.0: t = t / 255.0
            return t.cuda() if torch.cuda.is_available() else t
            
        return x