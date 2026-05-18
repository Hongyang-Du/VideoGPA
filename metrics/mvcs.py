import torch
import torch.nn.functional as F
import numpy as np
from metrics.base import Metric

class MVCSMetric(Metric):
    def __init__(self, device="cuda"):
        # 显式初始化父类
        super().__init__(name="MVCS")
        self.device = device

    def compute(self, *, gt, rep, depths, intrinsics, extrinsics, **kwargs):
        """
        计算 Multi-View Consistency Score (MVCS)
        
        Args:
            depths: [T, H, W, 1] 或 [T, 1, H, W] 的深度图
            intrinsics: [T, 3, 3] 或 [T, 4, 4] 的内参矩阵
            extrinsics: [T, 3, 4] 或 [T, 4, 4] 的外参矩阵 (W2C)
        """
        # 1. 数据设备与类型转换
        if isinstance(depths, np.ndarray):
            depths = torch.from_numpy(depths).to(self.device)
        depths = depths.float()
        intrinsics = intrinsics.to(self.device).float()
        extrinsics = extrinsics.to(self.device).float()

        # 2. 维度清洗: 处理 [T, H, W, 1] 或 [T, 1, H, W]
        if depths.ndim == 4:
            if depths.shape[1] == 1:
                depths = depths.squeeze(1)
            elif depths.shape[3] == 1:
                depths = depths.squeeze(3)
        
        T, H, W = depths.shape

        # 3. 几何矩阵格式标准化
        # 内参强制为 3x3
        if intrinsics.shape[-2:] == (4, 4):
            intrinsics = intrinsics[..., :3, :3]
            
        # 外参强制为 4x4 (解决 linalg.inv 无法对 3x4 求逆的问题)
        if extrinsics.shape[-2:] == (3, 4):
            bottom = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, 1, 4).expand(T, 1, 4)
            extrinsics = torch.cat([extrinsics, bottom], dim=1)

        mvcs_scores = []

        # 4. 准备像素网格坐标 (3, H*W)
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device), 
            torch.arange(W, device=self.device), 
            indexing='ij'
        )
        # 齐次坐标 [u, v, 1]
        coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().reshape(-1, 3).t()

        # 5. 跨帧一致性计算
        for i in range(T - 1):
            j = i + 1
            
            # --- Back-projection (Camera i) ---
            # p_3d_i = K_inv * coords * depth
            d_i = depths[i].reshape(1, -1)
            inv_K = torch.inverse(intrinsics[i])
            p_3d_i = inv_K @ coords * d_i # (3, HW)
            
            # --- Coordinate Transformation (i -> j) ---
            # T_rel = Ext_j * Ext_i_inv
            rel_pose = extrinsics[j] @ torch.inverse(extrinsics[i])
            R, t = rel_pose[:3, :3], rel_pose[:3, 3:4]
            p_3d_j = R @ p_3d_i + t # (3, HW)
            
            # --- Re-projection (Camera j) ---
            p_2d_j_homo = intrinsics[j] @ p_3d_j
            depth_proj_j = p_3d_j[2, :].reshape(H, W) # 投影后的 Z 即为理论深度值
            
            # 防止除以 0
            z_j = p_2d_j_homo[2, :].clamp(min=1e-8)
            u_j = (p_2d_j_homo[0, :] / z_j).reshape(H, W)
            v_j = (p_2d_j_homo[1, :] / z_j).reshape(H, W)
            
            # --- Grid Sampling (采样第 j 帧的预测深度) ---
            # 归一化坐标到 [-1, 1]
            grid_u = 2.0 * u_j / (W - 1) - 1.0
            grid_v = 2.0 * v_j / (H - 1) - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0) # [1, H, W, 2]
            
            sampled_depth_j = F.grid_sample(
                depths[j].view(1, 1, H, W), 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            ).squeeze() # [H, W]
            
            # --- Masking & Error ---
            # 只有投影在图像范围内且深度为正的点才计入误差
            mask = (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H) & (depth_proj_j > 0)
            
            if mask.any():
                # 核心误差：重投影深度 vs 原始预测深度
                err = (sampled_depth_j[mask] - depth_proj_j[mask]).pow(2).mean()
                mvcs_scores.append(err.item())

        # 返回全视频平均误差
        # return np.mean(mvcs_scores) if mvcs_scores else 0.0
        if not mvcs_scores:
            return 0.0
        
        avg_mse_err = np.mean(mvcs_scores)

        score = np.exp(-1.0 * avg_mse_err) 
        
        return score