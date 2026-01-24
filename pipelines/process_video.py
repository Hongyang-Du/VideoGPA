import os
import numpy as np
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from utils import sample_uniform_frames 
from utils import run_model_gpu       
from utils import get_colored_pointcloud, batch_reproject
from vggt.models.vggt import VGGT 

class VideoProcessor:
    def __init__(self, metrics, model_name="facebook/VGGT-1B", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            print(f"[VGGT] Loading locally: {model_name}")
            self.vggt_model = VGGT.from_pretrained(model_name, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"[VGGT] Local not found → Downloading from HF: {model_name}")
            self.vggt_model = VGGT.from_pretrained(model_name, local_files_only=False)

        self.vggt_model = self.vggt_model.to(self.device).eval()
        self.metrics = metrics

    def process(self, video_path , thresholds, num_frames, save_visuals=False, out_dir = None):
        """
        **无 I/O 管道**
        处理视频: 采样 -> 推理 -> 重投影 -> 度量。
        """

        frames_np_array = sample_uniform_frames(video_path, n_frames=num_frames)

        # ---------------------------
        # 步骤 2: VGGT 推理 (GPU)
        # ---------------------------
        preds = run_model_gpu(
            frames_np_array,      
            model=self.vggt_model,
            save_path=None,  
        )


        _, _, H, W = preds["images"].shape
        extrinsics, intrinsics = preds["extrinsic"], preds["intrinsic"]
        depths=preds["depth"] 
        gt_frames_for_metrics = frames_np_array 
        results = {}

        # ---------------------------
        # 步骤 3: 重投影
        # ---------------------------
        
        for th in thresholds:
            if save_visuals:
                th_dir = os.path.join(out_dir, f"th{th}")
                reproj_dir = os.path.join(th_dir, "reprojections")
                os.makedirs(reproj_dir, exist_ok=True)
                save_path = reproj_dir
            else:
                save_path = None 

            vertices_3d, colors_rgb = get_colored_pointcloud(preds, mode="depth", conf_thres=th)
            
            reprojected_frames = batch_reproject(
                vertices_3d, colors_rgb, intrinsics, extrinsics, H, W, 
                save_path=save_path
            )
            
            results[th] = self.compute_metrics(
                gt_frames_for_metrics, 
                reprojected_frames, 
                extrinsics,
                intrinsics=intrinsics,      
                depths=depths   
                )
        


        if isinstance(extrinsics, torch.Tensor):
            results["_extrinsic"] = extrinsics.detach().cpu().tolist()
        else:
            results["_extrinsic"] = extrinsics.tolist()
            
        return results

    def compute_metrics(self, gt_frames, rep_frames, extrinsics, intrinsics=None, depths=None):
        """
        度量计算逻辑。
        为 Consistency_Score 添加特殊处理以提取运动范数。
        """
        results = {}
        metrics_needing_extrinsics = ["lpips_motionnorm", "mse_motionnorm", "motion_norm", "Consistency_Score","MVCS"]

        for name, metric_fn in self.metrics.items():
            if name in metrics_needing_extrinsics:
                
                if name == "Consistency_Score":
                    # Consistency_Score.compute 返回 (final_score, motion_norm_val)
                    final_score, motion_norm_val = metric_fn.compute(
                        gt=gt_frames, rep=rep_frames, extrinsics=extrinsics
                    )

                    # 1. 一致性分数
                    results[name] = final_score

                    # 2. 运动范数
                    results["motion_norm"] = motion_norm_val
                elif "MVCS" in name:
                    results[name] = metric_fn.compute(
                        gt=gt_frames, 
                        rep=rep_frames,
                        depths=depths, 
                        intrinsics=intrinsics, 
                        extrinsics=extrinsics
                    )
                else:
                    results[name] = metric_fn.compute(gt=gt_frames, rep=rep_frames, extrinsics=extrinsics)
            else:
                results[name] = metric_fn.compute(gt=gt_frames, rep=rep_frames)

        return results