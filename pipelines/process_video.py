import os

import numpy as np
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse, unproject_depth
from utils import batch_reproject, get_colored_pointcloud, run_model_gpu, sample_uniform_frames
from vggt.models.vggt import VGGT


DEFAULT_VGGT_MODEL = "facebook/VGGT-1B"
DEFAULT_DA3_MODEL = "depth-anything/DA3-Large"


class VideoProcessor:
    def __init__(self, metrics, model_name=None, device=None, backbone=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics
        self.backbone = self._resolve_backbone(backbone, model_name)
        self.model_name = model_name or (
            DEFAULT_DA3_MODEL if self.backbone == "da3" else DEFAULT_VGGT_MODEL
        )

        if self.backbone == "da3":
            self.model = self._load_da3_model(self.model_name)
        else:
            self.model = self._load_vggt_model(self.model_name)

    def _resolve_backbone(self, backbone, model_name):
        if backbone:
            return backbone.lower()

        env_backbone = os.getenv("VIDEO_PROCESSOR_BACKBONE")
        if env_backbone:
            return env_backbone.lower()

        if model_name and "depth-anything" in model_name.lower():
            return "da3"
        return "vggt"

    def _load_vggt_model(self, model_name):
        try:
            print(f"[VGGT] Loading locally: {model_name}")
            model = VGGT.from_pretrained(model_name, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"[VGGT] Local not found -> Downloading from HF: {model_name}")
            model = VGGT.from_pretrained(model_name, local_files_only=False)
        return model.to(self.device).eval()

    def _load_da3_model(self, model_name):
        try:
            print(f"[DA3] Loading locally: {model_name}")
            model = DepthAnything3.from_pretrained(model_name, local_files_only=True)
        except Exception:
            print(f"[DA3] Local not found -> Downloading from HF: {model_name}")
            model = DepthAnything3.from_pretrained(model_name)
        return model.to(self.device).eval()

    def process(self, video_path, thresholds, num_frames, save_visuals=False, out_dir=None):
        if self.backbone == "da3":
            return self._process_da3(video_path, thresholds, num_frames, save_visuals, out_dir)
        return self._process_vggt(video_path, thresholds, num_frames, save_visuals, out_dir)

    def _process_vggt(self, video_path, thresholds, num_frames, save_visuals=False, out_dir=None):
        frames_np_array = sample_uniform_frames(video_path, n_frames=num_frames)
        preds = run_model_gpu(frames_np_array, model=self.model, save_path=None)

        _, _, height, width = preds["images"].shape
        extrinsics = preds["extrinsic"]
        intrinsics = preds["intrinsic"]
        depths = preds["depth"]

        results = {}
        for th in thresholds:
            save_path = None
            if save_visuals:
                th_dir = os.path.join(out_dir, f"th{th}")
                reproj_dir = os.path.join(th_dir, "reprojections")
                os.makedirs(reproj_dir, exist_ok=True)
                save_path = reproj_dir

            vertices_3d, colors_rgb = get_colored_pointcloud(preds, mode="depth", conf_thres=th)
            reprojected_frames = batch_reproject(
                vertices_3d, colors_rgb, intrinsics, extrinsics, height, width, save_path=save_path
            )

            results[th] = self.compute_metrics(
                frames_np_array,
                reprojected_frames,
                extrinsics,
                intrinsics=intrinsics,
                depths=depths,
            )

        results["_extrinsic"] = self._to_serializable(extrinsics)
        return results

    def _process_da3(self, video_path, thresholds, num_frames, save_visuals=False, out_dir=None):
        frames_np_array = sample_uniform_frames(video_path, n_frames=num_frames)
        preds, gt_frames, depths, intrinsics, extrinsics = self._build_da3_predictions(
            frames_np_array
        )

        height, width = gt_frames.shape[-2:]
        results = {}
        for th in thresholds:
            save_path = None
            if save_visuals:
                th_dir = os.path.join(out_dir, f"th{th}")
                reproj_dir = os.path.join(th_dir, "reprojections")
                os.makedirs(reproj_dir, exist_ok=True)
                save_path = reproj_dir

            vertices_3d, colors_rgb = get_colored_pointcloud(preds, mode="depth", conf_thres=th)
            reprojected_frames = batch_reproject(
                vertices_3d, colors_rgb, intrinsics, extrinsics, height, width, save_path=save_path
            )

            results[th] = self.compute_metrics(
                gt_frames,
                reprojected_frames,
                extrinsics,
                intrinsics=intrinsics,
                depths=depths,
            )

        results["_extrinsic"] = self._to_serializable(extrinsics)
        return results

    def _build_da3_predictions(self, frames_np_array):
        prediction = self.model.inference(
            [frames_np_array[i] for i in range(len(frames_np_array))]
        )

        images = torch.from_numpy(prediction.processed_images).float().to(self.device)
        if images.max() > 1.0:
            images = images / 255.0
        images = images.permute(0, 3, 1, 2).contiguous()

        extrinsics = torch.from_numpy(prediction.extrinsics).float().to(self.device)
        intrinsics = torch.from_numpy(prediction.intrinsics).float().to(self.device)
        depths = torch.from_numpy(prediction.depth).float().to(self.device)
        confidences = (
            torch.from_numpy(prediction.conf).float().to(self.device)
            if prediction.conf is not None
            else torch.ones_like(depths)
        )

        c2w = affine_inverse(extrinsics)
        world_points = unproject_depth(
            depths.unsqueeze(0).unsqueeze(-1),
            intrinsics.unsqueeze(0),
            c2w.unsqueeze(0),
        ).squeeze(0)

        preds = {
            "world_points_from_depth": world_points,
            "depth_conf": confidences,
            "images": images,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": depths,
        }
        return preds, images, depths, intrinsics, extrinsics

    def compute_metrics(self, gt_frames, rep_frames, extrinsics, intrinsics=None, depths=None):
        """
        Compute all registered metrics.
        Both VGGT and DA3 backbones use reprojected frames + camera extrinsics
        for Consistency_Score. Backbone selection happens in VideoProcessor.
        MVCS needs depths, intrinsics, and extrinsics.
        """
        results = {}

        for name, metric_fn in self.metrics.items():
            if name == "Consistency_Score":
                final_score, motion_norm_val = metric_fn.compute(
                    gt=gt_frames, rep=rep_frames, extrinsics=extrinsics
                )
                results[name] = final_score
                results["motion_norm"] = motion_norm_val

            elif name == "MVCS":
                results[name] = metric_fn.compute(
                    gt=gt_frames,
                    rep=rep_frames,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
            else:
                results[name] = metric_fn.compute(gt=gt_frames, rep=rep_frames)

        return results

    def _to_serializable(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value.tolist()
