# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input Processing Service
Handles different types of inputs (image, images, colmap, video)
"""

import glob
import json
import os
import subprocess
from typing import List, Tuple

import numpy as np
import typer

from ..utils.read_write_model import read_model


class InputHandler:
    """Base input handler class"""

    @staticmethod
    def validate_path(path: str, path_type: str = "file") -> str:
        """Validate path"""
        if not os.path.exists(path):
            raise typer.BadParameter(f"{path_type} not found: {path}")
        return path

    @staticmethod
    def handle_export_dir(export_dir: str, auto_cleanup: bool = False) -> str:
        """Handle export directory"""
        if os.path.exists(export_dir):
            if auto_cleanup:
                typer.echo(f"Auto-cleaning existing export directory: {export_dir}")
                import shutil

                shutil.rmtree(export_dir)
                os.makedirs(export_dir, exist_ok=True)
            else:
                typer.echo(f"Export directory '{export_dir}' already exists.")
                if typer.confirm("Do you want to clean it and continue?"):
                    import shutil

                    shutil.rmtree(export_dir)
                    os.makedirs(export_dir, exist_ok=True)
                    typer.echo(f"Cleaned export directory: {export_dir}")
                else:
                    typer.echo("Operation cancelled.")
                    raise typer.Exit(0)
        else:
            os.makedirs(export_dir, exist_ok=True)
        return export_dir


class ImageHandler(InputHandler):
    """Single image handler"""

    @staticmethod
    def process(image_path: str) -> List[str]:
        """Process single image"""
        InputHandler.validate_path(image_path, "Image file")
        return [image_path]


class ImagesHandler(InputHandler):
    """Image directory handler"""

    @staticmethod
    def process(images_dir: str, image_extensions: str = "png,jpg,jpeg") -> List[str]:
        """Process image directory"""
        InputHandler.validate_path(images_dir, "Images directory")

        # Parse extensions
        extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        # Find image files
        image_files = []
        for ext in extensions:
            pattern = f"*{ext}"
            image_files.extend(glob.glob(os.path.join(images_dir, pattern)))
            image_files.extend(glob.glob(os.path.join(images_dir, pattern.upper())))

        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort

        if not image_files:
            raise typer.BadParameter(
                f"No image files found in {images_dir} with extensions: {extensions}"
            )

        typer.echo(f"Found {len(image_files)} images to process")
        return image_files


class ColmapHandler(InputHandler):
    """COLMAP data handler"""

    @staticmethod
    def process(
        colmap_dir: str, sparse_subdir: str = ""
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Process COLMAP data"""
        InputHandler.validate_path(colmap_dir, "COLMAP directory")

        # Build paths
        images_dir = os.path.join(colmap_dir, "images")
        if sparse_subdir:
            sparse_dir = os.path.join(colmap_dir, "sparse", sparse_subdir)
        else:
            sparse_dir = os.path.join(colmap_dir, "sparse")

        InputHandler.validate_path(images_dir, "Images directory")
        InputHandler.validate_path(sparse_dir, "Sparse reconstruction directory")

        # Load COLMAP data
        typer.echo("Loading COLMAP reconstruction data...")
        try:
            cameras, images, points3D = read_model(sparse_dir)

            typer.echo(
                f"Loaded COLMAP data: {len(cameras)} cameras, {len(images)} images, "
                f"{len(points3D)} 3D points."
            )

            # Get image files and pose data
            image_files = []
            extrinsics = []
            intrinsics = []

            for image_id, image_data in images.items():
                image_name = image_data.name
                image_path = os.path.join(images_dir, image_name)

                if os.path.exists(image_path):
                    image_files.append(image_path)

                    # Get camera parameters
                    camera = cameras[image_data.camera_id]

                    # Convert quaternion to rotation matrix
                    R = image_data.qvec2rotmat()
                    t = image_data.tvec

                    # Create extrinsic matrix (world to camera)
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = R
                    extrinsic[:3, 3] = t
                    extrinsics.append(extrinsic)

                    # Create intrinsic matrix
                    if camera.model == "PINHOLE":
                        fx, fy, cx, cy = camera.params
                    elif camera.model == "SIMPLE_PINHOLE":
                        f, cx, cy = camera.params
                        fx = fy = f
                    else:
                        # For other models, use basic pinhole approximation
                        fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
                        cx = camera.width / 2
                        cy = camera.height / 2

                    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    intrinsics.append(intrinsic)

            if not image_files:
                raise typer.BadParameter("No valid images found in COLMAP data")

            typer.echo(f"Found {len(image_files)} valid images with pose data")

            return image_files, np.array(extrinsics), np.array(intrinsics)

        except Exception as e:
            raise typer.BadParameter(f"Failed to load COLMAP data: {e}")


class VideoHandler(InputHandler):
    """Video handler"""

    @staticmethod
    def _get_video_info(video_path: str) -> dict:
        """Get video properties using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise typer.BadParameter(f"Cannot open video: {video_path}. ffprobe error: {e}")

        if not info.get("streams"):
            raise typer.BadParameter(f"Cannot open video: {video_path}")

        stream = info["streams"][0]
        # Parse FPS from r_frame_rate (e.g. "30/1")
        r_frame_rate = stream.get("r_frame_rate", "30/1")
        num, den = map(int, r_frame_rate.split("/"))
        video_fps = num / den if den else 30.0
        total_frames = int(stream.get("nb_frames", 0))
        if total_frames == 0:
            # Fallback: compute from duration
            duration = float(stream.get("duration", 0))
            total_frames = int(duration * video_fps)

        return {"fps": video_fps, "total_frames": total_frames}

    @staticmethod
    def process(video_path: str, output_dir: str, fps: float = 1.0) -> List[str]:
        """Process video, extract frames using ffmpeg"""
        InputHandler.validate_path(video_path, "Video file")

        video_info = VideoHandler._get_video_info(video_path)
        video_fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        duration = total_frames / video_fps if video_fps else 0

        # Calculate frame interval (ensure at least 1)
        frame_interval = max(1, int(video_fps / fps))
        actual_fps = video_fps / frame_interval

        typer.echo(f"Video FPS: {video_fps:.2f}, Duration: {duration:.2f}s")

        # Warn if requested FPS is higher than video FPS
        if fps > video_fps:
            typer.echo(
                f"⚠️  Warning: Requested sampling FPS ({fps:.2f}) exceeds video FPS ({video_fps:.2f})",  # noqa: E501
                err=True,
            )
            typer.echo(
                f"⚠️  Using maximum available FPS: {actual_fps:.2f} (extracting every frame)",
                err=True,
            )

        typer.echo(f"Extracting frames at {actual_fps:.2f} FPS (every {frame_interval} frame(s))")

        # Create output directory
        frames_dir = os.path.join(output_dir, "input_images")
        os.makedirs(frames_dir, exist_ok=True)

        # Use ffmpeg to extract frames at the desired FPS
        output_pattern = os.path.join(frames_dir, "%06d.png")
        cmd = [
            "ffmpeg", "-loglevel", "error", "-hide_banner", "-y",
            "-i", video_path,
            "-vf", f"fps={actual_fps}",
            "-start_number", "0",
            output_pattern,
        ]
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise typer.BadParameter(f"Failed to extract frames from video: {e}")

        # Get frame file list
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        )
        if not frame_files:
            raise typer.BadParameter("No frames extracted from video")

        typer.echo(f"Extracted {len(frame_files)} frames to {frames_dir}")
        return [os.path.join(frames_dir, f) for f in frame_files]


def parse_export_feat(export_feat_str: str) -> List[int]:
    """Parse export_feat parameter"""
    if not export_feat_str:
        return []

    try:
        return [int(x.strip()) for x in export_feat_str.split(",") if x.strip()]
    except ValueError:
        raise typer.BadParameter(
            f"Invalid export_feat format: {export_feat_str}. "
            "Use comma-separated integers like '0,1,2'"
        )
