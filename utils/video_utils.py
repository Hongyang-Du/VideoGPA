import os
import cv2
import numpy as np
from decord import VideoReader, cpu

# ----------------------------------------------------
# video_utils.py
# ----------------------------------------------------

def center_crop_and_resize(frame, size=518):
    """Center crop then resize to size×size."""
    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = frame[top:top + side, left:left + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)

def sample_uniform_frames(video_path: str, n_frames: int = 48):

    try:
        # ctx=cpu(0) 确保在 CPU 上解码，避免占用过量显存
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"Decord 无法读取视频 {video_path}: {e}")

    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"视频帧数为 0: {video_path}")

    n_eff = min(n_frames, total)
    indices = np.linspace(0, total - 1, n_eff).astype(int)

    # return [T, H, W, C] Tensor
    frames_decord = vr.get_batch(indices).asnumpy() 

    processed_frames = []
    for i in range(len(frames_decord)):
        frame = frames_decord[i]
        
        frame = center_crop_and_resize(frame, size=518)
        processed_frames.append(frame)

    return np.stack(processed_frames, axis=0)
