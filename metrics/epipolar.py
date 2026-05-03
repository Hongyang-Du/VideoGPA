import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from kornia.geometry.epipolar import find_fundamental, sampson_epipolar_distance

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from metrics.base import Metric


class KeypointMatcher(ABC):
    """Abstract base class for keypoint detection and matching algorithms."""

    @abstractmethod
    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        pass


class SIFTMatcher(KeypointMatcher):
    """SIFT-based keypoint detection and matching."""

    def __init__(self, ratio_thresh: float = 0.75, min_matches: int = 20):
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, frame: np.ndarray):
        if len(frame.shape) == 3 and frame.shape[-1] == 3:
            if frame.shape[0] == 3:  # CHW format
                frame = frame.transpose(1, 2, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        kp1, desc1 = self.detect_and_compute(frame1)
        kp2, desc2 = self.detect_and_compute(frame2)

        metadata = {'keypoints1': len(kp1), 'keypoints2': len(kp2), 'descriptor_type': 'sift'}

        if len(kp1) < 8 or len(kp2) < 8 or desc1 is None or desc2 is None:
            return None, None, 0, metadata

        matches = self.match_features(desc1, desc2)

        if len(matches) < self.min_matches:
            metadata['error'] = f'Too few matches ({len(matches)})'
            return None, None, len(matches), metadata

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        return pts1, pts2, len(matches), metadata


class LightGlueMatcher(KeypointMatcher):
    """
    LightGlue-based keypoint detection using official cvg/LightGlue package.
    (More robust, does not depend on Kornia version)
    """

    def __init__(self, min_matches: int = 20, device: str = "cuda"):
        self.min_matches = min_matches
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            # Load official SuperPoint and LightGlue
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        except Exception as e:
            print(f"Error loading LightGlue models: {e}")
            raise e

    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        try:
            # 1. Preprocess: numpy (H,W,C) -> torch (1,1,H,W) normalized
            img1 = self._frame2tensor(frame1)
            img2 = self._frame2tensor(frame2)

            with torch.no_grad():
                # 2. Extract features
                feats0 = self.extractor.extract(img1)
                feats1 = self.extractor.extract(img2)

                # 3. Match
                matches01 = self.matcher({'image0': feats0, 'image1': feats1})

                # Remove batch dimension (kpts: 1xNx2 -> Nx2)
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

                # Get matched point indices
                matches = matches01['matches'] # indices
                kpts0 = feats0['keypoints'][matches[..., 0]]
                kpts1 = feats1['keypoints'][matches[..., 1]]

            num_matches = len(kpts0)
            metadata = {'descriptor_type': 'lightglue', 'total_matches': num_matches}

            if num_matches < self.min_matches:
                metadata['error'] = f'Too few matches ({num_matches})'
                return None, None, num_matches, metadata

            # Convert back to numpy
            pts1 = kpts0.cpu().numpy()
            pts2 = kpts1.cpu().numpy()

            return pts1, pts2, num_matches, metadata

        except Exception as e:
            return None, None, 0, {'error': str(e)}

    def _frame2tensor(self, frame):
        """Helper: numpy [H,W,C] -> torch [1,1,H,W] normalized"""
        if frame.shape[0] == 3: # Handle CHW
            frame = frame.transpose(1, 2, 0)
            
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # LightGlue expects [B, 1, H, W] in range [0, 1]
        return torch.from_numpy(gray / 255.0).float()[None, None].to(self.device)


class EpipolarMetric(Metric):
    """
    Video-level Epipolar Consistency Metric.
    Computes epipolar geometry error between consecutive frames using SIFT or LightGlue.
    """

    def __init__(self, descriptor_type: str = "sift", ratio_thresh: float = 0.75,
                 min_matches: int = 20, device: str = None):
        super().__init__(name="Epipolar")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.descriptor_type = descriptor_type

        if descriptor_type == "sift":
            self.matcher = SIFTMatcher(ratio_thresh=ratio_thresh, min_matches=min_matches)
        elif descriptor_type == "lightglue":
            self.matcher = LightGlueMatcher(min_matches=min_matches, device=self.device)
        else:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

    def compute(self, *, gt, rep, **kwargs) -> float:
        gt_frames = self._to_numpy_thwc(gt)
        # rep_frames = self._to_numpy_thwc(rep) # We only compute GT consistency, or only Rep consistency

        errors = []
        # Here we compute the temporal consistency of the input video itself
        for i in range(len(gt_frames) - 1):
            error = self._compute_frame_pair_error(gt_frames[i], gt_frames[i + 1])
            if error is not None:
                errors.append(error)

        if len(errors) == 0:
            return -1.0

        return float(np.mean(errors))

    def _compute_frame_pair_error(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[float]:
        pts1, pts2, num_matches, metadata = self.matcher.get_matched_points(frame1, frame2)

        if pts1 is None or pts2 is None:
            return None

        # Compute fundamental matrix (still using Kornia's geometry module, which is safe)
        F_matrix, points1_tensor, points2_tensor = self._compute_fundamental_matrix(pts1, pts2)

        if F_matrix is None:
            return None

        # Compute Sampson distance
        sampson_distances = self._compute_sampson_distances(F_matrix, points1_tensor, points2_tensor)

        if sampson_distances is None:
            return None

        return float(np.mean(sampson_distances))

    def _compute_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray):
        try:
            points1_tensor = torch.from_numpy(pts1).float().unsqueeze(0).to(self.device)
            points2_tensor = torch.from_numpy(pts2).float().unsqueeze(0).to(self.device)
            F_matrix = find_fundamental(points1_tensor, points2_tensor)
            if F_matrix is None or torch.isnan(F_matrix).any():
                return None, None, None
            return F_matrix, points1_tensor, points2_tensor
        except Exception:
            return None, None, None

    def _compute_sampson_distances(self, F_matrix, points1_tensor, points2_tensor):
        try:
            sampson_dist_squared = sampson_epipolar_distance(
                points1_tensor, points2_tensor, F_matrix, squared=True
            )
            sampson_dist = torch.sqrt(sampson_dist_squared + 1e-8).squeeze()
            return sampson_dist.cpu().numpy()
        except Exception:
            return None

    def _to_numpy_thwc(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if not isinstance(x, np.ndarray):
            raise ValueError(f"Expected Tensor or numpy array, got {type(x)}")
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        if x.shape[1] == 3 or x.shape[1] == 1:
            x = x.transpose(0, 2, 3, 1)
        if x.min() < 0:
            x = (x + 1.0) * 127.5
        elif x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x