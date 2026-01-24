"""
Metadata format description (meta_data.json):
{
    "groups": [
        {
            "group_id": "video1",
            "prompt": "A cat walking",
            "input_image_path": "images/cat.png",
            "original_video_path": "videos/cat_original.mp4",
            "videos": [
                {
                    "video_path": "generated/cat_variant1.mp4",
                    "generation_id": "1",
                    "consistency_score": 3.5,    # Lower is better
                    "motion_norm": 1.2,          # Motion magnitude
                    "latent_path": "dpo_latents/latent_cat_1.pt",
                    "condition_path": "dpo_latents/condition_cat_1.pt"
                },
                {
                    "video_path": "generated/cat_variant2.mp4",
                    "generation_id": "2",
                    "consistency_score": 8.7,  # Lower is better
                    "motion_norm": 1.1,
                    "latent_path": "dpo_latents/latent_cat_2.pt",
                    "condition_path": "dpo_latents/condition_cat_2.pt"
                }
            ]
        }
    ]
}
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
import numpy as np


def _torch_load_weights_only_compatible(path: Path):

    try:
        return torch.load(path, weights_only=True)
    except TypeError:

        return torch.load(path)


class DPODataset(Dataset):

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        metric_name: str = "consistency_score",
        metric_mode: str = "min",              # "min" = lower is better, "max" = higher is better
        min_gap: float = 0.1,                  # Minimum gap between Winner and Loser
        metric_threshold: Optional[float] = None,  # Threshold for the Winner
        motion_threshold: float = 0.001,         # Minimum motion magnitude
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            metadata_path: Path to the meta_data.json file.
            metric_name: Metric used for ranking (e.g., "consistency_score" or "motion_norm").
            metric_mode: "min" (lower is better) or "max" (higher is better).
            min_gap: Minimum gap/margin between Winner and Loser.
            metric_threshold: Threshold that the Winner must meet.
            motion_threshold: Minimum motion magnitude (used to filter out static videos).
            max_samples: Maximum number of samples to process (for debugging).
        """
        super().__init__()

        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.min_gap = min_gap
        self.metric_threshold = metric_threshold
        self.motion_threshold = motion_threshold

        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            data = json.load(f)

        if 'groups' not in data:
            raise ValueError("Invalid metadata format: missing 'groups' key")

        self.raw_groups = data['groups']
        print(f"Loaded {len(self.raw_groups)} groups")

        # Create preference pairs
        self.preference_pairs = self._create_preference_pairs()
        print(f"Created {len(self.preference_pairs)} preference pairs")

        if max_samples is not None:
            self.preference_pairs = self.preference_pairs[:max_samples]
            print(f"Limited to {max_samples} samples for debugging")

    def _create_preference_pairs(self) -> List[Dict[str, Any]]:
        """
        Construct preference pairs from groups.

        Logic:
        1. Iterate through each group (representing multiple generated variants of the same source video).
        2. Filter out videos missing scores or 'latent_path'.
        3. Filter out videos with 'motion_norm' < threshold.
        4. Sort by 'metric_name' to select the best (winner) and the worst (loser).
        5. Check if 'min_gap' and 'metric_threshold' constraints are met.
        """
        preference_pairs = []

        for group in self.raw_groups:
            group_id = group.get('group_id', 'unknown')
            prompt = group.get('prompt', '')
            input_image_path = group.get('input_image_path')
            original_video_path = group.get('original_video_path')
            videos = group.get('videos', [])

            if len(videos) < 2:
                continue

            # ========================================
            # Filter valid videos
            # ========================================
            valid_videos = []

            for video in videos:
                if (self.metric_name not in video or
                    'motion_norm' not in video):
                    continue

                if ('latent_path' not in video or
                    'condition_path' not in video):
                    continue
                
                full_latent_path = self.base_path / video['latent_path']
                full_cond_path = self.base_path / video['condition_path']

                if not full_latent_path.exists():
                    continue
                if not full_cond_path.exists():
                    continue

                if video['motion_norm'] < self.motion_threshold:
                    continue

                valid_videos.append(video)

            if len(valid_videos) < 2:
                continue

            # ========================================
            # Sorting
            # ========================================
            reverse = (self.metric_mode == "max")
            sorted_videos = sorted(
                valid_videos,
                key=lambda x: x[self.metric_name],
                reverse=reverse
            )

            if self.metric_mode == "min":
                winner = sorted_videos[0]
                loser = sorted_videos[-1]
            else:
                winner = sorted_videos[0]
                loser = sorted_videos[-1]

            winner_metric = winner[self.metric_name]
            loser_metric = loser[self.metric_name]


            # Check metric_threshold
            if self.metric_threshold is not None:
                if self.metric_mode == "min":
                    if winner_metric >= self.metric_threshold:
                        continue
                else:
                    if winner_metric <= self.metric_threshold:
                        continue

            # Check minimum gap
            gap = abs(winner_metric - loser_metric)
            if gap < self.min_gap:
                continue

            pair = {
                "group_id": group_id,
                "prompt": prompt,
                "input_image_path": input_image_path,
                "original_video_path": original_video_path,
                "winner": winner,
                "loser": loser,
                "metric_gap": gap,
            }
            preference_pairs.append(pair)

        return preference_pairs

    def __len__(self) -> int:
        return len(self.preference_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a preference pair.

        Returns:
            {
                "x_win": torch.Tensor,           # [C, F, H, W]
                "x_lose": torch.Tensor,          # [C, F, H, W]
                "prompt_emb": torch.Tensor,  # [seq_len, hidden_dim]
                "image_emb": torch.Tensor,   # [C, H, W] (Optional, I2V only)
                "prompt": str,
                "m_win": float,
                "m_lose": float,
            }
        """
        pair = self.preference_pairs[idx]
        winner = pair["winner"]
        loser = pair["loser"]

        # ========================================
        # Load latents
        # ========================================
        x_win = _torch_load_weights_only_compatible(self.base_path / winner["latent_path"])
        x_lose = _torch_load_weights_only_compatible(self.base_path / loser["latent_path"])

        # ========================================
        # Load conditions
        # ========================================
        shared_condition = _torch_load_weights_only_compatible(self.base_path / winner["condition_path"])

        # Load embeddings
        prompt_emb = shared_condition.get("encoder_hidden_states")
        image_emb = shared_condition.get("image_embeds")

        # ========================================
        # Return dictionary
        # ========================================
        result = {
            "x_win": x_win,
            "x_lose": x_lose,
            "prompt_emb": prompt_emb,
            "prompt": pair["prompt"],  
            "m_win": winner[self.metric_name],
            "m_lose": loser[self.metric_name],
        }

        # Add image embedding (I2V only)
        if image_emb is not None:
            result["image_emb"] = image_emb

        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}

    tensor_keys = ["x_win", "x_lose", "prompt_emb"]
    optional_tensor_keys = ["image_emb"]

    for key in tensor_keys:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])

    for key in optional_tensor_keys:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])

    string_keys = ["prompt"]
    for key in string_keys:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    for key in ["m_win", "m_lose"]:
        if key in batch[0]:
            result[key] = torch.tensor([item[key] for item in batch])

    return result