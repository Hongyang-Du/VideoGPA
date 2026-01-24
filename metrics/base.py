from abc import ABC, abstractmethod
from typing import Any

class Metric(ABC):
    """
    Base class for all metrics in the system.

    Metrics operate on entire videos:
        gt:  [T, H, W, C]
        rep: [T, H, W, C]

    Each metric returns a single float value for the entire video.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, *, gt, rep, **kwargs) -> float:
        """
        Compute the metric for the full video.

        Args:
            gt:  Ground-truth video frames, numpy array of shape [T, H, W, C]
            rep: Reprojected/generated video frames, same shape as gt
            **kwargs: Additional information (e.g., camera extrinsics)

        Returns:
            float: a single scalar metric value.
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Allows calling metric(gt=..., rep=...) directly."""
        return self.compute(*args, **kwargs)
