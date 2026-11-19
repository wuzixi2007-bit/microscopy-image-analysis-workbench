"""Shared interface both detection modes implement, so SamSegmenter never
needs to know whether a prompt came from Grounding-DINO text or PerSAM-style
reference matching."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionPrompt:
    """Points/boxes to feed SAM for one image. Either field may be empty —
    zero-shot mode typically yields boxes, few-shot mode yields points."""

    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    point_labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    scores: np.ndarray = field(default_factory=lambda: np.empty((0,)))


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> DetectionPrompt:
        """Return SAM-ready prompts for the given image (H, W, 3 RGB uint8)."""
        raise NotImplementedError
