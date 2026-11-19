"""Lightweight particle-vs-background classifier on frozen SAM embeddings,
trained offline (scripts/build_particle_classifier_dataset.py +
scripts/train_particle_classifier.py) to replace a fixed cosine-similarity
threshold when scanning a target image for *every* instance that resembles
the reference, not just the single best match (see README.md "穷举检测"
and "训练轻量分类头" sections for why the raw-threshold approach failed and
what this fixes).

Plain numpy 1-hidden-layer MLP (ReLU + sigmoid output) -- no new ML
dependency. Input is 257-d: the raw 256-d SAM embedding at a grid cell,
plus that cell's cosine similarity to the active reference (so the
classifier learns a joint boundary instead of a hard external AND-gate).
A linear (logistic-regression) version came first; the MLP replaced it
because recall stayed flat under a linear boundary (see README.md).
"""

import numpy as np


class ParticleClassifier:
    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.mean, self.std = data["mean"], data["std"]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """`features`: (..., 257) array, last dim = [256-d embedding, similarity-to-reference].
        Returns particle probability, same leading shape."""
        x = (features - self.mean) / self.std
        a1 = np.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))[..., 0]
