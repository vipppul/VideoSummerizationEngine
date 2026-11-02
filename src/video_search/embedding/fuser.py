"""
Multimodal embedding fusion strategies.
"""

from __future__ import annotations
import numpy as np


class EmbeddingFuser:
    """
    Combines visual embeddings and text embeddings into a unified representation.
    """

    def __init__(self, strategy: str = "weighted_average", alpha: float = 0.6):
        self.strategy = strategy.lower()
        self.alpha = float(alpha)

    def fuse(self, image_vecs: np.ndarray, text_vecs: np.ndarray) -> np.ndarray:
        """
        Fuses image and text vectors according to the configured strategy
        and normalizes the result to unit length.
        """
        if len(image_vecs) == 0:
            return text_vecs
        if len(text_vecs) == 0:
            return image_vecs

        if self.strategy == "image_only":
            fused = image_vecs.copy()
        elif self.strategy == "text_only":
            fused = text_vecs.copy()
        elif self.strategy == "concatenate":
            fused = np.concatenate([image_vecs, text_vecs], axis=-1)
        else:  # weighted_average
            fused = self.alpha * image_vecs + (1.0 - self.alpha) * text_vecs

        # L2 Normalize
        norms = np.linalg.norm(fused, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return (fused / norms).astype(np.float32)

