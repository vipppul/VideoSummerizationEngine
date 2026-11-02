"""
Multimodal embedding encoder using open_clip / CLIP.
"""

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import cv2
import torch
import open_clip

from ..config import EmbeddingSettings
from ..utils.logging import setup_logger
from ..utils.gpu import get_optimal_device

logger = setup_logger("embedding.encoder")


class MultimodalEncoder:
    """
    Encodes visual images and text strings into a shared multimodal vector space.
    """

    def __init__(self, settings: Optional[EmbeddingSettings] = None):
        self.settings = settings or EmbeddingSettings()
        self.device = get_optimal_device(self.settings.device)

        logger.info(
            f"Loading open_clip model '{self.settings.model}' "
            f"(pretrained='{self.settings.pretrained}') on {self.device}..."
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.settings.model,
            pretrained=self.settings.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.settings.model)
        self.model.eval()

    def embed_images(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Embeds a list of BGR numpy arrays or PIL Images into unit L2 normalized vectors.
        """
        if not images:
            return np.empty((0, self.settings.dim), dtype=np.float32)

        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # OpenCV BGR to RGB PIL Image
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
            else:
                pil_images.append(img)

        tensors = torch.stack([self.preprocess(p) for p in pil_images]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of text strings into unit L2 normalized vectors.
        """
        if not texts:
            return np.empty((0, self.settings.dim), dtype=np.float32)

        # Truncate texts if very long
        clean_texts = [t[:500] if len(t) > 500 else (t or "blank") for t in texts]
        tokens = self.tokenizer(clean_texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)

    def embed_single_query(self, query: str) -> np.ndarray:
        """Embeds a single query string and returns a 1D vector."""
        return self.embed_texts([query])[0]

