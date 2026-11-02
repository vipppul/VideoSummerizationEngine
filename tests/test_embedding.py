"""
Unit tests for Module 3: Vectorization and Fusion.
"""

import numpy as np
from video_search.embedding.fuser import EmbeddingFuser


def test_embedding_fuser_weighted_average():
    fuser = EmbeddingFuser(strategy="weighted_average", alpha=0.5)

    img_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    txt_vec = np.array([[0.0, 1.0]], dtype=np.float32)

    fused = fuser.fuse(img_vec, txt_vec)

    # Expected shape (1, 2)
    assert fused.shape == (1, 2)
    # Check unit L2 normalization
    norm = np.linalg.norm(fused, axis=-1)[0]
    assert np.isclose(norm, 1.0, atol=1e-5)
    # Check equal weights: [0.5, 0.5] / norm -> ~[0.707, 0.707]
    assert np.isclose(fused[0][0], fused[0][1], atol=1e-5)


def test_embedding_fuser_image_only():
    fuser = EmbeddingFuser(strategy="image_only")
    img_vec = np.array([[0.6, 0.8]], dtype=np.float32)
    txt_vec = np.array([[1.0, 0.0]], dtype=np.float32)

    fused = fuser.fuse(img_vec, txt_vec)
    assert np.allclose(fused, img_vec, atol=1e-5)

