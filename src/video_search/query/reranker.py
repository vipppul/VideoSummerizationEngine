"""
Cross-encoder re-ranking module.
"""

from __future__ import annotations
from typing import List, Optional
from sentence_transformers import CrossEncoder

from ..models import SearchResult
from ..config import QuerySettings
from ..utils.logging import setup_logger

logger = setup_logger("query.reranker")


class CrossEncoderReranker:
    """
    Re-scores candidate SearchResults using a cross-encoder model.
    """

    def __init__(self, settings: Optional[QuerySettings] = None):
        self.settings = settings or QuerySettings()
        self.model = None
        if self.settings.reranker_enabled and self.settings.reranker_type == "cross_encoder":
            try:
                logger.info(f"Loading re-ranker model '{self.settings.reranker_model}'...")
                self.model = CrossEncoder(self.settings.reranker_model)
                logger.info("Cross-encoder re-ranker loaded successfully!")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder ({e}). Will skip re-ranking.")
                self.model = None

    def rerank(self, query: str, results: List[SearchResult], top_n: Optional[int] = None) -> List[SearchResult]:
        if not results:
            return []

        limit = top_n or self.settings.reranker_top_n
        if self.model is None or not self.settings.reranker_enabled:
            return results[:limit]

        # Prepare pairs: query + textual frame evidence
        pairs = []
        for r in results:
            payload = r.payload
            text_context = (
                f"{payload.get('frame_caption', '')} "
                f"Objects: {payload.get('detection_summary', '')} "
                f"Text: {payload.get('ocr_full', '')}"
            ).strip()
            pairs.append([query, text_context])

        try:
            scores = self.model.predict(pairs)
            scored_results = []
            for r, score in zip(results, scores):
                # Sigmoid normalization for raw logits
                import math
                prob = 1.0 / (1.0 + math.exp(-float(score)))
                r.score = float(prob)
                scored_results.append(r)

            # Sort descending by cross-encoder score
            scored_results.sort(key=lambda x: x.score, reverse=True)
            return scored_results[:limit]
        except Exception as e:
            logger.error(f"Error during cross-encoder re-ranking: {e}")
            return results[:limit]

