"""
Hybrid search execution across temporal vector index and metadata filters.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from ..models import SearchResult
from ..storage.vector_store import QdrantVectorStore
from ..embedding.encoder import MultimodalEncoder
from .analyzer import AnalyzedQuery
from ..utils.logging import setup_logger

logger = setup_logger("query.searcher")


class HybridSearcher:
    """
    Executes hybrid semantic vector search with metadata filtering and temporal clustering.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        encoder: MultimodalEncoder,
        score_threshold: float = 0.15,
        temporal_window_sec: float = 5.0,
    ):
        self.vector_store = vector_store
        self.encoder = encoder
        self.score_threshold = score_threshold
        self.temporal_window_sec = temporal_window_sec

    def search(self, analyzed: AnalyzedQuery) -> List[SearchResult]:
        # Encode query text into shared vector space
        query_vec = self.encoder.embed_single_query(analyzed.search_text)

        # Primary vector search
        results = self.vector_store.search(
            query_vector=query_vec,
            vector_name=analyzed.vector_name,
            filter_obj=analyzed.filters,
            limit=analyzed.top_k,
            score_threshold=self.score_threshold,
        )

        # Graceful fallback: If strict filter yielded 0 results, retry without filter
        if not results and analyzed.filters is not None:
            logger.info("Filtered search returned 0 candidates. Falling back to semantic search without strict filter...")
            results = self.vector_store.search(
                query_vector=query_vec,
                vector_name=analyzed.vector_name,
                filter_obj=None,
                limit=analyzed.top_k,
                score_threshold=self.score_threshold,
            )

        logger.info(
            f"Search for '{analyzed.search_text}' on vector '{analyzed.vector_name}' "
            f"returned {len(results)} raw candidates"
        )

        return results

