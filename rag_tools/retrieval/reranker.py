"""
Cross-encoder reranker for improving retrieval results.
"""
import asyncio
from typing import List, Optional, Tuple, Union
import numpy as np
import httpx

from rag_tools.config.settings import settings, RerankerSettings, APIRerankerSettings
from rag_tools.storage.models import RetrievalResult



class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, config: Optional[RerankerSettings] = None):
        self.config = config or settings.reranker
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the reranker model."""
        if self._initialized:
            return

        # Import here to avoid heavy import at module load
        from sentence_transformers import CrossEncoder

        # Load model in thread to avoid blocking
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: CrossEncoder(
                self.config.model_name,
                max_length=512,
                device=self.config.device,
            )
        )

        self._initialized = True

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder.

        Args:
            query: Query text
            results: Initial retrieval results
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not results:
            return []

        if not self._initialized:
            await self.initialize()

        top_k = top_k or self.config.top_k

        # Prepare pairs: (query, document)
        pairs = [(query, self._get_doc_text(r)) for r in results]

        # Score in batch
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        )

        # Handle different score formats
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]

        # Add rerank scores to results
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score
        reranked = sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        return reranked[:top_k]

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents and return indices with scores.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            List of (document_index, score) tuples
        """
        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        top_k = top_k or len(documents)

        # Prepare pairs
        pairs = [(query, doc) for doc in documents]

        # Score
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        )

        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]

        # Sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def _get_doc_text(self, result: RetrievalResult) -> str:
        """Get document text from retrieval result."""
        if result.text:
            return result.text

        parts = [result.name, result.description]
        if result.metadata:
            tags = result.metadata.get("tags", [])
            if tags:
                parts.append(", ".join(tags))

        return " | ".join(filter(None, parts))

class APIReranker:
    """Remote reranker via HTTP API."""

    def __init__(self, config: Optional[APIRerankerSettings] = None):
        self.config = config or settings.reranker

        self._initialized = False

        self._url = config.url

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._initialized:
            return

        self._initialized = True

    @property
    def model_name(self) -> str:
        return  "api-reranker"

    async def get_scores(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Call remote reranker service.

        Expected API contract:
        POST {url}
        {
            "query": "...",
            "documents": ["doc1", "doc2"]
        }

        Response:
        {
            "scores": [0.9, 0.1]
        }
        """

        pairs = [(query, doc) for doc in documents]
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self._url,
                    json=pairs,
                )
                response.raise_for_status()

                scores = response.json()["scores"]

                # Normalize format
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                elif not isinstance(scores, list):
                    scores = [scores]

                return [float(s) for s in scores]

            except Exception as e:
                print(f"Reranker service error: {str(e)}")
                return [0.0] * len(documents)

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using API.
        """
        if not results:
            return []

        if not self._initialized:
            await self.initialize()

        top_k = top_k or self.config.top_k

        # Extract texts
        documents = [self._get_doc_text(r) for r in results]

        # Get scores
        scores = await self.get_scores(query, documents)

        # Attach scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort
        reranked = sorted(
            results,
            key=lambda r: r.rerank_score or 0,
            reverse=True,
        )

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        return reranked[:top_k]

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Return (index, score) tuples.
        """
        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        top_k = top_k or len(documents)

        scores = await self.get_scores(query, documents)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def _get_doc_text(self, result: RetrievalResult) -> str:
        """Same logic as local reranker."""
        if result.text:
            return result.text

        parts = [result.name, result.description]

        if result.metadata:
            tags = result.metadata.get("tags", [])
            if tags:
                parts.append(", ".join(tags))

        return " | ".join(filter(None, parts))


class SimpleReranker:
    """Lightweight reranker using keyword matching."""

    def __init__(self):
        pass

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Simple reranking based on keyword overlap.

        Args:
            query: Query text
            results: Results to rerank
            top_k: Number to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        top_k = top_k or len(results)

        query_terms = set(query.lower().split())

        for result in results:
            doc_terms = set(result.name.lower().split())
            doc_terms.update(result.description.lower().split())

            if result.metadata and "tags" in result.metadata:
                doc_terms.update(tag.lower() for tag in result.metadata["tags"])

            # Jaccard similarity
            overlap = len(query_terms & doc_terms)
            total = len(query_terms | doc_terms)

            result.rerank_score = overlap / total if total > 0 else 0

        # Sort by rerank score
        reranked = sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        return reranked[:top_k]


class HybridReranker:
    """Hybrid reranker combining vector and keyword similarity."""

    def __init__(self, cross_encoder: Optional[Reranker] = None):
        self.cross_encoder = cross_encoder or Reranker()
        self.vector_weight = 0.7
        self.keyword_weight = 0.3

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid reranking.

        Args:
            query: Query text
            results: Initial results
            top_k: Number to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        if not self.cross_encoder._initialized:
            await self.cross_encoder.initialize()

        top_k = top_k or self.cross_encoder.config.top_k

        # Get cross-encoder scores
        query_terms = set(query.lower().split())

        for result in results:
            # Keyword score
            doc_terms = set(result.name.lower().split())
            doc_terms.update(result.description.lower().split())

            overlap = len(query_terms & doc_terms)
            total = len(query_terms | doc_terms)
            keyword_score = overlap / total if total > 0 else 0

            # Vector score (normalized from 0-1)
            vector_score = result.score if result.score else 0

            # Combine scores
            result.rerank_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )

        # Sort by combined score
        reranked = sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        return reranked[:top_k]
