"""
Reranker abstractions and implementations.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import httpx
import re

from rag_tools.config.settings import settings, RerankerSettings, APIRerankerSettings
from rag_tools.storage.models import RetrievalResult


# =========================
# Base Class
# =========================

class BaseReranker(ABC):
    """Abstract reranker with shared ranking logic."""

    def __init__(self):
        self._initialized = False

    # ---------- Lifecycle ----------

    async def initialize(self) -> None:
        self._initialized = True

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    # ---------- Core API ----------

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:

        if not results:
            return []

        if not self._initialized:
            await self.initialize()

        documents = [self._get_doc_text(r) for r in results]
        scores = await self._score(query, documents)

        # attach scores
        for r, s in zip(results, scores):
            r.rerank_score = float(s)

        return self._sort_and_truncate(results, top_k)

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:

        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        scores = await self._score(query, documents)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        return indexed[: (top_k or len(documents))]

    @abstractmethod
    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Return relevance scores."""
        pass

    # ---------- Helpers ----------

    def _sort_and_truncate(
        self,
        results: List[RetrievalResult],
        top_k: Optional[int],
    ) -> List[RetrievalResult]:

        results = sorted(
            results,
            key=lambda r: r.rerank_score or 0,
            reverse=True,
        )

        for i, r in enumerate(results):
            r.rank = i + 1

        return results[: (top_k or len(results))]

    def _get_doc_text(self, result: RetrievalResult) -> str:
        if result.text:
            return result.text

        parts = [result.name, result.description]

        if result.metadata:
            tags = result.metadata.get("tags", [])
            if tags:
                parts.append(", ".join(tags))

        return " | ".join(filter(None, parts))


# =========================
# Cross Encoder
# =========================

class CrossEncoderReranker(BaseReranker):
    """Sentence-transformers cross-encoder."""

    def __init__(self, config: Optional[RerankerSettings] = None):
        super().__init__()
        self.config = config or settings.reranker
        self._model = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        
        # Import here to avoid heavy import at module load
        from sentence_transformers import CrossEncoder

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length or 512, 
                device=self.config.device,
            )
        )

        self._initialized = True

    @property
    def model_name(self) -> str:
        return self.config.model_name

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:

        pairs = [(query, doc) for doc in documents]

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

        return [float(s) for s in scores]
    

# =========================
# API Reranker
# =========================

class APIReranker(BaseReranker):
    """Remote reranker."""

    def __init__(self, config: Optional[APIRerankerSettings] = None):
        super().__init__()
        self.config = config or settings.reranker
        self._url = self.config.url

    @property
    def model_name(self) -> str:
        return "api-reranker"

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:


        pairs = [(query, doc) for doc in documents]
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self._url,
                    json=pairs,
                )
                response.raise_for_status()

                scores = response.json()["scores"]

                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                elif not isinstance(scores, list):
                    scores = [scores]

                return [float(s) for s in scores]

            except Exception as e:
                print(f"Reranker API error: {e}")
                return [0.0] * len(documents)
            

# =========================
# Simple (Keyword)
# =========================

class SimpleReranker(BaseReranker):
    """Keyword overlap reranker."""

    @property
    def model_name(self) -> str:
        return "keyword"

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:

        query_terms = set(query.lower().split())
        scores = []

        for doc in documents:
            doc_terms = set(doc.lower().split())

            overlap = len(query_terms & doc_terms)
            total = len(query_terms | doc_terms)

            scores.append(overlap / total if total > 0 else 0.0)

        return scores
    
# =========================
# BM25 Reranker
# =========================

class BM25Reranker(BaseReranker):
    """BM25-based reranker using rank_bm25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.k1 = k1
        self.b = b

    @property
    def model_name(self) -> str:
        return "bm25"

    @classmethod
    def tokenize(text: str):
        return re.findall(r"\w+", text.lower())

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:

        # Import here (optional dependency)
        from rank_bm25 import BM25Okapi

        # Tokenization (simple baseline)
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        tokenized_query = self.tokenize(query)

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        scores = bm25.get_scores(tokenized_query)

        return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
    

# =========================
# Hybrid (Composable)
# =========================

class HybridReranker(BaseReranker):
    """Combines multiple rerankers."""

    def __init__(
        self,
        rerankers: List[BaseReranker],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)

    @property
    def model_name(self) -> str:
        return "hybrid"

    async def initialize(self) -> None:
        if self._initialized:
            return

        for r in self.rerankers:
            await r.initialize()

        self._initialized = True

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:

        all_scores = []

        for reranker in self.rerankers:
            scores = await reranker._score(query, documents)
            all_scores.append(scores)

        # weighted sum
        final_scores = []
        for i in range(len(documents)):
            score = sum(
                w * all_scores[j][i]
                for j, w in enumerate(self.weights)
            )
            final_scores.append(score)

        return final_scores
   