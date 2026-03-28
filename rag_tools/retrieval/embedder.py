"""
Local embedding model for tool embeddings.
"""
import asyncio
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from functools import lru_cache
import httpx

from rag_tools.config.settings import settings, EmbeddingSettings, APIEmbeddingSettings


# =========================
# Base Class
# =========================

class BaseEmbedder(ABC):
    """Abstract base embedder defining common interface and utilities."""

    def __init__(self):
        self._initialized = False
        self._embedding_dim: Optional[int] = None

    # ---------- Lifecycle ----------

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("Embedder not initialized.")
        return self._embedding_dim

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    # ---------- Core API ----------

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = True,
    ) -> np.ndarray:
        if not self._initialized:
            await self.initialize()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = await self._embed_impl(texts, batch_size)

        embeddings = np.array(embeddings)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings[0] if single_input else embeddings

    @abstractmethod
    async def _embed_impl(
        self,
        texts: List[str],
        batch_size: Optional[int],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Actual embedding implementation (must be overridden)."""
        pass

    async def embed_query(self, query: str) -> np.ndarray:
        return await self.embed(query)

    async def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        e1 = embeddings1 / (norm1 + 1e-8)
        e2 = embeddings2 / (norm2 + 1e-8)

        return np.dot(e1, e2.T)

    def encode_sync(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return asyncio.run(self.embed(texts, batch_size=batch_size))


# =========================
# Local Embedder
# =========================

class Embedder(BaseEmbedder):
    """Sentence-transformers embedder."""

    def __init__(self, config: Optional[EmbeddingSettings] = None):
        super().__init__()
        self.config = config or settings.embedding
        self._model = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        
        # Import here to avoid heavy import at module load
        from sentence_transformers import SentenceTransformer

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        )

        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = self.config.max_seq_length

        self._initialized = True

    @property
    def model_name(self) -> str:
        return self.config.model_name

    async def _embed_impl(
        self,
        texts: List[str],
        batch_size: Optional[int],
    ) -> np.ndarray:
        batch_size = batch_size or self.config.batch_size

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
            )
        )
    
# =========================
# API Embedder
# =========================

class APIEmbedder(BaseEmbedder):
    """Remote HTTP embedding."""

    def __init__(self, config: Optional[APIEmbeddingSettings] = None):
        super().__init__()
        self.config = config or settings.api_embedding
        self._url = self.config.url

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            test = await self._embed_impl(["test"], None)
            if test:
                self._embedding_dim = len(test[0])
        except Exception:
            pass

        self._initialized = True

    @property
    def model_name(self) -> str:
        return "api-embedder"

    async def _embed_impl(
        self,
        texts: List[str],
        batch_size: Optional[int],
    ) -> List[np.ndarray]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(self._url, json=texts)
                response.raise_for_status()
                data = response.json()["embeddings"]
                return [np.array(vec, dtype=np.float32) for vec in data]

            except Exception as e:
                print(f"Embedding service error: {e}")
                return []

# =========================
# Cached Wrapper
# =========================

class CachedEmbedder(BaseEmbedder):
    """Cache wrapper around another embedder."""

    def __init__(self, base: BaseEmbedder, cache_size: int = 10000):
        super().__init__()
        self.base = base
        self._cache = {}
        self._order = []
        self._cache_size = cache_size

    async def initialize(self) -> None:
        await self.base.initialize()
        self._embedding_dim = self.base.embedding_dim
        self._initialized = True

    @property
    def model_name(self) -> str:
        return f"{self.base.model_name}-cached"

    async def _embed_impl(
        self,
        texts: List[str],
        batch_size: Optional[int],
    ) -> List[np.ndarray]:

        results = [None] * len(texts)
        uncached = []
        uncached_idx = []

        for i, text in enumerate(texts):
            key = self._key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached.append(text)
                uncached_idx.append(i)

        if uncached:
            embeddings = await self.base.embed(uncached, batch_size, normalize=False)

            for i, emb in zip(uncached_idx, embeddings):
                key = self._key(texts[i])
                self._add(key, emb)
                results[i] = emb

        return results

    def _key(self, text: str) -> str:
        return text[:100]

    def _add(self, key: str, value: np.ndarray):
        if len(self._cache) >= self._cache_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._order.append(key)

    def clear_cache(self):
        self._cache.clear()
        self._order.clear()