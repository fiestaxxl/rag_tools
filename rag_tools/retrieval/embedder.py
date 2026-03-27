"""
Local embedding model for tool embeddings.
"""
import asyncio
from typing import List, Optional, Union
import numpy as np
from functools import lru_cache
import httpx

from rag_tools.config.settings import settings, EmbeddingSettings, APIEmbeddingSettings



class Embedder:
    """Local embedding model using sentence-transformers."""

    def __init__(self, config: Optional[EmbeddingSettings] = None):
        self.config = config or settings.embedding
        self._model = None
        self._embedding_dim = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return

        # Import here to avoid heavy import at module load
        from sentence_transformers import SentenceTransformer

        # Load model in thread to avoid blocking
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        )

        # Get embedding dimension
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        # Set max sequence length
        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = self.config.max_seq_length

        self._initialized = True

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if not self._embedding_dim:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Returns:
            numpy array of embeddings
        """
        if not self._initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size
        normalize = normalize if normalize is not None else self.config.normalize_embeddings

        # Encode in thread to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100,
            )
        )

        return np.array(embeddings)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        embeddings = await self.embed([query])
        return embeddings[0]

    async def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix
        """
        # Normalize if not already
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        embeddings1_norm = embeddings1 / (norm1 + 1e-8)
        embeddings2_norm = embeddings2 / (norm2 + 1e-8)

        # Compute similarity
        return np.dot(embeddings1_norm, embeddings2_norm.T)

    def encode_sync(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Synchronous encode (use this only if already in async context).

        Args:
            texts: Texts to encode
            batch_size: Batch size

        Returns:
            Embeddings
        """
        if not self._initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
        )


class APIEmbedder:
    """Remote embedding model via HTTP API."""

    def __init__(self, config: Optional[APIEmbeddingSettings] = None):
        self.config = config or settings.api_embedding

        self._embedding_dim: Optional[int] = None
        self._initialized = False
        self._url = self.config.url

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._initialized:
            return

        # Optional: warmup request to detect embedding_dim
        try:
            test = await self.get_embeddings(["test"])
            if test:
                self._embedding_dim = len(test[0])
        except Exception:
            pass

        self._initialized = True



    @property
    def embedding_dim(self) -> int:
        if not self._embedding_dim:
            raise RuntimeError("Embedding dim unknown. Call initialize() first.")
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return "api-embedder"

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(self._url, json=texts)
                response.raise_for_status()

                data = response.json()["embeddings"]
                return [np.array(vec, dtype=np.float32) for vec in data]

            except Exception as e:
                print(f"Embedding service error: {str(e)} {self.config}")
                return []

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,   # not used but kept for API parity
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generate embeddings via API.
        """
        if not self._initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]

        normalize = (
            normalize
            if normalize is not None
            else self.config.normalize_embeddings
        )

        embeddings = await self.get_embeddings(texts)

        if not embeddings:
            return np.array([])

        embeddings = np.array(embeddings)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    async def embed_query(self, query: str) -> np.ndarray:
        embeddings = await self.embed([query])
        return embeddings[0]

    async def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Same as local version (pure numpy).
        """
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        embeddings1_norm = embeddings1 / (norm1 + 1e-8)
        embeddings2_norm = embeddings2 / (norm2 + 1e-8)

        return np.dot(embeddings1_norm, embeddings2_norm.T)

    def encode_sync(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sync wrapper (for compatibility).
        """
        return asyncio.run(self.embed(texts))


class CachedEmbedder(Embedder):
    """Embedder with caching for frequently used texts."""

    def __init__(self, config: Optional[EmbeddingSettings] = None, cache_size: int = 10000):
        super().__init__(config)
        self._cache: dict = {}
        self._cache_size = cache_size
        self._cache_order: List[str] = []

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Embed with caching."""
        if isinstance(texts, str):
            texts = [texts]
            was_string = True
        else:
            was_string = False

        # Check cache
        uncached_indices = []
        cached_embeddings = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_embeddings.append((i, self._cache[cache_key]))
            else:
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            uncached_embeddings = await super().embed(uncached_texts, batch_size, normalize)

            # Add to cache
            for idx, text_idx in enumerate(uncached_indices):
                cache_key = self._get_cache_key(texts[text_idx])
                self._add_to_cache(cache_key, uncached_embeddings[idx])

        # Combine results
        result = [None] * len(texts)
        for idx, emb in cached_embeddings:
            result[idx] = emb

        for idx, text_idx in enumerate(uncached_indices):
            result[text_idx] = uncached_embeddings[idx]

        if was_string:
            return np.array([result[0]])

        return np.array(result)

    def _get_cache_key(self, text: str) -> str:
        """Get cache key for text."""
        return text[:100]  # Truncate long texts for cache key

    def _add_to_cache(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to cache."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = embedding
        self._cache_order.append(key)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_order.clear()
