"""
Full RAG pipeline for tool retrieval.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from rag_tools.config.settings import settings
from rag_tools.storage.models import RetrievalResult
from rag_tools.retrieval.embedder import BaseEmbedder, LocalEmbedder
from rag_tools.retrieval.reranker import BaseReranker, SimpleReranker, CrossEncoderReranker
from rag_tools.retrieval.retriever import ToolRetriever, RetrievalConfig


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    top_k: int = 10
    rerank: bool = True
    rerank_top_k: int = 5
    min_relevance_score: float = 0.3
    use_chunks: bool = False
    server_filter: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None
    hybrid_rerank: bool = False


@dataclass
class PipelineResult:
    """Result from the RAG pipeline."""
    query: str
    results: List[RetrievalResult]
    config: PipelineConfig
    retrieval_time_ms: float
    rerank_time_ms: float = 0
    total_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRetrievalPipeline:
    """
    Full RAG pipeline for tool retrieval with optional reranking.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        retriever: Optional[ToolRetriever] = None,
    ):
        self.embedder = embedder or LocalEmbedder()
        self.reranker = reranker
        self.retriever = retriever

        # Simple reranker as fallback
        self.simple_reranker = SimpleReranker()

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.embedder.initialize()

        if self.reranker:
            await self.reranker.initialize()

        if self.retriever:
            self.retriever.embedder = self.embedder
            await self.retriever.initialize()

    async def retrieve(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Retrieve tools for a query using the full pipeline.

        Args:
            query: Search query
            config: Pipeline configuration

        Returns:
            PipelineResult with retrieval results and timing info
        """
        import time

        config = config or PipelineConfig()
        start_time = time.time()

        # Initialize if needed
        if not self.retriever:
            from rag_tools.storage.qdrant_client import QdrantClientWrapper
            qdrant = QdrantClientWrapper()
            await qdrant.connect()
            self.retriever = ToolRetriever(qdrant, self.embedder)
            await self.retriever.initialize()

        retrieval_start = time.time()

        # Build retrieval config
        retrieval_config = RetrievalConfig(
            top_k=config.top_k * 2 if config.rerank else config.top_k,
            min_score=config.min_relevance_score,
            use_chunks=config.use_chunks,
            server_filter=config.server_filter,
            tags_filter=config.tags_filter,
        )

        # Initial retrieval
        results = await self.retriever.retrieve(query, retrieval_config)

        retrieval_time = (time.time() - retrieval_start) * 1000

        rerank_time = 0
        if config.rerank and results:
            rerank_start = time.time()

            # Use cross-encoder reranker if available
            if self.reranker:
                results = await self.reranker.rerank(
                    query,
                    results,
                    top_k=config.rerank_top_k,
                )
            else:
                # Use simple reranker
                results = await self.simple_reranker.rerank(
                    query,
                    results,
                    top_k=config.rerank_top_k,
                )

            rerank_time = (time.time() - rerank_start) * 1000

        total_time = (time.time() - start_time) * 1000

        return PipelineResult(
            query=query,
            results=results,
            config=config,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            metadata={
                "total_results": len(results),
                "reranked": config.rerank,
            },
        )

    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[PipelineConfig] = None,
    ) -> List[PipelineResult]:
        """
        Retrieve tools for multiple queries.

        Args:
            queries: List of search queries
            config: Pipeline configuration

        Returns:
            List of PipelineResults
        """
        if not self.retriever:
            from rag_tools.storage.qdrant_client import QdrantClientWrapper
            qdrant = QdrantClientWrapper()
            await qdrant.connect()
            self.retriever = ToolRetriever(qdrant, self.embedder)
            await self.retriever.initialize()

        config = config or PipelineConfig()

        # Build retrieval config
        retrieval_config = RetrievalConfig(
            top_k=config.top_k * 2 if config.rerank else config.top_k,
            min_score=config.min_relevance_score,
            use_chunks=config.use_chunks,
            server_filter=config.server_filter,
            tags_filter=config.tags_filter,
        )

        # Batch retrieval
        all_results = await self.retriever.retrieve_batch(queries, retrieval_config)

        # Rerank if configured
        pipeline_results = []
        for query, results in zip(queries, all_results):
            if config.rerank and results:
                if self.reranker:
                    results = await self.reranker.rerank(
                        query,
                        results,
                        top_k=config.rerank_top_k,
                    )
                else:
                    results = await self.simple_reranker.rerank(
                        query,
                        results,
                        top_k=config.rerank_top_k,
                    )

            pipeline_results.append(
                PipelineResult(
                    query=query,
                    results=results,
                    config=config,
                    retrieval_time_ms=0,
                    total_time_ms=0,
                )
            )

        return pipeline_results

    async def retrieve_with_fallback(
        self,
        query: str,
        configs: Optional[List[PipelineConfig]] = None,
    ) -> PipelineResult:
        """
        Retrieve with fallback strategies.

        Tries multiple configurations until results are found.

        Args:
            query: Search query
            configs: List of fallback configs

        Returns:
            First successful result
        """
        if configs is None:
            configs = [
                PipelineConfig(top_k=10, rerank=True, min_relevance_score=0.3),
                PipelineConfig(top_k=20, rerank=True, min_relevance_score=0.1),
                PipelineConfig(top_k=30, rerank=False, min_relevance_score=0.0),
            ]

        for config in configs:
            result = await self.retrieve(query, config)
            if result.results:
                return result

        # Return empty result with last config
        return await self.retrieve(query, configs[-1])

    def get_default_tools(self) -> List[str]:
        """Get list of default tool names that should always be available."""
        # These could be configured externally
        return []


class StreamingPipeline:
    """Pipeline with streaming results."""

    def __init__(self, pipeline: ToolRetrievalPipeline):
        self.pipeline = pipeline

    async def retrieve_stream(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Retrieve tools with streaming results.

        Args:
            query: Search query
            config: Pipeline configuration

        Yields:
            Results as they become available
        """
        result = await self.pipeline.retrieve(query, config)

        # Yield initial results quickly
        for r in result.results[:3]:
            yield r

        # Yield remaining results
        for r in result.results[3:]:
            yield r


def create_pipeline(
    embedder: Optional[BaseEmbedder] = None,
    use_cross_encoder: bool = True,
    device: str = "cpu",
) -> ToolRetrievalPipeline:
    """
    Factory function to create a retrieval pipeline.

    Args:
        embedder: Optional embedder instance
        use_cross_encoder: Whether to use cross-encoder reranker
        device: Device for models (cpu/cuda)

    Returns:
        Configured retrieval pipeline
    """
    from rag_tools.config.settings import settings

    # Update settings based on device
    settings.embedding.device = device
    settings.cross_encoder_reranker.device = device

    # Create embedder
    if embedder is None:
        embedder = LocalEmbedder()

    # Create reranker
    reranker = None
    if use_cross_encoder:
        reranker = CrossEncoderReranker()

    return ToolRetrievalPipeline(embedder=embedder, reranker=reranker)
