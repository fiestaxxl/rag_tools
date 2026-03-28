"""
RAG Tools Module - Main entry point.

A RAG-based tool retrieval system for MCP (Model Context Protocol) servers.
"""
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from rag_tools.config.settings import settings, Settings
from rag_tools.storage.models import (
    MCPServer, MCPTool, RetrievalResult, ToolStatus
)
from rag_tools.storage.postgres_client import PostgresClient
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.ingestion.indexer import ToolIndexer
from rag_tools.retrieval.embedder import BaseEmbedder, LocalEmbedder, APIEmbedder
from rag_tools.retrieval.reranker import BaseReranker, APIReranker, CrossEncoderReranker
from rag_tools.retrieval.retriever import ToolRetriever, RetrievalConfig
from rag_tools.retrieval.pipeline import (
    ToolRetrievalPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline,
)
from rag_tools.evaluation.metrics import MetricsCalculator, MetricsAggregator
from rag_tools.evaluation.evaluator import (
    RAGEvaluator,
    EvaluationSuite,
    EvaluationReport,
    create_default_suite,
)
from rag_tools.tools.add_tool import ToolAdder, add_mcp_server
from rag_tools.tools.remove_tool import ToolRemover, remove_mcp_server, remove_tool_by_id
from rag_tools.tools.sync_mcp import MCPSyncer, SyncResult, sync_server_by_id, sync_all_servers


class RAGToolsManager:
    """
    Main manager class for the RAG Tools module.

    Provides a unified interface for managing MCP tools and performing RAG retrieval.
    """

    def __init__(self, config: Optional[Settings] = None,
                    embedder: Optional[BaseEmbedder] = None,
                    reranker: Optional[BaseReranker] = None,
):
        """
        Initialize the RAG Tools manager.

        Args:
            config: Optional settings override
            embedder: Optional, Embedder instance to use
            reranker: Optional, Reranker instance to use 
        """
        self.config = config or settings
        self._postgres: Optional[PostgresClient] = None
        self._qdrant: Optional[QdrantClientWrapper] = None
        self._embedder: Optional[BaseEmbedder] = embedder
        self._reranker: Optional[BaseReranker] = reranker
        self._indexer: Optional[ToolIndexer] = None
        self._retriever: Optional[ToolRetriever] = None
        self._pipeline: Optional[ToolRetrievalPipeline] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        # Initialize PostgreSQL
        self._postgres = PostgresClient(self.config.postgres)
        await self._postgres.initialize()

        # Initialize Qdrant
        self._qdrant = QdrantClientWrapper(self.config.qdrant)
        await self._qdrant.connect()

        # Initialize embedder
        if self._embedder is None:
            self._embedder = LocalEmbedder(self.config.embedding)

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.reranker)

        await self._embedder.initialize()
        await self._reranker.initialize()

        # Initialize indexer
        self._indexer = ToolIndexer(
            self._postgres,
            self._qdrant,
            self._embedder,
        )
        await self._indexer.initialize()

        # Initialize retriever
        self._retriever = ToolRetriever(self._qdrant, self._embedder)
        await self._retriever.initialize()

        # Initialize pipeline
        self._pipeline = ToolRetrievalPipeline(
            self._embedder,
            self._reranker,
            self._retriever,
        )

        self._initialized = True

    async def close(self) -> None:
        """Close all connections."""
        if self._postgres:
            await self._postgres.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    @asynccontextmanager
    async def session(self):
        """Context manager for the manager."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    # Server management
    async def add_server(
        self,
        url: str,
        name: str,
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        sync_tools: bool = True,
    ) -> MCPServer:
        """
        Add a new MCP server.

        Args:
            url: MCP server URL
            name: Server name
            description: Optional description
            headers: HTTP headers
            sync_tools: Whether to sync tools immediately

        Returns:
            Created MCPServer
        """
        await self.initialize()
        adder = ToolAdder(self._postgres, self._qdrant, self._indexer)
        return await adder.add_server(
            url=url,
            name=name,
            description=description,
            headers=headers,
            sync_tools=sync_tools,
        )

    async def remove_server(self, server_id: str) -> Dict[str, Any]:
        """Remove a server and all its tools."""
        await self.initialize()
        remover = ToolRemover(self._postgres, self._qdrant, self._indexer)
        return await remover.remove_server(server_id)

    async def sync_server(self, server_id: str) -> SyncResult:
        """Sync tools from a server."""
        await self.initialize()
        syncer = MCPSyncer(self._postgres, self._qdrant, self._indexer)
        return await syncer.sync_server(server_id)

    async def sync_all_servers(self) -> List[SyncResult]:
        """Sync all registered servers."""
        await self.initialize()
        syncer = MCPSyncer(self._postgres, self._qdrant, self._indexer)
        return await syncer.sync_all_servers()

    # Tool management
    async def add_tool(
        self,
        server_id: str,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> MCPTool:
        """Add a single tool manually."""
        await self.initialize()
        adder = ToolAdder(self._postgres, self._qdrant, self._indexer)
        return await adder.add_tool(
            server_id=server_id,
            name=name,
            description=description,
            input_schema=input_schema,
            tags=tags,
        )

    async def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool."""
        await self.initialize()
        remover = ToolRemover(self._postgres, self._qdrant, self._indexer)
        return await remover.remove_tool(tool_id)

    # Retrieval
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        rerank_top_k: int = 5,
        min_score: float = 0.3,
        use_chunks: bool = False,
        server_filter: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Retrieve relevant tools for a query.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            rerank: Whether to use reranking
            rerank_top_k: Number of results to reranl
            min_score: Minimum relevance score,
            use_chunks: Whether to use chunked collection to retrieve
            server_filter: Criterias to server filter
            tags_filter: Tags filter of tools

        Returns:
            PipelineResult with retrieval results
        """
        await self.initialize()

        config = PipelineConfig(
            top_k=top_k,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            min_relevance_score=min_score,
            use_chunks=use_chunks,
            server_filter=server_filter,
            tags_filter=tags_filter
        )

        return await self._pipeline.retrieve(query, config)

    async def retrieve_tools(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        rerank_top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve tools for a query (simplified interface).

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of RetrievalResult
        """
        result = await self.retrieve(query, top_k=top_k, rerank=rerank, min_score=min_score, rerank_top_k=rerank_top_k)
        return result.results

    # Evaluation
    def create_evaluator(self) -> RAGEvaluator:
        """Create an evaluator instance."""
        return RAGEvaluator(self._pipeline)

    async def evaluate(
        self,
        suite: EvaluationSuite,
    ) -> EvaluationReport:
        """Evaluate with a test suite."""
        await self.initialize()
        evaluator = self.create_evaluator()
        return await evaluator.evaluate_suite(suite)

    # Stats
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        await self.initialize()

        postgres_stats = await self._postgres.get_stats()

        try:
            qdrant_info = await self._qdrant.get_collection_info(self.config.tools_collection)
        except Exception:
            qdrant_info = None

        return {
            "postgres": postgres_stats,
            "qdrant": qdrant_info,
            "embedder": {
                "model": self.config.embedding.model_name,
                "dimension": self._embedder.embedding_dim,
            },
        }

    # Direct access to components
    @property
    def postgres(self) -> PostgresClient:
        """Get PostgreSQL client."""
        return self._postgres

    @property
    def qdrant(self) -> QdrantClientWrapper:
        """Get Qdrant client."""
        return self._qdrant

    @property
    def pipeline(self) -> ToolRetrievalPipeline:
        """Get retrieval pipeline."""
        return self._pipeline


# Convenience functions
async def create_manager(config: Optional[Settings] = None, 
                         embedder: Optional[BaseEmbedder] = None,
                         reranker: Optional[BaseReranker] = None) -> RAGToolsManager:
    """Create and initialize a RAG Tools manager."""
    manager = RAGToolsManager(config, embedder, reranker)
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "RAGToolsManager",
    "ToolIndexer",
    "ToolRetriever",
    "ToolRetrievalPipeline",
    "RAGEvaluator",
    "ToolAdder",
    "ToolRemover",
    "MCPSyncer",
    # Models
    "MCPServer",
    "MCPTool",
    "RetrievalResult",
    "PipelineResult",
    "EvaluationSuite",
    "EvaluationReport",
    "SyncResult",
    # Functions
    "create_manager",
    "add_mcp_server",
    "remove_mcp_server",
    "remove_tool_by_id",
    "sync_server_by_id",
    "sync_all_servers",
    "create_pipeline",
    "create_default_suite",
]


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = await create_manager()

        # Add a server
        server = await manager.add_server(
            url="http://localhost:8080/mcp",
            name="example-server",
            description="Example MCP server",
        )

        # Retrieve tools
        results = await manager.retrieve_tools("search for papers")

        print(f"Found {len(results)} relevant tools:")
        for r in results:
            print(f"  - {r.name} (score: {r.score:.3f})")

        await manager.close()

    asyncio.run(main())
