"""
Retriever for fetching tools from vector database.
"""
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from qdrant_client.http.models import FieldCondition, MatchValue, Filter

from rag_tools.config.settings import settings
from rag_tools.storage.models import RetrievalResult, ToolStatus
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.retrieval.embedder import BaseEmbedder, LocalEmbedder


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 10
    min_score: float = 0.0
    use_chunks: bool = False
    server_filter: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None


class ToolRetriever:
    """Retrieves tools from Qdrant based on query."""

    def __init__(
        self,
        qdrant_client: QdrantClientWrapper,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.qdrant = qdrant_client
        self.embedder = embedder or LocalEmbedder()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the retriever."""
        if self._initialized:
            return

        await self.embedder.initialize()
        await self.qdrant.set_embedding_dim(self.embedder.embedding_dim)
        self._initialized = True

    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant tools for a query.

        Args:
            query: Search query
            config: Retrieval configuration

        Returns:
            List of retrieval results
        """
        if not self._initialized:
            await self.initialize()

        config = config or RetrievalConfig(top_k=settings.rag.default_top_k)

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Build filter conditions
        filter_conditions = self._build_filters(config)

        # Determine collection
        collection = (
            settings.tools_chunks_collection
            if config.use_chunks
            else settings.tools_collection
        )

        # Search
        search_results = await self.qdrant.search(
            collection_name=collection,
            query_vector=query_embedding.tolist(),
            top_k=config.top_k * 2,  # Get more for filtering
            filter_conditions=filter_conditions,
            with_payload=True,
        )

        # Convert to RetrievalResult
        results = []
        for i, hit in enumerate(search_results):
            result = self._hit_to_result(hit, i + 1)
            results.append(result)

        # Apply minimum score filter
        if config.min_score > 0:
            results = [r for r in results if r.score >= config.min_score]

        # Deduplicate by tool_id (keep highest scoring)
        seen = {}
        for result in results:
            if result.tool_id not in seen:
                seen[result.tool_id] = result

        unique_results = list(seen.values())

        # Update ranks
        for i, result in enumerate(unique_results):
            result.rank = i + 1

        return unique_results[:config.top_k]

    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[RetrievalConfig] = None,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve tools for multiple queries.

        Args:
            queries: List of search queries
            config: Retrieval configuration

        Returns:
            List of result lists, one per query
        """
        if not self._initialized:
            await self.initialize()

        config = config or RetrievalConfig(top_k=settings.rag.default_top_k)

        # Generate query embeddings
        query_embeddings = await self.embedder.embed(queries)

        # Build filter conditions
        filter_conditions = self._build_filters(config)

        # Determine collection
        collection = (
            settings.tools_chunks_collection
            if config.use_chunks
            else settings.tools_collection
        )

        # Batch search
        all_results = await self.qdrant.search_batch(
            collection_name=collection,
            query_vectors=query_embeddings.tolist(),
            top_k=config.top_k * 2,
            filter_conditions=filter_conditions,
            with_payload=True,
        )

        # Convert to RetrievalResult
        final_results = []
        for query_results in all_results:
            results = []
            for i, hit in enumerate(query_results):
                result = self._hit_to_result(hit, i + 1)
                results.append(result)

            # Apply minimum score and deduplicate
            if config.min_score > 0:
                results = [r for r in results if r.score >= config.min_score]

            seen = {}
            for result in results:
                if result.tool_id not in seen:
                    seen[result.tool_id] = result

            unique_results = list(seen.values())

            # Update ranks
            for i, result in enumerate(unique_results):
                result.rank = i + 1

            final_results.append(unique_results[:config.top_k])

        return final_results

    def _build_filters(self, config: RetrievalConfig) -> Optional[List[FieldCondition]]:
        """Build filter conditions."""
        conditions = []

        # Server filter
        if config.server_filter:
            conditions.append(
                FieldCondition(
                    key="server_id",
                    match=MatchValue(value=config.server_filter[0]),
                )
            )

        # Tags filter (requires pre-indexing with tags)
        if config.tags_filter:
            conditions.append(
                FieldCondition(
                    key="tags",
                    match=MatchValue(value=config.tags_filter[0]),
                )
            )

        return conditions if conditions else None

    def _hit_to_result(self, hit: Dict[str, Any], rank: int) -> RetrievalResult:
        """Convert a Qdrant hit to RetrievalResult."""
        payload = hit.get("payload", {})

        return RetrievalResult(
            tool_id=payload.get("tool_id", hit.get("id", "")),
            server_id=payload.get("server_id", ""),
            name=payload.get("name", ""),
            description=payload.get("description", ""),
            score=hit.get("score", 0),
            rank=rank,
            metadata=payload,
        )

    async def get_tool_by_id(self, tool_id: str) -> Optional[RetrievalResult]:
        """Get a specific tool by ID."""
        results = await self.qdrant.retrieve(
            collection_name=settings.tools_collection,
            point_ids=[tool_id],
            with_payload=True,
        )

        if not results:
            return None

        return self._hit_to_result(results[0], 1)

    async def get_tools_by_ids(self, tool_ids: List[str]) -> List[RetrievalResult]:
        """Get multiple tools by IDs."""
        if not tool_ids:
            return []

        results = await self.qdrant.retrieve(
            collection_name=settings.tools_collection,
            point_ids=tool_ids,
            with_payload=True,
        )

        return [
            self._hit_to_result(hit, i + 1)
            for i, hit in enumerate(results)
        ]

    async def get_all_tools(
        self,
        limit: int = 100,
        offset: Optional[str] = None,
    ) -> tuple[List[RetrievalResult], Optional[str]]:
        """Get all tools with pagination."""
        results, next_offset = await self.qdrant.scroll(
            collection_name=settings.tools_collection,
            limit=limit,
            offset=offset,
            with_payload=True,
        )

        retrieval_results = [
            self._hit_to_result(hit, i + 1)
            for i, hit in enumerate(results)
        ]

        return retrieval_results, next_offset
