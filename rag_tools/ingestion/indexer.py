"""
Indexer for adding tools to the vector database.
"""
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

from rag_tools.config.settings import settings
from rag_tools.storage.models import MCPTool, ToolChunk, ToolStatus
from rag_tools.storage.postgres_client import PostgresClient
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.retrieval.embedder import Embedder
from rag_tools.ingestion.text_builder import TextBuilder, build_tool_metadata


class ToolIndexer:
    """Indexes tools to PostgreSQL and Qdrant."""

    def __init__(
        self,
        postgres_client: PostgresClient,
        qdrant_client: QdrantClientWrapper,
        embedder: Optional[Embedder] = None,
    ):
        self.postgres = postgres_client
        self.qdrant = qdrant_client
        self.embedder = embedder or Embedder()
        self.text_builder = TextBuilder()

    async def initialize(self) -> None:
        """Initialize collections and indexer."""
        # Initialize embedder
        await self.embedder.initialize()

        # Set embedding dimension
        self.qdrant.set_embedding_dim(self.embedder.embedding_dim)

        # Create collections
        self._ensure_collections()

        # Create payload indexes
        self._ensure_indexes()

    def _ensure_collections(self) -> None:
        """Ensure Qdrant collections exist."""
        # Main tools collection
        self.qdrant.create_collection(
            collection_name=settings.tools_collection,
            vector_size=self.embedder.embedding_dim,
        )

        # Chunks collection for larger tools
        self.qdrant.create_collection(
            collection_name=settings.tools_chunks_collection,
            vector_size=self.embedder.embedding_dim,
        )

    def _ensure_indexes(self) -> None:
        """Ensure payload indexes exist."""
        # Index on tool_id for fast lookups
        self.qdrant.create_payload_index(
            collection_name=settings.tools_collection,
            field_name="tool_id",
        )

        # Index on server_id for filtering
        self.qdrant.create_payload_index(
            collection_name=settings.tools_collection,
            field_name="server_id",
        )

        # Index on name for exact matches
        self.qdrant.create_payload_index(
            collection_name=settings.tools_collection,
            field_name="name",
        )

        # Chunk indexes
        self.qdrant.create_payload_index(
            collection_name=settings.tools_chunks_collection,
            field_name="tool_id",
        )

        self.qdrant.create_payload_index(
            collection_name=settings.tools_chunks_collection,
            field_name="server_id",
        )

    async def index_tool(self, tool: MCPTool, rebuild: bool = False) -> bool:
        """
        Index a single tool.

        Args:
            tool: Tool to index
            rebuild: If True, delete existing and reindex

        Returns:
            True if successful
        """
        if rebuild:
            await self.remove_tool(tool.tool_id)

        # Store in PostgreSQL
        await self.postgres.add_tool(tool)

        # Build search text
        text = self.text_builder.build_full_text(tool)

        # Get embedding
        embedding = await self.embedder.embed([text])
        embedding = embedding[0]

        # Build metadata
        metadata = build_tool_metadata(tool)

        # Store in Qdrant
        self.qdrant.upsert_points(
            collection_name=settings.tools_collection,
            vectors=[embedding],
            payloads=[metadata],
            ids=[tool.tool_id],
        )

        return True

    async def index_tools_batch(
        self,
        tools: List[MCPTool],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Index multiple tools in batches.

        Args:
            tools: List of tools to index
            batch_size: Batch size for processing
            show_progress: Show progress indicator

        Returns:
            Statistics dict
        """
        stats = {
            "total": len(tools),
            "indexed": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Store all tools in PostgreSQL first
        if tools:
            await self.postgres.bulk_add_tools(tools)

        # Build texts and metadata
        texts = []
        all_metadata = []
        for t in tools: #TODO we build here same text descriptions many times. Can fix it via text_builder.build_chunks yeild full_description
            await self.index_tool_chunks(t)
            texts.append(self.text_builder.build_full_text(t))
            all_metadata.append(build_tool_metadata(t))
        # texts = [self.text_builder.build_full_text(t) for t in tools]
        # all_metadata = [build_tool_metadata(t) for t in tools]

        # Process in batches
        for i in range(0, len(tools), batch_size):
            batch_end = min(i + batch_size, len(tools))
            batch_texts = texts[i:batch_end]
            batch_metadata = all_metadata[i:batch_end]
            batch_tools = tools[i:batch_end]

            try:
                # Get embeddings
                embeddings = await self.embedder.embed(batch_texts)

                # Prepare IDs
                ids = [t.tool_id for t in batch_tools]

                # Store in Qdrant
                self.qdrant.upsert_points(
                    collection_name=settings.tools_collection,
                    vectors=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                    payloads=batch_metadata,
                    ids=ids,
                )

                stats["indexed"] += len(batch_tools)

                if show_progress:
                    print(f"Indexed {stats['indexed']}/{stats['total']}")

            except Exception as e:
                stats["failed"] += len(batch_tools)
                print(f"Batch indexing failed: {e}")

        return stats

    async def index_tool_chunks(self, tool: MCPTool) -> int:
        """
        Index tool as chunks for better retrieval of long descriptions.

        Args:
            tool: Tool to index as chunks

        Returns:
            Number of chunks indexed
        """
        # Generate chunks
        chunks = self.text_builder.build_chunks(tool)

        if not chunks:
            return 0

        # Store chunks in PostgreSQL
        await self.postgres.add_chunks(chunks)

        # Build texts for embedding
        chunk_texts = [c.text for c in chunks]

        # Get embeddings
        embeddings = await self.embedder.embed(chunk_texts)

        # Prepare payloads
        payloads = [
            {
                "chunk_id": c.chunk_id,
                "tool_id": c.tool_id,
                "server_id": tool.server_id,
                "name": tool.name,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        # Store in Qdrant
        self.qdrant.upsert_points(
            collection_name=settings.tools_chunks_collection,
            vectors=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
            payloads=payloads,
            ids=[c.chunk_id for c in chunks],
        )

        return len(chunks)

    async def remove_tool(self, tool_id: str) -> bool:
        """
        Remove a tool from the index.

        Args:
            tool_id: Tool ID to remove

        Returns:
            True if successful
        """
        # Remove from PostgreSQL
        await self.postgres.delete_tool(tool_id)

        # Remove from Qdrant
        self.qdrant.delete_points(
            collection_name=settings.tools_collection,
            point_ids=[tool_id],
        )

        # Remove chunks
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        self.qdrant.delete_by_filter(
            collection_name=settings.tools_chunks_collection,
            filter_conditions=[
                FieldCondition(
                    key="tool_id",
                    match=MatchValue(value=tool_id),
                )
            ],
        )

        return True

    async def remove_server_tools(self, server_id: str) -> int:
        """
        Remove all tools for a server.

        Args:
            server_id: Server ID

        Returns:
            Number of tools removed
        """
        # Get tools to remove
        tools = await self.postgres.get_tools_by_server(server_id)
        count = len(tools)

        # Remove from PostgreSQL
        for tool in tools:
            await self.postgres.delete_tool(tool.tool_id)

        # Remove from Qdrant
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Remove from main collection
        self.qdrant.delete_by_filter(
            collection_name=settings.tools_collection,
            filter_conditions=[
                FieldCondition(
                    key="server_id",
                    match=MatchValue(value=server_id),
                )
            ],
        )

        # Remove from chunks collection
        self.qdrant.delete_by_filter(
            collection_name=settings.tools_chunks_collection,
            filter_conditions=[
                FieldCondition(
                    key="server_id",
                    match=MatchValue(value=server_id),
                )
            ],
        )

        return count

    async def reindex_tool(self, tool: MCPTool) -> bool:
        """
        Reindex a tool (delete and add).

        Args:
            tool: Tool to reindex

        Returns:
            True if successful
        """
        return await self.index_tool(tool, rebuild=True)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        stats = {}

        # Qdrant stats
        try:
            main_info = self.qdrant.get_collection_info(settings.tools_collection)
            chunks_info = self.qdrant.get_collection_info(settings.tools_chunks_collection)

            stats["qdrant"] = {
                "tools_collection": main_info,
                "chunks_collection": chunks_info,
            }
        except Exception as e:
            stats["qdrant"] = {"error": str(e)}

        # PostgreSQL stats
        try:
            stats["postgres"] = asyncio.get_event_loop().run_until_complete(
                self.postgres.get_stats()
            )
        except Exception as e:
            stats["postgres"] = {"error": str(e)}

        return stats
