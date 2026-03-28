"""
Tool management: Remove tools from the registry.
"""
import asyncio
from typing import List, Optional, Dict, Any

from rag_tools.storage.models import ToolStatus
from rag_tools.storage.postgres_client import PostgresClient
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.ingestion.indexer import ToolIndexer


class ToolRemover:
    """Remove tools from the registry and index."""

    def __init__(
        self,
        postgres: PostgresClient,
        qdrant: QdrantClientWrapper,
        indexer: Optional[ToolIndexer] = None,
    ):
        self.postgres = postgres
        self.qdrant = qdrant
        self.indexer = indexer or ToolIndexer(postgres, qdrant)

    async def remove_tool(self, tool_id: str) -> bool:
        """
        Remove a single tool.

        Args:
            tool_id: Tool ID to remove

        Returns:
            True if successful
        """
        # Remove from indexer (PostgreSQL and Qdrant)
        await self.indexer.remove_tool(tool_id)
        return True

    async def remove_tools_batch(self, tool_ids: List[str]) -> Dict[str, int]:
        """
        Remove multiple tools.

        Args:
            tool_ids: List of tool IDs to remove

        Returns:
            Statistics dict
        """
        stats = {
            "total": len(tool_ids),
            "removed": 0,
            "failed": 0,
        }

        for tool_id in tool_ids:
            try:
                await self.remove_tool(tool_id)
                stats["removed"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"Failed to remove {tool_id}: {e}")

        return stats

    async def remove_server(self, server_id: str) -> Dict[str, Any]:
        """
        Remove a server and all its tools.

        Args:
            server_id: Server ID to remove

        Returns:
            Statistics dict
        """
        stats = {
            "server_id": server_id,
            "tools_removed": 0,
            "credentials_removed": 0,
        }

        # Get tools for this server
        tools = await self.postgres.get_tools_by_server(server_id)
        stats["tools_removed"] = len(tools)

        # Remove all tools
        await self.indexer.remove_server_tools(server_id)

        # Remove server from PostgreSQL
        await self.postgres.delete_server(server_id)

        return stats

    async def remove_by_filter(
        self,
        server_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: Optional[ToolStatus] = None,
    ) -> Dict[str, int]:
        """
        Remove tools by filter criteria.

        Args:
            server_ids: Filter by server IDs
            tags: Filter by tags
            status: Filter by status

        Returns:
            Statistics dict
        """
        stats = {
            "total": 0,
            "removed": 0,
            "failed": 0,
        }

        # Get all servers
        servers = await self.postgres.list_servers()

        for server in servers:
            if server_ids and server.server_id not in server_ids:
                continue

            tools = await self.postgres.get_tools_by_server(server.server_id)

            for tool in tools:
                if status and tool.status != status:
                    continue

                if tags and not any(t in tool.tags for t in tags):
                    continue

                try:
                    await self.remove_tool(tool.tool_id)
                    stats["removed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    print(f"Failed to remove {tool.tool_id}: {e}")

                stats["total"] += 1

        return stats

    async def remove_inactive_tools(self, older_than_hours: int = 24) -> Dict[str, int]:
        """
        Remove tools marked as inactive.

        Args:
            older_than_hours: Only remove tools inactive for this many hours

        Returns:
            Statistics dict
        """
        from datetime import datetime, timedelta

        stats = {
            "total": 0,
            "removed": 0,
            "failed": 0,
        }

        # Get all servers
        servers = await self.postgres.list_servers()

        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)

        for server in servers:
            tools = await self.postgres.get_tools_by_server(server.server_id)

            for tool in tools:
                if tool.status == ToolStatus.INACTIVE:
                    if tool.updated_at and tool.updated_at < cutoff:
                        try:
                            await self.remove_tool(tool.tool_id)
                            stats["removed"] += 1
                        except Exception as e:
                            stats["failed"] += 1
                            print(f"Failed to remove {tool.tool_id}: {e}")

                        stats["total"] += 1

        return stats

    async def clear_all(self) -> Dict[str, Any]:
        """
        Clear all tools and servers (use with caution!).

        Returns:
            Statistics dict
        """
        stats = {
            "servers": 0,
            "tools": 0,
            "credentials": 0,
            "chunks": 0,
        }

        # Get all servers
        servers = await self.postgres.list_servers()
        stats["servers"] = len(servers)

        # Remove each server
        for server in servers:
            await self.remove_server(server.server_id)

        # Get stats
        final_stats = await self.postgres.get_stats()
        stats.update(final_stats)

        return stats


async def remove_mcp_server(server_id: str) -> Dict[str, Any]:
    """
    Convenience function to remove an MCP server.

    Args:
        server_id: Server ID to remove

    Returns:
        Statistics dict
    """
    postgres = PostgresClient()
    await postgres.initialize()

    qdrant = QdrantClientWrapper()
    await qdrant.connect()

    remover = ToolRemover(postgres, qdrant)

    try:
        stats = await remover.remove_server(server_id)
        return stats
    finally:
        await postgres.close()
        await qdrant.close()


async def remove_tool_by_id(tool_id: str) -> bool:
    """
    Convenience function to remove a single tool.

    Args:
        tool_id: Tool ID to remove

    Returns:
        True if successful
    """
    postgres = PostgresClient()
    await postgres.initialize()

    qdrant = QdrantClientWrapper()
    await qdrant.connect()

    remover = ToolRemover(postgres, qdrant)

    try:
        return await remover.remove_tool(tool_id)
    finally:
        await postgres.close()
        qdrant.close()
