"""
Tool management: Sync tools from MCP servers.
"""
import asyncio
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from rag_tools.storage.models import MCPServer, MCPTool, ToolStatus
from rag_tools.storage.postgres_client import PostgresClient
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.ingestion.parser import (
    mcp_tool_info_to_model,
    create_tool_id,
)
from rag_tools.ingestion.indexer import ToolIndexer

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

@dataclass
class SyncResult:
    """Result of a sync operation."""
    server_id: str
    server_name: str
    success: bool
    tools_added: int
    tools_removed: int
    tools_updated: int
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0


class MCPSyncer:
    """Syncs tools from MCP servers."""

    def __init__(
        self,
        postgres: PostgresClient,
        qdrant: QdrantClientWrapper,
        indexer: Optional[ToolIndexer] = None,
    ):
        self.postgres = postgres
        self.qdrant = qdrant
        self.indexer = indexer or ToolIndexer(postgres, qdrant)

    async def sync_server(
        self,
        server_id: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> SyncResult:
        """
        Sync tools from a single server.

        Args:
            server_id: Server ID to sync
            progress_callback: Optional callback for progress updates

        Returns:
            SyncResult with statistics
        """
        started_at = datetime.utcnow()
        result = SyncResult(
            server_id=server_id,
            server_name="",
            success=False,
            tools_added=0,
            tools_removed=0,
            tools_updated=0,
            errors=[],
            started_at=started_at,
        )

        # Get server
        server = await self.postgres.get_server(server_id)
        if not server:
            result.errors.append(f"Server {server_id} not found")
            result.completed_at = datetime.utcnow()
            result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000
            return result

        result.server_name = server.name

        try:
            # Get existing tools
            existing_tools = await self.postgres.get_tools_by_server(server_id)
            existing_by_name = {t.name: t for t in existing_tools}

            # Connect to MCP server
            async with sse_client(server.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    raw_tools = response.tools

                    new_tools = []
                    for tool_info in response.tools:
                        tool = mcp_tool_info_to_model(tool_info, server_id)

                        if tool.name in existing_by_name:
                            # Check if updated
                            existing = existing_by_name[tool.name]
                            if existing.description != tool.description:
                                result.tools_updated += 1
                        else:
                            new_tools.append(tool)
                            result.tools_added += 1

                    # Add new tools
                    if new_tools:
                        await self.postgres.bulk_add_tools(new_tools)
                        await self.indexer.index_tools_batch(new_tools)

                    # Mark server as active and update timestamp
                    server.status = ToolStatus.ACTIVE
                    server.last_synced = datetime.utcnow()
                    await self.postgres.add_server(server)

                    result.success = True          

        except Exception as e:
            result.errors.append(f"Error: {str(e)}")
            server.status = ToolStatus.ERROR
            await self.postgres.add_server(server)

        result.completed_at = datetime.utcnow()
        result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000

        return result   

    async def sync_all_servers(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[SyncResult]:
        """
        Sync all registered servers.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of SyncResults
        """
        servers = await self.postgres.list_servers()
        results = []

        for i, server in enumerate(servers):
            if progress_callback:
                progress_callback(f"Syncing {server.name}", i, len(servers))

            result = await self.sync_server(server.server_id)
            results.append(result)

        return results

    async def sync_with_retry(
        self,
        server_id: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> SyncResult:
        """
        Sync with automatic retry on failure.

        Args:
            server_id: Server ID to sync
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            SyncResult
        """
        for attempt in range(max_retries):
            result = await self.sync_server(server_id)

            if result.success:
                return result

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))

        return result

    async def create_server_and_sync(
        self,
        url: str,
        name: str,
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> SyncResult:
        """
        Create a new server and sync its tools.

        Args:
            url: MCP server URL
            name: Server name
            description: Optional description
            headers: HTTP headers
            timeout: Request timeout

        Returns:
            SyncResult
        """
        from rag_tools.tools.add_tool import ToolAdder

        adder = ToolAdder(self.postgres, self.qdrant, self.indexer)

        # Add server
        server = await adder.add_server(
            url=url,
            name=name,
            description=description,
            headers=headers,
            timeout=timeout,
            sync_tools=False,
        )

        # Sync tools
        result = await self.sync_server(server.server_id)

        return result

    async def schedule_periodic_sync(
        self,
        interval_hours: float,
        servers: Optional[List[str]] = None,
    ) -> asyncio.Task:
        """
        Schedule periodic sync as a background task.

        Args:
            interval_hours: Interval between syncs in hours
            servers: Optional list of specific server IDs (None = all)

        Returns:
            asyncio.Task that can be cancelled
        """
        async def sync_loop():
            while True:
                try:
                    if servers:
                        for server_id in servers:
                            await self.sync_with_retry(server_id)
                    else:
                        results = await self.sync_all_servers()
                        for r in results:
                            if not r.success:
                                print(f"Sync failed for {r.server_name}: {r.errors}")

                    # Wait for next interval
                    await asyncio.sleep(interval_hours * 3600)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Sync loop error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        return asyncio.create_task(sync_loop())


async def sync_server_by_id(server_id: str) -> SyncResult:
    """
    Convenience function to sync a server by ID.

    Args:
        server_id: Server ID to sync

    Returns:
        SyncResult
    """
    postgres = PostgresClient()
    await postgres.initialize()

    qdrant = QdrantClientWrapper()
    qdrant.connect()

    syncer = MCPSyncer(postgres, qdrant)

    try:
        result = await syncer.sync_server(server_id)
        return result
    finally:
        await postgres.close()
        qdrant.close()


async def sync_all_servers() -> List[SyncResult]:
    """
    Convenience function to sync all servers.

    Returns:
        List of SyncResults
    """
    postgres = PostgresClient()
    await postgres.initialize()

    qdrant = QdrantClientWrapper()
    qdrant.connect()

    syncer = MCPSyncer(postgres, qdrant)

    try:
        results = await syncer.sync_all_servers()
        return results
    finally:
        await postgres.close()
        qdrant.close()
