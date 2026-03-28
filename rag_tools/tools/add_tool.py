"""
Tool management: Add tools to the registry.
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rag_tools.config.settings import settings
from rag_tools.storage.models import MCPTool, MCPServer, ToolStatus
from rag_tools.storage.postgres_client import PostgresClient
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from rag_tools.ingestion.indexer import ToolIndexer
from rag_tools.ingestion.parser import (
    mcp_tool_info_to_model,
    create_tool_id,
    MCPToolInfo, 
    MCPToolsResponse
)


class ToolAdder:
    """Add tools to the registry and index."""

    def __init__(
        self,
        postgres: PostgresClient,
        qdrant: QdrantClientWrapper,
        indexer: Optional[ToolIndexer] = None,
    ):
        self.postgres = postgres
        self.qdrant = qdrant
        self.indexer = indexer or ToolIndexer(postgres, qdrant)

    async def add_server(
        self,
        url: str,
        name: str,
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        sync_tools: bool = True,
    ) -> MCPServer:
        """
        Add a new MCP server.

        Args:
            url: MCP server URL
            name: Server name
            description: Optional description
            headers: HTTP headers
            timeout: Request timeout
            sync_tools: Whether to sync tools immediately

        Returns:
            Created MCPServer
        """
        import hashlib

        server_id = hashlib.sha256(f"{url}:{name}".encode()).hexdigest()[:16]

        server = MCPServer(
            server_id=server_id,
            name=name,
            url=url,
            description=description,
            headers=headers or {},
            timeout=timeout,
            status=ToolStatus.SYNCING,
        )

        # Store in PostgreSQL
        await self.postgres.add_server(server)

        # Sync tools if requested
        if sync_tools:
            try:
                await self.sync_server_tools(server)
                server.status = ToolStatus.ACTIVE
            except Exception as e:
                server.status = ToolStatus.ERROR
                print(f"Error syncing tools: {e}")

            # Update server status
            server = await self.postgres.add_server(server)

        return server

    async def add_tool(
        self,
        server_id: str,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        index: bool = True,
    ) -> MCPTool:
        """
        Add a single tool manually.

        Args:
            server_id: Server ID
            name: Tool name
            description: Tool description
            input_schema: Input parameters schema
            output_schema: Output schema
            tags: Tool tags
            index: Whether to index immediately

        Returns:
            Created MCPTool
        """
        tool_id = create_tool_id(server_id, name)

        tool = MCPTool(
            tool_id=tool_id,
            server_id=server_id,
            name=name,
            description=description,
            input_schema=input_schema or {},
            output_schema=output_schema,
            tags=tags or [],
        )

        # Store in PostgreSQL
        await self.postgres.add_tool(tool)

        # Index if requested
        if index:
            await self.indexer.index_tool(tool)
            await self.indexer.index_tool_chunks(tool)

        return tool

    async def add_tools_batch(
        self,
        tools: List[MCPTool],
        index: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        Add multiple tools in batch.

        Args:
            tools: List of tools to add
            index: Whether to index
            batch_size: Batch size for indexing

        Returns:
            Statistics dict
        """
        stats = {
            "total": len(tools),
            "added": 0,
            "failed": 0,
        }

        # Store in PostgreSQL
        await self.postgres.bulk_add_tools(tools)
        stats["added"] = len(tools)

        # Index if requested
        if index:
            index_stats = await self.indexer.index_tools_batch(
                tools,
                batch_size=batch_size,
            )
            stats["indexed"] = index_stats["indexed"]

        return stats

    async def sync_server_tools(self, server: MCPServer) -> List[MCPTool]:
        """
        Sync tools from an MCP server.

        Args:
            server: Server to sync

        Returns:
            List of synced tools
        """
        try:
            async with streamable_http_client(server.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    raw_tools = response.tools

                    tools = []
                    for tool_info in raw_tools:
                        tool = MCPTool(
                            tool_id=create_tool_id(server.server_id, tool_info.name),   # adapt as needed
                            server_id=server.server_id,
                            name=tool_info.name,
                            description=tool_info.description,
                            input_schema=tool_info.inputSchema,
                            output_schema=getattr(tool_info, 'outputSchema', None),
                            tags=getattr(tool_info, 'tags', []),
                        )
                        tools.append(tool)

                    # Delete existing tools for this server
                    existing_tools = await self.postgres.get_tools_by_server(server.server_id)

                    for exist_tool in existing_tools:
                        await self.indexer.remove_tool(exist_tool.tool_id)

                    # Add new tools
                    for tool in tools:
                        await self.postgres.add_tool(tool)
                        await self.indexer.index_tool(tool)
                        await self.indexer.index_tool_chunks(tool)
                    # await self.postgres.bulk_add_tools(tools)

                    # Index tools
                    # await self.indexer.index_tools_batch(tools)

                    # Update server last_synced
                    server.last_synced = datetime.utcnow()
                    await self.postgres.add_server(server)

                    return tools

        except Exception as e:
            print(f"Error syncing tools: {e}")
            import traceback
            traceback.print_exc()
            raise   # re-raise so the caller knows


    async def add_credential(
        self,
        server_id: str,
        credential_type: str,
        credential_data: Dict[str, str],
        tool_id: Optional[str] = None,
    ) -> str:
        """
        Add credentials for a server or tool.

        Args:
            server_id: Server ID
            credential_type: Type (api_key, oauth, basic, bearer)
            credential_data: Credential values
            tool_id: Optional specific tool ID

        Returns:
            Credential ID
        """
        from rag_tools.storage.models import ToolCredential
        import uuid

        credential = ToolCredential(
            credential_id=str(uuid.uuid4()),
            server_id=server_id,
            tool_id=tool_id,
            credential_type=credential_type,
            credential_data=credential_data,
        )

        await self.postgres.add_credential(credential)

        return credential.credential_id


async def add_mcp_server(
    url: str,
    name: str,
    postgres_url: str = None,
    qdrant_url: str = None,
    **kwargs,
) -> MCPServer:
    """
    Convenience function to add an MCP server.

    Args:
        url: MCP server URL
        name: Server name
        postgres_url: PostgreSQL connection URL
        qdrant_url: Qdrant URL
        **kwargs: Additional arguments for ToolAdder

    Returns:
        Created MCPServer
    """
    # Initialize clients
    postgres = PostgresClient()
    await postgres.initialize()

    qdrant = QdrantClientWrapper()
    await qdrant.connect()

    adder = ToolAdder(postgres, qdrant)

    try:
        server = await adder.add_server(url, name, **kwargs)
        return server
    finally:
        await postgres.close()
        await qdrant.close()
