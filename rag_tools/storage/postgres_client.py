"""
PostgreSQL client for tool registry storage.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any
from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload

from rag_tools.config.settings import settings, PostgresSettings
from rag_tools.storage.models import (
    Base, ServerModel, ToolModel, CredentialModel, ChunkModel,
    MCPServer, MCPTool, ToolCredential, ToolChunk
)


class PostgresClient:
    """Async PostgreSQL client for managing tool registry."""

    def __init__(self, config: Optional[PostgresSettings] = None):
        self.config = config or settings.postgres
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize the database connection."""
        connection_url = (
            f"postgresql+asyncpg://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

        self._engine = create_async_engine(
            connection_url,
            pool_size=self.config.max_connections,
            # min_pool_size=self.config.min_connections,
            echo=False,
            max_overflow=0
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Server operations
    async def add_server(self, server: MCPServer) -> MCPServer:
        """Add or update an MCP server."""
        async with self.session() as session:
            # Check if server already exists
            result = await session.execute(
                select(ServerModel).where(ServerModel.server_id == server.server_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing server
                existing.name = server.name
                existing.url = server.url
                existing.description = server.description
                existing.session_id = server.session_id
                existing.headers = server.headers
                existing.timeout = server.timeout
                existing.status = server.status.value
                existing.last_synced = server.last_synced
                existing.updated_at = datetime.utcnow()  # Use your utc_now() helper
                model = existing
            else:
                # Create new server
                model = ServerModel(
                    server_id=server.server_id,
                    name=server.name,
                    url=server.url,
                    description=server.description,
                    session_id=server.session_id,
                    headers=server.headers,
                    timeout=server.timeout,
                    status=server.status.value,
                    last_synced=server.last_synced,
                    created_at=server.created_at or datetime.utcnow(),  # Use provided or now
                    updated_at=datetime.utcnow(),
                )
                session.add(model)
            
            await session.flush()
            
            return MCPServer(
                server_id=model.server_id,
                name=model.name,
                url=model.url,
                description=model.description,
                session_id=model.session_id,
                headers=model.headers,
                timeout=model.timeout,
                status=model.status,
                last_synced=model.last_synced,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

    async def get_server(self, server_id: str) -> Optional[MCPServer]:
        """Get a server by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(ServerModel).where(ServerModel.server_id == server_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return None

            return MCPServer(
                server_id=model.server_id,
                name=model.name,
                url=model.url,
                description=model.description,
                session_id=model.session_id,
                headers=model.headers,
                timeout=model.timeout,
                status=model.status,
                last_synced=model.last_synced,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

    async def list_servers(self, status: Optional[str] = None) -> List[MCPServer]:
        """List all servers, optionally filtered by status."""
        async with self.session() as session:
            query = select(ServerModel)
            if status:
                query = query.where(ServerModel.status == status)

            result = await session.execute(query)
            models = result.scalars().all()

            return [
                MCPServer(
                    server_id=m.server_id,
                    name=m.name,
                    url=m.url,
                    description=m.description,
                    session_id=m.session_id,
                    headers=m.headers,
                    timeout=m.timeout,
                    status=m.status,
                    last_synced=m.last_synced,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in models
            ]

    async def delete_server(self, server_id: str) -> bool:
        """Delete a server and all its tools."""
        async with self.session() as session:
            # Delete chunks
            tool_ids = await session.execute(
                select(ToolModel.tool_id).where(ToolModel.server_id == server_id)
            )
            tool_ids = [t for t in tool_ids.scalars().all()]
            if tool_ids:
                await session.execute(
                    delete(ChunkModel).where(ChunkModel.tool_id.in_(tool_ids))
                )
            # Delete tools 
            await session.execute(
                delete(ToolModel).where(ToolModel.server_id == server_id)
            )
            # Delete credentials
            await session.execute(
                delete(CredentialModel).where(CredentialModel.server_id == server_id)
            )
            # Delete server
            await session.execute(
                delete(ServerModel).where(ServerModel.server_id == server_id)
            )

            return True

    # Tool operations
    async def add_tool(self, tool: MCPTool) -> MCPTool:
        """Add or update a tool."""

        async with self.session() as session:
            # Check if tool already exists
            result = await session.execute(
                select(ToolModel).where(ToolModel.tool_id == tool.tool_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing server
                existing.name = tool.name
                existing.description = tool.description
                existing.input_schema = tool.input_schema
                existing.output_schema = tool.output_schema
                existing.tags = tool.tags
                existing.status = tool.status.value
                existing.version = tool.version
                existing.created_at = tool.created_at
                existing.updated_at = datetime.utcnow()  
                model = existing
            else:
                # Create new tool
                model = ToolModel(
                    tool_id=tool.tool_id,
                    server_id=tool.server_id,
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    tags=tool.tags,
                    status=tool.status.value,
                    version=tool.version,
                    created_at=tool.created_at,
                    updated_at=datetime.utcnow(),
                )
                session.add(model)
            
            await session.flush()
    
            return MCPTool(
                tool_id=model.tool_id,
                server_id=model.server_id,
                name=model.name,
                description=model.description,
                input_schema=model.input_schema,
                output_schema=model.output_schema,
                tags=model.tags,
                status=model.status,
                version=model.version,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

    async def get_tool(self, tool_id: str) -> Optional[MCPTool]:
        """Get a tool by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(ToolModel).where(ToolModel.tool_id == tool_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return None

            return MCPTool(
                tool_id=model.tool_id,
                server_id=model.server_id,
                name=model.name,
                description=model.description,
                input_schema=model.input_schema,
                output_schema=model.output_schema,
                tags=model.tags,
                status=model.status,
                version=model.version,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

    async def get_tools_by_server(self, server_id: str) -> List[MCPTool]:
        """Get all tools for a server."""
        async with self.session() as session:
            result = await session.execute(
                select(ToolModel).where(ToolModel.server_id == server_id)
            )
            models = result.scalars().all()

            return [
                MCPTool(
                    tool_id=m.tool_id,
                    server_id=m.server_id,
                    name=m.name,
                    description=m.description,
                    input_schema=m.input_schema,
                    output_schema=m.output_schema,
                    tags=m.tags,
                    status=m.status,
                    version=m.version,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in models
            ]

    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool and its chunks."""
        async with self.session() as session:
            # Delete chunks
            await session.execute(
                delete(ChunkModel).where(ChunkModel.tool_id == tool_id)
            )
            # Delete tool
            await session.execute(
                delete(ToolModel).where(ToolModel.tool_id == tool_id)
            )

            return True

    async def bulk_add_tools(self, tools: List[MCPTool]) -> int:
        """Bulk insert tools."""
        async with self.session() as session:
            models = [
                ToolModel(
                    tool_id=t.tool_id,
                    server_id=t.server_id,
                    name=t.name,
                    description=t.description,
                    input_schema=t.input_schema,
                    output_schema=t.output_schema,
                    tags=t.tags,
                    status=t.status.value,
                    version=t.version,
                    created_at=t.created_at,
                    updated_at=datetime.utcnow(),
                )
                for t in tools
            ]
            session.add_all(models)
            await session.flush()

            return len(models)

    # Chunk operations
    async def add_chunks(self, chunks: List[ToolChunk]) -> int:
        """Add tool chunks."""
        async with self.session() as session:
            models = [
                ChunkModel(
                    chunk_id=c.chunk_id,
                    tool_id=c.tool_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    chunk_metadata=c.metadata,
                )
                for c in chunks
            ]
            session.add_all(models)
            await session.flush()

            return len(models)

    async def get_chunk(self, chunk_id: str) -> Optional[ToolChunk]:
        """Get a chunk by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(ToolChunk).where(ToolChunk.chunk_id == chunk_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return None

            return ToolChunk(
                tool_id=model.tool_id,
                chunk_id=model.chunk_id,
                chunk_index=model.chunk_index,
                text=model.text,
                metadata=model.chunk_metadata,
            )

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete tool chunk."""
        async with self.session() as session:
            # Delete chunk
            await session.execute(
                delete(ChunkModel).where(ChunkModel.chunk_id == chunk_id)
            )

            return True

    async def get_chunks_by_tool(self, tool_id: str) -> List[ToolChunk]:
        """Get all chunks for a tool."""
        async with self.session() as session:
            result = await session.execute(
                select(ChunkModel)
                .where(ChunkModel.tool_id == tool_id)
                .order_by(ChunkModel.chunk_index)
            )
            models = result.scalars().all()

            return [
                ToolChunk(
                    chunk_id=m.chunk_id,
                    tool_id=m.tool_id,
                    chunk_index=m.chunk_index,
                    text=m.text,
                    metadata=m.chunk_metadata,
                )
                for m in models
            ]

    # Credential operations
    async def add_credential(self, credential: ToolCredential) -> ToolCredential:
        """Add a credential."""
        async with self.session() as session:
            model = CredentialModel(
                credential_id=credential.credential_id,
                server_id=credential.server_id,
                tool_id=credential.tool_id,
                credential_type=credential.credential_type,
                credential_data=credential.credential_data,
                created_at=credential.created_at,
                updated_at=datetime.utcnow(),
            )
            session.add(model)
            await session.flush()

            return credential

    async def get_credential(self, server_id: str, tool_id: Optional[str] = None) -> Optional[ToolCredential]:
        """Get credentials for a server or tool."""
        async with self.session() as session:
            query = select(CredentialModel).where(CredentialModel.server_id == server_id)
            if tool_id:
                query = query.where(CredentialModel.tool_id == tool_id)

            result = await session.execute(query)
            model = result.scalar_one_or_none()

            if not model:
                return None

            return ToolCredential(
                credential_id=str(model.credential_id),
                server_id=model.server_id,
                tool_id=model.tool_id,
                credential_type=model.credential_type,
                credential_data=model.credential_data,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

    async def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        async with self.session() as session:
            await session.execute(
                delete(CredentialModel).where(CredentialModel.credential_id == credential_id)
            )

            return True

    # Stats
    async def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        async with self.session() as session:
            servers = await session.execute(select(ServerModel))
            tools = await session.execute(select(ToolModel))
            credentials = await session.execute(select(CredentialModel))
            chunks = await session.execute(select(ChunkModel))

            return {
                "servers": len(servers.scalars().all()),
                "tools": len(tools.scalars().all()),
                "credentials": len(credentials.scalars().all()),
                "chunks": len(chunks.scalars().all()),
            }
