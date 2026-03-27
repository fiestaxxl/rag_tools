"""
Data models for the RAG Tools module.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, JSON, DateTime, Boolean, Integer, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid


class ToolStatus(str, Enum):
    """Status of a tool in the registry."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    SYNCING = "syncing"


class MCPServer(BaseModel):
    """MCP server metadata."""
    server_id: str = Field(..., description="Unique identifier for the MCP server")
    name: str = Field(..., description="Display name of the server")
    url: str = Field(..., description="HTTP URL of the MCP server")
    description: Optional[str] = Field(None, description="Description of the server")
    session_id: Optional[str] = Field(None, description="MCP session ID if applicable")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    timeout: int = Field(30, description="Request timeout in seconds")
    status: ToolStatus = Field(ToolStatus.ACTIVE, description="Server status")
    last_synced: Optional[datetime] = Field(None, description="Last sync timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MCPTool(BaseModel):
    """Single MCP tool."""
    tool_id: str = Field(..., description="Unique identifier (server_id:tool_name)")
    server_id: str = Field(..., description="Parent server ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = Field(None)
    tags: List[str] = Field(default_factory=list)
    status: ToolStatus = Field(ToolStatus.ACTIVE)
    version: str = Field("1.0.0")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ToolCredential(BaseModel):
    """Credentials for a tool or server."""
    credential_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str = Field(..., description="Associated server ID")
    tool_id: Optional[str] = Field(None, description="Associated tool ID if specific")
    credential_type: str = Field(..., description="Type: api_key, oauth, basic, bearer, etc.")
    credential_data: Dict[str, str] = Field(..., description="Encrypted credential data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ToolChunk(BaseModel):
    """Chunk of tool text for embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# SQLAlchemy ORM Models
Base = declarative_base()


class ServerModel(Base):
    """PostgreSQL model for MCP servers."""
    __tablename__ = "servers"

    server_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    session_id = Column(String(255), nullable=True)
    headers = Column(JSON, default=dict)
    timeout = Column(Integer, default=30)
    status = Column(String(50), default="active")
    last_synced = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_servers_status", "status"),
        Index("idx_servers_name", "name"),
    )


class ToolModel(Base):
    """PostgreSQL model for MCP tools."""
    __tablename__ = "tools"

    tool_id = Column(String(255), primary_key=True)
    server_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    input_schema = Column(JSON, default=dict)
    output_schema = Column(JSON, nullable=True)
    tags = Column(JSON, default=list)
    status = Column(String(50), default="active")
    version = Column(String(50), default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_tools_server_id", "server_id"),
        Index("idx_tools_name", "name"),
        Index("idx_tools_status", "status"),
    )


class CredentialModel(Base):
    """PostgreSQL model for credentials."""
    __tablename__ = "credentials"

    credential_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    server_id = Column(String(255), nullable=False)
    tool_id = Column(String(255), nullable=True)
    credential_type = Column(String(100), nullable=False)
    credential_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_credentials_server_id", "server_id"),
        Index("idx_credentials_tool_id", "tool_id"),
    )


class ChunkModel(Base):
    """PostgreSQL model for tool chunks."""
    __tablename__ = "chunks"

    chunk_id = Column(String(255), primary_key=True)
    tool_id = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, default=dict)

    __table_args__ = (
        Index("idx_chunks_tool_id", "tool_id"),
    )


class RetrievalResult(BaseModel):
    """Result from a retrieval query."""
    tool_id: str
    server_id: str
    name: str
    description: str
    score: float
    rerank_score: Optional[float] = None
    rank: int
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result from evaluation metrics."""
    metric_name: str
    value: float
    details: Dict[str, Any] = Field(default_factory=dict)
