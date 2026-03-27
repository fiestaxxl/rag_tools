"""
RAG Tools Module - A RAG-based tool retrieval system for MCP servers.
"""
from rag_tools.main import (
    RAGToolsManager,
    create_manager,
    create_pipeline,
    create_default_suite,
)
from rag_tools.storage.models import (
    MCPServer,
    MCPTool,
    RetrievalResult
)

from rag_tools.evaluation.evaluator import EvaluationSuite, EvaluationReport
from rag_tools.retrieval.pipeline import PipelineConfig, PipelineResult
from rag_tools.tools.add_tool import ToolAdder, add_mcp_server
from rag_tools.tools.remove_tool import ToolRemover, remove_mcp_server, remove_tool_by_id
from rag_tools.tools.sync_mcp import SyncResult, MCPSyncer, sync_server_by_id, sync_all_servers

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "RAGToolsManager",
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
    "PipelineConfig",
    # Functions
    "create_manager",
    "create_pipeline",
    "create_default_suite",
    "add_mcp_server",
    "remove_mcp_server",
    "remove_tool_by_id",
    "sync_server_by_id",
    "sync_all_servers",
]
