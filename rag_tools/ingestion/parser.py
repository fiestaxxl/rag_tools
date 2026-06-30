"""
Parser for MCP responses and tool definitions.
"""
import json
from typing import Any, Dict, List, Optional
from rag_tools.storage.models import MCPTool
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
import uuid


@dataclass
class MCPToolInfo:
    """Parsed tool information from MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]]
    tags: List[str]


@dataclass
class MCPToolsResponse:
    """Response from listing tools."""
    tools: List[MCPToolInfo]
    raw_response: Dict[str, Any]


def parse_mcp_tools_response(response: Dict[str, Any]) -> MCPToolsResponse:
    """
    Parse a raw MCP JSON-RPC response for tools/list.

    Args:
        response: Raw JSON-RPC response

    Returns:
        Parsed MCPToolsResponse
    """
    result = response.get("result", {})
    raw_tools = result.get("tools", [])

    tools = []
    for raw_tool in raw_tools:
        tool = _parse_single_tool(raw_tool)
        tools.append(tool)

    return MCPToolsResponse(tools=tools, raw_response=result)


def _parse_single_tool(raw_tool: Dict[str, Any]) -> MCPToolInfo:
    """Parse a single tool from raw MCP format."""
    # Handle nested _meta structure
    meta = raw_tool.get("_meta", {})
    tags = []

    if "fastmcp" in meta:
        tags = meta["fastmcp"].get("tags", [])

    # Try different schema field names
    input_schema = (
        raw_tool.get("inputSchema")
        or raw_tool.get("input_schema")
        or raw_tool.get("parameters")
        or {}
    )

    output_schema = (
        raw_tool.get("outputSchema")
        or raw_tool.get("output_schema")
        or None
    )

    return MCPToolInfo(
        name=raw_tool.get("name", ""),
        description=raw_tool.get("description", ""),
        input_schema=_normalize_json_schema(input_schema),
        output_schema=_normalize_json_schema(output_schema) if output_schema else None,
        tags=tags,
    )


def _normalize_json_schema(schema: Any) -> Dict[str, Any]:
    """
    Normalize JSON schema to ensure consistent format.

    Args:
        schema: Raw schema (dict or string)

    Returns:
        Normalized schema dict
    """
    if schema is None:
        return {}

    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return {"type": "object"}

    if not isinstance(schema, dict):
        return {"type": "object"}

    # Ensure required structure
    normalized = {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
        "additionalProperties": schema.get("additionalProperties", True),
    }

    # Copy other common fields
    for field in ["description", "title", "definitions", "$defs"]:
        if field in schema:
            normalized[field] = schema[field]

    return normalized


def extract_server_id_from_tool(tool_id: str) -> str:
    """Extract server ID from tool ID (format: server_id:tool_name)."""
    if ":" in tool_id:
        return tool_id.rsplit(":", 1)[0]
    return ""


def extract_tool_name_from_id(tool_id: str) -> str:
    """Extract tool name from tool ID."""
    if ":" in tool_id:
        return tool_id.rsplit(":", 1)[1]
    return tool_id


def create_tool_id(server_id: str, tool_name: str) -> str:
    """Create a unique tool ID from server ID and tool name."""
    return str(uuid.uuid4())


def mcp_tool_info_to_model(
    tool_info: MCPToolInfo,
    server_id: str,
) -> MCPTool:
    """
    Convert MCPToolInfo to MCPTool model.

    Args:
        tool_info: Parsed tool info
        server_id: Server ID

    Returns:
        MCPTool model
    """
    tool_id = create_tool_id(server_id, tool_info.name)

    # `tool_info` may be a raw MCP SDK Tool (camelCase: inputSchema / outputSchema,
    # and no `tags`) — which is what sync_server passes from session.list_tools() —
    # or our internal MCPToolInfo (snake_case). Accept both so sync doesn't crash
    # with AttributeError: 'Tool' object has no attribute 'input_schema'.
    raw_in = getattr(tool_info, "inputSchema", None)
    if raw_in is None:
        raw_in = getattr(tool_info, "input_schema", None) or {}
    raw_out = getattr(tool_info, "outputSchema", None)
    if raw_out is None:
        raw_out = getattr(tool_info, "output_schema", None)

    return MCPTool(
        tool_id=tool_id,
        server_id=server_id,
        name=tool_info.name,
        description=getattr(tool_info, "description", "") or "",
        input_schema=_normalize_json_schema(raw_in),
        output_schema=_normalize_json_schema(raw_out) if raw_out else None,
        tags=getattr(tool_info, "tags", None) or [],
    )


def parse_call_arguments(input_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse input schema into a list of argument definitions.

    Args:
        input_schema: Normalized JSON schema

    Returns:
        List of argument definitions
    """
    properties = input_schema.get("properties", {})

    args = []
    for name, prop in properties.items():
        arg = {
            "name": name,
            "type": prop.get("type", "string"),
            "description": prop.get("description", ""),
            "required": name in input_schema.get("required", []),
            "default": prop.get("default"),
        }

        # Add enum if present
        if "enum" in prop:
            arg["enum"] = prop["enum"]

        args.append(arg)

    return args


def validate_tool_response(response: Dict[str, Any]) -> bool:
    """
    Validate an MCP JSON-RPC response structure.

    Args:
        response: Raw response

    Returns:
        True if valid
    """
    if not isinstance(response, dict):
        return False

    if response.get("jsonrpc") != "2.0":
        return False

    # Must have either result or error
    if "result" not in response and "error" not in response:
        return False

    return True


def extract_error_message(response: Dict[str, Any]) -> Optional[str]:
    """Extract error message from MCP response."""
    error = response.get("error")
    if isinstance(error, dict):
        return error.get("message", str(error))
    return error
