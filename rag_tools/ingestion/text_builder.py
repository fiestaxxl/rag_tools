"""
Text builder for creating searchable tool representations.
"""
import json
from typing import Any, Dict, List, Optional, Tuple
from rag_tools.storage.models import MCPTool, ToolChunk
from rag_tools.config.settings import settings
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextBuilder:
    """Builds searchable text from tool definitions."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\n", ".", " ", ""
            ]
        )
    def build_full_text(self, tool: MCPTool) -> str:
        """
        Build full searchable text for a tool.

        Args:
            tool: Tool model

        Returns:
            Combined searchable text
        """
        parts = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
        ]

        # Add tags if present
        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")

        # Add input schema
        if tool.input_schema:
            schema_text = self._format_schema(tool.input_schema)
            parts.append(f"Parameters: {schema_text}")

        # Add output schema if present
        if tool.output_schema:
            output_text = self._format_schema(tool.output_schema)
            parts.append(f"Returns: {output_text}")

        return "\n".join(parts)

    def build_summary_text(self, tool: MCPTool) -> str:
        """
        Build a shorter summary text for quick matching.

        Args:
            tool: Tool model

        Returns:
            Summary text
        """
        parts = [
            tool.name,
            tool.description,
        ]

        if tool.tags:
            parts.append(", ".join(tool.tags))

        return " | ".join(parts)

    def build_name_text(self, tool: MCPTool) -> str:
        """
        Build text focused on tool name.

        Args:
            tool: Tool model

        Returns:
            Name-focused text
        """
        parts = [tool.name]

        if tool.tags:
            parts.extend(tool.tags)

        return " ".join(parts)

    def build_chunks(self, tool: MCPTool) -> List[ToolChunk]:
        """
        Split tool text into overlapping chunks using LangChain.
        """
        full_text = self.build_full_text(tool)

        # Use splitter
        split_texts = self._splitter.split_text(full_text)

        total_chunks = len(split_texts)

        chunks: List[ToolChunk] = []

        for idx, chunk_text in enumerate(split_texts):
            chunks.append(
                ToolChunk(
                    chunk_id=str(uuid.uuid4()),
                    tool_id=tool.tool_id,
                    chunk_index=idx,
                    text=chunk_text.strip(),
                    metadata={
                        "name": tool.name,
                        "tags": tool.tags,
                        "chunk_count": total_chunks,
                        # optional: approximate position
                        "chunk_start": idx * (self.chunk_size - self.chunk_overlap),
                    },
                )
            )

        return chunks

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format JSON schema as readable text."""
        if not schema or not isinstance(schema, dict):
            return ""

        parts = []
        properties = schema.get("properties", {})

        for name, prop in properties.items():
            prop_type = prop.get("type", "any")
            prop_desc = prop.get("description", "")
            required = name in schema.get("required", [])

            required_str = "[required]" if required else "[optional]"
            desc_str = f" - {prop_desc}" if prop_desc else ""

            # Handle enum
            if "enum" in prop:
                enum_vals = ", ".join(str(v) for v in prop["enum"][:5])
                if len(prop["enum"]) > 5:
                    enum_vals += ", ..."
                desc_str += f" (one of: {enum_vals})"

            # Handle default
            if "default" in prop:
                desc_str += f" [default: {prop['default']}]"

            parts.append(f"{name}: {prop_type}{required_str}{desc_str}")

        return "; ".join(parts)

    def build_search_text(
        self,
        tool: MCPTool,
        include_schema: bool = True,
        include_description: bool = True,
    ) -> str:
        """
        Build customized search text.

        Args:
            tool: Tool model
            include_schema: Include parameter schema
            include_description: Include full description

        Returns:
            Searchable text
        """
        parts = []

        # Always include name
        parts.append(f"Name: {tool.name}")

        # Include description
        if include_description:
            parts.append(f"Description: {tool.description}")

        # Include tags
        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")

        # Include schema
        if include_schema and tool.input_schema:
            parts.append(f"Parameters: {self._format_schema(tool.input_schema)}")

        return "\n".join(parts)


def build_tool_embedding_text(tool: MCPTool) -> str:
    """
    Build the default embedding text for a tool.

    Args:
        tool: Tool model

    Returns:
        Text for embedding
    """
    builder = TextBuilder()
    return builder.build_full_text(tool)


def build_tool_metadata(tool: MCPTool) -> Dict[str, Any]:
    """
    Build metadata for a tool.

    Args:
        tool: Tool model

    Returns:
        Metadata dict for storage
    """
    return {
        "tool_id": tool.tool_id,
        "server_id": tool.server_id,
        "name": tool.name,
        "description": tool.description[:500] if len(tool.description) > 500 else tool.description,
        "tags": tool.tags,
        "input_schema": json.dumps(tool.input_schema) if tool.input_schema else None,
        "created_at": tool.created_at.isoformat() if tool.created_at else None,
    }
