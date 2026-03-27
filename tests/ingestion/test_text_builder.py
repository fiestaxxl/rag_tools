import pytest
import pytest_asyncio
from datetime import datetime

from rag_tools.ingestion.text_builder import (
    TextBuilder,
    build_tool_embedding_text,
    build_tool_metadata,
)
from rag_tools.storage.models import MCPTool


pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_tool():
    return MCPTool(
        tool_id="s1:t1",
        server_id="s1",
        name="weather",
        description="Get weather information for a city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
            },
        },
        tags=["weather", "api"],
        created_at=datetime.utcnow(),
    )


# -----------------------
# FULL TEXT
# -----------------------
def test_build_full_text(sample_tool):
    builder = TextBuilder()

    text = builder.build_full_text(sample_tool)

    assert "Tool: weather" in text
    assert "Description:" in text
    assert "Tags:" in text
    assert "Parameters:" in text
    assert "Returns:" in text


# -----------------------
# SUMMARY TEXT
# -----------------------
def test_build_summary_text(sample_tool):
    builder = TextBuilder()

    text = builder.build_summary_text(sample_tool)

    assert "weather" in text
    assert "|" in text
    assert "api" in text


# -----------------------
# NAME TEXT
# -----------------------
def test_build_name_text(sample_tool):
    builder = TextBuilder()

    text = builder.build_name_text(sample_tool)

    assert text.startswith("weather")
    assert "api" in text


# -----------------------
# CHUNKING
# -----------------------
def test_build_chunks_small_text(sample_tool):
    builder = TextBuilder(chunk_size=1000)

    chunks = builder.build_chunks(sample_tool)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_build_chunks_large_text(sample_tool):
    builder = TextBuilder(chunk_size=30, chunk_overlap=10)

    chunks = builder.build_chunks(sample_tool)

    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.tool_id == sample_tool.tool_id
        assert "name" in chunk.metadata
        assert chunk.metadata["chunk_count"] == len(chunks)


# -----------------------
# SCHEMA FORMATTING
# -----------------------
def test_format_schema(sample_tool):
    builder = TextBuilder()

    text = builder._format_schema(sample_tool.input_schema)

    assert "city:" in text
    assert "[required]" in text
    assert "[optional]" in text
    assert "default" in text
    assert "one of:" in text


def test_format_schema_empty():
    builder = TextBuilder()

    assert builder._format_schema({}) == ""
    assert builder._format_schema(None) == ""


# -----------------------
# SEARCH TEXT
# -----------------------
def test_build_search_text_full(sample_tool):
    builder = TextBuilder()

    text = builder.build_search_text(sample_tool)

    assert "Name:" in text
    assert "Description:" in text
    assert "Tags:" in text
    assert "Parameters:" in text


def test_build_search_text_minimal(sample_tool):
    builder = TextBuilder()

    text = builder.build_search_text(
        sample_tool,
        include_schema=False,
        include_description=False,
    )

    assert "Name:" in text
    assert "Description:" not in text
    assert "Parameters:" not in text


# -----------------------
# EMBEDDING TEXT
# -----------------------
def test_build_tool_embedding_text(sample_tool):
    text = build_tool_embedding_text(sample_tool)

    assert "Tool:" in text
    assert "Description:" in text


# -----------------------
# METADATA
# -----------------------
def test_build_tool_metadata(sample_tool):
    metadata = build_tool_metadata(sample_tool)

    assert metadata["tool_id"] == "s1:t1"
    assert metadata["server_id"] == "s1"
    assert metadata["name"] == "weather"
    assert isinstance(metadata["input_schema"], str)
    assert metadata["created_at"] is not None


def test_build_tool_metadata_truncates_description():
    tool = MCPTool(
        tool_id="t1",
        server_id="s1",
        name="test",
        description="x" * 1000,
        input_schema={},
        output_schema=None,
        tags=[],
    )

    metadata = build_tool_metadata(tool)

    assert len(metadata["description"]) == 500