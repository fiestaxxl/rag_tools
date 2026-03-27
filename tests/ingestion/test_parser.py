import pytest
import json

from rag_tools.ingestion.parser import (
    parse_mcp_tools_response,
    extract_server_id_from_tool,
    extract_tool_name_from_id,
    create_tool_id,
    parse_call_arguments,
    validate_tool_response,
    extract_error_message,
    mcp_tool_info_to_model,
    _normalize_json_schema,
)


# ---------------------------
# Parsing MCP response
# ---------------------------

def test_parse_mcp_tools_response_basic():
    response = {"jsonrpc":"2.0",
                "id":2,
                "result":{
                    "tools":[
                        {"name":"search_entity","description":"Search for an entity (author, source, institution) in OpenAlex and return its ID.\nThis ID can be further used to search for papers using the search_papers or download_papers_from_search tools.\n\nArgs:\n    entity_type: Type of entity to search for (\"author\", \"source\", \"institution\")\n    entity_name: Name of the entity to search for (e.g., author name, journal name, institution name)","inputSchema":{"additionalProperties":False,"properties":{"entity_type":{"type":"string"},"entity_name":{"type":"string"}},"required":["entity_type","entity_name"],"type":"object"},"outputSchema":{"additionalProperties":True,"type":"object"},"_meta":{"fastmcp":{"tags":[]}}},
                        {"name":"search_papers","description":"Search papers in OpenAlex using filters and return normalized metadata.\n\nArgs:\n    keywords: Search keywords (e.g., \"crispr cas9\")\n    author_id: OpenAlex author ID (e.g., \"A123...\")\n    institution_id: OpenAlex institution ID\n    source_id: OpenAlex source ID\n    publication_year: Year filter (e.g., \"2025\", \">2020\")\n    open_access: Include only open access works\n    has_pdf: Include only works with PDF available\n    limit: Max number of results\n    sort: OpenAlex sort field (e.g., \"cited_by_count:desc\")","inputSchema":{"additionalProperties":False,"properties":{"keywords":{"default":None,"type":"string"},"author_id":{"default":None,"type":"string"},"institution_id":{"default":None,"type":"string"},"source_id":{"default":None,"type":"string"},"publication_year":{"default":None,"type":"string"},"open_access":{"default":True,"type":"boolean"},"has_pdf":{"default":True,"type":"boolean"},"limit":{"default":10,"type":"integer"},"sort":{"default":None,"type":"string"}},"type":"object"},"outputSchema":{"additionalProperties":True,"type":"object"},"_meta":{"fastmcp":{"tags":[]}}},
                        {"name":"download_papers_from_search","description":"Search papers in OpenAlex and upload found PDFs directly to S3.","inputSchema":{"additionalProperties":False,"properties":{"keywords":{"default":None,"type":"string"},"author_id":{"default":None,"type":"string"},"institution_id":{"default":None,"type":"string"},"source_id":{"default":None,"type":"string"},"publication_year":{"default":None,"type":"string"},"open_access":{"default":True,"type":"boolean"},"limit":{"default":10,"type":"integer"},"sort":{"default":None,"type":"string"},"session_id":{"default":"1","type":"string"},"user_id":{"default":"1","type":"string"}},"type":"object"},"outputSchema":{"additionalProperties":True,"type":"object"},"_meta":{"fastmcp":{"tags":[]}}}
                        ]
                    }
                }

    parsed = parse_mcp_tools_response(response)

    assert len(parsed.tools) == 3
    tool = parsed.tools[0]

    assert tool.name == "search_entity"
    assert tool.description == "Search for an entity (author, source, institution) in OpenAlex and return its ID.\nThis ID can be further used to search for papers using the search_papers or download_papers_from_search tools.\n\nArgs:\n    entity_type: Type of entity to search for (\"author\", \"source\", \"institution\")\n    entity_name: Name of the entity to search for (e.g., author name, journal name, institution name)"
    assert tool.input_schema["type"] == "object"
    assert "entity_name" in tool.input_schema["properties"]


def test_parse_mcp_tools_with_tags():
    response = {
        "result": {
            "tools": [
                {
                    "name": "tool1",
                    "_meta": {
                        "fastmcp": {
                            "tags": ["a", "b"]
                        }
                    }
                }
            ]
        }
    }

    parsed = parse_mcp_tools_response(response)

    assert parsed.tools[0].tags == ["a", "b"]


# ---------------------------
# Schema normalization
# ---------------------------

def test_normalize_schema_from_string():
    raw = json.dumps({
        "type": "object",
        "properties": {"x": {"type": "number"}}
    })

    normalized = _normalize_json_schema(raw)

    assert normalized["type"] == "object"
    assert "x" in normalized["properties"]


def test_normalize_schema_invalid_string():
    normalized = _normalize_json_schema("not json")

    assert normalized["type"] == "object"


def test_normalize_schema_none():
    normalized = _normalize_json_schema(None)

    assert normalized == {}


def test_normalize_schema_non_dict():
    normalized = _normalize_json_schema(123)

    assert normalized["type"] == "object"


# ---------------------------
# Tool ID helpers
# ---------------------------

def test_create_tool_id():
    assert create_tool_id("s1", "t1") == "s1:t1"


def test_extract_server_id():
    assert extract_server_id_from_tool("s1:t1") == "s1"


def test_extract_tool_name():
    assert extract_tool_name_from_id("s1:t1") == "t1"


def test_extract_without_colon():
    assert extract_server_id_from_tool("t1") == ""
    assert extract_tool_name_from_id("t1") == "t1"


# ---------------------------
# Model conversion
# ---------------------------

def test_mcp_tool_info_to_model():
    from rag_tools.ingestion.parser import MCPToolInfo

    info = MCPToolInfo(
        name="tool1",
        description="desc",
        input_schema={"type": "object"},
        output_schema=None,
        tags=["a"]
    )

    model = mcp_tool_info_to_model(info, "s1")

    assert model.tool_id == "s1:tool1"
    assert model.server_id == "s1"
    assert model.name == "tool1"


# ---------------------------
# Argument parsing
# ---------------------------

def test_parse_call_arguments():
    schema = {
        "type": "object",
        "properties": {
            "x": {
                "type": "string",
                "description": "test",
                "default": "a"
            }
        },
        "required": ["x"]
    }

    args = parse_call_arguments(schema)

    assert len(args) == 1
    arg = args[0]

    assert arg["name"] == "x"
    assert arg["required"] is True
    assert arg["default"] == "a"


def test_parse_call_arguments_with_enum():
    schema = {
        "properties": {
            "x": {
                "type": "string",
                "enum": ["a", "b"]
            }
        }
    }

    args = parse_call_arguments(schema)

    assert args[0]["enum"] == ["a", "b"]


# ---------------------------
# Validation
# ---------------------------

def test_validate_tool_response_valid():
    response = {
        "jsonrpc": "2.0",
        "result": {}
    }

    assert validate_tool_response(response) is True


def test_validate_tool_response_invalid_version():
    response = {
        "jsonrpc": "1.0",
        "result": {}
    }

    assert validate_tool_response(response) is False


def test_validate_tool_response_missing_fields():
    assert validate_tool_response({}) is False
    assert validate_tool_response("not dict") is False


# ---------------------------
# Error extraction
# ---------------------------

def test_extract_error_message():
    response = {
        "error": {
            "message": "fail"
        }
    }

    assert extract_error_message(response) == "fail"


def test_extract_error_message_non_dict():
    response = {
        "error": "something wrong"
    }

    assert extract_error_message(response) == "something wrong"


def test_extract_error_message_none():
    assert extract_error_message({}) is None