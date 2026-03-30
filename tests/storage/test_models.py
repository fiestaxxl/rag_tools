from rag_tools.storage import MCPServer, MCPProtocol, MCPTool, ToolCredential, ToolChunk, ToolStatus


def test_hhtp_mcp_server_creation():
    server = MCPServer(
        server_id="s1",
        name="Test Server",
        protocol=MCPProtocol.HTTP,
        url="http://localhost"
    )

    assert server.server_id == "s1"
    assert server.status == ToolStatus.ACTIVE

def test_stdio_mcp_server_creation():
    server = MCPServer(
        server_id="s1",
        name="Test Server",
        protocol=MCPProtocol.STDIO,
        command="npx",
        args=['heheh'],
        env={'env1': 'val1'}
    )

    assert server.server_id == "s1"
    assert server.status == ToolStatus.ACTIVE


def test_tool_defaults():
    tool = MCPTool(
        tool_id="s1:t1",
        server_id="s1",
        name="tool",
        description="desc"
    )

    assert tool.tags == []
    assert tool.status == ToolStatus.ACTIVE


def test_chunk_creation():
    chunk = ToolChunk(
        tool_id="t1",
        chunk_index=0,
        text="hello"
    )

    assert chunk.text == "hello"
    assert chunk.metadata == {}