import pytest
import pytest_asyncio
from rag_tools.storage import MCPServer, MCPTool, ToolChunk, ToolCredential, PostgresClient
from rag_tools.config.settings import PostgresSettings


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def postgres_client():
    # Use service name 'postgres' (docker-compose network) instead of localhost
    config = PostgresSettings(
        host="host.docker.internal",       # container name in docker-compose
        port=5432,
        user="rag_tools",
        password="rag_tools_password",
        database="rag_tools",
    )

    client = PostgresClient(config)
    await client.initialize()   # your async init

    yield client

    await client.close()

@pytest.mark.asyncio
async def test_add_and_get_server(postgres_client):
    server = MCPServer(
        server_id="s1",
        name="Server",
        url="http://test"
    )

    result = await postgres_client.add_server(server)

    result = await postgres_client.get_server("s1")

    assert result is not None
    assert result.server_id == "s1"

    result = await postgres_client.delete_server('s1')

@pytest.mark.asyncio
async def test_list_servers(postgres_client):
    await postgres_client.add_server(
        MCPServer(server_id="s1", name="A", url="http://a")
    )
    await postgres_client.add_server(
        MCPServer(server_id="s2", name="B", url="http://b")
    )

    servers = await postgres_client.list_servers()

    assert len(servers) == 2

    for server in servers:
        await postgres_client.delete_server(server.server_id)
    
    servers = await postgres_client.list_servers()

    assert len(servers) == 0

@pytest.mark.asyncio
async def test_add_and_get_server(postgres_client):
    server = MCPServer(
        server_id="s1",
        name="Server",
        url="http://test"
    )

    result = await postgres_client.add_server(server)

    result = await postgres_client.get_server("s1")

    assert result is not None
    assert result.server_id == "s1"

    result = await postgres_client.delete_server('s1')
    
@pytest.mark.asyncio
async def test_tool_crud(postgres_client):
    await postgres_client.add_server(
        MCPServer(server_id="s1", name="A", url="http://a")
    )

    tool = MCPTool(
        tool_id="s1:t1",
        server_id="s1",
        name="tool",
        description="desc"
    )

    await postgres_client.add_tool(tool)

    result = await postgres_client.get_tool("s1:t1")

    assert result is not None
    assert result.name == "tool"

    await postgres_client.delete_server("s1")

    assert await postgres_client.get_tool('s1:t1') is None

@pytest.mark.asyncio
async def test_dublicate_tool(postgres_client):
    await postgres_client.add_server(
        MCPServer(server_id="s1", name="A", url="http://a")
    )

    tool = MCPTool(
        tool_id="s1:t1",
        server_id="s1",
        name="tool",
        description="desc"
    )

    await postgres_client.add_tool(tool)

    result = await postgres_client.get_tool("s1:t1")

    assert result is not None
    assert result.name == "tool"

    await postgres_client.add_tool(tool)

    result = await postgres_client.get_tool("s1:t1")

    assert result is not None
    assert result.name == "tool"
    
    await postgres_client.delete_server("s1")

    assert await postgres_client.get_tool('s1:t1') is None



@pytest.mark.asyncio
async def test_chunks(postgres_client):
    await postgres_client.add_chunks([
        ToolChunk(tool_id="t1", chunk_index=0, text="a"),
        ToolChunk(tool_id="t1", chunk_index=1, text="b"),
    ])

    chunks = await postgres_client.get_chunks_by_tool("t1")

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0

    for chunk in chunks:
        await postgres_client.delete_chunk(chunk.chunk_id)




@pytest.mark.asyncio
async def test_credentials(postgres_client):
    cred = ToolCredential(
        server_id="s1",
        credential_type="api_key",
        credential_data={"key": "123"}
    )

    await postgres_client.add_credential(cred)

    result = await postgres_client.get_credential("s1")

    assert result is not None
    assert result.credential_data["key"] == "123"

    await postgres_client.delete_credential(result.credential_id)

    assert await postgres_client.get_credential("s1") is None

@pytest.mark.asyncio
async def test_cascade_delete_server(postgres_client):
    await postgres_client.add_server(
        MCPServer(server_id="s1", name="A", url="http://a")
    )

    await postgres_client.add_tool(
        MCPTool(tool_id="s1:t1", server_id="s1", name="t", description="d")
    )

    await postgres_client.add_chunks([
        ToolChunk(tool_id="s1:t1", chunk_index=0, text="x")
    ])

    await postgres_client.delete_server("s1")
    server = await postgres_client.get_server('s1')
    assert server is None

    tools = await postgres_client.get_tools_by_server("s1")
    assert tools == []

    chunks = await postgres_client.get_chunks_by_tool('s1:t1')
    assert chunks == []


