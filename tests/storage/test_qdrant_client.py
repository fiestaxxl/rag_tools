import pytest
import pytest_asyncio
from rag_tools.storage import QdrantClientWrapper  # assuming your wrapper
from rag_tools.config.settings import QdrantSettings
import uuid

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def qdrant_client():
    config = QdrantSettings(
        url="http://localhost:6333"  # container name from docker-compose or localhost
    )

    client = QdrantClientWrapper(config)
    await client.connect()
    await client.set_embedding_dim(4)  # make sure your tests expect this

    yield client

    await client.close()

@pytest.mark.asyncio
async def test_collection_lifecycle(qdrant_client):
    created = await qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert qdrant_client.collection_exists("test")

    info = await qdrant_client.get_collection_info("test")

    assert info["vector_size"] == 4

    result = await qdrant_client.delete_collection('test')
    assert result is True

    collection = await qdrant_client.collection_exists('test')

    assert collection is False

@pytest.mark.asyncio
async def test_upsert_and_search(qdrant_client):
    created = await qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert await qdrant_client.collection_exists("test")

    vectors = [[0.1, 0.2, 0.3, 0.4]]
    payloads = [{"tool_id": "t1"}]

    await qdrant_client.upsert_points("test", vectors, payloads)

    results = await qdrant_client.search("test", [0.1, 0.2, 0.3, 0.4])

    assert len(results) > 0
    assert results[0]["payload"]["tool_id"] == "t1"

    result = await qdrant_client.delete_collection('test')
    assert result is True

    collection = await qdrant_client.collection_exists('test')

    assert collection is False

@pytest.mark.asyncio
async def test_delete_points(qdrant_client):
    created = await qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert await qdrant_client.collection_exists("test")


    vectors = [[0, 0, 0, 1]]
    payloads = [{"x": 1}]
    ids = [str(uuid.uuid4())]

    await qdrant_client.upsert_points("test", vectors, payloads, ids)

    points = await qdrant_client.retrieve('test', ids, with_vectors=True)
    assert len(points) == len(vectors)
    assert points[0]['vector'] == vectors[0]

    await qdrant_client.delete_points("test", ids)

    count = await qdrant_client.count("test")

    assert count == 0

    result = await qdrant_client.delete_collection('test')
    assert result is True

    collection = await qdrant_client.collection_exists('test')

    assert collection is False


