import numpy as np
import pytest
from rag_tools.storage import QdrantClientWrapper  # assuming your wrapper
from rag_tools.config.settings import QdrantSettings
import uuid

@pytest.fixture
def qdrant_client():
    config = QdrantSettings(
        url="http://host.docker.internal:6333"  # container name from docker-compose
    )

    client = QdrantClientWrapper(config)
    client.connect()
    client.set_embedding_dim(4)  # make sure your tests expect this

    yield client

    client.close()

def test_collection_lifecycle(qdrant_client):
    created = qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert qdrant_client.collection_exists("test")

    info = qdrant_client.get_collection_info("test")

    assert info["vector_size"] == 4

    result = qdrant_client.delete_collection('test')
    assert result is True

    collection = qdrant_client.collection_exists('test')

    assert collection is False


def test_upsert_and_search(qdrant_client):
    created = qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert qdrant_client.collection_exists("test")

    vectors = [[0.1, 0.2, 0.3, 0.4]]
    payloads = [{"tool_id": "t1"}]

    qdrant_client.upsert_points("test", vectors, payloads)

    results = qdrant_client.search("test", [0.1, 0.2, 0.3, 0.4])

    assert len(results) > 0
    assert results[0]["payload"]["tool_id"] == "t1"

    result = qdrant_client.delete_collection('test')
    assert result is True

    collection = qdrant_client.collection_exists('test')

    assert collection is False


def test_delete_points(qdrant_client):
    created = qdrant_client.create_collection("test", vector_size=4)

    assert created is True
    assert qdrant_client.collection_exists("test")


    vectors = [[0, 0, 0, 1]]
    payloads = [{"x": 1}]
    ids = [str(uuid.uuid4())]

    qdrant_client.upsert_points("test", vectors, payloads, ids)

    points = qdrant_client.retrieve('test', ids, with_vectors=True)
    assert len(points) == len(vectors)
    assert points[0]['vector'] == vectors[0]

    qdrant_client.delete_points("test", ids)

    count = qdrant_client.count("test")

    assert count == 0

    result = qdrant_client.delete_collection('test')
    assert result is True

    collection = qdrant_client.collection_exists('test')

    assert collection is False


