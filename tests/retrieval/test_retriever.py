import pytest
import pytest_asyncio
import numpy as np

from rag_tools.retrieval.retriever import ToolRetriever, RetrievalConfig
from rag_tools.storage.models import RetrievalResult


pytestmark = pytest.mark.asyncio


class DummyQdrant:
    async def set_embedding_dim(self, dim):
        self.dim = dim

    async def search(self, **kwargs):
        return [
            {"id": "1", "score": 0.9, "payload": {"tool_id": "t1", "name": "A"}},
            {"id": "2", "score": 0.8, "payload": {"tool_id": "t2", "name": "B"}},
        ]

    async def search_batch(self, **kwargs):
        return [await self.search(), await self.search()]

    async def retrieve(self, **kwargs):
        return [{"id": "1", "payload": {"tool_id": "t1", "name": "A"}}]

    async def scroll(self, **kwargs):
        return (await self.search(), None)


class DummyEmbedder:
    async def initialize(self): pass

    @property
    def embedding_dim(self): return 4

    async def embed_query(self, q):
        return np.array([1, 0, 0, 0])

    async def embed(self, qs):
        return np.array([[1, 0, 0, 0] for _ in qs])


@pytest_asyncio.fixture
async def retriever():
    return ToolRetriever(DummyQdrant(), DummyEmbedder())

@pytest.mark.asyncio
async def test_retrieve(retriever):
    results = await retriever.retrieve("test")

    assert len(results) > 0
    assert isinstance(results[0], RetrievalResult)

@pytest.mark.asyncio
async def test_retrieve_batch(retriever):
    results = await retriever.retrieve_batch(["a", "b"])

    assert len(results) == 2
    assert len(results[0]) > 0

@pytest.mark.asyncio
async def test_get_tool_by_id(retriever):
    result = await retriever.get_tool_by_id("t1")

    assert result is not None
    assert result.tool_id == "t1"