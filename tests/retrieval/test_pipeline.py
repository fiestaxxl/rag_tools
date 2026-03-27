import pytest
import pytest_asyncio

from rag_tools.retrieval.pipeline import ToolRetrievalPipeline, PipelineConfig
from rag_tools.storage.models import RetrievalResult


pytestmark = pytest.mark.asyncio


class DummyRetriever:
    async def initialize(self): pass

    async def retrieve(self, query, config):
        return [
            RetrievalResult(tool_id="s1:t1", server_id='s1', name="a", description="x", score=0.5, rank=1)
        ]

    async def retrieve_batch(self, queries, config):
        return [[
            RetrievalResult(tool_id="s2:t1", server_id='s2', name="a", description="x", score=0.5, rank=1)
        ] for _ in queries]


class DummyReranker:
    async def initialize(self): pass

    async def rerank(self, query, results, top_k=None):
        for r in results:
            r.rerank_score = 1.0
        return results[:top_k]


@pytest_asyncio.fixture
async def pipeline():
    return ToolRetrievalPipeline(
        embedder=None,
        reranker=DummyReranker(),
        retriever=DummyRetriever(),
    )

@pytest.mark.asyncio
async def test_pipeline_retrieve(pipeline):
    result = await pipeline.retrieve("test", PipelineConfig())

    assert result.results
    assert isinstance(result.results, list)
    assert result.results[0].tool_id == 's1:t1'
    assert result.metadata["reranked"] is True

@pytest.mark.asyncio
async def test_pipeline_batch(pipeline):
    results = await pipeline.retrieve_batch(["a", "b"])

    assert len(results) == 2

@pytest.mark.asyncio
async def test_pipeline_fallback(pipeline):
    result = await pipeline.retrieve_with_fallback("test")

    assert result is not None