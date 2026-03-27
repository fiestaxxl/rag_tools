import pytest
import pytest_asyncio

from rag_tools.retrieval.reranker import APIReranker
from rag_tools.storage.models import RetrievalResult
from rag_tools.config.settings import APIRerankerSettings


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def api_reranker(monkeypatch):
    config = APIRerankerSettings(
        url="http://10.32.1.36:5001/rerank"
    )
    emb = APIReranker(config)
    return emb

def make_results():
    return [
        RetrievalResult(server_id='s1', tool_id="s1:t1", name="a", description="x", score=0.1, rank=1),
        RetrievalResult(server_id='s2', tool_id="s2:t1", name="b", description="y", score=0.2, rank=2),
    ]

@pytest.mark.asyncio
async def test_api_rerank(api_reranker):
    results = make_results()

    reranked = await api_reranker.rerank("query", results)

    assert len(reranked) == 2
    assert reranked[0].rerank_score >= reranked[1].rerank_score
