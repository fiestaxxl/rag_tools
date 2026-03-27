import pytest
import pytest_asyncio
import numpy as np

from rag_tools.retrieval.embedder import APIEmbedder
from rag_tools.config.settings import APIEmbeddingSettings


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def embedder(monkeypatch):
    config = APIEmbeddingSettings(
        url="http://10.32.1.36:5002/embed"
    )
    emb = APIEmbedder(config)
    return emb

@pytest.mark.asyncio
async def test_embed(embedder):
    result = await embedder.embed(["a", "b"])

    assert result.shape[0] == 2
    assert result.shape[1] > 0  # don't hardcode dim

    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

@pytest.mark.asyncio
async def test_embed_query(embedder):
    result = await embedder.embed_query("hello")

    assert result.shape == (1024,)

@pytest.mark.asyncio
async def test_similarity(embedder):
    a = np.array([[1, 0]])
    b = np.array([[1, 0]])

    sim = await embedder.compute_similarity(a, b)

    assert sim.shape == (1, 1)
    assert sim[0][0] == pytest.approx(1.0)