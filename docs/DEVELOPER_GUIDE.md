# Developer Guide

This guide provides information for developers who want to extend or contribute to RAG Tools.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Adding Custom Embedders](#adding-custom-embedders)
- [Adding Custom Rerankers](#adding-custom-rerankers)
- [Storage Backend Integration](#storage-backend-integration)
- [Running Tests](#running-tests)
- [Contributing](#contributing)

---

## Architecture Overview

RAG Tools follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│                  (RAGToolsManager)                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Core Components                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │
│  │  Ingestion    │  │   Retrieval   │  │    Evaluation     │    │
│  │               │  │               │  │                   │    │
│  │ • Parser      │  │ • Embedder    │  │ • Metrics         │    │
│  │ • TextBuilder │  │ • Reranker    │  │ • Evaluator       │    │
│  │ • Indexer     │  │ • Retriever   │  │ • Suite           │    │
│  │               │  │ • Pipeline    │  │                   │    │
│  └───────────────┘  └───────────────┘  └───────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Storage Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐        ┌─────────────────────────────┐│
│  │     PostgreSQL      │        │        Qdrant                ││
│  │                     │        │                             ││
│  │ • Servers           │        │ • Tool embeddings           ││
│  │ • Tools             │        │ • Tool chunks               ││
│  │ • Credentials       │        │                             ││
│  │ • Chunks            │        │                             ││
│  └─────────────────────┘        └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **RAGToolsManager**: Main entry point, coordinates all components
2. **Ingestion**: Parses MCP tools, builds searchable text, indexes to storage
3. **Retrieval**: Embeds queries, searches vectors, reranks results
4. **Evaluation**: Measures retrieval quality with various metrics
5. **Storage**: PostgreSQL for metadata, Qdrant for vectors

---

## Adding Custom Embedders

To add a new embedder, extend the `BaseEmbedder` class:

```python
import asyncio
from typing import List, Optional, Union
import numpy as np

from rag_tools.retrieval.embedder import BaseEmbedder

class MyCustomEmbedder(BaseEmbedder):
    """Custom embedder implementation."""

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._model = None

    async def initialize(self) -> None:
        """Initialize the embedder."""
        if self._initialized:
            return

        # Load your model here
        # self._model = load_my_model(self.model_path)
        # self._embedding_dim = self._model.get_dimension()

        self._initialized = True

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"custom-embedder-{self.model_path}"

    async def _embed_impl(
        self,
        texts: List[str],
        batch_size: Optional[int],
    ) -> np.ndarray:
        """Implement the actual embedding logic."""
        # Your embedding implementation
        embeddings = []
        for i in range(0, len(texts), batch_size or 32):
            batch = texts[i:i + batch_size or 32]
            # batch_embeddings = self._model.encode(batch)
            # embeddings.extend(batch_embeddings)
            pass  # Replace with actual implementation

        return np.array(embeddings)
```

### Registering the Custom Embedder

```python
from rag_tools import create_manager

manager = await create_manager(
    embedder=MyCustomEmbedder(model_path="/path/to/model")
)
```

---

## Adding Custom Rerankers

To add a new reranker, extend the `BaseReranker` class:

```python
from typing import List
from rag_tools.retrieval.reranker import BaseReranker
from rag_tools.storage.models import RetrievalResult

class MyCustomReranker(BaseReranker):
    """Custom reranker implementation."""

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._model = None

    async def initialize(self) -> None:
        """Initialize the reranker."""
        if self._initialized:
            return

        # Load your model
        # self._model = load_my_reranker_model(self.model_path)

        self._initialized = True

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"custom-reranker-{self.model_path}"

    async def _score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Compute relevance scores for query-document pairs."""
        # Your scoring implementation
        scores = []
        for doc in documents:
            # score = self._model.score(query, doc)
            # scores.append(score)
            pass  # Replace with actual implementation

        return scores
```

### Using with Pipeline

```python
from rag_tools import create_manager

manager = await create_manager(
    reranker=MyCustomReranker(model_path="/path/to/model")
)
```

---

## Storage Backend Integration

### Adding Support for Another Vector Database

To add support for a new vector database (e.g., Pinecone, Weaviate):

1. Create a new client class:

```python
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client.models import FieldCondition, MatchValue

class PineconeClientWrapper:
    """Wrapper for Pinecone vector database."""

    def __init__(self, config):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Connect to Pinecone."""
        import pinecone
        pinecone.init(api_key=self.config.api_key)
        self._client = pinecone.Index(self.config.index_name)

    async def create_collection(self, name: str, vector_size: int) -> bool:
        """Create index (if not exists)."""
        if name not in pinecone.list_indexes():
            pinecone.create_index(name, dimension=vector_size)
        return True

    async def upsert_points(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """Insert vectors."""
        # Convert to Pinecone format
        vectors_data = []
        for i, (vec, payload) in enumerate(zip(vectors, payloads)):
            idx = ids[i] if ids else str(i)
            vectors_data.append((idx, vec, payload))

        self._client.upsert(vectors_data)
        return True

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[List[FieldCondition]] = None,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Convert filter conditions if needed
        pinecone_filter = self._convert_filter(filter_conditions)

        results = self._client.query(
            vector=query_vector,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=with_payload
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "payload": match.metadata
            }
            for match in results.matches
        ]

    def _convert_filter(
        self,
        conditions: Optional[List[FieldCondition]]
    ) -> Optional[Dict]:
        """Convert Qdrant filter format to Pinecone format."""
        # Implement conversion logic
        pass
```

2. Update `ToolRetriever` to accept the new client:

```python
# In rag_tools/retrieval/retriever.py
from rag_tools.storage.qdrant_client import QdrantClientWrapper
from your_package.pinecone_client import PineconeClientWrapper

class ToolRetriever:
    def __init__(
        self,
        vector_client: Union[QdrantClientWrapper, PineconeClientWrapper],
        # ... other params
    ):
        self.vector_client = vector_client
```

---

## Running Tests

### Install Test Dependencies

```bash
pip install rag-tools[dev]
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Modules

```bash
# Test storage
pytest tests/storage/ -v

# Test retrieval
pytest tests/retrieval/ -v

# Test ingestion
pytest tests/ingestion/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=rag_tools --cov-report=html
```

### Write Tests

```python
# tests/retrieval/test_custom_embedder.py

import pytest
import asyncio
from rag_tools.retrieval.embedder import BaseEmbedder

class TestMyCustomEmbedder:
    @pytest.fixture
    def embedder(self):
        return MyCustomEmbedder(model_path="/test/model")

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        await embedder.initialize()
        assert embedder._initialized is True
        assert embedder.embedding_dim > 0

    @pytest.mark.asyncio
    async def test_embed_single(self, embedder):
        await embedder.initialize()
        embedding = await embedder.embed("test text")
        assert embedding.shape == (embedder.embedding_dim,)

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        await embedder.initialize()
        embeddings = await embedder.embed(["text1", "text2", "text3"])
        assert embeddings.shape == (3, embedder.embedding_dim)
```

---

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/fiestaxxl/rag-tools.git
cd rag-tools
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

4. Install in development mode:
```bash
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public APIs
- Maximum line length: 100 characters

### Before Submitting

1. Run tests:
```bash
pytest tests/ -v
```

2. Format code:
```bash
black rag_tools/
isort rag_tools/
```

3. Lint:
```bash
ruff rag_tools/
mypy rag_tools/
```

### Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and commit:
```bash
git add .
git commit -m "Add: description of your changes"
```

3. Push and create PR:
```bash
git push origin feature/your-feature-name
```

### Issue Guidelines

When reporting issues, include:
- Python version
- RAG Tools version
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

---

## Performance Benchmarks

### Embedding Speed (CPU)

| Model | Batch Size | Time per 1000 texts |
|-------|------------|---------------------|
| all-MiniLM-L6-v2 | 32 | ~15s |
| all-mpnet-base-v2 | 16 | ~45s |

### Embedding Speed (GPU)

| Model | Batch Size | Time per 1000 texts |
|-------|------------|---------------------|
| all-MiniLM-L6-v2 | 128 | ~2s |
| all-mpnet-base-v2 | 64 | ~5s |

### Retrieval Latency

| Collection Size | Initial Retrieval | With Reranking |
|-----------------|-------------------|----------------|
| 100 tools | ~5ms | ~20ms |
| 1,000 tools | ~8ms | ~25ms |
| 10,000 tools | ~15ms | ~35ms |

### Memory Usage

| Component | Memory |
|-----------|--------|
| all-MiniLM-L6-v2 (CPU) | ~200MB |
| all-MiniLM-L6-v2 (GPU) | ~200MB + GPU RAM |
| Cross-encoder | ~500MB |
| Qdrant (embedded) | ~100MB |
| PostgreSQL | Depends on data |

---

## Release Process

### Version Numbering

RAG Tools follows Semantic Versioning (SemVer):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

### Release Checklist

1. Update version in `rag_tools/__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build documentation
5. Create GitHub release
6. Publish to PyPI
