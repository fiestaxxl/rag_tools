# RAG Tools

A RAG-based tool retrieval system for MCP (Model Context Protocol) servers. This module enables intelligent, semantic search over MCP tools for AI agents.

## Features

- **Semantic Search**: Find relevant tools using natural language queries
- **RAG Pipeline**: Vector-based retrieval with optional cross-encoder reranking
- **Multi-Server Support**: Manage tools from multiple MCP servers
- **Async Architecture**: Fully async implementation for high performance
- **Evaluation Framework**: Built-in metrics for measuring retrieval quality
- **Scalable**: Designed for large tool collections

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  MCP Server │────▶│   Parser     │────▶│  Text       │
│  (HTTP)     │     │              │     │  Builder    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            ▼
                    │   Embedder   │◀────┌─────────────┐
                    │ (Local/API)  │     │  Indexer    │
                    └──────┬───────┘     └──────┬──────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Qdrant     │     │  PostgreSQL │
                    │   (Vectors)  │     │  (Registry) │
                    └──────────────┘     └─────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```
To install the module in another project (e.g., from a Git repository), you can:

```bash
# Install from a local directory (editable mode)
pip install -e /path/to/rag_tools

# Install from a Git repository
pip install git+https://github.com/fiesta_xxl/rag_tools.git

# Install with optional local models
pip install "rag_tools[local]"@git+https://github.com/fiesta_xxl/rag_tools.git
```

## Quick Start

### 1. Setup Infrastructure

```bash
cd docker
./setup.sh
```

### 2. Initialize and Use

```python
import asyncio
from rag_tools import create_manager

async def main():
    # Create manager
    manager = await create_manager()

    # Add an MCP server
    server = await manager.add_server(
        url="http://localhost:8080/mcp",
        name="my-server",
        description="My MCP server"
    )

    # Retrieve relevant tools
    results = await manager.retrieve_tools("search for academic papers")

    print(f"Found {len(results)} relevant tools:")
    for r in results:
        print(f"  - {r.name} (score: {r.score:.3f})")

    await manager.close()

asyncio.run(main())
```

## Usage Examples

### Adding a Server

```python
server = await manager.add_server(
    url="http://10.32.11.22:7331/mcp",
    name="openalex",
    description="OpenAlex academic search API",
    headers={"Authorization": "Bearer ..."},
    sync_tools=True  # Automatically fetch tools
)
```

### Retrieving Tools

```python
# Simple retrieval
results = await manager.retrieve_tools("search papers by author")

# With configuration
result = await manager.retrieve(
    query="download PDF documents",
    top_k=10,
    rerank=True,
    min_score=0.3
)

# Access detailed results
for r in result.results:
    print(f"{r.name}: {r.description[:100]}...")
    print(f"  Score: {r.score:.3f}, Rank: {r.rank}")
```

### Evaluation

```python
from rag_tools import create_default_suite, RAGEvaluator

# Create evaluation suite
suite = create_default_suite()

# Or load from file
evaluator = RAGEvaluator(pipeline)
report = await evaluator.evaluate(suite)

# View results
evaluator.print_report(report)
```

## Configuration

Configuration is managed via `config/settings.py` or environment variables, check `example/env.example`:

```bash
# Environment variables
export QDRANT__URL=http://localhost:6333
export POSTGRES__HOST=localhost
export POSTGRES__PORT=5432
export POSTGRES__USER=rag_tools
export POSTGRES__PASSWORD=secret
```

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding.model_name` | all-MiniLM-L6-v2 | Embedding model |
| `reranker.model_name` | ms-marco-MiniLM-L-6-v2 | Reranker model |
| `rag.default_top_k` | 10 | Default retrieval limit |
| `rag.rerank_top_k` | 5 | Results after reranking |

## API Reference

### RAGToolsManager

Main class for tool management and retrieval.

#### Methods

- `add_server(url, name, ...)` - Add an MCP server
- `remove_server(server_id)` - Remove a server
- `sync_server(server_id)` - Sync tools from server
- `add_tool(server_id, name, ...)` - Add tool manually
- `remove_tool(tool_id)` - Remove a tool
- `retrieve(query, ...)` - Retrieve with full result
- `retrieve_tools(query, ...)` - Simplified retrieval
- `evaluate(suite)` - Run evaluation
- `get_stats()` - Get statistics

### Pipeline

The retrieval pipeline supports:

- **Vector search** via Qdrant
- **Cross-encoder reranking** (optional)
- **Filtering** by server, tags
- **Chunking** for long descriptions

## Metrics

The evaluation module provides:

- **Recall@K** - Fraction of relevant tools retrieved
- **Precision@K** - Fraction of retrieved tools that are relevant
- **MRR** - Mean Reciprocal Rank
- **MAP** - Mean Average Precision
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **Hit Rate** - Whether any relevant tool is in top K

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Project Structure

```
rag_tools/
├── config/          # Configuration
├── storage/         # Database clients
├── ingestion/       # MCP ingestion
├── retrieval/       # RAG retrieval
├── evaluation/      # Metrics
├── tools/           # Tool management
└── main.py          # Main module
```

## Docker

Services are defined in `docker/docker-compose.yml`:

```bash
cd docker
docker-compose up -d
```

## License

MIT
