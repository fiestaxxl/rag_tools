# RAG Tools Documentation Index

Welcome to the RAG Tools documentation. This page provides an overview of all available documentation resources.

## Documentation Overview

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This document - overview and quick navigation |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API reference for all classes, methods, and functions |
| [EXAMPLES.md](EXAMPLES.md) | Practical code examples for common use cases |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Guide for extending RAG Tools and contributing |

## Getting Started

### 1. Installation
See: [README.md - Installation](README.md#installation)

```bash
pip install rag-tools
```

### 2. Quick Start
See: [README.md - Quick Start](README.md#quick-start)

```python
import asyncio
from rag_tools import create_manager

async def main():
    manager = await create_manager()
    server = await manager.add_server(
        protocol='http',
        url="http://localhost:8080/mcp",
        name="my-server"
    )
    results = await manager.retrieve_tools("search papers")
    await manager.close()

asyncio.run(main())
```

### 3. Configuration
See: [README.md - Configuration](README.md#configuration)

## Documentation Sections

### Core Documentation
- **Introduction**: What is RAG Tools and key features
- **Installation**: Prerequisites and setup
- **Quick Start**: Minimal working example
- **Core Concepts**: MCP servers, tools, embedders, rerankers
- **Architecture**: System design and data flow

### API Reference
- **Main Module**: RAGToolsManager and factory functions
- **Storage Module**: PostgresClient, QdrantClientWrapper
- **Retrieval Module**: Embedders, retrievers, rerankers, pipeline
- **Evaluation Module**: Metrics, evaluator, evaluation suite
- **Tool Management**: ToolAdder, ToolRemover, MCPSyncer
- **Configuration**: Settings classes

### Examples
- **Basic Examples**: Minimal setup, complete workflow, multi-server
- **Advanced Usage**: Custom models, chunked retrieval, evaluation
- **Integration Patterns**: FastAPI service, periodic sync, batch processing
- **Performance Optimization**: GPU acceleration, caching
- **Common Use Cases**: AI assistant tool selection, tool discovery, multi-tenant

### Developer Resources
- **Architecture Overview**: System components and relationships
- **Custom Embedders**: Implementing new embedder types
- **Custom Rerankers**: Implementing new reranker types
- **Storage Integration**: Adding support for new databases
- **Testing**: Running and writing tests
- **Contributing**: Code style and PR process

## Common Tasks

| Task | Documentation |
|------|---------------|
| Add MCP server | [README.md - Adding a Server](README.md#adding-a-server) |
| Retrieve tools | [README.md - Retrieving Tools](README.md#retrieving-tools) |
| Configure embeddings | [README.md - Configuration](README.md#configuration) |
| Run evaluation | [README.md - Evaluation](README.md#evaluation) |
| Use custom embedder | [EXAMPLES.md - Custom Embedder](EXAMPLES.md#example-4-custom-embedder-and-reranker) |
| Integrate with FastAPI | [EXAMPLES.md - FastAPI Service](EXAMPLES.md#example-7-as-a-fastapi-service) |
| Add custom reranker | [DEVELOPER_GUIDE.md - Custom Rerankers](DEVELOPER_GUIDE.md#adding-custom-rerankers) |
| Run tests | [DEVELOPER_GUIDE.md - Running Tests](DEVELOPER_GUIDE.md#running-tests) |

## Search Tips

### Finding Information by Topic

**Retrieval:**
- Pipeline: [README.md - Retrieval Pipeline](README.md#retrieval-pipeline)
- Configuration: [README.md - Pipeline Config](README.md#pipeline-config)
- Examples: [EXAMPLES.md - Basic Retrieval](EXAMPLES.md#example-1-basic-tool-retrieval)

**Embedders:**
- Base class: [API_REFERENCE.md - BaseEmbedder](API_REFERENCE.md#baseembedder)
- Local embedder: [API_REFERENCE.md - LocalEmbedder](API_REFERENCE.md#localembedder)
- Custom embedder: [DEVELOPER_GUIDE.md - Custom Embedders](DEVELOPER_GUIDE.md#adding-custom-embedders)

**Reranking:**
- Base class: [API_REFERENCE.md - BaseReranker](API_REFERENCE.md#basereranker)
- Cross-encoder: [API_REFERENCE.md - CrossEncoderReranker](API_REFERENCE.md#crossencoderreranker)
- BM25: [API_REFERENCE.md - BM25Reranker](API_REFERENCE.md#bm25reranker)
- Hybrid: [API_REFERENCE.md - HybridReranker](API_REFERENCE.md#hybridreranker)

**Evaluation:**
- Metrics: [README.md - Metrics](README.md#metrics)
- Calculator: [API_REFERENCE.md - MetricsCalculator](API_REFERENCE.md#metricscalculator)
- Evaluator: [API_REFERENCE.md - RAGEvaluator](API_REFERENCE.md#ragevaluator)
- Examples: [EXAMPLES.md - Evaluation](EXAMPLES.md#example-6-evaluation-suite)

**Storage:**
- PostgreSQL: [API_REFERENCE.md - PostgresClient](API_REFERENCE.md#postgresclient)
- Qdrant: [API_REFERENCE.md - QdrantClientWrapper](API_REFERENCE.md#qdrantclientwrapper)
- Models: [API_REFERENCE.md - Data Models](API_REFERENCE.md#data-models)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection errors | [README.md - Troubleshooting](README.md#troubleshooting) |
| Embedding issues | [README.md - Embedding Issues](README.md#embedding-issues) |
| Retrieval issues | [README.md - Retrieval Issues](README.md#retrieval-issues) |
| MCP sync issues | [README.md - MCP Sync Issues](README.md#mcp-sync-issues) |

## External Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Sentence-Transformers](https://www.sbert.net/)
- [MCP SDK](https://github.com/modelcontextprotocol/python-sdk)

## Version

Current version: 1.0.0

## License

MIT License
