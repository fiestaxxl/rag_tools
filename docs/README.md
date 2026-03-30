# RAG Tools Documentation

Welcome to the RAG Tools documentation! This library provides a comprehensive RAG (Retrieval-Augmented Generation) based tool retrieval system for MCP (Model Context Protocol) servers.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Introduction

### What is RAG Tools?

RAG Tools is a Python library designed for semantic search and retrieval of MCP (Model Context Protocol) tools. It enables intelligent tool discovery by indexing tool descriptions and capabilities, then retrieving the most relevant tools based on natural language queries.

### Key Features

- **Semantic Tool Retrieval**: Use natural language to find relevant tools from your MCP servers
- **Multi-Strategy Reranking**: Combine embedding-based and keyword-based retrieval for optimal results
- **MCP Server Integration**: Seamlessly sync tools from MCP servers with automatic indexing
- **Comprehensive Evaluation**: Built-in metrics for measuring retrieval quality (Recall, Precision, NDCG, MRR)
- **Flexible Embedding Support**: Use local sentence-transformers or remote API embeddings
- **Chunked Indexing**: Handle long tool descriptions with intelligent text chunking
- **Async Architecture**: Fully async design for high-performance applications

### Architecture Overview

RAG Tools is built on a modular architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Tools                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Ingestion │  │  Retrieval  │  │      Evaluation        │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────┤  │
│  │  - Parser   │  │  - Embedder │  │  - Metrics Calculator   │  │
│  │  - Text     │  │  - Reranker │  │  - RAG Evaluator       │  │
│  │    Builder  │  │  - Retriever│  │  - Evaluation Suite    │  │
│  │  - Indexer  │  │  - Pipeline │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      Storage Layer                          │ │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐ │ │
│  │  │  PostgreSQL Client  │  │   Qdrant Vector Database    │ │ │
│  │  │  - Servers          │  │   - Tool Embeddings         │ │ │
│  │  │  - Tools            │  │   - Tool Chunks             │ │ │
│  │  │  - Credentials      │  │                             │ │ │
│  │  │  - Chunks           │  │                             │ │ │
│  │  └──────────────────────┘  └──────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 14+ (for structured data storage)
- Qdrant (for vector storage)

### Standard Installation

Install RAG Tools using pip:

```bash
pip install rag-tools
```

### Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/rag-tools.git
cd rag-tools
pip install -e ".[dev]"
```

### Dependencies

RAG Tools requires the following dependencies:

**Core Dependencies:**
- `pydantic` >= 2.0
- `pydantic-settings` >= 2.0
- `sqlalchemy` >= 2.0 (with async support)
- `asyncpg` >= 0.28 (PostgreSQL async driver)
- `qdrant-client` >= 1.7

**ML Dependencies:**
- `sentence-transformers` >= 2.2 (for local embeddings)
- `langchain-text-splitters` >= 0.0.16 (for text chunking)
- `rank-bm25` >= 0.2 (for BM25 reranking)

**MCP Integration:**
- `mcp` >= 1.0

**Optional:**
- `torch` (for GPU acceleration)
- `fastapi` (for API deployment)

### Docker Setup

Use Docker Compose to set up PostgreSQL and Qdrant:

```bash
cd rag_tools/docker
docker-compose up -d
```

---

## Quick Start

### Basic Usage

Here's a minimal example to get you started:

```python
import asyncio
from rag_tools import create_manager

async def main():
    # Create and initialize the manager
    manager = await create_manager()

    # Add an MCP server
    server = await manager.add_server(
        url="http://localhost:8080/mcp",
        name="my-server",
        description="My MCP server",
        sync_tools=True
    )

    # Retrieve relevant tools
    results = await manager.retrieve_tools(
        query="search for academic papers",
        top_k=5
    )

    print(f"Found {len(results)} relevant tools:")
    for r in results:
        print(f"  - {r.name} (score: {r.score:.3f})")

    # Cleanup
    await manager.close()

asyncio.run(main())
```

### Complete Workflow

```python
import asyncio
from rag_tools import create_manager

async def complete_workflow():
    # Initialize manager
    manager = await create_manager()

    # 1. Add a server
    server = await manager.add_server(
        protocol='http',
        url="http://10.32.11.22:7332/mcp",
        name="example-server",
        description="Example MCP server"
    )

    # 2. Add tools manually (optional)
    tool = await manager.add_tool(
        server_id=server.server_id,
        name="search_papers",
        description="Search for academic papers",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        },
        tags=["search", "academic"]
    )

    # 3. Retrieve tools
    results = await manager.retrieve_tools(
        query="Get docking score for disease research",
        top_k=10,
        rerank=True,
        rerank_top_k=5
    )

    # 4. Get detailed results
    detailed = await manager.retrieve(
        query="search for papers",
        top_k=10,
        rerank=True
    )

    print(f"Retrieval time: {detailed.retrieval_time_ms:.2f}ms")
    print(f"Reranking time: {detailed.rerank_time_ms:.2f}ms")

    # 5. Sync server for updates
    sync_result = await manager.sync_server(server.server_id)
    print(f"Sync: {sync_result.success}")

    # 6. Get statistics
    stats = await manager.get_stats()
    print(f"Total tools: {stats['postgres']['tools']}")

    # Cleanup
    await manager.remove_server(server.server_id)
    await manager.close()

asyncio.run(complete_workflow())
```

---

## Core Concepts

### MCP Server

An MCP (Model Context Protocol) server is a service that exposes a set of tools. RAG Tools can connect to these servers and automatically discover, index, and retrieve their tools.

```python
from rag_tools import MCPServer

server = MCPServer(
    server_id="unique-id",
    name="my-server",
    protocol='http',
    url="http://localhost:8080/mcp",
    description="A server providing search tools"
)
```
or
```python
from rag_tools import MCPServer

server = MCPServer(
    server_id="unique-id",
    name="my-server",
    protocol='stdio',
    command="npx",
    args=['arg1', 'arg2'],
    env={'env1': 'val1'},
    description="A server providing search tools"
)
```

### Tool Model

Tools are the atomic units of functionality exposed by MCP servers:

```python
from rag_tools import MCPTool

tool = MCPTool(
    tool_id="server-id:tool-name",
    server_id="server-id",
    name="search_papers",
    description="Search for academic papers",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        }
    },
    tags=["search", "academic"]
)
```

### Embedders

Embedders convert text into dense vector representations for semantic similarity search.

**Local Embedder (Sentence-Transformers):**
```python
from rag_tools.retrieval import LocalEmbedder

embedder = LocalEmbedder(config)
await embedder.initialize()
embeddings = await embedder.embed(["text to embed"])
```

**API Embedder:**
```python
from rag_tools.retrieval import APIEmbedder

embedder = APIEmbedder(config)
```

**Cached Embedder:**
```python
from rag_tools.retrieval import CachedEmbedder

cached = CachedEmbedder(base_embedder, cache_size=10000)
```

### Retrievers

Retrievers search the vector database for relevant tools:

```python
from rag_tools.retrieval import ToolRetriever

retriever = ToolRetriever(qdrant_client, embedder)
await retriever.initialize()

results = await retriever.retrieve(
    query="search papers",
    config=RetrievalConfig(top_k=10)
)
```

### Rerankers

Rerankers refine initial retrieval results using more sophisticated models.

**Cross-Encoder Reranker:**
```python
from rag_tools.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(config)
await reranker.initialize()
reranked = await reranker.rerank(query, results, top_k=5)
```

**BM25 Reranker:**
```python
from rag_tools.retrieval import BM25Reranker

reranker = BM25Reranker(config)
```

**Hybrid Reranker:**
```python
from rag_tools.retrieval import HybridReranker

reranker = HybridReranker([embed_reranker, bm25_reranker], config)
```

### Retrieval Pipeline

The complete retrieval pipeline combines embedding, retrieval, and reranking:

```python
from rag_tools.retrieval import ToolRetrievalPipeline, PipelineConfig

pipeline = ToolRetrievalPipeline(embedder, reranker, retriever)
await pipeline.initialize()

result = await pipeline.retrieve(
    query="search for academic papers",
    config=PipelineConfig(top_k=10, rerank=True)
)
```

### Evaluation Suite

Evaluation suites define test cases for measuring retrieval quality:

```python
from rag_tools.evaluation import EvaluationSuite, GroundTruthEntry

suite = EvaluationSuite(
    name="academic-search",
    description="Test suite for academic search tools",
    queries=[
        GroundTruthEntry(
            query="search for papers",
            relevant_tools=["openalex:search_papers"]
        ),
        GroundTruthEntry(
            query="download PDFs",
            relevant_tools=["openalex:download_papers"]
        )
    ]
)
```

---

## API Reference

### RAGToolsManager

The main manager class providing a unified interface for all RAG Tools operations.

#### Constructor

```python
RAGToolsManager(
    config: Optional[Settings] = None,
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None
)
```

#### Methods

**async initialize()**
Initialize all components (databases, embedder, reranker, etc.).

**async close()**
Close all connections and cleanup resources.

**async session()**
Context manager for automatic initialization and cleanup.

```python
async with manager.session():
    results = await manager.retrieve_tools("search papers")
```

**async add_server()**
Add a new MCP server.

```python
server = await manager.add_server(
    protocol='http',
    url="http://localhost:8080/mcp",
    name="my-server",
    description="Server description",
    headers={"Authorization": "Bearer token"},
    sync_tools=True
)
```
or
```python
server = await manager.add_server(
    protocol='stdio',
    command="npx",
    args=['arg1', 'arg2'],
    env={'env1': 'val1'},
    name="my-server",
    description="Server description",
    headers={"Authorization": "Bearer token"},
    sync_tools=True
)
```

**async remove_server(server_id: str)**
Remove a server and all its tools.

**async sync_server(server_id: str) -> SyncResult**
Sync tools from a server.

**async sync_all_servers() -> List[SyncResult]**
Sync all registered servers.

**async add_tool()**
Add a single tool manually.

```python
tool = await manager.add_tool(
    server_id=server.server_id,
    name="search_papers",
    description="Search for academic papers",
    input_schema={"type": "object", "properties": {...}},
    tags=["search", "academic"]
)
```

**async remove_tool(tool_id: str) -> bool**
Remove a tool.

**async retrieve() -> PipelineResult**
Retrieve tools with full pipeline information.

```python
result = await manager.retrieve(
    query="search for papers",
    top_k=10,
    rerank=True,
    rerank_top_k=5,
    min_score=0.3,
    use_chunks=False,
    server_filter=["server-id-1"],
    tags_filter=["academic"]
)
```

**async retrieve_tools() -> List[RetrievalResult]**
Simplified retrieval returning just results.

**create_evaluator() -> RAGEvaluator**
Create an evaluator for this manager's pipeline.

**async evaluate(suite: EvaluationSuite) -> EvaluationReport**
Run evaluation with a test suite.

**async get_stats() -> Dict**
Get registry statistics.

#### Properties

- `postgres`: PostgreSQL client instance
- `qdrant`: Qdrant client instance
- `pipeline`: Retrieval pipeline instance

---

### Storage Module

#### PostgresClient

Async PostgreSQL client for structured data storage.

```python
from rag_tools.storage import PostgresClient

client = PostgresClient(config)
await client.initialize()

# Server operations
await client.add_server(server)
await client.get_server(server_id)
await client.list_servers(status="active")
await client.delete_server(server_id)

# Tool operations
await client.add_tool(tool)
await client.get_tool(tool_id)
await client.get_tools_by_server(server_id)
await client.delete_tool(tool_id)
await client.bulk_add_tools(tools)

# Chunk operations
await client.add_chunks(chunks)
await client.get_chunks_by_tool(tool_id)

# Credential operations
await client.add_credential(credential)
await client.get_credential(server_id)

# Statistics
stats = await client.get_stats()
```

#### QdrantClientWrapper

Async wrapper for Qdrant vector database.

```python
from rag_tools.storage import QdrantClientWrapper

client = QdrantClientWrapper(config)
await client.connect()

# Collection management
await client.create_collection("tools", vector_size=384)
await client.delete_collection("tools")
await client.collection_exists("tools")
info = await client.get_collection_info("tools")

# Point operations
await client.upsert_points("tools", vectors, payloads, ids)
await client.delete_points("tools", point_ids)
await client.delete_by_filter("tools", filter_conditions)

# Search operations
results = await client.search("tools", query_vector, top_k=10)
batch_results = await client.search_batch("tools", query_vectors, top_k=10)

# Retrieval
points = await client.retrieve("tools", point_ids)

# Scroll (pagination)
points, offset = await client.scroll("tools", limit=100)

# Indexing
await client.create_payload_index("tools", "tool_id")

# Health check
health = await client.health_check()
```

---

### Retrieval Module

#### BaseEmbedder

Abstract base class for embedders.

```python
class BaseEmbedder:
    async def initialize() -> None
    @property def embedding_dim() -> int
    @property def model_name() -> str
    async def embed(texts: Union[str, List[str]]) -> np.ndarray
    async def embed_query(query: str) -> np.ndarray
    async def compute_similarity(embeddings1, embeddings2) -> np.ndarray
```

#### LocalEmbedder

Sentence-transformers based embedder.

```python
from rag_tools.retrieval import LocalEmbedder

embedder = LocalEmbedder(EmbeddingSettings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    batch_size=32,
    max_seq_length=512
))
await embedder.initialize()
```

#### ToolRetriever

Retrieves tools from Qdrant.

```python
from rag_tools.retrieval import ToolRetriever, RetrievalConfig

retriever = ToolRetriever(qdrant_client, embedder)
await retriever.initialize()

results = await retriever.retrieve(
    query="search papers",
    config=RetrievalConfig(
        top_k=10,
        min_score=0.0,
        use_chunks=False,
        server_filter=["server-id"],
        tags_filter=["academic"]
    )
)
```

#### BaseReranker

Abstract base class for rerankers.

```python
class BaseReranker:
    async def initialize() -> None
    @property def model_name() -> str
    async def rerank(query, results, top_k) -> List[RetrievalResult]
    async def rerank_with_scores(query, documents, top_k) -> List[Tuple[int, float]]
```

#### CrossEncoderReranker

Cross-encoder based reranker using sentence-transformers.

```python
from rag_tools.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(CrossEncoderRerankerSettings(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
    batch_size=32,
    max_length=512
))
await reranker.initialize()
```

#### BM25Reranker

BM25 keyword-based reranker.

```python
from rag_tools.retrieval import BM25Reranker

reranker = BM25Reranker(BM25RerankerSettings(
    k1=1.5,
    b=0.75
))
```

#### HybridReranker

Combines multiple rerankers with weighted scoring.

```python
from rag_tools.retrieval import HybridReranker

reranker = HybridReranker(
    rerankers=[cross_encoder, bm25],
    config=HybridRerankerSettings(weights=[0.7, 0.3])
)
```

#### ToolRetrievalPipeline

Complete RAG retrieval pipeline.

```python
from rag_tools.retrieval import ToolRetrievalPipeline, PipelineConfig

pipeline = ToolRetrievalPipeline(embedder, reranker, retriever)
await pipeline.initialize()

result = await pipeline.retrieve(
    query="search for papers",
    config=PipelineConfig(
        top_k=10,
        rerank=True,
        rerank_top_k=5,
        min_relevance_score=0.3,
        use_chunks=False
    )
)

# result.query - the original query
# result.results - List[RetrievalResult]
# result.retrieval_time_ms
# result.rerank_time_ms
# result.total_time_ms
```

#### PipelineConfig

Configuration for retrieval pipeline.

```python
@dataclass
class PipelineConfig:
    top_k: int = 10                    # Number of results to retrieve
    rerank: bool = True                # Whether to use reranking
    rerank_top_k: int = 5              # Number of results after reranking
    min_relevance_score: float = 0.3  # Minimum relevance threshold
    use_chunks: bool = False          # Use chunked collection
    server_filter: Optional[List[str]] = None  # Filter by server IDs
    tags_filter: Optional[List[str]] = None    # Filter by tags
```

---

### Evaluation Module

#### RAGEvaluator

Evaluates RAG pipeline performance.

```python
from rag_tools.evaluation import RAGEvaluator, EvaluationSuite

evaluator = RAGEvaluator(pipeline, k_values=[1, 3, 5, 10])

# Evaluate a suite
report = await evaluator.evaluate_suite(suite, config)

# Evaluate custom query
evaluation = await evaluator.evaluate_custom(
    query="search papers",
    relevant_tools=["tool-id-1"],
    config=pipeline_config
)

# Print and export
evaluator.print_report(report)
evaluator.export_report(report, Path("report.json"))
```

#### MetricsCalculator

Calculates various retrieval metrics.

```python
from rag_tools.evaluation import MetricsCalculator

metrics = MetricsCalculator.calculate_all_metrics(
    results,
    relevant_ids=["tool-1", "tool-2"],
    k_values=[1, 3, 5, 10]
)

# Available metrics:
# - recall@k
# - precision@k
# - ndcg@k
# - hit_rate@k
# - mrr (Mean Reciprocal Rank)
# - map (Mean Average Precision)
# - diversity (unique server coverage)
```

#### EvaluationReport

Results from evaluation.

```python
@dataclass
class EvaluationReport:
    suite_name: str                                    # Name of test suite
    total_queries: int                                # Number of queries
    aggregated_metrics: Dict[str, Dict[str, float]]  # Mean, std, min, max for each metric
    per_query_results: List[QueryEvaluation]         # Individual query results
    metadata: Dict[str, Any]                          # Additional metadata
```

---

### Tool Management

#### ToolAdder

Add tools and servers to the registry.

```python
from rag_tools.tools import ToolAdder

adder = ToolAdder(postgres, qdrant, indexer)

# Add server
server = await adder.add_server(
    protocol='http',
    url="http://localhost:8080/mcp",
    name="my-server",
    description="Server description",
    headers={},
    timeout=30,
    sync_tools=True
)
```
or
```python
from rag_tools.tools import ToolAdder

adder = ToolAdder(postgres, qdrant, indexer)

# Add server
server = await adder.add_server(
    protocol='stdio',
    command="npx",
    args=['arg1', 'arg2'],
    env={'env1': 'val1'},
    name="my-server",
    description="Server description",
    headers={},
    timeout=30,
    sync_tools=True
)
```

# Add single tool
tool = await adder.add_tool(
    server_id=server_id,
    name="search",
    description="Search tool",
    input_schema={},
    tags=[]
)

# Add multiple tools
await adder.add_tools_batch(tools, batch_size=32)

# Sync from MCP server
await adder.sync_server_tools(server)
```

#### ToolRemover

Remove tools and servers from the registry.

```python
from rag_tools.tools import ToolRemover

remover = ToolRemover(postgres, qdrant, indexer)

# Remove single tool
await remover.remove_tool(tool_id)

# Remove multiple tools
stats = await remover.remove_tools_batch([id1, id2, id3])

# Remove server (and all its tools)
stats = await remover.remove_server(server_id)

# Remove by filter
stats = await remover.remove_by_filter(
    server_ids=["id1", "id2"],
    tags=["deprecated"],
    status=ToolStatus.INACTIVE
)

# Remove inactive tools
stats = await remover.remove_inactive_tools(older_than_hours=24)

# Clear all data
stats = await remover.clear_all()
```

#### MCPSyncer

Sync tools from MCP servers.

```python
from rag_tools.tools import MCPSyncer

syncer = MCPSyncer(postgres, qdrant, indexer)

# Sync single server
result = await syncer.sync_server(server_id)

# Sync all servers
results = await syncer.sync_all_servers()

# Sync with retry
result = await syncer.sync_with_retry(
    server_id,
    max_retries=3,
    retry_delay=1.0
)
```

```python
# Create server and sync
result = await syncer.create_server_and_sync(
    protocol='http',
    url="http://localhost:8080/mcp",
    name="new-server",
    description="New server"
)
```
or
```python
# Create server and sync
result = await syncer.create_server_and_sync(
    protocol='stdio',
    command="npx",
    args=['arg1', 'arg2'],
    env={'env1': 'val1'},
    name="new-server",
    description="New server"
)
```

# Schedule periodic sync
task = await syncer.schedule_periodic_sync(interval_hours=24)
# Later: task.cancel()
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Database Configuration
QDRANT__URL=http://localhost:6333
QDRANT__API_KEY=

POSTGRES__HOST=localhost
POSTGRES__PORT=5432
POSTGRES__USER=rag_tools
POSTGRES__PASSWORD=rag_tools_password
POSTGRES__DATABASE=rag_tools

# Embedding Configuration
EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING__DEVICE=cpu
EMBEDDING__BATCH_SIZE=32
EMBEDDING__MAX_SEQ_LENGTH=512

# API Embedding (optional)
API_EMBEDDING__URL=http://localhost:5000/embed
API_EMBEDDING__API_KEY=
API_EMBEDDING__NORMALIZE_EMBEDDINGS=false

# Reranker Configuration
RERANKER__MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER__DEVICE=cpu
RERANKER__BATCH_SIZE=32
RERANKER__TOP_K=20
RERANKER__MAX_LENGTH=512

# API Reranker (optional)
API_RERANKER__URL=http://localhost:5001/rerank
API_RERANKER__API_KEY=

# BM25 Reranker
BM_RERANKER__K1=1.5
BM_RERANKER__B=0.75

# Hybrid Reranker
HYBRID_RERANKER__WEIGHTS=[0.7, 0.3]

# RAG Configuration
RAG__DEFAULT_TOP_K=10
RAG__RERANK_TOP_K=5
RAG__MIN_RELEVANCE_SCORE=0.3
RAG__CHUNK_SIZE=512
RAG__CHUNK_OVERLAP=50

# Storage
WORK_DIR=/tmp/rag_tools
TOOLS_COLLECTION=mcp_tools
TOOLS_CHUNKS_COLLECTION=mcp_tools_chunks
```

### Configuration Classes

For programmatic configuration:

```python
from rag_tools.config import Settings, QdrantSettings, PostgresSettings

settings = Settings(
    qdrant=QdrantSettings(
        url="http://localhost:6333",
        api_key=None,
        timeout=30
    ),
    postgres=PostgresSettings(
        host="localhost",
        port=5432,
        user="rag_tools",
        password="password",
        database="rag_tools"
    ),
    embedding=EmbeddingSettings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32
    ),
    rag=RAGSettings(
        default_top_k=10,
        rerank_top_k=5,
        min_relevance_score=0.3,
        chunk_size=512,
        chunk_overlap=50
    )
)
```

### Collection Names

Customize Qdrant collection names:

```python
settings.tools_collection = "custom_tools"
settings.tools_chunks_collection = "custom_tools_chunks"
```

---

## Examples

### Example 1: Basic Tool Retrieval

```python
import asyncio
from rag_tools import create_manager

async def basic_retrieval():
    manager = await create_manager()

    # Add a server
    server = await manager.add_server(
        protocol='http',
        url="http://10.32.11.22:7332/mcp",
        name="chemistry-tools",
        sync_tools=True
    )

    # Retrieve tools
    results = await manager.retrieve_tools(
        query="Get docking score for Alzheimer disease",
        top_k=10,
        rerank=True,
        rerank_top_k=5
    )

    for r in results:
        print(f"{r.name}: {r.description[:50]}... (score: {r.score:.3f})")

    await manager.close()

asyncio.run(basic_retrieval())
```

### Example 2: MCP Server Sync

```python
import asyncio
from rag_tools import create_manager

async def sync_example():
    manager = await create_manager()

    # Add and sync server
    server = await manager.add_server(
        protocol='http',
        url="http://openalex.example.com/mcp",
        name="openalex-api",
        description="OpenAlex academic search API",
        sync_tools=True
    )

    print(f"Server status: {server.status}")
    print(f"Last synced: {server.last_synced}")

    # Manually sync again
    sync_result = await manager.sync_server(server.server_id)
    print(f"Sync: {sync_result.success}")
    print(f"Tools added: {sync_result.tools_added}")
    print(f"Tools updated: {sync_result.tools_updated}")

    await manager.close()

asyncio.run(sync_example())
```

### Example 3: Evaluation

```python
import asyncio
from rag_tools import create_manager, EvaluationSuite, GroundTruthEntry

async def evaluation_example():
    manager = await create_manager()

    # Create test suite
    suite = EvaluationSuite(
        name="academic-search-eval",
        description="Evaluation for academic search tools",
        queries=[
            GroundTruthEntry(
                query="how to search for papers",
                relevant_tools=["openalex:search_papers"]
            ),
            GroundTruthEntry(
                query="download PDF documents",
                relevant_tools=["openalex:download_papers"]
            ),
            GroundTruthEntry(
                query="find author details",
                relevant_tools=["openalex:get_author_info"]
            )
        ]
    )

    # Run evaluation
    evaluator = manager.create_evaluator()
    report = await evaluator.evaluate_suite(suite)

    # Print results
    evaluator.print_report(report)

    # Export report
    evaluator.export_report(report, "/tmp/evaluation_report.json")

    await manager.close()

asyncio.run(evaluation_example())
```

### Example 4: Custom Embedder and Reranker

```python
import asyncio
from rag_tools import create_manager
from rag_tools.retrieval import (
    APIEmbedder,
    APIReranker,
    BM25Reranker,
    HybridReranker
)
from rag_tools.config import Settings

async def custom_config_example():
    settings = Settings()

    # Configure API embedder
    embedder = APIEmbedder(settings.api_embedding)

    # Configure hybrid reranker
    api_reranker = APIReranker(settings.api_reranker)
    bm25_reranker = BM25Reranker(settings.bm_reranker)
    hybrid_reranker = HybridReranker(
        [api_reranker, bm25_reranker],
        settings.hybrid_reranker
    )

    # Create manager with custom components
    manager = await create_manager(
        settings,
        embedder=embedder,
        reranker=hybrid_reranker
    )

    results = await manager.retrieve_tools(
        query="search for papers",
        rerank=True
    )

    await manager.close()

asyncio.run(custom_config_example())
```

### Example 5: Filtering and Chunked Retrieval

```python
import asyncio
from rag_tools import create_manager

async def filtering_example():
    manager = await create_manager()

    # Filter by server
    results = await manager.retrieve_tools(
        query="search",
        server_filter=["server-id-1"]
    )

    # Filter by tags
    results = await manager.retrieve_tools(
        query="search",
        tags_filter=["academic", "research"]
    )

    # Use chunked collection for long descriptions
    results = await manager.retrieve(
        query="search for papers",
        use_chunks=True,
        top_k=20
    )

    await manager.close()

asyncio.run(filtering_example())
```

---

## Evaluation

### Creating Evaluation Suites

```python
from rag_tools.evaluation import EvaluationSuite, GroundTruthEntry

suite = EvaluationSuite(
    name="my-evaluation",
    description="Description of evaluation",
    queries=[
        GroundTruthEntry(
            query="natural language query",
            relevant_tools=["tool-id-1", "tool-id-2"]
        ),
        # ... more queries
    ]
)
```

### Loading Suites from JSON

```python
from rag_tools.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)
suite = evaluator.load_suite_from_file(Path("test_suite.json"))

# JSON format:
# {
#     "name": "my-suite",
#     "description": "Test suite",
#     "queries": [
#         {"query": "search papers", "relevant_tools": ["tool-id"]}
#     ]
# }
```

### Available Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Recall@K | Fraction of relevant items retrieved in top K | [0, 1] |
| Precision@K | Fraction of retrieved items that are relevant | [0, 1] |
| NDCG@K | Normalized Discounted Cumulative Gain | [0, 1] |
| Hit Rate@K | Whether any relevant item is in top K | [0, 1] |
| MRR | Mean Reciprocal Rank | [0, 1] |
| MAP | Mean Average Precision | [0, 1] |
| Diversity | Unique server coverage in top K | [0, 1] |

### Interpreting Results

```python
# Access aggregated metrics
for metric_name, values in report.aggregated_metrics.items():
    print(f"{metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"  Range: [{values['min']:.4f}, {values['max']:.4f}]")

# Access per-query results
for query_eval in report.per_query_results:
    print(f"Query: {query_eval.query}")
    print(f"  Predicted: {query_eval.predicted_tools}")
    print(f"  Relevant: {query_eval.relevant_tools}")
    print(f"  Recall@5: {query_eval.metrics['recall@5']:.4f}")
```

---

## Architecture

### System Overview

```
                                    ┌─────────────────────┐
                                    │   Application       │
                                    │   (RAGToolsManager)  │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
        ┌───────────▼───────────┐    ┌────────▼────────┐    ┌────────────▼─────────┐
        │     Ingestion          │    │    Retrieval     │    │     Evaluation        │
        ├───────────────────────┤    ├─────────────────┤    ├──────────────────────┤
        │ • Parser              │    │ • Embedder       │    │ • MetricsCalculator   │
        │   (MCP tools)          │    │   (Local/API)    │    │ • RAGEvaluator        │
        │ • TextBuilder         │    │ • Reranker       │    │ • EvaluationSuite     │
        │   (Search text)        │    │   (Cross-Encoder │    │                       │
        │ • Indexer             │    │    / BM25)       │    │                       │
        │   (Postgres + Qdrant) │    │ • Retriever      │    │                       │
        └───────────────────────┘    │   (Qdrant)       │    └──────────────────────┘
                                      └────────┬─────────┘
                                               │
            ┌───────────────────────────────────┼───────────────────────────────────┐
            │                                   │                                   │
┌───────────▼───────────┐            ┌──────────▼──────────┐
│     PostgreSQL        │            │      Qdrant        │
├───────────────────────┤            ├────────────────────┤
│ • servers             │            │ • mcp_tools        │
│   - server_id (PK)    │            │   (embeddings)     │
│   - name              │            │                    │
│   - url               │            │ • mcp_tools_chunks │
│   - status            │            │   (chunked text)   │
│   - last_synced       │            │                    │
│                       │            │                    │
│ • tools               │            │                    │
│   - tool_id (PK)      │            │                    │
│   - server_id (FK)    │            │                    │
│   - name, description │            │                    │
│   - input_schema      │            │                    │
│   - tags              │            │                    │
│                       │            │                    │
│ • chunks              │            │                    │
│ • credentials         │            │                    │
└───────────────────────┘            └────────────────────┘
```

### Data Flow

**Ingestion Flow:**
```
MCP Server → Parser → MCPTool → TextBuilder → Indexer
                                    ↓
                        ┌───────────┴───────────┐
                        ↓                       ↓
                   PostgreSQL              Qdrant
                   (structured)            (vectors)
```

**Retrieval Flow:**
```
Query → Embedder → Query Vector → Retriever → Initial Results
                                              ↓
                                       Reranker
                                              ↓
                                      Final Ranked Results
```

### Storage Strategy

- **PostgreSQL**: Stores structured tool metadata, server configurations, and credentials
- **Qdrant**: Stores vector embeddings for semantic similarity search
- **Dual Indexing**: Tools are indexed both as complete units and as overlapping text chunks for better retrieval of long descriptions

---

## Troubleshooting

### Common Issues

#### Connection Errors

**PostgreSQL connection failed:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify credentials
psql -h localhost -p 5432 -U rag_tools -d rag_tools
```

**Qdrant connection failed:**
```bash
# Check Qdrant is running
curl http://localhost:6333/healthz
```

#### Embedding Issues

**CUDA out of memory:**
```python
# Use CPU instead
settings.embedding.device = "cpu"
settings.reranker.device = "cpu"
```

**Model not found:**
```bash
# Install sentence-transformers
pip install sentence-transformers

# Or download model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### Retrieval Issues

**No results returned:**
- Check if tools are indexed: `manager.get_stats()`
- Lower `min_score` threshold
- Check server filter is correct

**Poor retrieval quality:**
- Try using reranking (`rerank=True`)
- Experiment with hybrid reranking
- Increase `top_k` for initial retrieval

#### MCP Sync Issues

**Server connection timeout:**
```python
server = await manager.add_server(
    protocol='http',
    url="http://server:port/mcp",
    timeout=60  # Increase timeout
)
```

**Tool parsing errors:**
- Verify MCP server returns valid JSON-RPC responses
- Check server logs for errors
- Try syncing with `sync_tools=False` and add tools manually

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger("rag_tools").setLevel(logging.DEBUG)
```

### Performance Optimization

**For large-scale deployments:**

1. **Use GPU for embeddings:**
```python
settings.embedding.device = "cuda"
```

2. **Batch processing:**
```python
await indexer.index_tools_batch(tools, batch_size=64)
```

3. **Connection pooling:**
```python
settings.postgres.max_connections = 20
```

4. **Cached embeddings:**
```python
cached_embedder = CachedEmbedder(embedder, cache_size=10000)
```

---

## License

RAG Tools is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines for more information.

## Support

For issues and questions, please open an issue on GitHub.
