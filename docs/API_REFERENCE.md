# API Reference

This page provides detailed documentation for all public APIs in RAG Tools.

## Table of Contents

- [Main Module](#main-module)
- [Storage Module](#storage-module)
- [Retrieval Module](#retrieval-module)
- [Evaluation Module](#evaluation-module)
- [Tool Management Module](#tool-management-module)
- [Configuration Module](#configuration-module)
- [Ingestion Module](#ingestion-module)

---

## Main Module

### `rag_tools`

```python
from rag_tools import (
    RAGToolsManager,
    create_manager,
    create_pipeline,
    create_default_suite,
    MCPServer,
    MCPTool,
    RetrievalResult,
    EvaluationSuite,
    EvaluationReport,
    SyncResult,
    PipelineConfig,
)
```

### RAGToolsManager

```python
class RAGToolsManager
```

Main manager class for the RAG Tools module.

#### Constructor

```python
def __init__(
    self,
    config: Optional[Settings] = None,
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Optional[Settings]` | Configuration settings. Uses global settings if not provided. |
| `embedder` | `Optional[BaseEmbedder]` | Custom embedder. Uses LocalEmbedder if not provided. |
| `reranker` | `Optional[BaseReranker]` | Custom reranker. Uses CrossEncoderReranker if not provided. |

#### Methods

##### `async initialize() -> None`

Initialize all components (databases, embedder, reranker, etc.).

##### `async close() -> None`

Close all connections and cleanup resources.

##### `async session() -> AsyncGenerator[RAGToolsManager, None]`

Context manager for automatic initialization and cleanup.

```python
async with manager.session():
    results = await manager.retrieve_tools("search papers")
```

##### `async add_server() -> MCPServer`

Add a new MCP server to the registry.

```python
async def add_server(
    self,
    name: str,
    protocol: str = MCPProtocol.HTTP,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    sync_tools: bool = True,
) -> MCPServer
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Display name for the server |
| `protocol` | `str` | transport to connect to MCP server |
| `url` | `str` | MCP server URL if protocol is http |
| `command` | `str` | command for MCP server if protocol is stdio |
| `args` | `List[str]` | args for MCP server if protocol is stdio |
| `env` | `Dict[str,str]` | env variables for MCP server if protocol is stdio |
| `description` | `Optional[str]` | Server description |
| `headers` | `Optional[Dict[str, str]]` | HTTP headers for authentication |
| `sync_tools` | `bool` | Whether to immediately sync tools from server |

**Returns:** The created `MCPServer` object.

##### `async remove_server(server_id: str) -> Dict[str, Any]`

Remove a server and all its tools.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_id` | `str` | Server ID to remove |

**Returns:** Dictionary with removal statistics.

##### `async sync_server(server_id: str) -> SyncResult`

Sync tools from a specific server.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_id` | `str` | Server ID to sync |

**Returns:** `SyncResult` with sync statistics.

##### `async sync_all_servers() -> List[SyncResult]`

Sync all registered servers.

**Returns:** List of `SyncResult` for each server.

##### `async add_tool() -> MCPTool`

Add a single tool manually.

```python
async def add_tool(
    self,
    server_id: str,
    name: str,
    description: str,
    input_schema: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> MCPTool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_id` | `str` | Parent server ID |
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |
| `input_schema` | `Optional[Dict[str, Any]]` | JSON schema for input parameters |
| `tags` | `Optional[List[str]]` | Tool categorization tags |

**Returns:** The created `MCPTool` object.

##### `async remove_tool(tool_id: str) -> bool`

Remove a tool from the registry.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_id` | `str` | Tool ID to remove |

**Returns:** `True` if successful.

##### `async retrieve() -> PipelineResult`

Retrieve tools using the full RAG pipeline.

```python
async def retrieve(
    self,
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    rerank_top_k: int = 5,
    min_score: float = 0.3,
    use_chunks: bool = False,
    server_filter: Optional[List[str]] = None,
    tags_filter: Optional[List[str]] = None,
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | - | Search query |
| `top_k` | `int` | `10` | Number of initial results to retrieve |
| `rerank` | `bool` | `True` | Whether to apply reranking |
| `rerank_top_k` | `int` | `5` | Number of final results after reranking |
| `min_score` | `float` | `0.3` | Minimum relevance score threshold |
| `use_chunks` | `bool` | `False` | Use chunked collection for retrieval |
| `server_filter` | `Optional[List[str]]` | `None` | Filter by server IDs |
| `tags_filter` | `Optional[List[str]]` | `None` | Filter by tags |

**Returns:** `PipelineResult` with retrieval results and timing.

##### `async retrieve_tools() -> List[RetrievalResult]`

Simplified retrieval interface.

```python
async def retrieve_tools(
    self,
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    rerank_top_k: int = 5,
    min_score: float = 0.3,
) -> List[RetrievalResult]
```

**Returns:** List of `RetrievalResult` objects.

##### `create_evaluator() -> RAGEvaluator`

Create an evaluator for the retrieval pipeline.

**Returns:** `RAGEvaluator` instance.

##### `async evaluate() -> EvaluationReport`

Run evaluation with a test suite.

```python
async def evaluate(
    self,
    suite: EvaluationSuite,
) -> EvaluationReport
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `suite` | `EvaluationSuite` | Test suite with ground truth |

**Returns:** `EvaluationReport` with metrics.

##### `async get_stats() -> Dict[str, Any]`

Get registry statistics.

**Returns:** Dictionary with PostgreSQL and Qdrant stats.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `postgres` | `PostgresClient` | PostgreSQL client instance |
| `qdrant` | `QdrantClientWrapper` | Qdrant client instance |
| `pipeline` | `ToolRetrievalPipeline` | Retrieval pipeline instance |

---

### Factory Functions

#### `create_manager()`

```python
async def create_manager(
    config: Optional[Settings] = None,
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
) -> RAGToolsManager
```

Create and initialize a RAGToolsManager.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Optional[Settings]` | Configuration settings |
| `embedder` | `Optional[BaseEmbedder]` | Custom embedder |
| `reranker` | `Optional[BaseReranker]` | Custom reranker |

**Returns:** Initialized `RAGToolsManager` instance.

#### `create_pipeline()`

```python
def create_pipeline(
    embedder: Optional[BaseEmbedder] = None,
    use_cross_encoder: bool = True,
    device: str = "cpu",
) -> ToolRetrievalPipeline
```

Create a retrieval pipeline with optional components.

#### `create_default_suite() -> EvaluationSuite`

Create a default evaluation suite for testing.

---

## Storage Module

### `rag_tools.storage`

```python
from rag_tools.storage import (
    MCPServer,
    MCPTool,
    ToolCredential,
    ToolChunk,
    RetrievalResult,
    EvaluationResult,
    ToolStatus,
    PostgresClient,
    QdrantClientWrapper,
)
```

### Data Models

#### `MCPServer`

```python
class MCPServer(BaseModel)
```

MCP server metadata.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `server_id` | `str` | Unique identifier |
| `name` | `str` | Display name |
| `protocol` | `str` | transport for MCP server connection |
| `url` | `str` | HTTP URL if protocol is http |
| `command` | `str` | Command for MCP server if protocol is stdio |
| `args` | `List[str]` | Arguments for MCP server if protocol is stdio |
| `env` | `Dict[str,str]` | Enviroment variables for MCP server if protocol is stdio |
| `description` | `Optional[str]` | Description |
| `session_id` | `Optional[str]` | MCP session ID |
| `headers` | `Dict[str, str]` | HTTP headers |
| `timeout` | `int` | Request timeout (seconds) |
| `status` | `ToolStatus` | Server status |
| `last_synced` | `Optional[datetime]` | Last sync timestamp |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Update timestamp |

#### `MCPTool`

```python
class MCPTool(BaseModel)
```

Single MCP tool.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tool_id` | `str` | Unique identifier |
| `server_id` | `str` | Parent server ID |
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |
| `input_schema` | `Dict[str, Any]` | Input JSON schema |
| `output_schema` | `Optional[Dict[str, Any]]` | Output JSON schema |
| `tags` | `List[str]` | Categorization tags |
| `status` | `ToolStatus` | Tool status |
| `version` | `str` | Version string |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Update timestamp |

#### `ToolChunk`

```python
class ToolChunk(BaseModel)
```

Text chunk for tool embedding.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | Unique chunk identifier |
| `tool_id` | `str` | Parent tool ID |
| `chunk_index` | `int` | Position in chunk sequence |
| `text` | `str` | Chunk text content |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### `RetrievalResult`

```python
class RetrievalResult(BaseModel)
```

Result from retrieval query.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tool_id` | `str` | Tool identifier |
| `server_id` | `str` | Server identifier |
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |
| `score` | `float` | Relevance score |
| `rerank_score` | `Optional[float]` | Reranked score |
| `rank` | `int` | Result rank |
| `chunk_id` | `Optional[str]` | Chunk ID if from chunks |
| `text` | `Optional[str]` | Retrieved text |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### `ToolStatus`

```python
class ToolStatus(str, Enum)
```

Status enumeration.

**Values:**

| Value | Description |
|-------|-------------|
| `ACTIVE` | Tool is active and available |
| `INACTIVE` | Tool is inactive |
| `ERROR` | Tool encountered an error |
| `SYNCING` | Tool is being synced |

### PostgresClient

```python
class PostgresClient
```

Async PostgreSQL client.

#### Constructor

```python
def __init__(self, config: Optional[PostgresSettings] = None) -> None
```

#### Methods

**Server Operations:**

- `async add_server(server: MCPServer) -> MCPServer`
- `async get_server(server_id: str) -> Optional[MCPServer]`
- `async list_servers(status: Optional[str] = None) -> List[MCPServer]`
- `async delete_server(server_id: str) -> bool`

**Tool Operations:**

- `async add_tool(tool: MCPTool) -> MCPTool`
- `async get_tool(tool_id: str) -> Optional[MCPTool]`
- `async get_tools_by_server(server_id: str) -> List[MCPTool]`
- `async delete_tool(tool_id: str) -> bool`
- `async bulk_add_tools(tools: List[MCPTool]) -> int`

**Chunk Operations:**

- `async add_chunks(chunks: List[ToolChunk]) -> int`
- `async get_chunk(chunk_id: str) -> Optional[ToolChunk]`
- `async delete_chunk(chunk_id: str) -> bool`
- `async get_chunks_by_tool(tool_id: str) -> List[ToolChunk]`

**Credential Operations:**

- `async add_credential(credential: ToolCredential) -> ToolCredential`
- `async get_credential(server_id: str, tool_id: Optional[str] = None) -> Optional[ToolCredential]`
- `async delete_credential(credential_id: str) -> bool`

**Utilities:**

- `async get_stats() -> Dict[str, int]`
- `async initialize() -> None`
- `async close() -> None`

### QdrantClientWrapper

```python
class QdrantClientWrapper
```

Async Qdrant vector database wrapper.

#### Constructor

```python
def __init__(self, config: Optional[QdrantSettings] = None) -> None
```

#### Methods

**Connection:**

- `async connect() -> None`
- `async close() -> None`
- `async set_embedding_dim(dim: int) -> None`
- `async health_check() -> Dict[str, Any]`

**Collection Management:**

- `async create_collection(collection_name: str, vector_size: int, distance: Distance = Distance.COSINE) -> bool`
- `async delete_collection(collection_name: str) -> bool`
- `async collection_exists(collection_name: str) -> bool`
- `async get_collection_info(collection_name: str) -> Optional[Dict[str, Any]]`
- `async list_collections() -> List[str]`

**Point Operations:**

- `async upsert_points(collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> bool`
- `async delete_points(collection_name: str, point_ids: List[str]) -> bool`
- `async delete_by_filter(collection_name: str, filter_conditions: List[FieldCondition]) -> int`

**Search Operations:**

- `async search(collection_name: str, query_vector: List[float], top_k: int = 10, score_threshold: Optional[float] = None, filter_conditions: Optional[List[FieldCondition]] = None, with_payload: bool = True) -> List[Dict[str, Any]]`
- `async search_batch(collection_name: str, query_vectors: List[List[float]], top_k: int = 10, score_threshold: Optional[float] = None, filter_conditions: Optional[List[FieldCondition]] = None, with_payload: bool = True) -> List[List[Dict[str, Any]]]`

**Retrieval:**

- `async retrieve(collection_name: str, point_ids: List[str], with_payload: bool = True) -> List[Dict[str, Any]]`
- `async scroll(collection_name: str, limit: int = 100, offset: Optional[str] = None, filter_conditions: Optional[List[FieldCondition]] = None, with_payload: bool = True) -> Tuple[List[Dict[str, Any]], Optional[str]]`

**Indexing:**

- `async create_payload_index(collection_name: str, field_name: str, field_schema: Optional[Any] = None) -> bool`
- `async count(collection_name: str, filter_conditions: Optional[List[FieldCondition]] = None) -> int`

---

## Retrieval Module

### `rag_tools.retrieval`

```python
from rag_tools.retrieval import (
    BaseEmbedder,
    LocalEmbedder,
    CachedEmbedder,
    APIEmbedder,
    BaseReranker,
    CrossEncoderReranker,
    SimpleReranker,
    BM25Reranker,
    HybridReranker,
    ToolRetriever,
    RetrievalConfig,
    ToolRetrievalPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline,
)
```

### Embedders

#### BaseEmbedder

```python
class BaseEmbedder(ABC)
```

Abstract base class for embedders.

**Properties:**

- `embedding_dim: int` - Embedding dimension
- `model_name: str` - Model name

**Methods:**

- `async initialize() -> None` - Initialize the embedder
- `async embed(texts: Union[str, List[str]], batch_size: Optional[int] = None, normalize: Optional[bool] = True) -> np.ndarray` - Generate embeddings
- `async embed_query(query: str) -> np.ndarray` - Embed a query
- `async compute_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray` - Compute similarity
- `encode_sync(texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray` - Synchronous encoding

#### LocalEmbedder

```python
class LocalEmbedder(BaseEmbedder)
```

Sentence-transformers based embedder.

```python
LocalEmbedder(config: Optional[EmbeddingSettings] = None)
```

#### CachedEmbedder

```python
class CachedEmbedder(BaseEmbedder)
```

Cache wrapper for embedders.

```python
CachedEmbedder(base: BaseEmbedder, cache_size: int = 10000)
```

**Additional Methods:**

- `clear_cache() -> None` - Clear the embedding cache

#### APIEmbedder

```python
class APIEmbedder(BaseEmbedder)
```

Remote HTTP embedding service.

```python
APIEmbedder(config: Optional[APIEmbeddingSettings] = None)
```

### Retrievers

#### ToolRetriever

```python
class ToolRetriever
```

Retrieves tools from Qdrant.

```python
ToolRetriever(
    qdrant_client: QdrantClientWrapper,
    embedder: Optional[BaseEmbedder] = None
)
```

**Methods:**

- `async initialize() -> None` - Initialize retriever
- `async retrieve(query: str, config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]` - Retrieve tools
- `async retrieve_batch(queries: List[str], config: Optional[RetrievalConfig] = None) -> List[List[RetrievalResult]]` - Batch retrieval
- `async get_tool_by_id(tool_id: str) -> Optional[RetrievalResult]` - Get single tool
- `async get_tools_by_ids(tool_ids: List[str]) -> List[RetrievalResult]` - Get multiple tools
- `async get_all_tools(limit: int = 100, offset: Optional[str] = None) -> Tuple[List[RetrievalResult], Optional[str]]` - Get all tools with pagination

#### RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    top_k: int = 10
    min_score: float = 0.0
    use_chunks: bool = False
    server_filter: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None
```

### Rerankers

#### BaseReranker

```python
class BaseReranker(ABC)
```

Abstract base class for rerankers.

**Methods:**

- `async initialize() -> None` - Initialize the reranker
- `async rerank(query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> List[RetrievalResult]` - Rerank results
- `async rerank_with_scores(query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]` - Get reranked indices and scores

#### CrossEncoderReranker

```python
class CrossEncoderReranker(BaseReranker)
```

Cross-encoder reranker using sentence-transformers.

```python
CrossEncoderReranker(config: Optional[CrossEncoderRerankerSettings] = None)
```

#### BM25Reranker

```python
class BM25Reranker(BaseReranker)
```

BM25 keyword-based reranker.

```python
BM25Reranker(config: Optional[BM25RerankerSettings] = None)
```

#### HybridReranker

```python
class HybridReranker(BaseReranker)
```

Combines multiple rerankers.

```python
HybridReranker(
    rerankers: List[BaseReranker],
    config: Optional[HybridRerankerSettings] = None
)
```

#### SimpleReranker

```python
class SimpleReranker(BaseReranker)
```

Keyword overlap reranker (baseline).

### Pipeline

#### ToolRetrievalPipeline

```python
class ToolRetrievalPipeline
```

Full RAG retrieval pipeline.

```python
ToolRetrievalPipeline(
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    retriever: Optional[ToolRetriever] = None
)
```

**Methods:**

- `async initialize() -> None` - Initialize all components
- `async retrieve(query: str, config: Optional[PipelineConfig] = None) -> PipelineResult` - Full retrieval with timing
- `async retrieve_batch(queries: List[str], config: Optional[PipelineConfig] = None) -> List[PipelineResult]` - Batch retrieval
- `async retrieve_with_fallback(query: str, configs: Optional[List[PipelineConfig]] = None) -> PipelineResult` - Retrieve with fallback strategies

#### PipelineConfig

```python
@dataclass
class PipelineConfig:
    top_k: int = 10
    rerank: bool = True
    rerank_top_k: int = 5
    min_relevance_score: float = 0.3
    use_chunks: bool = False
    server_filter: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None
```

#### PipelineResult

```python
@dataclass
class PipelineResult:
    query: str
    results: List[RetrievalResult]
    config: PipelineConfig
    retrieval_time_ms: float
    rerank_time_ms: float
    total_time_ms: float
    metadata: Dict[str, Any]
```

---

## Evaluation Module

### `rag_tools.evaluation`

```python
from rag_tools.evaluation import (
    MetricsCalculator,
    MetricsAggregator,
    GroundTruthEntry,
    RAGEvaluator,
    EvaluationSuite,
    EvaluationReport,
    QueryEvaluation,
    create_default_suite,
)
```

### Evaluation Models

#### GroundTruthEntry

```python
@dataclass
class GroundTruthEntry:
    query: str
    relevant_tools: List[str]
```

#### EvaluationSuite

```python
@dataclass
class EvaluationSuite:
    name: str
    description: str
    queries: List[GroundTruthEntry]
```

#### QueryEvaluation

```python
@dataclass
class QueryEvaluation:
    query: str
    predicted_tools: List[str]
    relevant_tools: List[str]
    metrics: Dict[str, float]
    retrieval_result: PipelineResult
```

#### EvaluationReport

```python
@dataclass
class EvaluationReport:
    suite_name: str
    total_queries: int
    aggregated_metrics: Dict[str, Dict[str, float]]
    per_query_results: List[QueryEvaluation]
    metadata: Dict[str, Any]
```

### MetricsCalculator

```python
class MetricsCalculator
```

Calculate retrieval metrics.

**Static Methods:**

- `recall_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int) -> float`
- `precision_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int) -> float`
- `mean_reciprocal_rank(results: List[RetrievalResult], relevant_ids: List[str]) -> float`
- `average_precision(results: List[RetrievalResult], relevant_ids: List[str]) -> float`
- `ndcg_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int, relevance_scores: Optional[Dict[str, float]] = None) -> float`
- `hit_rate_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int) -> float`
- `mrr_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int) -> float`
- `coverage_at_k(results: List[RetrievalResult], relevant_ids: List[str], k: int) -> float`
- `diversity_at_k(results: List[RetrievalResult], k: int) -> float`
- `calculate_all_metrics(results, relevant_ids, k_values=[1, 3, 5, 10]) -> Dict[str, float]`

### MetricsAggregator

```python
class MetricsAggregator
```

Aggregates metrics across queries.

**Methods:**

- `add(metrics: Dict[str, float]) -> None` - Add query metrics
- `aggregate() -> Dict[str, Dict[str, float]]` - Aggregate all metrics
- `reset() -> None` - Reset aggregator

### RAGEvaluator

```python
class RAGEvaluator
```

Evaluates RAG pipeline performance.

```python
RAGEvaluator(
    pipeline: ToolRetrievalPipeline,
    k_values: Optional[List[int]] = None
)
```

**Methods:**

- `async evaluate_suite(suite: EvaluationSuite, config: Optional[PipelineConfig] = None) -> EvaluationReport`
- `async evaluate_custom(query: str, relevant_tools: List[str], config: Optional[PipelineConfig] = None) -> QueryEvaluation`
- `load_suite_from_file(file_path: Path) -> EvaluationSuite`
- `save_suite_to_file(suite: EvaluationSuite, file_path: Path) -> None`
- `print_report(report: EvaluationReport) -> None`
- `export_report(report: EvaluationReport, file_path: Path) -> None`

---

## Tool Management Module

### `rag_tools.tools`

```python
from rag_tools.tools import (
    ToolAdder,
    ToolRemover,
    MCPSyncer,
    SyncResult,
    add_mcp_server,
    remove_mcp_server,
    remove_tool_by_id,
    sync_server_by_id,
    sync_all_servers,
)
```

### ToolAdder

```python
class ToolAdder
```

Add tools to the registry and index.

```python
ToolAdder(
    postgres: PostgresClient,
    qdrant: QdrantClientWrapper,
    indexer: Optional[ToolIndexer] = None
)
```

**Methods:**

- `async add_server(name: str, protocol: str = 'http', url: Optional[str] = None, command: Optional[str] = None, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None,  description: Optional[str] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30, sync_tools: bool = True) -> MCPServer`
- `async add_tool(server_id: str, name: str, description: str, input_schema: Optional[Dict[str, Any]] = None, output_schema: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None, index: bool = True) -> MCPTool`
- `async add_tools_batch(tools: List[MCPTool], index: bool = True, batch_size: int = 32) -> Dict[str, int]`
- `async sync_server_tools(server: MCPServer) -> List[MCPTool]`
- `async add_credential(server_id: str, credential_type: str, credential_data: Dict[str, str], tool_id: Optional[str] = None) -> str`

### ToolRemover

```python
class ToolRemover
```

Remove tools from the registry and index.

```python
ToolRemover(
    postgres: PostgresClient,
    qdrant: QdrantClientWrapper,
    indexer: Optional[ToolIndexer] = None
)
```

**Methods:**

- `async remove_tool(tool_id: str) -> bool`
- `async remove_tools_batch(tool_ids: List[str]) -> Dict[str, int]`
- `async remove_server(server_id: str) -> Dict[str, Any]`
- `async remove_by_filter(server_ids: Optional[List[str]] = None, tags: Optional[List[str]] = None, status: Optional[ToolStatus] = None) -> Dict[str, int]`
- `async remove_inactive_tools(older_than_hours: int = 24) -> Dict[str, int]`
- `async clear_all() -> Dict[str, Any]`

### MCPSyncer

```python
class MCPSyncer
```

Sync tools from MCP servers.

```python
MCPSyncer(
    postgres: PostgresClient,
    qdrant: QdrantClientWrapper,
    indexer: Optional[ToolIndexer] = None
)
```

**Methods:**

- `async sync_server(server_id: str, progress_callback: Optional[Callable[[str, int, int], None]] = None) -> SyncResult`
- `async sync_all_servers(progress_callback: Optional[Callable[[str, int, int], None]] = None) -> List[SyncResult]`
- `async sync_with_retry(server_id: str, max_retries: int = 3, retry_delay: float = 1.0) -> SyncResult`
- `async create_server_and_sync(url: str, name: str, description: Optional[str] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> SyncResult`
- `async schedule_periodic_sync(interval_hours: float, servers: Optional[List[str]] = None) -> asyncio.Task`

### SyncResult

```python
@dataclass
class SyncResult:
    server_id: str
    server_name: str
    success: bool
    tools_added: int
    tools_removed: int
    tools_updated: int
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: float
```

### Convenience Functions

- `async add_mcp_server(url: str, name: str, **kwargs) -> MCPServer`
- `async remove_mcp_server(server_id: str) -> Dict[str, Any]`
- `async remove_tool_by_id(tool_id: str) -> bool`
- `async sync_server_by_id(server_id: str) -> SyncResult`
- `async sync_all_servers() -> List[SyncResult]`

---

## Configuration Module

### `rag_tools.config`

```python
from rag_tools.config import Settings, get_settings
from rag_tools.config.settings import (
    QdrantSettings,
    PostgresSettings,
    EmbeddingSettings,
    APIEmbeddingSettings,
    CrossEncoderRerankerSettings,
    APIRerankerSettings,
    BM25RerankerSettings,
    HybridRerankerSettings,
    RAGSettings,
)
```

### Settings Classes

#### QdrantSettings

```python
class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
```

#### PostgresSettings

```python
class PostgresSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "rag_tools"
    password: str = "rag_tools_password"
    database: str = "rag_tools"
    min_connections: int = 2
    max_connections: int = 20
```

#### EmbeddingSettings

```python
class EmbeddingSettings(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
```

#### CrossEncoderRerankerSettings

```python
class CrossEncoderRerankerSettings(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32
    top_k: int = 20
    max_length: int = 512
```

#### BM25RerankerSettings

```python
class BM25RerankerSettings(BaseModel):
    k1: float = 1.5
    b: float = 0.75
```

#### HybridRerankerSettings

```python
class HybridRerankerSettings(BaseModel):
    weights: List[float] = [0.7, 0.3]
```

#### RAGSettings

```python
class RAGSettings(BaseModel):
    default_top_k: int = 10
    rerank_top_k: int = 5
    min_relevance_score: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 50
```

#### Settings

```python
class Settings(BaseSettings):
    qdrant: QdrantSettings = QdrantSettings()
    postgres: PostgresSettings = PostgresSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    api_embedding: APIEmbeddingSettings = APIEmbeddingSettings()
    cross_encoder_reranker: CrossEncoderRerankerSettings = CrossEncoderRerankerSettings()
    api_reranker: APIRerankerSettings = APIRerankerSettings()
    bm_reranker: BM25RerankerSettings = BM25RerankerSettings()
    hybrid_reranker: HybridRerankerSettings = HybridRerankerSettings()
    rag: RAGSettings = RAGSettings()
    work_dir: Path = Path("/tmp/rag_tools")
    tools_collection: str = "mcp_tools"
    tools_chunks_collection: str = "mcp_tools_chunks"
```

### Global Instance

```python
settings = Settings()
get_settings() -> Settings
```

---

## Ingestion Module

### `rag_tools.ingestion`

```python
from rag_tools.ingestion import (
    MCPToolInfo,
    MCPToolsResponse,
    parse_mcp_tools_response,
    create_tool_id,
    mcp_tool_info_to_model,
    TextBuilder,
    build_tool_embedding_text,
    ToolIndexer,
)
```

### Parser Functions

#### `parse_mcp_tools_response(response: Dict[str, Any]) -> MCPToolsResponse`

Parse MCP JSON-RPC response.

#### `mcp_tool_info_to_model(tool_info: MCPToolInfo, server_id: str) -> MCPTool`

Convert MCPToolInfo to MCPTool model.

#### `create_tool_id(server_id: str, tool_name: str) -> str`

Create unique tool ID.

### TextBuilder

```python
class TextBuilder
```

Build searchable text representations.

```python
TextBuilder(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
)
```

**Methods:**

- `build_full_text(tool: MCPTool) -> str` - Full searchable text
- `build_summary_text(tool: MCPTool) -> str` - Short summary
- `build_name_text(tool: MCPTool) -> str` - Name-focused text
- `build_chunks(tool: MCPTool) -> List[ToolChunk]` - Text chunks
- `build_search_text(tool: MCPTool, include_schema: bool = True, include_description: bool = True) -> str` - Custom search text

### ToolIndexer

```python
class ToolIndexer
```

Index tools to PostgreSQL and Qdrant.

```python
ToolIndexer(
    postgres_client: PostgresClient,
    qdrant_client: QdrantClientWrapper,
    embedder: Optional[BaseEmbedder] = None
)
```

**Methods:**

- `async initialize() -> None` - Initialize indexer
- `async index_tool(tool: MCPTool, rebuild: bool = False) -> bool` - Index single tool
- `async index_tools_batch(tools: List[MCPTool], batch_size: int = 32, show_progress: bool = True) -> Dict[str, int]` - Batch indexing
- `async index_tool_chunks(tool: MCPTool) -> int` - Index as chunks
- `async remove_tool(tool_id: str) -> bool` - Remove from index
- `async remove_server_tools(server_id: str) -> int` - Remove all server tools
- `async reindex_tool(tool: MCPTool) -> bool` - Reindex tool
- `get_index_stats() -> Dict[str, Any]` - Get indexing statistics
