# Examples Guide

This guide provides practical examples for using RAG Tools in various scenarios.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Advanced Usage](#advanced-usage)
- [Integration Patterns](#integration-patterns)
- [Performance Optimization](#performance-optimization)
- [Common Use Cases](#common-use-cases)

---

## Basic Examples

### Example 1: Minimal Setup

The simplest way to get started with RAG Tools:

```python
import asyncio
from rag_tools import create_manager

async def minimal_example():
    # Create and initialize manager with default settings
    manager = await create_manager()

    # Add a server (sync_tools=False for demo without real MCP server)
    server = await manager.add_server(
        protocol='http',
        url="http://localhost:8080/mcp",
        name="demo-server",
        sync_tools=False
    )

    # Add a test tool
    tool = await manager.add_tool(
        server_id=server.server_id,
        name="search",
        description="Search for information in the database"
    )

    # Retrieve
    results = await manager.retrieve_tools("find data")
    print(f"Found: {results[0].name if results else 'None'}")

    await manager.close()

asyncio.run(minimal_example())
```

### Example 2: Complete Workflow

Full workflow from server addition to retrieval:

```python
import asyncio
from rag_tools import create_manager

async def complete_workflow():
    # Initialize
    manager = await create_manager()

    # 1. Add MCP Server
    server = await manager.add_server(
        protocol='http',
        url="http://10.32.11.22:7332/mcp",
        name="chemistry-tools",
        description="Tools for chemistry research",
        sync_tools=True
    )

    print(f"Added server: {server.name} (ID: {server.server_id})")
    print(f"Status: {server.status}")

    # 2. Add manual tools (if needed)
    custom_tool = await manager.add_tool(
        server_id=server.server_id,
        name="molecular_search",
        description="Search for molecules by structure or properties",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"}
            },
            "required": ["query"]
        },
        tags=["chemistry", "molecular", "search"]
    )

    # 3. Retrieve tools
    results = await manager.retrieve_tools(
        query="Find molecules for drug discovery",
        top_k=10,
        rerank=True,
        rerank_top_k=5
    )

    print("\nRetrieval Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.name}")
        print(f"   Score: {r.score:.4f}")
        print(f"   Rerank: {r.rerank_score:.4f}" if r.rerank_score else "")

    # 4. Get detailed retrieval info
    detailed = await manager.retrieve(
        query="molecular docking",
        top_k=10,
        rerank=True
    )
    print(f"\nTiming:")
    print(f"  Retrieval: {detailed.retrieval_time_ms:.2f}ms")
    print(f"  Reranking: {detailed.rerank_time_ms:.2f}ms")

    # 5. Sync for updates
    sync_result = await manager.sync_server(server.server_id)
    print(f"\nSync: {'Success' if sync_result.success else 'Failed'}")
    print(f"  Added: {sync_result.tools_added}")
    print(f"  Updated: {sync_result.tools_updated}")

    # 6. Cleanup
    await manager.remove_tool(custom_tool.tool_id)
    await manager.close()

asyncio.run(complete_workflow())
```

### Example 3: Multiple Server Management

Managing multiple MCP servers:

```python
import asyncio
from rag_tools import create_manager

async def multi_server():
    manager = await create_manager()

    # Add multiple servers
    servers = []
    for name, url in [
        ("server-a", "http://server-a:8080/mcp"),
        ("server-b", "http://server-b:8080/mcp"),
        ("server-c", "http://server-c:8080/mcp"),
    ]:
        try:
            server = await manager.add_server(
                protocol='http',
                url=url,
                name=name,
                sync_tools=False  # Demo mode
            )
            servers.append(server)
            print(f"Added: {server.name}")
        except Exception as e:
            print(f"Failed to add {name}: {e}")

    # Sync all servers
    sync_results = await manager.sync_all_servers()
    for result in sync_results:
        print(f"\n{result.server_name}:")
        print(f"  Success: {result.success}")
        print(f"  Tools: +{result.tools_added}, ~{result.tools_updated}")

    # Filter by server
    results = await manager.retrieve_tools(
        query="search",
        server_filter=[servers[0].server_id]  # Only search first server
    )
    print(f"\nResults from {servers[0].name}: {len(results)}")

    # Remove servers
    for server in servers:
        await manager.remove_server(server.server_id)

    await manager.close()

asyncio.run(multi_server())
```

---

## Advanced Usage

### Example 4: Custom Embedder and Reranker

Using custom embedding and reranking models:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.retrieval import (
    APIEmbedder,
    APIReranker,
    BM25Reranker,
    HybridReranker,
    CrossEncoderReranker,
    LocalEmbedder
)
from rag_tools.config import Settings

async def custom_models():
    # Create custom settings
    settings = Settings()

    # Option 1: API Embedding
    api_embedder = APIEmbedder(settings.api_embedding)

    # Option 2: Local Embedding with GPU
    local_embedder = LocalEmbedder(settings.embedding)

    # Custom rerankers
    cross_encoder = CrossEncoderReranker(settings.cross_encoder_reranker)
    bm25 = BM25Reranker(settings.bm_reranker)

    # Hybrid reranker combining multiple strategies
    hybrid_reranker = HybridReranker(
        rerankers=[cross_encoder, bm25],
        config=settings.hybrid_reranker
    )

    # Create manager with custom components
    manager = await create_manager(
        settings,
        embedder=local_embedder,
        reranker=hybrid_reranker
    )

    # Add test data
    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test",
        sync_tools=False
    )

    # Add diverse tools
    tools = [
        ("web_search", "Search the web for information", ["search", "web"]),
        ("file_search", "Search files on the filesystem", ["search", "files"]),
        ("image_search", "Search for images", ["search", "images"]),
        ("code_search", "Search code repositories", ["search", "code"]),
        ("data_analysis", "Analyze data and generate insights", ["analysis", "data"]),
    ]

    for name, desc, tags in tools:
        await manager.add_tool(
            server_id=server.server_id,
            name=name,
            description=desc,
            tags=tags
        )

    # Test different retrieval strategies
    queries = ["find information", "search for files", "analyze data"]

    for query in queries:
        # With reranking
        results = await manager.retrieve_tools(
            query=query,
            rerank=True
        )

        print(f"\nQuery: '{query}'")
        print("Top 3 with reranking:")
        for r in results[:3]:
            print(f"  - {r.name} ({r.rerank_score:.4f})")

    await manager.close()

asyncio.run(custom_models())
```

### Example 5: Chunked Retrieval

For tools with long descriptions:

```python
import asyncio
from rag_tools import create_manager

async def chunked_retrieval():
    manager = await create_manager()

    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test",
        sync_tools=False
    )

    # Add tool with long description
    long_description = """
    This is a comprehensive tool for searching academic papers from multiple databases.
    It supports advanced features including:
    - Boolean queries with AND, OR, NOT operators
    - Date range filtering
    - Citation tracking
    - Full-text search
    - Abstract extraction
    - Author disambiguation
    - Institution linking
    - Journal impact filtering
    - Peer review status
    - Open access filtering

    The tool returns structured results including:
    - Paper metadata (title, authors, abstract)
    - Citation counts
    - Publication venue
    - DOI and URLs
    - Related papers

    Performance characteristics:
    - Query latency: <100ms for simple queries
    - Batch processing: up to 1000 papers per request
    - Rate limiting: 100 requests per minute
    - Caching: 24-hour TTL for common queries
    """

    await manager.add_tool(
        server_id=server.server_id,
        name="academic_search",
        description=long_description,
        tags=["academic", "papers", "search"]
    )

    # Compare standard vs chunked retrieval
    queries = [
        "search for papers about machine learning",
        "find articles with high citation counts",
        "get papers on neural networks"
    ]

    for query in queries:
        # Standard retrieval
        standard = await manager.retrieve(
            query=query,
            use_chunks=False
        )

        # Chunked retrieval
        chunked = await manager.retrieve(
            query=query,
            use_chunks=True
        )

        print(f"\nQuery: '{query}'")
        print(f"  Standard: {len(standard.results)} results ({standard.retrieval_time_ms:.2f}ms)")
        print(f"  Chunked:  {len(chunked.results)} results ({chunked.retrieval_time_ms:.2f}ms)")

    await manager.close()

asyncio.run(chunked_retrieval())
```

### Example 6: Evaluation Suite

Creating and running evaluation:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.evaluation import EvaluationSuite, GroundTruthEntry, RAGEvaluator

async def evaluation_example():
    manager = await create_manager()

    # Setup test data
    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test-server",
        sync_tools=False
    )

    test_tools = [
        ("search_papers", "Search for academic papers"),
        ("download_papers", "Download paper PDFs"),
        ("get_author_info", "Get author information"),
        ("list_journals", "List academic journals"),
        ("find_citations", "Find citation relationships"),
    ]

    for name, desc in test_tools:
        await manager.add_tool(
            server_id=server.server_id,
            name=name,
            description=desc,
            tags=["academic", "research"]
        )

    # Create evaluation suite
    suite = EvaluationSuite(
        name="academic-tools-eval",
        description="Evaluation for academic search tools",
        queries=[
            GroundTruthEntry(
                query="how to search for papers",
                relevant_tools=[f"{server.server_id}:search_papers"]
            ),
            GroundTruthEntry(
                query="download PDF documents",
                relevant_tools=[f"{server.server_id}:download_papers"]
            ),
            GroundTruthEntry(
                query="find author details",
                relevant_tools=[f"{server.server_id}:get_author_info"]
            ),
            GroundTruthEntry(
                query="get journal list",
                relevant_tools=[f"{server.server_id}:list_journals"]
            ),
            GroundTruthEntry(
                query="academic search",
                relevant_tools=[
                    f"{server.server_id}:search_papers",
                    f"{server.server_id}:find_citations"
                ]
            ),
        ]
    )

    # Run evaluation
    evaluator = manager.create_evaluator()
    report = await evaluator.evaluate_suite(suite)

    # Print formatted results
    evaluator.print_report(report)

    # Export for analysis
    evaluator.export_report(report, "/tmp/eval_report.json")
    print("\nReport exported to /tmp/eval_report.json")

    # Access individual metrics
    print("\nDetailed Metrics:")
    for metric_name, values in report.aggregated_metrics.items():
        print(f"  {metric_name}: {values['mean']:.4f}")

    await manager.close()

asyncio.run(evaluation_example())
```

---

## Integration Patterns

### Example 7: As a FastAPI Service

Integrate RAG Tools into a FastAPI application:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio

from rag_tools import create_manager

# Global manager
manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = await create_manager()
    yield
    if manager:
        await manager.close()

app = FastAPI(lifespan=lifespan)

class ServerRequest(BaseModel):
    url: str
    name: str
    description: str = ""
    sync_tools: bool = True

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True

@app.post("/servers")
async def add_server(request: ServerRequest):
    try:
        server = await manager.add_server(**request.dict())
        return {"server_id": server.server_id, "name": server.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/servers")
async def list_servers():
    stats = await manager.get_stats()
    return stats

@app.post("/retrieve")
async def retrieve(request: RetrievalRequest):
    results = await manager.retrieve_tools(
        query=request.query,
        top_k=request.top_k,
        rerank=request.rerank
    )
    return {
        "query": request.query,
        "count": len(results),
        "results": [
            {
                "name": r.name,
                "description": r.description,
                "score": r.score,
                "rerank_score": r.rerank_score
            }
            for r in results
        ]
    }

@app.delete("/servers/{server_id}")
async def remove_server(server_id: str):
    result = await manager.remove_server(server_id)
    return result

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Example 8: Periodic Sync Task

Schedule periodic synchronization:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.tools import MCPSyncer

async def periodic_sync_example():
    manager = await create_manager()

    # Add servers
    server1 = await manager.add_server(
        protocol='http',
        url="http://server1:8080/mcp",
        name="server-1",
        sync_tools=True
    )
    server2 = await manager.add_server(
        protocol='http',
        url="http://server2:8080/mcp",
        name="server-2",
        sync_tools=True
    )

    # Create syncer
    syncer = MCPSyncer(manager.postgres, manager.qdrant)

    # Schedule sync every 6 hours
    sync_task = await syncer.schedule_periodic_sync(
        interval_hours=6,
        servers=[server1.server_id, server2.server_id]
    )

    print("Sync scheduled. Running for 1 hour...")

    # Run for demo purposes
    try:
        await asyncio.sleep(3600)  # 1 hour
    except asyncio.CancelledError:
        pass
    finally:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass

    await manager.close()

asyncio.run(periodic_sync_example())
```

### Example 9: Batch Processing

Process multiple queries efficiently:

```python
import asyncio
from rag_tools import create_manager

async def batch_retrieval():
    manager = await create_manager()

    # Setup
    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test",
        sync_tools=False
    )

    tools = [
        ("search", "Search for items"),
        ("filter", "Filter results"),
        ("sort", "Sort items"),
        ("aggregate", "Aggregate data"),
        ("visualize", "Create visualizations"),
    ]

    for name, desc in tools:
        await manager.add_tool(server_id=server.server_id, name=name, description=desc)

    # Batch queries
    queries = [
        "how to search",
        "filtering data",
        "sorting results",
        "data aggregation",
        "create charts",
        "find items",
        "organize data",
    ]

    # Method 1: Sequential
    print("Sequential retrieval:")
    for query in queries[:3]:
        results = await manager.retrieve_tools(query)
        print(f"  '{query}': {len(results)} results")

    # Method 2: Using pipeline batch (if supported)
    pipeline = manager.pipeline
    if hasattr(pipeline, 'retrieve_batch'):
        print("\nBatch retrieval:")
        results = await pipeline.retrieve_batch(queries)
        for query, result in zip(queries, results):
            print(f"  '{query}': {len(result.results)} results")

    await manager.close()

asyncio.run(batch_retrieval())
```

---

## Performance Optimization

### Example 10: GPU Acceleration

Use GPU for faster embedding:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.config import Settings

async def gpu_acceleration():
    settings = Settings()

    # Enable CUDA for embeddings
    settings.embedding.device = "cuda"
    settings.embedding.batch_size = 128

    # Enable CUDA for reranker
    settings.cross_encoder_reranker.device = "cuda"
    settings.cross_encoder_reranker.batch_size = 64

    manager = await create_manager(settings)

    # Setup test data
    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test",
        sync_tools=False
    )

    # Add many tools
    for i in range(100):
        await manager.add_tool(
            server_id=server.server_id,
            name=f"tool_{i}",
            description=f"Description for tool {i} with various capabilities"
        )

    # Benchmark
    import time

    queries = ["search", "analyze", "process", "calculate"]

    # Warm up
    await manager.retrieve_tools("warm up query")

    # Benchmark
    start = time.time()
    for _ in range(10):
        for query in queries:
            await manager.retrieve_tools(query)
    elapsed = time.time() - start

    print(f"Processed {10 * len(queries)} queries in {elapsed:.2f}s")
    print(f"Average: {elapsed / (10 * len(queries)) * 1000:.2f}ms/query")

    await manager.close()

asyncio.run(gpu_acceleration())
```

### Example 11: Caching

Use cached embeddings:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.retrieval import LocalEmbedder, CachedEmbedder

async def caching_example():
    settings = Settings()

    # Create base embedder
    base_embedder = LocalEmbedder(settings.embedding)

    # Wrap with cache
    cached_embedder = CachedEmbedder(
        base=base_embedder,
        cache_size=10000  # Cache up to 10000 embeddings
    )

    manager = await create_manager(
        settings,
        embedder=cached_embedder
    )

    # Setup
    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="test",
        sync_tools=False
    )

    for i in range(50):
        await manager.add_tool(
            server_id=server.server_id,
            name=f"tool_{i}",
            description=f"Tool {i} description"
        )

    # First query - cache miss
    import time
    start = time.time()
    await manager.retrieve_tools("search query 1")
    first_time = time.time() - start

    # Repeated queries - cache hits
    times = []
    for i in range(5):
        start = time.time()
        await manager.retrieve_tools("search query 1")  # Same query
        times.append(time.time() - start)

    avg_cached = sum(times) / len(times)

    print(f"First query: {first_time*1000:.2f}ms")
    print(f"Cached queries: {avg_cached*1000:.2f}ms")
    print(f"Speedup: {first_time/avg_cached:.2f}x")

    # Clear cache
    cached_embedder.clear_cache()

    await manager.close()

asyncio.run(caching_example())
```

---

## Common Use Cases

### Use Case 1: AI Assistant Tool Selection

Integrate with AI assistant for automatic tool selection:

```python
import asyncio
from rag_tools import create_manager

async def ai_assistant_example():
    manager = await create_manager()

    # Setup tools
    server = await manager.add_server(
        protocol='http',
        url="http://mcp-server:8080/mcp",
        name="assistant-tools",
        sync_tools=True
    )

    async def select_tools_for_request(user_request: str, max_tools: int = 5):
        """Select appropriate tools for a user request."""
        results = await manager.retrieve_tools(
            query=user_request,
            top_k=max_tools,
            rerank=True
        )
        return results

    # Example requests
    requests = [
        "I need to search for recent papers on quantum computing",
        "Can you find information about the author John Smith?",
        "Download the PDF for paper ID 12345",
        "List all journals in the field of machine learning",
    ]

    for request in requests:
        tools = await select_tools_for_request(request)

        print(f"\nRequest: '{request}'")
        print("Selected tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description[:50]}...")

    await manager.close()

asyncio.run(ai_assistant_example())
```

### Use Case 2: Tool Discovery

Help users discover available tools:

```python
import asyncio
from rag_tools import create_manager

async def tool_discovery():
    manager = await create_manager()

    server = await manager.add_server(
        protocol='http',
        url="http://test.com/mcp",
        name="demo",
        sync_tools=False
    )

    # Add diverse tools
    tools = [
        ("web_search", "Search the web", ["search", "web"]),
        ("file_search", "Search local files", ["search", "files"]),
        ("image_search", "Search for images", ["search", "images"]),
        ("video_search", "Search for videos", ["search", "videos"]),
        ("document_search", "Search documents", ["search", "documents"]),
        ("code_search", "Search code repositories", ["search", "code"]),
        ("data_analyzer", "Analyze datasets", ["analysis", "data"]),
        ("visualizer", "Create charts and graphs", ["visualization"]),
    ]

    for name, desc, tags in tools:
        await manager.add_tool(
            server_id=server.server_id,
            name=name,
            description=desc,
            tags=tags
        )

    async def discover_related_tools(tool_name: str):
        """Find tools related to a given tool."""
        # Get the tool
        tool = await manager.postgres.get_tool(
            f"{server.server_id}:{tool_name}"
        )

        if not tool:
            print(f"Tool '{tool_name}' not found")
            return

        # Search by name and tags
        results = await manager.retrieve_tools(
            query=f"{tool.name} {' '.join(tool.tags)}",
            top_k=5
        )

        return [r for r in results if r.name != tool_name]

    # Discover related tools
    related = await discover_related_tools("web_search")
    print("Tools related to 'web_search':")
    for r in related:
        print(f"  - {r.name} ({r.score:.4f})")

    await manager.close()

asyncio.run(tool_discovery())
```

### Use Case 3: Multi-Tenant Tool Registry

Support multiple tenants with isolated tool registries:

```python
import asyncio
from rag_tools import create_manager
from rag_tools.config import Settings

async def multi_tenant():
    # Create separate managers for different tenants
    tenant_a_settings = Settings()
    tenant_a_settings.tools_collection = "tenant_a_tools"

    tenant_b_settings = Settings()
    tenant_b_settings.tools_collection = "tenant_b_tools"

    manager_a = await create_manager(tenant_a_settings)
    manager_b = await create_manager(tenant_b_settings)

    # Tenant A adds their tools
    server_a = await manager_a.add_server(
        protocol='http',
        url="http://tenant-a-server/mcp",
        name="tenant-a-server",
        sync_tools=False
    )
    await manager_a.add_tool(
        server_id=server_a.server_id,
        name="private_search",
        description="Tenant A's private search tool",
        tags=["private", "tenant-a"]
    )

    # Tenant B adds their tools
    server_b = await manager_b.add_server(
        protocol='http',
        url="http://tenant-b-server/mcp",
        name="tenant-b-server",
        sync_tools=False
    )
    await manager_b.add_tool(
        server_id=server_b.server_id,
        name="private_search",
        description="Tenant B's private search tool",
        tags=["private", "tenant-b"]
    )

    # Each tenant only sees their own tools
    results_a = await manager_a.retrieve_tools("search")
    results_b = await manager_b.retrieve_tools("search")

    print(f"Tenant A sees: {[r.name for r in results_a]}")
    print(f"Tenant B sees: {[r.name for r in results_b]}")

    # Cleanup
    await manager_a.close()
    await manager_b.close()

asyncio.run(multi_tenant())
```
