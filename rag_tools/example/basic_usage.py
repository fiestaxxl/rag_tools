"""
Example: Basic Usage of RAG Tools

This example demonstrates the basic usage of the RAG Tools module
for managing and retrieving MCP tools.
"""
import asyncio
from rag_tools import create_manager, MCPServer
from rag_tools.config.settings import get_settings
from rag_tools.retrieval import APIEmbedder, APIReranker, BM25Reranker, HybridReranker


async def main():
    """Basic usage example."""
    print("RAG Tools - Basic Usage Example")
    print("=" * 50)

    # Create and initialize manager
    settings = get_settings()
    print(settings)

    embedder = APIEmbedder(settings.api_embedding)
    api_reranker = APIReranker(settings.api_reranker)
    bm2_reranker = BM25Reranker(settings.bm_reranker)
    reranker = HybridReranker([api_reranker, bm2_reranker], settings.hybrid_reranker)
    manager = await create_manager(settings, embedder, reranker)
    print("Manager initialized")

    # Example 1: Add a server manually
    print("\n1. Adding a server...")
    server = await manager.add_server(
        url="http://10.32.11.22:7332/mcp",
        name="example-server-chemistry",
        description="Example MCP server for demonstration",
        sync_tools=True
    )
    print(f"   Added server: {server.server_id}")

    # Example 2: Add a tool manually
    print("\n2. Adding a tool...")
    tool = await manager.add_tool(
        server_id=server.server_id,
        name="search_papers",
        description="Search for academic papers in the database. Supports filtering by author, year, and keywords.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        },
        tags=["search", "academic", "papers"]
    )
    print(f"   Added tool: {tool.name}")

    # Example 3: Retrieve tools
    print("\n3. Retrieving tools...")
    results = await manager.retrieve_tools("Get docking score for Alzheimer desease", rerank=True,
                                           top_k=10,
                                           rerank_top_k=5)

    print(f"   Found {len(results)} relevant tools:")
    for r in results:
        print(f"   - {r.name} (score: {r.score:.3f}, rerank_score: {r.rerank_score:.3f})")

    # Example 4: Get detailed results
    print("\n4. Detailed retrieval...")
    result = await manager.retrieve(
        query="Get docking score for Alzheimer desease",
        top_k=10,
        rerank=True,
        rerank_top_k=8,
        min_score=0.0
    )

    print(f"   Query: {result.query}")
    print(f"   Total results: {len(result.results)}")
    print(f"   Retrieval time: {result.retrieval_time_ms:.2f}ms")
    print(f"   Rerank time: {result.rerank_time_ms:.2f}ms")

    # Example 5: Get statistics
    print("\n5. Statistics...")
    stats = await manager.get_stats()
    print(f"   Tools: {stats['postgres'].get('tools', 'N/A')}")
    print(f"   Servers: {stats['postgres'].get('servers', 'N/A')}")
    print(f"   Embedding dimension: {stats['embedder']['dimension']}")

    # Cleanup
    print("\n6. Cleaning up...")
    await manager.remove_tool(tool.tool_id)
    await manager.remove_server(server.server_id)
    print("   Removed test data")

    await manager.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
