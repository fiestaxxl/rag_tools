"""
Example: MCP Server Sync

This example demonstrates syncing tools from an actual MCP server.
"""
import asyncio
from rag_tools import create_manager
from rag_tools.ingestion.mcp_client import test_mcp_connection


async def main():
    """MCP sync example."""
    print("RAG Tools - MCP Server Sync Example")
    print("=" * 50)

    # Test MCP connection (example with OpenAlex-style server)
    test_url = "http://10.32.11.22:7331/mcp"

    print(f"\nTesting connection to {test_url}...")
    connected = await test_mcp_connection(test_url)
    print(f"Connection: {'Success' if connected else 'Failed'}")

    if not connected:
        print("\nCould not connect to test server. Skipping sync example.")
        return

    # Create manager
    manager = await create_manager()

    # Add and sync server
    print("\nAdding and syncing MCP server...")
    server = await manager.add_server(
        url=test_url,
        name="openalex-api",
        description="OpenAlex API for academic paper search",
        sync_tools=True  # This will fetch tools from the server
    )

    print(f"Server: {server.name}")
    print(f"Status: {server.status}")

    # Get stats
    stats = await manager.get_stats()
    print(f"\nTotal tools indexed: {stats['postgres'].get('tools', 0)}")

    # Retrieve some tools
    if stats['postgres'].get('tools', 0) > 0:
        print("\nRetrieving tools for 'search papers':")
        results = await manager.retrieve_tools("search papers by keyword")
        for r in results[:3]:
            print(f"  - {r.name}: {r.description[:50]}...")

    # Sync again to check for updates
    print("\nSyncing server again...")
    sync_result = await manager.sync_server(server.server_id)
    print(f"Sync status: {'Success' if sync_result.success else 'Failed'}")
    print(f"Tools updated: {sync_result.tools_updated}")

    # Cleanup
    await manager.remove_server(server.server_id)
    await manager.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
