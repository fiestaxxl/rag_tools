"""
Example: Evaluation with RAG Tools

This example demonstrates how to evaluate retrieval quality.
"""
import asyncio
from rag_tools import create_manager, RAGEvaluator, EvaluationSuite, GroundTruthEntry


async def main():
    """Evaluation example."""
    print("RAG Tools - Evaluation Example")
    print("=" * 50)

    # Create manager
    manager = await create_manager()
    print("Manager initialized")

    # Add some test tools
    print("\nAdding test tools...")
    server = await manager.add_server(
        url="http://test.com/mcp",
        name="test-server",
        sync_tools=False
    )

    test_tools = [
        ("search_papers", "Search for academic papers"),
        ("download_papers", "Download paper PDFs"),
        ("get_author_info", "Get information about an author"),
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

    print(f"Added {len(test_tools)} test tools")

    # Create evaluation suite
    print("\nCreating evaluation suite...")
    suite = EvaluationSuite(
        name="test-evaluation",
        description="Evaluation for test tools",
        queries=[
            GroundTruthEntry(
                query="how to search for papers",
                relevant_tools=[f"{server.server_id}:search_papers"],
            ),
            GroundTruthEntry(
                query="download PDF documents",
                relevant_tools=[f"{server.server_id}:download_papers"],
            ),
            GroundTruthEntry(
                query="find author details",
                relevant_tools=[f"{server.server_id}:get_author_info"],
            ),
            GroundTruthEntry(
                query="academic search",
                relevant_tools=[
                    f"{server.server_id}:search_papers",
                    f"{server.server_id}:find_citations",
                ],
            ),
        ]
    )

    print(f"Created suite with {len(suite.queries)} queries")

    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = manager.create_evaluator()
    report = await evaluator.evaluate_suite(suite)

    # Print results
    evaluator.print_report(report)

    # Export report
    print("\nExporting report...")
    evaluator.export_report(report, "/tmp/evaluation_report.json")
    print("Report saved to /tmp/evaluation_report.json")

    # Cleanup
    print("\nCleaning up...")
    await manager.remove_server(server.server_id)
    await manager.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
