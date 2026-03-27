"""
Evaluator for testing RAG retrieval quality.
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from rag_tools.config.settings import settings
from rag_tools.storage.models import RetrievalResult
from rag_tools.evaluation.metrics import MetricsCalculator, MetricsAggregator, GroundTruthEntry
from rag_tools.retrieval.pipeline import ToolRetrievalPipeline, PipelineConfig, PipelineResult


@dataclass
class EvaluationSuite:
    """A collection of test queries with ground truth."""
    name: str
    description: str
    queries: List[GroundTruthEntry]


@dataclass
class QueryEvaluation:
    """Results for a single query evaluation."""
    query: str
    predicted_tools: List[str]
    relevant_tools: List[str]
    metrics: Dict[str, float]
    retrieval_result: PipelineResult


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    suite_name: str
    total_queries: int
    aggregated_metrics: Dict[str, Dict[str, float]]
    per_query_results: List[QueryEvaluation]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """Evaluates RAG pipeline performance."""

    def __init__(
        self,
        pipeline: ToolRetrievalPipeline,
        k_values: Optional[List[int]] = None,
    ):
        self.pipeline = pipeline
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics_calc = MetricsCalculator()

    async def evaluate_suite(
        self,
        suite: EvaluationSuite,
        config: Optional[PipelineConfig] = None,
    ) -> EvaluationReport:
        """
        Evaluate a test suite.

        Args:
            suite: Test suite with queries and ground truth
            config: Pipeline configuration

        Returns:
            EvaluationReport with results
        """
        aggregator = MetricsAggregator()
        per_query_results = []

        for entry in suite.queries:
            query_eval = await self._evaluate_single_query(entry, config)
            aggregator.add(query_eval.metrics)
            per_query_results.append(query_eval)

        return EvaluationReport(
            suite_name=suite.name,
            total_queries=len(suite.queries),
            aggregated_metrics=aggregator.aggregate(),
            per_query_results=per_query_results,
            metadata={
                "k_values": self.k_values,
                "config": config.__dict__ if config else {},
            },
        )

    async def _evaluate_single_query(
        self,
        entry: GroundTruthEntry,
        config: Optional[PipelineConfig],
    ) -> QueryEvaluation:
        """Evaluate a single query."""
        # Run retrieval
        result = await self.pipeline.retrieve(entry.query, config)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            result.results,
            entry.relevant_tools,
            k_values=self.k_values,
        )

        return QueryEvaluation(
            query=entry.query,
            predicted_tools=[r.tool_id for r in result.results],
            relevant_tools=entry.relevant_tools,
            metrics=metrics,
            retrieval_result=result,
        )

    async def evaluate_custom(
        self,
        query: str,
        relevant_tools: List[str],
        config: Optional[PipelineConfig] = None,
    ) -> QueryEvaluation:
        """
        Evaluate a custom query.

        Args:
            query: Search query
            relevant_tools: List of relevant tool IDs
            config: Pipeline configuration

        Returns:
            QueryEvaluation with results
        """
        entry = GroundTruthEntry(
            query=query,
            relevant_tools=relevant_tools,
        )

        return await self._evaluate_single_query(entry, config)

    def load_suite_from_file(self, file_path: Path) -> EvaluationSuite:
        """Load evaluation suite from JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        queries = [
            GroundTruthEntry(
                query=q["query"],
                relevant_tools=q.get("relevant_tools", q.get("relevant", [])),
            )
            for q in data.get("queries", data.get("test_cases", []))
        ]

        return EvaluationSuite(
            name=data.get("name", file_path.stem),
            description=data.get("description", ""),
            queries=queries,
        )

    def save_suite_to_file(self, suite: EvaluationSuite, file_path: Path) -> None:
        """Save evaluation suite to JSON file."""
        data = {
            "name": suite.name,
            "description": suite.description,
            "queries": [
                {
                    "query": q.query,
                    "relevant_tools": q.relevant_tools,
                }
                for q in suite.queries
            ],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def print_report(self, report: EvaluationReport) -> None:
        """Print evaluation report in a readable format."""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {report.suite_name}")
        print(f"{'='*60}")
        print(f"Total queries: {report.total_queries}")
        print()

        # Print aggregated metrics
        print("Aggregated Metrics:")
        print("-" * 40)

        # Group by metric type
        recall_metrics = {k: v for k, v in report.aggregated_metrics.items() if "recall@" in k}
        precision_metrics = {k: v for k, v in report.aggregated_metrics.items() if "precision@" in k}
        ndcg_metrics = {k: v for k, v in report.aggregated_metrics.items() if "ndcg@" in k}
        other_metrics = {k: v for k, v in report.aggregated_metrics.items()
                        if "recall@" not in k and "precision@" not in k and "ndcg@" not in k}

        for group_name, metrics in [
            ("Recall", recall_metrics),
            ("Precision", precision_metrics),
            ("NDCG", ndcg_metrics),
            ("Other", other_metrics),
        ]:
            if metrics:
                print(f"\n{group_name}:")
                for name, values in sorted(metrics.items()):
                    print(f"  {name:20s}: {values['mean']:.4f} ± {values['std']:.4f} "
                          f"(min: {values['min']:.4f}, max: {values['max']:.4f})")

        print()

    def export_report(self, report: EvaluationReport, file_path: Path) -> None:
        """Export evaluation report to JSON."""
        data = {
            "suite_name": report.suite_name,
            "total_queries": report.total_queries,
            "aggregated_metrics": report.aggregated_metrics,
            "per_query_results": [
                {
                    "query": q.query,
                    "predicted_tools": q.predicted_tools,
                    "relevant_tools": q.relevant_tools,
                    "metrics": q.metrics,
                }
                for q in report.per_query_results
            ],
            "metadata": report.metadata,
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


class SyntheticGroundTruth:
    """Generates synthetic ground truth for testing."""

    @staticmethod
    def generate_from_tool_descriptions(
        tools: List[Dict[str, Any]],
        queries_per_tool: int = 3,
    ) -> List[GroundTruthEntry]:
        """
        Generate synthetic queries based on tool descriptions.

        Args:
            tools: List of tool dicts with name and description
            queries_per_tool: Number of queries to generate per tool

        Returns:
            List of ground truth entries
        """
        entries = []

        for tool in tools:
            tool_id = tool.get("tool_id", tool.get("name", ""))
            name = tool.get("name", "")
            description = tool.get("description", "")

            # Generate query based on description
            if description:
                queries = SyntheticGroundTruth._extract_query_candidates(description)

                for query in queries[:queries_per_tool]:
                    entries.append(
                        GroundTruthEntry(
                            query=query,
                            relevant_tools=[tool_id],
                        )
                    )

        return entries

    @staticmethod
    def _extract_query_candidates(description: str) -> List[str]:
        """Extract potential queries from description."""
        # Simple heuristic: split by common patterns
        queries = []

        # Use the name
        queries.append(f"how to use {name}" if "name" in locals() else description[:50])

        # Extract verbs and nouns
        words = description.lower().split()
        if len(words) > 5:
            queries.append(" ".join(words[:8]))

        # Truncated description
        queries.append(description[:80])

        return list(set(queries))


def create_default_suite() -> EvaluationSuite:
    """Create a default evaluation suite for testing."""
    return EvaluationSuite(
        name="default",
        description="Default evaluation suite",
        queries=[
            GroundTruthEntry(
                query="search for academic papers",
                relevant_tools=["openalex:search_papers"],
            ),
            GroundTruthEntry(
                query="download PDFs from search results",
                relevant_tools=["openalex:download_papers_from_search"],
            ),
            GroundTruthEntry(
                query="find author information",
                relevant_tools=["openalex:search_entity"],
            ),
            GroundTruthEntry(
                query="search entity in database",
                relevant_tools=["openalex:search_entity"],
            ),
        ],
    )
