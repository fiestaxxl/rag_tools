"""
RAG metrics for evaluating retrieval quality.
"""
from typing import List, Dict, Any, Set, Optional
import numpy as np
from dataclasses import dataclass

from rag_tools.storage.models import RetrievalResult, EvaluationResult


@dataclass
class GroundTruthEntry:
    """A ground truth entry for evaluation."""
    query: str
    relevant_tools: List[str]  # List of relevant tool IDs


class MetricsCalculator:
    """Calculate various RAG metrics for retrieval evaluation."""

    @staticmethod
    def recall_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_ids:
            return 0.0

        top_k_results = results[:k]
        retrieved_ids = {r.tool_id for r in top_k_results}
        relevant_set = set(relevant_ids)

        if not relevant_set:
            return 0.0

        return len(retrieved_ids & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position

        Returns:
            Precision@K score (0-1)
        """
        if k == 0:
            return 0.0

        top_k_results = results[:k]
        retrieved_ids = {r.tool_id for r in top_k_results}
        relevant_set = set(relevant_ids)

        if not retrieved_ids:
            return 0.0

        return len(retrieved_ids & relevant_set) / min(k, len(retrieved_ids))

    @staticmethod
    def mean_reciprocal_rank(
        results: List[RetrievalResult],
        relevant_ids: List[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs

        Returns:
            MRR score (0-1)
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        for i, result in enumerate(results):
            if result.tool_id in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def average_precision(
        results: List[RetrievalResult],
        relevant_ids: List[str],
    ) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs

        Returns:
            AP score (0-1)
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)
        num_relevant = len(relevant_set)

        if num_relevant == 0:
            return 0.0

        precisions = []
        num_hits = 0

        for i, result in enumerate(results):
            if result.tool_id in relevant_set:
                num_hits += 1
                precisions.append(num_hits / (i + 1))

        if not precisions:
            return 0.0

        return sum(precisions) / num_relevant

    @staticmethod
    def ndcg_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position
            relevance_scores: Optional dict of tool_id -> relevance score

        Returns:
            NDCG@K score (0-1)
        """
        if not relevant_ids:
            return 0.0

        relevance_scores = relevance_scores or {}
        relevant_set = set(relevant_ids)

        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            rel = relevance_scores.get(result.tool_id, 1.0 if result.tool_id in relevant_set else 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because position starts at 1

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        num_relevant = min(k, len(relevant_ids))
        for i in range(num_relevant):
            rel = relevance_scores.get(relevant_ids[i], 1.0)
            idcg += rel / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Hit Rate@K (whether any relevant item is in top K).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position

        Returns:
            Hit rate (0-1)
        """
        if not relevant_ids:
            return 0.0

        top_k_results = results[:k]
        retrieved_ids = {r.tool_id for r in top_k_results}
        relevant_set = set(relevant_ids)

        return 1.0 if retrieved_ids & relevant_set else 0.0

    @staticmethod
    def mrr_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate MRR@K (only considering first K results).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position

        Returns:
            MRR@K score (0-1)
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        for i, result in enumerate(results[:k]):
            if result.tool_id in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def coverage_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Coverage@K (what fraction of relevant items are retrieved in top K).

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k: Cutoff position

        Returns:
            Coverage score (0-1)
        """
        if not relevant_ids:
            return 0.0

        top_k_results = results[:k]
        retrieved_ids = {r.tool_id for r in top_k_results}
        relevant_set = set(relevant_ids)

        return len(retrieved_ids & relevant_set) / len(relevant_set)

    @staticmethod
    def diversity_at_k(
        results: List[RetrievalResult],
        k: int,
    ) -> float:
        """
        Calculate diversity as unique server coverage in top K.

        Args:
            results: Retrieved results
            k: Cutoff position

        Returns:
            Diversity score (0-1)
        """
        if k == 0:
            return 0.0

        top_k_results = results[:k]
        unique_servers = {r.server_id for r in top_k_results}

        return len(unique_servers) / k

    @staticmethod
    def calculate_all_metrics(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.

        Args:
            results: Retrieved results
            relevant_ids: List of relevant tool IDs
            k_values: List of K values to evaluate
            relevance_scores: Optional relevance scores for NDCG

        Returns:
            Dict of metric_name -> score
        """
        metrics = {}

        for k in k_values:
            metrics[f"recall@{k}"] = MetricsCalculator.recall_at_k(results, relevant_ids, k)
            metrics[f"precision@{k}"] = MetricsCalculator.precision_at_k(results, relevant_ids, k)
            metrics[f"ndcg@{k}"] = MetricsCalculator.ndcg_at_k(results, relevant_ids, k, relevance_scores)
            metrics[f"hit_rate@{k}"] = MetricsCalculator.hit_rate_at_k(results, relevant_ids, k)

        metrics["mrr"] = MetricsCalculator.mean_reciprocal_rank(results, relevant_ids)
        metrics["map"] = MetricsCalculator.average_precision(results, relevant_ids)
        metrics["diversity"] = MetricsCalculator.diversity_at_k(results, 10)

        return metrics


class MetricsAggregator:
    """Aggregates metrics across multiple queries."""

    def __init__(self):
        self.metrics_list: List[Dict[str, float]] = []

    def add(self, metrics: Dict[str, float]) -> None:
        """Add metrics for a single query."""
        self.metrics_list.append(metrics)

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across all queries.

        Returns:
            Dict with mean, std, and min/max for each metric
        """
        if not self.metrics_list:
            return {}

        all_metrics = {}

        # Collect all metric names
        for m in self.metrics_list:
            all_metrics.update(m)

        # Aggregate each metric
        aggregated = {}
        for metric_name in all_metrics.keys():
            values = [m.get(metric_name, 0) for m in self.metrics_list]

            aggregated[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

        return aggregated

    def reset(self) -> None:
        """Reset the aggregator."""
        self.metrics_list.clear()
