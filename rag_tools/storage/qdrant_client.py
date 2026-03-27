"""
Qdrant vector database client for tool embeddings.
"""
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from rag_tools.config.settings import settings, QdrantSettings


class QdrantClientWrapper:
    """Wrapper for Qdrant vector database operations."""

    def __init__(self, config: Optional[QdrantSettings] = None):
        self.config = config or settings.qdrant
        self._client = None
        self._embedding_dim = None

    def connect(self) -> None:
        """Connect to Qdrant server."""
        self._client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            prefer_grpc=self.config.prefer_grpc,
        )

    def close(self) -> None:
        """Close the connection."""
        self._client = None

    def set_embedding_dim(self, dim: int) -> None:
        """Set the embedding dimension."""
        self._embedding_dim = dim

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        if self._client is None:
            raise RuntimeError("Qdrant not connected. Call connect() first.")
        return self._client

    # Collection management
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Create a new collection."""
        try:
            self.client.get_collection(collection_name)
            return False  # Collection exists
        except (UnexpectedResponse, Exception):
            pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        return True

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except UnexpectedResponse:
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(collection_name)
            return True
        except UnexpectedResponse:
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        try:
            info = self.client.get_collection(collection_name)

            return {
                "name": collection_name,                     # use the argument
                "vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
            }
        except UnexpectedResponse:
            return None

    def list_collections(self) -> List[str]:
        """List all collections."""
        collections = self.client.get_collections()
        return [c for c in collections.collections]

    # Point operations
    def upsert_points(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """Insert or update points in a collection."""
        if not vectors or not payloads:
            return False

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        return True

    def delete_points(
        self,
        collection_name: str,
        point_ids: List[str],
    ) -> bool:
        """Delete points from a collection."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=point_ids,
            ),
        )
        return True

    def delete_by_filter(
        self,
        collection_name: str,
        filter_conditions: List[FieldCondition],
    ) -> int:
        """Delete points by filter condition."""
        result = self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=filter_conditions,
                )
            ),
        )

        return result.status in ('acknowledged', 'completed') or 0

    # Search operations
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[List[FieldCondition]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        search_params = models.SearchParams(hnsw_ef=128)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            search_params=search_params,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def search_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[List[FieldCondition]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Batch search for similar vectors."""
        search_params = models.SearchParams(hnsw_ef=128)

        all_results = self.client.search_batch(
            collection_name=collection_name,
            requests=[
                models.SearchRequest(
                    vector=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    filter=models.Filter(must=filter_conditions) if filter_conditions else None,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                    search_params=search_params,
                )
                for vector in query_vectors
            ],
        )

        return [
            [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in results.points
            ]
            for results in all_results
        ]

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        filter_conditions: Optional[List[FieldCondition]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Scroll through points in a collection."""
        results, next_page_offset = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            scroll_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        points = [
            {
                "id": hit.id,
                "payload": hit.payload,
                "vector": hit.vector if with_vectors else None,
            }
            for hit in results.points
        ]

        return points, next_page_offset

    def retrieve(
        self,
        collection_name: str,
        point_ids: List[str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by ID."""
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        return [
            {
                "id": hit.id,
                "payload": hit.payload,
                "vector": hit.vector if with_vectors else None,
            }
            for hit in results
        ]

    # Indexing operations
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Optional[Any] = None,
    ) -> bool:
        """Create a payload index for faster filtering."""
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema or models.PayloadSchemaType.KEYWORD,
            )
            return True
        except UnexpectedResponse:
            return False

    # Count operations
    def count(
        self,
        collection_name: str,
        filter_conditions: Optional[List[FieldCondition]] = None,
    ) -> int:
        """Count points in a collection."""
        result = self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
        )
        return result.count

    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health status."""
        info = self.client.get_collections()
        return {
            "status": "ok",
            "collections": len(info.collections),
        }
