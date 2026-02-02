"""
Vector database service supporting multiple backends.
Provides abstraction layer for Pinecone, Qdrant, Weaviate, and Milvus.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

from ..core.config import settings


@dataclass
class SearchResult:
    """Container for vector search results."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations."""
    
    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create a new collection/index."""
        pass
    
    @abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Insert or update vectors."""
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> None:
        """Delete vectors by IDs."""
        pass


class QdrantVectorDB(VectorDBInterface):
    """Qdrant vector database implementation."""
    
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.Distance = Distance
        self.VectorParams = VectorParams
    
    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.models import VectorParams, Distance
        
        collections = self.client.get_collections().collections
        
        if not any(col.name == collection_name for col in collections):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
    
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Upsert vectors to Qdrant."""
        from qdrant_client.models import PointStruct
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        points = []
        for i, (vec, text, meta) in enumerate(zip(vectors, texts, metadata)):
            payload = {
                "text": text,
                **meta
            }
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vec,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return ids
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Qdrant for similar vectors."""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_dict
        )
        
        return [
            SearchResult(
                id=str(result.id),
                content=result.payload.get("text", ""),
                score=result.score,
                metadata={k: v for k, v in result.payload.items() if k != "text"}
            )
            for result in results
        ]
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> None:
        """Delete vectors from Qdrant."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids
        )


class PineconeVectorDB(VectorDBInterface):
    """Pinecone vector database implementation."""
    
    def __init__(self):
        import pinecone
        
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        self.pinecone = pinecone
        self.index = None
    
    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create a Pinecone index."""
        if collection_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = self.pinecone.Index(collection_name)
    
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Upsert vectors to Pinecone."""
        if self.index is None or self.index._index_name != collection_name:
            self.index = self.pinecone.Index(collection_name)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        vectors_to_upsert = []
        for i, (vec, text, meta) in enumerate(zip(vectors, texts, metadata)):
            metadata_with_text = {
                "text": text,
                **meta
            }
            vectors_to_upsert.append((ids[i], vec, metadata_with_text))
        
        self.index.upsert(vectors=vectors_to_upsert)
        
        return ids
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone for similar vectors."""
        if self.index is None or self.index._index_name != collection_name:
            self.index = self.pinecone.Index(collection_name)
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        return [
            SearchResult(
                id=match.id,
                content=match.metadata.get("text", ""),
                score=match.score,
                metadata={k: v for k, v in match.metadata.items() if k != "text"}
            )
            for match in results.matches
        ]
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        if self.index is None or self.index._index_name != collection_name:
            self.index = self.pinecone.Index(collection_name)
        
        self.index.delete(ids=ids)


class VectorDBFactory:
    """Factory for creating vector database instances."""
    
    _instances: Dict[str, VectorDBInterface] = {}
    
    @classmethod
    def get_vector_db(cls, db_type: str = None) -> VectorDBInterface:
        """
        Get or create a vector database instance.
        
        Args:
            db_type: Type of vector DB ("qdrant", "pinecone", etc.)
            
        Returns:
            VectorDBInterface instance
        """
        db_type = db_type or settings.VECTOR_DB_TYPE
        
        if db_type not in cls._instances:
            if db_type == "qdrant":
                cls._instances[db_type] = QdrantVectorDB()
            elif db_type == "pinecone":
                cls._instances[db_type] = PineconeVectorDB()
            else:
                raise ValueError(f"Unsupported vector DB type: {db_type}")
        
        return cls._instances[db_type]


def get_collection_name() -> str:
    """Get the collection name from settings."""
    if settings.VECTOR_DB_TYPE == "qdrant":
        return settings.QDRANT_COLLECTION_NAME
    elif settings.VECTOR_DB_TYPE == "pinecone":
        return settings.PINECONE_INDEX_NAME
    else:
        return "documents"