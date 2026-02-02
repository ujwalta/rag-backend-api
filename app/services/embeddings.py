"""
Embeddings service using local sentence-transformers model.
Provides text embedding generation for document chunks.
"""
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings using local models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing (default from config)
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        batch_size = batch_size or settings.BATCH_SIZE
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(valid_texts) > 100
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        import numpy as np
        
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Global embedding service instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service