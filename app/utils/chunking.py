"""
Document chunking strategies for text splitting.
Implements multiple strategies for different use cases.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize fixed-size chunking strategy.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks with overlap."""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence ending within last 100 chars
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                    "strategy": "fixed_size"
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
            
            start = end - self.overlap if end < text_length else text_length
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on paragraphs and sentences."""
    
    def __init__(
        self, 
        max_chunk_size: int = 1000, 
        min_chunk_size: int = 100,
        respect_paragraphs: bool = True
    ):
        """
        Initialize semantic chunking strategy.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            respect_paragraphs: Whether to respect paragraph boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_paragraphs = respect_paragraphs
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into semantic chunks based on structure."""
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # Split into paragraphs
        if self.respect_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text)
        else:
            paragraphs = [text]
        
        current_chunk = []
        current_size = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_size = len(paragraph)
            
            # If single paragraph exceeds max size, split by sentences
            if para_size > self.max_chunk_size:
                # Flush current chunk if any
                if current_chunk:
                    self._add_chunk(chunks, current_chunk, metadata)
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    sent_size = len(sentence)
                    
                    if temp_size + sent_size > self.max_chunk_size and temp_chunk:
                        self._add_chunk(chunks, temp_chunk, metadata)
                        temp_chunk = [sentence]
                        temp_size = sent_size
                    else:
                        temp_chunk.append(sentence)
                        temp_size += sent_size
                
                if temp_chunk:
                    self._add_chunk(chunks, temp_chunk, metadata)
            
            # If adding paragraph would exceed max size, create new chunk
            elif current_size + para_size > self.max_chunk_size and current_chunk:
                self._add_chunk(chunks, current_chunk, metadata)
                current_chunk = [paragraph]
                current_size = para_size
            
            # Add paragraph to current chunk
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            self._add_chunk(chunks, current_chunk, metadata)
        
        return chunks
    
    def _add_chunk(
        self, 
        chunks: List[Dict[str, Any]], 
        content_parts: List[str], 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add a chunk to the chunks list."""
        content = '\n\n'.join(content_parts).strip()
        
        if len(content) < self.min_chunk_size and chunks:
            # Merge with previous chunk if too small
            chunks[-1]["content"] += '\n\n' + content
        else:
            chunk_metadata = {
                "chunk_index": len(chunks),
                "strategy": "semantic",
                "paragraph_count": len(content_parts)
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            chunks.append({
                "content": content,
                "metadata": chunk_metadata
            })


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    _strategies = {
        "fixed_size": FixedSizeChunking,
        "semantic": SemanticChunking
    }
    
    @classmethod
    def create_strategy(
        cls, 
        strategy_name: str, 
        **kwargs
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy by name.
        
        Args:
            strategy_name: Name of the strategy ("fixed_size" or "semantic")
            **kwargs: Strategy-specific parameters
            
        Returns:
            ChunkingStrategy instance
            
        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_class = cls._strategies.get(strategy_name)
        
        if not strategy_class:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        
        return strategy_class(**kwargs)
    
    @classmethod
    def available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())