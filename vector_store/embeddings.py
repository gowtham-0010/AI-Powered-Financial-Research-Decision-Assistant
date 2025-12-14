"""
Embedding utilities for converting text to vectors.
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import get_settings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings using Sentence Transformers."""
    
    def __init__(self):
        """Initialize embedding model."""
        settings = get_settings()
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector (dimension,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to embed()")
            return np.zeros(self.dimension, dtype=np.float32)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert multiple texts to embeddings.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings (n, dimension)
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.astype(np.float32)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
        )
        return float(similarity)

# Global instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create embedding manager (singleton)."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
