"""
FAISS Vector Store: Manages semantic document retrieval with FAISS.
Handles embedding storage, indexing, and similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import faiss

from vector_store.embeddings import get_embedding_manager
from config.settings import get_settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for financial documents.
    
    Features:
    - Fast similarity search (sub-100ms for 1000+ docs)
    - Minimal memory footprint
    - CPU-only operation
    - Persistent storage
    """
    
    def __init__(self):
        """Initialize FAISS vector store."""
        logger.info("Initializing FAISSVectorStore...")
        self.settings = get_settings()
        self.embedding_manager = get_embedding_manager()
        self.index = None
        self.documents = []
        self.document_index = {}
        
        # Load existing index or create new
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing FAISS index or create new one."""
        index_path = Path(self.settings.vector_db_path) / "faiss_index.bin"
        docs_path = Path(self.settings.vector_db_path) / "documents.json"
        
        if index_path.exists() and docs_path.exists():
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(str(index_path))
            
            with open(docs_path, 'r') as f:
                self.documents = json.load(f)
            
            # Rebuild document index
            for i, doc in enumerate(self.documents):
                self.document_index[i] = doc
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        else:
            logger.info("Creating new FAISS index...")
            dimension = self.settings.embedding_dimension
            self.index = faiss.IndexFlatL2(dimension)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents with 'content' field
        """
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Extract texts
        texts = [doc.get('content', '') for doc in documents]
        
        # Embed
        embeddings = self.embedding_manager.batch_embed(texts)
        
        # Add to index
        start_idx = len(self.documents)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        for i, doc in enumerate(documents):
            doc_id = start_idx + i
            self.documents.append(doc)
            self.document_index[doc_id] = doc
        
        # Save
        self.save_index()
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector (dimension,)
            top_k: Number of top results
            filters: Optional filters (e.g., {'company': 'AAPL'})
            
        Returns:
            List[Dict]: Top-K similar documents with scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert numpy scalar to native Python integer
            idx = int(idx)
            
            # Check for invalid index (FAISS returns -1 for invalid results)
            if idx < 0:
                continue
            
            doc = self.document_index.get(idx)
            if not doc:
                continue
            
            # Apply filters if provided
            if filters:
                if not self._matches_filters(doc, filters):
                    continue
            
            results.append({
                **doc,
                'similarity_score': float(1.0 / (1.0 + dist))  # Convert distance to similarity
            })
        
        return results[:top_k]
    
    def _matches_filters(self, document: Dict, filters: Dict) -> bool:
        """Check if document matches filter criteria."""
        for key, value in filters.items():
            if key not in document:
                return False
            
            doc_value = str(document[key]).upper()
            filter_value = str(value).upper()
            
            if doc_value != filter_value:
                return False
        
        return True
    
    def save_index(self):
        """Save index and documents to disk."""
        db_path = Path(self.settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        index_path = db_path / "faiss_index.bin"
        docs_path = db_path / "documents.json"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save documents
        with open(docs_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
    
    def delete(self, document_id: int):
        """Delete document (requires index rebuild)."""
        if document_id in self.document_index:
            del self.document_index[document_id]
            logger.warning("FAISS doesn't support deletion. Consider using Chromadb.")
    
    def info(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.settings.embedding_dimension,
            'storage_type': 'FAISS IndexFlatL2'
        }


# Global instance
_vector_store = None


def get_vector_store() -> FAISSVectorStore:
    """Get or create vector store (singleton)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store


if __name__ == "__main__":
    # Test vector store
    store = get_vector_store()
    
    # Add sample documents
    sample_docs = [
        {
            'title': 'Apple Q3 2024 Earnings',
            'content': 'Apple reported strong Q3 results with revenue of $95.7B...',
            'source': 'Investor Relations',
            'company': 'AAPL'
        },
        {
            'title': 'Tech Sector Analysis',
            'content': 'Technology sector continues to benefit from AI adoption...',
            'source': 'Morgan Stanley',
            'company': 'TECH'
        }
    ]
    
    # Add documents
    store.add_documents(sample_docs)
    
    # Test search
    query_embedding = store.embedding_manager.embed("Apple financial results")
    results = store.search(query_embedding, top_k=2)
    
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result['title']} (similarity: {result['similarity_score']:.2f})")
    
    # Store info
    print(f"\nStore info: {store.info()}")