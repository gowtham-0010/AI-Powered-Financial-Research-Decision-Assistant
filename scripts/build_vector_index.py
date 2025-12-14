"""
Script to build vector index from prepared documents.
Creates FAISS embeddings for semantic search.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_vector_index():
    """Build vector index from prepared documents."""
    logger.info("Building vector index...")
    
    # Import dependencies
    from vector_store.faiss_store import get_vector_store
    
    # Load prepared documents
    docs_path = Path("data/processed/documents.json")
    
    if not docs_path.exists():
        logger.error("Prepared documents not found. Run prepare_data.py first.")
        return False
    
    with open(docs_path, 'r') as f:
        documents = json.load(f)
    
    logger.info(f"Loading {len(documents)} documents into vector store...")
    
    # Get vector store
    vector_store = get_vector_store()
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Verify
    info = vector_store.info()
    logger.info(f"✓ Vector index built successfully")
    logger.info(f"  Total documents: {info['total_documents']}")
    logger.info(f"  Index size: {info['index_size']}")
    logger.info(f"  Dimension: {info['dimension']}")
    logger.info(f"  Storage type: {info['storage_type']}")
    
    return True


def test_vector_search():
    """Test vector search functionality."""
    logger.info("Testing vector search...")
    
    from vector_store.faiss_store import get_vector_store
    
    vector_store = get_vector_store()
    
    # Test queries
    test_queries = [
        "Apple financial results",
        "technology stock analysis",
        "earnings report"
    ]
    
    for query in test_queries:
        query_embedding = vector_store.embedding_manager.embed(query)
        results = vector_store.search(query_embedding, top_k=3)
        
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"  Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            logger.info(
                f"    [{i}] {result.get('title', 'Unknown')} "
                f"(sim: {result.get('similarity_score', 0):.3f})"
            )


if __name__ == "__main__":
    logger.info("Starting vector index build process...")
    
    # Build index
    success = build_vector_index()
    
    if success:
        # Test search
        test_vector_search()
        logger.info("✅ Vector index build complete!")
    else:
        logger.error("❌ Vector index build failed")
