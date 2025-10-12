"""
Qdrant Hybrid Search with LlamaIndex Integration
Combines dense and sparse vectors for comprehensive semantic search.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
from pathlib import Path

try:
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logging.warning("LlamaIndex not available. Install with: pip install llama-index llama-index-vector-stores-qdrant")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")

from .qdrant_core import QdrantCore
from .base_agent_config import AgentConfig


class QdrantHybridSearchLlamaIndex:
    """
    Advanced hybrid search combining Qdrant vector storage with LlamaIndex query capabilities.
    
    Features:
    - Dense vector search (semantic similarity)
    - Sparse vector search (keyword matching) 
    - Hybrid scoring and ranking
    - Integration with agent configurations
    - Document indexing and retrieval
    """
    
    def __init__(self,
                 qdrant_core: Optional[QdrantCore] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 sparse_model: str = "prithvida/Splade_PP_en_v1",
                 collection_name: str = "hybrid_search",
                 vector_size: int = 384):
        """
        Initialize hybrid search system.
        
        Args:
            qdrant_core: Existing QdrantCore instance or None to create new
            embedding_model: Dense embedding model name
            sparse_model: Sparse embedding model name  
            collection_name: Qdrant collection for hybrid search
            vector_size: Vector dimensionality
        """
        if not LLAMAINDEX_AVAILABLE or not QDRANT_AVAILABLE:
            raise ImportError("Required packages not available. Install with: pip install llama-index llama-index-vector-stores-qdrant qdrant-client")
        
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize or use existing Qdrant core
        if qdrant_core is None:
            # Create with data directory path
            self.qdrant_core = QdrantCore(persist_path="hybrid_search")
        else:
            self.qdrant_core = qdrant_core
        
        # Initialize embedding models
        try:
            self.dense_embedder = HuggingFaceEmbedding(model_name=embedding_model)
            self.logger.info(f"Loaded dense embedding model: {embedding_model}")
        except Exception as e:
            self.logger.warning(f"Failed to load dense embedding model: {e}")
            self.dense_embedder = None
        
        # TODO: Implement sparse embedding model loading
        self.sparse_model_name = sparse_model
        self.sparse_embedder = None
        
        # Configure LlamaIndex settings
        if self.dense_embedder:
            Settings.embed_model = self.dense_embedder
        
        # Initialize vector store and index
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize Qdrant vector store and LlamaIndex integration."""
        try:
            # Create collection if it doesn't exist
            self.qdrant_core.create_collection(
                collection_name=self.collection_name,
                vector_size=self.vector_size,
                distance=Distance.COSINE
            )
            
            # Create vector store
            self.vector_store = QdrantVectorStore(
                collection_name=self.collection_name,
                client=self.qdrant_core.client
            )
            
            # Create index
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            self.logger.info(f"Initialized vector store and index for collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None
            self.index = None
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the hybrid search index."""
        try:
            if not self.index:
                self.logger.error("Index not initialized")
                return False
            
            # Add documents to index (LlamaIndex handles embedding generation)
            for doc in documents:
                self.index.insert(doc)
            
            self.logger.info(f"Added {len(documents)} documents to hybrid search index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    def add_text_documents(self, 
                          texts: List[str], 
                          metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Add text documents with optional metadata."""
        try:
            if metadata is None:
                metadata = [{}] * len(texts)
            
            documents = [
                Document(text=text, metadata=meta)
                for text, meta in zip(texts, metadata)
            ]
            
            return self.add_documents(documents)
            
        except Exception as e:
            self.logger.error(f"Failed to add text documents: {e}")
            return False
    
    def create_query_engine(self, 
                           similarity_top_k: int = 10,
                           response_mode: str = "compact") -> Optional[BaseQueryEngine]:
        """Create a query engine for the hybrid search index."""
        try:
            if not self.index:
                self.logger.error("Index not initialized")
                return None
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode=response_mode
            )
            
            self.logger.info(f"Created query engine with top_k={similarity_top_k}")
            return query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create query engine: {e}")
            return None
    
    def hybrid_search(self, 
                     query: str,
                     top_k: int = 10,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Args:
            query: Search query
            top_k: Number of results to return
            dense_weight: Weight for dense vector search
            sparse_weight: Weight for sparse vector search
            
        Returns:
            List of search results with hybrid scores
        """
        try:
            if not self.dense_embedder:
                self.logger.error("Dense embedder not available")
                return []
            
            # Generate dense embedding for query
            dense_vector = self.dense_embedder.get_text_embedding(query)
            
            # Perform dense vector search
            dense_results = self.qdrant_core.search_vectors(
                collection_name=self.collection_name,
                query_vector=dense_vector,
                limit=top_k * 2  # Get more results for hybrid ranking
            )
            
            # TODO: Implement sparse vector search when sparse embedder is available
            # For now, return dense results with adjusted scoring
            hybrid_results = []
            for result in dense_results[:top_k]:
                hybrid_score = result["score"] * dense_weight
                hybrid_results.append({
                    "id": result["id"],
                    "score": hybrid_score,
                    "dense_score": result["score"],
                    "sparse_score": 0.0,  # Placeholder
                    "payload": result["payload"],
                    "query": query
                })
            
            self.logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    def semantic_query(self, 
                      query: str, 
                      top_k: int = 5,
                      response_synthesis: bool = True) -> Dict[str, Any]:
        """
        Perform semantic query with response synthesis.
        
        Args:
            query: Natural language query
            top_k: Number of source documents to consider
            response_synthesis: Whether to synthesize a response
            
        Returns:
            Query results with optional synthesized response
        """
        try:
            query_engine = self.create_query_engine(similarity_top_k=top_k)
            if not query_engine:
                return {"error": "Failed to create query engine"}
            
            if response_synthesis:
                # Use query engine for response synthesis
                response = query_engine.query(query)
                return {
                    "query": query,
                    "response": str(response),
                    "source_nodes": [
                        {
                            "text": node.node.text,
                            "score": node.score,
                            "metadata": node.node.metadata
                        }
                        for node in response.source_nodes
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Just return search results
                results = self.hybrid_search(query, top_k=top_k)
                return {
                    "query": query,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Semantic query failed: {e}")
            return {"error": str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid search collection."""
        try:
            info = self.qdrant_core.get_collection_info(self.collection_name)
            if info:
                return {
                    "collection_name": self.collection_name,
                    "total_documents": info["points_count"],
                    "vector_size": info["config"]["vector_size"],
                    "distance_metric": info["config"]["distance"],
                    "indexed_vectors": info["indexed_vectors_count"],
                    "segments": info["segments_count"]
                }
            else:
                return {"error": "Collection not found"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the hybrid search collection."""
        return self.qdrant_core.delete_collection(self.collection_name)


# Factory function for easy initialization
def create_hybrid_search_system(persist_path: str = "hybrid_search",
                               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                               collection_name: str = "multimodal_hybrid") -> Optional[QdrantHybridSearchLlamaIndex]:
    """
    Factory function to create a hybrid search system with proper data directory paths.
    
    Args:
        persist_path: Path within data directory for Qdrant storage
        embedding_model: Dense embedding model to use
        collection_name: Name for the search collection
        
    Returns:
        Configured hybrid search system or None if setup fails
    """
    try:
        # Create QdrantCore with data directory path
        qdrant_core = QdrantCore(persist_path=persist_path)
        
        # Create hybrid search system
        hybrid_search = QdrantHybridSearchLlamaIndex(
            qdrant_core=qdrant_core,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        return hybrid_search
        
    except Exception as e:
        logging.error(f"Failed to create hybrid search system: {e}")
        return None