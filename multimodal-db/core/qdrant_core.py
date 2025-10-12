"""
Qdrant Vector Database Core Implementation
Handles vector storage, embedding generation, and semantic search functionality.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import uuid
from datetime import datetime
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")
    
    # Create mock classes to prevent NameError
    class Distance:
        COSINE = "cosine"
        DOT = "dot"
        EUCLID = "euclid"
    
    class VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance
    
    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

from .base_agent_config import AgentConfig


class QdrantCore:
    """
    Core Qdrant vector database implementation for multimodal-db.
    
    Handles:
    - Vector storage and retrieval 
    - Collection management
    - Semantic search
    - Integration with agent configurations
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 use_memory: bool = False,
                 persist_path: str = None):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host (for server mode)
            port: Qdrant server port (for server mode)  
            use_memory: Use in-memory storage (for development/testing)
            persist_path: Local persistence path (for local mode)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize client based on mode
        if use_memory:
            self.client = QdrantClient(":memory:")
            self.logger.info("Initialized Qdrant in memory mode")
        elif persist_path:
            # Ensure path is in data directory
            data_path = Path("data") / "qdrant_db" / persist_path
            data_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(data_path))
            self.logger.info(f"Initialized Qdrant with local persistence: {data_path}")
        else:
            # Server mode
            self.client = QdrantClient(host=host, port=port)
            self.logger.info(f"Initialized Qdrant client for server at {host}:{port}")
        
        # Standard collections for multimodal-db
        self.collections = {
            "knowledge_documents": {
                "vector_size": 384,  # Default for all-MiniLM-L6-v2
                "distance": Distance.COSINE
            },
            "agent_conversations": {
                "vector_size": 384,
                "distance": Distance.COSINE
            },
            "research_data": {
                "vector_size": 384,
                "distance": Distance.COSINE
            },
            "alignment_documents": {
                "vector_size": 384,
                "distance": Distance.COSINE
            }
        }
    
    def create_collection(self, 
                         collection_name: str, 
                         vector_size: int = 384,
                         distance: Distance = Distance.COSINE,
                         recreate: bool = False) -> bool:
        """Create a new collection in Qdrant."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                if recreate:
                    self.client.delete_collection(collection_name)
                    self.logger.info(f"Deleted existing collection: {collection_name}")
                else:
                    self.logger.info(f"Collection {collection_name} already exists")
                    return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            
            self.logger.info(f"Created collection: {collection_name} (size: {vector_size}, distance: {distance})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def initialize_standard_collections(self) -> Dict[str, bool]:
        """Initialize all standard collections for multimodal-db."""
        results = {}
        
        for collection_name, config in self.collections.items():
            success = self.create_collection(
                collection_name=collection_name,
                vector_size=config["vector_size"],
                distance=config["distance"]
            )
            results[collection_name] = success
        
        return results
    
    def add_vectors(self, 
                   collection_name: str,
                   vectors: List[List[float]],
                   payloads: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> bool:
        """Add vectors to a collection."""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            self.logger.info(f"Added {len(points)} vectors to collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to {collection_name}: {e}")
            return False
    
    def search_vectors(self, 
                      collection_name: str,
                      query_vector: List[float],
                      limit: int = 10,
                      filter_conditions: Optional[Dict[str, Any]] = None,
                      score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a collection."""
        try:
            search_filter = None
            if filter_conditions:
                # Convert filter conditions to Qdrant filter format
                # This is a basic implementation - can be extended for complex filters
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold
            )
            
            # Convert results to standard format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            self.logger.info(f"Found {len(formatted_results)} results in {collection_name}")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed in {collection_name}: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.distance
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        try:
            collections = self.list_collections()
            return {
                "status": "healthy",
                "collections_count": len(collections),
                "collections": collections,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }