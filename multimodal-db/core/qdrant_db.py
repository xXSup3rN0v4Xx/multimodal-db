"""
Minimal Qdrant Vector Database
Optimized for essential vector operations only.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantDB:
    """Minimal, fast Qdrant vector database."""
    
    def __init__(self, persist_path: str = "vectors"):
        self.persist_path = Path("data") / "qdrant" / persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.available = QDRANT_AVAILABLE
        
        if self.available:
            self.client = QdrantClient(path=str(self.persist_path))
        else:
            self.client = None
    
    def create_collection(self, name: str, vector_size: int = 384) -> bool:
        """Create vector collection."""
        if not self.available:
            return False
        
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            return True
        except:
            return False  # Collection might already exist
    
    def store_vector(self, collection: str, vector: List[float], 
                    metadata: Dict[str, Any]) -> str:
        """Store vector with metadata."""
        if not self.available:
            return "mock_id"
        
        point_id = str(uuid.uuid4())
        
        try:
            self.client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata
                )]
            )
            return point_id
        except:
            return ""
    
    def search_vectors(self, collection: str, query_vector: List[float], 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.available:
            return []
        
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit
            )
            
            return [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
        except:
            return []
    
    def list_collections(self) -> List[str]:
        """List available collections."""
        if not self.available:
            return []
        
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except:
            return []