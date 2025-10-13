"""
Enhanced Qdrant Integration
Optimized for embeddings and multimodal vector search with LlamaIndex support.
"""
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import uuid
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Create mock Distance class for type hints
    class Distance:
        COSINE = "Cosine"
        DOT = "Dot"
        EUCLID = "Euclid"

try:
    from .agent_config import MediaType
except ImportError:
    from agent_config import MediaType


class QdrantVectorDB:
    """
    Enhanced Qdrant vector database for multimodal embeddings.
    Supports text, image, audio, and video embeddings with metadata filtering.
    """
    
    def __init__(self, persist_path: str = "vectors"):
        self.persist_path = Path("data") / "qdrant" / persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.available = QDRANT_AVAILABLE
        
        if self.available:
            self.client = QdrantClient(path=str(self.persist_path))
        else:
            self.client = None
        
        # Standard collections for multimodal data
        self.collections = {
            "agent_knowledge": {
                "description": "Agent knowledge base and conversations",
                "vector_size": 768,  # Nomic embeddings
                "distance": Distance.COSINE
            },
            "text_embeddings": {
                "description": "Text document embeddings",
                "vector_size": 768,
                "distance": Distance.COSINE
            },
            "image_embeddings": {
                "description": "Image feature embeddings",
                "vector_size": 512,  # Future: CLIP embeddings
                "distance": Distance.COSINE
            },
            "audio_embeddings": {
                "description": "Audio feature embeddings", 
                "vector_size": 768,  # Future: audio models
                "distance": Distance.COSINE
            },
            "video_embeddings": {
                "description": "Video feature embeddings",
                "vector_size": 512,  # Future: video models
                "distance": Distance.COSINE
            },
            "multimodal_fusion": {
                "description": "Fused multimodal embeddings",
                "vector_size": 1024,  # Future: fusion models
                "distance": Distance.COSINE
            }
        }
    
    def initialize_collections(self) -> bool:
        """Initialize all standard collections."""
        if not self.available:
            return False
        
        success_count = 0
        for collection_name, config in self.collections.items():
            if self.create_collection(
                collection_name, 
                config["vector_size"], 
                config["distance"]
            ):
                success_count += 1
        
        return success_count == len(self.collections)
    
    def create_collection(self, name: str, vector_size: int, distance: Distance = Distance.COSINE) -> bool:
        """Create vector collection with error handling."""
        if not self.available:
            return False
        
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            return True
        except Exception:
            # Collection might already exist
            return True
    
    def store_embedding(self, collection: str, vector: List[float], 
                       metadata: Dict[str, Any], point_id: str = None) -> str:
        """Store vector embedding with rich metadata."""
        if not self.available:
            return "mock_id"
        
        point_id = point_id or str(uuid.uuid4())
        
        # Ensure metadata has required fields for filtering
        metadata.update({
            "stored_at": metadata.get("stored_at", ""),
            "media_type": metadata.get("media_type", "text"),
            "agent_id": metadata.get("agent_id", ""),
            "content_type": metadata.get("content_type", "document")
        })
        
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
        except Exception:
            return ""
    
    def search_similar(self, collection: str, query_vector: List[float], 
                      limit: int = 10, filters: Dict[str, Any] = None,
                      min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional filtering."""
        if not self.available:
            return []
        
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values (OR condition)
                        for v in value:
                            conditions.append(FieldCondition(key=key, match={"value": v}))
                    else:
                        conditions.append(FieldCondition(key=key, match={"value": value}))
                
                if conditions:
                    filter_conditions = Filter(should=conditions) if len(conditions) > 1 else Filter(must=[conditions[0]])
            
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=min_score
            )
            
            return [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
        
        except Exception:
            return []
    
    def search_by_agent(self, collection: str, query_vector: List[float],
                       agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search vectors for specific agent."""
        return self.search_similar(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            filters={"agent_id": agent_id}
        )
    
    def search_by_media_type(self, collection: str, query_vector: List[float],
                           media_type: MediaType, limit: int = 10) -> List[Dict[str, Any]]:
        """Search vectors by media type."""
        return self.search_similar(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            filters={"media_type": media_type.value}
        )
    
    def store_agent_knowledge(self, agent_id: str, text: str, embedding: List[float],
                            content_type: str = "conversation", metadata: Dict[str, Any] = None) -> str:
        """Store agent knowledge with text embedding."""
        payload = {
            "agent_id": agent_id,
            "text": text,
            "content_type": content_type,
            "media_type": MediaType.TEXT.value,
            "stored_at": str(uuid.uuid4()),
            **(metadata or {})
        }
        
        return self.store_embedding("agent_knowledge", embedding, payload)
    
    def search_agent_knowledge(self, agent_id: str, query_embedding: List[float],
                             content_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search agent's knowledge base."""
        filters = {"agent_id": agent_id}
        if content_types:
            filters["content_type"] = content_types
        
        return self.search_similar("agent_knowledge", query_embedding, limit, filters)
    
    def store_multimodal_content(self, agent_id: str, content: str, 
                               embedding: List[float], media_type: MediaType,
                               media_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Store multimodal content embedding."""
        collection_map = {
            MediaType.TEXT: "text_embeddings",
            MediaType.IMAGE: "image_embeddings", 
            MediaType.AUDIO: "audio_embeddings",
            MediaType.VIDEO: "video_embeddings"
        }
        
        collection = collection_map.get(media_type, "text_embeddings")
        
        payload = {
            "agent_id": agent_id,
            "content": content,
            "media_type": media_type.value,
            "media_id": media_id,
            "stored_at": str(uuid.uuid4()),
            **(metadata or {})
        }
        
        return self.store_embedding(collection, embedding, payload)
    
    def hybrid_search(self, collections: List[str], query_vector: List[float],
                     agent_id: str = None, limit_per_collection: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Perform hybrid search across multiple collections."""
        results = {}
        
        for collection in collections:
            if collection in self.collections:
                filters = {"agent_id": agent_id} if agent_id else None
                search_results = self.search_similar(
                    collection=collection,
                    query_vector=query_vector,
                    limit=limit_per_collection,
                    filters=filters
                )
                results[collection] = search_results
        
        return results
    
    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.available:
            return {}
        
        try:
            info = self.client.get_collection(collection)
            return {
                "name": collection,
                "vectors_count": info.vectors_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
        except Exception:
            return {}
    
    def list_collections(self) -> List[str]:
        """List available collections."""
        if not self.available:
            return list(self.collections.keys())
        
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception:
            return []
    
    def delete_collection(self, collection: str) -> bool:
        """Delete a collection."""
        if not self.available:
            return False
        
        try:
            self.client.delete_collection(collection)
            return True
        except Exception:
            return False
    
    def delete_agent_data(self, agent_id: str) -> int:
        """Delete all vectors for a specific agent."""
        if not self.available:
            return 0
        
        deleted_count = 0
        for collection in self.list_collections():
            try:
                # Delete points with matching agent_id
                self.client.delete(
                    collection_name=collection,
                    points_selector=Filter(
                        must=[FieldCondition(key="agent_id", match={"value": agent_id})]
                    )
                )
                deleted_count += 1
            except Exception:
                continue
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        if not self.available:
            return {"available": False, "collections": list(self.collections.keys())}
        
        stats = {
            "available": True,
            "persist_path": str(self.persist_path),
            "collections": {}
        }
        
        for collection in self.list_collections():
            stats["collections"][collection] = self.get_collection_stats(collection)
        
        return stats