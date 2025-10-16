"""
Graphiti Knowledge Graph Database Integration
Temporal RAG with entity/relationship extraction and graph-based knowledge storage.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import uuid

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.llm_client import LLMConfig, OpenAIClient
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


class GraphitiDB:
    """
    Knowledge graph database using Graphiti for temporal RAG.
    Provides entity extraction, relationship mapping, and time-aware knowledge retrieval.
    """
    
    def __init__(self, db_path: str = "graphiti_db", llm_config: Dict[str, Any] = None):
        """
        Initialize Graphiti database.
        
        Args:
            db_path: Path to store graph database
            llm_config: LLM configuration for entity/relationship extraction
                       (defaults to Ollama if available)
        """
        self.db_path = Path("data") / db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.available = GRAPHITI_AVAILABLE
        
        if not self.available:
            self.client = None
            return
        
        # Initialize LLM config
        if llm_config is None:
            # Default to Ollama for local processing
            llm_config = {
                "model": "qwen2.5-coder:3b",
                "api_base": "http://localhost:11434",
                "api_type": "ollama"
            }
        
        self.llm_config = llm_config
        
        # Initialize Graphiti client
        try:
            # Create LLM client configuration
            client_config = LLMConfig(
                model=llm_config.get("model", "qwen2.5-coder:3b"),
                base_url=llm_config.get("api_base", "http://localhost:11434"),
            )
            
            # Initialize Graphiti
            self.client = Graphiti(
                llm_client=OpenAIClient(config=client_config),
                vector_db_path=str(self.db_path)
            )
        except Exception as e:
            print(f"Warning: Graphiti initialization failed: {e}")
            self.client = None
            self.available = False
    
    async def add_episode(self, 
                         agent_id: str,
                         content: str,
                         episode_type: str = "conversation",
                         source: str = "user",
                         metadata: Dict[str, Any] = None) -> str:
        """
        Add an episode (conversation turn, document, observation) to the graph.
        
        Args:
            agent_id: Agent identifier
            content: Text content of the episode
            episode_type: Type of episode (conversation, document, observation)
            source: Source of the episode (user, system, agent)
            metadata: Additional metadata
        
        Returns:
            Episode ID
        """
        if not self.available or not self.client:
            return ""
        
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Prepare episode data
        episode_data = {
            "episode_id": episode_id,
            "agent_id": agent_id,
            "content": content,
            "episode_type": episode_type,
            "source": source,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            # Add episode to Graphiti
            await self.client.add_episode(
                name=f"{episode_type}_{episode_id[:8]}",
                episode_body=content,
                episode_type=EpisodeType.message if episode_type == "conversation" else EpisodeType.text,
                reference_time=timestamp,
                source=source,
                entity_config={"agent_id": agent_id, **metadata} if metadata else {"agent_id": agent_id}
            )
            
            return episode_id
        except Exception as e:
            print(f"Error adding episode: {e}")
            return ""
    
    async def search_episodes(self,
                            query: str,
                            agent_id: Optional[str] = None,
                            limit: int = 10,
                            time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Search episodes using semantic search with temporal awareness.
        
        Args:
            query: Search query
            agent_id: Filter by agent ID
            limit: Maximum results
            time_range: Optional (start_time, end_time) tuple
        
        Returns:
            List of matching episodes
        """
        if not self.available or not self.client:
            return []
        
        try:
            # Build search filters
            filters = {}
            if agent_id:
                filters["agent_id"] = agent_id
            
            # Search episodes
            results = await self.client.search(
                query=query,
                limit=limit,
                filter=filters if filters else None
            )
            
            # Format results
            episodes = []
            for result in results:
                episodes.append({
                    "episode_id": result.id,
                    "content": result.content,
                    "score": result.score,
                    "timestamp": result.timestamp,
                    "metadata": result.metadata
                })
            
            return episodes
        except Exception as e:
            print(f"Error searching episodes: {e}")
            return []
    
    async def get_entities(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all entities extracted from episodes.
        
        Args:
            agent_id: Filter by agent ID
        
        Returns:
            List of entities
        """
        if not self.available or not self.client:
            return []
        
        try:
            # Get entities
            entities = await self.client.get_entities(
                filter={"agent_id": agent_id} if agent_id else None
            )
            
            # Format entities
            result = []
            for entity in entities:
                result.append({
                    "entity_id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "metadata": entity.metadata
                })
            
            return result
        except Exception as e:
            print(f"Error getting entities: {e}")
            return []
    
    async def get_relationships(self, 
                               entity_id: Optional[str] = None,
                               agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get relationships between entities.
        
        Args:
            entity_id: Filter by entity ID
            agent_id: Filter by agent ID
        
        Returns:
            List of relationships
        """
        if not self.available or not self.client:
            return []
        
        try:
            # Build filters
            filters = {}
            if entity_id:
                filters["entity_id"] = entity_id
            if agent_id:
                filters["agent_id"] = agent_id
            
            # Get relationships
            relationships = await self.client.get_edges(
                filter=filters if filters else None
            )
            
            # Format relationships
            result = []
            for rel in relationships:
                result.append({
                    "relationship_id": rel.id,
                    "source_entity": rel.source,
                    "target_entity": rel.target,
                    "relationship_type": rel.type,
                    "description": rel.description,
                    "metadata": rel.metadata
                })
            
            return result
        except Exception as e:
            print(f"Error getting relationships: {e}")
            return []
    
    async def temporal_search(self,
                            query: str,
                            time_context: datetime,
                            agent_id: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform temporal-aware search considering time context.
        
        Args:
            query: Search query
            time_context: Reference time for temporal relevance
            agent_id: Filter by agent ID
            limit: Maximum results
        
        Returns:
            Time-aware search results
        """
        if not self.available or not self.client:
            return []
        
        try:
            # Perform temporal search
            results = await self.client.search(
                query=query,
                reference_time=time_context,
                limit=limit,
                filter={"agent_id": agent_id} if agent_id else None
            )
            
            # Format results with temporal relevance
            episodes = []
            for result in results:
                episodes.append({
                    "episode_id": result.id,
                    "content": result.content,
                    "score": result.score,
                    "temporal_relevance": result.temporal_relevance,
                    "timestamp": result.timestamp,
                    "metadata": result.metadata
                })
            
            return episodes
        except Exception as e:
            print(f"Error in temporal search: {e}")
            return []
    
    async def get_agent_knowledge_graph(self, agent_id: str) -> Dict[str, Any]:
        """
        Get complete knowledge graph for an agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Knowledge graph data (nodes and edges)
        """
        if not self.available or not self.client:
            return {"nodes": [], "edges": []}
        
        try:
            # Get entities (nodes)
            entities = await self.get_entities(agent_id=agent_id)
            
            # Get relationships (edges)
            relationships = await self.get_relationships(agent_id=agent_id)
            
            return {
                "agent_id": agent_id,
                "nodes": entities,
                "edges": relationships,
                "node_count": len(entities),
                "edge_count": len(relationships)
            }
        except Exception as e:
            print(f"Error getting knowledge graph: {e}")
            return {"nodes": [], "edges": []}
    
    async def add_document(self,
                          agent_id: str,
                          document: str,
                          title: str,
                          source: str = "document",
                          metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the knowledge graph with automatic chunking.
        
        Args:
            agent_id: Agent identifier
            document: Document text
            title: Document title
            source: Document source
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        if not self.available or not self.client:
            return ""
        
        doc_id = str(uuid.uuid4())
        
        # Split document into chunks for better processing
        chunk_size = 1000  # characters
        chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]
        
        # Add each chunk as an episode
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_id": doc_id,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            chunk_id = await self.add_episode(
                agent_id=agent_id,
                content=chunk,
                episode_type="document",
                source=source,
                metadata=chunk_metadata
            )
            chunk_ids.append(chunk_id)
        
        return doc_id
    
    async def delete_agent_data(self, agent_id: str) -> bool:
        """
        Delete all graph data for an agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Success status
        """
        if not self.available or not self.client:
            return False
        
        try:
            # Delete all episodes for agent
            await self.client.delete_episodes(
                filter={"agent_id": agent_id}
            )
            
            return True
        except Exception as e:
            print(f"Error deleting agent data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.available:
            return {
                "available": False,
                "message": "Graphiti not installed"
            }
        
        return {
            "available": True,
            "db_path": str(self.db_path),
            "llm_config": self.llm_config
        }


# Synchronous wrapper for easier integration
class GraphitiDBSync:
    """Synchronous wrapper for GraphitiDB."""
    
    def __init__(self, db_path: str = "graphiti_db", llm_config: Dict[str, Any] = None):
        self.db = GraphitiDB(db_path, llm_config)
    
    def add_episode(self, *args, **kwargs) -> str:
        """Add episode synchronously."""
        return asyncio.run(self.db.add_episode(*args, **kwargs))
    
    def search_episodes(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Search episodes synchronously."""
        return asyncio.run(self.db.search_episodes(*args, **kwargs))
    
    def get_entities(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get entities synchronously."""
        return asyncio.run(self.db.get_entities(*args, **kwargs))
    
    def get_relationships(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get relationships synchronously."""
        return asyncio.run(self.db.get_relationships(*args, **kwargs))
    
    def temporal_search(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Temporal search synchronously."""
        return asyncio.run(self.db.temporal_search(*args, **kwargs))
    
    def get_agent_knowledge_graph(self, *args, **kwargs) -> Dict[str, Any]:
        """Get knowledge graph synchronously."""
        return asyncio.run(self.db.get_agent_knowledge_graph(*args, **kwargs))
    
    def add_document(self, *args, **kwargs) -> str:
        """Add document synchronously."""
        return asyncio.run(self.db.add_document(*args, **kwargs))
    
    def delete_agent_data(self, *args, **kwargs) -> bool:
        """Delete agent data synchronously."""
        return asyncio.run(self.db.delete_agent_data(*args, **kwargs))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats synchronously."""
        return self.db.get_stats()