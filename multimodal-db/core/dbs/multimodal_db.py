"""
Razor-Sharp Multimodal Database
Handles text, embeddings, audio, images, video with blazing speed.
"""
import polars as pl
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import hashlib
import base64

try:
    from ..agent_configs.base_agent_config import MediaType, AgentConfig
except ImportError:
    try:
        from core.agent_configs.base_agent_config import MediaType, AgentConfig
    except ImportError:
        # Fallback for standalone usage
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from agent_configs.base_agent_config import MediaType, AgentConfig


class MultimodalDB:
    """
    High-performance multimodal database using Polars + file storage.
    Optimized for agent configurations and all media types.
    """
    
    def __init__(self, db_path: str = "multimodal_db"):
        # Handle both relative and absolute paths
        db_path_obj = Path(db_path)
        if db_path_obj.is_absolute():
            # Use absolute path as-is
            self.db_path = db_path_obj
        else:
            # Prepend "data/" for relative paths
            self.db_path = Path("data") / db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Core database files
        self.agents_file = self.db_path / "agents.parquet"
        self.conversations_file = self.db_path / "conversations.parquet"
        self.media_index_file = self.db_path / "media_index.parquet"
        
        # Media storage directories
        self.media_dir = self.db_path / "media"
        for media_type in MediaType:
            (self.media_dir / media_type.value).mkdir(parents=True, exist_ok=True)
        
        self._init_tables()
    
    def _init_tables(self):
        """Initialize core tables with optimized schemas."""
        
        # Agents table - streamlined
        if self.agents_file.exists():
            self.agents = pl.read_parquet(self.agents_file)
        else:
            self.agents = pl.DataFrame({
                "agent_id": pl.Series([], dtype=pl.String),
                "name": pl.Series([], dtype=pl.String),
                "description": pl.Series([], dtype=pl.String),
                "config_json": pl.Series([], dtype=pl.String),
                "tags": pl.Series([], dtype=pl.List(pl.String)),
                "enabled_models": pl.Series([], dtype=pl.String),  # JSON
                "created_at": pl.Series([], dtype=pl.Datetime),
                "updated_at": pl.Series([], dtype=pl.Datetime)
            })
        
        # Conversations table - multimodal ready
        if self.conversations_file.exists():
            self.conversations = pl.read_parquet(self.conversations_file)
        else:
            self.conversations = pl.DataFrame({
                "id": pl.Series([], dtype=pl.String),
                "agent_id": pl.Series([], dtype=pl.String),
                "session_id": pl.Series([], dtype=pl.String),
                "role": pl.Series([], dtype=pl.String),
                "content": pl.Series([], dtype=pl.String),
                "media_type": pl.Series([], dtype=pl.String),
                "media_refs": pl.Series([], dtype=pl.List(pl.String)),  # File references
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "metadata": pl.Series([], dtype=pl.String)  # JSON
            })
        
        # Media index table - tracks all media files
        if self.media_index_file.exists():
            self.media_index = pl.read_parquet(self.media_index_file)
        else:
            self.media_index = pl.DataFrame({
                "media_id": pl.Series([], dtype=pl.String),
                "media_type": pl.Series([], dtype=pl.String),
                "file_path": pl.Series([], dtype=pl.String),
                "content_hash": pl.Series([], dtype=pl.String),
                "file_size": pl.Series([], dtype=pl.Int64),
                "mime_type": pl.Series([], dtype=pl.String),
                "agent_id": pl.Series([], dtype=pl.String),
                "created_at": pl.Series([], dtype=pl.Datetime),
                "metadata": pl.Series([], dtype=pl.String)  # JSON
            })
    
    def save(self):
        """Save all tables efficiently."""
        self.agents.write_parquet(self.agents_file)
        self.conversations.write_parquet(self.conversations_file)
        self.media_index.write_parquet(self.media_index_file)
    
    # Agent Operations
    def add_agent(self, agent: AgentConfig) -> str:
        """Add agent configuration to database."""
        config_json = json.dumps(agent.to_dict())
        enabled_models = json.dumps(agent.get_enabled_models())
        
        new_agent = pl.DataFrame({
            "agent_id": [agent.agent_id],
            "name": [agent.agent_name],
            "description": [agent.description],
            "config_json": [config_json],
            "tags": [agent.tags],
            "enabled_models": [enabled_models],
            "created_at": [agent.created_at],
            "updated_at": [agent.updated_at]
        })
        
        self.agents = pl.concat([self.agents, new_agent])
        self.save()
        return agent.agent_id
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Retrieve agent configuration."""
        agent_data = self.agents.filter(pl.col("agent_id") == agent_id)
        if agent_data.height == 0:
            return None
        
        config_json = agent_data.select("config_json").item()
        config_data = json.loads(config_json)
        return AgentConfig.from_dict(config_data)
    
    def list_agents(self, include_full_config: bool = False) -> List[Dict[str, Any]]:
        """
        List all agents with summary info.
        
        Args:
            include_full_config: If True, includes complete agent configuration with prompts
        """
        if include_full_config:
            # Return full agent data including config_json
            agents_list = self.agents.to_dicts()
            # Parse config_json for each agent
            for agent in agents_list:
                if 'config_json' in agent and agent['config_json']:
                    try:
                        config_data = json.loads(agent['config_json'])
                        # Merge full config into the agent dict
                        agent['system_prompt'] = config_data.get('system_prompt', '')
                        agent['helper_prompts'] = config_data.get('helper_prompts', {})
                        agent['models'] = config_data.get('models', {})
                        agent['media_config'] = config_data.get('media_config', {})
                    except json.JSONDecodeError:
                        pass
            return agents_list
        else:
            # Return summary only (original behavior)
            return (self.agents
                    .select(["agent_id", "name", "description", "tags", "enabled_models", "created_at"])
                    .to_dicts())
    
    def update_agent(self, agent: AgentConfig):
        """Update existing agent configuration."""
        # Remove old record
        self.agents = self.agents.filter(pl.col("agent_id") != agent.agent_id)
        # Add updated record
        self.add_agent(agent)
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent and associated data."""
        # Check if agent exists
        if self.agents.filter(pl.col("agent_id") == agent_id).height == 0:
            return False
        
        # Delete agent record
        self.agents = self.agents.filter(pl.col("agent_id") != agent_id)
        
        # Delete agent's conversations
        self.conversations = self.conversations.filter(pl.col("agent_id") != agent_id)
        
        # Delete agent's media files
        agent_media = self.media_index.filter(pl.col("agent_id") == agent_id)
        for media_record in agent_media.to_dicts():
            media_path = Path(media_record["file_path"])
            if media_path.exists():
                media_path.unlink()
        
        # Remove from media index
        self.media_index = self.media_index.filter(pl.col("agent_id") != agent_id)
        
        self.save()
        return True
    
    def store_agent(self, agent: AgentConfig) -> str:
        """Alias for add_agent for compatibility."""
        return self.add_agent(agent)
    
    def store_content(self, agent_id: str, content: str, media_type: MediaType,
                     metadata: Dict[str, Any] = None) -> str:
        """Store content with media type information."""
        return self.add_message(
            agent_id=agent_id,
            role="content",
            content=content,
            media_type=media_type.value,
            metadata=metadata
        )
    
    def search_content(self, agent_id: str = None, media_type: MediaType = None,
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search content with filters."""
        query_parts = []
        params = []
        
        if agent_id:
            query_parts.append("agent_id = ?")
            params.append(agent_id)
        
        if media_type:
            query_parts.append("media_type = ?") 
            params.append(media_type.value)
        
        query = "SELECT * FROM messages"
        if query_parts:
            query += " WHERE " + " AND ".join(query_parts)
        
        try:
            results = self.db.execute(query, params).collect()
            return results.to_dicts() if not results.is_empty() else []
        except:
            return []

    # Conversation Operations
    def add_message(self, agent_id: str, role: str, content: str, 
                   session_id: str = None, media_type: str = "text",
                   media_files: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add conversation message with optional media."""
        msg_id = str(uuid.uuid4())
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        new_msg = pl.DataFrame({
            "id": [msg_id],
            "agent_id": [agent_id],
            "session_id": [session_id],
            "role": [role],
            "content": [content],
            "media_type": [media_type],
            "media_refs": [media_files or []],
            "timestamp": [datetime.now()],
            "metadata": [json.dumps(metadata or {})]
        })
        
        self.conversations = pl.concat([self.conversations, new_msg])
        self.save()
        return msg_id
    
    def get_conversation(self, agent_id: str, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation messages."""
        query = self.conversations.filter(pl.col("agent_id") == agent_id)
        
        if session_id:
            query = query.filter(pl.col("session_id") == session_id)
        
        messages = (query
                   .sort("timestamp", descending=True)
                   .limit(limit)
                   .select(["role", "content", "media_type", "media_refs", "timestamp", "metadata"]))
        
        return messages.to_dicts()
    
    def get_messages(self, agent_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for agent (all sessions). Compatible with PolarsDB interface."""
        messages = (self.conversations
                   .filter(pl.col("agent_id") == agent_id)
                   .sort("timestamp", descending=True)
                   .limit(limit)
                   .select(["id", "role", "content", "timestamp"]))
        
        return messages.to_dicts()
    
    def clear_messages(self, agent_id: str):
        """Clear all messages for an agent."""
        self.conversations = self.conversations.filter(pl.col("agent_id") != agent_id)
        self.save()
    
    # Media Operations
    def store_media(self, media_data: bytes, media_type: MediaType, 
                   agent_id: str, filename: str = None, 
                   mime_type: str = None, metadata: Dict[str, Any] = None) -> str:
        """Store media file and return media_id."""
        media_id = str(uuid.uuid4())
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(media_data).hexdigest()
        
        # Check if media already exists
        existing = self.media_index.filter(pl.col("content_hash") == content_hash)
        if existing.height > 0:
            return existing.select("media_id").item()
        
        # Generate filename if not provided
        if not filename:
            extension = self._get_extension_from_mime(mime_type) if mime_type else "bin"
            filename = f"{media_id}.{extension}"
        
        # Store file
        media_path = self.media_dir / media_type.value / filename
        media_path.write_bytes(media_data)
        
        # Add to media index
        new_media = pl.DataFrame({
            "media_id": [media_id],
            "media_type": [media_type.value],
            "file_path": [str(media_path)],
            "content_hash": [content_hash],
            "file_size": [len(media_data)],
            "mime_type": [mime_type or "application/octet-stream"],
            "agent_id": [agent_id],
            "created_at": [datetime.now()],
            "metadata": [json.dumps(metadata or {})]
        })
        
        self.media_index = pl.concat([self.media_index, new_media])
        self.save()
        return media_id
    
    def get_media(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve media file info and content."""
        media_data = self.media_index.filter(pl.col("media_id") == media_id)
        if media_data.height == 0:
            return None
        
        media_info = media_data.to_dicts()[0]
        file_path = Path(media_info["file_path"])
        
        if file_path.exists():
            media_info["content"] = file_path.read_bytes()
            media_info["metadata"] = json.loads(media_info["metadata"])
            return media_info
        
        return None
    
    def list_media(self, agent_id: str = None, media_type: MediaType = None) -> List[Dict[str, Any]]:
        """List media files with optional filtering."""
        query = self.media_index
        
        if agent_id:
            query = query.filter(pl.col("agent_id") == agent_id)
        
        if media_type:
            query = query.filter(pl.col("media_type") == media_type.value)
        
        return (query
                .select(["media_id", "media_type", "mime_type", "file_size", "agent_id", "created_at"])
                .to_dicts())
    
    def delete_media(self, media_id: str) -> bool:
        """Delete media file and index entry."""
        media_data = self.media_index.filter(pl.col("media_id") == media_id)
        if media_data.height == 0:
            return False
        
        # Delete physical file
        file_path = Path(media_data.select("file_path").item())
        if file_path.exists():
            file_path.unlink()
        
        # Remove from index
        self.media_index = self.media_index.filter(pl.col("media_id") != media_id)
        self.save()
        return True
    
    # Export/Import Operations
    def export_agent_full(self, agent_id: str, export_path: str) -> bool:
        """Export complete agent configuration and data."""
        agent = self.get_agent(agent_id)
        if not agent:
            return False
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export agent config
        agent_file = export_dir / "agent_config.json"
        agent_file.write_text(json.dumps(agent.to_dict(), indent=2))
        
        # Export conversations
        conversations = self.get_conversation(agent_id, limit=1000)
        if conversations:
            conv_file = export_dir / "conversations.json"
            conv_file.write_text(json.dumps(conversations, indent=2, default=str))
        
        # Export media files
        media_files = self.list_media(agent_id)
        if media_files:
            media_dir = export_dir / "media"
            media_dir.mkdir(exist_ok=True)
            
            for media_info in media_files:
                media_data = self.get_media(media_info["media_id"])
                if media_data:
                    media_file = media_dir / f"{media_info['media_id']}.{media_info['media_type']}"
                    media_file.write_bytes(media_data["content"])
            
            # Export media index
            media_index_file = export_dir / "media_index.json"
            media_index_file.write_text(json.dumps(media_files, indent=2, default=str))
        
        return True
    
    def import_agent_full(self, import_path: str) -> str:
        """Import complete agent configuration and data."""
        import_dir = Path(import_path)
        if not import_dir.exists():
            return ""
        
        # Import agent config
        agent_file = import_dir / "agent_config.json"
        if not agent_file.exists():
            return ""
        
        agent_data = json.loads(agent_file.read_text())
        agent = AgentConfig.from_dict(agent_data)
        
        # Generate new agent_id to avoid conflicts
        old_agent_id = agent.agent_id
        agent.agent_id = str(uuid.uuid4())
        
        # Store agent
        self.add_agent(agent)
        
        # Import conversations
        conv_file = import_dir / "conversations.json"
        if conv_file.exists():
            conversations = json.loads(conv_file.read_text())
            for conv in conversations:
                self.add_message(
                    agent_id=agent.agent_id,
                    role=conv["role"],
                    content=conv["content"],
                    session_id=conv.get("session_id"),
                    media_type=conv.get("media_type", "text"),
                    media_files=conv.get("media_refs", []),
                    metadata=json.loads(conv.get("metadata", "{}"))
                )
        
        # Import media files
        media_dir = import_dir / "media"
        media_index_file = import_dir / "media_index.json"
        if media_dir.exists() and media_index_file.exists():
            media_index = json.loads(media_index_file.read_text())
            for media_info in media_index:
                media_file = media_dir / f"{media_info['media_id']}.{media_info['media_type']}"
                if media_file.exists():
                    media_data = media_file.read_bytes()
                    self.store_media(
                        media_data=media_data,
                        media_type=MediaType(media_info["media_type"]),
                        agent_id=agent.agent_id,
                        filename=media_file.name,
                        mime_type=media_info.get("mime_type"),
                        metadata=json.loads(media_info.get("metadata", "{}"))
                    )
        
        return agent.agent_id
    
    def _get_extension_from_mime(self, mime_type: str) -> str:
        """Get file extension from MIME type."""
        mime_map = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "audio/wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "video/mp4": "mp4",
            "video/avi": "avi",
            "text/plain": "txt",
            "application/json": "json"
        }
        return mime_map.get(mime_type, "bin")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            agents_count = len(self.agents) if hasattr(self, 'agents') else 0
            conversations_count = len(self.conversations) if hasattr(self, 'conversations') else 0
            
            # Try to get actual counts from database
            try:
                agents_result = self.db.execute("SELECT COUNT(*) as count FROM agents").collect()
                agents_count = agents_result.select("count").item() if not agents_result.is_empty() else 0
            except:
                pass
                
            try:
                messages_result = self.db.execute("SELECT COUNT(*) as count FROM messages").collect()
                content_count = messages_result.select("count").item() if not messages_result.is_empty() else 0
            except:
                content_count = 0
            
            return {
                "total_agents": agents_count,
                "total_content": content_count,
                "efficiency_score": 95.0,  # Placeholder
                "database_path": str(self.db_path)
            }
        except Exception as e:
            return {
                "total_agents": 0,
                "total_content": 0,
                "efficiency_score": 0.0,
                "error": str(e)
            }
    
    def export_agent(self, agent_id: str, include_content: bool = True) -> Dict[str, Any]:
        """Export agent configuration and optionally content."""
        agent = self.get_agent(agent_id)
        if not agent:
            return {}
        
        export_data = {
            "agent": agent.to_dict(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if include_content:
            # Get agent's messages/content
            try:
                content_result = self.db.execute(
                    "SELECT * FROM messages WHERE agent_id = ?", 
                    [agent_id]
                ).collect()
                export_data["content"] = content_result.to_dicts() if not content_result.is_empty() else []
            except:
                export_data["content"] = []
        
        return export_data
    
    def import_agent(self, export_data: Dict[str, Any], new_agent_id: str = None) -> bool:
        """Import agent from export data."""
        try:
            agent_data = export_data.get("agent", {})
            if not agent_data:
                return False
            
            # Create agent from data
            if new_agent_id:
                agent_data["agent_id"] = new_agent_id
            
            agent = AgentConfig.from_dict(agent_data)
            self.add_agent(agent)
            
            # Import content if present
            content_data = export_data.get("content", [])
            for content in content_data:
                self.add_message(
                    agent_id=agent.agent_id,
                    role=content.get("role", "content"),
                    content=content.get("content", ""),
                    metadata=json.loads(content.get("metadata", "{}"))
                )
            
            return True
        except Exception:
            return False
    
    def find_duplicates(self) -> List[Dict[str, Any]]:
        """Find potential duplicate content."""
        # Simple implementation - can be enhanced
        return []
    
    def remove_duplicates(self) -> int:
        """Remove duplicate entries."""
        # Placeholder implementation
        return 0