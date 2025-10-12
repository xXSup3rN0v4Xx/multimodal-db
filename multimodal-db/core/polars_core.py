import polars as pl
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from .base_agent_config import AgentConfig

class PolarsDBHandler:
    """
    A comprehensive Polars-based database handler for managing agent configurations,
    conversation histories, knowledge bases, and research collections.
    """
    
    def __init__(self, db_path: str = "agent_database"):
        """
        Initialize the Polars database handler.
        
        Args:
            db_path: Directory name within data/ for storing Polars database files (.parquet)
        """
        # Ensure all Polars database files go in the data directory
        # This is separate from Neo4j (Graphiti) and Qdrant vector databases
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        self.db_path = data_dir / db_path
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize database schemas
        self._init_schemas()
        
        # Load existing data or create empty DataFrames
        self._load_or_create_tables()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_schemas(self):
        """Initialize the database schemas for different tables."""
        
        # Agent Matrix Schema - Core agent configurations
        self.agent_matrix_schema = {
            "agent_id": pl.String,
            "agent_name": pl.String,
            "created_at": pl.Datetime,
            "updated_at": pl.Datetime,
            "config_json": pl.String,  # Serialized agent config
            "description": pl.String,
            "tags": pl.List(pl.String),
            "version": pl.String,
            "is_active": pl.Boolean
        }
        
        # Conversation History Schema
        self.conversation_schema = {
            "conversation_id": pl.String,
            "agent_id": pl.String,
            "message_id": pl.String,
            "timestamp": pl.Datetime,
            "role": pl.String,  # user, assistant, system
            "content": pl.String,
            "message_type": pl.String,  # text, image, audio, etc.
            "metadata": pl.String,  # JSON metadata
            "session_id": pl.String
        }
        
        # Knowledge Base Schema
        self.knowledge_base_schema = {
            "kb_id": pl.String,
            "agent_id": pl.String,
            "document_id": pl.String,
            "title": pl.String,
            "content": pl.String,
            "content_type": pl.String,  # text, json, markdown, etc.
            "source": pl.String,
            "created_at": pl.Datetime,
            "updated_at": pl.Datetime,
            "tags": pl.List(pl.String),
            "metadata": pl.String,
            "embedding_status": pl.String  # pending, processed, failed
        }
        
        # Research Collection Schema
        self.research_schema = {
            "research_id": pl.String,
            "agent_id": pl.String,
            "query": pl.String,
            "results": pl.String,  # JSON serialized results
            "source_urls": pl.List(pl.String),
            "created_at": pl.Datetime,
            "research_type": pl.String,  # web_search, document_analysis, etc.
            "status": pl.String,  # completed, pending, failed
            "metadata": pl.String
        }
        
        # Template Files Schema
        self.template_schema = {
            "template_id": pl.String,
            "template_name": pl.String,
            "template_type": pl.String,  # prompt, config, workflow
            "content": pl.String,
            "created_at": pl.Datetime,
            "updated_at": pl.Datetime,
            "tags": pl.List(pl.String),
            "description": pl.String
        }
    
    def _load_or_create_tables(self):
        """Load existing tables or create empty ones."""
        
        # Agent Matrix Table
        agent_matrix_file = self.db_path / "agent_matrix.parquet"
        if agent_matrix_file.exists():
            self.agent_matrix = pl.read_parquet(agent_matrix_file)
        else:
            self.agent_matrix = pl.DataFrame(schema=self.agent_matrix_schema)
        
        # Conversation History Table
        conversation_file = self.db_path / "conversations.parquet"
        if conversation_file.exists():
            self.conversations = pl.read_parquet(conversation_file)
        else:
            self.conversations = pl.DataFrame(schema=self.conversation_schema)
        
        # Knowledge Base Table
        knowledge_file = self.db_path / "knowledge_base.parquet"
        if knowledge_file.exists():
            self.knowledge_base = pl.DataFrame(schema=self.knowledge_base_schema)
        else:
            self.knowledge_base = pl.DataFrame(schema=self.knowledge_base_schema)
        
        # Research Collection Table
        research_file = self.db_path / "research_collection.parquet"
        if research_file.exists():
            self.research_collection = pl.read_parquet(research_file)
        else:
            self.research_collection = pl.DataFrame(schema=self.research_schema)
        
        # Template Files Table
        template_file = self.db_path / "templates.parquet"
        if template_file.exists():
            self.templates = pl.read_parquet(template_file)
        else:
            self.templates = pl.DataFrame(schema=self.template_schema)
    
    def save_tables(self):
        """Save all tables to parquet files."""
        self.agent_matrix.write_parquet(self.db_path / "agent_matrix.parquet")
        self.conversations.write_parquet(self.db_path / "conversations.parquet")
        self.knowledge_base.write_parquet(self.db_path / "knowledge_base.parquet")
        self.research_collection.write_parquet(self.db_path / "research_collection.parquet")
        self.templates.write_parquet(self.db_path / "templates.parquet")
    
    # Agent Matrix Operations
    def add_agent_config(self, agent_config: Union[AgentConfig, Dict[str, Any]], 
                        agent_name: str = None, description: str = "", 
                        tags: List[str] = None) -> str:
        """Add a new agent configuration to the matrix."""
        
        # Handle both AgentConfig objects and legacy dict format
        if isinstance(agent_config, AgentConfig):
            agent_id = agent_config.agent_id
            config_dict = agent_config.to_dict()
            agent_name = agent_name or agent_config.agent_name
            description = description or agent_config.description
            tags = tags or agent_config.tags
            version = agent_config.version
            created_at = agent_config.created_at
            updated_at = agent_config.updated_at
        else:
            # Legacy dict format - should be a dictionary
            if hasattr(agent_config, 'get'):
                agent_id = agent_config.get("agent_id", str(uuid.uuid4()))
            else:
                # This shouldn't happen, but let's handle it gracefully
                raise TypeError(f"Expected AgentConfig object or dict, got {type(agent_config)}")
                
            config_dict = agent_config
            config_dict = agent_config
            agent_name = agent_name or agent_id
            version = "1.0.0"
            created_at = datetime.now()
            updated_at = datetime.now()
        
        new_agent = pl.DataFrame({
            "agent_id": [agent_id],
            "agent_name": [agent_name],
            "created_at": [created_at],
            "updated_at": [updated_at],
            "config_json": [json.dumps(config_dict, default=str)],
            "description": [description],
            "tags": [tags or []],
            "version": [version],
            "is_active": [True]
        })
        
        self.agent_matrix = pl.concat([self.agent_matrix, new_agent])
        self.save_tables()
        return agent_id
    
    def get_agent_config(self, agent_id: str, as_object: bool = True) -> Optional[Union[AgentConfig, Dict[str, Any]]]:
        """Retrieve an agent configuration by ID."""
        result = self.agent_matrix.filter(pl.col("agent_id") == agent_id)
        if result.height > 0:
            config_json = result.select("config_json").to_series()[0]
            config_dict = json.loads(config_json)
            
            if as_object:
                try:
                    return AgentConfig.from_dict(config_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to convert config to AgentConfig object: {e}")
                    return config_dict
            else:
                return config_dict
        return None
    
    def update_agent_config(self, agent_id: str, agent_config: Union[AgentConfig, Dict[str, Any]]):
        """Update an existing agent configuration."""
        
        # Handle both AgentConfig objects and legacy dict format
        if isinstance(agent_config, AgentConfig):
            agent_config.updated_at = datetime.now()
            config_dict = agent_config.to_dict()
            agent_name = agent_config.agent_name
            description = agent_config.description
            tags = agent_config.tags
            version = agent_config.version
        else:
            # Legacy dict format
            config_dict = agent_config
            # Get existing data for fields not in the dict
            existing = self.agent_matrix.filter(pl.col("agent_id") == agent_id)
            if existing.height > 0:
                existing_row = existing.to_dicts()[0]
                agent_name = existing_row.get("agent_name", agent_id)
                description = existing_row.get("description", "")
                tags = existing_row.get("tags", [])
                version = existing_row.get("version", "1.0.0")
            else:
                agent_name = agent_id
                description = ""
                tags = []
                version = "1.0.0"
        
        self.agent_matrix = self.agent_matrix.with_columns([
            pl.when(pl.col("agent_id") == agent_id)
            .then(json.dumps(config_dict, default=str))
            .otherwise(pl.col("config_json"))
            .alias("config_json"),
            
            pl.when(pl.col("agent_id") == agent_id)
            .then(datetime.now())
            .otherwise(pl.col("updated_at"))
            .alias("updated_at"),
            
            pl.when(pl.col("agent_id") == agent_id)
            .then(agent_name)
            .otherwise(pl.col("agent_name"))
            .alias("agent_name"),
            
            pl.when(pl.col("agent_id") == agent_id)
            .then(description)
            .otherwise(pl.col("description"))
            .alias("description"),
            
            pl.when(pl.col("agent_id") == agent_id)
            .then(tags)
            .otherwise(pl.col("tags"))
            .alias("tags"),
            
            pl.when(pl.col("agent_id") == agent_id)
            .then(version)
            .otherwise(pl.col("version"))
            .alias("version")
        ])
        self.save_tables()
    
    def list_agents(self, active_only: bool = True) -> pl.DataFrame:
        """List all agents in the matrix."""
        df = self.agent_matrix
        if active_only:
            df = df.filter(pl.col("is_active") == True)
        return df.select(["agent_id", "agent_name", "description", "tags", "created_at", "updated_at"])
    
    def delete_agent(self, agent_id: str, soft_delete: bool = True):
        """Delete an agent (soft delete by default)."""
        if soft_delete:
            self.agent_matrix = self.agent_matrix.with_columns([
                pl.when(pl.col("agent_id") == agent_id)
                .then(False)
                .otherwise(pl.col("is_active"))
                .alias("is_active")
            ])
        else:
            self.agent_matrix = self.agent_matrix.filter(pl.col("agent_id") != agent_id)
        self.save_tables()
    
    def search_agents(self, query: str, search_fields: List[str] = None) -> pl.DataFrame:
        """Search agents by name, description, or tags."""
        if search_fields is None:
            search_fields = ["agent_name", "description"]
        
        conditions = []
        for field in search_fields:
            conditions.append(pl.col(field).str.contains(query, literal=False))
        
        # Combine conditions with OR
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        
        return self.agent_matrix.filter(combined_condition)
    
    # Conversation Operations
    def add_conversation_message(self, agent_id: str, role: str, content: str, 
                               session_id: str = None, message_type: str = "text", 
                               metadata: Dict[str, Any] = None) -> str:
        """Add a message to conversation history."""
        message_id = str(uuid.uuid4())
        conversation_id = f"{agent_id}_{session_id or 'default'}"
        
        new_message = pl.DataFrame({
            "conversation_id": [conversation_id],
            "agent_id": [agent_id],
            "message_id": [message_id],
            "timestamp": [datetime.now()],
            "role": [role],
            "content": [content],
            "message_type": [message_type],
            "metadata": [json.dumps(metadata or {})],
            "session_id": [session_id or "default"]
        })
        
        self.conversations = pl.concat([self.conversations, new_message])
        self.save_tables()
        return message_id
    
    def get_conversation_history(self, agent_id: str, session_id: str = None, 
                               limit: int = 100) -> pl.DataFrame:
        """Get conversation history for an agent."""
        df = self.conversations.filter(pl.col("agent_id") == agent_id)
        
        if session_id:
            df = df.filter(pl.col("session_id") == session_id)
        
        return df.sort("timestamp", descending=True).limit(limit)
    
    def clear_conversation_history(self, agent_id: str, session_id: str = None):
        """Clear conversation history for an agent."""
        if session_id:
            self.conversations = self.conversations.filter(
                ~((pl.col("agent_id") == agent_id) & (pl.col("session_id") == session_id))
            )
        else:
            self.conversations = self.conversations.filter(pl.col("agent_id") != agent_id)
        self.save_tables()
    
    # Knowledge Base Operations
    def add_knowledge_document(self, agent_id: str, title: str, content: str, 
                             content_type: str = "text", source: str = "", 
                             tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the knowledge base."""
        kb_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_doc = pl.DataFrame({
            "kb_id": [kb_id],
            "agent_id": [agent_id],
            "document_id": [document_id],
            "title": [title],
            "content": [content],
            "content_type": [content_type],
            "source": [source],
            "created_at": [now],
            "updated_at": [now],
            "tags": [tags or []],
            "metadata": [json.dumps(metadata or {})],
            "embedding_status": ["pending"]
        })
        
        self.knowledge_base = pl.concat([self.knowledge_base, new_doc])
        self.save_tables()
        return kb_id
    
    def search_knowledge_base(self, agent_id: str, query: str, 
                            content_type: str = None, tags: List[str] = None) -> pl.DataFrame:
        """Search the knowledge base."""
        df = self.knowledge_base.filter(pl.col("agent_id") == agent_id)
        
        # Text search in title and content
        df = df.filter(
            pl.col("title").str.contains(query, literal=False) |
            pl.col("content").str.contains(query, literal=False)
        )
        
        if content_type:
            df = df.filter(pl.col("content_type") == content_type)
        
        if tags:
            # Filter by tags (if any of the provided tags match)
            tag_conditions = [pl.col("tags").list.contains(tag) for tag in tags]
            combined_tag_condition = tag_conditions[0]
            for condition in tag_conditions[1:]:
                combined_tag_condition = combined_tag_condition | condition
            df = df.filter(combined_tag_condition)
        
        return df.sort("updated_at", descending=True)
    
    def get_knowledge_documents(self, agent_id: str, limit: int = 100) -> pl.DataFrame:
        """Get all knowledge documents for an agent."""
        return (self.knowledge_base
                .filter(pl.col("agent_id") == agent_id)
                .sort("updated_at", descending=True)
                .limit(limit))
    
    def update_embedding_status(self, kb_id: str, status: str):
        """Update the embedding status of a knowledge document."""
        self.knowledge_base = self.knowledge_base.with_columns([
            pl.when(pl.col("kb_id") == kb_id)
            .then(status)
            .otherwise(pl.col("embedding_status"))
            .alias("embedding_status")
        ])
        self.save_tables()
    
    # Research Collection Operations
    def add_research_result(self, agent_id: str, query: str, results: Dict[str, Any], 
                          source_urls: List[str] = None, research_type: str = "web_search",
                          metadata: Dict[str, Any] = None) -> str:
        """Add research results to the collection."""
        research_id = str(uuid.uuid4())
        
        new_research = pl.DataFrame({
            "research_id": [research_id],
            "agent_id": [agent_id],
            "query": [query],
            "results": [json.dumps(results)],
            "source_urls": [source_urls or []],
            "created_at": [datetime.now()],
            "research_type": [research_type],
            "status": ["completed"],
            "metadata": [json.dumps(metadata or {})]
        })
        
        self.research_collection = pl.concat([self.research_collection, new_research])
        self.save_tables()
        return research_id
    
    def search_research_collection(self, agent_id: str, query: str, 
                                 research_type: str = None) -> pl.DataFrame:
        """Search the research collection."""
        df = self.research_collection.filter(pl.col("agent_id") == agent_id)
        
        df = df.filter(pl.col("query").str.contains(query, literal=False))
        
        if research_type:
            df = df.filter(pl.col("research_type") == research_type)
        
        return df.sort("created_at", descending=True)
    
    # Template Operations
    def add_template(self, template_name: str, template_type: str, content: str,
                    description: str = "", tags: List[str] = None) -> str:
        """Add a template to the collection."""
        template_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_template = pl.DataFrame({
            "template_id": [template_id],
            "template_name": [template_name],
            "template_type": [template_type],
            "content": [content],
            "created_at": [now],
            "updated_at": [now],
            "tags": [tags or []],
            "description": [description]
        })
        
        self.templates = pl.concat([self.templates, new_template])
        self.save_tables()
        return template_id
    
    def get_template(self, template_id: str = None, template_name: str = None) -> Optional[Dict[str, Any]]:
        """Get a template by ID or name."""
        if template_id:
            result = self.templates.filter(pl.col("template_id") == template_id)
        elif template_name:
            result = self.templates.filter(pl.col("template_name") == template_name)
        else:
            return None
        
        if result.height > 0:
            return result.to_dicts()[0]
        return None
    
    def list_templates(self, template_type: str = None) -> pl.DataFrame:
        """List all templates."""
        df = self.templates
        if template_type:
            df = df.filter(pl.col("template_type") == template_type)
        return df.sort("updated_at", descending=True)
    
    # Export/Import Operations
    def export_agent_config(self, agent_id: str, filepath: str):
        """Export an agent configuration to a JSON file."""
        config = self.get_agent_config(agent_id)
        if config:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        return False
    
    def import_agent_config(self, filepath: str, agent_name: str = None) -> str:
        """Import an agent configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return self.add_agent_config(config, agent_name or config.get("agent_id", "imported_agent"))
    
    def export_database_backup(self, backup_path: str):
        """Export the entire database as a backup."""
        backup_dir = Path(backup_path)
        backup_dir.mkdir(exist_ok=True)
        
        # Export all tables
        self.agent_matrix.write_parquet(backup_dir / "agent_matrix_backup.parquet")
        self.conversations.write_parquet(backup_dir / "conversations_backup.parquet")
        self.knowledge_base.write_parquet(backup_dir / "knowledge_base_backup.parquet")
        self.research_collection.write_parquet(backup_dir / "research_collection_backup.parquet")
        self.templates.write_parquet(backup_dir / "templates_backup.parquet")
        
        # Export metadata
        metadata = {
            "backup_timestamp": datetime.now().isoformat(),
            "agent_count": self.agent_matrix.height,
            "conversation_count": self.conversations.height,
            "knowledge_count": self.knowledge_base.height,
            "research_count": self.research_collection.height,
            "template_count": self.templates.height
        }
        
        with open(backup_dir / "backup_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "agent_count": self.agent_matrix.height,
            "active_agent_count": self.agent_matrix.filter(pl.col("is_active") == True).height,
            "conversation_count": self.conversations.height,
            "knowledge_document_count": self.knowledge_base.height,
            "research_result_count": self.research_collection.height,
            "template_count": self.templates.height,
            "database_size_mb": sum(
                f.stat().st_size for f in self.db_path.glob("*.parquet")
            ) / (1024 * 1024)
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        return {
            "total_agents": self.agent_matrix.height,
            "active_agents": self.agent_matrix.filter(pl.col("is_active") == True).height,
            "inactive_agents": self.agent_matrix.filter(pl.col("is_active") == False).height
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "total_messages": self.conversations.height,
            "unique_sessions": self.conversations.select("session_id").n_unique(),
            "unique_conversations": self.conversations.select("conversation_id").n_unique()
        }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_entries": self.knowledge_base.height,
            "embedded_entries": self.knowledge_base.filter(pl.col("embedding_status") == "completed").height,
            "pending_entries": self.knowledge_base.filter(pl.col("embedding_status") == "pending").height
        }
    
    def get_agent_by_id(self, agent_id: str) -> Optional[pl.DataFrame]:
        """Get agent data by ID."""
        result = self.agent_matrix.filter(pl.col("agent_id") == agent_id)
        return result if result.height > 0 else None
    
    # New Methods for JSONL Export and Conversation Generation
    def export_conversations_jsonl(self, agent_id: str, output_path: str) -> bool:
        """
        Export conversations in JSONL format for training/fine-tuning.
        
        Args:
            agent_id: Agent ID to export conversations for
            output_path: Path to save JSONL file
            
        Returns:
            bool: Success status
        """
        try:
            conversations = self.conversations.filter(
                pl.col("agent_id") == agent_id
            ).sort("timestamp")
            
            if conversations.height == 0:
                self.logger.warning(f"No conversations found for agent {agent_id}")
                return False
            
            # Group by session and create conversation pairs
            jsonl_data = []
            
            for session_id in conversations["session_id"].unique():
                session_msgs = conversations.filter(
                    pl.col("session_id") == session_id
                ).sort("timestamp")
                
                # Create conversation pairs (user -> assistant)
                user_msg = None
                for row in session_msgs.to_dicts():
                    if row["role"] == "user":
                        user_msg = row["content"]
                    elif row["role"] == "assistant" and user_msg:
                        jsonl_data.append({
                            "messages": [
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": row["content"]}
                            ],
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "timestamp": str(row["timestamp"])  # Convert datetime to string
                        })
                        user_msg = None
            
            # Write JSONL file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in jsonl_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Exported {len(jsonl_data)} conversation pairs to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export conversations to JSONL: {e}")
            return False
    
    def export_data(self, table_name: str, format: str = "csv", file_path: str = None) -> str:
        """Export data from a table to various formats.
        
        Args:
            table_name: Name of the table to export ('agents', 'conversations', 'knowledge', 'research', 'templates')
            format: Export format ('csv', 'parquet', 'json', 'jsonl')
            file_path: Optional custom file path
            
        Returns:
            Path to the exported file
        """
        try:
            # Get the appropriate table
            table_map = {
                'agents': self.agent_matrix,
                'conversations': self.conversations,
                'knowledge': self.knowledge_base,
                'research': self.research_collection,
                'templates': self.templates
            }
            
            if table_name not in table_map:
                raise ValueError(f"Unknown table: {table_name}. Available: {list(table_map.keys())}")
            
            table = table_map[table_name]
            
            # Generate file path if not provided
            if file_path is None:
                file_path = str(self.db_path / f"{table_name}.{format}")
            
            # Export based on format
            if format == "csv":
                table.write_csv(file_path)
            elif format == "parquet":
                table.write_parquet(file_path)
            elif format in ["json", "jsonl"]:
                table.write_ndjson(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {table_name} to {file_path} ({format} format)")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to export {table_name}: {e}")
            raise