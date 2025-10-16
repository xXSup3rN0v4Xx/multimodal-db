"""
Optimized Polars Database - Essential Operations Only
Fast, lightweight, focused on core functionality.
"""
import polars as pl
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..agent_configs.base_agent_config import AgentConfig


class PolarsDB:
    """Minimal, high-performance Polars database for agents."""
    
    def __init__(self, db_path: str = "core_db"):
        self.db_path = Path("data") / db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Core tables only
        self.agents_file = self.db_path / "agents.parquet"
        self.conversations_file = self.db_path / "conversations.parquet"
        
        self._init_tables()
    
    def _init_tables(self):
        """Initialize core tables."""
        # Agents table
        if self.agents_file.exists():
            self.agents = pl.read_parquet(self.agents_file)
        else:
            self.agents = pl.DataFrame({
                "agent_id": pl.Series([], dtype=pl.String),
                "name": pl.Series([], dtype=pl.String),
                "config": pl.Series([], dtype=pl.String),  # JSON
                "created_at": pl.Series([], dtype=pl.Datetime)
            })
        
        # Conversations table  
        if self.conversations_file.exists():
            self.conversations = pl.read_parquet(self.conversations_file)
        else:
            self.conversations = pl.DataFrame({
                "id": pl.Series([], dtype=pl.String),
                "agent_id": pl.Series([], dtype=pl.String),
                "role": pl.Series([], dtype=pl.String),
                "content": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime)
            })
    
    def save(self):
        """Save all tables to disk."""
        self.agents.write_parquet(self.agents_file)
        self.conversations.write_parquet(self.conversations_file)
    
    def add_agent(self, agent: AgentConfig, name: str = None) -> str:
        """Add agent to database."""
        agent_id = str(uuid.uuid4())
        config_json = json.dumps({
            "agent_name": agent.agent_name,
            "description": agent.description,
            "system_prompt": agent.system_prompt,  # Now works with property
            "helper_prompts": agent.helper_prompts,  # Now works with property
            "tags": agent.tags
        })
        
        new_agent = pl.DataFrame({
            "agent_id": [agent_id],
            "name": [name or agent.agent_name],
            "config": [config_json],
            "created_at": [datetime.now()]
        })
        
        self.agents = pl.concat([self.agents, new_agent])
        self.save()
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent by ID."""
        agent_data = self.agents.filter(pl.col("agent_id") == agent_id)
        if agent_data.height == 0:
            return None
        
        config_json = agent_data.select("config").item()
        config_data = json.loads(config_json)
        
        # Use from_dict to properly reconstruct AgentConfig
        return AgentConfig.from_dict(config_data)
    
    def add_message(self, agent_id: str, role: str, content: str) -> str:
        """Add conversation message."""
        msg_id = str(uuid.uuid4())
        
        new_msg = pl.DataFrame({
            "id": [msg_id],
            "agent_id": [agent_id],
            "role": [role],
            "content": [content],
            "timestamp": [datetime.now()]
        })
        
        self.conversations = pl.concat([self.conversations, new_msg])
        self.save()
        return msg_id
    
    def get_messages(self, agent_id: str, limit: int = 50) -> list:
        """Get conversation messages for agent."""
        messages = (self.conversations
                   .filter(pl.col("agent_id") == agent_id)
                   .sort("timestamp", descending=True)
                   .limit(limit)
                   .select(["role", "content", "timestamp"]))
        
        return messages.to_dicts()
    
    def clear_messages(self, agent_id: str):
        """Clear all messages for an agent."""
        self.conversations = self.conversations.filter(pl.col("agent_id") != agent_id)
        self.save()
    
    def list_agents(self) -> list:
        """List all agents."""
        return (self.agents
                .select(["agent_id", "name", "created_at"])
                .to_dicts())