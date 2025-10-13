"""
Razor-Sharp Multimodal-DB Core
Optimized components for high-performance multimodal database operations.
"""

# Core optimized components  
from .agent_config import AgentConfig, ModelType, MediaType, create_corecoder_agent, create_multimodal_agent
from .multimodal_db import MultimodalDB
from .vector_db import QdrantVectorDB

# Legacy components (maintained for compatibility)
from .base_agent_config import AgentConfig as LegacyAgentConfig
from .polars_db import PolarsDB
from .qdrant_db import QdrantDB
from .simple_ollama import SimpleOllamaClient

__all__ = [
    # Core razor-sharp components
    "AgentConfig", 
    "ModelType",
    "MediaType", 
    "create_corecoder_agent",
    "create_multimodal_agent",
    "MultimodalDB",
    "QdrantVectorDB",
    
    # Legacy components
    "LegacyAgentConfig",
    "PolarsDB",
    "QdrantDB",
    "SimpleOllamaClient"
]