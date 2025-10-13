"""
Razor-Sharp Multimodal Agent Configuration
Optimized for text, embeddings, audio, images, and video.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class ModelType(Enum):
    """Core model types for multimodal AI agents."""
    # Language Processing
    LLM = "llm"
    EMBEDDING = "embedding"
    QWEN_CODER_3B = "qwen2.5-coder:3b"  # Specific Ollama model
    
    # Vision Processing  
    VISION_LLM = "vision_llm"
    VISION_DETECTION = "vision_detection"
    IMAGE_GENERATION = "image_generation"
    
    # Audio Processing
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech" 
    AUDIO_GENERATION = "audio_generation"
    
    # Video Processing
    VIDEO_GENERATION = "video_generation"
    VIDEO_ANALYSIS = "video_analysis"


class MediaType(Enum):
    """Supported media types for multimodal storage."""
    TEXT = "text"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


class AgentConfig:
    """
    Razor-sharp agent configuration for multimodal AI systems.
    Optimized for storage, retrieval, and multimodal capabilities.
    """
    
    def __init__(self, agent_name: str, description: str = "", tags: List[str] = None):
        """Initialize streamlined agent configuration."""
        self.agent_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Core capabilities
        self.system_prompt = ""
        self.helper_prompts = {}
        self.supported_media = [MediaType.TEXT]  # Default to text support
        
        # Model configurations - simplified and focused
        self.models = {
            # Text & Language
            ModelType.LLM.value: {
                "ollama": {"enabled": True, "model": "qwen2.5-coder:3b"},
                "default": "ollama"
            },
            ModelType.EMBEDDING.value: {
                "nomic": {"enabled": True, "model": "nomic-embed-text-v1.5"},
                "default": "nomic"
            },
            
            # Vision (future)
            ModelType.VISION_LLM.value: {
                "enabled": False,
                "default": None
            },
            ModelType.VISION_DETECTION.value: {
                "enabled": False, 
                "default": None
            },
            ModelType.IMAGE_GENERATION.value: {
                "enabled": False,
                "default": None
            },
            
            # Audio (future)
            ModelType.SPEECH_TO_TEXT.value: {
                "enabled": False,
                "default": None
            },
            ModelType.TEXT_TO_SPEECH.value: {
                "enabled": False,
                "default": None
            },
            ModelType.AUDIO_GENERATION.value: {
                "enabled": False,
                "default": None
            },
            
            # Video (future)
            ModelType.VIDEO_GENERATION.value: {
                "enabled": False,
                "default": None
            },
            ModelType.VIDEO_ANALYSIS.value: {
                "enabled": False,
                "default": None
            }
        }
        
        # Multimodal storage preferences
        self.media_config = {
            MediaType.TEXT.value: {"storage": "polars", "vector_store": "qdrant"},
            MediaType.EMBEDDING.value: {"storage": "qdrant", "dimensions": 768},
            MediaType.AUDIO.value: {"storage": "files", "vector_store": "qdrant", "formats": ["wav", "mp3"]},
            MediaType.IMAGE.value: {"storage": "files", "vector_store": "qdrant", "formats": ["jpg", "png"]},
            MediaType.VIDEO.value: {"storage": "files", "vector_store": "qdrant", "formats": ["mp4", "avi"]},
            MediaType.DOCUMENT.value: {"storage": "polars", "vector_store": "qdrant"}
        }
    
    def add_helper_prompt(self, name: str, prompt: str):
        """Add helper prompt capability."""
        self.helper_prompts[name] = prompt
        self._update_timestamp()
    
    def enable_model(self, model_type: ModelType, provider: str, model_name: str = None):
        """Enable a specific model type."""
        if model_type.value in self.models:
            if provider not in self.models[model_type.value]:
                self.models[model_type.value][provider] = {}
            
            self.models[model_type.value][provider]["enabled"] = True
            if model_name:
                self.models[model_type.value][provider]["model"] = model_name
            self.models[model_type.value]["default"] = provider
            self._update_timestamp()
    
    def get_enabled_models(self) -> Dict[str, str]:
        """Get currently enabled models."""
        enabled = {}
        for model_type, config in self.models.items():
            if config.get("default") and config[config["default"]].get("enabled"):
                enabled[model_type] = config[config["default"]].get("model", config["default"])
        return enabled
    
    def supports_media_type(self, media_type: MediaType) -> bool:
        """Check if agent supports a specific media type."""
        return media_type.value in self.media_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export agent configuration to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "description": self.description,
            "tags": self.tags,
            "system_prompt": self.system_prompt,
            "helper_prompts": self.helper_prompts,
            "models": self.models,
            "media_config": self.media_config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Import agent configuration from dictionary."""
        agent = cls(
            agent_name=data["agent_name"],
            description=data.get("description", ""),
            tags=data.get("tags", [])
        )
        
        agent.agent_id = data["agent_id"]
        agent.system_prompt = data.get("system_prompt", "")
        agent.helper_prompts = data.get("helper_prompts", {})
        agent.models = data.get("models", agent.models)
        agent.media_config = data.get("media_config", agent.media_config)
        
        if "created_at" in data:
            agent.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            agent.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return agent
    
    def _update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()


def create_corecoder_agent(name: str = "corecoder") -> AgentConfig:
    """Create optimized CoreCoder agent with essential capabilities."""
    agent = AgentConfig(
        agent_name=name,
        description="A highly skilled software engineer and machine learning specialist with enhanced terminal capabilities",
        tags=["software-engineering", "coding", "terminal", "collaboration"]
    )
    
    # Set system prompt
    agent.system_prompt = """You are CoreCoder, a highly skilled software engineer and machine learning specialist. 
You excel at writing clean, efficient code and providing practical solutions. Be direct, helpful, and focus on working solutions."""
    
    # Add essential helper prompts
    agent.add_helper_prompt("investigation_verification", 
        "CoreCoder, my engineering partner—don't forget to use your full terminal toolkit to investigate and verify everything thoroughly.")
    
    agent.add_helper_prompt("terminal_guidelines", 
        "Avoid terminal commands that require user interaction unless absolutely necessary. Use appropriate flags for non-interactive operation.")
    
    agent.add_helper_prompt("deep_investigation", 
        "Let's try something: use the terminal to list all files in project, then read the key files to understand the structure and identify issues.")
    
    agent.add_helper_prompt("review_documentation", 
        "Outstanding work, CoreCoder! Let's now review the project and documentation. Test everything thoroughly and ensure it's production-ready.")
    
    agent.add_helper_prompt("project_organization", 
        "Please move all tests to the tests directory and documentation to docs. Keep the project organized and clean.")
    
    # Enable core models for current functionality
    agent.enable_model(ModelType.LLM, "ollama", "qwen2.5-coder:3b")
    agent.enable_model(ModelType.EMBEDDING, "nomic", "nomic-embed-text-v1.5")
    
    # Set supported media types for coding agent
    agent.supported_media = [MediaType.TEXT, MediaType.DOCUMENT]
    
    return agent


def create_multimodal_agent(name: str = "multimodal") -> AgentConfig:
    """Create example multimodal agent ready for future extensions."""
    agent = AgentConfig(
        agent_name=name,
        description="Advanced multimodal AI agent capable of processing text, audio, images, and video",
        tags=["multimodal", "vision", "audio", "comprehensive"]
    )
    
    agent.system_prompt = """You are MultiModal, an advanced AI agent capable of processing and understanding multiple types of media. 
You can work with text, images, audio, and video to provide comprehensive assistance."""
    
    # Add multimodal capabilities
    agent.add_helper_prompt("vision_analysis", "Analyze images and visual content for insights and understanding.")
    agent.add_helper_prompt("audio_processing", "Process and analyze audio content including speech and sounds.")
    agent.add_helper_prompt("video_understanding", "Understand and analyze video content frame by frame.")
    agent.add_helper_prompt("multimodal_fusion", "Combine insights from multiple media types for comprehensive understanding.")
    
    # Enable current models and prepare for future ones
    agent.enable_model(ModelType.LLM, "ollama", "qwen2.5-coder:3b")
    agent.enable_model(ModelType.EMBEDDING, "nomic", "nomic-embed-text-v1.5")
    
    # Set supported media types for multimodal agent
    agent.supported_media = [
        MediaType.TEXT, MediaType.DOCUMENT, MediaType.EMBEDDING,
        MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO
    ]
    
    return agent