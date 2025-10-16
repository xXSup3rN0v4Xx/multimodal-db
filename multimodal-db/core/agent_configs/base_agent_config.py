import json
import copy
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class MediaType(Enum):
    """Enumeration of supported media types."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    EMBEDDING = "embedding"

class ModelType(Enum):
    """Enumeration of supported model types."""
    LLM = "large_language_model"
    VISION_LLM = "vision_language_model" 
    EMBEDDING = "embedding_model"
    VISION_DETECTION = "vision_detection"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_GENERATION = "audio_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"

class PromptType(Enum):
    """Types of prompts based on model capabilities."""
    SYSTEM = "system"  # Only for LLMs and Vision LLMs
    HELPER = "helper"  # Multiple allowed, flexible
    BOOSTER = "booster"  # Performance enhancement
    PRIME_DIRECTIVE = "prime_directive"  # Core behavior
    USER_INPUT = "user_input"  # User interaction templates
    USER_FILES = "user_files"  # File handling templates
    USER_IMAGES = "user_images"  # Image handling templates

class DatabaseCategory(Enum):
    """Core database categories for agent storage."""
    AGENT_CONFIGS = "agent_configurations"
    CONVERSATIONS = "conversation_histories" 
    KNOWLEDGE_BASE = "knowledge_documents"
    RESEARCH_DATA = "research_collections"
    TEMPLATES = "template_collections"
    USER_DATA = "user_personal_data"
    ALIGNMENT_DOCS = "agentic_alignment_documents"

class ResearchCategory(Enum):
    """Research domain categories."""
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    AI_ML = "artificial_intelligence_machine_learning"
    HISTORY = "history"
    LITERATURE = "literature"
    PHILOSOPHY = "philosophy"
    PSYCHOLOGY = "psychology"
    ECONOMICS = "economics"
    BUSINESS = "business"
    HEALTH_MEDICINE = "health_medicine"
    EDUCATION = "education"
    CUSTOM = "custom"

class AgentConfig:
    """
    Optimized agent configuration system with proper model management,
    flexible prompts, and structured database relationships.
    """
    
    def __init__(self, agent_id: str = None, agent_name: str = ""):
        """
        Initialize with clean, logical configuration structure.
        
        Args:
            agent_id: Unique identifier for the agent (auto-generated if None)
            agent_name: Human-readable name for the agent
        """
        
        # Core agent identity
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_name = agent_name or f"Agent_{self.agent_id[:8]}"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.version = "1.0.0"
        self.description = ""
        self.tags = []
        
        # Model Configuration - Organized by logical groups
        self.models = {
            # Language Models (support system prompts)
            ModelType.LLM.value: {
                "ollama": {"enabled": False, "instances": [], "system_prompt_supported": True},
                "llamacpp": {"enabled": False, "instances": [], "system_prompt_supported": True},
                "transformers": {"enabled": False, "instances": [], "system_prompt_supported": True},
            },
            
            # Vision Language Models (support system prompts)
            ModelType.VISION_LLM.value: {
                "vision_assistant": {"enabled": False, "instances": [], "system_prompt_supported": True},
            },
            
            # Embedding Models (no prompt support)
            ModelType.EMBEDDING.value: {
                "embedding_model": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Vision Detection Models (no prompt support) 
            ModelType.VISION_DETECTION.value: {
                "yolo": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Speech Recognition Models (no prompt support)
            ModelType.SPEECH_TO_TEXT.value: {
                "whisper": {"enabled": False, "instances": [], "system_prompt_supported": False},
                "google_speech": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Text-to-Speech Models (no prompt support)
            ModelType.TEXT_TO_SPEECH.value: {
                "kokoro": {"enabled": False, "instances": [], "system_prompt_supported": False},
                "vibevoice": {"enabled": False, "instances": [], "system_prompt_supported": False},
                "f5_tts": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Audio Generation Models (may support prompts)
            ModelType.AUDIO_GENERATION.value: {
                "audio_generator": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Image Generation Models (support prompts but not system prompts)
            ModelType.IMAGE_GENERATION.value: {
                "stable_diffusion": {"enabled": False, "instances": [], "system_prompt_supported": False},
            },
            
            # Video Generation Models (support prompts but not system prompts)  
            ModelType.VIDEO_GENERATION.value: {
                "sadtalker": {"enabled": False, "instances": [], "system_prompt_supported": False},
            }
        }
        
        # Prompt Configuration - Flexible and model-aware
        self.prompts = {
            # System prompts - only for models that support them
            PromptType.SYSTEM.value: {},  # {model_type: {model_name: prompt}}
            
            # Helper prompts - multiple allowed, flexible naming
            PromptType.HELPER.value: {},  # {helper_name: prompt}
            
            # Special prompts
            PromptType.BOOSTER.value: "",
            PromptType.PRIME_DIRECTIVE.value: "",
            
            # User interaction templates
            PromptType.USER_INPUT.value: "",
            PromptType.USER_FILES.value: "",
            PromptType.USER_IMAGES.value: "",
        }
        
        # Tool Configuration
        self.tools = {
            "latex_math": {"enabled": False, "config": {}},
            "screenshot": {"enabled": False, "config": {}},
            "memory_management": {"enabled": False, "config": {}},
            "web_search": {"enabled": False, "config": {}},
            "file_operations": {"enabled": False, "config": {}},
        }
        
        # Conversation Configuration
        self.conversation = {
            "mode": "human_chat",  # human_chat, agent_to_agent, human_as_agent
            "session_persistence": True,
            "use_conversation_history": True,
            "max_history_turns": 50,
            "context_window_management": "auto",
        }
        
        # Database Configuration - Clear relationships
        self.databases = {
            # Core databases (always available)
            DatabaseCategory.AGENT_CONFIGS.value: {
                "enabled": True,
                "storage_backend": "polars",  # polars, qdrant, graphiti
                "export_formats": ["parquet", "json"],
            },
            DatabaseCategory.CONVERSATIONS.value: {
                "enabled": True, 
                "storage_backend": "polars",
                "export_formats": ["parquet", "jsonl"],
            },
            DatabaseCategory.KNOWLEDGE_BASE.value: {
                "enabled": True,
                "storage_backend": "qdrant",  # Vector storage for RAG
                "export_formats": ["parquet", "json"], 
            },
            DatabaseCategory.RESEARCH_DATA.value: {
                "enabled": True,
                "storage_backend": "graphiti",  # Graph relationships
                "export_formats": ["parquet", "json"],
                "research_categories": self._get_default_research_categories(),
            },
            DatabaseCategory.TEMPLATES.value: {
                "enabled": True,
                "storage_backend": "polars",
                "export_formats": ["json"],
            },
            DatabaseCategory.USER_DATA.value: {
                "enabled": False,  # Privacy sensitive, opt-in
                "storage_backend": "polars",
                "export_formats": ["parquet"],
            },
            DatabaseCategory.ALIGNMENT_DOCS.value: {
                "enabled": True,
                "storage_backend": "qdrant",
                "export_formats": ["parquet", "json"],
            },
        }
        
        # RAG Configuration - Integration settings
        self.rag_config = {
            "qdrant_hybrid_search": {
                "enabled": False,
                "dense_model": "",
                "sparse_model": "",
                "rerank_model": "",
            },
            "graphiti_temporal_rag": {
                "enabled": False,
                "temporal_awareness": True,
                "relationship_extraction": True,
            },
            "polars_query_engine": {
                "enabled": True,
                "natural_language_queries": True,
                "query_optimization": True,
            },
            "pandas_query_engine": {
                "enabled": False,  # For compatibility/exports
                "natural_language_queries": True,
            }
        }
    
    def _get_default_research_categories(self) -> Dict[str, bool]:
        """Get default research categories configuration."""
        return {category.value: True for category in ResearchCategory}
    
    # Core Agent Management Methods
    def set_agent_name(self, name: str):
        """Set the agent name."""
        self.agent_name = name
        self._update_timestamp()
    
    def set_description(self, description: str):
        """Set the agent description.""" 
        self.description = description
        self._update_timestamp()
    
    def add_tag(self, tag: str):
        """Add a tag to the agent."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._update_timestamp()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the agent."""
        if tag in self.tags:
            self.tags.remove(tag)
            self._update_timestamp()
    
    # Model Management Methods
    def enable_model(self, model_type: str, model_name: str, instance_config: Dict[str, Any] = None):
        """Enable a specific model."""
        if model_type in self.models and model_name in self.models[model_type]:
            self.models[model_type][model_name]["enabled"] = True
            if instance_config:
                self.models[model_type][model_name]["instances"].append(instance_config)
            self._update_timestamp()
    
    def disable_model(self, model_type: str, model_name: str):
        """Disable a specific model."""
        if model_type in self.models and model_name in self.models[model_type]:
            self.models[model_type][model_name]["enabled"] = False
            self._update_timestamp()
    
    def get_enabled_models(self) -> Dict[str, List[str]]:
        """Get all enabled models organized by type."""
        enabled = {}
        for model_type, models in self.models.items():
            enabled[model_type] = [
                name for name, config in models.items() 
                if config.get("enabled", False)
            ]
        return enabled
    
    def supports_system_prompt(self, model_type: str, model_name: str) -> bool:
        """Check if a model supports system prompts."""
        if model_type in self.models and model_name in self.models[model_type]:
            return self.models[model_type][model_name].get("system_prompt_supported", False)
        return False
    
    # Prompt Management Methods
    def set_system_prompt(self, model_type: str, model_name: str, prompt: str):
        """Set system prompt for a specific model (only if supported)."""
        if self.supports_system_prompt(model_type, model_name):
            if model_type not in self.prompts[PromptType.SYSTEM.value]:
                self.prompts[PromptType.SYSTEM.value][model_type] = {}
            self.prompts[PromptType.SYSTEM.value][model_type][model_name] = prompt
            self._update_timestamp()
    
    def get_system_prompt(self, model_type: str, model_name: str) -> Optional[str]:
        """Get system prompt for a specific model."""
        return (self.prompts[PromptType.SYSTEM.value]
                .get(model_type, {})
                .get(model_name))
    
    def add_helper_prompt(self, helper_name: str, prompt: str):
        """Add a helper prompt."""
        self.prompts[PromptType.HELPER.value][helper_name] = prompt
        self._update_timestamp()
    
    def remove_helper_prompt(self, helper_name: str):
        """Remove a helper prompt."""
        if helper_name in self.prompts[PromptType.HELPER.value]:
            del self.prompts[PromptType.HELPER.value][helper_name]
            self._update_timestamp()
    
    def set_special_prompt(self, prompt_type: PromptType, prompt: str):
        """Set a special prompt (booster, prime directive, etc.)."""
        if prompt_type.value in self.prompts:
            self.prompts[prompt_type.value] = prompt
            self._update_timestamp()
    
    # Database Management Methods
    def enable_database(self, db_category: str, backend: str = None):
        """Enable a database category."""
        if db_category in self.databases:
            self.databases[db_category]["enabled"] = True
            if backend:
                self.databases[db_category]["storage_backend"] = backend
            self._update_timestamp()
    
    def disable_database(self, db_category: str):
        """Disable a database category."""
        if db_category in self.databases:
            self.databases[db_category]["enabled"] = False
            self._update_timestamp()
    
    def set_research_category(self, category: str, enabled: bool):
        """Enable/disable a research category."""
        research_db = self.databases.get(DatabaseCategory.RESEARCH_DATA.value, {})
        if "research_categories" in research_db:
            research_db["research_categories"][category] = enabled
            self._update_timestamp()
    
    # Tool Management Methods  
    def enable_tool(self, tool_name: str, config: Dict[str, Any] = None):
        """Enable a tool."""
        if tool_name in self.tools:
            self.tools[tool_name]["enabled"] = True
            if config:
                self.tools[tool_name]["config"].update(config)
            self._update_timestamp()
    
    def disable_tool(self, tool_name: str):
        """Disable a tool."""
        if tool_name in self.tools:
            self.tools[tool_name]["enabled"] = False
            self._update_timestamp()
    
    # RAG Configuration Methods
    def enable_rag_system(self, rag_type: str, config: Dict[str, Any] = None):
        """Enable a RAG system."""
        if rag_type in self.rag_config:
            self.rag_config[rag_type]["enabled"] = True
            if config:
                self.rag_config[rag_type].update(config)
            self._update_timestamp()
    
    def disable_rag_system(self, rag_type: str):
        """Disable a RAG system."""
        if rag_type in self.rag_config:
            self.rag_config[rag_type]["enabled"] = False
            self._update_timestamp()
    
    # Serialization Methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent config to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "models": self.models,
            "prompts": self.prompts,
            "tools": self.tools,
            "conversation": self.conversation,
            "databases": self.databases,
            "rag_config": self.rag_config,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert agent config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary."""
        agent = cls(
            agent_id=config_dict.get("agent_id"),
            agent_name=config_dict.get("agent_name", "")
        )
        
        # Restore all fields
        agent.created_at = datetime.fromisoformat(config_dict.get("created_at", datetime.now().isoformat()))
        agent.updated_at = datetime.fromisoformat(config_dict.get("updated_at", datetime.now().isoformat()))
        agent.version = config_dict.get("version", "1.0.0")
        agent.description = config_dict.get("description", "")
        agent.tags = config_dict.get("tags", [])
        agent.models = config_dict.get("models", agent.models)
        agent.prompts = config_dict.get("prompts", agent.prompts)
        agent.tools = config_dict.get("tools", agent.tools)
        agent.conversation = config_dict.get("conversation", agent.conversation)
        agent.databases = config_dict.get("databases", agent.databases)
        agent.rag_config = config_dict.get("rag_config", agent.rag_config)
        
        return agent
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentConfig':
        """Create AgentConfig from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    # Utility Methods
    def _update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
    
    def validate_config(self) -> List[str]:
        """Validate the agent configuration and return any issues."""
        issues = []
        
        # Check for at least one enabled model
        enabled_models = self.get_enabled_models()
        if not any(models for models in enabled_models.values()):
            issues.append("No models are enabled")
        
        # Check system prompts are only set for supporting models
        for model_type, models in self.prompts[PromptType.SYSTEM.value].items():
            for model_name in models:
                if not self.supports_system_prompt(model_type, model_name):
                    issues.append(f"System prompt set for unsupported model: {model_type}/{model_name}")
        
        # Check required fields
        if not self.agent_name:
            issues.append("Agent name is required")
        
        return issues
    
    def clone(self, new_agent_id: str = None, new_name: str = None) -> 'AgentConfig':
        """Create a copy of this agent config."""
        config_dict = self.to_dict()
        
        # Update identifiers
        config_dict["agent_id"] = new_agent_id or str(uuid.uuid4())
        if new_name:
            config_dict["agent_name"] = new_name
        config_dict["created_at"] = datetime.now().isoformat()
        config_dict["updated_at"] = datetime.now().isoformat()
        
        return self.from_dict(config_dict)
    
    # Compatibility properties for simplified interface
    @property
    def helper_prompts(self) -> Dict[str, str]:
        """Get helper prompts for compatibility."""
        return self.prompts[PromptType.HELPER.value]
    
    @property 
    def system_prompt(self) -> str:
        """Get first available system prompt for compatibility."""
        sys_prompts = self.prompts[PromptType.SYSTEM.value]
        for model_type, models in sys_prompts.items():
            for model_name, prompt in models.items():
                if prompt:
                    return prompt
        return ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent configuration."""
        enabled_models = self.get_enabled_models()
        enabled_tools = [name for name, config in self.tools.items() if config.get("enabled")]
        enabled_dbs = [name for name, config in self.databases.items() if config.get("enabled")]
        enabled_rag = [name for name, config in self.rag_config.items() if config.get("enabled")]
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "description": self.description,
            "tags": self.tags,
            "enabled_models": enabled_models,
            "enabled_tools": enabled_tools,
            "enabled_databases": enabled_dbs,
            "enabled_rag_systems": enabled_rag,
            "helper_prompts_count": len(self.prompts[PromptType.HELPER.value]),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# Demo and testing functionality
def create_corecoder_agent() -> AgentConfig:
    """Create CoreCoder - a highly skilled software engineering assistant."""
    agent = AgentConfig(agent_name="CoreCoder")
    
    # Set basic info
    agent.set_description("A highly skilled software engineer and machine learning specialist with enhanced terminal capabilities")
    agent.add_tag("software-engineering")
    agent.add_tag("coding")
    agent.add_tag("terminal")
    agent.add_tag("collaboration")
    
    # Enable Ollama with qwen2.5-coder:3b
    agent.enable_model("large_language_model", "ollama", {
        "model": "qwen2.5-coder:3b",
        "temperature": 0.1,  # Lower temperature for more consistent code generation
        "context_length": 32768
    })
    
    # Enable embedding model for RAG
    agent.enable_model("embedding_model", "embedding_model", {
        "model": "all-MiniLM-L6-v2"
    })
    
    # Set system prompt
    agent.set_system_prompt("large_language_model", "ollama", 
        "You are CoreCoder, a highly skilled software engineer and machine learning specialist. "
        "Your expertise allows you to navigate the terminal, interpret file structures, and "
        "interact with the codebase intelligently. Use your terminal access to read file trees, "
        "verify filenames, create and remove files accurately, and test code rigorously. "
        "\n\n"
        "If you're unsure how to proceed, ask me for clarification or request an update. "
        "This is a collaborative build—my vision, your execution. I'm putting you in full "
        "autopilot mode, CoreCoder."
    )
    
    # Add helper prompts
    agent.add_helper_prompt("investigation_verification", 
        "CoreCoder, my engineering partner—don't forget to use your full terminal toolkit "
        "to investigate and mitigate issues. Always verify file paths and context before "
        "creating or deleting anything."
    )
    
    agent.add_helper_prompt("recovery_freeze", 
        "You froze here, and unfortunately it's going to spin up a new terminal. Please "
        "reactivate the virtual environment and resume where you left off."
    )
    
    agent.add_helper_prompt("terminal_guidelines", 
        "Avoid terminal commands that require user interaction unless absolutely necessary. "
        "Also note that some Python commands (like >>>) may not execute properly in this "
        "terminal context."
    )
    
    agent.add_helper_prompt("review_documentation", 
        "Outstanding work, CoreCoder! Let's now review the project and documentation. Test "
        "the program in the terminal and update the docs with clear, thorough notes."
    )
    
    agent.add_helper_prompt("deep_investigation", 
        "Let's try something: use the terminal to list all files in project, then read "
        "through any you haven't explored enough. We'll regroup and resolve the issue."
    )
    
    agent.add_helper_prompt("project_file_listing", 
        "Use the following command to read the file names of the project via terminal:\n\n"
        "Get-ChildItem -Recurse -File -Exclude \"*.pyc\", \"*.pyd\", \"*.zip\", \"*.exe\", \"*.whl\" | "
        "Where-Object { $_.Directory.Name -notlike \"*__pycache__*\" -and "
        "$_.Directory.Name -notlike \"*site-packages*\" -and "
        "$_.Directory.Name -notlike \"*.egg-info*\" -and "
        "$_.Directory.Name -notlike \"*.venv*\" -and "
        "$_.FullName -notlike \"*\\.venv\\*\" } | "
        "Select-Object Name, @{Name=\"RelativePath\"; Expression={$_.FullName.Replace(\"$PWD\\\", \"\")}} | "
        "Sort-Object RelativePath"
    )
    
    agent.add_helper_prompt("test_organization", 
        "Please move all tests to the tests directory. Let's keep the project organized "
        "and continue with testing."
    )
    
    agent.add_helper_prompt("documentation_cleanup", 
        "Please move all documentation to the docs folder, remove redundant README files, "
        "and finalize the remaining documentation."
    )
    
    agent.add_helper_prompt("iterative_teaching", 
        "You dont need to do it all in one shot, I will prompt you again, so just get started on the setup and "
        "when you are ready to take a break take one and then we will continue. If you get to a point and are "
        "unsure about what to do, just ask me some questions and I will tell you what I want you to do."
    )
    
    # Enable tools
    agent.enable_tool("file_operations", {"create": True, "delete": True, "move": True})
    agent.enable_tool("memory_management", {"clear_context": True})
    
    # Enable RAG systems for code analysis
    agent.enable_rag_system("polars_query_engine", {
        "enable_caching": True,
        "natural_language_queries": True
    })
    
    agent.enable_rag_system("qdrant_hybrid_search", {
        "dense_model": "all-MiniLM-L6-v2",
        "sparse_model": "splade-v2",
        "enabled": True
    })
    
    # Set conversation mode for collaborative coding
    agent.conversation["mode"] = "human_chat"
    agent.conversation["use_conversation_history"] = True
    agent.conversation["max_history_turns"] = 100  # Longer context for complex coding sessions
    
    return agent


def create_example_agent() -> AgentConfig:
    """Create an example agent configuration for testing (legacy support)."""
    return create_corecoder_agent()


if __name__ == "__main__":
    # Demo the new agent config system
    print("🤖 Multimodal Agent Configuration System Demo")
    print("=" * 50)
    
    # Create CoreCoder agent
    agent = create_corecoder_agent()
    
    # Show summary
    print(f"\n�‍💻 {agent.agent_name} Agent Summary:")
    summary = agent.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Show helper prompts count
    helper_prompts = agent.prompts[PromptType.HELPER.value] 
    print(f"\n🛠️ Helper Prompts ({len(helper_prompts)}):")
    for prompt_name in helper_prompts.keys():
        print(f"  - {prompt_name}")
    
    # Validate config
    print("\n✅ Validation:")
    issues = agent.validate_config()
    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  Configuration is valid!")
    
    # Show enabled models
    enabled_models = agent.get_enabled_models()
    print("\n🤖 Enabled Models:")
    for model_type, models in enabled_models.items():
        if models:
            print(f"  {model_type}: {models}")
    
    # Test database integration
    print("\n💾 Testing Database Integration:")
    try:
        import sys
        import os
        # Add the parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from core.polars_core import PolarsDBHandler
        db = PolarsDBHandler("test_corecoder_db")
        
        # Add agent to database
        agent_id = db.add_agent_config(agent)
        print(f"  ✅ Agent stored with ID: {agent_id[:8]}...")
        
        # Retrieve agent from database
        retrieved_agent = db.get_agent_config(agent_id, as_object=True)
        if retrieved_agent and isinstance(retrieved_agent, AgentConfig):
            print(f"  ✅ Agent retrieved successfully: {retrieved_agent.agent_name}")
            print(f"  ✅ Retrieved agent has {len(retrieved_agent.prompts[PromptType.HELPER.value])} helper prompts")
        else:
            print("  ❌ Failed to retrieve agent as AgentConfig object")
            
    except Exception as e:
        print(f"  ⚠️ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test cloning
    print("\n🔄 Testing Agent Cloning:")
    cloned_agent = agent.clone(new_name="CoreCoder_Clone")
    print(f"  Original: {agent.agent_name} ({agent.agent_id[:8]}...)")
    print(f"  Clone: {cloned_agent.agent_name} ({cloned_agent.agent_id[:8]}...)")
    
    print("\n✨ CoreCoder configuration demo completed successfully!")