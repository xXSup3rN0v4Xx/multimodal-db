# Agent Configuration System

## Overview

The **Agent Configuration System** is the core framework for defining, managing, and persisting AI agent personalities, capabilities, and behaviors in the multimodal-db project. It provides a flexible, type-safe, and extensible way to configure agents that can work with multiple AI models, databases, and RAG (Retrieval-Augmented Generation) systems.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture](#architecture)
3. [Configuration Components](#configuration-components)
4. [Usage Guide](#usage-guide)
5. [Advanced Features](#advanced-features)
6. [Examples](#examples)
7. [Best Practices](#best-practices)

---

## Core Concepts

### What is an AgentConfig?

An `AgentConfig` is a complete specification of an AI agent that includes:

- **Identity**: Unique ID, name, description, and tags
- **Models**: Which AI models the agent can use (LLMs, vision models, embeddings, etc.)
- **Prompts**: System prompts, helper prompts, and special instructions
- **Tools**: Enabled capabilities (file operations, web search, LaTeX math, etc.)
- **Databases**: Which database backends to use for storage
- **RAG Systems**: Configuration for retrieval-augmented generation
- **Conversation Settings**: How the agent interacts with users

### Key Design Principles

1. **Flexibility**: Support multiple model types with different capabilities
2. **Type Safety**: Use enumerations to prevent configuration errors
3. **Model-Aware**: Only allow system prompts for models that support them
4. **Persistence**: Easy serialization to JSON/Parquet for database storage
5. **Extensibility**: Simple to add new model types, tools, or features

---

## Architecture

### Class Hierarchy

```
AgentConfig (main configuration class)
├── Enumerations
│   ├── MediaType (text, audio, image, video, document, embedding)
│   ├── ModelType (LLM, Vision LLM, Embedding, Vision Detection, TTS, STT, etc.)
│   ├── PromptType (system, helper, booster, prime_directive, user_input, etc.)
│   ├── DatabaseCategory (agent_configs, conversations, knowledge_base, etc.)
│   └── ResearchCategory (mathematics, physics, AI/ML, etc.)
└── Core Attributes
    ├── models: Model configuration dictionary
    ├── prompts: Prompt configuration dictionary
    ├── tools: Tool configuration dictionary
    ├── databases: Database configuration dictionary
    └── rag_config: RAG system configuration dictionary
```

### Data Structure

The `AgentConfig` class uses nested dictionaries organized by logical categories:

```python
agent = AgentConfig(agent_name="MyAgent")

# Structure examples:
agent.models[ModelType.LLM.value]["ollama"] = {
    "enabled": True,
    "instances": [...],
    "system_prompt_supported": True
}

agent.prompts[PromptType.HELPER.value]["my_helper"] = "Helper prompt text..."

agent.databases[DatabaseCategory.KNOWLEDGE_BASE.value] = {
    "enabled": True,
    "storage_backend": "qdrant",
    "export_formats": ["parquet", "json"]
}
```

---

## Configuration Components

### 1. Model Configuration

The system supports nine model types, each with specific capabilities:

#### Language Models (Support System Prompts)
- **LLM** (`large_language_model`): Text-based language models
  - Backends: `ollama`, `llamacpp`, `transformers`
  - Example: Qwen2.5-coder, Llama, GPT models

#### Vision Language Models (Support System Prompts)
- **Vision LLM** (`vision_language_model`): Multimodal vision+text models
  - Backends: `vision_assistant`
  - Example: LLaVA, GPT-4 Vision, Claude with images

#### Specialized Models (No System Prompt Support)
- **Embedding** (`embedding_model`): Text-to-vector embeddings
- **Vision Detection** (`vision_detection`): Object detection models (YOLO, etc.)
- **Speech-to-Text** (`speech_to_text`): Audio transcription (Whisper, Google Speech)
- **Text-to-Speech** (`text_to_speech`): Voice synthesis (Kokoro, VibeVoice, F5-TTS)
- **Audio Generation** (`audio_generation`): Music/audio generation
- **Image Generation** (`image_generation`): Stable Diffusion, DALL-E
- **Video Generation** (`video_generation`): SadTalker, video synthesis

#### Example: Enabling Models

```python
# Enable Ollama LLM
agent.enable_model("large_language_model", "ollama", {
    "model": "qwen2.5-coder:3b",
    "temperature": 0.1,
    "context_length": 32768
})

# Enable embedding model
agent.enable_model("embedding_model", "embedding_model", {
    "model": "all-MiniLM-L6-v2"
})

# Enable vision detection
agent.enable_model("vision_detection", "yolo", {
    "model": "yolov8n.pt",
    "confidence": 0.5
})
```

### 2. Prompt Configuration

The system uses a **model-aware prompt system** that respects which models support what types of prompts.

#### Prompt Types

1. **System Prompts** (`PromptType.SYSTEM`)
   - Only for LLMs and Vision LLMs
   - Defines core behavior and personality
   - Per-model configuration: `{model_type: {model_name: prompt}}`

2. **Helper Prompts** (`PromptType.HELPER`)
   - Flexible, multiple allowed
   - Named instructions for specific scenarios
   - Example: `investigation_verification`, `recovery_freeze`

3. **Special Prompts**
   - **Booster** (`PromptType.BOOSTER`): Performance enhancement instructions
   - **Prime Directive** (`PromptType.PRIME_DIRECTIVE`): Core unchangeable behavior
   - **User Input** (`PromptType.USER_INPUT`): User interaction templates
   - **User Files** (`PromptType.USER_FILES`): File handling templates
   - **User Images** (`PromptType.USER_IMAGES`): Image handling templates

#### Example: Setting Prompts

```python
# Set system prompt (only for LLM/Vision LLM)
agent.set_system_prompt("large_language_model", "ollama", 
    "You are a helpful coding assistant specialized in Python."
)

# Add helper prompts
agent.add_helper_prompt("debugging", 
    "When debugging, always check logs and verify assumptions."
)

agent.add_helper_prompt("code_review",
    "Review code for efficiency, readability, and best practices."
)

# Set special prompts
agent.set_special_prompt(PromptType.PRIME_DIRECTIVE,
    "Always prioritize user safety and data privacy."
)
```

### 3. Database Configuration

Agents can use multiple database backends for different purposes:

#### Database Categories

1. **AGENT_CONFIGS**: Store agent configurations (Polars)
2. **CONVERSATIONS**: Store chat history (Polars)
3. **KNOWLEDGE_BASE**: Vector embeddings for RAG (Qdrant)
4. **RESEARCH_DATA**: Graph-based knowledge (Graphiti)
5. **TEMPLATES**: Reusable prompt templates (Polars)
6. **USER_DATA**: User-specific data (opt-in, privacy-sensitive)
7. **ALIGNMENT_DOCS**: Alignment and safety documents (Qdrant)

#### Storage Backends

- **Polars**: High-performance DataFrame storage (Parquet files)
- **Qdrant**: Vector database for embeddings and semantic search
- **Graphiti**: Graph database for temporal knowledge graphs

#### Example: Database Configuration

```python
# Enable knowledge base with Qdrant backend
agent.enable_database("knowledge_documents", backend="qdrant")

# Enable research data with Graphiti backend
agent.enable_database("research_collections", backend="graphiti")

# Configure research categories
agent.set_research_category("computer_science", enabled=True)
agent.set_research_category("artificial_intelligence_machine_learning", enabled=True)

# Privacy-sensitive: disable user data by default
agent.disable_database("user_personal_data")
```

### 4. RAG Configuration

The system supports multiple RAG (Retrieval-Augmented Generation) approaches:

#### RAG Systems

1. **Qdrant Hybrid Search**
   - Dense embeddings + Sparse vectors (BM25-style)
   - Neural reranking
   - Best for: Document retrieval, semantic search

2. **Graphiti Temporal RAG**
   - Time-aware knowledge graphs
   - Entity and relationship extraction
   - Best for: Conversational memory, complex reasoning

3. **Polars Query Engine**
   - Natural language queries on DataFrames
   - High-speed analytics
   - Best for: Data analysis, structured queries

4. **Pandas Query Engine**
   - Compatibility with Pandas workflows
   - Export/analysis tasks
   - Best for: Legacy code, data science workflows

#### Example: RAG Configuration

```python
# Enable hybrid search
agent.enable_rag_system("qdrant_hybrid_search", {
    "dense_model": "BAAI/bge-small-en-v1.5",
    "sparse_model": "splade-v2",
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "enabled": True
})

# Enable temporal RAG
agent.enable_rag_system("graphiti_temporal_rag", {
    "temporal_awareness": True,
    "relationship_extraction": True,
    "enabled": True
})

# Enable Polars query engine
agent.enable_rag_system("polars_query_engine", {
    "natural_language_queries": True,
    "query_optimization": True,
    "enable_caching": True
})
```

### 5. Tool Configuration

Tools extend agent capabilities with specific functions:

#### Available Tools

- **latex_math**: Mathematical notation and equation rendering
- **screenshot**: Screen capture capabilities
- **memory_management**: Context clearing and memory optimization
- **web_search**: Internet search integration
- **file_operations**: Create, read, update, delete files

#### Example: Tool Configuration

```python
# Enable file operations
agent.enable_tool("file_operations", {
    "create": True,
    "delete": True,
    "move": True,
    "max_file_size": "10MB"
})

# Enable web search
agent.enable_tool("web_search", {
    "search_engine": "duckduckgo",
    "max_results": 10,
    "safe_search": True
})
```

### 6. Conversation Configuration

Control how the agent interacts with users:

```python
agent.conversation = {
    "mode": "human_chat",  # or "agent_to_agent", "human_as_agent"
    "session_persistence": True,
    "use_conversation_history": True,
    "max_history_turns": 50,
    "context_window_management": "auto"
}
```

---

## Usage Guide

### Creating an Agent

```python
from multimodal_db.core.agent_configs.base_agent_config import (
    AgentConfig, ModelType, PromptType, DatabaseCategory
)

# Create basic agent
agent = AgentConfig(agent_name="MyAssistant")

# Set description and tags
agent.set_description("A helpful AI assistant for coding tasks")
agent.add_tag("coding")
agent.add_tag("python")
agent.add_tag("helpful")
```

### Configuring Models

```python
# Enable LLM
agent.enable_model(ModelType.LLM.value, "ollama", {
    "model": "qwen2.5-coder:3b",
    "temperature": 0.7
})

# Enable embeddings
agent.enable_model(ModelType.EMBEDDING.value, "embedding_model", {
    "model": "all-MiniLM-L6-v2"
})

# Set system prompt
agent.set_system_prompt(ModelType.LLM.value, "ollama",
    "You are a helpful coding assistant."
)
```

### Saving and Loading

```python
# Save to JSON
json_str = agent.to_json(indent=2)
with open("my_agent.json", "w") as f:
    f.write(json_str)

# Load from JSON
with open("my_agent.json", "r") as f:
    json_str = f.read()
agent = AgentConfig.from_json(json_str)

# Convert to dictionary
config_dict = agent.to_dict()

# Load from dictionary
agent = AgentConfig.from_dict(config_dict)
```

### Database Integration

```python
from multimodal_db.core.dbs.polars_db import PolarsDB

# Create database
db = PolarsDB(db_path="my_agents")

# Add agent to database
agent_id = db.add_agent(agent, name="MyAssistant")

# Retrieve agent
retrieved_agent = db.get_agent(agent_id)

# List all agents
agents = db.list_agents()
```

### Validation

```python
# Validate configuration
issues = agent.validate_config()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid!")
```

---

## Advanced Features

### 1. Cloning Agents

Create copies of agents with modified identities:

```python
# Clone agent with new name
cloned_agent = agent.clone(
    new_name="MyAssistant_V2",
    new_agent_id=None  # Auto-generate new ID
)
```

### 2. Agent Summary

Get a high-level overview of agent configuration:

```python
summary = agent.get_summary()
print(summary)
# Output:
# {
#     "agent_id": "abc123...",
#     "agent_name": "MyAssistant",
#     "enabled_models": {...},
#     "enabled_tools": [...],
#     "enabled_databases": [...],
#     "helper_prompts_count": 5,
#     ...
# }
```

### 3. Dynamic Model Checking

```python
# Check if model supports system prompts
if agent.supports_system_prompt(ModelType.LLM.value, "ollama"):
    agent.set_system_prompt(ModelType.LLM.value, "ollama", "...")

# Get all enabled models
enabled = agent.get_enabled_models()
for model_type, models in enabled.items():
    print(f"{model_type}: {models}")
```

### 4. Research Categories

For agents doing research tasks:

```python
# Enable specific research domains
from multimodal_db.core.agent_configs.base_agent_config import ResearchCategory

agent.set_research_category(ResearchCategory.AI_ML.value, enabled=True)
agent.set_research_category(ResearchCategory.COMPUTER_SCIENCE.value, enabled=True)
agent.set_research_category(ResearchCategory.MATHEMATICS.value, enabled=True)

# Custom research category
agent.set_research_category(ResearchCategory.CUSTOM.value, enabled=True)
```

---

## Examples

### Example 1: Simple Chat Agent

```python
agent = AgentConfig(agent_name="ChatBot")
agent.set_description("A friendly chatbot")

# Enable LLM only
agent.enable_model(ModelType.LLM.value, "ollama", {
    "model": "llama2:7b",
    "temperature": 0.9  # More creative
})

# Simple system prompt
agent.set_system_prompt(ModelType.LLM.value, "ollama",
    "You are a friendly chatbot. Be helpful and conversational."
)

# Enable conversation history
agent.conversation["use_conversation_history"] = True
agent.conversation["max_history_turns"] = 20
```

### Example 2: Coding Assistant (CoreCoder)

```python
agent = AgentConfig(agent_name="CoreCoder")
agent.set_description("Advanced coding assistant with terminal access")
agent.add_tag("coding")
agent.add_tag("terminal")

# Enable coding-optimized LLM
agent.enable_model(ModelType.LLM.value, "ollama", {
    "model": "qwen2.5-coder:3b",
    "temperature": 0.1,  # Low temp for consistent code
    "context_length": 32768
})

# Enable embeddings for code search
agent.enable_model(ModelType.EMBEDDING.value, "embedding_model", {
    "model": "all-MiniLM-L6-v2"
})

# Detailed system prompt
agent.set_system_prompt(ModelType.LLM.value, "ollama",
    "You are CoreCoder, an expert software engineer. "
    "You have terminal access and can navigate codebases. "
    "Always verify file paths before operations."
)

# Multiple helper prompts
agent.add_helper_prompt("verification",
    "Always verify file paths and context before creating or deleting files."
)

agent.add_helper_prompt("testing",
    "Test code rigorously before marking tasks complete."
)

# Enable tools
agent.enable_tool("file_operations", {"create": True, "delete": True})
agent.enable_tool("memory_management", {"clear_context": True})

# Enable RAG for code analysis
agent.enable_rag_system("polars_query_engine", {
    "natural_language_queries": True,
    "enable_caching": True
})

# Longer conversation history for complex projects
agent.conversation["max_history_turns"] = 100
```

### Example 3: Multimodal Research Agent

```python
agent = AgentConfig(agent_name="ResearchBot")
agent.set_description("Multimodal research assistant")
agent.add_tag("research")
agent.add_tag("multimodal")

# Enable multiple model types
agent.enable_model(ModelType.LLM.value, "ollama", {
    "model": "mistral:7b"
})

agent.enable_model(ModelType.VISION_LLM.value, "vision_assistant", {
    "model": "llava:13b"
})

agent.enable_model(ModelType.EMBEDDING.value, "embedding_model", {
    "model": "BAAI/bge-base-en-v1.5"
})

# System prompts for both LLM types
agent.set_system_prompt(ModelType.LLM.value, "ollama",
    "You are a research assistant. Analyze documents and synthesize information."
)

agent.set_system_prompt(ModelType.VISION_LLM.value, "vision_assistant",
    "Analyze images and diagrams from research papers."
)

# Enable knowledge bases
agent.enable_database(DatabaseCategory.KNOWLEDGE_BASE.value, backend="qdrant")
agent.enable_database(DatabaseCategory.RESEARCH_DATA.value, backend="graphiti")

# Configure research categories
agent.set_research_category(ResearchCategory.AI_ML.value, True)
agent.set_research_category(ResearchCategory.COMPUTER_SCIENCE.value, True)

# Enable advanced RAG
agent.enable_rag_system("qdrant_hybrid_search", {
    "dense_model": "BAAI/bge-base-en-v1.5",
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
})

agent.enable_rag_system("graphiti_temporal_rag", {
    "temporal_awareness": True,
    "relationship_extraction": True
})
```

### Example 4: Voice Assistant

```python
agent = AgentConfig(agent_name="VoiceAssistant")
agent.set_description("Voice-enabled AI assistant")

# Enable voice models
agent.enable_model(ModelType.SPEECH_TO_TEXT.value, "whisper", {
    "model": "whisper-base"
})

agent.enable_model(ModelType.TEXT_TO_SPEECH.value, "kokoro", {
    "voice": "af_sarah",
    "speed": 1.0
})

agent.enable_model(ModelType.LLM.value, "ollama", {
    "model": "llama2:7b"
})

# Configure for voice interaction
agent.conversation["mode"] = "human_chat"
agent.set_special_prompt(PromptType.USER_INPUT,
    "Process spoken input and respond naturally."
)
```

---

## Best Practices

### 1. Model Configuration

- **Use appropriate temperatures**: Low (0.1-0.3) for coding/facts, high (0.7-1.0) for creativity
- **Match context length to task**: Longer for complex projects, shorter for simple chats
- **Enable only needed models**: Each model adds overhead

### 2. Prompt Engineering

- **Be specific in system prompts**: Clear instructions yield better results
- **Use helper prompts for scenarios**: Create helpers for common situations
- **Test prompts iteratively**: Refine based on agent behavior
- **Keep prompts focused**: Don't try to cover everything in one prompt

### 3. Database Selection

- **Polars**: Structured data, agent configs, conversations
- **Qdrant**: Embeddings, semantic search, RAG retrieval
- **Graphiti**: Knowledge graphs, temporal relationships, complex reasoning

### 4. RAG Configuration

- **Start simple**: Enable Polars query engine first
- **Add hybrid search**: For better retrieval quality
- **Use Graphiti**: When temporal or relationship context matters
- **Tune parameters**: Adjust chunk sizes, embedding models, reranking

### 5. Validation and Testing

```python
# Always validate before deployment
issues = agent.validate_config()
assert len(issues) == 0, f"Config issues: {issues}"

# Test serialization round-trip
json_str = agent.to_json()
loaded_agent = AgentConfig.from_json(json_str)
assert loaded_agent.agent_name == agent.agent_name

# Verify enabled models
enabled = agent.get_enabled_models()
assert ModelType.LLM.value in enabled
```

### 6. Version Control

- **Save configs to files**: Keep JSON configs in version control
- **Document changes**: Add comments explaining configuration choices
- **Use descriptive names**: Make agent purposes clear
- **Tag versions**: Use tags to track different agent iterations

### 7. Performance Optimization

- **Limit conversation history**: Balance context vs. memory usage
- **Use efficient embeddings**: Smaller models for better speed
- **Enable caching**: In Polars query engine and RAG systems
- **Batch operations**: When processing multiple agents

---

## Integration with Databases

### Polars Integration

```python
from multimodal_db.core.dbs.polars_db import PolarsDB

db = PolarsDB(db_path="agents")

# Add agent
agent_id = db.add_agent(agent, name="MyAgent")

# Get agent (returns AgentConfig object)
agent = db.get_agent(agent_id)

# List agents
agents = db.list_agents()

# Add conversation
db.add_message(agent_id, role="user", content="Hello")
db.add_message(agent_id, role="assistant", content="Hi there!")

# Get conversation history
messages = db.get_messages(agent_id, limit=50)
```

### Multimodal Database Integration

```python
from multimodal_db.core.dbs.multimodal_db import MultimodalDB

db = MultimodalDB(db_path="multimodal")

# Add agent with full capabilities
agent_id = db.add_agent(agent)

# Store media for agent
media_id = db.store_media(
    media_data=image_bytes,
    media_type=MediaType.IMAGE,
    agent_id=agent_id,
    filename="screenshot.png"
)

# Export agent with all data
export_path = db.export_agent_full(agent_id, "exports/my_agent")
```

### Vector Database Integration

```python
from multimodal_db.core.dbs.vector_db import QdrantVectorDB

vector_db = QdrantVectorDB(persist_path="agent_vectors")

# Store agent knowledge
vector_db.store_agent_knowledge(
    agent_id=agent.agent_id,
    text="Python is a programming language",
    embedding=embedding_vector,
    content_type="fact"
)

# Search agent knowledge
results = vector_db.search_agent_knowledge(
    agent_id=agent.agent_id,
    query_embedding=query_vector,
    limit=10
)
```

---

## API and CLI Integration

### REST API

Agents can be managed through the FastAPI endpoints:

```python
# POST /agents - Create agent
# GET /agents/{agent_id} - Get agent
# PUT /agents/{agent_id} - Update agent
# DELETE /agents/{agent_id} - Delete agent
# GET /agents - List all agents
```

### CLI Commands

```bash
# Create agent from config file
python -m multimodal_db.cli.cli agent create --config agent.json

# List agents
python -m multimodal_db.cli.cli agent list

# Get agent details
python -m multimodal_db.cli.cli agent get <agent_id>

# Export agent
python -m multimodal_db.cli.cli agent export <agent_id> --output exports/
```

---

## Troubleshooting

### Common Issues

1. **"System prompt set for unsupported model"**
   - Solution: Check `supports_system_prompt()` before setting system prompts
   - Only LLMs and Vision LLMs support system prompts

2. **"No models are enabled"**
   - Solution: Enable at least one model using `enable_model()`

3. **Serialization errors**
   - Solution: Ensure all custom objects are JSON-serializable
   - Use `to_dict()` method for custom conversions

4. **Database connection failures**
   - Solution: Verify database paths exist and have write permissions
   - Check that database backends (Qdrant, Graphiti) are installed

### Debug Tips

```python
# Print agent summary
summary = agent.get_summary()
print(json.dumps(summary, indent=2))

# Validate configuration
issues = agent.validate_config()
for issue in issues:
    print(f"Issue: {issue}")

# Check enabled models
enabled = agent.get_enabled_models()
print("Enabled models:", enabled)

# Inspect prompts
print("System prompts:", agent.prompts[PromptType.SYSTEM.value])
print("Helper prompts:", list(agent.prompts[PromptType.HELPER.value].keys()))
```

---

## Future Enhancements

Planned features for the agent configuration system:

1. **Agent Templates**: Pre-built configurations for common use cases
2. **Configuration Inheritance**: Base configs that agents can extend
3. **Dynamic Model Loading**: Hot-swap models without recreating agent
4. **Performance Metrics**: Track agent performance and optimize
5. **Multi-Agent Orchestration**: Coordinate multiple agents on complex tasks
6. **Fine-tuning Integration**: Connect agents to fine-tuned models
7. **Plugin System**: Third-party model/tool integrations

---

## Conclusion

The Agent Configuration System provides a powerful, flexible framework for defining AI agents with:

- ✅ **Type-safe configuration** using Python enumerations
- ✅ **Model-aware prompt management** respecting capabilities
- ✅ **Multiple database backends** for different data types
- ✅ **Advanced RAG integration** for knowledge retrieval
- ✅ **Easy serialization** to JSON/Parquet for persistence
- ✅ **Validation and error checking** to catch issues early

Use this system to create agents ranging from simple chatbots to complex multimodal research assistants with full RAG capabilities.

For more information, see:
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture overview
- [API.md](./api_docs/API.md) - REST API documentation
- [CLI.md](./cli_docs/CLI.md) - CLI usage guide
- [LIBRARY.md](./library_docs/LIBRARY.md) - Python library reference
