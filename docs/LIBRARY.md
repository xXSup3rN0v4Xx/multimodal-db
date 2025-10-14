# Python Library Reference

Complete guide for using Multimodal-DB as a Python library in your scripts.

> **Note**: This documents the Python library interface (classes, methods, functions you import).  
> For the REST API (HTTP endpoints), see [API.md](API.md).

## Installation

```bash
pip install -r requirements.txt
```

## Importing Components

```python
# Core components
from multimodal_db.core import (
    AgentConfig, 
    ModelType, 
    MediaType,
    create_corecoder_agent,
    create_multimodal_agent,
    MultimodalDB,
    QdrantVectorDB,
    SimpleOllamaClient
)
```

---

## AgentConfig

Lightweight agent configuration with multimodal support.

### Creating Agents

```python
from multimodal_db.core import create_corecoder_agent, AgentConfig, ModelType

# Quick creation - CoreCoder agent
agent = create_corecoder_agent(name="my_coder")

# Quick creation - Multimodal agent
agent = create_multimodal_agent(name="my_assistant")

# Custom agent from scratch
agent = AgentConfig(
    agent_name="custom_agent",
    description="My specialized AI agent",
    tags=["tag1", "tag2", "nlp"]
)
```

### Working with Agent Configuration

```python
# Add helper prompts
agent.add_helper_prompt("style", "Write clean, documented code")
agent.add_helper_prompt("format", "Use type hints and docstrings")

# Set system prompt
agent.set_system_prompt("You are an expert Python developer")

# Add model
agent.add_model(
    name="qwen2.5-coder:3b",
    model_type=ModelType.QWEN_CODER_3B,
    enabled=True
)

# Access configuration
print(f"Agent ID: {agent.agent_id}")
print(f"Name: {agent.agent_name}")
print(f"System Prompt: {agent.system_prompt}")
print(f"Helper Prompts: {agent.helper_prompts}")
print(f"Models: {agent.models}")
print(f"Tags: {agent.tags}")

# Serialize to dict
config_dict = agent.to_dict()

# Deserialize from dict
agent2 = AgentConfig.from_dict(config_dict)
```

### ModelType Enum

```python
from multimodal_db.core import ModelType

# Language models
ModelType.LLM              # General language model
ModelType.EMBEDDING        # Embedding model
ModelType.QWEN_CODER_3B   # Specific Ollama model

# Vision models
ModelType.VISION_LLM       # Vision language model
ModelType.VISION_DETECTION # Object detection
ModelType.IMAGE_GENERATION # Image generation

# Audio models
ModelType.SPEECH_TO_TEXT   # Speech recognition
ModelType.TEXT_TO_SPEECH   # Speech synthesis
ModelType.AUDIO_GENERATION # Audio generation

# Video models
ModelType.VIDEO_GENERATION # Video generation
ModelType.VIDEO_ANALYSIS   # Video understanding
```

### MediaType Enum

```python
from multimodal_db.core import MediaType

MediaType.TEXT      # Text content
MediaType.EMBEDDING # Vector embeddings
MediaType.IMAGE     # Images
MediaType.AUDIO     # Audio files
MediaType.VIDEO     # Video files
MediaType.DOCUMENT  # Documents (PDF, etc.)
```

---

## MultimodalDB

High-performance Polars-based database for agents and content.

### Initialization

```python
from multimodal_db.core import MultimodalDB

# Default path (data/multimodal_db)
db = MultimodalDB()

# Custom relative path (prepends "data/")
db = MultimodalDB(db_path="my_custom_db")

# Absolute path
db = MultimodalDB(db_path="/absolute/path/to/db")
```

### Agent Operations

```python
# Store agent
agent_id = db.store_agent(agent)
print(f"Stored agent: {agent_id}")

# Alias: add_agent (same as store_agent)
agent_id = db.add_agent(agent)

# Retrieve agent
retrieved_agent = db.get_agent(agent_id)
if retrieved_agent:
    print(f"Found: {retrieved_agent.agent_name}")

# List all agents (summary)
agents = db.list_agents()
for agent in agents:
    print(f"- {agent['name']}: {agent['agent_id']}")

# List agents with full configuration
agents_full = db.list_agents(include_full_config=True)
for agent in agents_full:
    print(f"- {agent['name']}")
    print(f"  Helper Prompts: {agent.get('helper_prompts', {})}")
    print(f"  System Prompt: {agent.get('system_prompt', '')}")
    print(f"  Models: {agent.get('models', [])}")

# Update agent
agent.description = "Updated description"
db.update_agent(agent)

# Delete agent
success = db.delete_agent(agent_id)
print(f"Deleted: {success}")
```

### Content Operations

```python
from multimodal_db.core import MediaType

# Store text content
content_id = db.store_content(
    agent_id=agent_id,
    content="Important information here",
    media_type=MediaType.TEXT,
    metadata={"category": "notes", "priority": "high"}
)

# Store with different media types
content_id = db.store_content(
    agent_id=agent_id,
    content="path/to/image.png",
    media_type=MediaType.IMAGE,
    metadata={"source": "camera", "resolution": "1920x1080"}
)

# Retrieve content by ID
content = db.get_content(content_id)
print(content)

# Search content
results = db.search_content(
    agent_id=agent_id,            # Optional: filter by agent
    media_type=MediaType.TEXT      # Optional: filter by type
)

for item in results:
    print(f"- {item['media_type']}: {item['content'][:50]}...")
```

### Message/Conversation Operations

```python
# Store message
message_id = db.store_message(
    agent_id=agent_id,
    role="user",
    content="Hello, how are you?",
    session_id="session_123",
    metadata={"source": "web"}
)

# Get conversation history
messages = db.get_messages(
    agent_id=agent_id,
    session_id="session_123",  # Optional
    limit=50
)

for msg in messages:
    print(f"{msg['role']}: {msg['content']}")
```

### Database Statistics

```python
# Get comprehensive stats
stats = db.get_stats()

print(f"Total agents: {stats['agents']['total']}")
print(f"Total content: {stats['content']['total']}")
print(f"Content by type: {stats['content']['by_type']}")
print(f"Database size: {stats['database']['size_mb']} MB")
```

### Manual Save

```python
# Database auto-saves, but you can force save
db.save()
```

---

## QdrantVectorDB

Vector database with 6 specialized collections for embeddings.

### Initialization

```python
from multimodal_db.core import QdrantVectorDB

# Default path (data/qdrant/vectors)
vector_db = QdrantVectorDB()

# Custom path
vector_db = QdrantVectorDB(persist_path="my_vectors")

# Initialize all 6 collections
vector_db.initialize_collections()
```

### Available Collections

```python
# Text embeddings (768-dimensional)
"text_embeddings"

# Image embeddings (512-dimensional)
"image_embeddings"

# Audio embeddings (768-dimensional)
"audio_embeddings"

# Video embeddings (512-dimensional)
"video_embeddings"

# Agent-specific knowledge (768-dimensional)
"agent_knowledge"

# Cross-modal fusion (768-dimensional)
"multimodal_fusion"
```

### Storing Embeddings

```python
import numpy as np

# Generate or load embedding (768-dim for text)
embedding = np.random.rand(768).tolist()

# Store in collection
point_id = vector_db.store_embedding(
    collection="text_embeddings",
    vector=embedding,
    metadata={
        "agent_id": agent_id,
        "content_id": content_id,
        "text": "Original text content",
        "source": "document.pdf"
    }
)

print(f"Stored vector: {point_id}")
```

### Searching Vectors

```python
# Generate query embedding
query_embedding = np.random.rand(768).tolist()

# Search for similar vectors
results = vector_db.search_similar(
    collection="text_embeddings",
    query_vector=query_embedding,
    limit=10,
    score_threshold=0.7  # Optional: minimum similarity score
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Metadata: {result['payload']}")
    print(f"Point ID: {result['id']}")
```

### Filtered Search

```python
# Search with metadata filters
results = vector_db.search_similar(
    collection="text_embeddings",
    query_vector=query_embedding,
    limit=10,
    filters={"agent_id": agent_id}  # Only from specific agent
)
```

### Collection Management

```python
# List all collections
collections = vector_db.list_collections()
print(f"Collections: {collections}")

# Get collection stats
stats = vector_db.get_stats()
for collection, info in stats.items():
    print(f"{collection}: {info['vectors_count']} vectors")

# Create custom collection
vector_db.create_collection(
    name="custom_embeddings",
    vector_size=384,
    distance="Cosine"  # or "Euclidean", "Dot"
)

# Check if collection exists
exists = vector_db.collection_exists("text_embeddings")
```

---

## SimpleOllamaClient

Integration with Ollama for local AI chat.

### Initialization

```python
from multimodal_db.core import SimpleOllamaClient

# Default model (qwen2.5-coder:3b)
client = SimpleOllamaClient()

# Custom model and timeout
client = SimpleOllamaClient(
    model="llama2:7b",
    timeout=60
)

# Check availability
if client.available:
    print("Ollama is ready!")
else:
    print("Ollama not available. Install from https://ollama.ai")
```

### Generating Responses

```python
# Simple generation
response = client.generate(
    prompt="Explain Python decorators in one sentence"
)

if response["success"]:
    print(response["content"])
else:
    print(f"Error: {response.get('error')}")

# With system prompt
response = client.generate(
    prompt="Write a function to reverse a string",
    system_prompt="You are an expert Python developer. Write clean, documented code."
)

print(response["content"])
```

### Using with AgentConfig

```python
# Get agent's prompts
agent = db.get_agent(agent_id)
system_prompt = agent.system_prompt

# Combine with helper prompts
helper_context = "\n".join([
    f"{key}: {value}" 
    for key, value in agent.helper_prompts.items()
])

full_system_prompt = f"{system_prompt}\n\n{helper_context}"

# Generate with agent context
response = client.generate(
    prompt="How should I structure this Python project?",
    system_prompt=full_system_prompt
)
```

---

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Example: Create agent, store content, generate embeddings, search.
"""
import numpy as np
from multimodal_db.core import (
    create_corecoder_agent,
    MultimodalDB,
    QdrantVectorDB,
    SimpleOllamaClient,
    MediaType
)

def main():
    # Initialize components
    db = MultimodalDB()
    vector_db = QdrantVectorDB()
    vector_db.initialize_collections()
    ollama = SimpleOllamaClient()
    
    # Create and store agent
    agent = create_corecoder_agent(name="CodeHelper")
    agent.add_helper_prompt("style", "Write clean, tested code")
    agent_id = db.store_agent(agent)
    print(f"Created agent: {agent_id}")
    
    # Store some content
    content_items = [
        "Python is a high-level programming language",
        "FastAPI is a modern web framework for Python",
        "Polars is a fast dataframe library"
    ]
    
    for text in content_items:
        content_id = db.store_content(
            agent_id=agent_id,
            content=text,
            media_type=MediaType.TEXT,
            metadata={"category": "knowledge"}
        )
        
        # Generate embedding (in real use, use actual embedding model)
        embedding = np.random.rand(768).tolist()
        
        # Store in vector database
        vector_db.store_embedding(
            collection="text_embeddings",
            vector=embedding,
            metadata={
                "agent_id": agent_id,
                "content_id": content_id,
                "text": text
            }
        )
        print(f"Stored: {text[:50]}...")
    
    # Chat with agent (if Ollama available)
    if ollama.available:
        response = ollama.generate(
            prompt="What is Python?",
            system_prompt=agent.system_prompt
        )
        print(f"\nAI Response: {response['content']}")
    
    # Search content
    results = db.search_content(agent_id=agent_id)
    print(f"\nFound {len(results)} content items")
    
    # Get statistics
    stats = db.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Agents: {stats['agents']['total']}")
    print(f"  Content: {stats['content']['total']}")

if __name__ == "__main__":
    main()
```

---

## Testing Your Scripts

```python
# Run with pytest
pytest your_script.py -v

# Or run directly
python your_script.py
```

## Best Practices

1. **Always use context managers or manual save**:
   ```python
   db = MultimodalDB()
   # ... operations ...
   db.save()  # Ensure data is persisted
   ```

2. **Check Ollama availability**:
   ```python
   if ollama.available:
       response = ollama.generate(...)
   ```

3. **Use MediaType enum** (not strings):
   ```python
   # Good
   media_type=MediaType.TEXT
   
   # Bad
   media_type="text"
   ```

4. **Handle None returns**:
   ```python
   agent = db.get_agent(agent_id)
   if agent is None:
       print("Agent not found")
   ```

5. **Use proper embedding dimensions**:
   - Text/Audio/Agent Knowledge: 768-dimensional
   - Image/Video: 512-dimensional

## Error Handling

```python
try:
    agent_id = db.store_agent(agent)
    content_id = db.store_content(...)
except Exception as e:
    print(f"Error: {e}")
    # Handle appropriately
```

## See Also

- **REST API Documentation**: [API.md](API.md) - HTTP endpoints
- **CLI Documentation**: [CLI.md](CLI.md) - Command-line tools
- **Examples**: [EXAMPLES.md](EXAMPLES.md) - Usage examples
