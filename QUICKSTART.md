# 🚀 Quick Start Guide - Multimodal-DB

## Installation

```bash
cd multimodal-db
pip install -r requirements.txt
```

### Optional Dependencies

For full functionality, install these additional packages:

```bash
# LlamaIndex for natural language queries
pip install llama-index llama-index-experimental
pip install llama-index-llms-ollama llama-index-embeddings-huggingface
pip install llama-index-vector-stores-qdrant

# Graphiti for temporal knowledge graphs
pip install graphiti-core

# Make sure Ollama is running
ollama pull qwen2.5-coder:3b
```

## Basic Usage

### 1. Create an Agent

```python
from core import AgentConfig, MultimodalDB

# Create agent
agent = AgentConfig(agent_name="MyBot")
agent.enable_model("large_language_model", "ollama", {
    "model": "qwen2.5-coder:3b"
})
agent.set_system_prompt("large_language_model", "ollama",
    "You are a helpful assistant.")

# Store in database
db = MultimodalDB()
agent_id = db.add_agent(agent)
```

### 2. Store Conversations

```python
# Add messages
db.add_message(agent_id, "user", "Hello!")
db.add_message(agent_id, "assistant", "Hi there!")

# Retrieve history
history = db.get_conversation(agent_id, limit=10)
```

### 3. Export Data

```python
from core import ParquetExporter

exporter = ParquetExporter()
exports = exporter.export_multimodal_db(db)
print(f"Exported to: {exports}")
```

### 4. Query with Natural Language

```python
from core import PolarsNLQueryEngine

engine = PolarsNLQueryEngine()
result = engine.query_from_parquet(
    "agents.parquet",
    "How many agents are there?"
)
print(result["response"])
```

### 5. Hybrid Search RAG

```python
from core import QdrantHybridSearch

rag = QdrantHybridSearch()
rag.add_documents(["Doc 1", "Doc 2"], agent_id=agent_id)

response = rag.query_with_context(
    query="What is in the docs?",
    agent_id=agent_id
)
print(response["response"])
```

### 6. Temporal Knowledge Graphs

```python
from core import GraphitiDBSync

graphiti = GraphitiDBSync()
graphiti.add_episode(
    agent_id=agent_id,
    content="User asked about Python",
    episode_type="conversation"
)

results = graphiti.search_episodes(
    query="Python",
    agent_id=agent_id
)
```

## Running the API

```bash
# Start the API server
cd multimodal-db
python -m uvicorn multimodal-db.api.main:app --reload --port 8000
```

### API Endpoints

- `GET /` - API health check
- `GET /agents/` - List all agents
- `POST /agents/` - Create agent
- `GET /agents/{agent_id}` - Get specific agent
- `POST /content/` - Store content
- `POST /search/content` - Search content
- `POST /chat/message` - Chat with agent
- `GET /admin/stats` - System statistics

## Integration with chatbot-python-core

### Unified API Setup

```python
# multimodal-db API on port 8000 (storage & RAG)
# chatbot-python-core API on port 8001 (models & processing)

# Frontend calls both:
# - Vision, speech, generation → chatbot-python-core
# - Storage, RAG, queries → multimodal-db
```

### Example: Vision + Storage

```python
# In chatbot-python-core:
from chatbot_python_core.core.vision import YoloOBB
yolo = YoloOBB()
detections = yolo.detect(image)

# In multimodal-db:
from core import PolarsDB
db = PolarsDB()
for det in detections:
    db.add_detection(det)

# Query later
from core import PolarsNLQueryEngine
engine = PolarsNLQueryEngine()
result = engine.analyze_vision_detections("detections.parquet")
```

## Examples

Run the integration examples:

```bash
cd multimodal-db
python examples/core/integration_examples.py
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design and integration patterns.

## Testing

```bash
# Run tests (when available)
pytest tests/

# Test individual components
python -m multimodal-db.core.dbs.polars_db
python -m multimodal-db.core.db_tools.export_data_as_parquet
```

## Project Structure

```
multimodal-db/
├── core/
│   ├── agent_configs/      # Agent configuration system
│   ├── dbs/                # Database implementations
│   └── db_tools/           # Query engines & exporters
├── api/                    # FastAPI REST API
├── examples/               # Integration examples
├── tests/                  # Test suites
└── data/                   # Database storage (auto-created)
```

## Key Features

✅ **High Performance**: Polars for 10-100x faster operations
✅ **Multimodal**: Audio, images, video, text, embeddings
✅ **RAG Ready**: Hybrid search, temporal graphs, NL queries
✅ **Flexible**: Works standalone or with chatbot-python-core
✅ **Production Ready**: REST API, exports, monitoring

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Check [examples/core/integration_examples.py](examples/core/integration_examples.py)
3. Review API docs at `http://localhost:8000/docs`
4. Explore chatbot-python-core integration patterns

## Support

For issues or questions, check:
- Architecture documentation
- Integration examples
- API interactive docs
- Test files for usage patterns

---

**Built for AI Agent Ecosystems** 🤖💾🚀
