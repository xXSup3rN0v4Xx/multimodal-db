# Multimodal-DB Architecture

> High-performance multimodal database system for AI agent ecosystems

## 🏗️ System Architecture

### Core Components

#### 1. **Agent Configuration System** (`core/agent_configs/`)
- **`base_agent_config.py`** - Complete agent configuration management
  - Model types: LLM, Vision LLM, Embedding, Vision Detection, Speech-to-Text, Text-to-Speech, Audio/Image/Video Generation
  - Prompt management: System, Helper, Booster, Prime Directive, User Input/Files/Images
  - Database categories: Agent Configs, Conversations, Knowledge Base, Research Data, Templates, User Data, Alignment Docs
  - RAG configuration: Qdrant Hybrid Search, Graphiti Temporal RAG, Polars/Pandas Query Engines
  - Supports chatbot-python-core models: Ollama, Whisper, Kokoro, YOLO, Stable Diffusion

#### 2. **Database Implementations** (`core/dbs/`)

##### **PolarsDB** - High-speed tabular storage
- Lightning-fast Parquet-based storage
- Agent configurations and conversations
- Zero-copy data operations
- Perfect for structured data

##### **QdrantDB** - Vector search (Simple)
- Basic vector storage and retrieval
- Cosine similarity search
- Metadata filtering

##### **QdrantVectorDB** - Vector search (Enhanced)
- Multimodal embeddings (text, image, audio, video)
- Agent-specific knowledge bases
- Temporal awareness
- Hybrid search ready

##### **MultimodalDB** - Unified multimodal storage
- Combines Polars + File Storage
- Media management (audio, image, video, documents)
- Agent configurations with full context
- Conversation histories
- Export/Import capabilities

##### **GraphitiDB** - Temporal knowledge graphs
- Entity and relationship extraction
- Time-aware knowledge retrieval
- Semantic search with temporal context
- Document chunking and processing
- Perfect for RAG with temporal reasoning

#### 3. **Database Tools** (`core/db_tools/`)

##### **ParquetExporter** - Universal export system
- Export from any database to Parquet
- Per-database exports (Polars, Qdrant, Multimodal, Graphiti)
- Complete agent exports (config + conversations + media + embeddings + graph)
- Metadata tracking
- Pandas/Polars conversion

##### **PandasNLQueryEngine** - Natural language queries on Pandas
- LlamaIndex integration
- Ollama-powered code generation
- Natural language to Pandas code
- Conversation/agent analysis
- Custom prompt support

##### **PolarsNLQueryEngine** - Natural language queries on Polars
- High-speed analytics
- Streaming data support (for YOLO detections)
- Batch query processing
- Vision detection analysis
- Media analysis
- Conversation/agent analysis

##### **QdrantHybridSearch** - Advanced RAG
- Dense + Sparse vector search
- BM25-style retrieval
- Neural reranking with LLM
- Agent-specific knowledge bases
- Context-aware responses
- Query with automatic context retrieval

### 4. **API Layer** (`api/`)
- FastAPI-based REST API
- CORS enabled for Next.js frontend
- WebSocket support for real-time chat
- Agent management endpoints
- Content storage and retrieval
- Real-time search
- Admin endpoints for health and stats

## 🔄 Integration with chatbot-python-core

### Separation of Concerns
- **multimodal-db**: Database layer (storage, retrieval, RAG)
- **chatbot-python-core**: Multimodal toolkit (speech, vision, generation)

### Integration Points

#### 1. **Model Storage**
```python
# Agent config stores chatbot-python-core model configurations
agent = AgentConfig()
agent.enable_model("large_language_model", "ollama", {
    "model": "qwen2.5-coder:3b",
    "temperature": 0.1
})
agent.enable_model("vision_detection", "yolo", {
    "model": "yolov8n.pt"
})
agent.enable_model("text_to_speech", "kokoro", {
    "voice": "af_sarah"
})
```

#### 2. **Streaming Data (YOLO)**
```python
# High-speed vision data through Polars
polars_db = PolarsDB()
for detection in yolo_stream:
    polars_db.add_detection(detection)  # Batch insert

# Query with natural language
query_engine = PolarsNLQueryEngine()
results = query_engine.analyze_vision_detections("detections.parquet")
```

#### 3. **RAG Integration**
```python
# Store conversations in Graphiti for temporal RAG
graphiti_db = GraphitiDBSync()
graphiti_db.add_episode(
    agent_id=agent_id,
    content=user_message,
    episode_type="conversation"
)

# Hybrid search for knowledge retrieval
hybrid_search = QdrantHybridSearch()
response = hybrid_search.query_with_context(
    query=user_question,
    agent_id=agent_id
)
```

#### 4. **Unified API Mode**
```python
# Both APIs can run together
# multimodal-db API: http://localhost:8000
# chatbot-python-core API: http://localhost:8001

# Frontend calls both for complete functionality
# - Speech/Vision/Generation → chatbot-python-core
# - Storage/RAG/Query → multimodal-db
```

## 📊 Data Flow Scenarios

### Scenario 1: Agent with Conversation History
```
User → API → Agent Config → Ollama (chatbot-python-core)
                          → MultimodalDB (store message)
                          → GraphitiDB (extract entities)
                          → QdrantVectorDB (store embedding)
```

### Scenario 2: Vision Detection Stream
```
Camera → YOLO (chatbot-python-core) → PolarsDB (batch insert)
                                    → PolarsNLQueryEngine (analyze)
                                    → Parquet Export (archive)
```

### Scenario 3: RAG Query
```
User Query → QdrantHybridSearch → Dense Search (embeddings)
                                → Sparse Search (BM25)
                                → Neural Rerank
                                → GraphitiDB (temporal context)
                                → Ollama (generate response)
```

### Scenario 4: Knowledge Base Building
```
Documents → GraphitiDB.add_document() → Entity Extraction
                                      → Relationship Mapping
                                      → Temporal Indexing
         → QdrantHybridSearch.add_documents() → Dense Embeddings
                                                → Sparse Vectors
                                                → Hybrid Index
```

## 🚀 Usage Examples

### Basic Agent Creation
```python
from core import AgentConfig, MultimodalDB

# Create agent
agent = AgentConfig(agent_name="CodeHelper")
agent.enable_model("large_language_model", "ollama", {
    "model": "qwen2.5-coder:3b"
})
agent.set_system_prompt("large_language_model", "ollama", 
    "You are a helpful coding assistant.")

# Store in database
db = MultimodalDB()
agent_id = db.add_agent(agent)
```

### Natural Language Queries
```python
from core import PolarsNLQueryEngine

# Load data and query
engine = PolarsNLQueryEngine()
result = engine.query_from_parquet(
    "conversations.parquet",
    "What are the top 5 topics discussed?"
)
print(result["response"])
print(result["polars_code"])  # See generated code
```

### Hybrid Search RAG
```python
from core import QdrantHybridSearch

# Initialize hybrid search
rag = QdrantHybridSearch()

# Add knowledge documents
rag.add_agent_knowledge(
    agent_id="agent-123",
    documents=["Document 1", "Document 2"],
    document_types=["manual", "faq"]
)

# Query with context
response = rag.query_with_context(
    query="How do I configure the API?",
    agent_id="agent-123"
)
print(response["response"])
print(f"Used {response['num_contexts']} context docs")
```

### Export Everything
```python
from core import ParquetExporter

exporter = ParquetExporter()

# Export complete agent data
exports = exporter.export_agent_complete(
    agent_id="agent-123",
    multimodal_db=db,
    qdrant_db=vector_db,
    graphiti_db=graphiti_db
)

print(f"Exported to: {exports['export_dir']}")
print(f"Files: {list(exports['files'].keys())}")
```

## 🎯 Key Features

### Performance
- ⚡ Polars: 10-100x faster than Pandas
- 🚀 Zero-copy operations where possible
- 📦 Efficient Parquet storage
- 🔥 Streaming support for high-throughput data

### Flexibility
- 🔧 Works standalone or with chatbot-python-core
- 🎨 Multiple databases for different use cases
- 🌐 RESTful API + WebSocket
- 📊 Natural language queries

### Intelligence
- 🧠 Temporal RAG with Graphiti
- 🔍 Hybrid search (dense + sparse)
- 🎯 Neural reranking
- 📚 Knowledge graph construction

### Reliability
- 💾 Parquet-based persistence
- 🔒 Metadata tracking
- 📤 Export/Import capabilities
- ✅ Type safety with Enums

## 🔮 Future Enhancements

1. **Agent Collaboration**
   - Multi-agent conversations
   - Shared knowledge bases
   - Agent-to-agent RAG

2. **Advanced Streaming**
   - Real-time YOLO → Polars pipeline
   - Audio streaming analysis
   - Video frame processing

3. **Enhanced RAG**
   - Multi-hop reasoning
   - Cross-modal retrieval
   - Adaptive reranking

4. **Production Features**
   - Authentication & authorization
   - Rate limiting
   - Monitoring & logging
   - Distributed deployment

## 📝 Configuration Best Practices

### For Conversation Agents
```python
agent = create_corecoder_agent()  # Pre-configured
# Enable RAG
agent.enable_rag_system("graphiti_temporal_rag", {
    "temporal_awareness": True
})
agent.enable_rag_system("qdrant_hybrid_search", {
    "dense_model": "BAAI/bge-small-en-v1.5"
})
```

### For Vision Agents
```python
agent = AgentConfig(agent_name="VisionAgent")
agent.enable_model("vision_detection", "yolo")
agent.enable_database("research_data", backend="polars")
# Use PolarsNLQueryEngine for high-speed analysis
```

### For Knowledge Management
```python
agent = AgentConfig(agent_name="KnowledgeAgent")
agent.enable_database("knowledge_base", backend="qdrant")
agent.enable_database("research_data", backend="graphiti")
agent.enable_rag_system("qdrant_hybrid_search")
agent.enable_rag_system("graphiti_temporal_rag")
```

## 🤝 Contributing

When extending multimodal-db:
1. Keep databases focused (single responsibility)
2. Maintain separation from chatbot-python-core
3. Add natural language query support where applicable
4. Include export functionality
5. Update this architecture document

---

**Built for the AI agent ecosystem** 🤖💾🚀
