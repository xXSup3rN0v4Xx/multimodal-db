# Multimodal-DB

A comprehensive data management layer for AI agents with multi-database support, vector search, and RAG capabilities.

## 🎯 Project Overview

Multimodal-DB is designed as the **data management foundation** for AI agent systems. It provides a clean separation between data storage/retrieval and model execution, allowing for scalable and maintainable AI applications.

### Architecture Philosophy
- **Data Layer**: Multimodal-DB handles all data operations (storage, search, retrieval)
- **Execution Layer**: External systems (like chatbot-python-core) handle model inference
- **Clean Interfaces**: Well-defined APIs between layers for maximum flexibility

## ✅ **What's Working**

### 🤖 Agent Configuration System
- **Type-Safe Design**: Python enums for ModelType, PromptType, DatabaseCategory
- **Smart Prompt System**: Knows which models support system prompts vs. helper prompts
- **CoreCoder Agent**: Production-ready example with 9 specialized helper prompts
- **Flexible Architecture**: Supports LLM, Vision, Audio, and specialized models
- **Full CRUD Operations**: Create, read, update, delete agent configurations

### 📊 Polars Database Layer
- **Lightning Fast**: Rust-based dataframe operations with `.parquet` storage
- **Agent Storage**: Complete serialization/deserialization of agent configurations
- **Conversation History**: Structure ready for multi-agent conversation tracking
- **Knowledge Base**: Organized storage for structured data
- **Research Collections**: Categorized research data management
- **Backward Compatible**: Handles both new AgentConfig objects and legacy dicts

### 🔍 Qdrant Vector Database
- **Vector Operations**: Store, retrieve, and search high-dimensional vectors
- **Collection Management**: 4 standard collections (knowledge_documents, agent_conversations, research_data, alignment_documents)
- **Semantic Search**: Vector similarity search with configurable thresholds
- **Local & Server Modes**: Flexible deployment from development to production
- **Data Organization**: Clean `data/qdrant_db/` structure

### 🦙 LlamaIndex Integration
- **Hybrid Search**: Dense + sparse vector search capabilities
- **Document Indexing**: Automatic text processing and embedding generation
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` model integrated
- **Query Engines**: Natural language querying infrastructure
- **RAG Foundation**: Core components for retrieval-augmented generation

### 🗂️ Database Organization
- **Clean Separation**: Different database types properly isolated
- **Data Directory**: All storage under unified `data/` structure
- **Path Management**: Automatic directory creation and organization
- **Test Isolation**: Test databases separate from production data

### 🧪 Testing Infrastructure
- **Integration Tests**: Core functionality verified (agent creation, storage, retrieval)
- **Database Path Tests**: File organization and structure validation
- **Qdrant Tests**: Vector operations and search functionality
- **Comprehensive Coverage**: All working components have test coverage

## ⚠️ **What's Not Working / Incomplete**

### 🔄 Conversation Systems
- **Status**: Hardcoded responses removed, but integration incomplete
- **Issue**: `conversation_generator.py` and `conversation_modes.py` have placeholder responses
- **Needs**: Integration with actual model execution layer for real conversations
- **Impact**: Demo functions exist but don't generate real agent conversations

### 🕸️ Graphiti Knowledge Graphs
- **Status**: Implementation exists but requires Neo4j setup
- **Issue**: No Neo4j server configured, knowledge graph features unavailable
- **Needs**: Neo4j installation and configuration
- **Impact**: Advanced relationship mapping and knowledge graphs not functional

### 📝 Polars Query Engine (LlamaIndex)
- **Status**: Documentation exists but integration not tested
- **Issue**: `llama-index-experimental` Polars support may need additional setup
- **Needs**: Verification and testing of natural language → Polars code generation
- **Impact**: Can't query dataframes with natural language yet

### 🌐 API Layer
- **Status**: Empty directories (`api/`, `cli/`)
- **Issue**: No REST API or command-line interface implemented
- **Needs**: FastAPI or similar framework implementation
- **Impact**: No external interface for other applications to use

### 📦 Project Packaging
- **Status**: `pyproject.toml` placeholder
- **Issue**: No proper Python package configuration
- **Needs**: Complete package metadata, build configuration, entry points
- **Impact**: Can't install as a proper Python package

## 🚀 **Getting Started**

### Prerequisites
- Python 3.11+
- Virtual environment recommended

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd multimodal-db
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Test

Run the integration tests to verify everything is working:

```bash
# Test core agent functionality
python tests/test_integration.py

# Test database organization
python tests/test_database_paths.py

# Test vector search capabilities
python tests/test_qdrant_integration.py
```

### Basic Usage

```python
from multimodal_db.core.base_agent_config import create_corecoder_agent
from multimodal_db.core.polars_core import PolarsDBHandler
from multimodal_db.core.qdrant_core import QdrantCore

# Create and store an agent
agent = create_corecoder_agent()
db = PolarsDBHandler("my_agents")
agent_id = db.add_agent_config(agent)

# Set up vector search
qdrant = QdrantCore(persist_path="my_vectors")
qdrant.initialize_standard_collections()

print(f"Agent stored: {agent_id}")
```

## 🏗️ **Architecture**

```
multimodal-db/
├── data/                          # All database files
│   ├── [db_name]/                # Polars parquet files
│   ├── qdrant_db/                # Vector storage
│   │   ├── [collection]/         # Vector collections
│   │   └── meta.json             # Qdrant metadata
│   └── sessions/                 # Conversation sessions
├── multimodal-db/
│   ├── core/                     # Core implementations
│   │   ├── base_agent_config.py  # ✅ Agent management
│   │   ├── polars_core.py        # ✅ Fast dataframes
│   │   ├── qdrant_core.py        # ✅ Vector search
│   │   ├── qdrant_hybrid_search_llama_index.py  # ✅ RAG
│   │   ├── conversation_*.py     # ⚠️ Needs model integration
│   │   └── graphiti_pipe.py      # ⚠️ Needs Neo4j
│   ├── api/                      # 🔄 Empty - needs implementation
│   ├── cli/                      # 🔄 Empty - needs implementation
│   └── utils/                    # 🔄 Minimal
├── tests/                        # ✅ Comprehensive test suite
├── docs/                         # 📚 Documentation and examples
└── requirements.txt              # ✅ All dependencies
```

## 📋 **Roadmap**

### Immediate Next Steps
1. **Model Integration**: Connect with actual LLM execution layer
2. **Conversation System**: Implement real agent-to-agent conversations
3. **API Development**: REST API for external applications
4. **Neo4j Setup**: Enable knowledge graph functionality

### Future Enhancements
- Multi-modal support (images, audio, video)
- Advanced RAG patterns
- Distributed deployment support
- Performance optimization
- Comprehensive benchmarking

## 🤝 **Contributing**

This project is under active development. Current focus areas:

- **Model Integration**: Connect with chatbot-python-core or similar
- **Conversation Generation**: Real agent conversations vs. placeholders
- **API Development**: REST/GraphQL interfaces
- **Documentation**: More examples and tutorials
- **Testing**: Expand test coverage for edge cases

## 📄 **License**

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🔧 **Current Status: Development/Alpha**

- **Core Data Layer**: ✅ Production ready
- **Vector Search**: ✅ Fully functional
- **Agent Management**: ✅ Complete
- **Conversation System**: ⚠️ Needs model integration
- **API Layer**: 🔄 Not implemented
- **Knowledge Graphs**: ⚠️ Needs Neo4j setup

**Perfect for**: Building data layers, prototyping agent systems, vector search applications
**Not ready for**: Production conversations without external model integration

---

*Built with ❤️ for the AI agent community*