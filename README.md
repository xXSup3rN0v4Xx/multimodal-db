# 💾 Multimodal-DB 💾

> High-performance multimodal database system for AI agent ecosystems

A comprehensive data management layer for AI agents with multi-database support, vector search capabilities, and RAG (Retrieval-Augmented Generation) integration. Built with Polars for fast data operations and Qdrant for semantic search.

## Features

- **Multi-Database Architecture** - Polars for structured data, Qdrant for vector search
- **Multimodal Support** - Text, embeddings, images, audio, video, and documents
- **Agent Configuration System** - Type-safe agent management with flexible model support
- **FastAPI Backend** - RESTful API for integration with external systems
- **Gradio UI** - Interactive web interface for agent management
- **Vector Search** - Semantic search with hybrid dense+sparse capabilities

## Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (optional, for AI model execution)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/xXSup3rN0v4Xx/multimodal-db.git
   cd multimodal-db
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Pull AI models**
   ```bash
   ollama pull qwen2.5-coder:3b
   ollama pull nomic-embed-text
   ```

## Usage

### Python Library

```python
from multimodal_db.core import (
    MultimodalDB, 
    QdrantVectorDB,
    create_corecoder_agent
)

# Initialize database
db = MultimodalDB()

# Create and store an agent
agent = create_corecoder_agent("my_coder")
agent_id = db.add_agent(agent)

# Store multimodal content
content_id = db.store_content(
    agent_id=agent_id,
    content="Python optimization techniques",
    media_type=MediaType.TEXT,
    metadata={"category": "coding"}
)
```

### FastAPI Server

Start the API server for external integrations:

```bash
python multimodal-db/api/run_api.py
```

Access API documentation at `http://localhost:8000/docs`

### Gradio UI

Launch the interactive web interface:

```bash
python examples/enhanced_gradio_ui.py
```

## Integration with Chatbot-Python-Core

🔥 **NEW!** Multimodal-DB now integrates seamlessly with [Chatbot-Python-Core](https://github.com/xXSup3rN0v4Xx/chatbot-python-core)!

**Chatbot-Python-Core** provides AI model execution (Ollama, YOLO, Whisper, Kokoro, SDXL), while **Multimodal-DB** provides storage, querying, and analytics. Together, they form a **complete AI application platform**.

### Quick Integration Example

```python
import requests

# 1. Chat with AI (Chatbot-Python-Core)
response = requests.post("http://localhost:8000/api/v1/ollama/chat", json={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
})

# 2. Store conversation (Multimodal-DB)
requests.post("http://localhost:8001/api/v1/conversations/message", json={
    "agent_id": "agent-123",
    "role": "user",
    "content": "Hello!"
})
```

### Integration Documentation

- **[🚀 Quick Reference](docs/QUICK_REFERENCE.md)** - Fast lookup for commands and APIs
- **[📖 How It Works Together](docs/HOW_IT_WORKS_TOGETHER.md)** - Complete integration guide with examples
- **[🎨 Architecture Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)** - Visual representations of integration patterns
- **[🔍 Integration Analysis](docs/INTEGRATION_ANALYSIS.md)** - Technical specifications and compatibility
- **[📊 Integration Summary](docs/INTEGRATION_SUMMARY.md)** - Implementation roadmap and status

**Start here:** [docs/README.md](docs/README.md) - Choose the right document for your needs!

## Documentation

- **[Library Usage](docs/LIBRARY.md)** - Core components and Python API
- **[API Reference](docs/API.md)** - FastAPI endpoints and integration
- **[Examples](docs/EXAMPLES.md)** - Common usage patterns and recipes
- **[Testing](docs/TESTING.md)** - Running tests and validation
- **[CLI Commands](docs/CLI.md)** - Command-line utilities

### Technology-Specific Guides

- **[Polars Integration](docs/polars/polars_docs.md)** - DataFrame operations
- **[Qdrant Vector DB](docs/qdrant/qdrant_docs.md)** - Vector search setup
- **[LlamaIndex](docs/llamaindex/)** - RAG and query engines
- **[Graphiti](docs/graphiti/graphiti_docs.md)** - Knowledge graph integration

## Project Structure

```
multimodal-db/
├── multimodal-db/          # Core package
│   ├── api/                # FastAPI backend
│   ├── core/               # Core components
│   └── utils/              # Utilities
├── data/                   # Database storage
├── docs/                   # Documentation
├── examples/               # Example scripts
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

## Architecture

Multimodal-DB uses a layered architecture:

1. **Storage Layer** - Polars (structured data) + Qdrant (vectors)
2. **Core Layer** - Agent management, content storage, search
3. **API Layer** - FastAPI endpoints for external access
4. **UI Layer** - Gradio interface for human interaction

This separation enables flexible deployment and scaling.

## Contributing

Contributions are welcome! Please ensure tests pass before submitting PRs:

```bash
pytest tests/test_optimized.py -v
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details

## Links

- **Repository**: https://github.com/xXSup3rN0v4Xx/multimodal-db
- **Issues**: https://github.com/xXSup3rN0v4Xx/multimodal-db/issues
- **Documentation**: [docs/](docs/)

---

Built for the AI agent ecosystem by [xXSup3rN0v4Xx](https://github.com/xXSup3rN0v4Xx)