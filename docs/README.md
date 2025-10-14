# Documentation

Complete documentation for Multimodal-DB.

## Quick Links

| Documentation | Description |
|---------------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | One-page command reference |
| [LIBRARY.md](LIBRARY.md) | Python library reference |
| [API.md](API.md) | REST API endpoints |
| [CLI.md](CLI.md) | Command-line tools |
| [EXAMPLES.md](EXAMPLES.md) | UI applications |
| [TESTING.md](TESTING.md) | Test suite guide |

## Usage Guides

### For Python Developers

**[LIBRARY.md](LIBRARY.md)** - Complete Python library reference

Learn how to use Multimodal-DB as a library in your Python scripts:
- AgentConfig - Create and configure AI agents
- MultimodalDB - Store agents, content, and messages
- QdrantVectorDB - Vector embeddings and similarity search
- SimpleOllamaClient - AI chat integration
- Complete examples and best practices

**Start here if you want to**: Write Python scripts using the core library

---

### For REST API Users

**[API.md](API.md)** - Complete REST API documentation

Use the FastAPI backend from any language or platform:
- Agent endpoints (CRUD operations)
- Content management
- AI chat endpoints
- Search and statistics
- Request/response examples in cURL and Python

**Start here if you want to**: Build web apps, mobile apps, or use from other languages

---

### For Command-Line Users

**[CLI.md](CLI.md)** - Command-line tools documentation

Use the provided CLI utilities:
- cleanup_agents.py - Database cleanup tool
- initialize_corecoder.py - Quick agent setup
- run_api.py - Start the API server
- Creating custom CLI tools

**Start here if you want to**: Automate tasks, manage database, quick operations

---

### For UI Users

**[EXAMPLES.md](EXAMPLES.md)** - Example applications and interfaces

Two Gradio web interfaces included:
- Enhanced UI - Full-featured professional interface
- Simple UI - Minimal testing interface
- Creating custom UIs
- Deployment guides

**Start here if you want to**: Use the web interface or build custom UIs

---

### For Testers

**[TESTING.md](TESTING.md)** - Testing documentation

Comprehensive testing guide:
- Running the test suite (pytest)
- Interactive testing (Jupyter notebook)
- Writing custom tests
- CI/CD integration

**Start here if you want to**: Run tests, write new tests, ensure quality

---

## Core Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - One-page quick reference with commands and examples
- **[STATUS.md](STATUS.md)** - Current project status and roadmap
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines

## Technical Documentation

### Integration Guides
- **[llamaindex/](llamaindex/)** - LlamaIndex integration examples and guides
- **[qdrant/](qdrant/)** - Qdrant vector database documentation
- **[polars/](polars/)** - Polars dataframe library documentation
- **[graphiti/](graphiti/)** - Graphiti knowledge graph integration

### Key Files
- `qdrant/qdrant_docs.md` - Qdrant setup and usage
- `polars/polars_docs.md` - Polars operations and best practices
- `llamaindex/qdrant_llama_index_hybrid_search.md` - RAG implementation guide
- `llamaindex/*.ipynb` - Interactive notebooks for query engines

## Getting Started Workflows

### I want to write Python scripts

1. Read [LIBRARY.md](LIBRARY.md)
2. Check examples in the "Complete Example Script" section
3. Try the test script: `tests/test_optimized.py`
4. Reference [QUICKSTART.md](QUICKSTART.md) for quick lookups

### I want to use the REST API

1. Start the API: `python multimodal-db/api/run_api.py`
2. Read [API.md](API.md) for endpoint documentation
3. Try interactive docs: http://localhost:8000/docs
4. Use examples from the "Complete Workflow" section

### I want to use the command-line tools

1. Read [CLI.md](CLI.md)
2. Try cleanup tool: `python scripts/cleanup_agents.py`
3. Create custom scripts using templates provided
4. Automate with cron/Task Scheduler

### I want to use the web interface

1. Start API: `python multimodal-db/api/run_api.py`
2. Start UI: `python examples/enhanced_gradio_ui.py`
3. Read [EXAMPLES.md](EXAMPLES.md) for feature guide
4. Access at http://localhost:7860

### I want to run tests

1. Install test dependencies: `pip install pytest pytest-cov`
2. Read [TESTING.md](TESTING.md)
3. Run tests: `pytest tests/ -v`
4. Check coverage: `pytest tests/ --cov=multimodal_db`

## Documentation Structure

```
docs/
├── README.md              # This file - Documentation index
├── QUICKSTART.md          # Quick reference (start here)
├── LIBRARY.md             # Python library usage ⭐
├── API.md                 # REST API endpoints ⭐
├── CLI.md                 # Command-line tools ⭐
├── EXAMPLES.md            # UI applications ⭐
├── TESTING.md             # Test suite guide ⭐
├── STATUS.md              # Project status
├── CHANGELOG.md           # Version history
├── CONTRIBUTING.md        # Contribution guide
├── qdrant/                # Vector DB docs
├── polars/                # DataFrame docs
├── llamaindex/            # RAG integration docs
└── graphiti/              # Knowledge graph docs
```

## External Resources

- **Main README**: [../README.md](../README.md) - Project overview
- **Repository**: https://github.com/xXSup3rN0v4Xx/multimodal-db
- **Issues**: https://github.com/xXSup3rN0v4Xx/multimodal-db/issues
- **API Docs (Interactive)**: http://localhost:8000/docs (when running)

## Need Help?

1. Check [QUICKSTART.md](QUICKSTART.md) for common commands
2. Search the appropriate guide:
   - Python code → [LIBRARY.md](LIBRARY.md)
   - HTTP requests → [API.md](API.md)
   - CLI usage → [CLI.md](CLI.md)
   - UI usage → [EXAMPLES.md](EXAMPLES.md)
   - Testing → [TESTING.md](TESTING.md)
3. Check [STATUS.md](STATUS.md) for known limitations
4. Open an issue on GitHub

---

**Note**: All documentation reflects v1.0.0. For latest features, check [STATUS.md](STATUS.md) and [CHANGELOG.md](CHANGELOG.md).
