# Quick Reference: Multimodal-DB ↔ Chatbot-Python-Core Integration

## System Comparison

| Feature | Chatbot-Python-Core | Multimodal-DB |
|---------|---------------------|---------------|
| **Purpose** | AI Model Execution | Data Storage & Querying |
| **Port** | 8000 | 8001 |
| **Core Function** | Process (Brain) | Remember (Memory) |
| **Models** | Ollama, YOLO, Whisper, Kokoro, SDXL | N/A |
| **Databases** | None | Polars, Qdrant, Graphiti |
| **Stores Data?** | ❌ No | ✅ Yes |
| **Runs AI?** | ✅ Yes | ❌ No |

## Quick Start Commands

```bash
# Terminal 1: Start Chatbot-Python-Core
cd chatbot-python-core
python run_api.py --port 8000

# Terminal 2: Start Multimodal-DB  
cd multimodal-db
python run_api.py --port 8001

# Terminal 3: Test integration
curl http://localhost:8000/health  # Should return healthy
curl http://localhost:8001/health  # Should return healthy
```

## Integration Patterns

### 1️⃣ API-to-API (Production)
```python
# Multimodal-DB orchestrates Chatbot-Python-Core
response = requests.post("http://localhost:8000/api/v1/ollama/chat", json={...})
requests.post("http://localhost:8001/api/v1/conversations/store", json={...})
```

### 2️⃣ Direct Import (Development)
```python
# Both imported directly
from chatbot_python_core.core.ollama import OllamaServiceOrchestrator
from multimodal_db.core import PolarsDB
```

### 3️⃣ Unified API (Best of Both)
```python
# Single endpoint does everything
response = requests.post("http://localhost:8003/api/v1/chat", json={
    "agent_id": "agent-123",
    "message": "Hello",
    "store_conversation": True
})
```

## Common Workflows

### Chat with Memory
```python
# 1. Get agent config (Multimodal-DB)
agent = GET /api/v1/agents/{id}

# 2. Get history (Multimodal-DB)
history = GET /api/v1/conversations/{agent_id}

# 3. Chat (Chatbot-Python-Core)
response = POST /api/v1/ollama/chat + history

# 4. Store (Multimodal-DB)
POST /api/v1/conversations/message
```

### YOLO Detection
```python
# 1. Detect (Chatbot-Python-Core)
detections = POST /api/v1/vision/detect

# 2. Store (Multimodal-DB)
POST /api/v1/detections/store

# 3. Query (Multimodal-DB)
result = POST /api/v1/query/polars
# "How many cars detected today?"
```

### RAG Pipeline
```python
# 1. Search KB (Multimodal-DB)
docs = POST /api/v1/search/hybrid

# 2. Chat with context (Chatbot-Python-Core)
answer = POST /api/v1/ollama/chat + docs

# 3. Store Q&A (Multimodal-DB)
POST /api/v1/conversations/message
```

## Key APIs

### Chatbot-Python-Core (Port 8000)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/ollama/chat` | Chat with LLM |
| `POST /api/v1/audio/tts` | Text-to-Speech |
| `POST /api/v1/audio/stt` | Speech-to-Text |
| `POST /api/v1/vision/detect` | Object Detection |
| `POST /api/v1/image/generate` | Image Generation |

### Multimodal-DB (Port 8001)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/agents` | Create Agent |
| `GET /api/v1/agents/{id}` | Get Agent |
| `POST /api/v1/conversations/message` | Store Message |
| `GET /api/v1/conversations/{agent_id}` | Get History |
| `POST /api/v1/query/polars` | Natural Language Query |
| `POST /api/v1/search/hybrid` | Semantic Search |

## CLI Commands

```bash
# Agent management
python run_cli.py agent create --name "MyAgent"
python run_cli.py agent list
python run_cli.py agent show <agent_id>

# Conversations
python run_cli.py conversation show <agent_id>
python run_cli.py conversation add <agent_id> "Hello"

# Queries
python run_cli.py query run "How many agents?"

# Export
python run_cli.py database export --format parquet
```

## Code Templates

### Simple Chat
```python
import requests

# Chat
response = requests.post("http://localhost:8000/api/v1/ollama/chat", json={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
})
print(response.json()['message']['content'])

# Store
requests.post("http://localhost:8001/api/v1/conversations/message", json={
    "agent_id": "agent-123",
    "role": "assistant",
    "content": response.json()['message']['content']
})
```

### Detection Pipeline
```python
import cv2, requests, base64

frame = cv2.imread("image.jpg")
_, buffer = cv2.imencode('.jpg', frame)
img_base64 = base64.b64encode(buffer).decode()

# Detect
detections = requests.post("http://localhost:8000/api/v1/vision/detect", json={
    "image": img_base64,
    "model": "yolov8n"
}).json()

# Store
for det in detections['detections']:
    requests.post("http://localhost:8001/api/v1/detections/store", json=det)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Check services running: `curl http://localhost:8000/health` |
| Agent not found | List agents: `python run_cli.py agent list` |
| Model unavailable | Pull model: `ollama pull llama3.2` |
| Slow queries | Use smaller model: `qwen2.5-coder:3b` |
| Out of memory | Use batch processing or streaming |

## Best Practices

✅ **DO:**
- Store all interactions
- Use consistent agent configs
- Implement error handling
- Export data regularly
- Use natural language queries

❌ **DON'T:**
- Hardcode model names
- Skip storing context
- Ignore API errors
- Query in loops
- Load unlimited history

## Architecture Diagram

```
┌─────────────┐
│    USER     │
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│    Integration Layer            │
│  (Your Application/API)         │
└──────┬──────────────────────┬───┘
       │                      │
┌──────▼──────┐      ┌────────▼────────┐
│  Chatbot    │      │  Multimodal-DB  │
│  Python     │◄────►│                 │
│  Core       │      │                 │
│  (Port 8000)│      │  (Port 8001)    │
└──────┬──────┘      └────────┬────────┘
       │                      │
┌──────▼──────┐      ┌────────▼────────┐
│ AI Models   │      │  Databases      │
│ - Ollama    │      │  - Polars       │
│ - YOLO      │      │  - Qdrant       │
│ - Whisper   │      │  - Graphiti     │
└─────────────┘      └─────────────────┘
```

## Data Flow

1. **User Request** → Your App
2. **Retrieve Config** → Multimodal-DB
3. **Execute AI Task** → Chatbot-Python-Core
4. **Store Results** → Multimodal-DB
5. **Response** → User

## Example Use Cases

| Use Case | Chatbot-Core Role | Multimodal-DB Role |
|----------|-------------------|-------------------|
| **Chat Bot** | Generate responses | Store conversations |
| **Security Cam** | Detect objects | Store + query detections |
| **Content Gen** | Create images/audio | Store artifacts |
| **RAG System** | Generate answers | Search knowledge base |
| **Research** | Process papers | Build knowledge graph |

## Next Steps

1. ✅ Read full guide: `docs/HOW_IT_WORKS_TOGETHER.md`
2. ✅ Review architecture: `docs/INTEGRATION_ANALYSIS.md`
3. ✅ Check examples: `examples/integration_examples.py`
4. ✅ Try CLI: `python run_cli.py --help`
5. ✅ Test APIs: Use Postman or curl

## Resources

- **Full Documentation:** `docs/HOW_IT_WORKS_TOGETHER.md`
- **Technical Analysis:** `docs/INTEGRATION_ANALYSIS.md`
- **Implementation Status:** `docs/INTEGRATION_SUMMARY.md`
- **CLI Reference:** `python run_cli.py --help`
- **API Docs:** `/docs` endpoint on each service

---

**Quick Test:**
```bash
# Create agent
python run_cli.py agent create --name "TestAgent"

# Send a chat message (requires services running)
curl -X POST http://localhost:8000/api/v1/ollama/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello!"}]}'

# Check it worked
python run_cli.py agent list
```

**You're ready to build! 🚀**
