# REST API Documentation

Complete reference for the FastAPI REST API endpoints.

## Base URL

```
http://localhost:8000
```

## API Documentation

- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## Authentication

Currently, no authentication is required. Future versions will add API key support.

---

## Response Format

All endpoints return JSON responses.

### Success Response
```json
{
  "status": "success",
  "data": { ... }
}
```

### Error Response
```json
{
  "detail": "Error message description"
}
```

---

## Endpoints

### Root / Health Check

#### `GET /`

Get API information and health status.

**Response:**
```json
{
  "status": "operational",
  "system": "Multimodal-DB Unified API",
  "version": "1.0.0",
  "razor_sharp": true,
  "components": {
    "database": "MultimodalDB",
    "vector_db": "QdrantVectorDB",
    "collections": 6
  }
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

## Agent Endpoints

### List All Agents

#### `GET /agents/`

List all agents with summary information.

**Query Parameters:**
- `include_full` (boolean, optional) - If `true`, includes full configuration with helper_prompts, system_prompt, and models

**Response (summary):**
```json
[
  {
    "agent_id": "uuid-here",
    "name": "CoreCoder",
    "description": "Expert coding assistant",
    "tags": ["coding", "python"],
    "created_at": "2025-10-13T10:00:00",
    "updated_at": "2025-10-13T10:00:00"
  }
]
```

**Response (full config with `include_full=true`):**
```json
[
  {
    "agent_id": "uuid-here",
    "name": "CoreCoder",
    "description": "Expert coding assistant",
    "tags": ["coding", "python"],
    "created_at": "2025-10-13T10:00:00",
    "updated_at": "2025-10-13T10:00:00",
    "helper_prompts": {
      "style": "Write clean code",
      "format": "Use type hints"
    },
    "system_prompt": "You are an expert developer",
    "models": [
      {
        "name": "qwen2.5-coder:3b",
        "model_type": "qwen2.5-coder:3b",
        "enabled": true
      }
    ]
  }
]
```

**Examples:**
```bash
# Summary view
curl http://localhost:8000/agents/

# Full configuration
curl http://localhost:8000/agents/?include_full=true

# Using Python
import requests
response = requests.get("http://localhost:8000/agents/?include_full=true")
agents = response.json()
```

### Create Agent

#### `POST /agents/`

Create a new agent.

**Request Body:**
```json
{
  "name": "my_agent",
  "agent_type": "corecoder",
  "description": "My custom agent",
  "tags": ["tag1", "tag2"]
}
```

**Parameters:**
- `name` (string, required) - Agent name
- `agent_type` (string, optional) - Type: `"corecoder"` or `"multimodal"` (default: `"corecoder"`)
- `description` (string, optional) - Agent description
- `tags` (array of strings, optional) - Tags for categorization

**Response:**
```json
{
  "agent_id": "uuid-here",
  "name": "my_agent",
  "agent_type": "corecoder",
  "message": "Agent created successfully"
}
```

**Examples:**
```bash
# cURL
curl -X POST http://localhost:8000/agents/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_agent",
    "agent_type": "corecoder",
    "description": "Test agent",
    "tags": ["test", "demo"]
  }'

# Python
import requests
response = requests.post(
    "http://localhost:8000/agents/",
    json={
        "name": "my_agent",
        "agent_type": "corecoder",
        "description": "Test agent",
        "tags": ["test", "demo"]
    }
)
print(response.json())
```

### Get Specific Agent

#### `GET /agents/{agent_id}`

Get detailed information about a specific agent.

**Path Parameters:**
- `agent_id` (string, required) - Agent UUID

**Response:**
```json
{
  "agent_id": "uuid-here",
  "name": "CoreCoder",
  "description": "Expert coding assistant",
  "config_json": "{...}",
  "tags": ["coding"],
  "created_at": "2025-10-13T10:00:00",
  "updated_at": "2025-10-13T10:00:00"
}
```

**Examples:**
```bash
curl http://localhost:8000/agents/550e8400-e29b-41d4-a716-446655440000

# Python
import requests
agent_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.get(f"http://localhost:8000/agents/{agent_id}")
agent = response.json()
```

### Delete Agent

#### `DELETE /agents/{agent_id}`

Delete an agent and its associated data.

**Path Parameters:**
- `agent_id` (string, required) - Agent UUID

**Response:**
```json
{
  "success": true,
  "agent_id": "uuid-here",
  "message": "Agent deleted successfully"
}
```

**Examples:**
```bash
curl -X DELETE http://localhost:8000/agents/550e8400-e29b-41d4-a716-446655440000

# Python
import requests
agent_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.delete(f"http://localhost:8000/agents/{agent_id}")
print(response.json())
```

---

## Content Endpoints

### Upload Content

#### `POST /content/`

Upload content associated with an agent.

**Request Body:**
```json
{
  "agent_id": "uuid-here",
  "content": "Text content or file path",
  "media_type": "text",
  "metadata": {
    "category": "notes",
    "priority": "high"
  }
}
```

**Parameters:**
- `agent_id` (string, required) - Associated agent UUID
- `content` (string, required) - Content text or file path
- `media_type` (string, required) - One of: `"text"`, `"image"`, `"audio"`, `"video"`, `"document"`, `"embedding"`
- `metadata` (object, optional) - Custom metadata as JSON object

**Response:**
```json
{
  "content_id": "uuid-here",
  "agent_id": "uuid-here",
  "media_type": "text",
  "message": "Content stored successfully"
}
```

**Examples:**
```bash
# cURL
curl -X POST http://localhost:8000/content/ \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "Important notes here",
    "media_type": "text",
    "metadata": {"category": "meeting"}
  }'

# Python
import requests
response = requests.post(
    "http://localhost:8000/content/",
    json={
        "agent_id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "Important notes",
        "media_type": "text",
        "metadata": {"category": "meeting"}
    }
)
print(response.json())
```

### List Content

#### `GET /content/`

List content with optional filters.

**Query Parameters:**
- `agent_id` (string, optional) - Filter by agent
- `media_type` (string, optional) - Filter by media type
- `limit` (integer, optional) - Max results (default: 50)

**Response:**
```json
[
  {
    "id": "uuid-here",
    "agent_id": "agent-uuid",
    "content": "Content text",
    "media_type": "text",
    "timestamp": "2025-10-13T10:00:00",
    "metadata": {"category": "notes"}
  }
]
```

**Examples:**
```bash
# All content
curl http://localhost:8000/content/

# Filter by agent
curl "http://localhost:8000/content/?agent_id=550e8400-e29b-41d4-a716-446655440000"

# Filter by media type
curl "http://localhost:8000/content/?media_type=text&limit=20"

# Python
import requests
response = requests.get(
    "http://localhost:8000/content/",
    params={"agent_id": "550e8400-e29b-41d4-a716-446655440000", "media_type": "text"}
)
content = response.json()
```

### Search Content

#### `POST /search/content`

Search content with query and filters.

**Request Body:**
```json
{
  "query": "search term",
  "agent_id": "uuid-here",
  "media_type": "text",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "id": "uuid-here",
    "content": "Matching content...",
    "media_type": "text",
    "score": 0.95
  }
]
```

**Example:**
```bash
curl -X POST http://localhost:8000/search/content \
  -H "Content-Type: application/json" \
  -d '{
    "query": "python programming",
    "media_type": "text",
    "limit": 10
  }'
```

---

## AI Chat Endpoints

### Check Ollama Status

#### `GET /chat/status`

Check if Ollama is available and ready for chat.

**Response:**
```json
{
  "ollama_available": true,
  "model": "qwen2.5-coder:3b",
  "timeout": 60,
  "status": "ready",
  "message": "Ollama is ready for chat"
}
```

**Example:**
```bash
curl http://localhost:8000/chat/status

# Python
import requests
response = requests.get("http://localhost:8000/chat/status")
status = response.json()
if status["ollama_available"]:
    print("Ready to chat!")
```

### Send Chat Message

#### `POST /chat/message`

Send a message and get AI response using agent's context.

**Request Body:**
```json
{
  "agent_id": "uuid-here",
  "message": "Your question here",
  "session_id": "session_123"
}
```

**Parameters:**
- `agent_id` (string, required) - Agent UUID for context
- `message` (string, required) - User message
- `session_id` (string, optional) - Session identifier for conversation continuity

**Response:**
```json
{
  "status": "success",
  "ai_response": "The AI's response based on agent context",
  "agent_id": "uuid-here",
  "session_id": "session_123",
  "conversation_history": [
    {"role": "user", "content": "Your question"},
    {"role": "assistant", "content": "AI response"}
  ]
}
```

**Examples:**
```bash
# cURL
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Explain Python decorators"
  }'

# Python
import requests
response = requests.post(
    "http://localhost:8000/chat/message",
    json={
        "agent_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "Explain Python decorators",
        "session_id": "my_session"
    },
    timeout=90  # AI responses can take time
)
data = response.json()
print(data["ai_response"])
```

### WebSocket Chat

#### `WS /chat/ws/{agent_id}`

Real-time chat via WebSocket for streaming responses.

**Path Parameters:**
- `agent_id` (string, required) - Agent UUID

**Message Format (Client → Server):**
```json
{
  "message": "Your question here"
}
```

**Message Format (Server → Client):**
```json
{
  "type": "response",
  "content": "AI response chunk",
  "done": false
}
```

**Example (Python):**
```python
import asyncio
import websockets
import json

async def chat():
    agent_id = "550e8400-e29b-41d4-a716-446655440000"
    uri = f"ws://localhost:8000/chat/ws/{agent_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({"message": "Hello!"}))
        
        # Receive response
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(data["content"], end="")
            if data.get("done"):
                break

asyncio.run(chat())
```

---

## Admin Endpoints

### System Statistics

#### `GET /admin/stats`

Get comprehensive system statistics.

**Response:**
```json
{
  "agents": {
    "total": 5,
    "by_name": {
      "CoreCoder": 2,
      "MultimodalAgent": 3
    }
  },
  "content": {
    "total": 150,
    "by_type": {
      "text": 100,
      "image": 30,
      "audio": 20
    }
  },
  "vectors": {
    "collections": 6,
    "details": {
      "text_embeddings": {"vectors": 1000},
      "image_embeddings": {"vectors": 500}
    }
  },
  "database": {
    "size_mb": 125.5,
    "agents_size_mb": 10.2,
    "content_size_mb": 115.3
  }
}
```

**Example:**
```bash
curl http://localhost:8000/admin/stats

# Python
import requests
response = requests.get("http://localhost:8000/admin/stats")
stats = response.json()
print(f"Total agents: {stats['agents']['total']}")
print(f"Total content: {stats['content']['total']}")
```

### Health Check

#### `GET /admin/health`

Detailed health check of all components.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "database": "operational",
    "vector_db": "operational",
    "ollama": "available"
  },
  "uptime_seconds": 3600
}
```

**Example:**
```bash
curl http://localhost:8000/admin/health
```

### List Vector Collections

#### `GET /search/collections`

List all available vector collections and their stats.

**Response:**
```json
{
  "collections": [
    "text_embeddings",
    "image_embeddings",
    "audio_embeddings",
    "video_embeddings",
    "agent_knowledge",
    "multimodal_fusion"
  ],
  "total": 6,
  "available": true,
  "stats": {
    "text_embeddings": {"vectors_count": 1000, "indexed_vectors_count": 1000}
  }
}
```

**Example:**
```bash
curl http://localhost:8000/search/collections
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Ollama not running |

---

## Rate Limiting

Currently no rate limiting. Future versions will implement:
- 100 requests/minute for general endpoints
- 10 requests/minute for chat endpoints

---

## CORS

API allows CORS from:
- `http://localhost:3000` (for chatbot-nextjs-webui)

To add more origins, modify `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    ...
)
```

---

## Example: Complete Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Create agent
response = requests.post(
    f"{BASE_URL}/agents/",
    json={
        "name": "ResearchBot",
        "agent_type": "multimodal",
        "description": "Research assistant",
        "tags": ["research", "analysis"]
    }
)
agent = response.json()
agent_id = agent["agent_id"]
print(f"Created agent: {agent_id}")

# 2. Upload content
for i in range(3):
    requests.post(
        f"{BASE_URL}/content/",
        json={
            "agent_id": agent_id,
            "content": f"Research document {i+1}",
            "media_type": "text",
            "metadata": {"doc_number": i+1}
        }
    )
print("Uploaded 3 documents")

# 3. Check status
status = requests.get(f"{BASE_URL}/chat/status").json()
if status["ollama_available"]:
    # 4. Chat with agent
    response = requests.post(
        f"{BASE_URL}/chat/message",
        json={
            "agent_id": agent_id,
            "message": "Summarize the research documents"
        },
        timeout=90
    )
    data = response.json()
    print(f"AI: {data['ai_response']}")

# 5. Get statistics
stats = requests.get(f"{BASE_URL}/admin/stats").json()
print(f"\nTotal agents: {stats['agents']['total']}")
print(f"Total content: {stats['content']['total']}")

# 6. Delete agent (cleanup)
requests.delete(f"{BASE_URL}/agents/{agent_id}")
print(f"Deleted agent: {agent_id}")
```

---

## Testing API

Use the interactive documentation for testing:
1. Navigate to http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Fill in parameters
4. Click "Execute"

---

## See Also

- **Python Library**: [LIBRARY.md](LIBRARY.md) - For scripting
- **CLI Tools**: [CLI.md](CLI.md) - Command-line tools
- **Examples**: [EXAMPLES.md](EXAMPLES.md) - Usage examples
