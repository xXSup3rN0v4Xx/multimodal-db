# 🚀 FastAPI Unified API - Integration Roadmap

## Overview
Build a comprehensive FastAPI layer that serves as the unified interface between:
- **Multimodal-DB** (this system): Data management & vector search
- **chatbot-python-core**: AI utilities & model execution
- **chatbot-nextjs-webui**: Frontend interface & user experience

## 🎯 Phase 1: Core API Foundation

### 1.1 Project Structure
```
multimodal-db/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── dependencies.py     # Auth, database connections
│   ├── middleware.py       # CORS, rate limiting, logging
│   └── routers/
│       ├── __init__.py
│       ├── agents.py       # Agent CRUD operations
│       ├── content.py      # Content management
│       ├── search.py       # Vector & hybrid search
│       ├── chat.py         # Real-time conversations
│       └── admin.py        # System administration
├── models/
│   ├── __init__.py
│   ├── requests.py         # Pydantic request models
│   ├── responses.py        # Pydantic response models
│   └── schemas.py          # Shared data schemas
└── tests/
    ├── test_api.py         # API endpoint tests
    ├── test_integration.py # Integration tests
    └── test_performance.py # Load testing
```

### 1.2 Core Endpoints Design

#### Agent Management (`/agents/`)
```python
# Agent CRUD operations
GET    /agents/                    # List all agents
POST   /agents/                    # Create new agent
GET    /agents/{agent_id}          # Get specific agent
PUT    /agents/{agent_id}          # Update agent
DELETE /agents/{agent_id}          # Delete agent

# Agent specialized operations
POST   /agents/corecoder           # Create CoreCoder agent
POST   /agents/multimodal          # Create multimodal agent
POST   /agents/{agent_id}/export   # Export agent with content
POST   /agents/import              # Import agent from export
```

#### Content Management (`/content/`)
```python
# Content CRUD operations
GET    /content/                   # List content (with filters)
POST   /content/                   # Store new content
GET    /content/{content_id}       # Get specific content
PUT    /content/{content_id}       # Update content
DELETE /content/{content_id}       # Delete content

# Media-specific operations
POST   /content/text              # Store text content
POST   /content/document          # Store document content
POST   /content/embedding         # Store embedding vectors
POST   /content/image             # Store image content (future)
POST   /content/audio             # Store audio content (future)
POST   /content/video             # Store video content (future)
```

#### Search Operations (`/search/`)
```python
# Vector & similarity search
POST   /search/similarity         # Similarity search
POST   /search/hybrid             # Hybrid search (dense + sparse)
POST   /search/agent              # Search within agent's content
POST   /search/media              # Search by media type

# Advanced search patterns
POST   /search/semantic           # Semantic search with embeddings
POST   /search/multimodal         # Cross-modal search (future)
GET    /search/collections        # List available collections
```

#### Real-time Chat (`/chat/`)
```python
# Conversation management
GET    /chat/conversations        # List conversations
POST   /chat/conversations        # Start new conversation
GET    /chat/conversations/{id}   # Get conversation history
DELETE /chat/conversations/{id}   # Delete conversation

# Real-time messaging (WebSocket)
WS     /chat/ws/{agent_id}        # WebSocket connection for real-time chat
POST   /chat/message              # Send message to agent
GET    /chat/stream/{conversation_id} # Server-sent events stream
```

#### System Administration (`/admin/`)
```python
# System health & monitoring
GET    /admin/health              # System health check
GET    /admin/stats               # Database statistics
GET    /admin/metrics             # Performance metrics

# Database operations
POST   /admin/backup              # Create system backup
POST   /admin/restore             # Restore from backup
POST   /admin/cleanup             # Clean up old data
GET    /admin/logs                # System logs
```

## 🔧 Phase 2: Integration Layer

### 2.1 chatbot-python-core Integration

#### Model Execution Bridge
```python
# In api/routers/chat.py
from chatbot_python_core import ModelExecutor, ConversationManager

class ChatRouter:
    def __init__(self):
        self.db = MultimodalDB()
        self.model_executor = ModelExecutor()  # From chatbot-python-core
        
    async def process_message(self, agent_id: str, message: str):
        # 1. Get agent config from our DB
        agent = self.db.get_agent(agent_id)
        
        # 2. Pass to chatbot-python-core for execution
        response = await self.model_executor.generate(
            agent_config=agent.to_dict(),
            message=message,
            model_type="qwen2.5-coder:3b"
        )
        
        # 3. Store conversation in our DB
        self.db.add_message(agent_id, "user", message)
        self.db.add_message(agent_id, "assistant", response)
        
        return response
```

#### Configuration Sharing
```python
# Shared configuration between systems
class IntegrationConfig:
    """Shared configuration for all systems."""
    
    # Database connections
    MULTIMODAL_DB_PATH = "data/multimodal_db"
    QDRANT_PATH = "data/qdrant/vectors"
    
    # Model configurations
    DEFAULT_LLM = "qwen2.5-coder:3b"
    DEFAULT_EMBEDDING = "nomic-embed-text-v1.5"
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_CONTENT_LENGTH = 10_000_000  # 10MB
    
    # chatbot-python-core settings
    MODEL_TIMEOUT = 30
    MAX_CONVERSATION_LENGTH = 50
```

### 2.2 chatbot-nextjs-webui Integration

#### WebSocket Real-time Communication
```typescript
// In chatbot-nextjs-webui
interface MultimodalDBClient {
  // Agent management
  createAgent(config: AgentConfig): Promise<string>
  getAgent(agentId: string): Promise<AgentConfig>
  listAgents(): Promise<AgentConfig[]>
  
  // Real-time chat
  connectChat(agentId: string): WebSocket
  sendMessage(message: string): void
  onMessage(callback: (message: string) => void): void
  
  // Content operations
  uploadContent(file: File, mediaType: MediaType): Promise<string>
  searchContent(query: string, filters?: any): Promise<SearchResult[]>
}
```

#### React Components Integration
```typescript
// Chat interface with our API
function ChatInterface({ agentId }: { agentId: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [ws, setWs] = useState<WebSocket | null>(null)
  
  useEffect(() => {
    // Connect to our FastAPI WebSocket
    const websocket = new WebSocket(`ws://localhost:8000/chat/ws/${agentId}`)
    
    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data)
      setMessages(prev => [...prev, message])
    }
    
    setWs(websocket)
    return () => websocket.close()
  }, [agentId])
  
  const sendMessage = (text: string) => {
    if (ws) {
      ws.send(JSON.stringify({ message: text, agent_id: agentId }))
    }
  }
  
  return (
    <div>
      <MessageList messages={messages} />
      <MessageInput onSend={sendMessage} />
    </div>
  )
}
```

## 🚀 Phase 3: Advanced Features

### 3.1 Authentication & Security
```python
# JWT token authentication
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# CORS for frontend integration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # chatbot-nextjs-webui
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 Performance Optimization
```python
# Async database operations
from asyncio import gather
from concurrent.futures import ThreadPoolExecutor

# Caching layer
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Background tasks
from fastapi import BackgroundTasks

async def background_embed_content(content_id: str):
    """Generate embeddings in background."""
    # Use our QdrantVectorDB for embedding storage
    pass
```

### 3.3 Monitoring & Observability
```python
# Metrics collection
from prometheus_client import Counter, Histogram
from fastapi_prometheus_metrics import PrometheusMetrics

# Logging
import structlog
logger = structlog.get_logger()

# Health checks
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": await check_database_health(),
        "vector_db": await check_qdrant_health(),
        "model_executor": await check_model_health()
    }
```

## 🎯 Implementation Timeline

### Week 1: Foundation
- [ ] Set up FastAPI project structure
- [ ] Implement basic CRUD endpoints for agents
- [ ] Add Pydantic models for requests/responses
- [ ] Create basic integration with MultimodalDB

### Week 2: Core Features
- [ ] Implement content management endpoints
- [ ] Add vector search endpoints with QdrantVectorDB
- [ ] Create WebSocket chat interface
- [ ] Add chatbot-python-core integration

### Week 3: Frontend Integration
- [ ] Implement CORS and security middleware
- [ ] Create WebSocket handlers for real-time chat
- [ ] Build TypeScript client for chatbot-nextjs-webui
- [ ] Add authentication and rate limiting

### Week 4: Production Ready
- [ ] Add comprehensive error handling
- [ ] Implement monitoring and logging
- [ ] Performance testing and optimization
- [ ] Documentation and deployment guides

## 🎉 Expected Outcome

A unified API system that provides:
- **Clean separation** between data (multimodal-db), AI (chatbot-python-core), and UX (chatbot-nextjs-webui)
- **Real-time capabilities** for live agent conversations
- **Scalable architecture** ready for production deployment
- **Comprehensive testing** ensuring reliability
- **Full integration** between all three systems

This creates a powerful, modular AI agent platform where each component can be developed, tested, and deployed independently while working seamlessly together! 🚀