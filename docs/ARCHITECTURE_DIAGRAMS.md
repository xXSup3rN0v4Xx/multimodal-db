# Integration Architecture Diagrams

Visual representations of how Multimodal-DB and Chatbot-Python-Core work together.

---

## System Overview

```
╔═══════════════════════════════════════════════════════════════════════╗
║                        COMPLETE AI PLATFORM                            ║
╚═══════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────┐
│                              USER                                      │
│                    (Web UI / Mobile / CLI / API)                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │   INTEGRATION LAYER           │
                │  (Your Application Logic)     │
                └───────┬───────────────┬───────┘
                        │               │
        ┌───────────────▼───┐   ┌──────▼────────────────┐
        │  CHATBOT-PYTHON   │   │   MULTIMODAL-DB       │
        │      CORE         │   │                       │
        │  ┌─────────────┐  │   │  ┌─────────────────┐  │
        │  │ AI MODELS   │  │◄──┼─►│  DATABASES      │  │
        │  │             │  │   │  │                 │  │
        │  │ - Ollama    │  │   │  │ - Polars        │  │
        │  │ - YOLO      │  │   │  │ - Qdrant        │  │
        │  │ - Whisper   │  │   │  │ - Graphiti      │  │
        │  │ - Kokoro    │  │   │  │                 │  │
        │  │ - SDXL      │  │   │  │ Query Engines   │  │
        │  └─────────────┘  │   │  │ Search Tools    │  │
        │                   │   │  │ Export Tools    │  │
        │  Port 8000        │   │  └─────────────────┘  │
        └───────────────────┘   │                       │
                                │  Port 8001            │
                                └───────────────────────┘

        ┌─────────────────────────────────────────────┐
        │  THE BRAIN              THE MEMORY           │
        │  Processes              Stores & Retrieves   │
        │  AI Tasks               Data                 │
        └─────────────────────────────────────────────┘
```

---

## Integration Pattern 1: API-to-API

**Best for:** Production deployments, microservices

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: User sends message                                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Multimodal-DB retrieves agent config & history          │
│                                                                  │
│  ┌──────────────────┐    ┌─────────────────────────────────┐   │
│  │  HTTP Request    │───►│  Polars Database                │   │
│  │  GET /agents/123 │    │  - Agent config                 │   │
│  │                  │    │  - Conversation history         │   │
│  └──────────────────┘    └─────────────────────────────────┘   │
│                                                                  │
│  Multimodal-DB (Port 8001)                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Call Chatbot-Python-Core for AI processing              │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │  HTTP Request                         │                       │
│  │  POST /api/v1/ollama/chat            │                       │
│  │                                       │                       │
│  │  Body:                                │                       │
│  │  {                                    │                       │
│  │    "model": "llama3.2",              │                       │
│  │    "messages": [                      │                       │
│  │      {...previous messages...},       │                       │
│  │      {"role": "user", "content": "?"} │                       │
│  │    ]                                  │                       │
│  │  }                                    │                       │
│  └────────────────┬─────────────────────┘                       │
│                   │                                              │
│  Chatbot-Python-Core (Port 8000)                                │
└───────────────────┼──────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Ollama generates response                               │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │  Ollama LLM (Port 11434)             │                       │
│  │  Processing with full context...     │                       │
│  │                                       │                       │
│  │  Response: "Based on our previous    │                       │
│  │  conversation about X..."             │                       │
│  └──────────────────────────────────────┘                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Store conversation in Multimodal-DB                     │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │  HTTP Request                         │                       │
│  │  POST /api/v1/conversations/message   │                       │
│  │                                       │                       │
│  │  User message → Polars DB             │                       │
│  │  Assistant response → Polars DB       │                       │
│  │                                       │                       │
│  │  ✓ Stored with timestamp              │                       │
│  │  ✓ Indexed for retrieval              │                       │
│  │  ✓ Available for future context       │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
│  Multimodal-DB (Port 8001)                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Response returned to user                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Pattern 2: Direct Integration

**Best for:** Development, examples, single-process apps

```
┌─────────────────────────────────────────────────────────────────┐
│              YOUR PYTHON APPLICATION                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  from chatbot_python_core.core.ollama import Ollama    │    │
│  │  from multimodal_db.core import PolarsDB               │    │
│  │                                                         │    │
│  │  # Direct function calls, no HTTP                      │    │
│  │  db = PolarsDB("my_db")                                │    │
│  │  ollama = OllamaServiceOrchestrator()                  │    │
│  │                                                         │    │
│  │  agent = db.get_agent("agent-123")                     │    │
│  │  response = ollama.chat(model="llama3.2", ...)         │    │
│  │  db.add_message(agent_id, "user", message)             │    │
│  │  db.add_message(agent_id, "assistant", response)       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│         ┌──────────────────┐      ┌──────────────────┐         │
│         │  Direct Imports  │      │  Direct Imports  │         │
│         └────────┬─────────┘      └────────┬─────────┘         │
│                  │                         │                    │
│         ┌────────▼─────────┐      ┌────────▼─────────┐         │
│         │ Chatbot-Python   │      │  Multimodal-DB   │         │
│         │ Core Modules     │      │  Modules         │         │
│         └──────────────────┘      └──────────────────┘         │
│                                                                  │
│  ✓ Lower latency (no HTTP)                                      │
│  ✓ Easier debugging                                             │
│  ✓ Shared memory space                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## YOLO Detection Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│ VIDEO STREAM                                                        │
└────────────────┬───────────────────────────────────────────────────┘
                 │ Frame 1, Frame 2, Frame 3...
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ CHATBOT-PYTHON-CORE: YOLO Object Detection                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/vision/detect                              │     │
│  │                                                           │     │
│  │  Input: Video frame (base64 encoded)                     │     │
│  │                                                           │     │
│  │  YOLO Processing:                                         │     │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │     │
│  │  │ Person │  │  Car   │  │  Dog   │  │  Bike  │         │     │
│  │  │ 95%    │  │  89%   │  │  87%   │  │  82%   │         │     │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │     │
│  │                                                           │     │
│  │  Output: Detections with bounding boxes & confidence     │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Port 8000                                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Detections
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ MULTIMODAL-DB: High-Speed Storage                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/detections/store                           │     │
│  │                                                           │     │
│  │  Polars Database (Optimized for Speed):                  │     │
│  │                                                           │     │
│  │  ┌────────────────────────────────────────────────┐      │     │
│  │  │ timestamp | object_class | confidence | bbox  │      │     │
│  │  ├────────────────────────────────────────────────┤      │     │
│  │  │ 10:00:01  | person       | 0.95       | [...]  │      │     │
│  │  │ 10:00:01  | car          | 0.89       | [...]  │      │     │
│  │  │ 10:00:02  | person       | 0.93       | [...]  │      │     │
│  │  │ 10:00:02  | dog          | 0.87       | [...]  │      │     │
│  │  │ ...       | ...          | ...        | ...    │      │     │
│  │  └────────────────────────────────────────────────┘      │     │
│  │                                                           │     │
│  │  ✓ 10-100x faster than Pandas                            │     │
│  │  ✓ Handles millions of detections                        │     │
│  │  ✓ Zero-copy operations                                  │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Port 8001                                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ QUERY ANALYTICS                                                     │
│                                                                     │
│  User: "How many cars detected in the last hour?"                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/query/polars                               │     │
│  │                                                           │     │
│  │  Polars Query Engine:                                     │     │
│  │  1. Natural language → Polars code                        │     │
│  │  2. Execute query at lightning speed                      │     │
│  │  3. Return: "47 cars detected"                           │     │
│  │                                                           │     │
│  │  Generated Code:                                          │     │
│  │  df.filter(                                               │     │
│  │    (pl.col('timestamp') > one_hour_ago) &                 │     │
│  │    (pl.col('object_class') == 'car')                      │     │
│  │  ).count()                                                │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Response: "47 cars were detected in the last hour"                │
└────────────────────────────────────────────────────────────────────┘
```

---

## RAG (Retrieval Augmented Generation) Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│ STEP 1: User asks question                                         │
│ "What are the safety protocols for handling chemicals?"            │
└────────────────┬───────────────────────────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 2: MULTIMODAL-DB - Search Knowledge Base                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/search/hybrid                              │     │
│  │                                                           │     │
│  │  Qdrant Vector Database:                                 │     │
│  │  ┌────────────────────────────────────────────┐          │     │
│  │  │  Dense Search (Semantic)                   │          │     │
│  │  │  + Sparse Search (BM25 Keywords)           │          │     │
│  │  │  + Neural Reranking                        │          │     │
│  │  │                                             │          │     │
│  │  │  Top 3 Results:                            │          │     │
│  │  │  1. Safety Document Section 3.2            │          │     │
│  │  │  2. Chemical Handling Guide p12            │          │     │
│  │  │  3. Emergency Procedures Manual             │          │     │
│  │  └────────────────────────────────────────────┘          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Port 8001                                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Relevant Documents
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 3: Build Context with Retrieved Documents                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  Context Assembly:                                        │     │
│  │                                                           │     │
│  │  "Based on the following information:                    │     │
│  │                                                           │     │
│  │  Document 1: Always wear protective equipment...         │     │
│  │  Document 2: Store chemicals in ventilated areas...      │     │
│  │  Document 3: In case of spill, evacuate and...          │     │
│  │                                                           │     │
│  │  Question: What are the safety protocols?"               │     │
│  └──────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 4: CHATBOT-PYTHON-CORE - Generate Answer                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/ollama/chat                                │     │
│  │                                                           │     │
│  │  Ollama LLM with Context:                                │     │
│  │  ┌────────────────────────────────────────────┐          │     │
│  │  │  "Based on the provided documents, the     │          │     │
│  │  │  key safety protocols for handling         │          │     │
│  │  │  chemicals are:                            │          │     │
│  │  │                                             │          │     │
│  │  │  1. Always wear appropriate PPE...         │          │     │
│  │  │  2. Ensure proper ventilation...           │          │     │
│  │  │  3. Follow emergency procedures...         │          │     │
│  │  │  4. Store chemicals properly...            │          │     │
│  │  │                                             │          │     │
│  │  │  Sources: Safety Document 3.2, Chemical    │          │     │
│  │  │  Handling Guide, Emergency Procedures"     │          │     │
│  │  └────────────────────────────────────────────┘          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Port 8000                                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 5: MULTIMODAL-DB - Store Q&A for Future Context               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  POST /api/v1/conversations/message                      │     │
│  │                                                           │     │
│  │  Stored:                                                 │     │
│  │  - User question                                         │     │
│  │  - Assistant answer                                      │     │
│  │  - Source documents used                                 │     │
│  │  - Timestamp                                             │     │
│  │                                                           │     │
│  │  ✓ Available for future context                          │     │
│  │  ✓ Can be queried later                                  │     │
│  │  ✓ Improves next interactions                            │     │
│  └──────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 6: Return answer with sources to user                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Storage Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL-DB STORAGE                          │
└────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────┐
│   POLARS DATABASE    │  │  QDRANT VECTOR DB    │  │  GRAPHITI DB   │
│   (High-Speed        │  │  (Semantic Search)   │  │  (Knowledge    │
│    Analytics)        │  │                      │  │   Graph)       │
├──────────────────────┤  ├──────────────────────┤  ├────────────────┤
│                      │  │                      │  │                │
│ ┌─────────────────┐  │  │ ┌─────────────────┐  │  │ ┌──────────┐   │
│ │ Agent Configs   │  │  │ │ Text Embeddings │  │  │ │ Entities │   │
│ │ - Model settings│  │  │ │ - Documents     │  │  │ │ - Dates  │   │
│ │ - Prompts       │  │  │ │ - Conversations │  │  │ │ - Topics │   │
│ │ - Tools         │  │  │ │ - FAQs          │  │  │ └────┬─────┘   │
│ └─────────────────┘  │  │ └─────────────────┘  │  │      │         │
│                      │  │                      │  │      │         │
│ ┌─────────────────┐  │  │ ┌─────────────────┐  │  │ ┌────▼──────┐  │
│ │ Conversations   │  │  │ │ Image Embeddings│  │  │ │Relationships│
│ │ - Messages      │  │  │ │ - Generated imgs│  │  │ │ - Temporal │
│ │ - Timestamps    │  │  │ │ - Uploaded imgs │  │  │ │ - Causal   │
│ │ - Roles         │  │  │ └─────────────────┘  │  │ └───────────┘  │
│ └─────────────────┘  │  │                      │  │                │
│                      │  │ ┌─────────────────┐  │  │ ┌──────────┐   │
│ ┌─────────────────┐  │  │ │ Audio Embeddings│  │  │ │ Queries  │   │
│ │ YOLO Detections │  │  │ │ - Speech files  │  │  │ │ - History│   │
│ │ - Object class  │  │  │ │ - Transcripts   │  │  │ │ - Results│   │
│ │ - Confidence    │  │  │ └─────────────────┘  │  │ └──────────┘   │
│ │ - Bounding box  │  │  │                      │  │                │
│ │ - Timestamps    │  │  │ Hybrid Search:       │  │ Time-Aware:    │
│ └─────────────────┘  │  │ - Dense vectors      │  │ - "What was   │
│                      │  │ - Sparse vectors     │  │   known in    │
│ ┌─────────────────┐  │  │ - BM25               │  │   2023?"      │
│ │ Media Files     │  │  │ - Reranking          │  │                │
│ │ - File paths    │  │  └──────────────────────┘  └────────────────┘
│ │ - Metadata      │  │
│ │ - Types         │  │
│ └─────────────────┘  │
│                      │
│ Export to:           │
│ - Parquet            │
│ - JSON               │
│ - CSV                │
└──────────────────────┘

Natural Language Queries ───────┐
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  QUERY ENGINES                                                     │
│                                                                    │
│  Polars Query Engine  ──► "How many agents created today?"        │
│  Pandas Query Engine  ──► "Show top 5 detections by confidence"   │
│  Graphiti Temporal    ──► "What concepts emerged in Q2 2024?"     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Complete Workflow Example: Customer Support Bot

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. SETUP PHASE (One-time)                                           │
└─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐
    │ Create Agent Config │
    │ - Name: "Support"   │
    │ - Model: Ollama     │
    │ - System Prompt     │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐       ┌──────────────────────┐
    │ Upload FAQs to      │──────►│ Qdrant Vector DB     │
    │ Knowledge Base      │       │ (Hybrid Search)      │
    └─────────────────────┘       └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. CUSTOMER INTERACTION (Runtime)                                   │
└─────────────────────────────────────────────────────────────────────┘

Customer: "I forgot my password, how do I reset it?"
    │
    ▼
┌────────────────────────────────────────────────────┐
│ MULTIMODAL-DB: Search FAQs                         │
│ ┌────────────────────────────────────────────────┐ │
│ │ Hybrid Search finds:                           │ │
│ │ - "Password Reset Guide"                       │ │
│ │ - "Account Recovery Steps"                     │ │
│ │ - "Security FAQs"                              │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ MULTIMODAL-DB: Get Conversation History            │
│ ┌────────────────────────────────────────────────┐ │
│ │ Previous messages with this customer:          │ │
│ │ - "Hello, welcome back!"                       │ │
│ │ - "I need help with billing"                   │ │
│ │ - (resolved)                                   │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ CHATBOT-PYTHON-CORE: Generate Response             │
│ ┌────────────────────────────────────────────────┐ │
│ │ Ollama with:                                   │ │
│ │ - System prompt (be helpful)                   │ │
│ │ - Conversation history                         │ │
│ │ - Retrieved FAQ context                        │ │
│ │                                                │ │
│ │ Response:                                      │ │
│ │ "Hi again! I see you need help with password  │ │
│ │ reset. Here are the steps:                    │ │
│ │ 1. Click 'Forgot Password' on login page      │ │
│ │ 2. Enter your email                           │ │
│ │ 3. Check your email for reset link            │ │
│ │ 4. Follow the link and create new password    │ │
│ │                                                │ │
│ │ If you don't receive the email within 5       │ │
│ │ minutes, check your spam folder."             │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ MULTIMODAL-DB: Store Interaction                   │
│ ┌────────────────────────────────────────────────┐ │
│ │ ✓ Customer question stored                     │ │
│ │ ✓ Bot response stored                          │ │
│ │ ✓ FAQ sources logged                           │ │
│ │ ✓ Timestamp recorded                           │ │
│ │ ✓ Available for next interaction               │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────┐
    │ Customer receives helpful        │
    │ response with context from       │
    │ previous conversation!           │
    └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. ANALYTICS PHASE (Later)                                          │
└─────────────────────────────────────────────────────────────────────┘

Manager: "What are the top 5 most asked questions today?"
    │
    ▼
┌────────────────────────────────────────────────────┐
│ MULTIMODAL-DB: Natural Language Query              │
│ ┌────────────────────────────────────────────────┐ │
│ │ Polars Query Engine:                           │ │
│ │                                                │ │
│ │ Generated Code:                                │ │
│ │ df.filter(pl.col('timestamp') >= today)        │ │
│ │   .groupby('question_category')                │ │
│ │   .count()                                     │ │
│ │   .sort('count', descending=True)              │ │
│ │   .head(5)                                     │ │
│ │                                                │ │
│ │ Results:                                       │ │
│ │ 1. Password Reset - 47 times                   │ │
│ │ 2. Billing Questions - 32 times                │ │
│ │ 3. Account Setup - 28 times                    │ │
│ │ 4. Feature Questions - 21 times                │ │
│ │ 5. Technical Issues - 18 times                 │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Summary

```
╔═══════════════════════════════════════════════════════════════════╗
║                    KEY TAKEAWAYS                                   ║
╚═══════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────┐
│ CHATBOT-PYTHON-CORE                                                │
│ ────────────────────────────────────────────────────────────────  │
│ Role: The BRAIN                                                    │
│ Does: Executes AI models, generates content                       │
│ Port: 8000                                                         │
│ Provides: Chat, Detection, TTS, STT, Image Gen                    │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ MULTIMODAL-DB                                                      │
│ ────────────────────────────────────────────────────────────────  │
│ Role: The MEMORY                                                   │
│ Does: Stores, queries, analyzes data                              │
│ Port: 8001                                                         │
│ Provides: Storage, Search, Queries, Analytics, Export             │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ TOGETHER = COMPLETE AI PLATFORM                                    │
│ ────────────────────────────────────────────────────────────────  │
│ ✓ Persistent memory                                               │
│ ✓ Context-aware conversations                                     │
│ ✓ Knowledge base with RAG                                         │
│ ✓ High-speed analytics                                            │
│ ✓ Natural language queries                                        │
│ ✓ Complete audit trail                                            │
│ ✓ Production-ready                                                │
└───────────────────────────────────────────────────────────────────┘
```

---

**For more details, see:**
- `HOW_IT_WORKS_TOGETHER.md` - Complete guide
- `QUICK_REFERENCE.md` - Quick reference
- `INTEGRATION_ANALYSIS.md` - Technical analysis
