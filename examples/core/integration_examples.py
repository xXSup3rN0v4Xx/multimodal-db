"""
Multimodal-DB Integration Examples
Demonstrates key integration patterns between multimodal-db and chatbot-python-core
"""

# =============================================================================
# Example 1: Basic Agent with Conversation Storage
# =============================================================================

def example_1_basic_agent():
    """Create agent, chat, and store conversations."""
    from core import AgentConfig, MultimodalDB
    
    print("=" * 60)
    print("Example 1: Basic Agent with Conversation Storage")
    print("=" * 60)
    
    # Create agent
    agent = AgentConfig(agent_name="ChatBot")
    agent.enable_model("large_language_model", "ollama", {
        "model": "qwen2.5-coder:3b",
        "temperature": 0.7
    })
    agent.set_system_prompt("large_language_model", "ollama",
        "You are a helpful AI assistant.")
    
    # Store agent
    db = MultimodalDB()
    agent_id = db.add_agent(agent)
    print(f"✅ Agent created: {agent_id}")
    
    # Simulate conversation
    conversations = [
        ("user", "Hello! How are you?"),
        ("assistant", "Hello! I'm doing well, thank you for asking!"),
        ("user", "Can you help me with Python?"),
        ("assistant", "Of course! I'd be happy to help with Python. What do you need?")
    ]
    
    for role, content in conversations:
        msg_id = db.add_message(agent_id, role, content)
        print(f"  💬 Stored {role} message")
    
    # Retrieve conversation
    history = db.get_conversation(agent_id, limit=10)
    print(f"\n📚 Conversation History ({len(history)} messages):")
    for msg in history[-4:]:  # Show last 4
        print(f"  {msg['role']}: {msg['content']}")
    
    return agent_id, db


# =============================================================================
# Example 2: RAG with Hybrid Search
# =============================================================================

def example_2_rag_hybrid_search():
    """Add knowledge documents and perform hybrid search."""
    from core import QdrantHybridSearch
    
    print("\n" + "=" * 60)
    print("Example 2: RAG with Hybrid Search")
    print("=" * 60)
    
    # Initialize hybrid search
    rag = QdrantHybridSearch(
        collection_name="demo_knowledge",
        dense_model="BAAI/bge-small-en-v1.5"
    )
    
    if not rag.available:
        print("⚠️  Hybrid search not available (missing dependencies)")
        return
    
    print("✅ Hybrid search initialized")
    
    # Add knowledge documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "FastAPI is a modern web framework for building APIs with Python.",
        "Polars is a blazingly fast DataFrame library written in Rust.",
        "LlamaIndex provides tools for building LLM applications with external data.",
        "Qdrant is a vector similarity search engine for ML applications."
    ]
    
    agent_id = "demo-agent-123"
    doc_ids = rag.add_documents(documents, agent_id=agent_id)
    print(f"📚 Added {len(doc_ids)} knowledge documents")
    
    # Perform hybrid search
    query = "What is FastAPI?"
    results = rag.hybrid_search(query, top_k=3, agent_id=agent_id)
    
    print(f"\n🔍 Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] Score: {result['score']:.3f}")
        print(f"      Text: {result['text'][:80]}...")
    
    # Query with context and response
    response = rag.query_with_context(query, top_k=2, agent_id=agent_id)
    if response["success"]:
        print(f"\n🤖 AI Response:")
        print(f"  {response['response']}")
        print(f"  (Used {response['num_contexts']} context documents)")
    
    return rag


# =============================================================================
# Example 3: Natural Language Queries on Data
# =============================================================================

def example_3_nl_queries():
    """Export data and query with natural language."""
    from core import MultimodalDB, ParquetExporter, PolarsNLQueryEngine
    import polars as pl
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print("Example 3: Natural Language Queries on Data")
    print("=" * 60)
    
    # Create some sample data
    db = MultimodalDB("demo_nl_db")
    
    # Add sample agents
    from core import AgentConfig
    for name in ["CodeBot", "ChatBot", "VisionBot"]:
        agent = AgentConfig(agent_name=name)
        agent.add_tag("demo")
        db.add_agent(agent)
    
    print(f"✅ Created demo database with agents")
    
    # Export to Parquet
    exporter = ParquetExporter("exports/demo")
    exports = exporter.export_multimodal_db(db, output_name="demo_agents")
    agents_file = exports.get("agents")
    
    if not agents_file:
        print("⚠️  Export failed")
        return
    
    print(f"📤 Exported to: {agents_file}")
    
    # Query with natural language
    engine = PolarsNLQueryEngine()
    
    if not engine.available:
        print("⚠️  NL query engine not available (missing LlamaIndex)")
        # Fallback: Direct Polars query
        df = pl.read_parquet(agents_file)
        print(f"\n📊 Direct query result: {df.height} agents")
        print(df.select(["name", "created_at"]))
        return
    
    print("✅ NL Query Engine initialized")
    
    # Natural language queries
    queries = [
        "How many agents are there?",
        "List all agent names",
        "What are the unique tags?"
    ]
    
    for query in queries:
        result = engine.query_from_parquet(agents_file, query)
        if result["success"]:
            print(f"\n❓ Query: {query}")
            print(f"   Answer: {result['response']}")
            if result.get("polars_code"):
                print(f"   Code: {result['polars_code']}")
    
    return engine


# =============================================================================
# Example 4: Temporal Knowledge Graphs
# =============================================================================

def example_4_temporal_graphs():
    """Build temporal knowledge graph with Graphiti."""
    from core import GraphitiDBSync
    from datetime import datetime
    
    print("\n" + "=" * 60)
    print("Example 4: Temporal Knowledge Graphs")
    print("=" * 60)
    
    # Initialize Graphiti
    graphiti = GraphitiDBSync("demo_graph")
    
    if not graphiti.db.available:
        print("⚠️  Graphiti not available (missing dependencies)")
        return
    
    print("✅ Graphiti initialized")
    
    agent_id = "demo-agent-456"
    
    # Add episodes over time
    episodes = [
        "The user asked about Python programming basics.",
        "We discussed variables, functions, and classes in Python.",
        "The user wanted to learn about web development with FastAPI.",
        "We covered REST APIs, routing, and request handling."
    ]
    
    for i, content in enumerate(episodes):
        episode_id = graphiti.add_episode(
            agent_id=agent_id,
            content=content,
            episode_type="conversation",
            source="demo"
        )
        print(f"  📝 Added episode {i+1}")
    
    # Search episodes
    query = "What did we discuss about Python?"
    results = graphiti.search_episodes(query, agent_id=agent_id, limit=3)
    
    print(f"\n🔍 Search: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result['content'][:60]}...")
    
    # Get knowledge graph
    graph = graphiti.get_agent_knowledge_graph(agent_id)
    print(f"\n🕸️  Knowledge Graph:")
    print(f"  Entities: {graph['node_count']}")
    print(f"  Relationships: {graph['edge_count']}")
    
    return graphiti


# =============================================================================
# Example 5: Complete Export for Backup
# =============================================================================

def example_5_complete_export():
    """Export complete agent data for backup/migration."""
    from core import MultimodalDB, ParquetExporter, AgentConfig
    
    print("\n" + "=" * 60)
    print("Example 5: Complete Agent Export")
    print("=" * 60)
    
    # Create agent with rich data
    db = MultimodalDB("demo_export_db")
    
    agent = AgentConfig(agent_name="ExportDemo")
    agent.set_description("Demo agent for export testing")
    agent.add_tag("demo")
    agent.add_tag("export")
    agent.enable_model("large_language_model", "ollama")
    
    agent_id = db.add_agent(agent)
    print(f"✅ Created agent: {agent_id}")
    
    # Add conversations
    for i in range(5):
        db.add_message(agent_id, "user", f"User message {i+1}")
        db.add_message(agent_id, "assistant", f"Assistant response {i+1}")
    
    print(f"  💬 Added 10 conversation messages")
    
    # Export everything
    exporter = ParquetExporter("exports/complete")
    exports = exporter.export_agent_complete(
        agent_id=agent_id,
        multimodal_db=db
    )
    
    print(f"\n📦 Complete Export Summary:")
    print(f"  Export Dir: {exports['export_dir']}")
    print(f"  Files:")
    for name, path in exports['files'].items():
        from pathlib import Path
        size = Path(path).stat().st_size if Path(path).exists() else 0
        print(f"    - {name}: {size} bytes")
    
    return exports


# =============================================================================
# Example 6: Streaming Vision Data (Conceptual)
# =============================================================================

def example_6_streaming_vision():
    """Conceptual example of streaming YOLO data through Polars."""
    print("\n" + "=" * 60)
    print("Example 6: Streaming Vision Data (Conceptual)")
    print("=" * 60)
    
    print("""
This example shows how YOLO detection data from chatbot-python-core
would be streamed through multimodal-db's Polars database:

# From chatbot-python-core:
from chatbot_python_core.core.vision import YoloOBB
yolo = YoloOBB(model_path="yolov8n.pt")

# In multimodal-db:
from core import PolarsDB
import polars as pl

polars_db = PolarsDB("vision_stream")

# Stream detections
for frame in video_stream:
    detections = yolo.detect(frame)
    
    # Batch insert into Polars (high-speed)
    detection_records = [
        {
            "timestamp": detection.timestamp,
            "class": detection.class_name,
            "confidence": detection.confidence,
            "bbox": detection.bbox,
            "frame_id": frame.id
        }
        for detection in detections
    ]
    
    # Efficient batch insert
    polars_db.batch_insert_detections(detection_records)

# Query with natural language
from core import PolarsNLQueryEngine
engine = PolarsNLQueryEngine()

result = engine.query(
    polars_db.detections,
    "What are the top 5 detected objects by count?"
)
print(result["response"])

# Export for analysis
from core import ParquetExporter
exporter = ParquetExporter()
exporter.export_polars_db(polars_db, include_tables=["detections"])
    """)
    
    print("✅ See ARCHITECTURE.md for more integration patterns")


# =============================================================================
# Run All Examples
# =============================================================================

def run_all_examples():
    """Run all integration examples."""
    print("\n")
    print("🚀" * 30)
    print("Multimodal-DB Integration Examples")
    print("🚀" * 30)
    
    try:
        example_1_basic_agent()
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
    
    try:
        example_2_rag_hybrid_search()
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
    
    try:
        example_3_nl_queries()
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
    
    try:
        example_4_temporal_graphs()
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
    
    try:
        example_5_complete_export()
    except Exception as e:
        print(f"❌ Example 5 failed: {e}")
    
    try:
        example_6_streaming_vision()
    except Exception as e:
        print(f"❌ Example 6 failed: {e}")
    
    print("\n" + "✅" * 30)
    print("Examples completed!")
    print("✅" * 30 + "\n")


if __name__ == "__main__":
    run_all_examples()
