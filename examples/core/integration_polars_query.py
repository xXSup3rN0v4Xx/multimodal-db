"""
Integration Example: Using PolarsQueryEngine with PolarsDB
Demonstrates natural language queries on agent conversations and data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodal_db.core.dbs.polars_db import PolarsDB
from multimodal_db.core.db_tools.llamaindex_polars_query_engine import PolarsNLQueryEngine, LLAMAINDEX_AVAILABLE
from multimodal_db.core.agent_configs.base_agent_config import AgentConfig


def example_query_conversations():
    """Example: Natural language queries on conversation history."""
    print("\n" + "="*70)
    print("Example: Query Conversation History with Natural Language")
    print("="*70)
    
    # Initialize database and query engine
    db = PolarsDB(db_path="test_db")
    query_engine = PolarsNLQueryEngine(verbose=True)
    
    # Create a test agent if needed
    agents = db.list_agents()
    if not agents:
        print("\n📝 Creating test agent...")
        test_agent = AgentConfig(
            agent_name="TestAssistant",
            description="A test agent for demonstrating queries",
            system_prompt="You are a helpful assistant.",
            tags=["test", "demo"]
        )
        agent_id = db.add_agent(test_agent)
        
        # Add some test messages
        print("📝 Adding test conversations...")
        messages = [
            ("user", "Hello, how are you?"),
            ("assistant", "I'm doing well, thank you! How can I help you today?"),
            ("user", "Can you explain quantum computing?"),
            ("assistant", "Quantum computing uses quantum mechanics principles to process information. It uses qubits instead of classical bits."),
            ("user", "That's interesting, thanks!"),
            ("assistant", "You're welcome! Feel free to ask if you have more questions."),
        ]
        for role, content in messages:
            db.add_message(agent_id, role, content)
    else:
        agent_id = agents[0]["agent_id"]
        print(f"\n📊 Using existing agent: {agents[0]['name']}")
    
    # Get conversation dataframe
    print(f"\n🔍 Querying conversations for agent: {agent_id}")
    
    # Access the conversations DataFrame directly
    conversations_df = db.conversations.filter(
        db.conversations["agent_id"] == agent_id
    )
    
    print(f"\n📈 Total messages: {conversations_df.height}")
    print("\nDataFrame preview:")
    print(conversations_df.head())
    
    # Natural language queries
    queries = [
        "How many messages are there in total?",
        "How many messages per role?",
        "What is the average content length by role?",
        "Show the most recent 3 messages",
    ]
    
    print("\n" + "="*70)
    print("Running Natural Language Queries")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📊 Query {i}: {query}")
        print("-" * 70)
        
        result = query_engine.query(
            conversations_df,
            query,
            synthesize_response=True
        )
        
        if result["success"]:
            print(f"✅ Response: {result['response']}")
            if result.get('polars_code'):
                print(f"🔧 Generated Polars code:")
                print(f"   {result['polars_code']}")
        else:
            print(f"❌ Error: {result['error']}")


def example_query_agents():
    """Example: Analyze agent configurations."""
    print("\n" + "="*70)
    print("Example: Analyze Agent Configurations")
    print("="*70)
    
    db = PolarsDB(db_path="test_db")
    query_engine = PolarsNLQueryEngine(verbose=True)
    
    # Get agents dataframe
    agents_df = db.agents
    
    if agents_df.height == 0:
        print("\n⚠️  No agents found in database. Run example_query_conversations first.")
        return
    
    print(f"\n📊 Total agents: {agents_df.height}")
    print("\nAgents DataFrame:")
    print(agents_df)
    
    # Query agents
    queries = [
        "How many agents are there?",
        "Show all agent names and creation dates",
        "What is the most recently created agent?",
    ]
    
    print("\n" + "="*70)
    print("Running Agent Analysis Queries")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📊 Query {i}: {query}")
        print("-" * 70)
        
        result = query_engine.query(
            agents_df,
            query,
            synthesize_response=True
        )
        
        if result["success"]:
            print(f"✅ Response: {result['response']}")
            if result.get('polars_code'):
                print(f"🔧 Generated code: {result['polars_code']}")
        else:
            print(f"❌ Error: {result['error']}")


def example_analyze_from_parquet():
    """Example: Analyze using pre-configured methods."""
    print("\n" + "="*70)
    print("Example: Pre-configured Analysis Methods")
    print("="*70)
    
    db = PolarsDB(db_path="test_db")
    query_engine = PolarsNLQueryEngine(verbose=True)
    
    # Ensure we have data
    agents = db.list_agents()
    if agents:
        db.save()  # Ensure files are written
        
        # Use the pre-configured analysis method
        conversations_file = str(db.conversations_file)
        
        print(f"\n📊 Analyzing: {conversations_file}")
        
        result = query_engine.analyze_conversations(
            conversations_file,
            analysis_queries=[
                "How many total messages are there?",
                "What is the distribution of roles (user vs assistant)?",
                "What is the average content length per message?",
            ]
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n✅ Analysis Results:")
            print(f"   Total records: {result['total_records']}")
            print(f"   Columns: {result['columns']}")
            
            print("\n📈 Query Results:")
            for i, analysis in enumerate(result['analyses'], 1):
                if analysis['success']:
                    print(f"\n   {i}. {analysis['query']}")
                    print(f"      Response: {analysis['response']}")
                else:
                    print(f"\n   {i}. {analysis['query']}")
                    print(f"      Error: {analysis['error']}")
    else:
        print("\n⚠️  No data found. Run example_query_conversations first.")


def example_inspect_prompts():
    """Example: Inspect the prompts used by the query engine."""
    print("\n" + "="*70)
    print("Example: Inspect Query Engine Prompts")
    print("="*70)
    
    import polars as pl
    
    # Create sample DataFrame
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"]
    })
    
    query_engine = PolarsNLQueryEngine(verbose=True)
    
    # Get prompts
    print("\n🔍 Inspecting query engine prompts...")
    prompts = query_engine.get_prompts(df)
    
    if prompts:
        for name, prompt_info in prompts.items():
            print(f"\n📝 Prompt: {name}")
            print("-" * 70)
            template = prompt_info['template']
            # Show first 500 characters
            if len(template) > 500:
                print(template[:500] + "\n... (truncated)")
            else:
                print(template)
    else:
        print("❌ Could not retrieve prompts")


def example_custom_prompts():
    """Example: Use custom prompts for specific analysis."""
    print("\n" + "="*70)
    print("Example: Custom Prompt for Code Generation")
    print("="*70)
    
    db = PolarsDB(db_path="test_db")
    
    # Get conversations dataframe
    conversations_df = db.conversations
    
    if conversations_df.height == 0:
        print("\n⚠️  No conversations found. Run example_query_conversations first.")
        return
    
    query_engine = PolarsNLQueryEngine(verbose=True)
    
    # Custom prompt that emphasizes efficiency
    custom_prompt = """\
You are an expert Polars developer. Generate highly optimized Polars code.

DataFrame: `df`
Schema:
{df_str}

Instructions:
{instruction_str}

User Query: {query_str}

Generate the most efficient Polars expression (single line preferred):
"""
    
    print("\n🔧 Using custom prompt for optimized code generation...")
    print("\nQuery: Find the longest message content by role")
    
    result = query_engine.custom_prompt_query(
        conversations_df,
        "Find the longest message content by role",
        custom_prompt
    )
    
    if result["success"]:
        print(f"\n✅ Response: {result['response']}")
        if result.get('polars_code'):
            print(f"\n🔧 Generated Polars code:")
            print(f"   {result['polars_code']}")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 PolarsDB + PolarsQueryEngine Integration Examples")
    print("="*70)
    
    # Check availability
    if not LLAMAINDEX_AVAILABLE:
        print("\n❌ LlamaIndex not available. Please install:")
        print("   pip install llama-index llama-index-experimental polars")
        sys.exit(1)
    
    print("\n✅ LlamaIndex available")
    print("🔧 Make sure Ollama is running with qwen2.5-coder:3b model")
    print("   Run: ollama run qwen2.5-coder:3b")
    
    try:
        # Run examples in sequence
        example_query_conversations()
        example_query_agents()
        example_analyze_from_parquet()
        example_inspect_prompts()
        example_custom_prompts()
        
        print("\n" + "="*70)
        print("✅ All integration examples completed!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
