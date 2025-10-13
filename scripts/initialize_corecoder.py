"""
Initialize CoreCoder Agent in the Database
Creates the full CoreCoder agent configuration and stores it in the agent matrix.
"""
import sys
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'multimodal-db'))

from core.base_agent_config import create_corecoder_agent, AgentConfig
from core.multimodal_db import MultimodalDB  # Use the same DB as the API!

def initialize_corecoder():
    """Create and store CoreCoder agent in the database."""
    
    print("🤖 Initializing CoreCoder Agent")
    print("=" * 50)
    
    # Create the CoreCoder agent configuration
    print("\n📝 Creating CoreCoder configuration...")
    corecoder = create_corecoder_agent()
    
    print(f"✅ Agent Name: {corecoder.agent_name}")
    print(f"✅ Agent ID: {corecoder.agent_id}")
    print(f"✅ Description: {corecoder.description[:60]}...")
    print(f"✅ Tags: {corecoder.tags}")
    print(f"✅ Helper Prompts: {len(corecoder.helper_prompts)}")
    
    # Initialize database (same as API uses)
    print("\n💾 Connecting to database...")
    db = MultimodalDB()  # Uses default "multimodal_db" path
    
    # Store the agent
    print(f"\n💾 Storing CoreCoder in agent matrix...")
    agent_id = db.store_agent(corecoder)
    
    print(f"\n✅ SUCCESS! CoreCoder stored in database")
    print(f"   Agent ID: {agent_id}")
    
    # Verify storage
    print("\n🔍 Verifying storage...")
    retrieved = db.get_agent(agent_id)
    
    if retrieved:
        print(f"✅ Verified! Agent retrieved successfully")
        print(f"   Name: {retrieved.agent_name}")
        print(f"   Description: {retrieved.description[:60]}...")
        print(f"   Helper Prompts: {len(retrieved.helper_prompts)}")
    else:
        print("❌ Error: Could not retrieve agent")
    
    # Show all agents
    print("\n📊 Current agents in database:")
    all_agents = db.list_agents()
    for agent in all_agents:
        print(f"   - {agent['name']} (ID: {agent['agent_id'][:8]}...)")
    
    print("\n🎉 CoreCoder initialization complete!")
    print(f"   You can now use agent_id: {agent_id}")
    print(f"   Or search by name: CoreCoder")

if __name__ == "__main__":
    initialize_corecoder()
