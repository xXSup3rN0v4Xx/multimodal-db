#!/usr/bin/env python3
"""
Test script for CoreCoder agent and database integration
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path to access multimodal-db module
sys.path.insert(0, str(Path(__file__).parent.parent / 'multimodal-db'))

from core.base_agent_config import create_corecoder_agent
from core.polars_core import PolarsDBHandler

def main():
    print("🧪 CoreCoder Integration Test")
    print("=" * 40)
    
    # Change to parent directory so database paths work correctly
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    
    # Create CoreCoder agent
    print("1. Creating CoreCoder agent...")
    agent = create_corecoder_agent()
    print(f"   ✅ Agent created: {agent.agent_name}")
    print(f"   📝 Description: {agent.description}")
    print(f"   🏷️ Tags: {agent.tags}")
    
    # Create database handler (will create data/integration_test_db/)
    print("\n2. Creating database handler...")
    db = PolarsDBHandler("integration_test_db")
    print("   ✅ Database handler created (data/integration_test_db/)")
    
    # Test agent storage
    print("\n3. Storing agent in database...")
    try:
        agent_id = db.add_agent_config(agent)
        print(f"   ✅ Agent stored with ID: {agent_id[:8]}...")
        
        # Test agent retrieval
        print("\n4. Retrieving agent from database...")
        retrieved_agent = db.get_agent_config(agent_id, as_object=True)
        
        if retrieved_agent:
            print(f"   ✅ Agent retrieved: {retrieved_agent.agent_name}")
            print(f"   📊 Helper prompts: {len(retrieved_agent.prompts['helper'])}")
            print(f"   🤖 Enabled models: {len([m for models in retrieved_agent.get_enabled_models().values() for m in models])}")
        else:
            print("   ❌ Failed to retrieve agent")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed!")
    
    # Restore original working directory
    os.chdir(original_cwd)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)