"""
Database Cleanup Script
Interactive tool to clean up duplicate and test agents from the database.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "multimodal-db"))

from core import MultimodalDB


def main():
    """Interactive agent cleanup tool."""
    print("🧹 Multimodal-DB Agent Cleanup Tool\n")
    
    # Initialize database
    db = MultimodalDB()
    agents = db.list_agents()
    
    print(f"📊 Total agents in database: {len(agents)}\n")
    
    # Group agents by name
    agent_groups = {}
    for agent in agents:
        name = agent.get('name', 'Unknown')
        if name not in agent_groups:
            agent_groups[name] = []
        agent_groups[name].append(agent)
    
    # Display summary
    print("📋 Agent Summary:")
    for name, group in agent_groups.items():
        count = len(group)
        status = "⚠️  DUPLICATE" if count > 1 else "✅"
        print(f"  {status} {name}: {count} instance(s)")
    
    print("\n" + "="*60 + "\n")
    
    # Show all agents with details
    print("🔍 All Agents:\n")
    for i, agent in enumerate(agents, 1):
        name = agent.get('name', 'Unknown')
        agent_id = agent.get('agent_id', 'N/A')
        created = agent.get('created_at', 'N/A')
        
        # Truncate ID and timestamp for display
        id_short = agent_id[:24] + "..." if len(agent_id) > 24 else agent_id
        created_short = created[:19] if created and len(created) > 19 else created
        
        print(f"  [{i}] {name}")
        print(f"      ID: {id_short}")
        print(f"      Created: {created_short}")
        print()
    
    print("="*60 + "\n")
    
    # Cleanup options
    print("🗑️  Cleanup Options:\n")
    print("  [1] Remove all 'test_db_agent' entries")
    print("  [2] Remove duplicate CoreCoder agents (keep newest)")
    print("  [3] Remove specific agent by number")
    print("  [4] Remove multiple agents by numbers (e.g., 1,2,3)")
    print("  [5] Exit without changes")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        # Remove all test agents
        test_agents = [a for a in agents if a.get('name') == 'test_db_agent']
        print(f"\n⚠️  This will delete {len(test_agents)} test agents!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        
        if confirm == 'yes':
            deleted = 0
            for agent in test_agents:
                if db.delete_agent(agent['agent_id']):
                    deleted += 1
            print(f"✅ Deleted {deleted} test agents")
        else:
            print("❌ Cancelled")
    
    elif choice == "2":
        # Remove duplicate CoreCoders, keep newest
        corecoder_agents = [a for a in agents if a.get('name') == 'CoreCoder']
        if len(corecoder_agents) <= 1:
            print("✅ No duplicate CoreCoder agents found")
        else:
            # Sort by created_at, keep the newest
            corecoder_agents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            to_delete = corecoder_agents[1:]  # All except the first (newest)
            
            print(f"\n⚠️  This will delete {len(to_delete)} duplicate CoreCoder agent(s)")
            print(f"   Keeping: {corecoder_agents[0]['agent_id'][:24]}... (newest)")
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            
            if confirm == 'yes':
                deleted = 0
                for agent in to_delete:
                    if db.delete_agent(agent['agent_id']):
                        deleted += 1
                print(f"✅ Deleted {deleted} duplicate CoreCoder agent(s)")
            else:
                print("❌ Cancelled")
    
    elif choice == "3":
        # Remove specific agent
        agent_num = input("Enter agent number to delete: ").strip()
        try:
            idx = int(agent_num) - 1
            if 0 <= idx < len(agents):
                agent = agents[idx]
                print(f"\n⚠️  Delete: {agent['name']} ({agent['agent_id'][:24]}...)?")
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    if db.delete_agent(agent['agent_id']):
                        print(f"✅ Deleted agent: {agent['name']}")
                    else:
                        print(f"❌ Failed to delete agent")
                else:
                    print("❌ Cancelled")
            else:
                print("❌ Invalid agent number")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "4":
        # Remove multiple agents
        numbers = input("Enter agent numbers separated by commas (e.g., 1,2,3): ").strip()
        try:
            indices = [int(n.strip()) - 1 for n in numbers.split(',')]
            to_delete = [agents[i] for i in indices if 0 <= i < len(agents)]
            
            if not to_delete:
                print("❌ No valid agents selected")
            else:
                print(f"\n⚠️  This will delete {len(to_delete)} agent(s):")
                for agent in to_delete:
                    print(f"   - {agent['name']} ({agent['agent_id'][:24]}...)")
                
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    deleted = 0
                    for agent in to_delete:
                        if db.delete_agent(agent['agent_id']):
                            deleted += 1
                    print(f"✅ Deleted {deleted} agent(s)")
                else:
                    print("❌ Cancelled")
        except (ValueError, IndexError):
            print("❌ Invalid input")
    
    elif choice == "5":
        print("👋 Exiting without changes")
    
    else:
        print("❌ Invalid option")
    
    # Show final count
    print()
    remaining = db.list_agents()
    print(f"📊 Final agent count: {len(remaining)}")
    print("✨ Cleanup complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
