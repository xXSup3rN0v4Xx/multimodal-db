"""
Multi-Agent Conversation Generator with JSONL Export
Integrates with Graphiti and Ollama for conversation generation and export.
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import polars as pl

from .polars_db import PolarsDBHandler
from .graphiti_pipe import GraphitiRAGFramework
from .base_agent_config import AgentConfig

class ConversationGenerator:
    """
    Generates multi-agent conversations and exports them in various formats.
    """
    
    def __init__(self, db_handler: PolarsDBHandler, graphiti_framework: Optional[GraphitiRAGFramework]):
        """
        Initialize conversation generator.
        
        Args:
            db_handler: Database handler for storage
            graphiti_framework: RAG framework for context (optional for demo mode)
        """
        self.db_handler = db_handler
        self.graphiti_framework = graphiti_framework
        self.logger = logging.getLogger(__name__)
        
    def generate_conversation(self, 
                            agents: List[str], 
                            topic: str,
                            num_turns: int = 10,
                            conversation_id: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a multi-agent conversation on a given topic.
        
        Args:
            agents: List of agent IDs to participate
            topic: Conversation topic
            num_turns: Number of conversation turns
            conversation_id: Optional conversation ID
            context: Optional context information
            
        Returns:
            Dictionary containing conversation data
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            
        conversation = {
            "conversation_id": conversation_id,
            "topic": topic,
            "participants": agents,
            "turns": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "num_turns": num_turns,
                "context": context or {}
            }
        }
        
        # Get agent configurations
        agent_configs = {}
        for agent_id in agents:
            config = self.db.get_agent_config(agent_id)
            if config:
                agent_configs[agent_id] = config
        
        # Generate conversation turns
        for turn in range(num_turns):
            current_agent = agents[turn % len(agents)]
            
            # Get relevant context from knowledge base
            knowledge_context = []
            try:
                # Search knowledge base directly through DB handler
                knowledge_df = self.db.search_knowledge_base(current_agent, topic)
                if knowledge_df.height > 0:
                    knowledge_context = knowledge_df.head(3).to_dicts()
            except Exception as e:
                self.logger.warning(f"Failed to get knowledge context: {e}")
                knowledge_context = []
            
            # Simulate conversation turn (in real implementation, this would call Ollama)
            turn_data = self._generate_turn(
                agent_id=current_agent,
                agent_config=agent_configs.get(current_agent, {}),
                topic=topic,
                previous_turns=conversation["turns"],
                knowledge_context=knowledge_context,
                turn_number=turn
            )
            
            conversation["turns"].append(turn_data)
            
            # Store turn in database
            self._store_conversation_turn(conversation_id, turn_data)
        
        return conversation
    
    def _generate_turn(self,
                      agent_id: str,
                      agent_config: Dict[str, Any],
                      topic: str,
                      previous_turns: List[Dict[str, Any]],
                      knowledge_context: List[Dict[str, Any]],
                      turn_number: int) -> Dict[str, Any]:
        """
        Generate a single conversation turn.
        
        Args:
            agent_id: ID of the speaking agent
            agent_config: Agent configuration
            topic: Conversation topic
            previous_turns: Previous conversation turns
            knowledge_context: Relevant knowledge entries
            turn_number: Current turn number
            
        Returns:
            Turn data dictionary
        """
        # Extract agent personality from config
        prompts = agent_config.get("prompt_config", {})
        agent_personality = prompts.get("agent", {}).get("llmSystem", "")
        prime_directive = prompts.get("primeDirective", "")
        
        # Build context from previous turns
        context_text = ""
        if previous_turns:
            context_text = "\n".join([
                f"{turn['agent_id']}: {turn['content']}" 
                for turn in previous_turns[-3:]  # Last 3 turns for context
            ])
        
        # Build knowledge context
        knowledge_text = ""
        if knowledge_context:
            knowledge_text = "\n".join([
                f"- {entry['title']}: {entry['content'][:200]}..." 
                for entry in knowledge_context
            ])
        
        # TODO: Replace with actual model execution from chatbot-python-core
        # This should:
        # 1. Load the agent's configuration and prompts
        # 2. Build a context-aware prompt with topic, knowledge, and history
        # 3. Call the appropriate model (Ollama, etc.) via chatbot-python-core
        # 4. Return the generated response
        
        # Placeholder response indicating model integration needed
        response_content = (
            f"[Agent {agent_id}] Turn {turn_number} on '{topic}': "
            f"This is a placeholder response from the database layer. "
            f"Actual conversation generation requires integration with the "
            f"chatbot-python-core model execution system."
            f" (Knowledge context: {len(knowledge_context)} entries, "
            f"Previous turns: {len(previous_turns)})"
        )
        
        turn_data = {
            "turn_number": turn_number,
            "agent_id": agent_id,
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "agent_type": agent_config.get("agent_type", "unknown"),
                "context_used": len(previous_turns),
                "knowledge_entries": len(knowledge_context),
                "word_count": len(response_content.split())
            }
        }
        
        return turn_data
    
    def _store_conversation_turn(self, conversation_id: str, turn_data: Dict[str, Any]):
        """Store a conversation turn in the database."""
        self.db.add_conversation_message(
            agent_id=turn_data["agent_id"],
            role="assistant",  # Since this is the agent's response
            content=turn_data["content"],
            session_id=conversation_id,
            metadata=turn_data["metadata"]
        )
    
    def export_conversation_jsonl(self, 
                                 conversation_id: str, 
                                 output_path: str,
                                 include_metadata: bool = True) -> str:
        """
        Export a conversation to JSONL format.
        
        Args:
            conversation_id: ID of conversation to export (session_id)
            output_path: Path to save JSONL file
            include_metadata: Whether to include metadata in export
            
        Returns:
            Path to exported file
        """
        # Get conversation data from database by session_id
        conversation_df = self.db.conversations.filter(
            pl.col("session_id") == conversation_id
        ).sort("timestamp")
        
        if conversation_df.height == 0:
            raise ValueError(f"No conversation found with ID: {conversation_id}")
        
        # Convert to JSONL format
        jsonl_lines = []
        
        for row in conversation_df.to_dicts():
            entry = {
                "conversation_id": row["conversation_id"],
                "agent_id": row["agent_id"],
                "role": row["role"],
                "content": row["content"],
                "session_id": row["session_id"],
                "timestamp": str(row["timestamp"])
            }
            
            if include_metadata and row["metadata"]:
                try:
                    entry["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    entry["metadata"] = {"raw": row["metadata"]}
            
            jsonl_lines.append(json.dumps(entry, ensure_ascii=False))
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(jsonl_lines))
        
        print(f"Exported {len(jsonl_lines)} conversation turns to: {output_file}")
        return str(output_file)
    
    def export_multiple_conversations_jsonl(self,
                                          conversation_ids: List[str],
                                          output_path: str,
                                          include_metadata: bool = True) -> str:
        """
        Export multiple conversations to a single JSONL file.
        
        Args:
            conversation_ids: List of conversation IDs to export
            output_path: Path to save JSONL file
            include_metadata: Whether to include metadata in export
            
        Returns:
            Path to exported file
        """
        all_lines = []
        
        for conv_id in conversation_ids:
            try:
                # Get conversation data
                conversation_df = self.db.get_conversation_history(conv_id)
                
                for row in conversation_df.iter_rows(named=True):
                    entry = {
                        "conversation_id": row["conversation_id"],
                        "turn_number": row["turn_number"], 
                        "agent_id": row["agent_id"],
                        "message": row["message"],
                        "timestamp": row["timestamp"]
                    }
                    
                    if include_metadata and row["metadata"]:
                        try:
                            entry["metadata"] = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            entry["metadata"] = {"raw": row["metadata"]}
                    
                    all_lines.append(json.dumps(entry, ensure_ascii=False))
                    
            except Exception as e:
                print(f"Warning: Could not export conversation {conv_id}: {e}")
                continue
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_lines))
        
        print(f"Exported {len(all_lines)} total turns from {len(conversation_ids)} conversations to: {output_file}")
        return str(output_file)
    
    def generate_training_dataset(self,
                                topic_list: List[str],
                                agents: List[str],
                                turns_per_conversation: int = 8,
                                output_path: str = "training_dataset.jsonl") -> str:
        """
        Generate a training dataset of multi-agent conversations.
        
        Args:
            topic_list: List of topics for conversations
            agents: List of agent IDs to participate
            turns_per_conversation: Number of turns per conversation
            output_path: Path to save training dataset
            
        Returns:
            Path to exported dataset
        """
        conversation_ids = []
        
        print(f"Generating training dataset with {len(topic_list)} conversations...")
        
        for i, topic in enumerate(topic_list):
            print(f"Generating conversation {i+1}/{len(topic_list)}: {topic}")
            
            conversation = self.generate_conversation(
                agents=agents,
                topic=topic,
                num_turns=turns_per_conversation,
                context={"dataset_generation": True, "topic_index": i}
            )
            
            conversation_ids.append(conversation["conversation_id"])
        
        # Export all conversations to JSONL
        dataset_path = self.export_multiple_conversations_jsonl(
            conversation_ids=conversation_ids,
            output_path=output_path,
            include_metadata=True
        )
        
        print(f"Training dataset generated: {dataset_path}")
        return dataset_path

def demo_conversation_generator():
    """Demonstrate the conversation generator functionality."""
    print("=== Multi-Agent Conversation Generator Demo ===")
    
    # Initialize components
    db_handler = PolarsDBHandler("demo_conversation_db")
    
    # Create and store sample agents
    from .base_agent_config import create_corecoder_agent, AgentConfig
    
    print("Creating sample agents...")
    corecoder = create_corecoder_agent()
    agent_id = db_handler.add_agent_config(corecoder)
    print(f"Created CoreCoder agent: {agent_id[:8]}...")
    
    # For demo purposes, we can reference agents by their stored IDs
    # In a real scenario, these would be actual agent IDs from the database
    agents = [agent_id, "assistant_001", "researcher_001"]  # Mix of real and placeholder IDs
    
    # Initialize GraphitiRAGFramework and ConversationGenerator
    # Note: This is a placeholder - actual implementation requires Neo4j setup
    try:
        graphiti_framework = GraphitiRAGFramework(db_path="demo_conversation_db")
        generator = ConversationGenerator(db_handler, graphiti_framework)
    except Exception as e:
        print(f"⚠️  Graphiti initialization failed (requires Neo4j): {e}")
        print("Using minimal demo mode...")
        generator = ConversationGenerator(db_handler, None)  # Allow None for demo
    
    # Generate a single conversation
    print("\n1. Generating single conversation...")
    conversation = generator.generate_conversation(
        agents=agents,
        topic="Artificial Intelligence and Machine Learning",
        num_turns=6
    )
    
    print(f"Generated conversation with {len(conversation['turns'])} turns")
    print(f"Conversation ID: {conversation['conversation_id']}")
    
    # Export to JSONL
    print("\n2. Exporting conversation to JSONL...")
    jsonl_path = generator.export_conversation_jsonl(
        conversation_id=conversation["conversation_id"],
        output_path="single_conversation.jsonl"
    )
    
    # Generate training dataset
    print("\n3. Generating training dataset...")
    topics = [
        "Database Optimization Strategies",
        "Multi-Agent System Coordination", 
        "Deep Learning Architecture Design",
        "Distributed Computing Patterns",
        "AI Safety and Alignment"
    ]
    
    dataset_path = generator.generate_training_dataset(
        topic_list=topics,
        agents=agents,
        turns_per_conversation=4,
        output_path="multi_agent_training_dataset.jsonl"
    )
    
    print(f"\nDemo completed!")
    print(f"Single conversation: {jsonl_path}")
    print(f"Training dataset: {dataset_path}")
    
    return generator

if __name__ == "__main__":
    demo_conversation_generator()
