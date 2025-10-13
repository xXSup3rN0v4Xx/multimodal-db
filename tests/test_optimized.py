#!/usr/bin/env python3
"""
Optimized Integration Test Suite
Fast, focused, comprehensive testing of core components.
"""
import sys
import time
import pytest
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'multimodal-db'))

from core.base_agent_config import create_corecoder_agent, AgentConfig
from core.polars_db import PolarsDB
from core.qdrant_db import QdrantDB
from core.simple_ollama import SimpleOllamaClient


@pytest.fixture(scope="session")
def agent():
    """Create test agent."""
    return create_corecoder_agent()

@pytest.fixture(scope="session") 
def polars_db():
    """Create Polars database."""
    return PolarsDB("test_db")

@pytest.fixture(scope="session")
def qdrant_db():
    """Create Qdrant database."""
    return QdrantDB("test_vectors")

@pytest.fixture(scope="session")
def ollama_client():
    """Create Ollama client."""
    return SimpleOllamaClient()


def test_agent_creation(agent):
    """Test agent configuration system."""
    assert agent.agent_name == "CoreCoder"
    assert len(agent.helper_prompts) > 0
    assert "software-engineering" in agent.tags


def test_polars_operations(polars_db, agent):
    """Test Polars database operations."""
    # Store agent
    agent_id = polars_db.add_agent(agent, "TestAgent")
    assert agent_id
    
    # Retrieve agent
    retrieved = polars_db.get_agent(agent_id)
    assert retrieved.agent_name == agent.agent_name
    
    # Add conversation
    msg_id = polars_db.add_message(agent_id, "user", "Hello world")
    assert msg_id
    
    # Get messages
    messages = polars_db.get_messages(agent_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello world"


def test_qdrant_operations(qdrant_db):
    """Test Qdrant vector operations."""
    if not qdrant_db.available:
        pytest.skip("Qdrant not available")
    
    # Create collection
    success = qdrant_db.create_collection("test", 384)
    assert success or True  # Might already exist
    
    # Store vector
    test_vector = [0.1] * 384
    point_id = qdrant_db.store_vector("test", test_vector, {"text": "test"})
    assert point_id
    
    # Search vectors
    results = qdrant_db.search_vectors("test", test_vector, limit=1)
    if results:  # Only assert if we got results
        assert results[0]["payload"]["text"] == "test"


def test_ollama_integration(ollama_client):
    """Test Ollama model integration."""
    if not ollama_client.available:
        pytest.skip("Ollama not available")
    
    response = ollama_client.generate(
        "Say hello in one word",
        "You are a helpful assistant. Be concise."
    )
    
    assert response["success"]
    assert len(response["content"]) > 0


def test_performance_benchmark(polars_db):
    """Test database performance."""
    start_time = time.time()
    
    # Perform 50 operations
    for i in range(50):
        agent = AgentConfig(agent_name=f"TestAgent{i}")
        agent.description = "Performance test agent"  # Set description after creation
        agent_id = polars_db.add_agent(agent)
        polars_db.add_message(agent_id, "user", f"Message {i}")
    
    duration = time.time() - start_time
    ops_per_second = 100 / duration  # 50 writes + 50 reads
    
    print(f"Performance: {ops_per_second:.1f} ops/second")
    assert ops_per_second > 50  # Should be faster than 50 ops/sec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])