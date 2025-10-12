#!/usr/bin/env python3
"""
Comprehensive test for Qdrant + LlamaIndex integration
Tests vector storage, search, and hybrid functionality
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path to access multimodal-db module
sys.path.insert(0, str(Path(__file__).parent.parent / 'multimodal-db'))

from core.base_agent_config import create_corecoder_agent
from core.polars_core import PolarsDBHandler
from core.qdrant_core import QdrantCore
from core.qdrant_hybrid_search_llama_index import create_hybrid_search_system


def test_qdrant_functionality():
    """Test complete Qdrant vector database functionality."""
    
    print("🔍 Qdrant + LlamaIndex Integration Test")
    print("=" * 50)
    
    # Change to parent directory so database paths work correctly
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    
    try:
        # Test 1: Basic Qdrant Core
        print("\n1. Testing Qdrant Core functionality...")
        qdrant = QdrantCore(persist_path="test_qdrant_full")
        
        # Initialize standard collections
        results = qdrant.initialize_standard_collections()
        print(f"   ✅ Standard collections created: {sum(results.values())}/{len(results)}")
        
        # Test collection listing
        collections = qdrant.list_collections()
        print(f"   📁 Collections found: {collections}")
        
        # Test health check
        health = qdrant.health_check()
        print(f"   🏥 Health status: {health['status']}")
        
        # Test 2: Vector Storage and Search
        print("\n2. Testing vector storage and search...")
        
        # Create some test vectors (simplified embeddings)
        test_vectors = [
            [0.1, 0.2, 0.3, 0.4] * 96,  # 384-dim vector
            [0.5, 0.6, 0.7, 0.8] * 96,  # 384-dim vector
            [0.9, 0.1, 0.2, 0.3] * 96,  # 384-dim vector
        ]
        
        test_payloads = [
            {"text": "This is about machine learning and AI", "category": "technology"},
            {"text": "Python programming and software development", "category": "programming"},
            {"text": "Data science and analytics", "category": "data"}
        ]
        
        # Add vectors to knowledge_documents collection
        success = qdrant.add_vectors(
            collection_name="knowledge_documents",
            vectors=test_vectors,
            payloads=test_payloads
        )
        print(f"   ✅ Vectors added successfully: {success}")
        
        # Test search
        query_vector = [0.2, 0.3, 0.4, 0.5] * 96  # 384-dim query
        search_results = qdrant.search_vectors(
            collection_name="knowledge_documents",
            query_vector=query_vector,
            limit=2
        )
        print(f"   🔍 Search results found: {len(search_results)}")
        if search_results:
            print(f"      Best match: {search_results[0]['payload']['text'][:50]}...")
            print(f"      Score: {search_results[0]['score']:.4f}")
        
        # Test 3: Hybrid Search System (if LlamaIndex is available)
        print("\n3. Testing LlamaIndex hybrid search...")
        try:
            hybrid_system = create_hybrid_search_system(
                persist_path="test_hybrid_search",
                collection_name="test_hybrid"
            )
            
            if hybrid_system:
                print("   ✅ Hybrid search system created")
                
                # Test document addition
                test_texts = [
                    "Artificial intelligence is transforming software development",
                    "Machine learning models require large datasets for training",
                    "Python is an excellent language for data science projects"
                ]
                
                test_metadata = [
                    {"topic": "AI", "complexity": "high"},
                    {"topic": "ML", "complexity": "medium"},
                    {"topic": "Python", "complexity": "low"}
                ]
                
                success = hybrid_system.add_text_documents(test_texts, test_metadata)
                print(f"   📝 Documents added: {success}")
                
                if success:
                    # Test semantic search
                    query_results = hybrid_system.semantic_query(
                        "What is artificial intelligence?",
                        top_k=2,
                        response_synthesis=False
                    )
                    
                    if 'results' in query_results:
                        print(f"   🔍 Semantic search results: {len(query_results['results'])}")
                        if query_results['results']:
                            best_result = query_results['results'][0]
                            print(f"      Best match score: {best_result['score']:.4f}")
                    else:
                        print(f"   ⚠️  Search results: {query_results}")
                
                # Test collection stats
                stats = hybrid_system.get_collection_stats()
                print(f"   📊 Collection stats: {stats}")
                
            else:
                print("   ⚠️  Hybrid search system creation failed")
                
        except Exception as e:
            print(f"   ⚠️  LlamaIndex hybrid search test failed: {e}")
        
        # Test 4: Integration with Agent System
        print("\n4. Testing integration with agent system...")
        
        # Create agent and store in Polars
        agent = create_corecoder_agent()
        db = PolarsDBHandler("test_qdrant_integration")
        agent_id = db.add_agent_config(agent)
        print(f"   👤 Agent stored: {agent_id[:8]}...")
        
        # Create vector representation of agent (simplified)
        agent_vector = [0.7, 0.8, 0.9, 0.1] * 96  # 384-dim
        agent_payload = {
            "agent_id": agent_id,
            "agent_name": agent.agent_name,
            "description": agent.description,
            "tags": agent.tags
        }
        
        success = qdrant.add_vectors(
            collection_name="agent_conversations",
            vectors=[agent_vector],
            payloads=[agent_payload]
        )
        print(f"   🤖 Agent vector stored: {success}")
        
        # Search for similar agents
        similar_agents = qdrant.search_vectors(
            collection_name="agent_conversations",
            query_vector=agent_vector,
            limit=1
        )
        print(f"   🔍 Similar agents found: {len(similar_agents)}")
        if similar_agents:
            found_agent = similar_agents[0]['payload']
            print(f"      Found: {found_agent['agent_name']}")
        
        print("\n✅ All Qdrant integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    success = test_qdrant_functionality()
    sys.exit(0 if success else 1)