"""
Quick test of PolarsQueryEngine with Ollama
"""
import polars as pl
import sys
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent / "multimodal-db"))

# Test with direct LlamaIndex import
try:
    from llama_index.experimental.query_engine import PolarsQueryEngine
    from llama_index.core import Settings
    from llama_index_llms_ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    print(f"Import error: {e}")


def test_basic_query():
    """Test 1: Simple DataFrame Query (from docs)"""
    print("\n" + "="*70)
    print("TEST 1: Simple City/Population Query")
    print("="*70)
    
    # Create sample data (same as documentation)
    df = pl.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin"],
        "population": [2930000, 13960000, 3645000],
    })
    
    print("\n📊 DataFrame:")
    print(df)
    
    # Initialize query engine
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    # Test query without synthesis
    print("\n🔍 Query: 'What is the city with the highest population?'")
    print("   (without response synthesis)")
    print("-" * 70)
    
    result = engine.query(
        df,
        "What is the city with the highest population?",
        synthesize_response=False
    )
    
    if result["success"]:
        print(f"\n✅ Raw Result: {result['response']}")
        if result.get('polars_code'):
            print(f"\n🔧 Generated Polars Code:")
            print(f"   {result['polars_code']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return result["success"]


def test_synthesized_response():
    """Test 2: Query with LLM Response Synthesis"""
    print("\n" + "="*70)
    print("TEST 2: Query with Response Synthesis")
    print("="*70)
    
    df = pl.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin", "New York", "London"],
        "population": [2930000, 13960000, 3645000, 8336000, 8982000],
        "country": ["Canada", "Japan", "Germany", "USA", "UK"]
    })
    
    print("\n📊 DataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    print("\n🔍 Query: 'What is the city with the highest population? Give both city and population.'")
    print("   (with response synthesis)")
    print("-" * 70)
    
    result = engine.query(
        df,
        "What is the city with the highest population? Give both the city and population",
        synthesize_response=True
    )
    
    if result["success"]:
        print(f"\n✅ Synthesized Response: {result['response']}")
        if result.get('polars_code'):
            print(f"\n🔧 Generated Polars Code:")
            print(f"   {result['polars_code']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return result["success"]


def test_batch_queries():
    """Test 3: Multiple Queries on Same DataFrame"""
    print("\n" + "="*70)
    print("TEST 3: Batch Queries")
    print("="*70)
    
    df = pl.DataFrame({
        "product": ["Apple", "Banana", "Orange", "Grape", "Mango"],
        "price": [1.20, 0.50, 0.80, 2.50, 1.80],
        "stock": [100, 150, 80, 60, 90]
    })
    
    print("\n📊 DataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    queries = [
        "How many products are there?",
        "What is the average price?",
        "Which product has the highest stock?",
    ]
    
    print("\n🔍 Running batch queries...")
    print("-" * 70)
    
    results = engine.batch_query(df, queries, synthesize_response=True)
    
    all_success = True
    for i, result in enumerate(results, 1):
        print(f"\n📊 Query {i}: {result['query']}")
        if result["success"]:
            print(f"   ✅ Response: {result['response']}")
            if result.get('polars_code'):
                print(f"   🔧 Code: {result['polars_code']}")
        else:
            print(f"   ❌ Error: {result['error']}")
            all_success = False
    
    return all_success


def test_aggregation():
    """Test 4: Direct Aggregation (without LLM)"""
    print("\n" + "="*70)
    print("TEST 4: Direct Aggregation")
    print("="*70)
    
    # Simulate detection data
    df = pl.DataFrame({
        "object_class": ["person", "car", "person", "car", "bike", "person"],
        "confidence": [0.95, 0.88, 0.92, 0.85, 0.78, 0.96],
        "frame": [1, 1, 2, 2, 2, 3]
    })
    
    print("\n📊 DataFrame (detection data):")
    print(df)
    
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    print("\n🔍 Aggregating by object_class...")
    print("-" * 70)
    
    result = engine.aggregate_streaming_data(
        df,
        group_by=["object_class"],
        aggregations={
            "confidence": "mean",
            "object_class": "count"
        }
    )
    
    if result["success"]:
        print("\n✅ Aggregation Results:")
        for row in result["result"]:
            print(f"   {row}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return result["success"]


def test_prompt_inspection():
    """Test 5: Inspect Prompts"""
    print("\n" + "="*70)
    print("TEST 5: Prompt Inspection")
    print("="*70)
    
    df = pl.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30]
    })
    
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    print("\n🔍 Getting prompts...")
    prompts = engine.get_prompts(df)
    
    if prompts:
        print("\n✅ Available Prompts:")
        for name, prompt_info in prompts.items():
            print(f"\n   📝 {name}:")
            template = prompt_info['template']
            # Show first 300 chars
            preview = template[:300] + "..." if len(template) > 300 else template
            print(f"      {preview}")
        return True
    else:
        print("\n❌ Could not retrieve prompts")
        return False


def test_custom_prompt():
    """Test 6: Custom Prompt Template"""
    print("\n" + "="*70)
    print("TEST 6: Custom Prompt")
    print("="*70)
    
    df = pl.DataFrame({
        "department": ["Sales", "Engineering", "Marketing"],
        "employees": [10, 25, 8],
        "budget": [50000, 120000, 40000]
    })
    
    print("\n📊 DataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    custom_prompt = """\
You are a Polars expert. Generate optimized code.

DataFrame: `df`
{df_str}

Instructions:
{instruction_str}

Query: {query_str}

Optimized Polars expression:
"""
    
    print("\n🔍 Query with custom prompt: 'What is the total budget?'")
    print("-" * 70)
    
    result = engine.custom_prompt_query(
        df,
        "What is the total budget?",
        custom_prompt
    )
    
    if result["success"]:
        print(f"\n✅ Response: {result['response']}")
        if result.get('polars_code'):
            print(f"\n🔧 Generated Code:")
            print(f"   {result['polars_code']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return result["success"]


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 POLARS QUERY ENGINE TESTS WITH OLLAMA")
    print("="*70)
    
    # Check availability
    if not LLAMAINDEX_AVAILABLE:
        print("\n❌ LlamaIndex not available!")
        print("Install with: pip install llama-index llama-index-experimental")
        return
    
    print("\n✅ LlamaIndex available")
    print("🤖 Using Ollama model: qwen2.5-coder:3b")
    print("🌐 Ollama URL: http://localhost:11434")
    
    # Run tests
    results = {}
    
    try:
        results['test_1'] = test_basic_query()
        results['test_2'] = test_synthesized_response()
        results['test_3'] = test_batch_queries()
        results['test_4'] = test_aggregation()
        results['test_5'] = test_prompt_inspection()
        results['test_6'] = test_custom_prompt()
        
        # Summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
