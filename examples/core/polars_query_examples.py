"""
Examples demonstrating PolarsQueryEngine usage.
Based on official LlamaIndex documentation.
"""
import polars as pl
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodal_db.core.db_tools.llamaindex_polars_query_engine import PolarsNLQueryEngine


def example_1_simple_dataframe():
    """Example 1: Simple city/population dataframe query."""
    print("\n" + "="*60)
    print("Example 1: Simple DataFrame Query")
    print("="*60)
    
    # Create sample data
    df = pl.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin"],
        "population": [2930000, 13960000, 3645000],
    })
    
    print("\nDataFrame:")
    print(df)
    
    # Initialize query engine with verbose mode
    engine = PolarsNLQueryEngine(
        llm_model="qwen2.5-coder:3b",
        verbose=True
    )
    
    # Query without response synthesis
    print("\n--- Query 1: Without response synthesis ---")
    result = engine.query(
        df,
        "What is the city with the highest population?",
        synthesize_response=False
    )
    
    if result["success"]:
        print(f"Response: {result['response']}")
        if result['polars_code']:
            print(f"Generated Polars code:\n{result['polars_code']}")
    
    # Query with response synthesis
    print("\n--- Query 2: With response synthesis ---")
    result = engine.query(
        df,
        "What is the city with the highest population? Give both the city and population",
        synthesize_response=True
    )
    
    if result["success"]:
        print(f"Response: {result['response']}")
        if result['polars_code']:
            print(f"Generated Polars code:\n{result['polars_code']}")


def example_2_batch_queries():
    """Example 2: Multiple queries on the same DataFrame."""
    print("\n" + "="*60)
    print("Example 2: Batch Queries")
    print("="*60)
    
    # Create sample data
    df = pl.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin", "New York", "London"],
        "population": [2930000, 13960000, 3645000, 8336000, 8982000],
        "country": ["Canada", "Japan", "Germany", "USA", "UK"]
    })
    
    print("\nDataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine(verbose=True)
    
    queries = [
        "How many cities are there?",
        "What is the average population?",
        "Which country has the largest city?",
    ]
    
    print("\n--- Running batch queries ---")
    results = engine.batch_query(df, queries, synthesize_response=True)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        if result["success"]:
            print(f"Response: {result['response']}")
        else:
            print(f"Error: {result['error']}")


def example_3_custom_prompt():
    """Example 3: Using custom prompt template."""
    print("\n" + "="*60)
    print("Example 3: Custom Prompt")
    print("="*60)
    
    df = pl.DataFrame({
        "product": ["Apple", "Banana", "Orange"],
        "price": [1.20, 0.50, 0.80],
        "stock": [100, 150, 80]
    })
    
    print("\nDataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine(verbose=True)
    
    # First, get the current prompts
    print("\n--- Inspecting current prompts ---")
    prompts = engine.get_prompts(df)
    if prompts:
        for name, prompt_info in prompts.items():
            print(f"\n{name}:")
            print(prompt_info['template'][:200] + "...")
    
    # Use custom prompt
    custom_prompt = """\
You are working with a polars dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Generate efficient Polars code.
Expression: """
    
    print("\n--- Using custom prompt ---")
    result = engine.custom_prompt_query(
        df,
        "What is the total value of all products (price * stock)?",
        custom_prompt
    )
    
    if result["success"]:
        print(f"Response: {result['response']}")
        if result['polars_code']:
            print(f"Generated code:\n{result['polars_code']}")


def example_4_parquet_file():
    """Example 4: Query data from Parquet file."""
    print("\n" + "="*60)
    print("Example 4: Query from Parquet File")
    print("="*60)
    
    # Check if conversations file exists
    conversations_file = Path("data/core_db/conversations.parquet")
    
    if not conversations_file.exists():
        print(f"\nSkipping: {conversations_file} not found")
        return
    
    engine = PolarsNLQueryEngine(verbose=True)
    
    print(f"\nQuerying: {conversations_file}")
    result = engine.query_from_parquet(
        str(conversations_file),
        "How many conversations are there in total?",
        synthesize_response=True
    )
    
    if result["success"]:
        print(f"Response: {result['response']}")
        if result['polars_code']:
            print(f"Generated code:\n{result['polars_code']}")
    else:
        print(f"Error: {result['error']}")


def example_5_aggregations():
    """Example 5: Direct aggregations on streaming data."""
    print("\n" + "="*60)
    print("Example 5: Aggregations")
    print("="*60)
    
    # Simulate detection data
    df = pl.DataFrame({
        "object_class": ["person", "car", "person", "car", "bike", "person"],
        "confidence": [0.95, 0.88, 0.92, 0.85, 0.78, 0.96],
        "timestamp": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-03"]
    })
    
    print("\nDataFrame:")
    print(df)
    
    engine = PolarsNLQueryEngine()
    
    print("\n--- Aggregating by object class ---")
    result = engine.aggregate_streaming_data(
        df,
        group_by=["object_class"],
        aggregations={
            "confidence": "mean",
            "object_class": "count"
        }
    )
    
    if result["success"]:
        print("\nAggregation results:")
        for row in result["result"]:
            print(row)
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Polars Query Engine Examples")
    print("Based on LlamaIndex Official Documentation")
    print("="*60)
    
    # Check if LlamaIndex is available
    from multimodal_db.core.db_tools.llamaindex_polars_query_engine import LLAMAINDEX_AVAILABLE
    
    if not LLAMAINDEX_AVAILABLE:
        print("\n⚠️  LlamaIndex not available. Please install:")
        print("   pip install llama-index llama-index-experimental")
        sys.exit(1)
    
    # Run examples
    try:
        example_1_simple_dataframe()
        example_2_batch_queries()
        example_3_custom_prompt()
        example_4_parquet_file()
        example_5_aggregations()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
