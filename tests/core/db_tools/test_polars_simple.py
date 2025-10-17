"""
Quick test of PolarsQueryEngine with Ollama - Direct LlamaIndex Usage
"""
import polars as pl
from llama_index.experimental.query_engine import PolarsQueryEngine
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# Initialize Ollama
Settings.llm = Ollama(
    model="qwen2.5-coder:latest",
    base_url="http://localhost:11434",
    request_timeout=60.0
)

print("\n" + "="*70)
print("🧪 POLARS QUERY ENGINE TEST WITH OLLAMA")
print("="*70)

# Test 1: Simple DataFrame
print("\n--- Test 1: Simple City/Population Query ---")
df = pl.DataFrame({
    "city": ["Toronto", "Tokyo", "Berlin"],
    "population": [2930000, 13960000, 3645000],
})

print("\nDataFrame:")
print(df)

query_engine = PolarsQueryEngine(df=df, verbose=True)

print("\n🔍 Query: 'What is the city with the highest population?'")
response = query_engine.query("What is the city with the highest population?")

print(f"\n✅ Response: {response}")
print(f"\n📊 Metadata: {response.metadata}")

# Test 2: With response synthesis
print("\n\n--- Test 2: Query with Response Synthesis ---")
query_engine_synth = PolarsQueryEngine(df=df, verbose=True, synthesize_response=True)

print("\n🔍 Query: 'What is the city with the highest population? Give both city and population.'")
response2 = query_engine_synth.query("What is the city with the highest population? Give both city and population")

print(f"\n✅ Synthesized Response: {response2}")

# Test 3: More complex data
print("\n\n--- Test 3: Complex DataFrame ---")
df2 = pl.DataFrame({
    "product": ["Apple", "Banana", "Orange", "Grape", "Mango"],
    "price": [1.20, 0.50, 0.80, 2.50, 1.80],
    "stock": [100, 150, 80, 60, 90]
})

print("\nDataFrame:")
print(df2)

query_engine2 = PolarsQueryEngine(df=df2, verbose=True, synthesize_response=True)

print("\n🔍 Query: 'What is the average price?'")
response3 = query_engine2.query("What is the average price?")
print(f"\n✅ Response: {response3}")

print("\n🔍 Query: 'Which product has the highest stock?'")
response4 = query_engine2.query("Which product has the highest stock?")
print(f"\n✅ Response: {response4}")

print("\n\n" + "="*70)
print("✅ ALL TESTS COMPLETED!")
print("="*70)
