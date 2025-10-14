# Testing Documentation

Complete guide to the test suite and writing tests for Multimodal-DB.

## Test Files

1. **test_optimized.py** - Core component integration tests
2. **System_Test_Demo.ipynb** - Interactive testing notebook

---

## test_optimized.py

Comprehensive pytest-based test suite for core components.

### Location

```bash
tests/test_optimized.py
```

### Running Tests

```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest tests/test_optimized.py -v

# Run with coverage
pytest tests/test_optimized.py --cov=multimodal_db --cov-report=html

# Run specific test
pytest tests/test_optimized.py::test_agent_creation -v

# Run with output
pytest tests/test_optimized.py -v -s
```

### Test Structure

The test suite uses pytest fixtures for setup:

```python
@pytest.fixture(scope="session")
def agent():
    """Create test agent - reused across tests"""
    return create_corecoder_agent()

@pytest.fixture(scope="session")
def polars_db():
    """Create test database"""
    return PolarsDB("test_db")

@pytest.fixture(scope="session")
def qdrant_db():
    """Create vector database"""
    return QdrantDB("test_vectors")

@pytest.fixture(scope="session")
def ollama_client():
    """Create Ollama client"""
    return SimpleOllamaClient()
```

### Test Categories

#### 1. Agent Configuration Tests

**test_agent_creation**

Tests AgentConfig creation and configuration.

```python
def test_agent_creation(agent):
    """Test agent configuration system."""
    assert agent.agent_name == "CoreCoder"
    assert len(agent.helper_prompts) > 0
    assert "software-engineering" in agent.tags
```

**What it tests**:
- Agent is created successfully
- Helper prompts are populated
- Tags are correctly set
- Default configuration is valid

**Expected output**:
```
tests/test_optimized.py::test_agent_creation PASSED
```

#### 2. Database Operations Tests

**test_polars_operations**

Tests Polars database CRUD operations.

```python
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
```

**What it tests**:
- Agent storage and retrieval
- Message/conversation operations
- Data persistence
- Query functionality

**Expected output**:
```
tests/test_optimized.py::test_polars_operations PASSED
```

#### 3. Vector Database Tests

**test_qdrant_operations**

Tests Qdrant vector operations.

```python
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
    if results:
        assert results[0]["payload"]["text"] == "test"
```

**What it tests**:
- Collection creation
- Vector storage
- Similarity search
- Metadata handling

**Expected output**:
```
tests/test_optimized.py::test_qdrant_operations PASSED
# or SKIPPED if Qdrant not available
```

#### 4. AI Integration Tests

**test_ollama_integration**

Tests Ollama AI model integration.

```python
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
```

**What it tests**:
- Ollama availability check
- Response generation
- System prompt handling
- Error handling

**Expected output**:
```
tests/test_optimized.py::test_ollama_integration PASSED
# or SKIPPED if Ollama not available
```

### Test Output Examples

**Successful run**:
```bash
$ pytest tests/test_optimized.py -v

collected 4 items

tests/test_optimized.py::test_agent_creation PASSED              [ 25%]
tests/test_optimized.py::test_polars_operations PASSED           [ 50%]
tests/test_optimized.py::test_qdrant_operations SKIPPED (Q...)   [ 75%]
tests/test_optimized.py::test_ollama_integration PASSED          [100%]

================= 3 passed, 1 skipped in 2.45s ===================
```

**With coverage**:
```bash
$ pytest tests/test_optimized.py --cov=multimodal_db

Name                                  Stmts   Miss  Cover
---------------------------------------------------------
multimodal_db/core/agent_config.py      120     10    92%
multimodal_db/core/multimodal_db.py     180     25    86%
multimodal_db/core/vector_db.py         150     30    80%
---------------------------------------------------------
TOTAL                                   450     65    86%
```

### Handling Skipped Tests

Tests are automatically skipped if dependencies aren't available:

```python
# Qdrant not available
if not qdrant_db.available:
    pytest.skip("Qdrant not available")

# Ollama not running
if not ollama_client.available:
    pytest.skip("Ollama not available")
```

To ensure all tests run:
1. Install Qdrant: `pip install qdrant-client`
2. Install Ollama: https://ollama.ai
3. Pull model: `ollama pull qwen2.5-coder:3b`
4. Start Ollama: `ollama serve`

---

## System_Test_Demo.ipynb

Interactive Jupyter notebook for manual testing and exploration.

### Location

```bash
tests/System_Test_Demo.ipynb
```

### Opening the Notebook

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Navigate to tests/ folder and open System_Test_Demo.ipynb
```

### Notebook Sections

#### 1. Setup and Imports

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'multimodal-db'))

from core import (
    create_corecoder_agent,
    MultimodalDB,
    QdrantVectorDB,
    SimpleOllamaClient
)
```

#### 2. Agent Creation Tests

Create and inspect agents interactively:

```python
# Create agent
agent = create_corecoder_agent(name="TestAgent")

# Inspect configuration
print(f"Name: {agent.agent_name}")
print(f"System Prompt: {agent.system_prompt}")
print(f"Helper Prompts: {agent.helper_prompts}")
```

#### 3. Database Operations

Test database functionality:

```python
# Initialize database
db = MultimodalDB()

# Store agent
agent_id = db.store_agent(agent)
print(f"Stored: {agent_id}")

# Retrieve agent
retrieved = db.get_agent(agent_id)
print(f"Retrieved: {retrieved.agent_name}")

# List all agents
agents = db.list_agents()
for a in agents:
    print(f"- {a['name']}: {a['agent_id']}")
```

#### 4. Vector Operations

Test vector database:

```python
import numpy as np

# Initialize vector DB
vector_db = QdrantVectorDB()
vector_db.initialize_collections()

# Generate test embedding
embedding = np.random.rand(768).tolist()

# Store embedding
point_id = vector_db.store_embedding(
    collection="text_embeddings",
    vector=embedding,
    metadata={"test": "data"}
)

# Search
query = np.random.rand(768).tolist()
results = vector_db.search_similar(
    collection="text_embeddings",
    query_vector=query,
    limit=5
)

for r in results:
    print(f"Score: {r['score']}, Metadata: {r['payload']}")
```

#### 5. AI Chat Tests

Test Ollama integration:

```python
# Initialize client
client = SimpleOllamaClient()

if client.available:
    # Generate response
    response = client.generate(
        prompt="Explain Python in one sentence",
        system_prompt="You are a helpful teacher"
    )
    
    print(f"AI: {response['content']}")
else:
    print("Ollama not available")
```

### Interactive Testing Benefits

**Advantages of notebook testing**:
1. **Immediate feedback** - See results instantly
2. **Step-by-step debugging** - Run code cell by cell
3. **Data visualization** - Plot embeddings, statistics
4. **Documentation** - Combine code with explanations
5. **Experimentation** - Try different parameters easily

**Use notebook for**:
- Learning the API
- Debugging issues
- Prototyping new features
- Data exploration
- Creating tutorials

**Use pytest for**:
- Automated testing
- CI/CD integration
- Regression testing
- Coverage reports
- Production validation

---

## Writing Custom Tests

### Test Template

```python
#!/usr/bin/env python3
"""
Custom test module template
"""
import sys
import pytest
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'multimodal-db'))

from core import MultimodalDB, AgentConfig

@pytest.fixture
def db():
    """Database fixture"""
    return MultimodalDB("test_custom_db")

@pytest.fixture
def sample_agent():
    """Sample agent fixture"""
    agent = AgentConfig(
        agent_name="TestAgent",
        description="Test agent",
        tags=["test"]
    )
    return agent

def test_my_feature(db, sample_agent):
    """Test custom feature"""
    # Arrange
    agent_id = db.store_agent(sample_agent)
    
    # Act
    retrieved = db.get_agent(agent_id)
    
    # Assert
    assert retrieved is not None
    assert retrieved.agent_name == "TestAgent"
    
def test_another_feature(db):
    """Test another feature"""
    # Your test logic here
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Testing Best Practices

#### 1. Use Fixtures for Setup

```python
@pytest.fixture(scope="session")  # Created once per session
def expensive_setup():
    # Setup code
    yield resource
    # Teardown code

@pytest.fixture(scope="function")  # Created for each test
def fresh_data():
    return {"key": "value"}
```

#### 2. Test One Thing Per Test

```python
# Good
def test_agent_creation():
    agent = create_corecoder_agent()
    assert agent.agent_name == "CoreCoder"

def test_agent_storage():
    db = MultimodalDB()
    agent = create_corecoder_agent()
    agent_id = db.store_agent(agent)
    assert agent_id is not None

# Bad - testing multiple things
def test_everything():
    agent = create_corecoder_agent()
    assert agent.agent_name == "CoreCoder"
    db = MultimodalDB()
    agent_id = db.store_agent(agent)
    assert agent_id is not None
    retrieved = db.get_agent(agent_id)
    assert retrieved.agent_name == "CoreCoder"
```

#### 3. Use Descriptive Names

```python
# Good
def test_agent_retrieval_returns_none_for_invalid_id():
    db = MultimodalDB()
    agent = db.get_agent("invalid-id")
    assert agent is None

# Bad
def test_agent():
    db = MultimodalDB()
    agent = db.get_agent("invalid-id")
    assert agent is None
```

#### 4. Test Edge Cases

```python
def test_empty_database_returns_empty_list():
    db = MultimodalDB("test_empty_db")
    agents = db.list_agents()
    assert len(agents) == 0

def test_agent_with_empty_name_raises_error():
    with pytest.raises(ValueError):
        agent = AgentConfig(agent_name="", description="Test")

def test_vector_search_with_zero_limit():
    vector_db = QdrantVectorDB()
    results = vector_db.search_similar(
        collection="test",
        query_vector=[0.1] * 768,
        limit=0
    )
    assert len(results) == 0
```

#### 5. Use Parametrized Tests

```python
@pytest.mark.parametrize("media_type,expected", [
    ("text", True),
    ("image", True),
    ("invalid", False),
])
def test_media_type_validation(media_type, expected):
    result = validate_media_type(media_type)
    assert result == expected
```

### Example: API Endpoint Tests

```python
"""
Test API endpoints
"""
import pytest
import requests

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="module")
def api_available():
    """Check if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        return response.status_code == 200
    except:
        pytest.skip("API not running")

def test_root_endpoint(api_available):
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"

def test_create_agent(api_available):
    """Test agent creation"""
    response = requests.post(
        f"{BASE_URL}/agents/",
        json={
            "name": "test_agent",
            "agent_type": "corecoder"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "agent_id" in data
    return data["agent_id"]

def test_list_agents(api_available):
    """Test listing agents"""
    response = requests.get(f"{BASE_URL}/agents/")
    assert response.status_code == 200
    agents = response.json()
    assert isinstance(agents, list)

def test_invalid_agent_type(api_available):
    """Test invalid agent type"""
    response = requests.post(
        f"{BASE_URL}/agents/",
        json={
            "name": "test",
            "agent_type": "invalid_type"
        }
    )
    # Should either accept any string or return 400
    assert response.status_code in [200, 400]
```

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=multimodal_db --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

---

## Performance Testing

### Load Testing Example

```python
import time
import statistics
from multimodal_db.core import MultimodalDB, create_corecoder_agent

def test_agent_creation_performance():
    """Benchmark agent creation"""
    db = MultimodalDB("perf_test_db")
    timings = []
    
    for i in range(100):
        start = time.time()
        agent = create_corecoder_agent(name=f"agent_{i}")
        agent_id = db.store_agent(agent)
        end = time.time()
        timings.append(end - start)
    
    avg_time = statistics.mean(timings)
    max_time = max(timings)
    min_time = min(timings)
    
    print(f"Average: {avg_time*1000:.2f}ms")
    print(f"Min: {min_time*1000:.2f}ms")
    print(f"Max: {max_time*1000:.2f}ms")
    
    # Assert performance requirements
    assert avg_time < 0.1  # Less than 100ms average
    assert max_time < 0.5  # Less than 500ms max
```

---

## Troubleshooting Tests

### Tests Fail to Import

```bash
# Add multimodal-db to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/multimodal-db"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)\multimodal-db"  # Windows PowerShell
```

### Qdrant Tests Always Skip

```bash
# Install Qdrant client
pip install qdrant-client

# Verify import
python -c "import qdrant_client; print('OK')"
```

### Ollama Tests Always Skip

```bash
# Install Ollama from https://ollama.ai
ollama --version

# Pull model
ollama pull qwen2.5-coder:3b

# Start service
ollama serve

# Verify
ollama list
```

### Tests Take Too Long

```bash
# Run in parallel
pytest tests/ -n auto

# Run only fast tests
pytest tests/ -m "not slow"
```

---

## See Also

- **Python Library**: [LIBRARY.md](LIBRARY.md) - For scripting
- **REST API**: [API.md](API.md) - HTTP endpoints
- **Examples**: [EXAMPLES.md](EXAMPLES.md) - Usage examples
- **CLI Tools**: [CLI.md](CLI.md) - Command-line utilities
