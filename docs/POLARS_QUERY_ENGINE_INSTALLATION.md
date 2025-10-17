# Polars Query Engine Installation Guide

## Overview

This guide documents how to install and configure the LlamaIndex Polars Query Engine. As of October 2025, the PyPI release of `llama-index-experimental` (v0.6.3) does **not** include the `PolarsQueryEngine` module, so installation directly from the GitHub repository is required.

## The Problem

The `PolarsQueryEngine` was recently added to the LlamaIndex repository but has not yet been included in the PyPI release:

```bash
# ❌ This will NOT work - PyPI package is outdated
pip install llama-index-experimental

# The module doesn't exist in the PyPI version:
# ImportError: cannot import name 'PolarsQueryEngine' from 'llama_index.experimental.query_engine'
```

## Correct Installation Steps

### Step 1: Install from GitHub Repository

Install `llama-index-experimental` directly from the GitHub main branch:

```bash
pip install --force-reinstall "git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-experimental"
```

**Important Notes:**
- Use `--force-reinstall` to ensure the GitHub version replaces any existing PyPI installation
- Do **NOT** use `--no-deps` flag as it will break dependency chain
- This installs from commit `0c43fc569efd597536334b79f925aeeddb667fd3` (or latest main)

### Step 2: Install Ollama Integration (Optional)

If you plan to use Ollama as your LLM:

```bash
pip install llama-index-llms-ollama
```

### Step 3: Verify Installation

Confirm the module is available:

```bash
python -c "from llama_index.experimental.query_engine import PolarsQueryEngine; print('✅ SUCCESS!')"
```

You should see: `✅ SUCCESS!`

### Step 4: Check Installation Location

Verify the module exists in your environment:

**Windows (PowerShell):**
```powershell
Test-Path "path\to\.venv\Lib\site-packages\llama_index\experimental\query_engine\polars"
```

**Linux/Mac:**
```bash
ls path/to/.venv/lib/python3.x/site-packages/llama_index/experimental/query_engine/polars
```

Should return `True` or show directory contents.

## Complete Installation Script

Here's a complete installation script for a fresh environment:

**Windows (PowerShell):**
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install polars

# Install LlamaIndex experimental from GitHub
pip install --force-reinstall "git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-experimental"

# Install Ollama integration (optional)
pip install llama-index-llms-ollama

# Verify installation
python -c "from llama_index.experimental.query_engine import PolarsQueryEngine; print('✅ Installation successful!')"
```

**Linux/Mac (Bash):**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install polars

# Install LlamaIndex experimental from GitHub
pip install --force-reinstall "git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-experimental"

# Install Ollama integration (optional)
pip install llama-index-llms-ollama

# Verify installation
python -c "from llama_index.experimental.query_engine import PolarsQueryEngine; print('✅ Installation successful!')"
```

## Basic Usage Example

Once installed, here's a minimal working example:

```python
import polars as pl
from llama_index.experimental.query_engine import PolarsQueryEngine
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# Configure LLM (using Ollama)
Settings.llm = Ollama(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    request_timeout=60.0
)

# Create a Polars DataFrame
df = pl.DataFrame({
    "city": ["Toronto", "Tokyo", "Berlin"],
    "population": [2930000, 13960000, 3645000]
})

# Create query engine
query_engine = PolarsQueryEngine(df=df, verbose=True)

# Query the data
response = query_engine.query("What is the city with the highest population?")
print(response)  # Output: Tokyo
```

## Correct Import Paths

### ✅ Correct Imports

```python
from llama_index.experimental.query_engine import PolarsQueryEngine
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama  # For Ollama
```

### ❌ Common Import Mistakes

```python
# Wrong - old/incorrect package name
from llama_index_llms_ollama import Ollama

# Wrong - module doesn't exist in PyPI version
from llama_index.experimental.query_engine.polars import PolarsQueryEngine
```

## Installed Package Details

After successful installation from GitHub, you should have:

- **Package**: `llama-index-experimental`
- **Version**: `0.6.3` (GitHub build, not PyPI)
- **Wheel size**: ~28KB
- **Module location**: `llama_index/experimental/query_engine/polars/`

## Dependencies

The GitHub installation will automatically install:

- `llama-index-core` (>=0.13.0, <0.15)
- `llama-index-finetuning` (>=0.4, <0.5)
- `pandas` (<2.3.0)
- `duckdb` (>=1.0.0, <2)
- Plus all transitive dependencies (aiohttp, pydantic, etc.)

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'llama_index_llms_ollama'"

**Solution**: Use correct import path:
```python
# Correct
from llama_index.llms.ollama import Ollama

# Wrong
from llama_index_llms_ollama import Ollama
```

### Issue: "ImportError: cannot import name 'PolarsQueryEngine'"

**Solution**: You have the PyPI version installed. Reinstall from GitHub:
```bash
pip install --force-reinstall "git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-experimental"
```

### Issue: Polars folder doesn't exist in site-packages

**Check**: Verify you didn't use `--no-deps` flag during installation. This breaks the dependency chain.

**Solution**: Reinstall without `--no-deps`:
```bash
pip install --force-reinstall "git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-experimental"
```

### Issue: Dependency conflicts with gradio/other packages

The GitHub installation may update some packages (like `pydantic` or `pillow`) which might conflict with other installed packages like `gradio`. 

**Solution**: Install in a separate virtual environment or accept the dependency warnings if they don't affect your use case.

## When Will PyPI Be Updated?

As of October 16, 2025, the `PolarsQueryEngine` exists in the GitHub repository at:
```
llama-index-experimental/llama_index/experimental/query_engine/polars/
```

But has not been released to PyPI. Monitor the [LlamaIndex releases page](https://github.com/run-llama/llama_index/releases) for updates.

Once a new version (>0.6.3) is released to PyPI with the Polars Query Engine included, you can switch back to:
```bash
pip install llama-index-experimental
```

## Additional Resources

- **GitHub Repository**: https://github.com/run-llama/llama_index
- **Polars Query Engine Source**: https://github.com/run-llama/llama_index/tree/main/llama-index-experimental/llama_index/experimental/query_engine/polars
- **LlamaIndex Documentation**: https://docs.llamaindex.ai/
- **Polars Documentation**: https://pola-rs.github.io/polars/

## Testing Your Installation

Run this test script to verify everything works:

```python
import polars as pl
from llama_index.experimental.query_engine import PolarsQueryEngine
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

print("🧪 Testing Polars Query Engine Installation...")

# Configure Ollama
Settings.llm = Ollama(model="qwen2.5-coder:3b", base_url="http://localhost:11434")

# Create test DataFrame
df = pl.DataFrame({
    "city": ["Toronto", "Tokyo", "Berlin"],
    "population": [2930000, 13960000, 3645000]
})

# Create query engine
query_engine = PolarsQueryEngine(df=df, verbose=True)

# Run test query
response = query_engine.query("What is the city with the highest population?")

print(f"\n✅ Test Result: {response}")
print("\n🎉 Installation successful! Polars Query Engine is working correctly.")
```

## Version Information

- **Document Created**: October 16, 2025
- **llama-index-experimental PyPI Version**: 0.6.3 (incomplete)
- **GitHub Commit**: 0c43fc569efd597536334b79f925aeeddb667fd3
- **Python Version Tested**: 3.11
- **Polars Version**: Compatible with polars >= 0.x

---

**Last Updated**: October 16, 2025  
**Status**: ⚠️ PyPI package incomplete - GitHub installation required
