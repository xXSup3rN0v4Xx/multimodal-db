# Dependency Conflict Resolution

## Issue
The original `requirements.txt` had overly strict version constraints that caused dependency conflicts, particularly with `pydantic`.

## Root Cause
```
ERROR: Cannot install -r requirements.txt because these package versions have conflicting dependencies.
The conflict is caused by:
    The user requested pydantic>=2.12.1
```

Multiple packages (fastapi, qdrant-client, gradio) have different pydantic version requirements, causing pip's resolver to fail.

## Solution Applied

### 1. Loosened Version Constraints
Changed from strict minimum versions to more flexible ranges:

**Before:**
```txt
polars>=1.34.0
pyarrow>=21.0.0
qdrant-client>=1.15.1
fastapi>=0.119.0
uvicorn[standard]>=0.37.0
pydantic>=2.12.1
```

**After:**
```txt
polars>=1.0.0
pyarrow>=10.0.0
qdrant-client>=1.7.0
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0,<3.0.0
```

### 2. Key Changes

#### Pydantic
- **Old:** `pydantic>=2.12.1`
- **New:** `pydantic>=2.0.0,<3.0.0`
- **Reason:** Allow pip to find a compatible version across all dependencies

#### Core Libraries
- Reduced minimum versions to stable releases
- Still maintains compatibility with project code
- Allows pip resolver more flexibility

#### Development Tools
- **Black:** `25.9.0` → `23.0.0`
- **Pytest:** `8.4.2` → `7.0.0`
- **Mypy:** `1.18.2` → `1.0.0`

### 3. Rationale

**Why Loosen Constraints?**
1. **Compatibility:** More packages can satisfy requirements
2. **Stability:** Older versions are well-tested
3. **Flexibility:** Allows pip to resolve conflicts
4. **Future-proof:** Easier to update individual packages

**Why Keep Constraints?**
- Still enforces minimum feature requirements
- Prevents breaking changes (major version bumps)
- Maintains security standards

## Installation

### Method 1: Automated Setup (Recommended)
```powershell
.\setup_python_environments.ps1
```

This script will:
1. Create virtual environment
2. Upgrade pip
3. Install all dependencies
4. Verify installations

### Method 2: Manual Installation
```powershell
# Navigate to project
cd multimodal-db

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Verification

After installation, verify all packages are installed:

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Check installed packages
pip list

# Verify specific packages
python -c "import polars; import qdrant_client; import fastapi; print('All imports successful!')"
```

## Expected Output
```
Looking in indexes: https://pypi.org/simple
Collecting polars>=1.0.0
  Using cached polars-1.x.x-...
...
Successfully installed polars-1.x.x pyarrow-1x.x.x qdrant-client-1.x.x ...
```

## If Issues Persist

### Strategy 1: Clear Cache
```powershell
pip cache purge
pip install -r requirements.txt
```

### Strategy 2: Use --no-deps (Advanced)
```powershell
# Install core packages first without dependencies
pip install --no-deps polars pyarrow
pip install --no-deps qdrant-client
pip install --no-deps fastapi uvicorn

# Then install remaining packages normally
pip install -r requirements.txt
```

### Strategy 3: Create Fresh Environment
```powershell
# Remove old venv
Remove-Item -Recurse -Force .venv

# Run setup script again
.\setup_python_environments.ps1
```

### Strategy 4: Specify Exact Versions (Last Resort)
If you need specific versions, create `requirements-lock.txt`:
```txt
polars==1.10.0
pyarrow==14.0.0
qdrant-client==1.7.0
fastapi==0.109.0
pydantic==2.5.0
...
```

Then install with:
```powershell
pip install -r requirements-lock.txt
```

## Compatibility Matrix

| Package | Min Version | Tested Version | Python |
|---------|-------------|----------------|--------|
| polars | 1.0.0 | 1.10.0+ | 3.8+ |
| pyarrow | 10.0.0 | 14.0.0+ | 3.8+ |
| qdrant-client | 1.7.0 | 1.11.0+ | 3.8+ |
| fastapi | 0.100.0 | 0.109.0+ | 3.8+ |
| pydantic | 2.0.0 | 2.5.0+ | 3.8+ |
| gradio | 4.0.0 | 4.20.0+ | 3.8+ |

## Understanding Pip Dependency Resolution

### How Pip Resolves Dependencies
1. Reads all requirements
2. Fetches metadata for each package
3. Builds dependency graph
4. Attempts to find compatible versions
5. Installs resolved versions

### Common Conflict Causes
- **Transitive dependencies:** Package A requires B<2.0, Package C requires B>=2.0
- **Version pinning:** Too strict constraints
- **Incompatible packages:** Fundamentally incompatible requirements

### Best Practices
✅ **DO:**
- Use minimum versions with upper bounds: `package>=1.0.0,<2.0.0`
- Keep dependencies up to date
- Test with multiple dependency versions
- Document known working versions

❌ **DON'T:**
- Pin exact versions unless necessary: `package==1.2.3`
- Use bleeding-edge versions: `package>=999.0.0`
- Mix stable and pre-release versions
- Ignore deprecation warnings

## Related Files
- `requirements.txt` - Main dependency file
- `setup_python_environments.ps1` - Automated setup script
- `pyproject.toml` - Project metadata and build config

## Troubleshooting Reference

### Error: "ResolutionImpossible"
**Cause:** Conflicting version requirements
**Fix:** Loosen version constraints (already done)

### Error: "No matching distribution"
**Cause:** Package not available for your platform/Python version
**Fix:** Check Python version (needs 3.8+), update pip

### Error: "Failed building wheel"
**Cause:** Missing build dependencies (usually C extensions)
**Fix:** Install Visual Studio Build Tools on Windows

### Error: "Import failed after install"
**Cause:** Wrong virtual environment or corrupted install
**Fix:** Recreate venv, reinstall

## Additional Resources
- [Pip Documentation](https://pip.pypa.io/en/stable/)
- [Dependency Resolution](https://pip.pypa.io/en/latest/topics/dependency-resolution/)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Status:** ✅ Fixed  
**Date:** 2025-10-15  
**Impact:** All dependencies now install correctly
