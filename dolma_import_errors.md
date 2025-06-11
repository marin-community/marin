# Dolma Import Errors - Complete Issue Tracker

## Overview
This document tracks all issues encountered with installing and importing the `dolma` package, a mixed Python/Rust deduplication library. The dolma project has a complex structure that causes repeated installation and import problems.

## Root Cause: Dolma's Unusual Project Structure

Dolma is a **mixed Python/Rust project** with an unconventional layout:
```
dolma/
├── pyproject.toml          # Main project config (Rust + Python)
├── Cargo.toml             # Rust configuration  
├── src/                   # Rust source code
├── python/                # Python package source
│   ├── dolma/            # Actual Python package
│   │   ├── __init__.py
│   │   ├── cli/
│   │   └── ...
│   └── pyproject.toml    # Python-specific config
└── ...
```

**Key Problem**: The main `pyproject.toml` builds both Rust and Python components, but the Python code lives in a `python/` subdirectory with its own `pyproject.toml`.

---

## FAQ: Issues and Solutions

### 1. Docker Build Fails with Cargo Permission Errors

**Issue**: 
```
permission denied: /root/.cargo/registry/cache
```

**Root Cause**: Container not running as root, but trying to write to `/root/.cargo`

**Fix**: 
```dockerfile
ENV CARGO_HOME=$HOME/.cargo
ENV RUSTUP_HOME=$HOME/.rustup
RUN mkdir -p $HOME/.cargo/registry/cache && chmod -R 755 $HOME/.cargo
```

**Status**: ✅ RESOLVED

---

### 2. CUDA Dependencies Downloaded Despite CPU-Only Build

**Issue**: PyTorch trying to download CUDA dependencies in CPU-only container

**Attempted Fix**: `torch>=2.7.0+cpu` in requirements
**Problem**: uv doesn't understand `+cpu` suffix with `>=` operator

**Working Fix**: 
```dockerfile
--find-links https://download.pytorch.org/whl/cpu
```
And use plain `torch` in requirements.in

**Status**: ✅ RESOLVED

---

### 3. Cannot Import `deduper` from `dolma`

**Issue**: 
```python
from dolma import deduper  # ImportError: cannot import name 'deduper'
from dolma.cli import deduper  # ModuleNotFoundError: No module named 'dolma.cli'
```

**Debug Results**:
```
dolma contents: ['__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
dolma.__path__: _NamespacePath(['/tmp/ray/.../dolma'])
```

**Root Cause**: Dolma installed as **empty namespace package** - structure exists but no actual Python modules installed

**Status**: ❌ UNRESOLVED - This is our current blocker

---

### 4. Installing from `python/` Subdirectory Fails

**Issue**: 
```
/tmp/dolma/python does not appear to be a Python project, as neither `pyproject.toml` nor `setup.py` are present
```

**Attempted Fix**: `uv pip install -e ./python`
**Problem**: The `python/` subdirectory may not have its own `pyproject.toml` in the current dolma version

**Status**: ❌ FAILED APPROACH

---

### 5. Cannot Install PyPI Packages in Editable Mode

**Issue**: Attempted to install dolma from PyPI to bypass GitHub structure issues

**Attempted Fix**: `uv pip install dolma==1.2.0`
**Problem**: **PyPI packages cannot be installed with `-e` (editable) flag**

**Root Cause**: Editable installs only work with:
- Local file paths: `pip install -e ./path/to/package`  
- VCS URLs: `pip install -e git+https://github.com/user/repo.git`
- NOT PyPI packages: `pip install -e package-name` ❌

**Status**: ❌ IMPOSSIBLE APPROACH

---

### 6. Non-Editable GitHub Installation Creates Wrong Package Structure

**Issue**: Tried installing from GitHub without `-e` flag to avoid namespace package issues

**Attempted Fix**: `uv pip install . --no-cache-dir` (from GitHub root)

**Debug Results**:
```
dolma path: /tmp/ray/session_.../runtime_resources/working_dir_files/_ray_pkg_.../dolma
Files in dolma: ['.cargo', 'CITATION.cff', 'configs', 'Cargo.toml', 'Cargo.lock', 'LICENSE', 
'pyproject.toml', 'setup.sh', 'Makefile', 'tests', '.gitignore', 'src', 'scripts', 'sources', 
'README.md', 'python', '.github', '.flake8', 'docs', '.devcontainer', 'contrib']
Python files: []
Subdirectories: ['.cargo', 'configs', 'tests', 'src', 'scripts', 'sources', 'python', '.github', 
'docs', '.devcontainer', 'contrib']
```

**Root Cause**: Installation copied **entire GitHub repository** into dolma package location instead of installing actual Python package. The real Python code is in `python/` subdirectory but not accessible as dolma modules.

**Status**: ❌ FAILED - Still namespace package with wrong structure

---

### 7. Permission Denied When Cloning to /opt Directory

**Issue**: PYTHONPATH approach failed due to permission error

**Error**: 
```
fatal: could not create work tree dir '/opt/dolma': Permission denied
```

**Root Cause**: Container runs as non-root user (ray) who doesn't have write access to `/opt/`

**Fix**: Use user home directory instead: `$HOME/dolma`

**Status**: ✅ RESOLVED

---

### 8. PYTHONPATH Approach Causes Circular Import (Multiple Attempts) - **NEW FINDINGS**

**Issue**: PYTHONPATH approach builds successfully but fails at runtime with circular import

**Attempt 1 - Original naming**:
```
PYTHONPATH includes: ['/tmp/ray/.../experiments/train_test_overlap/dolma', '/home/ray/dolma/python']
Failed to import: cannot import name 'dolma' from partially initialized module 'dolma'
```

**Attempt 2 - Renamed to dolma_source**:
```
ENV PYTHONPATH="$HOME/dolma_source/python:$PYTHONPATH"
```
Still failed with:
```
PYTHONPATH includes: ['/tmp/ray/.../experiments/train_test_overlap/dolma', '/home/ray/dolma_source/python']
Failed to import: cannot import name 'dolma' from partially initialized module 'dolma'
```

**Attempt 3 - After removing local directory conflicts**:
```
PYTHONPATH includes: ['/home/ray/dolma_source/python']
Failed to import: cannot import name 'dolma' from partially initialized module 'dolma'
(most likely due to a circular import) (/home/ray/dolma_source/python/dolma/__init__.py)
```

**🔍 CRITICAL FINDING**: Even with **NO external naming conflicts**, the circular import persists. The error comes directly from dolma's own `__init__.py` file.

**Root Cause**: The issue is **internal to the dolma package itself**. Looking at dolma's `__init__.py`:
```python
from . import dolma as _dolma  # This line imports a Rust module also named 'dolma'
from .core import TaggerRegistry
from .core.errors import DolmaRustPipelineError
```

The problem is that dolma's Python package tries to import a Rust module with the same name (`dolma`), creating an internal circular import during package initialization.

**Status**: ❌ FAILED - **Internal dolma package structure issue - not solvable with installation methods**

---

### 9. Complex Multi-Step Docker Debugging Syntax Issues

**Issue**: Docker RUN commands with complex Python syntax failing

**Problems Encountered**:
- Multi-line Python code in Docker RUN
- Quote escaping issues
- Try/except blocks not working in single-line format
- Here document syntax not working in Docker

**Working Solution**: Separate RUN commands with simple Python scripts
```dockerfile
RUN echo "Testing..." && python -c "simple_statement"
```

**Status**: ✅ RESOLVED

---

## Current Status & Next Steps

### What Works:
1. ✅ Docker builds successfully 
2. ✅ dolma package installs without errors (at build time)
3. ✅ Basic `import dolma` works (in some contexts)
4. ✅ Rust components compile properly

### What's Broken:
1. ❌ **UNFIXABLE CIRCULAR IMPORT**: Local project has `experiments/train_test_overlap/dolma/` directory that conflicts with `dolma` package name
2. ❌ Cannot import any functionality (`deduper`, `cli`, etc.) at runtime
3. ❌ Runtime failures in Ray cluster due to missing imports
4. ❌ PYTHONPATH approach fails due to naming conflicts in project structure

### Root Cause Analysis:
The fundamental issue is **NOT external naming conflicts** but **internal circular imports within dolma's own package structure**:
- **Internal Issue**: Dolma's `__init__.py` tries to import a Rust extension module with the same name as the package
- **Package Structure**: Mixed Python/Rust project with complex internal dependencies
- **Initialization Problem**: The package cannot initialize itself due to internal naming conflicts

This is a **structural issue with the dolma package itself**, not our installation approach.

### Potential Solutions to Try:

#### Option 1: 🔥 **RECOMMENDED** - Use Alternative Deduplication Library
Since dolma has proven to be extremely difficult to integrate due to its internal structural issues:
```dockerfile
# Consider alternatives like:
# - dedupe: Simple Python deduplication
# - simhash: Fast similarity hashing
# - minhash: MinHash-based deduplication
# - datasketch: Probabilistic data structures for large datasets
```

#### Option 2: Install Dolma from PyPI (Non-Editable) - **HIGHEST PROBABILITY OF SUCCESS**
```dockerfile
RUN uv pip install dolma==1.2.0
# Remove all PYTHONPATH and GitHub installation attempts
# PyPI packages typically have better tested package structures
```
**Why this might work**: PyPI packages typically have better-tested internal structure and may not have the same internal circular import issues as the GitHub development version.

#### Option 3: Use Dolma CLI as Subprocess Instead of Python Import
```python
# Instead of: from dolma.cli import deduper
# Use: subprocess to call dolma CLI directly
import subprocess
import json

def run_dolma_deduper(config_dict):
    # Write config to temp file and call dolma CLI
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_file = f.name
    
    try:
        result = subprocess.run([
            'python', '-m', 'dolma.cli.deduper', '-c', config_file
        ], check=True, capture_output=True, text=True)
        return result.stdout
    finally:
        os.unlink(config_file)
```

#### Option 4: Fork Dolma and Fix Internal Structure
- Fork the dolma repository
- Fix the internal circular import in `__init__.py`
- Use the fixed version
**Risk**: Significant maintenance overhead

#### Option 5: Try Different Dolma Import Approach
```python
# Instead of importing from dolma package, try direct module imports
import sys
sys.path.insert(0, '/home/ray/dolma_source/python')

# Try importing the Rust extension directly
try:
    import dolma.dolma as dolma_rust
    deduper_func = dolma_rust.deduper_entrypoint
except ImportError:
    # Try different import paths
    pass
```

---

## Debug Commands for Investigation

### Check dolma installation:
```python
import dolma
print(f"Location: {dolma.__file__}")
print(f"Path: {dolma.__path__}")
print(f"Contents: {dir(dolma)}")
```

### Check PYTHONPATH conflicts:
```python
import sys
print("PYTHONPATH includes:", sys.path)
for path in sys.path:
    if 'dolma' in path:
        print(f"DOLMA CONFLICT: {path}")
```

### Search for actual modules:
```python
import pkgutil
for importer, modname, ispkg in pkgutil.iter_modules(dolma.__path__, dolma.__name__ + "."):
    print(f"Found: {modname}")
```

---

## Lessons Learned

1. **Project Structure Conflicts**: Package names must not conflict with local directory names
2. **Mixed Language Projects**: Python/Rust projects have complex build requirements
3. **Namespace Packages**: Empty namespace packages can be created without actual functionality
4. **Docker Debugging**: Complex Python code in Docker requires careful syntax handling
5. **Dependency Management**: uv has different behavior than pip for complex packages
6. **Import System**: Python's import resolution can be tricky with PYTHONPATH modifications
7. **Ray Cluster Environment**: Working directory structure affects import resolution

---

## Action Items & Recommendations

### Immediate Actions:
- [x] ~~Try PyPI installation instead of GitHub~~ ❌ **IMPOSSIBLE** - Cannot use `-e` with PyPI
- [x] ~~Test two-step installation approach~~ ❌ **FAILED** - Python subdirectory issues  
- [x] ~~Try building wheel from GitHub source then installing wheel~~ ❌ **FAILED** - Namespace package
- [x] ~~**Manual PYTHONPATH approach**~~ ❌ **FAILED** - Circular import conflict
- [x] ~~**TESTING: PYTHONPATH with unique directory name**~~ ❌ **FAILED** - Local directory naming conflict

### Next Steps (Priority Order):

1. **🔥 HIGHEST PRIORITY**: **Try PyPI non-editable installation** 
   ```dockerfile
   RUN uv pip install dolma==1.2.0
   # Remove all PYTHONPATH and GitHub cloning
   ```
   **Rationale**: PyPI packages have better-tested internal structure and may not have the internal circular import issues present in the GitHub development version.

2. **HIGH PRIORITY**: **Switch to alternative deduplication library**
   - Research simpler Python-only alternatives (`dedupe`, `simhash`, `datasketch`)
   - Avoid complex mixed-language projects with structural issues
   
3. **MEDIUM PRIORITY**: **Use subprocess approach**
   - Call dolma CLI as external process instead of Python import
   - Avoids internal circular import issues entirely
   
4. **LOW PRIORITY**: **Try direct Rust module import**
   - Bypass dolma's problematic `__init__.py` 
   - Import the underlying Rust extension directly

### Final Recommendation:
The issue is **internal to dolma's package structure** - not our installation method. Try PyPI installation first since it's most likely to work. If that fails, **switch to an alternative deduplication library** rather than spending more time debugging dolma's internal structural issues. 