# Local Haliax Installation Progress Tracking

## Problem Statement
- Need to install local haliax version (with debug prints) instead of PyPI version
- Runtime shows haliax loading from `/opt/levanter/.venv/lib/python3.10/site-packages/haliax/partitioning.py` (PyPI version)
- Should load from `/opt/levanter/haliax/src/haliax/partitioning.py` (local version with debug prints)

## Debugging Evidence
**WandB Debug Info Confirmed Issue:**
```
Haliax partitioning loaded from: /opt/levanter/.venv/lib/python3.10/site-packages/haliax/partitioning.py
```
Should be: `/opt/levanter/haliax/src/haliax/partitioning.py`

## Docker Installation Attempts

### Attempt 1: Original - Used conflicting flags
```bash
pip install -e /opt/levanter/haliax --target /opt/levanter/.venv/lib/python3.10/site-packages --upgrade
```
**Result**: FAILED - `--target` conflicts with `-e` (editable installs)

### Attempt 2: Removed target flag, used venv pip
```bash
/opt/levanter/.venv/bin/pip install -e /opt/levanter/haliax
```
**Result**: FAILED - `/opt/levanter/.venv/bin/pip` doesn't exist in container

### Current State
- ✅ Local haliax directory exists: `/opt/levanter/haliax/`
- ✅ Debug prints confirmed in local version (grep shows all debug lines)
- ❌ System still loads PyPI version from `.venv/lib/python3.10/site-packages/`
- ❌ venv pip binary missing: `/opt/levanter/.venv/bin/pip: not found`

## Root Cause Analysis
The levanter base image uses a different Python environment setup where:
1. Python path includes `/opt/levanter/.venv/lib/python3.10/site-packages/`
2. But venv pip binary doesn't exist at `/opt/levanter/.venv/bin/pip`
3. System pip exists at `/usr/local/bin/pip` but may not install to venv location

## Next Solutions to Try

### Option 1: Force install to correct location
```bash
pip install -e /opt/levanter/haliax --target /opt/levanter/.venv/lib/python3.10/site-packages --upgrade --force-reinstall
```

### Option 2: Manual file replacement
```bash
# Remove PyPI version
rm -rf /opt/levanter/.venv/lib/python3.10/site-packages/haliax*
# Create symlink to local version
ln -s /opt/levanter/haliax/src/haliax /opt/levanter/.venv/lib/python3.10/site-packages/haliax
```

### Option 3: PYTHONPATH override
```bash
# Prepend local haliax to PYTHONPATH so it's found first
ENV PYTHONPATH=/opt/levanter/haliax/src:$PYTHONPATH
```

### Option 4: Direct file copy
```bash
# Copy local haliax over PyPI version
cp -r /opt/levanter/haliax/src/haliax/* /opt/levanter/.venv/lib/python3.10/site-packages/haliax/
```

## Key Constraints
- ❌ NO haliax imports on local machine (no TPU available)
- ❌ NO redundant attempts of the same approach
- ✅ Local haliax has debug prints confirmed via grep
- ✅ Need to install to `/opt/levanter/.venv/lib/python3.10/site-packages/` location

## Current Docker Build Status
- Local haliax detected ✅
- Debug prints found in local version ✅
- Installation command failed ❌
- Runtime still uses PyPI version ❌

## ATTEMPT 3: PYTHONPATH Override (IMPLEMENTING NOW)

**Strategy**: Prepend local haliax to PYTHONPATH so Python finds it first
**Implementation**: Add `ENV PYTHONPATH=/opt/levanter/haliax/src:$PYTHONPATH` to Dockerfile
**Expected Result**: Python will import from local haliax before checking venv site-packages

### Implementation Details
- Location: After local haliax directory is copied but before runtime
- Command: `ENV PYTHONPATH=/opt/levanter/haliax/src:$PYTHONPATH`
- Verification: WandB debug should show `/opt/levanter/haliax/src/haliax/partitioning.py`

### Implementation Status
✅ **COMPLETED**: Added `ENV PYTHONPATH=/opt/levanter/haliax/src:$PYTHONPATH` to Dockerfile
- Location: Line 72 in docker/tpu/Dockerfile.marin_incremental
- Placement: After `ADD . /opt/levanter` (line 69)
- Before: Haliax installation debug section

### Expected Behavior
- Python will check `/opt/levanter/haliax/src/` first when importing haliax
- Local haliax (with debug prints) should take precedence over PyPI version
- WandB debug info should show: `/opt/levanter/haliax/src/haliax/partitioning.py`
- Debug prints from partitioning.py should be active during runtime

### Next Steps
1. Rebuild Docker image with PYTHONPATH override
2. Run evaluation job
3. Check WandB debug artifact for haliax path confirmation
4. Verify debug prints appear in logs during model loading
5. Update tracking file with results

### ATTEMPT 3 RESULTS: SUCCESS! ✅

**Docker Build Output Analysis:**
- ✅ **Local haliax found**: `/opt/levanter/haliax/` directory detected
- ✅ **Uninstall successful**: `Successfully uninstalled haliax-1.4.dev400`
- ✅ **Editable install successful**: `Successfully installed haliax-1.4`
- ✅ **Editable location correct**: `Editable project location: /opt/levanter/haliax`
- ✅ **Debug prints confirmed**: All debug print lines found in local partitioning.py
- ✅ **PYTHONPATH override in place**: Should prioritize local haliax

**Key Success Indicators:**
```
#22 11.01 Location: /usr/local/lib/python3.10/site-packages
#22 11.01 Editable project location: /opt/levanter/haliax
```

**Critical Observation:**
The pip installation shows it's in `/usr/local/lib/python3.10/site-packages`, but with PYTHONPATH override `/opt/levanter/haliax/src:$PYTHONPATH`, Python should import from the local version first.

**Next Testing Phase:**
- 🔄 Run evaluation job to check WandB debug artifact
- 🔄 Verify import path shows `/opt/levanter/haliax/src/haliax/partitioning.py`
- 🔄 Check if debug prints appear during model loading
- 🔄 Monitor for 126GB allocation error with debug context

### ATTEMPT 3 FINAL RESULTS: COMPLETE SUCCESS! 🎉

**Runtime Verification - WandB Debug Artifact:**
```
Haliax partitioning loaded from: /opt/levanter/haliax/src/haliax/partitioning.py ✅
Current working directory: /opt/levanter
Python path: ['/opt/levanter/.venv/lib/python3.10/site-packages/ray/thirdparty_files', '/opt/levanter/src/levanter/main', '/opt/levanter/haliax/src']
```

**SUCCESS INDICATORS:**
- ✅ **Correct import path**: `/opt/levanter/haliax/src/haliax/partitioning.py` (local version)
- ✅ **PYTHONPATH working**: `/opt/levanter/haliax/src` appears in Python path
- ✅ **Debug prints visible**: Terminal now shows debug output from partitioning.py
- ✅ **Local modifications active**: Custom debug code is executing

---

## 🔑 KEY LESSONS LEARNED FOR FUTURE LOCAL REPO CO-DEVELOPMENT

### Problem: Docker + Virtual Environment + Editable Installs = Complex
**Root Issue**: Docker containers with existing virtual environments make it tricky to override pip-installed packages with local development versions.

### ❌ What DOESN'T Work:
1. **Conflicting pip flags**: `pip install -e /path/to/local --target /venv/site-packages` - `--target` conflicts with `-e`
2. **Missing venv pip**: `/opt/container/.venv/bin/pip` may not exist in pre-built containers
3. **Editable installs alone**: Even successful editable installs may not override existing packages in complex Python path setups

### ✅ WINNING SOLUTION: PYTHONPATH Override
**Implementation**: `ENV PYTHONPATH=/path/to/local/repo/src:$PYTHONPATH`
**Why it works**: Python checks PYTHONPATH locations BEFORE site-packages, ensuring local code takes precedence

### 🛠️ PROVEN DOCKERFILE PATTERN FOR LOCAL REPO OVERRIDE:

```dockerfile
# 1. Copy local repo to container
ADD . /opt/container

# 2. CRITICAL: Set PYTHONPATH to prioritize local repo
ENV PYTHONPATH=/opt/container/local_repo/src:$PYTHONPATH

# 3. OPTIONAL: Also do editable install (for completeness, but PYTHONPATH is key)
RUN pip install -e /opt/container/local_repo

# 4. Verification during build
RUN echo "Verifying local repo override:" && \
    ls -la /opt/container/local_repo/src/ && \
    grep -n "YOUR_DEBUG_PATTERN" /opt/container/local_repo/src/target_file.py
```

### 🔍 DEBUGGING VERIFICATION PATTERN:
1. **Add import path logging** to your application:
   ```python
   import your_module
   print(f"Module loaded from: {your_module.__file__}")
   ```
2. **Use WandB/logging** to capture and verify paths in cloud environments
3. **Add debug prints** to local modifications to confirm they're active

### 🚀 FUTURE APPLICATIONS:
- **Multiple local repos**: Extend PYTHONPATH with multiple local repo paths
- **Development workflows**: Same pattern works for any Python package you're co-developing
- **Testing environments**: Verify local changes in containerized environments

---

## STATUS: MISSION ACCOMPLISHED ✅
Local haliax installation successful. Debug prints active. Ready to debug the 126GB allocation issue with full visibility into partitioning operations.
