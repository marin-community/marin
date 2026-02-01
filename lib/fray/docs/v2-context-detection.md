# Fray v2 Context Auto-Detection

## Overview

This document describes the auto-detection mechanism for Fray v2 client contexts, eliminating the need for explicit `FRAY_CLIENT_SPEC` environment variables in most cases. The new approach detects whether code is running inside Iris or Ray environments and automatically configures the appropriate client.

## Motivation

### Current Problems

1. **Duplication**: `FRAY_CLIENT_SPEC=iris://host:port` duplicates information already present in `IRIS_CONTROLLER_ADDRESS=http://host:port`. The same controller address is stored twice with different URL schemes.

2. **Fragile round-trip parsing**: Iris strips `http://` to build `iris://`, then Fray strips `iris://` and re-prepends `http://`. This lossy round-trip breaks on `https://` URLs, non-standard port formats, and other edge cases.

3. **Doesn't compose with existing context APIs**: Iris already has `create_context_from_env()` which reads `IRIS_CONTROLLER_ADDRESS`, `IRIS_BUNDLE_GCS_PATH`, job info, etc. and produces a fully-configured `IrisContext`. The Fray `iris://` parsing path reconstructs a subset of this by hand.

4. **Maintenance burden**: If Iris adds authentication, changes address formats, or adds context fields, both `build_iris_env()` and `_parse_client_spec()` need coordinated updates. Auto-detection leverages existing context initialization for free.

5. **User confusion**: Users must explicitly set `FRAY_CLIENT_SPEC` even when running inside managed environments that already provide all necessary context through their own environment variables.

## Design

### Resolution Order

The new `current_client()` resolution order is:

1. **Explicitly set client** (via `set_current_client()` context manager)
   - Highest priority - allows programmatic overrides
   - Use case: Testing, custom client configurations

2. **FRAY_CLIENT_SPEC environment variable** (explicit override)
   - Second priority - allows manual override of auto-detection
   - Backward compatible with existing code that sets this variable
   - Use case: Override auto-detection in specific scenarios

3. **Auto-detect Iris environment** (`IRIS_CONTROLLER_ADDRESS` is set)
   - Third priority - auto-configure when running inside Iris jobs
   - Use case: Tasks launched by Iris cluster

4. **Auto-detect Ray environment** (`ray.is_initialized()` returns True)
   - Fourth priority - auto-configure when Ray is already initialized
   - Use case: Code running inside Ray workers or after `ray.init()`

5. **LocalClient() default**
   - Lowest priority - fallback for development and testing
   - Use case: Local development, scripts outside managed environments

### Detection Logic

#### Iris Detection

Iris environment is detected by checking for `IRIS_CONTROLLER_ADDRESS` environment variable:

```python
controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
if controller_address is not None:
    from fray.v2.iris_backend import FrayIrisClient
    bundle_gcs_path = os.environ.get("IRIS_BUNDLE_GCS_PATH")
    return FrayIrisClient(controller_address, bundle_gcs_path=bundle_gcs_path)
```

**Rationale**: `IRIS_CONTROLLER_ADDRESS` is always set by Iris when launching tasks. It contains the full controller URL (including `http://` or `https://` scheme), so no parsing or reconstruction is needed.

#### Ray Detection

Ray environment is detected by checking if Ray is already initialized:

```python
try:
    import ray
    if ray.is_initialized():
        from fray.v2.ray.backend import RayClient
        return RayClient()
except ImportError:
    pass  # Ray not installed, skip detection
```

**Rationale**:
- `ray.is_initialized()` is the canonical way to check if Ray is running
- Ray manages its own connection state through `ray.init()` or environment variables (`RAY_ADDRESS`)
- `RayClient()` defaults to `address="auto"`, which automatically connects to the current Ray cluster
- Import is wrapped in try/except since Ray is an optional dependency

### Backward Compatibility

The auto-detection design maintains full backward compatibility:

1. **Existing `FRAY_CLIENT_SPEC` usage continues to work**: The environment variable is checked before auto-detection, so existing code that sets it will see no behavior change.

2. **Explicit client setting still takes precedence**: Code using `set_current_client()` context manager continues to work exactly as before.

3. **LocalClient fallback unchanged**: Code that doesn't set any environment variables or run in managed environments defaults to `LocalClient()` as before.

4. **Migration is opt-in**: Removing `FRAY_CLIENT_SPEC` injection from Iris is a separate step that can be done after auto-detection is deployed.

## Implementation Plan

### Phase 1: Add Auto-Detection (Backward Compatible)

1. **Update `lib/fray/src/fray/v2/client.py`**:
   - Modify `current_client()` to add Iris and Ray auto-detection steps
   - Add detection between `FRAY_CLIENT_SPEC` check and `LocalClient()` fallback
   - Wrap Ray detection in try/except to handle missing Ray dependency

2. **Update documentation**:
   - Update `current_client()` docstring with new resolution order
   - Add this design document to `lib/fray/docs/`

3. **Test the new auto-detection**:
   - Verify Iris tasks can use `current_client()` without `FRAY_CLIENT_SPEC`
   - Verify Ray tasks can use `current_client()` without `FRAY_CLIENT_SPEC`
   - Verify backward compatibility (existing code still works)

### Phase 2: Remove FRAY_CLIENT_SPEC Injection

1. **Update `lib/iris/src/iris/cluster/worker/task_attempt.py`**:
   - Remove `FRAY_CLIENT_SPEC` construction and injection from `build_iris_env()`
   - Remove logging related to `FRAY_CLIENT_SPEC`
   - Keep all other environment variables unchanged

2. **Update tests**:
   - Remove tests that explicitly check for `FRAY_CLIENT_SPEC`
   - Verify integration tests still pass with auto-detection

### Phase 3: Cleanup (Optional)

1. **Simplify `_parse_client_spec()`**:
   - Consider removing `iris://` scheme support if no longer needed
   - Keep `local` and `ray` schemes for manual configuration
   - Document remaining supported schemes

2. **Update examples and tutorials**:
   - Remove `FRAY_CLIENT_SPEC` from example code
   - Show auto-detection as the default approach
   - Document manual override scenarios

## Migration Path

### For Iris Users

**Before**:
```python
# FRAY_CLIENT_SPEC=iris://host:port set by Iris
from fray.v2.client import current_client

client = current_client()  # Returns FrayIrisClient via FRAY_CLIENT_SPEC
```

**After**:
```python
# IRIS_CONTROLLER_ADDRESS=http://host:port set by Iris
from fray.v2.client import current_client

client = current_client()  # Returns FrayIrisClient via auto-detection
```

No code changes needed! Auto-detection happens transparently.

### For Ray Users

**Before**:
```python
import ray
ray.init(address="auto")

# Must set FRAY_CLIENT_SPEC=ray manually
os.environ["FRAY_CLIENT_SPEC"] = "ray"
from fray.v2.client import current_client

client = current_client()  # Returns RayClient via FRAY_CLIENT_SPEC
```

**After**:
```python
import ray
ray.init(address="auto")

from fray.v2.client import current_client

client = current_client()  # Returns RayClient via auto-detection
```

No need to set `FRAY_CLIENT_SPEC`! Auto-detection sees Ray is initialized.

### For Local Development

**Before and After** (unchanged):
```python
from fray.v2.client import current_client

client = current_client()  # Returns LocalClient() as fallback
```

### Override Examples

If you need to override auto-detection:

```python
# Override via environment variable
os.environ["FRAY_CLIENT_SPEC"] = "local?threads=16"
client = current_client()  # Returns LocalClient(max_threads=16)

# Override programmatically
from fray.v2.client import set_current_client
from fray.v2.local import LocalClient

with set_current_client(LocalClient(max_threads=4)):
    client = current_client()  # Returns the explicitly set client
```

## Testing Strategy

### Unit Tests

1. **Test Iris auto-detection**:
   - Set `IRIS_CONTROLLER_ADDRESS` environment variable
   - Verify `current_client()` returns `FrayIrisClient`
   - Verify correct controller address is used

2. **Test Ray auto-detection**:
   - Mock `ray.is_initialized()` to return True
   - Verify `current_client()` returns `RayClient`

3. **Test resolution order**:
   - Set explicit client via `set_current_client()`
   - Set `FRAY_CLIENT_SPEC` environment variable
   - Set `IRIS_CONTROLLER_ADDRESS` environment variable
   - Verify explicit client takes precedence over all others

4. **Test backward compatibility**:
   - Set `FRAY_CLIENT_SPEC=iris://host:port`
   - Verify behavior is unchanged from current implementation

### Integration Tests

1. **Iris integration test**:
   - Launch a real Iris task
   - Verify task can use `current_client()` without `FRAY_CLIENT_SPEC`
   - Verify sub-tasks inherit correct context

2. **Ray integration test**:
   - Initialize Ray cluster
   - Verify tasks can use `current_client()` without `FRAY_CLIENT_SPEC`

## Alternatives Considered

### Alternative 1: Remove FRAY_CLIENT_SPEC entirely

**Rejected**: Keeping `FRAY_CLIENT_SPEC` as an override mechanism is useful for testing and edge cases. Making it optional (via auto-detection) gives the best of both worlds.

### Alternative 2: Use dependency injection instead of global detection

**Rejected**: Would require changing all call sites to pass client explicitly. Auto-detection provides better ergonomics for the common case while still supporting explicit configuration via `set_current_client()`.

### Alternative 3: Environment-specific modules (e.g., `fray.v2.iris.current_client()`)

**Rejected**: Creates API fragmentation. A single `current_client()` that auto-detects is more discoverable and easier to use.

### Alternative 4: Require explicit client creation everywhere

**Rejected**: Adds boilerplate to every script. Auto-detection makes the simple case (running in a managed environment) simple while still supporting explicit configuration when needed.

## Security Considerations

Auto-detection reads environment variables to determine which client to create. This is the same security boundary as the current `FRAY_CLIENT_SPEC` approach - code that can set environment variables can control client behavior.

The resolution order ensures explicit configuration (via `set_current_client()` or `FRAY_CLIENT_SPEC`) takes precedence over auto-detection, so code can always override detected settings if needed.

## Performance Impact

Auto-detection adds minimal overhead:

1. **Iris detection**: Single `os.environ.get()` call
2. **Ray detection**: One `try/except` block and one `ray.is_initialized()` call
3. **No network calls**: All detection is local environment inspection

The overhead is negligible compared to client initialization and job submission costs.

## Future Work

### Workspace Detection

Iris auto-detection currently doesn't set the `workspace` parameter. Future work could:

1. Add `IRIS_WORKSPACE` environment variable to Iris task environments
2. Auto-detect workspace from `IRIS_BUNDLE_GCS_PATH` or current working directory
3. Document workspace override via `FRAY_CLIENT_SPEC` query parameters

### Ray Namespace Detection

Ray auto-detection uses default namespace behavior. Future work could:

1. Detect namespace from `ray.get_runtime_context()` if available
2. Add `RAY_NAMESPACE` environment variable support
3. Document namespace override via `FRAY_CLIENT_SPEC` query parameters

### Additional Environment Detection

As Fray supports more execution environments (e.g., Kubernetes, SLURM), add detection for those environments following the same pattern:

1. Check for environment-specific variables
2. Create appropriate client with environment-provided configuration
3. Insert in resolution order based on specificity

## Summary

Auto-detection eliminates the need for `FRAY_CLIENT_SPEC` in common cases while maintaining full backward compatibility. The design:

- ✅ Removes duplication between Iris environment variables and FRAY_CLIENT_SPEC
- ✅ Eliminates fragile URL scheme round-tripping
- ✅ Composes with existing Iris and Ray context APIs
- ✅ Reduces maintenance burden by leveraging existing environment initialization
- ✅ Improves user experience by removing boilerplate
- ✅ Maintains backward compatibility with existing code
- ✅ Supports explicit overrides when needed

Implementation is straightforward and can be done in phases, with each phase independently valuable and backward compatible.
