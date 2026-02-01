# Design: iris_run.py - Iris Job Submission CLI

## Overview

This document compares two implementation approaches for adding CLI-based Iris job submission similar to `ray_run.py`. The goal is to provide a convenient way to submit jobs to Iris clusters with automatic SSH tunneling, workspace bundling, and log streaming.

## Background

Issue #2566 requests a way to submit Iris jobs with a command like:

```bash
scripts/iris_run.py --config=.../eu-west4.yaml submit <normal args for ray run>
```

This would:
- Establish SSH tunnel to the Iris controller
- Bundle the current workspace
- Submit the job with appropriate environment variables and resources
- Stream logs back to the terminal

## Comparison: Two Approaches

### Approach 1: Add `--iris` flag to ray_run.py

Modify the existing `ray_run.py` to accept an `--iris` flag that switches from Ray to Iris backend.

```bash
uv run lib/marin/src/marin/run/ray_run.py --iris --cluster eu-west4 -- python my_script.py
```

#### Implementation Sketch

```python
# In ray_run.py
parser.add_argument("--iris", action="store_true", help="Use Iris instead of Ray")

async def submit_and_track_job(..., use_iris: bool = False):
    if use_iris:
        # Iris path: bundle workspace, use IrisClient
        from iris.client import IrisClient
        from iris.cluster.client import BundleCreator

        bundle = BundleCreator(workspace).create_bundle()
        client = IrisClient.remote(controller_url, bundle_blob=bundle)
        # ... iris-specific logic
    else:
        # Ray path: existing logic
        client = make_client()
        runtime_dict = build_runtime_env_for_packages(...)
        # ... ray-specific logic
```

#### Pros
- Single entry point for users (`ray_run.py` works for both backends)
- Can share environment variable handling (`.marin.yaml`, `-e` flags)
- Can share submission ID generation logic
- Users don't need to learn a new command

#### Cons
- **Fundamental incompatibility**: Ray and Iris have completely different abstractions
  - Ray: `runtime_env` with pip dependencies, working_dir as string, entrypoint as shell command
  - Iris: `BundleCreator` with zip files, workspace as Path, entrypoint as callable or command
- **Resource model mismatch**:
  - Ray: `entrypoint_num_cpus`, `entrypoint_num_gpus`, `entrypoint_resources` dict
  - Iris: `ResourceSpec(cpu, memory, device=tpu_device(type))`
- **Heavy branching**: Would need `if use_iris: ... else: ...` throughout the code
- **Code size**: `ray_run.py` is already 415 lines; adding Iris would push it to 600+
- **Maintenance burden**: Changes to either backend affect the other
- **Testing complexity**: Need to test both paths with overlapping and unique flags
- **Import coupling**: Forces Ray dependencies in Iris-only environments

### Approach 2: Create separate `iris_run.py`

Create a new `lib/iris/scripts/iris_run.py` dedicated to Iris job submission.

```bash
uv run lib/iris/scripts/iris_run.py --config lib/iris/examples/eu-west4.yaml -- python my_script.py
```

#### Implementation Sketch

```python
# lib/iris/scripts/iris_run.py
import argparse
from pathlib import Path
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, tpu_device
from iris.cluster.vm.debug import controller_tunnel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Iris cluster config YAML")
    parser.add_argument("--env_vars", "-e", action="append", nargs="+")
    parser.add_argument("--tpu", type=str, help="TPU type (e.g., v5litepod-16)")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Parse cluster config to get zone/project
    config = load_config(args.config)

    # Parse environment variables (can share logic with ray_run.py via utility)
    env_vars = parse_env_vars(args.env_vars)

    # Build resources
    resources = ResourceSpec(cpu=1, memory="2GB")
    if args.tpu:
        resources.device = tpu_device(args.tpu)

    # Establish tunnel and submit
    with controller_tunnel(config.zone, config.project_id) as controller_url:
        client = IrisClient.remote(controller_url, workspace=Path.cwd())

        # Build entrypoint (support both callables and commands)
        entrypoint = Entrypoint.from_command(args.cmd)

        job = client.submit(
            entrypoint=entrypoint,
            name=generate_job_name(args.cmd),
            resources=resources,
            environment=EnvironmentSpec(env=env_vars),
        )

        if not args.no_wait:
            job.wait(stream_logs=True)
```

#### Pros
- **Clean separation**: Iris-specific code stays in `lib/iris/`
- **Optimized for Iris**: Can use native Iris idioms (bundles, ResourceSpec, etc.)
- **Easier maintenance**: Changes to Iris submission don't affect Ray
- **Simpler code**: No conditional logic, single responsibility
- **Independent evolution**: Can add Iris-specific features (ports, coscheduling, constraints) without Ray concerns
- **Clear ownership**: Lives in `lib/iris/scripts/` alongside other Iris tools

#### Cons
- **Two commands**: Users need to know `ray_run.py` vs `iris_run.py`
- **Some duplication**: Environment variable parsing, `.marin.yaml` loading might be duplicated
  - Mitigation: Extract shared logic to `lib/marin/src/marin/run/common.py`

## Recommendation: Approach 2 (Separate iris_run.py)

Create a dedicated `lib/iris/scripts/iris_run.py`.

### Rationale

1. **Backend Incompatibility**: Ray and Iris have fundamentally different resource models, workspace handling, and job lifecycle management. Trying to unify them creates more complexity than value.

2. **Code Clarity**: A focused 200-line `iris_run.py` is easier to understand and maintain than a 600-line branched `ray_run.py`.

3. **User Experience**: While two commands seems like a con, it's actually clearer: users explicitly choose their backend rather than remembering to add `--iris`.

4. **Maintenance**: Iris and Ray evolve independently. Coupling them slows down both.

5. **Precedent**: We already have separate `lib/iris/scripts/cluster-tools.py` and `lib/iris/scripts/smoke-test.py`.

## API Surface for iris_run.py

### Required Arguments
- `--config PATH`: Path to Iris cluster YAML (e.g., `lib/iris/examples/eu-west4.yaml`)
- `cmd`: Command to run (e.g., `-- python my_script.py --arg value`)

### Optional Arguments
- `--env_vars KEY [VALUE]`, `-e KEY [VALUE]`: Set environment variables (repeatable)
- `--tpu TYPE`: TPU type to request (e.g., `v5litepod-16`)
- `--cpu N`: Number of CPUs (default: 1)
- `--memory SIZE`: Memory size (e.g., `2GB`, `512MB`)
- `--no-wait`: Don't wait for job completion
- `--job-name NAME`: Custom job name (default: auto-generated)
- `--replicas N`: Number of tasks for gang scheduling (default: 1)
- `--max-retries N`: Max retries on failure (default: 0)
- `--timeout SECONDS`: Job timeout (default: 0 = no timeout)

### Future Arguments (not in initial implementation)
- `--ports NAME [NAME...]`: Allocate named ports
- `--constraints KEY=VALUE`: Scheduling constraints
- `--coscheduling-timeout SECONDS`: Atomic multi-task scheduling timeout

### Examples

```bash
# Simple CPU job
uv run lib/iris/scripts/iris_run.py \
  --config lib/iris/examples/eu-west4.yaml \
  -- python experiments/train.py

# TPU job with environment variables
uv run lib/iris/scripts/iris_run.py \
  --config lib/iris/examples/eu-west4.yaml \
  --tpu v5litepod-16 \
  -e WANDB_API_KEY $WANDB_API_KEY \
  -e HF_TOKEN $HF_TOKEN \
  -- python experiments/pretrain.py --model grug-xl

# Submit and detach
uv run lib/iris/scripts/iris_run.py \
  --config lib/iris/examples/eu-west4.yaml \
  --tpu v5litepod-16 \
  --no-wait \
  -- python experiments/long_job.py
```

## Workspace and Bundle Handling

### Current Behavior (Iris)

`IrisClient.remote(controller_url, workspace=Path("."))` automatically:
1. Creates a `BundleCreator` with the workspace path
2. Bundles files using `git ls-files` (or pattern-based exclusion if not in git repo)
3. Sends the bundle blob to the controller
4. Controller stores it in GCS (configured via `bundle_prefix` in cluster YAML)
5. Workers download the bundle and extract to job container

### iris_run.py Behavior

- Always bundle the current working directory (`Path.cwd()`)
- Use `BundleCreator` as-is (no changes needed)
- Pass `workspace=Path.cwd()` to `IrisClient.remote()`

### Comparison to ray_run.py

Ray uses `runtime_env`:
```python
runtime_dict = {
    "working_dir": current_dir,  # String path
    "config": {"setup_timeout_seconds": 1800},
    "excludes": [".git", "docs/", "**/*.pack"],
}
runtime_dict = build_runtime_env_for_packages(extra=["tpu"], env_vars=env_vars) | runtime_dict
```

Iris uses bundles:
```python
client = IrisClient.remote(controller_url, workspace=Path.cwd())
# BundleCreator handles exclusions via git or patterns
```

Key difference: Ray ships dependencies via pip at runtime; Iris bundles the entire workspace as a zip file. Both approaches work, but Iris is simpler for development workflows.

## Environment Variable Handling

### Shared Logic (to extract)

Both `ray_run.py` and `iris_run.py` need:
1. Load defaults from `.marin.yaml` if present
2. Auto-include `HF_TOKEN` and `WANDB_API_KEY` from environment
3. Parse `-e KEY VALUE` flags and override defaults
4. Validate keys don't contain `=`

Extract to `lib/marin/src/marin/run/env_utils.py`:

```python
def load_env_vars(env_flags: list[list[str]] | None) -> dict[str, str]:
    """Load environment variables from .marin.yaml and merge with flags.

    Args:
        env_flags: List of [KEY, VALUE] or [KEY] pairs from argparse

    Returns:
        Merged environment variables

    Raises:
        ValueError: If key contains '=' or other validation fails
    """
    # 1. Load from .marin.yaml
    env_vars = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        with open(marin_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("env"), dict):
            for k, v in cfg["env"].items():
                env_vars[str(k)] = "" if v is None else str(v)

    # 2. Auto-include tokens from environment
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if key not in env_vars and os.environ.get(key):
            env_vars[key] = os.environ[key]

    # 3. Merge flags
    if env_flags:
        for item in env_flags:
            if len(item) > 2:
                raise ValueError(f"Too many values for env var: {' '.join(item)}")
            if "=" in item[0]:
                raise ValueError(f"Key cannot contain '=': {item[0]}")
            env_vars[item[0]] = item[1] if len(item) == 2 else ""

    return env_vars
```

Both scripts can import and use this function.

## Config File Parsing

Iris cluster configs are YAML files with structure:
```yaml
project_id: hai-gcp-models
region: europe-west4
zone: europe-west4-b

controller_vm:
  gcp:
    image: ...
    machine_type: n2-standard-4

scale_groups:
  tpu_v5e_16:
    provider:
      tpu:
        project_id: hai-gcp-models
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    ...
```

We need to parse:
- `project_id`: For `controller_tunnel(zone, project)`
- `zone`: For `controller_tunnel(zone, project)`

Create helper in `lib/iris/src/iris/cluster/config.py`:

```python
@dataclass
class ClusterConfig:
    """Parsed Iris cluster configuration."""
    project_id: str
    zone: str
    region: str
    # Add other fields as needed

def load_cluster_config(path: Path) -> ClusterConfig:
    """Load and parse Iris cluster config from YAML.

    Args:
        path: Path to cluster YAML file

    Returns:
        Parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return ClusterConfig(
        project_id=data["project_id"],
        zone=data["zone"],
        region=data["region"],
    )
```

## Entrypoint Handling

### Current State

`Entrypoint.from_callable(fn, *args, **kwargs)` exists for Python callables.

For CLI commands, we need to support string commands like `"python my_script.py --arg value"`.

### Two Options

#### Option A: Add `Entrypoint.from_command(cmd: str)` or `Entrypoint.from_command(cmd: list[str])`

```python
# In iris.cluster.types
class Entrypoint:
    @classmethod
    def from_command(cls, cmd: str | list[str]) -> "Entrypoint":
        """Create entrypoint from shell command.

        Args:
            cmd: Command string or list of arguments

        Returns:
            Entrypoint that runs the command in a shell
        """
        if isinstance(cmd, list):
            cmd = " ".join(shlex.quote(arg) for arg in cmd)

        # Use a wrapper function that calls subprocess
        def _run_command():
            import subprocess
            result = subprocess.run(cmd, shell=True, check=False)
            return result.returncode

        return cls.from_callable(_run_command)
```

#### Option B: Use `Entrypoint.from_callable()` with a wrapper

```python
# In iris_run.py
def make_command_entrypoint(cmd: list[str]) -> Entrypoint:
    """Wrap a shell command as an Iris entrypoint."""
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    def run_command():
        import subprocess
        import sys
        result = subprocess.run(cmd_str, shell=True, check=False)
        sys.exit(result.returncode)

    return Entrypoint.from_callable(run_command)
```

**Recommendation**: Option B for initial implementation (no changes to core Iris). Can extract to a helper function or add Option A later if useful for other scripts.

## SSH Tunnel Management

Use existing `controller_tunnel()` context manager from `iris.cluster.vm.debug`:

```python
from iris.cluster.vm.debug import controller_tunnel

config = load_cluster_config(args.config)

with controller_tunnel(config.zone, config.project_id) as controller_url:
    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    # ... submit job
```

This handles:
- Controller VM discovery via `gcloud compute instances list`
- SSH tunnel establishment with port forwarding
- Cleanup on exit

## Testing Strategy

### Unit Tests

`lib/iris/tests/test_iris_run.py`:

```python
def test_parse_args_basic():
    """Test basic argument parsing."""
    args = parse_args(["--config", "cluster.yaml", "--", "python", "script.py"])
    assert args.config == "cluster.yaml"
    assert args.cmd == ["--", "python", "script.py"]

def test_parse_args_with_env_vars():
    """Test environment variable parsing."""
    args = parse_args([
        "--config", "cluster.yaml",
        "-e", "KEY1", "value1",
        "-e", "KEY2",
        "--", "python", "script.py"
    ])
    # Verify env_vars parsed correctly

def test_load_env_vars_from_marin_yaml(tmp_path):
    """Test loading env vars from .marin.yaml."""
    marin_yaml = tmp_path / ".marin.yaml"
    marin_yaml.write_text("env:\n  FOO: bar\n")
    # ... test loading

def test_load_cluster_config(tmp_path):
    """Test cluster config parsing."""
    config_yaml = tmp_path / "cluster.yaml"
    config_yaml.write_text("""
    project_id: test-project
    zone: us-central1-a
    region: us-central1
    """)
    config = load_cluster_config(config_yaml)
    assert config.project_id == "test-project"
    assert config.zone == "us-central1-a"
```

### Integration Tests

`lib/iris/tests/test_iris_run_integration.py`:

```python
def test_iris_run_local_cluster():
    """Test iris_run.py with local cluster."""
    # Start local cluster
    with IrisClient.local() as client:
        # Mock controller_tunnel to return local URL
        # Run iris_run.py subprocess with test script
        # Verify job completes successfully
        pass

def test_iris_run_env_vars():
    """Test environment variables passed to job."""
    # Submit job that prints env vars
    # Verify they match what was passed to iris_run.py
    pass
```

### Manual Validation (eu-west4)

Per issue requirements, validate by submitting actual tasks to eu-west4 cluster:

```bash
# 1. Start cluster
uv run iris cluster --config lib/iris/examples/eu-west4.yaml start

# 2. Submit test job
uv run lib/iris/scripts/iris_run.py \
  --config lib/iris/examples/eu-west4.yaml \
  --tpu v5litepod-16 \
  -- python -c "import jax; print(jax.devices())"

# 3. Verify:
# - SSH tunnel established
# - Job submitted
# - Logs streamed
# - Job completed successfully

# 4. Cleanup
uv run iris cluster --config lib/iris/examples/eu-west4.yaml stop
```

## Implementation Plan

### Phase 1: Core Functionality
1. Create `lib/iris/scripts/iris_run.py` with basic argument parsing
2. Extract `load_env_vars()` to `lib/marin/src/marin/run/env_utils.py`
3. Add `load_cluster_config()` to `lib/iris/src/iris/cluster/config.py`
4. Implement `make_command_entrypoint()` helper
5. Wire together: config parsing → tunnel → client → submit → wait

### Phase 2: Testing
1. Add unit tests for argument parsing, config loading, env var merging
2. Add integration test with local cluster
3. Manual validation against eu-west4 cluster

### Phase 3: Polish
1. Add `--help` documentation
2. Add progress messages (tunnel established, job submitted, etc.)
3. Handle errors gracefully (tunnel failure, job submission errors)
4. Add to CI (run unit tests, skip integration tests requiring GCP)

### Phase 4: Documentation
1. Update `lib/iris/README.md` with iris_run.py usage
2. Add examples to Iris docs
3. Update issue #2566 with resolution

## Open Questions

1. **Should we support `--auto` TPU detection like ray_run.py?**
   - ray_run.py can auto-detect TPU type from cluster config
   - For Iris, users would need to know the scale group name
   - Decision: Add in Phase 3 if useful

2. **Should we add a `--scale-group` flag?**
   - Users could specify `--scale-group tpu_v5e_16` instead of `--tpu v5litepod-16`
   - Might be more natural for Iris users
   - Decision: Add as alias in Phase 3

3. **Should we support `--cluster REGION` shorthand like ray_run.py?**
   - ray_run.py has `find_config_by_region()` to map regions to config files
   - For Iris, we could add similar logic
   - Decision: Not needed for MVP; can add later

## Summary

**Recommendation**: Create separate `lib/iris/scripts/iris_run.py` (Approach 2).

This provides:
- Clean separation between Ray and Iris backends
- Iris-optimized implementation using native abstractions
- Easier maintenance and independent evolution
- Clear ownership within the Iris subproject

The main implementation steps are:
1. Create iris_run.py with argument parsing
2. Extract shared env var logic to common utility
3. Add config parsing helper
4. Wire together using existing Iris APIs (controller_tunnel, IrisClient, BundleCreator)
5. Add comprehensive tests
6. Validate against eu-west4 cluster

Estimated effort: ~300 lines of new code, ~200 lines of tests.
