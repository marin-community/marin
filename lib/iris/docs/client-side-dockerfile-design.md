# Design: Client-Side Dockerfile Generation

**Issue:** [#2572](https://github.com/marin-community/marin/issues/2572) - Add support for custom docker images or configuration

## Problem

Workers currently own the entire Dockerfile template and image-building logic (`builder.py`). The client sends high-level hints (`pip_packages`, `extras`, `python_version`) and the worker interprets them into a Dockerfile. This makes it hard to:

- Use custom base images (e.g., vllm, GPU-specific images)
- Run different Python versions for CommandLine entrypoints
- Control the build process from the client side

## Design

### Core Idea

Move Dockerfile generation to the client. The proto gains a `dockerfile` field (the full Dockerfile text) sent alongside the existing bundle. Workers become "dumb builders" — they just run `docker build` with whatever Dockerfile they receive, caching on `hash(dockerfile + bundle)`.

### Proto Changes

```protobuf
message EnvironmentConfig {
  repeated string pip_packages = 1;  // DEPRECATED - now encoded in dockerfile
  map<string, string> env_vars = 2;  // Keep: injected at container run time
  repeated string extras = 3;        // DEPRECATED - now encoded in dockerfile
  string python_version = 4;         // DEPRECATED - now encoded in dockerfile
  string dockerfile = 5;             // NEW: Full Dockerfile content for image build
}
```

The `dockerfile` field contains the complete Dockerfile text. Old fields (`pip_packages`, `extras`, `python_version`) are kept for wire compatibility but ignored when `dockerfile` is set.

### Template Variables

The Dockerfile may contain template placeholders that the **worker** resolves at build time:

- `{{BUNDLE_PATH}}` — not needed, bundle is always `COPY . .` into `/app`

Actually, we don't need template variables. The Dockerfile is self-contained. The worker just:
1. Extracts the bundle to a directory
2. Writes the `dockerfile` field as `Dockerfile.iris` in that directory
3. Runs `docker build`

### Client-Side Dockerfile Generation

`EnvironmentSpec.to_proto()` in `types.py` generates the Dockerfile from the user's configuration:

```python
def _generate_dockerfile(
    python_version: str,
    extras: list[str],
    pip_packages: list[str],
    pre_install_packages: list[str] | None = None,
) -> str:
    """Generate a Dockerfile from environment configuration.

    The generated Dockerfile:
    1. Starts from python:{version} or python:{version}-slim
    2. Installs system dependencies (git, curl, build-essential, Rust)
    3. Copies uv from the official image
    4. Optionally pre-installs common packages for better cache hits
    5. Copies bundle and runs uv sync
    6. Installs additional pip packages
    """
    base_image = f"python:{python_version}" if pip_packages else f"python:{python_version}-slim"

    extras_flags = " ".join(f"--extra {e}" for e in extras) if extras else ""

    pip_install_step = ""
    if pip_packages:
        packages_str = " ".join(f'"{pkg}"' for pkg in pip_packages)
        pip_install_step = f"\nRUN uv pip install {packages_str}\n"

    # Pre-install step: install frozen deps before copying full bundle
    # so that source-only changes still hit the Docker layer cache
    pre_install_step = ""
    if pre_install_packages:
        pkgs = " ".join(f'"{p}"' for p in pre_install_packages)
        pre_install_step = f"\nRUN uv pip install {pkgs}\n"

    return DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        extras_flags=extras_flags,
        pip_install_step=pip_install_step,
        pre_install_step=pre_install_step,
    )
```

### Pre-Install Optimization

To improve cache hits when only source code changes (not dependencies), the client can generate a "pre-install" layer. Before the `COPY . .` step, it:

1. Runs `uv export --no-dev` or reads `uv.lock` to extract pinned package versions
2. Adds a `RUN uv pip install <frozen-deps>` step early in the Dockerfile
3. The `COPY . .` and `uv sync` steps come after, so changing source code doesn't invalidate the pre-install layer

The Dockerfile template becomes:

```dockerfile
FROM {base_image}

RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

{pre_install_step}

COPY . .

RUN --mount=type=cache,id=iris-uv-global,sharing=shared,target=/opt/uv-cache \
    --mount=type=cache,id=iris-cargo,sharing=shared,target=/root/.cargo/registry \
    --mount=type=cache,id=iris-cargo-git,sharing=shared,target=/root/.cargo/git \
    uv sync {extras_flags}

ENV PATH="/app/.venv/bin:$PATH"

RUN uv pip install cloudpickle
{pip_install_step}
```

### Worker-Side Changes

`ImageCache.build()` changes signature:

```python
def build(
    self,
    bundle_path: Path,
    dockerfile: str,
    job_id: str,
    task_logs: TaskLogs | None = None,
) -> BuildResult:
    # Hash = sha256(dockerfile + bundle contents)
    # Same caching logic, but dockerfile replaces the old parameter soup
```

`TaskAttempt._build_image()` simplifies to:

```python
def _build_image(self) -> None:
    env_config = self.request.environment
    dockerfile = env_config.dockerfile
    if not dockerfile:
        raise RuntimeError("No dockerfile in environment config")

    build_result = self._image_provider.build(
        bundle_path=self._bundle_path,
        dockerfile=dockerfile,
        job_id=self.job_id,
        task_logs=self.logs,
    )
```

### ImageProvider Protocol Update

```python
class ImageProvider(Protocol):
    def build(
        self,
        bundle_path: Path,
        dockerfile: str,
        job_id: str,
        task_logs: TaskLogs | None = None,
    ) -> BuildResult: ...

    def protect(self, tag: str) -> None: ...
    def unprotect(self, tag: str) -> None: ...
```

### Custom Docker Image Support

Users can provide a custom Dockerfile via `EnvironmentSpec`:

```python
@dataclass
class EnvironmentSpec:
    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None
    dockerfile: str | None = None  # NEW: custom Dockerfile override
```

When `dockerfile` is set, it's used as-is. When not set, the default is generated from `pip_packages`/`extras`/`python_version`.

### Cache Key Changes

Currently: `hash(pyproject.toml + uv.lock + base_image + pip_packages + extras)`
After: `hash(dockerfile + pyproject.toml + uv.lock)`

The Dockerfile itself encodes all the configuration, so hashing it plus the lock files captures everything needed for cache correctness.

## Implementation Stages

### Stage 1: Proto + Client Dockerfile Generation
- Add `dockerfile` field to `EnvironmentConfig` proto
- Regenerate proto files
- Add `_generate_dockerfile()` function to `types.py`
- Update `EnvironmentSpec.to_proto()` to populate `dockerfile`
- Tests: unit test for dockerfile generation

### Stage 2: Worker Accepts Dockerfile
- Update `ImageProvider` protocol to accept `dockerfile: str`
- Update `ImageCache.build()` to use `dockerfile` parameter
- Update `TaskAttempt._build_image()` to pass dockerfile from proto
- Remove old Dockerfile template from `builder.py`
- Tests: unit test for new ImageCache.build()

### Stage 3: Custom Dockerfile Support + Pre-Install
- Add `dockerfile` field to `EnvironmentSpec`
- Add pre-install optimization (read uv.lock for frozen deps)
- Tests: e2e test with custom dockerfile, test pre-install caching

### Stage 4: Cleanup
- Remove deprecated fields usage (pip_packages, extras, python_version used for image building)
- Update documentation
- Run full test suite

## Files Modified

| File | Change |
|------|--------|
| `src/iris/rpc/cluster.proto` | Add `dockerfile` field to `EnvironmentConfig` |
| `src/iris/rpc/cluster_pb2.py` | Regenerated |
| `src/iris/cluster/types.py` | Add `_generate_dockerfile()`, update `EnvironmentSpec` |
| `src/iris/cluster/worker/builder.py` | Simplify `ImageCache.build()`, remove `DOCKERFILE_TEMPLATE` |
| `src/iris/cluster/worker/task_attempt.py` | Simplify `_build_image()` |
| `src/iris/cluster/client/remote_client.py` | Pass through dockerfile in environment |
| `tests/cluster/worker/test_builder.py` | Update tests for new interface |
| `tests/cluster/test_types.py` | Test dockerfile generation |
