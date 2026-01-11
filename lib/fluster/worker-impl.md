# Fluster Worker Implementation Plan

## Overview

Implement the WorkerService defined in `lib/fluster/src/fluster/proto/cluster.proto` as a robust job execution daemon. The worker accepts job requests over Connect RPC, manages job lifecycle in isolated containers, and provides both RPC and web interfaces.

**Key Requirements:**
- Bundle caching with fsspec (GCS paths assumed unique)
- LRU cache for venvs and Docker images with shared UV cache
- Job monitoring with cgroup isolation (v2 required) and proper cleanup
- Integration tests against real Docker runtime using local file paths

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WorkerServer                                │
│  ┌─────────────────┐  ┌───────────────────────────────────────┐ │
│  │  Connect RPC    │  │        Web Dashboard                  │ │
│  │  WorkerService  │  │  /health, /jobs, /jobs/{id}/logs     │ │
│  └────────┬────────┘  └───────────────────────────────────────┘ │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    JobManager                               │ │
│  │  - Job lifecycle orchestration                             │ │
│  │  - Semaphore-based concurrency                             │ │
│  │  - Port allocation                                          │ │
│  │  - Status tracking                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│           │           │              │                          │
│           ▼           ▼              ▼                          │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐                │
│  │ BundleCache  │ │  Builder │ │   Runtime    │                │
│  │  (fsspec)    │ │ (Docker) │ │  (Docker)    │                │
│  └──────────────┘ └──────────┘ └──────────────┘                │
│                        │                                        │
│                        ▼                                        │
│               ┌────────────────┐                               │
│               │   VenvCache    │                               │
│               │  (LRU + UV)    │                               │
│               └────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
lib/fluster/src/fluster/cluster/worker/
├── __init__.py          # Public API exports
├── bundle.py            # Bundle download/caching (fsspec)
├── builder.py           # Docker image + venv building with LRU cache
├── runtime.py           # Container execution (Docker-focused)
├── manager.py           # Job lifecycle management + port allocation
├── service.py           # WorkerService RPC implementation
├── server.py            # HTTP server (Connect RPC + web dashboard)
├── types.py             # Worker-internal types (Job dataclass, etc.)
└── main.py              # Click-based CLI entry point
```

---

## Stage 1: Worker Internal Types

**Goal:** Define internal worker types that wrap proto messages and track job state.

**Files:** `types.py`

**Implementation:**

```python
# types.py
from dataclasses import dataclass, field
import asyncio
from fluster import cluster_pb2

@dataclass
class Job:
    """Internal job tracking state."""
    job_id: str
    request: cluster_pb2.RunJobRequest
    status: int = cluster_pb2.JOB_STATE_PENDING
    exit_code: int | None = None
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    ports: dict[str, int] = field(default_factory=dict)

    # Internals
    container_id: str | None = None
    log_queue: asyncio.Queue | None = None
    task: asyncio.Task | None = None

    def to_proto(self) -> cluster_pb2.JobStatus:
        """Convert to proto JobStatus."""
        return cluster_pb2.JobStatus(
            job_id=self.job_id,
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            started_at_ms=self.started_at_ms or 0,
            finished_at_ms=self.finished_at_ms or 0,
            ports=self.ports,
        )
```

**Exit Conditions:**
- [ ] `Job` dataclass with all fields needed for lifecycle tracking
- [ ] Proto conversion methods work correctly

---

## Stage 2: Bundle Cache (fsspec)

**Goal:** Download and cache workspace bundles from GCS using fsspec. Assume GCS paths are unique identifiers.

**Files:** `bundle.py`

**Implementation:**

```python
# bundle.py
import fsspec
import hashlib
import zipfile
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BundleCache:
    """Cache for workspace bundles downloaded from GCS.

    Assumes GCS paths are unique - uses path as cache key.
    Two-level caching: zip files + extracted directories.

    Supports both gs:// paths (requires GCS auth) and file:// paths
    for local testing.
    """

    def __init__(self, cache_dir: Path, max_bundles: int = 100):
        self._cache_dir = cache_dir
        self._bundles_dir = cache_dir / "bundles"
        self._extracts_dir = cache_dir / "extracts"
        self._max_bundles = max_bundles
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._bundles_dir.mkdir(parents=True, exist_ok=True)
        self._extracts_dir.mkdir(parents=True, exist_ok=True)

    def _path_to_key(self, gcs_path: str) -> str:
        """Convert GCS path to cache key (hash)."""
        return hashlib.sha256(gcs_path.encode()).hexdigest()[:16]

    async def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        """Get bundle path, downloading if needed.

        Args:
            gcs_path: gs://bucket/path/bundle.zip or file:///local/path.zip
            expected_hash: Optional SHA256 hash for verification

        Returns:
            Path to extracted bundle directory
        """
        key = self._path_to_key(gcs_path)
        extract_path = self._extracts_dir / key

        if extract_path.exists():
            return extract_path

        # Download and extract
        zip_path = self._bundles_dir / f"{key}.zip"
        if not zip_path.exists():
            await self._download(gcs_path, zip_path)

        if expected_hash:
            actual_hash = await self._compute_hash(zip_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Bundle hash mismatch: {actual_hash} != {expected_hash}")

        await self._extract(zip_path, extract_path)
        await self._evict_old_bundles()

        return extract_path

    async def _download(self, gcs_path: str, local_path: Path) -> None:
        """Download bundle using fsspec."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._download_sync, gcs_path, local_path)

    def _download_sync(self, gcs_path: str, local_path: Path) -> None:
        # fsspec handles gs://, file://, and other protocols
        with fsspec.open(gcs_path, "rb") as src:
            with open(local_path, "wb") as dst:
                dst.write(src.read())

    async def _extract(self, zip_path: Path, extract_path: Path) -> None:
        """Extract zip to directory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._extract_sync, zip_path, extract_path)

    def _extract_sync(self, zip_path: Path, extract_path: Path) -> None:
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_path)

    async def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._compute_hash_sync, path)

    def _compute_hash_sync(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    async def _evict_old_bundles(self) -> None:
        """LRU eviction when over max_bundles."""
        extracts = list(self._extracts_dir.iterdir())
        if len(extracts) <= self._max_bundles:
            return

        # Sort by mtime, remove oldest
        extracts.sort(key=lambda p: p.stat().st_mtime)
        for path in extracts[: len(extracts) - self._max_bundles]:
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            # Also remove corresponding zip
            zip_path = self._bundles_dir / f"{path.name}.zip"
            if zip_path.exists():
                zip_path.unlink()
```

**Exit Conditions:**
- [ ] `BundleCache` downloads from GCS via fsspec
- [ ] Supports `file://` paths for local testing
- [ ] Cache key derived from path (assumed unique)
- [ ] Zip extraction with verification
- [ ] LRU eviction when cache exceeds limit
- [ ] Test: download local bundle, verify caching
- [ ] Test: verify hash checking works

---

## Stage 3: VenvCache (Shared UV)

**Goal:** LRU cache for pre-built virtual environments using shared UV cache.

**Files:** `builder.py` (part 1)

**Implementation:**

```python
# builder.py (VenvCache portion)
import xxhash
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass
import time
import asyncio

@dataclass
class VenvCacheEntry:
    """Cached venv metadata."""
    deps_hash: str
    created_at: float
    size_bytes: int

class VenvCache:
    """LRU cache for pre-built Python virtual environments.

    Uses shared UV cache directory for wheel reuse across builds.
    Stores compressed venvs keyed by deps_hash (pyproject.toml + uv.lock).

    Permissions: The UV cache directory must be writable by container user
    (typically UID 1000). Call ensure_permissions() after creating.
    """

    def __init__(
        self,
        cache_dir: Path,
        uv_cache_dir: Path,
        max_entries: int = 20,
    ):
        self._cache_dir = cache_dir / "venvs"
        self._uv_cache_dir = uv_cache_dir
        self._max_entries = max_entries
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._uv_cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_permissions(self, uid: int = 1000, gid: int = 1000) -> None:
        """Ensure UV cache has correct permissions for container user."""
        import shutil
        shutil.chown(self._uv_cache_dir, uid, gid)
        for path in self._uv_cache_dir.rglob("*"):
            try:
                shutil.chown(path, uid, gid)
            except PermissionError:
                pass

    def compute_deps_hash(self, bundle_path: Path) -> str:
        """Compute hash from pyproject.toml + uv.lock."""
        h = xxhash.xxh64()
        for fname in ["pyproject.toml", "uv.lock"]:
            fpath = bundle_path / fname
            if fpath.exists():
                h.update(fpath.read_bytes())
        return h.hexdigest()

    def get(self, deps_hash: str) -> Path | None:
        """Get cached venv archive path if exists."""
        venv_archive = self._cache_dir / f"{deps_hash}.tar.zst"
        if venv_archive.exists():
            # Update mtime for LRU
            venv_archive.touch()
            return venv_archive
        return None

    async def build_venv(self, bundle_path: Path, extras: list[str]) -> tuple[Path, str]:
        """Build venv for bundle, returning (venv_path, deps_hash).

        Uses shared UV cache for fast builds.
        """
        deps_hash = self.compute_deps_hash(bundle_path)

        # Check cache first
        cached = self.get(deps_hash)
        if cached:
            venv_path = await self._extract_venv(cached, bundle_path)
            return venv_path, deps_hash

        # Build new venv with shared UV cache
        venv_path = bundle_path / ".venv"
        await self._run_uv_sync(bundle_path, extras)

        # Cache the built venv
        await self._archive_venv(venv_path, deps_hash)
        await self._evict_lru()

        return venv_path, deps_hash

    async def _run_uv_sync(self, bundle_path: Path, extras: list[str]) -> None:
        """Run uv sync with shared cache."""
        cmd = ["uv", "sync", "--frozen", "--all-packages"]
        for extra in extras:
            cmd.extend(["--extra", extra])

        env = {
            **os.environ,
            "UV_CACHE_DIR": str(self._uv_cache_dir),
            "UV_LINK_MODE": "copy",
        }

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=bundle_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"uv sync failed: {stderr.decode()}")

    async def _archive_venv(self, venv_path: Path, deps_hash: str) -> None:
        """Compress venv with zstd for storage."""
        archive_path = self._cache_dir / f"{deps_hash}.tar.zst"
        # Use zstd with parallel threads for speed
        proc = await asyncio.create_subprocess_shell(
            f"tar -cf - -C {venv_path.parent} .venv | zstd -T0 -o {archive_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _extract_venv(self, archive: Path, target_bundle: Path) -> Path:
        """Extract cached venv to target bundle."""
        proc = await asyncio.create_subprocess_shell(
            f"zstd -d -c {archive} | tar -xf - -C {target_bundle}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return target_bundle / ".venv"

    async def _evict_lru(self) -> None:
        """Remove oldest entries when over max_entries."""
        archives = list(self._cache_dir.glob("*.tar.zst"))
        if len(archives) <= self._max_entries:
            return

        archives.sort(key=lambda p: p.stat().st_mtime)
        for path in archives[: len(archives) - self._max_entries]:
            path.unlink()
```

**Exit Conditions:**
- [ ] `VenvCache` builds venvs with `uv sync`
- [ ] Shared UV cache directory for wheel reuse
- [ ] Content-addressed caching via deps_hash
- [ ] zstd compression for venv archives
- [ ] LRU eviction policy
- [ ] `ensure_permissions()` for container user access
- [ ] Test: build venv, verify caching works
- [ ] Test: verify shared UV cache speeds up subsequent builds

---

## Stage 4: Docker Image Builder

**Goal:** Build Docker images with layered caching. LRU cache for built images. Workspace-based only (no docker_image support).

**Files:** `builder.py` (part 2)

**Implementation:**

```python
# builder.py (ImageBuilder portion)
from dataclasses import dataclass
import time

@dataclass
class BuildResult:
    """Result of image build."""
    image_tag: str
    deps_hash: str
    build_time_ms: int
    from_cache: bool

# Dockerfile template with UV and shared cache
# Only supports workspace-based builds (not pre-built docker images)
DOCKERFILE_TEMPLATE = '''
FROM {base_image}

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Setup environment
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Layer 1: Dependencies (cached when pyproject.toml/uv.lock unchanged)
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/opt/uv-cache \\
    uv sync --frozen --no-install-project {extras_flags}

# Layer 2: Source code + install
COPY . .
RUN --mount=type=cache,target=/opt/uv-cache \\
    uv sync --frozen --no-editable {extras_flags}

# Use the venv python
ENV PATH="/app/.venv/bin:$PATH"
'''

class ImageBuilder:
    """Builds Docker images with intelligent caching.

    Image tag: {registry}/fluster-job-{job_id}:{deps_hash[:8]}
    Uses Docker BuildKit cache mounts for UV cache sharing.

    Only supports workspace-based builds. The EnvironmentConfig.workspace
    field must be set; docker_image is not supported.
    """

    def __init__(
        self,
        cache_dir: Path,
        registry: str,
        max_images: int = 50,
    ):
        self._cache_dir = cache_dir / "images"
        self._registry = registry
        self._max_images = max_images
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
    ) -> BuildResult:
        """Build Docker image for job.

        Returns cached image if deps_hash matches.
        """
        image_tag = f"{self._registry}/fluster-job-{job_id}:{deps_hash[:8]}"

        # Check if image exists locally
        if await self._image_exists(image_tag):
            return BuildResult(
                image_tag=image_tag,
                deps_hash=deps_hash,
                build_time_ms=0,
                from_cache=True,
            )

        # Build image
        start = time.time()
        extras_flags = " ".join(f"--extra {e}" for e in extras) if extras else ""
        dockerfile = DOCKERFILE_TEMPLATE.format(
            base_image=base_image,
            extras_flags=extras_flags,
        )
        await self._docker_build(bundle_path, dockerfile, image_tag)
        build_time_ms = int((time.time() - start) * 1000)

        await self._evict_old_images()

        return BuildResult(
            image_tag=image_tag,
            deps_hash=deps_hash,
            build_time_ms=build_time_ms,
            from_cache=False,
        )

    async def _docker_build(self, context: Path, dockerfile: str, tag: str) -> None:
        """Run docker build with BuildKit."""
        dockerfile_path = context / "Dockerfile.fluster"
        dockerfile_path.write_text(dockerfile)

        cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", tag,
            "--build-arg", "BUILDKIT_INLINE_CACHE=1",
            str(context),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker build failed: {stderr.decode()}")

        # Cleanup generated dockerfile
        dockerfile_path.unlink()

    async def _image_exists(self, tag: str) -> bool:
        """Check if image exists locally."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", tag,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0

    async def _evict_old_images(self) -> None:
        """Remove old fluster images when over limit."""
        # List images matching our pattern
        proc = await asyncio.create_subprocess_exec(
            "docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
            "--filter", f"reference={self._registry}/fluster-job-*",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        lines = stdout.decode().strip().split("\n")
        if len(lines) <= self._max_images:
            return

        # Parse and sort by creation time
        images = []
        for line in lines:
            if "\t" in line:
                tag, created = line.split("\t", 1)
                images.append((tag, created))

        images.sort(key=lambda x: x[1])  # Sort by created time

        # Remove oldest
        for tag, _ in images[: len(images) - self._max_images]:
            await asyncio.create_subprocess_exec("docker", "rmi", tag)
```

**Exit Conditions:**
- [ ] `ImageBuilder` generates content-addressed image tags
- [ ] Dockerfile with UV + BuildKit cache mounts
- [ ] Cache hit detection via `docker image inspect`
- [ ] LRU eviction for old images
- [ ] Registry passed as constructor argument
- [ ] Test: build image, verify caching
- [ ] Test: verify deps_hash change triggers rebuild

---

## Stage 5: Docker Runtime

**Goal:** Execute jobs in Docker containers with cgroups v2 resource limits.

**Files:** `runtime.py`

**Implementation:**

```python
# runtime.py
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
import asyncio
import time

@dataclass
class ContainerConfig:
    """Configuration for container execution."""
    image: str
    command: list[str]
    env: dict[str, str]
    workdir: str = "/app"
    cpu_millicores: int | None = None
    memory_mb: int | None = None
    timeout_seconds: int | None = None
    mounts: list[tuple[str, str, str]] = field(default_factory=list)  # (host, container, mode)
    ports: dict[str, int] = field(default_factory=dict)  # name -> host_port

@dataclass
class ContainerResult:
    """Result of container execution."""
    container_id: str
    exit_code: int
    started_at: float
    finished_at: float
    error: str | None = None

class DockerRuntime:
    """Execute jobs in Docker containers with cgroups v2 resource limits.

    Requires cgroups v2 (no v1 fallback). Security hardening:
    - no-new-privileges
    - cap-drop ALL
    """

    async def run(
        self,
        config: ContainerConfig,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> ContainerResult:
        """Run container to completion.

        Args:
            config: Container configuration
            log_callback: Called with (stream, data) for stdout/stderr

        Returns:
            ContainerResult with exit code and timing
        """
        container_id = await self._create_container(config)
        started_at = time.time()

        try:
            await self._start_container(container_id)

            # Stream logs if callback provided
            log_task = None
            if log_callback:
                log_task = asyncio.create_task(
                    self._stream_logs(container_id, log_callback)
                )

            # Wait for completion with timeout
            exit_code = await self._wait_container(
                container_id,
                timeout=config.timeout_seconds,
            )

            if log_task:
                log_task.cancel()
                try:
                    await log_task
                except asyncio.CancelledError:
                    pass

            return ContainerResult(
                container_id=container_id,
                exit_code=exit_code,
                started_at=started_at,
                finished_at=time.time(),
            )
        except asyncio.TimeoutError:
            await self.kill(container_id, force=True)
            return ContainerResult(
                container_id=container_id,
                exit_code=-1,
                started_at=started_at,
                finished_at=time.time(),
                error="Timeout exceeded",
            )

    async def _create_container(self, config: ContainerConfig) -> str:
        """Create container with cgroups v2 resource limits."""
        cmd = [
            "docker", "create",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-w", config.workdir,
        ]

        # Resource limits (cgroups v2)
        if config.cpu_millicores:
            cpus = config.cpu_millicores / 1000
            cmd.extend(["--cpus", str(cpus)])
        if config.memory_mb:
            cmd.extend(["--memory", f"{config.memory_mb}m"])

        # Environment variables
        for k, v in config.env.items():
            cmd.extend(["-e", f"{k}={v}"])

        # Mounts
        for host, container, mode in config.mounts:
            cmd.extend(["-v", f"{host}:{container}:{mode}"])

        # Port mappings
        for name, host_port in config.ports.items():
            cmd.extend(["-p", f"{host_port}:{host_port}"])

        cmd.append(config.image)
        cmd.extend(config.command)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create container: {stderr.decode()}")
        return stdout.decode().strip()

    async def _start_container(self, container_id: str) -> None:
        """Start a created container."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "start", container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start container: {stderr.decode()}")

    async def _wait_container(self, container_id: str, timeout: int | None) -> int:
        """Wait for container to exit."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "wait", container_id,
            stdout=asyncio.subprocess.PIPE,
        )

        if timeout:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        else:
            stdout, _ = await proc.communicate()

        return int(stdout.decode().strip())

    async def _stream_logs(
        self,
        container_id: str,
        callback: Callable[[str, str], None],
    ) -> None:
        """Stream container logs to callback."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def stream(pipe, name):
            async for line in pipe:
                callback(name, line.decode())

        await asyncio.gather(
            stream(proc.stdout, "stdout"),
            stream(proc.stderr, "stderr"),
        )

    async def kill(self, container_id: str, force: bool = False) -> None:
        """Kill container."""
        signal = "SIGKILL" if force else "SIGTERM"
        await asyncio.create_subprocess_exec(
            "docker", "kill", f"--signal={signal}", container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def remove(self, container_id: str) -> None:
        """Remove container."""
        await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
```

**Exit Conditions:**
- [ ] `DockerRuntime` creates containers with cgroups v2 limits
- [ ] Security hardening (no-new-privileges, cap-drop)
- [ ] Port mapping support
- [ ] Log streaming via callback
- [ ] Timeout handling with forced kill
- [ ] Test: run simple container, verify exit code
- [ ] Test: verify resource limits applied (check cgroup files)
- [ ] Test: verify timeout kills container

---

## Stage 6: Port Allocator

**Goal:** Manage port allocation for jobs with named ports.

**Files:** `manager.py` (part 1)

**Implementation:**

```python
# manager.py (PortAllocator portion)
import socket
import asyncio

class PortAllocator:
    """Allocate ephemeral ports for jobs.

    Tracks allocated ports to avoid conflicts.
    Ports are released when jobs terminate.
    """

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self, count: int = 1) -> list[int]:
        """Allocate N unused ports."""
        async with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    async def release(self, ports: list[int]) -> None:
        """Release allocated ports."""
        async with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        """Find an unused port in range."""
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        """Check if port is free on host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False
```

**Exit Conditions:**
- [ ] `PortAllocator` finds free ports in range
- [ ] Tracks allocated ports to avoid conflicts
- [ ] Release ports on job termination
- [ ] Test: allocate ports, verify they're usable
- [ ] Test: verify no port reuse before release

---

## Stage 7: JobManager

**Goal:** Orchestrate job lifecycle with all components.

**Files:** `manager.py` (part 2)

**Implementation:**

```python
# manager.py (JobManager portion)
import cloudpickle
import uuid
import time
from typing import AsyncIterator

from fluster import cluster_pb2
from .types import Job
from .bundle import BundleCache
from .builder import VenvCache, ImageBuilder
from .runtime import DockerRuntime, ContainerConfig

class JobManager:
    """Orchestrates job lifecycle.

    Phases:
    1. PENDING: Job submitted, waiting for resources
    2. BUILDING: Downloading bundle, building image
    3. RUNNING: Container executing
    4. SUCCEEDED/FAILED/KILLED: Terminal states

    Cleanup: Removes containers after completion. Logs are retained
    in memory (no rotation - assumes adequate disk space).
    """

    def __init__(
        self,
        bundle_cache: BundleCache,
        venv_cache: VenvCache,
        image_builder: ImageBuilder,
        runtime: DockerRuntime,
        port_allocator: PortAllocator,
        max_concurrent_jobs: int = 10,
    ):
        self._bundle_cache = bundle_cache
        self._venv_cache = venv_cache
        self._image_builder = image_builder
        self._runtime = runtime
        self._port_allocator = port_allocator
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit_job(self, request: cluster_pb2.RunJobRequest) -> str:
        """Submit job for execution.

        Returns job_id immediately, execution happens in background.
        """
        job_id = request.job_id or str(uuid.uuid4())

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = await self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports))

        job = Job(
            job_id=job_id,
            request=request,
            status=cluster_pb2.JOB_STATE_PENDING,
            ports=ports,
            log_queue=asyncio.Queue(),
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Start execution in background
        job.task = asyncio.create_task(self._execute_job(job))

        return job_id

    async def _execute_job(self, job: Job) -> None:
        """Execute job through all phases."""
        async with self._semaphore:
            try:
                # Phase 1: Download bundle
                job.status = cluster_pb2.JOB_STATE_BUILDING
                job.started_at_ms = int(time.time() * 1000)

                bundle_path = await self._bundle_cache.get_bundle(
                    job.request.bundle_gcs_path,
                    job.request.environment.bundle_hash if job.request.environment.bundle_hash else None,
                )

                # Phase 2: Build image
                env_config = job.request.environment
                extras = list(env_config.extras)

                # Compute deps_hash for caching
                deps_hash = self._venv_cache.compute_deps_hash(bundle_path)

                build_result = await self._image_builder.build(
                    bundle_path=bundle_path,
                    base_image="python:3.11-slim",
                    extras=extras,
                    job_id=job.job_id,
                    deps_hash=deps_hash,
                )

                # Phase 3: Run container
                job.status = cluster_pb2.JOB_STATE_RUNNING

                # Deserialize entrypoint
                entrypoint = cloudpickle.loads(job.request.serialized_entrypoint)
                command = self._build_command(entrypoint)

                # Build environment
                env = dict(job.request.env_vars)
                env.update(dict(env_config.env_vars))
                env["FLUSTER_JOB_ID"] = job.job_id
                for name, port in job.ports.items():
                    env[f"FLUSTER_PORT_{name.upper()}"] = str(port)

                config = ContainerConfig(
                    image=build_result.image_tag,
                    command=command,
                    env=env,
                    cpu_millicores=job.request.resources.cpu * 1000 if job.request.resources.cpu else None,
                    memory_mb=self._parse_memory(job.request.resources.memory),
                    timeout_seconds=job.request.timeout_seconds or None,
                    ports=job.ports,
                )

                # Setup log streaming
                def log_callback(stream: str, data: str):
                    entry = cluster_pb2.LogEntry(
                        timestamp_ms=int(time.time() * 1000),
                        source=stream,
                        data=data,
                    )
                    job.log_queue.put_nowait(entry)

                result = await self._runtime.run(config, log_callback)
                job.container_id = result.container_id

                # Phase 4: Complete
                job.exit_code = result.exit_code
                job.finished_at_ms = int(time.time() * 1000)

                if result.error:
                    job.status = cluster_pb2.JOB_STATE_FAILED
                    job.error = result.error
                elif result.exit_code == 0:
                    job.status = cluster_pb2.JOB_STATE_SUCCEEDED
                else:
                    job.status = cluster_pb2.JOB_STATE_FAILED
                    job.error = f"Exit code: {result.exit_code}"

            except Exception as e:
                job.status = cluster_pb2.JOB_STATE_FAILED
                job.error = str(e)
                job.finished_at_ms = int(time.time() * 1000)
            finally:
                # Cleanup: release ports, remove container
                await self._port_allocator.release(list(job.ports.values()))
                if job.container_id:
                    await self._runtime.remove(job.container_id)

    def _build_command(self, entrypoint) -> list[str]:
        """Build command to run entrypoint."""
        # Serialize entrypoint and run via python -c
        serialized = cloudpickle.dumps(entrypoint)
        import base64
        encoded = base64.b64encode(serialized).decode()
        return [
            "python", "-c",
            f"import cloudpickle, base64; e = cloudpickle.loads(base64.b64decode('{encoded}')); e.callable(*e.args, **e.kwargs)"
        ]

    def _parse_memory(self, memory_str: str) -> int | None:
        """Parse memory string like '8g' to MB."""
        if not memory_str:
            return None
        memory_str = memory_str.lower().strip()
        if memory_str.endswith("g"):
            return int(float(memory_str[:-1]) * 1024)
        elif memory_str.endswith("m"):
            return int(float(memory_str[:-1]))
        return int(memory_str)

    async def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def list_jobs(self, namespace: str | None = None) -> list[Job]:
        """List all jobs."""
        return list(self._jobs.values())

    async def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill a running job."""
        job = self._jobs.get(job_id)
        if not job or not job.container_id:
            return False

        if job.status not in (cluster_pb2.JOB_STATE_RUNNING, cluster_pb2.JOB_STATE_BUILDING):
            return False

        # Send SIGTERM
        await self._runtime.kill(job.container_id, force=False)

        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(
                self._wait_for_termination(job),
                timeout=term_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            # Force kill
            await self._runtime.kill(job.container_id, force=True)

        job.status = cluster_pb2.JOB_STATE_KILLED
        job.finished_at_ms = int(time.time() * 1000)
        return True

    async def _wait_for_termination(self, job: Job) -> None:
        """Wait until job reaches terminal state."""
        while job.status in (cluster_pb2.JOB_STATE_RUNNING, cluster_pb2.JOB_STATE_BUILDING):
            await asyncio.sleep(0.1)

    async def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.LogEntry]:
        """Get logs for a job.

        Args:
            job_id: Job ID
            start_line: Starting line number. If negative, returns last N lines
                       (e.g., start_line=-100 returns last 100 lines for tailing).

        Returns:
            List of log entries
        """
        job = self._jobs.get(job_id)
        if not job:
            return []

        # Drain queue to list
        logs = []
        while not job.log_queue.empty():
            try:
                logs.append(job.log_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Put back for future reads
        for log in logs:
            job.log_queue.put_nowait(log)

        # Handle negative start_line (tail behavior)
        if start_line < 0:
            return logs[start_line:]
        return logs[start_line:]
```

**Exit Conditions:**
- [ ] `JobManager` orchestrates full job lifecycle
- [ ] Phase transitions (PENDING → BUILDING → RUNNING → terminal)
- [ ] Port allocation integrated
- [ ] Log collection via queue
- [ ] Graceful kill with timeout
- [ ] Cleanup removes containers
- [ ] Negative `start_line` for log tailing
- [ ] Test: full job execution lifecycle
- [ ] Test: concurrent job limiting
- [ ] Test: job killing

---

## Stage 8: WorkerService (RPC)

**Goal:** Implement WorkerService RPC interface using Connect RPC.

**Files:** `service.py`

**Implementation:**

```python
# service.py
import time
import re
from connectrpc import RequestContext

from fluster import cluster_pb2
from .manager import JobManager
from fluster import cluster_pb2

class WorkerServiceImpl:
    """Implements WorkerService RPC interface.

    This class conforms to the WorkerService protocol generated by connectrpc.
    Mount using WorkerServiceASGIApplication from cluster_connect.py.

    Methods:
    - run_job: Submit job for execution
    - get_job_status: Query job status
    - list_jobs: List jobs (optionally filtered)
    - fetch_logs: Get job logs with filtering (negative start_line for tailing)
    - kill_job: Terminate job
    - health_check: Worker health status
    """

    def __init__(self, manager: JobManager):
        self._manager = manager
        self._start_time = time.time()

    async def run_job(
        self,
        request: cluster_pb2.RunJobRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.RunJobResponse:
        """Submit job for execution."""
        job_id = await self._manager.submit_job(request)
        job = await self._manager.get_job(job_id)

        return cluster_pb2.RunJobResponse(
            job_id=job_id,
            state=job.to_proto().state,
        )

    async def get_job_status(
        self,
        request: cluster_pb2.GetStatusRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.JobStatus:
        """Get job status."""
        job = await self._manager.get_job(request.job_id)
        if not job:
            from connectrpc import ConnectError, Code
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
        return job.to_proto()

    async def list_jobs(
        self,
        request: cluster_pb2.ListJobsRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.ListJobsResponse:
        """List jobs."""
        jobs = await self._manager.list_jobs(request.namespace or None)
        return cluster_pb2.ListJobsResponse(
            jobs=[job.to_proto() for job in jobs],
        )

    async def fetch_logs(
        self,
        request: cluster_pb2.FetchLogsRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.FetchLogsResponse:
        """Fetch job logs with filtering.

        Supports:
        - start_line: Line offset. Negative values for tailing (e.g., -100 for last 100 lines)
        - start_ms/end_ms: Time range filter
        - regex: Content filter
        - max_lines: Limit results
        """
        # Get logs with start_line handling (negative = tail)
        start_line = request.filter.start_line if request.filter.start_line else 0
        logs = await self._manager.get_logs(request.job_id, start_line=start_line)

        # Apply additional filters
        result = []
        for entry in logs:
            # Time range filter
            if request.filter.start_ms and entry.timestamp_ms < request.filter.start_ms:
                continue
            if request.filter.end_ms and entry.timestamp_ms > request.filter.end_ms:
                continue
            # Regex filter
            if request.filter.regex:
                if not re.search(request.filter.regex, entry.data):
                    continue

            result.append(entry)

            # Max lines limit
            if request.filter.max_lines and len(result) >= request.filter.max_lines:
                break

        return cluster_pb2.FetchLogsResponse(logs=result)

    async def kill_job(
        self,
        request: cluster_pb2.KillJobRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.Empty:
        """Kill running job."""
        success = await self._manager.kill_job(
            request.job_id,
            term_timeout_ms=request.term_timeout_ms or 5000,
        )
        if not success:
            from connectrpc import ConnectError, Code
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found or not running")
        return cluster_pb2.Empty()

    async def health_check(
        self,
        request: cluster_pb2.Empty,
        ctx: RequestContext,
    ) -> cluster_pb2.HealthResponse:
        """Worker health status."""
        jobs = await self._manager.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)

        return cluster_pb2.HealthResponse(
            healthy=True,
            uptime_ms=int((time.time() - self._start_time) * 1000),
            running_jobs=running,
        )
```

**Exit Conditions:**
- [ ] All 6 WorkerService RPCs implemented
- [ ] Conforms to generated `WorkerService` protocol
- [ ] FetchLogs supports filtering (regex, time range, max_lines)
- [ ] FetchLogs supports negative start_line for tailing
- [ ] KillJob with graceful timeout
- [ ] HealthCheck reports running jobs
- [ ] Test: RPC round-trip for each method

---

## Stage 9: HTTP Server + Web Dashboard

**Goal:** HTTP server with Connect RPC + simple web dashboard.

**Files:** `server.py`

**Implementation:**

```python
# server.py
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, HTMLResponse
from starlette.requests import Request

from fluster.cluster_connect import WorkerServiceASGIApplication
from .service import WorkerServiceImpl
from fluster import cluster_pb2

# Simple dashboard HTML
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
  <title>Fluster Worker</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .status-running { color: blue; }
    .status-succeeded { color: green; }
    .status-failed { color: red; }
  </style>
</head>
<body>
  <h1>Fluster Worker Dashboard</h1>
  <div id="stats"></div>
  <h2>Jobs</h2>
  <table id="jobs">
    <tr><th>ID</th><th>Status</th><th>Started</th><th>Finished</th><th>Error</th></tr>
  </table>
  <script>
    async function refresh() {
      const stats = await fetch('/api/stats').then(r => r.json());
      document.getElementById('stats').innerHTML =
        `<b>Running:</b> ${stats.running} | <b>Pending:</b> ${stats.pending} | ` +
        `<b>Building:</b> ${stats.building} | <b>Completed:</b> ${stats.completed}`;

      const jobs = await fetch('/api/jobs').then(r => r.json());
      const tbody = jobs.map(j => {
        const started = j.started_at ? new Date(j.started_at).toLocaleString() : '-';
        const finished = j.finished_at ? new Date(j.finished_at).toLocaleString() : '-';
        return `<tr>
          <td>${j.job_id.slice(0, 8)}...</td>
          <td class="status-${j.status}">${j.status}</td>
          <td>${started}</td>
          <td>${finished}</td>
          <td>${j.error || '-'}</td>
        </tr>`;
      }).join('');
      document.getElementById('jobs').innerHTML =
        '<tr><th>ID</th><th>Status</th><th>Started</th><th>Finished</th><th>Error</th></tr>' + tbody;
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
'''

class WorkerServer:
    """HTTP server with Connect RPC and web dashboard.

    Connect RPC is mounted at /fluster.cluster.WorkerService
    Web dashboard at /
    REST API for dashboard at /api/*
    """

    def __init__(
        self,
        service: WorkerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        # Create Connect RPC application
        rpc_app = WorkerServiceASGIApplication(service=self._service)

        routes = [
            # Web dashboard
            Route("/", self._dashboard),

            # REST API (for dashboard)
            Route("/api/stats", self._stats),
            Route("/api/jobs", self._list_jobs),
            Route("/api/jobs/{job_id}", self._get_job),
            Route("/api/jobs/{job_id}/logs", self._get_logs),

            # Connect RPC
            Mount(rpc_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    async def _dashboard(self, request: Request) -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    async def _stats(self, request: Request) -> JSONResponse:
        jobs = await self._service._manager.list_jobs()
        return JSONResponse({
            "running": sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING),
            "pending": sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_PENDING),
            "building": sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_BUILDING),
            "completed": sum(1 for j in jobs if j.status in (
                cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_KILLED
            )),
        })

    async def _list_jobs(self, request: Request) -> JSONResponse:
        jobs = await self._service._manager.list_jobs()
        return JSONResponse([
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "started_at": j.started_at_ms,
                "finished_at": j.finished_at_ms,
                "error": j.error,
            }
            for j in jobs
        ])

    async def _get_job(self, request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        job = await self._service._manager.get_job(job_id)
        if not job:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse({
            "job_id": job.job_id,
            "status": job.status.value,
            "started_at": job.started_at_ms,
            "finished_at": job.finished_at_ms,
            "exit_code": job.exit_code,
            "error": job.error,
            "ports": job.ports,
        })

    async def _get_logs(self, request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        # Support ?tail=N for last N lines
        tail = request.query_params.get("tail")
        start_line = -int(tail) if tail else 0
        logs = await self._service._manager.get_logs(job_id, start_line=start_line)
        return JSONResponse([
            {
                "timestamp": entry.timestamp_ms,
                "source": entry.source,
                "data": entry.data,
            }
            for entry in logs
        ])

    def run(self) -> None:
        """Run server (blocking)."""
        import uvicorn
        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        """Run server (async)."""
        import uvicorn
        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        server = uvicorn.Server(config)
        await server.serve()
```

**Exit Conditions:**
- [ ] HTTP server with Starlette
- [ ] Connect RPC mounted at generated path (`/fluster.cluster.WorkerService`)
- [ ] Web dashboard with live job stats
- [ ] REST API for dashboard consumption
- [ ] Tail support via query param
- [ ] Test: dashboard loads
- [ ] Test: Connect RPC client can call worker

---

## Stage 10: CLI Entry Point

**Goal:** Click-based CLI for starting worker with configuration.

**Files:** `main.py`

**Implementation:**

```python
# main.py
import click
import asyncio
from pathlib import Path

from .bundle import BundleCache
from .builder import VenvCache, ImageBuilder
from .runtime import DockerRuntime
from .manager import JobManager, PortAllocator
from .service import WorkerServiceImpl
from .server import WorkerServer


@click.group()
def cli():
    """Fluster Worker - Job execution daemon."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", default="~/.cache/fluster-worker", help="Cache directory")
@click.option("--uv-cache-dir", default="~/.cache/uv", help="Shared UV cache directory")
@click.option("--registry", required=True, help="Docker registry for built images")
@click.option("--max-concurrent-jobs", default=10, type=int, help="Max concurrent jobs")
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option("--max-bundles", default=100, type=int, help="Max cached bundles")
@click.option("--max-venvs", default=20, type=int, help="Max cached venvs")
@click.option("--max-images", default=50, type=int, help="Max cached Docker images")
def serve(
    host: str,
    port: int,
    cache_dir: str,
    uv_cache_dir: str,
    registry: str,
    max_concurrent_jobs: int,
    port_range: str,
    max_bundles: int,
    max_venvs: int,
    max_images: int,
):
    """Start the Fluster worker service."""
    asyncio.run(_serve(
        host, port, cache_dir, uv_cache_dir, registry,
        max_concurrent_jobs, port_range, max_bundles, max_venvs, max_images,
    ))


async def _serve(
    host: str,
    port: int,
    cache_dir: str,
    uv_cache_dir: str,
    registry: str,
    max_concurrent_jobs: int,
    port_range: str,
    max_bundles: int,
    max_venvs: int,
    max_images: int,
):
    cache_path = Path(cache_dir).expanduser()
    uv_cache_path = Path(uv_cache_dir).expanduser()

    port_start, port_end = map(int, port_range.split("-"))

    # Initialize components
    bundle_cache = BundleCache(cache_path, max_bundles=max_bundles)
    venv_cache = VenvCache(cache_path, uv_cache_path, max_entries=max_venvs)
    image_builder = ImageBuilder(cache_path, registry=registry, max_images=max_images)
    runtime = DockerRuntime()
    port_allocator = PortAllocator((port_start, port_end))

    # Ensure UV cache permissions for container user
    venv_cache.ensure_permissions()

    manager = JobManager(
        bundle_cache=bundle_cache,
        venv_cache=venv_cache,
        image_builder=image_builder,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    service = WorkerServiceImpl(manager)
    server = WorkerServer(service, host, port)

    click.echo(f"Starting Fluster worker on {host}:{port}")
    click.echo(f"  Registry: {registry}")
    click.echo(f"  Cache dir: {cache_path}")
    click.echo(f"  Max concurrent jobs: {max_concurrent_jobs}")
    await server.run_async()


@cli.command()
@click.option("--cache-dir", default="~/.cache/fluster-worker", help="Cache directory")
def cleanup(cache_dir: str):
    """Clean up cached bundles, venvs, and images."""
    import shutil
    cache_path = Path(cache_dir).expanduser()
    if cache_path.exists():
        shutil.rmtree(cache_path)
        click.echo(f"Removed cache directory: {cache_path}")
    else:
        click.echo(f"Cache directory does not exist: {cache_path}")


if __name__ == "__main__":
    cli()
```

**Exit Conditions:**
- [ ] Click-based CLI with all configuration options
- [ ] `serve` command starts worker
- [ ] `cleanup` command removes cache
- [ ] Registry is required argument
- [ ] Components initialized in correct order
- [ ] UV cache permissions set
- [ ] Test: CLI help works
- [ ] Test: serve starts server

---

## Stage 11: Integration Tests

**Goal:** Full integration tests against real Docker runtime using local file paths.

**Files:** `tests/cluster/worker/test_integration.py`

**Implementation:**

```python
# tests/cluster/worker/test_integration.py
import pytest
import asyncio
import tempfile
import zipfile
from pathlib import Path
import cloudpickle

from fluster import cluster_pb2
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import VenvCache, ImageBuilder
from fluster.cluster.worker.runtime import DockerRuntime, ContainerConfig
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster.worker.server import WorkerServer
from fluster import cluster_pb2


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def test_bundle(tmp_path):
    """Create a minimal test bundle."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Minimal pyproject.toml
    (bundle_dir / "pyproject.toml").write_text('''
[project]
name = "test-job"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
''')

    # Create zip
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in bundle_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_dir))

    return f"file://{zip_path}"


def create_test_request(bundle_path: str) -> cluster_pb2.RunJobRequest:
    """Create a test job request."""
    def test_entrypoint():
        print("Hello from test job!")
        return 42

    from fluster.cluster.types import Entrypoint
    entrypoint = Entrypoint(callable=test_entrypoint)

    return cluster_pb2.RunJobRequest(
        job_id="test-job-123",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=bundle_path,
        environment=cluster_pb2.EnvironmentConfig(
            workspace="/app",
        ),
        resources=cluster_pb2.ResourceSpec(
            cpu=1,
            memory="512m",
        ),
    )


class TestBundleCache:
    """Test bundle downloading and caching."""

    async def test_download_local_bundle(self, cache_dir, test_bundle):
        """Test downloading a bundle from local file."""
        cache = BundleCache(cache_dir)
        bundle_path = await cache.get_bundle(test_bundle)
        assert bundle_path.exists()
        assert (bundle_path / "pyproject.toml").exists()

    async def test_cache_hit(self, cache_dir, test_bundle):
        """Test that second download uses cache."""
        cache = BundleCache(cache_dir)
        path1 = await cache.get_bundle(test_bundle)
        path2 = await cache.get_bundle(test_bundle)
        assert path1 == path2


class TestDockerRuntime:
    """Test Docker container execution."""

    @pytest.mark.docker
    async def test_run_simple_container(self):
        """Run a simple container and verify exit code."""
        runtime = DockerRuntime()
        result = await runtime.run(ContainerConfig(
            image="alpine:latest",
            command=["echo", "hello"],
            env={},
        ))
        assert result.exit_code == 0

    @pytest.mark.docker
    async def test_log_streaming(self):
        """Test log streaming from container."""
        runtime = DockerRuntime()
        logs = []

        def callback(stream, data):
            logs.append((stream, data))

        result = await runtime.run(
            ContainerConfig(
                image="alpine:latest",
                command=["echo", "test output"],
                env={},
            ),
            log_callback=callback,
        )
        assert result.exit_code == 0
        assert any("test output" in data for _, data in logs)

    @pytest.mark.docker
    async def test_resource_limits(self):
        """Verify cgroups v2 resource limits are applied."""
        runtime = DockerRuntime()
        result = await runtime.run(ContainerConfig(
            image="alpine:latest",
            command=["cat", "/sys/fs/cgroup/memory.max"],
            env={},
            memory_mb=128,
        ))
        assert result.exit_code == 0

    @pytest.mark.docker
    async def test_timeout(self):
        """Test timeout kills container."""
        runtime = DockerRuntime()
        result = await runtime.run(ContainerConfig(
            image="alpine:latest",
            command=["sleep", "60"],
            env={},
            timeout_seconds=1,
        ))
        assert result.error == "Timeout exceeded"


class TestPortAllocator:
    """Test port allocation."""

    async def test_allocate_ports(self):
        allocator = PortAllocator((40000, 40100))
        ports = await allocator.allocate(3)
        assert len(ports) == 3
        assert len(set(ports)) == 3  # All unique

    async def test_release_ports(self):
        allocator = PortAllocator((40000, 40010))
        ports1 = await allocator.allocate(5)
        await allocator.release(ports1)
        ports2 = await allocator.allocate(5)
        assert set(ports1) == set(ports2)  # Same ports reused


class TestJobManager:
    """Test full job lifecycle."""

    @pytest.fixture
    async def manager(self, cache_dir):
        bundle_cache = BundleCache(cache_dir)
        venv_cache = VenvCache(cache_dir, cache_dir / "uv")
        image_builder = ImageBuilder(cache_dir, registry="localhost:5000")
        runtime = DockerRuntime()
        port_allocator = PortAllocator()

        return JobManager(
            bundle_cache=bundle_cache,
            venv_cache=venv_cache,
            image_builder=image_builder,
            runtime=runtime,
            port_allocator=port_allocator,
            max_concurrent_jobs=2,
        )

    @pytest.mark.docker
    async def test_submit_and_poll(self, manager, test_bundle):
        """Test job submission and status polling."""
        request = create_test_request(test_bundle)
        job_id = await manager.submit_job(request)

        job = await manager.get_job(job_id)
        assert job is not None
        assert job.status in (cluster_pb2.JOB_STATE_PENDING, cluster_pb2.JOB_STATE_BUILDING, cluster_pb2.JOB_STATE_RUNNING)

    @pytest.mark.docker
    async def test_concurrent_limit(self, manager, test_bundle):
        """Test concurrent job limit is enforced."""
        # Submit 4 jobs to manager with max_concurrent=2
        requests = [create_test_request(test_bundle) for _ in range(4)]
        job_ids = [await manager.submit_job(r) for r in requests]

        await asyncio.sleep(0.5)  # Let jobs start

        jobs = await manager.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)
        assert running <= 2


class TestWorkerService:
    """Test RPC service end-to-end."""

    @pytest.fixture
    async def service(self, cache_dir):
        bundle_cache = BundleCache(cache_dir)
        venv_cache = VenvCache(cache_dir, cache_dir / "uv")
        image_builder = ImageBuilder(cache_dir, registry="localhost:5000")
        runtime = DockerRuntime()
        port_allocator = PortAllocator()

        manager = JobManager(
            bundle_cache=bundle_cache,
            venv_cache=venv_cache,
            image_builder=image_builder,
            runtime=runtime,
            port_allocator=port_allocator,
        )

        return WorkerServiceImpl(manager)

    @pytest.mark.docker
    async def test_health_check(self, service):
        """Test HealthCheck RPC."""
        from connectrpc import RequestContext
        response = await service.health_check(cluster_pb2.Empty(), RequestContext())
        assert response.healthy

    @pytest.mark.docker
    async def test_fetch_logs_tail(self, service, test_bundle):
        """Test FetchLogs with negative start_line for tailing."""
        from connectrpc import RequestContext
        ctx = RequestContext()

        # Submit a job first
        request = create_test_request(test_bundle)
        await service.run_job(request, ctx)

        # Fetch last 10 lines
        log_request = cluster_pb2.FetchLogsRequest(
            job_id=request.job_id,
            filter=cluster_pb2.FetchLogsFilter(start_line=-10),
        )
        response = await service.fetch_logs(log_request, ctx)
        assert isinstance(response.logs, list)
```

**Exit Conditions:**
- [ ] Bundle cache tests with local file paths
- [ ] Docker runtime tests with real containers
- [ ] Port allocator tests
- [ ] Job lifecycle tests end-to-end
- [ ] Concurrent job limit tests
- [ ] RPC service tests
- [ ] FetchLogs tail behavior tested
- [ ] All tests pass with `uv run pytest lib/fluster/tests/cluster/worker/ -m docker`

---

## Verification Plan

1. **Unit Tests (each stage):**
   - Run with `uv run pytest lib/fluster/tests/cluster/worker/`

2. **Integration Tests (Stage 11):**
   - Run with `uv run pytest lib/fluster/tests/cluster/worker/ -m docker`
   - Requires Docker daemon running

3. **Manual Testing:**
   - Start worker: `uv run python -m fluster.cluster.worker.main serve --registry localhost:5000`
   - Open dashboard: http://localhost:8080
   - Submit test job via Connect RPC client

4. **End-to-End:**
   - Create test bundle with simple Python script
   - Create zip and use `file://` path
   - Submit job via RPC
   - Verify execution and logs
   - Verify log tailing with negative start_line

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Log streaming | Batch via FetchLogs | Simpler; negative start_line enables tailing |
| cgroups | v2 only | Modern systems; no v1 fallback complexity |
| UV cache permissions | chown 1000:1000 | Standard container user; explicit call |
| Registry | Required CLI arg | No default; explicit configuration |
| Log rotation | None | Adequate disk space assumed |
| Environment source | Workspace only | Removed docker_image support for simplicity |
| Test bundles | Local file:// paths | GCS auth not required for tests |
| Connect RPC mounting | WorkerServiceASGIApplication | Generated code provides path property |
