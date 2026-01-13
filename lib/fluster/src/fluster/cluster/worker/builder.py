# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Virtual environment and Docker image caching with UV support."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import xxhash

from fluster.cluster.worker.docker import DockerImageBuilder
from fluster.cluster.worker.worker_types import JobLogs


@dataclass
class VenvCacheEntry:
    """Cached venv metadata."""

    deps_hash: str
    created_at: float
    size_bytes: int


class VenvCache:
    """UV cache manager for dependency caching.

    UV handles dependency caching natively via BuildKit cache mounts.
    This class provides utilities for computing dependency hashes and
    ensuring proper permissions for container access.

    Permissions: The UV cache directory must be writable by container user
    (typically UID 1000). Call ensure_permissions() after creating.
    """

    def __init__(self, uv_cache_dir: Path):
        self._uv_cache_dir = uv_cache_dir
        self._uv_cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_permissions(self, uid: int = 1000, gid: int = 1000) -> None:
        """Ensure UV cache has correct permissions for container user.

        Attempts to chown the UV cache directory to the specified uid/gid.
        Silently ignores permission errors (e.g., on macOS or when not running as root).
        """
        try:
            shutil.chown(self._uv_cache_dir, uid, gid)
        except (PermissionError, OSError):
            # Cannot chown (e.g., macOS, running as non-root user)
            # This is expected in local development environments
            pass

        for path in self._uv_cache_dir.rglob("*"):
            try:
                shutil.chown(path, uid, gid)
            except (PermissionError, OSError):
                pass

    def compute_deps_hash(self, bundle_path: Path) -> str:
        """Compute hash from pyproject.toml + uv.lock."""
        h = xxhash.xxh64()
        for fname in ["pyproject.toml", "uv.lock"]:
            fpath = bundle_path / fname
            if fpath.exists():
                h.update(fpath.read_bytes())
        return h.hexdigest()


@dataclass
class BuildResult:
    """Result of Docker image build."""

    image_tag: str
    deps_hash: str
    build_time_ms: int
    from_cache: bool


DOCKERFILE_TEMPLATE = """FROM {base_image}

# Install git (required for git-based dependencies) and UV
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV
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
"""


class ImageCache:
    """Manages Docker image building with caching.

    Image tag: {registry}/fluster-job-{job_id}:{deps_hash[:8]}
    Uses Docker BuildKit cache mounts for UV cache sharing.

    Delegates actual Docker operations to DockerImageBuilder, keeping
    caching logic separate from container runtime specifics.
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
        self._docker = DockerImageBuilder(registry)

    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        job_logs: JobLogs | None = None,
    ) -> BuildResult:
        """Build Docker image for job.

        Returns cached image if deps_hash matches.
        """
        image_tag = f"{self._registry}/fluster-job-{job_id}:{deps_hash[:8]}"

        # Check if image exists locally
        if self._docker.exists(image_tag):
            if job_logs:
                job_logs.add("build", f"Using cached image: {image_tag}")
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
        self._docker.build(bundle_path, dockerfile, image_tag, job_logs)
        build_time_ms = int((time.time() - start) * 1000)

        self._evict_old_images()

        return BuildResult(
            image_tag=image_tag,
            deps_hash=deps_hash,
            build_time_ms=build_time_ms,
            from_cache=False,
        )

    def _evict_old_images(self) -> None:
        """Remove old fluster images when over limit."""
        pattern = f"{self._registry}/fluster-job-*"
        images = self._docker.list_images(pattern)

        if len(images) <= self._max_images:
            return

        # Sort by creation time and remove oldest
        images.sort(key=lambda x: x.created_at)
        for image in images[: len(images) - self._max_images]:
            self._docker.remove(image.tag)
