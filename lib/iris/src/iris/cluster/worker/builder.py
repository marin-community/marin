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

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from iris.cluster.worker.docker import DockerImageBuilder
from iris.cluster.worker.worker_types import TaskLogs

logger = logging.getLogger(__name__)


def _find_all_recursive(bundle_path: Path, pattern: str) -> list[Path]:
    return list(bundle_path.rglob(pattern))


class VenvCache:
    """Utility for computing dependency hashes for cache invalidation.

    UV handles dependency caching natively via BuildKit cache mounts with
    explicit global cache ID (iris-uv-global). This ensures all workspaces
    share the same BuildKit-managed cache for dependency reuse.

    This class provides utilities for computing dependency hashes from
    pyproject.toml and uv.lock files to determine when Docker image layers
    can be reused.
    """

    def compute_deps_hash(self, bundle_path: Path) -> str:
        h = hashlib.sha256()
        for fname in ["pyproject.toml", "uv.lock"]:
            at_least_one_found = False
            for fpath in _find_all_recursive(bundle_path, fname):
                if fpath.exists():
                    h.update(fpath.read_bytes())
                    at_least_one_found = True
            if not at_least_one_found:
                logger.warning(f"File {fname} not found inside {bundle_path}")
        return h.hexdigest()


@dataclass
class BuildResult:
    image_tag: str
    deps_hash: str
    build_time_ms: int
    from_cache: bool


class ImageProvider(Protocol):
    """Protocol for Docker image management."""

    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        task_logs: TaskLogs | None = None,
    ) -> BuildResult: ...

    def protect(self, tag: str) -> None:
        """Mark an image as protected from eviction (used by a running job)."""
        ...

    def unprotect(self, tag: str) -> None:
        """Remove protection from an image (job completed)."""
        ...


DOCKERFILE_TEMPLATE = """FROM {base_image}

# Install git (required for git-based dependencies) and UV
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# TODO -- install Cargo here.
# How do we make Rust stuff build faster?
# We could pre-build something similar or at least fetch Rust deps to cache?

# Configure UV
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

RUN --mount=type=cache,id=iris-uv-global,sharing=shared,target=/opt/uv-cache \\
    {pyproject_mounts} \\
    uv sync {extras_flags}

# Copy workspace contents
# Path dependencies referenced in [tool.uv.sources] must be present before uv sync.
# We copy everything upfront to support workspaces with local path dependencies.
COPY . .

# Use the venv python
ENV PATH="/app/.venv/bin:$PATH"

# Always install cloudpickle - required by iris to unpickle job entrypoints
RUN uv pip install cloudpickle
"""


class ImageCache:
    """Manages Docker image building with caching.

    Image tag: {registry}/iris-job-{job_id}:{deps_hash[:8]}
    Uses Docker BuildKit cache mounts with explicit global cache ID
    (iris-uv-global) to ensure all workspaces share the same UV cache.

    Cache behavior:
    - All builds use id=iris-uv-global for the UV cache mount
    - Different workspaces reuse cached dependencies automatically
    - BuildKit manages cache storage in /var/lib/buildkit/

    Delegates actual Docker operations to DockerImageBuilder, keeping
    caching logic separate from container runtime specifics.
    """

    def __init__(
        self,
        cache_dir: Path,
        registry: str,
        max_images: int = 100,
    ):
        self._cache_dir = cache_dir / "images"
        self._registry = registry
        self._max_images = max_images
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._docker = DockerImageBuilder(registry)
        # Refcount of images currently in use by jobs (should not be evicted)
        self._image_refcounts: dict[str, int] = {}

    def protect(self, tag: str) -> None:
        """Increment refcount for an image (job using it)."""
        self._image_refcounts[tag] = self._image_refcounts.get(tag, 0) + 1

    def unprotect(self, tag: str) -> None:
        """Decrement refcount for an image (job done with it)."""
        if tag in self._image_refcounts:
            self._image_refcounts[tag] -= 1
            if self._image_refcounts[tag] <= 0:
                del self._image_refcounts[tag]

    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        task_logs: TaskLogs | None = None,
    ) -> BuildResult:
        if self._registry:
            image_tag = f"{self._registry}/iris-job-{job_id}:{deps_hash[:8]}"
        else:
            image_tag = f"iris-job-{job_id}:{deps_hash[:8]}"

        # Check if image exists locally
        if self._docker.exists(image_tag):
            if task_logs:
                task_logs.add("build", f"Using cached image: {image_tag}")
            return BuildResult(
                image_tag=image_tag,
                deps_hash=deps_hash,
                build_time_ms=0,
                from_cache=True,
            )

        # Build image
        start = time.time()
        extras_flags = " ".join(f"--extra {e}" for e in extras) if extras else ""

        uv_locks_files = _find_all_recursive(bundle_path, "pyproject.toml") + _find_all_recursive(bundle_path, "uv.lock")
        if not uv_locks_files:
            logger.warning("No pyproject.toml or uv.lock files found in the bundle path")

        pyproject_mounts = " \\\n".join(
            f"--mount=type=bind,source={f.relative_to(bundle_path)},target={f.relative_to(bundle_path)}"
            for f in uv_locks_files
        )

        dockerfile = DOCKERFILE_TEMPLATE.format(
            base_image=base_image, extras_flags=extras_flags, pyproject_mounts=pyproject_mounts
        )
        self._docker.build(bundle_path, dockerfile, image_tag, task_logs)
        build_time_ms = int((time.time() - start) * 1000)

        self._evict_old_images()

        return BuildResult(
            image_tag=image_tag,
            deps_hash=deps_hash,
            build_time_ms=build_time_ms,
            from_cache=False,
        )

    def _evict_old_images(self) -> None:
        if self._registry:
            pattern = f"{self._registry}/iris-job-*"
        else:
            pattern = "iris-job-*"
        images = self._docker.list_images(pattern)

        if len(images) <= self._max_images:
            return

        # Filter out protected images (in use by running jobs)
        evictable = [img for img in images if img.tag not in self._image_refcounts]

        if len(evictable) <= self._max_images:
            return

        # Sort by created_at (oldest first), with tag as tiebreaker for determinism
        evictable.sort(key=lambda x: (x.created_at, x.tag))
        to_remove = evictable[: len(evictable) - self._max_images]
        for image in to_remove:
            self._docker.remove(image.tag)
