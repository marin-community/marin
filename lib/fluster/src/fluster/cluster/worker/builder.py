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

"""Virtual environment and Docker image builder with caching."""

import asyncio
import os
import shlex
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import xxhash


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
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"uv sync failed: {stderr.decode()}")

    async def _archive_venv(self, venv_path: Path, deps_hash: str) -> None:
        """Compress venv with zstd for storage."""
        archive_path = self._cache_dir / f"{deps_hash}.tar.zst"
        # Use zstd with parallel threads for speed
        cmd = f"tar -cf - -C {shlex.quote(str(venv_path.parent))} .venv | zstd -T0 -o {shlex.quote(str(archive_path))}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to archive venv: {stderr.decode()}")

    async def _extract_venv(self, archive: Path, target_bundle: Path) -> Path:
        """Extract cached venv to target bundle."""
        cmd = f"zstd -d -c {shlex.quote(str(archive))} | tar -xf - -C {shlex.quote(str(target_bundle))}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to extract venv: {stderr.decode()}")
        return target_bundle / ".venv"

    async def _evict_lru(self) -> None:
        """Remove oldest entries when over max_entries."""
        archives = list(self._cache_dir.glob("*.tar.zst"))
        if len(archives) <= self._max_entries:
            return

        archives.sort(key=lambda p: p.stat().st_mtime)
        for path in archives[: len(archives) - self._max_entries]:
            path.unlink()


@dataclass
class BuildResult:
    """Result of Docker image build."""

    image_tag: str
    deps_hash: str
    build_time_ms: int
    from_cache: bool


DOCKERFILE_TEMPLATE = """FROM {base_image}

# Install UV
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

        try:
            cmd = [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                tag,
                "--build-arg",
                "BUILDKIT_INLINE_CACHE=1",
                str(context),
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Docker build failed: {stderr.decode()}")
        finally:
            # Cleanup generated dockerfile
            dockerfile_path.unlink(missing_ok=True)

    async def _image_exists(self, tag: str) -> bool:
        """Check if image exists locally."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            tag,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0

    async def _evict_old_images(self) -> None:
        """Remove old fluster images when over limit."""
        # List images matching our pattern
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "images",
            "--format",
            "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
            "--filter",
            f"reference={self._registry}/fluster-job-*",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()

        # Filter empty lines
        lines = [line for line in stdout.decode().strip().split("\n") if line]
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
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rmi",
                tag,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
