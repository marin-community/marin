# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Mochi native-control cache in the Marin CPU runtime."""

import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from marin.inference.vllm_cache import VllmCompilationCache, VllmCompileIdentity
from rigging.filesystem.s3_compat import configure_coreweave_s3

CACHE_ROOT_ENV = "VLLM_CACHE_ROOT"


@dataclass(frozen=True)
class CachePreparation:
    launcher_identity: str
    native_cli: tuple[str, ...]
    expected_cache: Path


@dataclass(frozen=True)
class CacheReceipt:
    native_cache_restored: str
    python_executable: str
    files: int
    bytes: int


def restore_cache(request: CachePreparation) -> CacheReceipt:
    """Restore compiler caches to the requested native path."""
    cache = VllmCompilationCache.prepare(
        launcher_identity=request.launcher_identity,
        compile_identity=VllmCompileIdentity.from_vllm_args(
            model_name_or_path=request.native_cli[1], extra_cli_args=request.native_cli[2:]
        ),
        environment=dict(os.environ),
    )
    with tempfile.TemporaryDirectory(prefix="mochi-native-cache-") as temporary:
        staged = Path(temporary) / "cache"
        try:
            source = Path(cache.environment()[CACHE_ROOT_ENV]).parent
            if source != request.expected_cache:
                raise RuntimeError(f"Native cache identity mismatch: {source} != {request.expected_cache}")
            shutil.copytree(source, staged)
        finally:
            cache.close()
        shutil.copytree(staged, request.expected_cache)
    files = [path for path in request.expected_cache.rglob("*") if path.is_file()]
    return CacheReceipt(
        native_cache_restored=str(request.expected_cache),
        python_executable=sys.executable,
        files=len(files),
        bytes=sum(path.stat().st_size for path in files),
    )


def main() -> None:
    configure_coreweave_s3()
    profile = json.load(sys.stdin)
    request = CachePreparation(
        launcher_identity=profile["cache_launcher_identity"],
        native_cli=tuple(profile["native_cli"]),
        expected_cache=Path(profile["values"][CACHE_ROOT_ENV]).parent,
    )
    print(json.dumps(asdict(restore_cache(request))), flush=True)


if __name__ == "__main__":
    main()
