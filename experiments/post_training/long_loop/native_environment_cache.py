# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Mochi native-control cache in the Marin CPU runtime."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from marin.inference.vllm_cache import VllmCompilationCache, VllmCompileIdentity
from rigging.filesystem.s3_compat import configure_coreweave_s3


def main() -> None:
    configure_coreweave_s3()
    profile = json.load(sys.stdin)
    argv = profile["native_cli"]
    cache = VllmCompilationCache.prepare(
        launcher_identity=profile["cache_launcher_identity"],
        compile_identity=VllmCompileIdentity.from_vllm_args(model_name_or_path=argv[1], extra_cli_args=tuple(argv[2:])),
        environment=dict(os.environ),
    )
    with tempfile.TemporaryDirectory(prefix="mochi-native-cache-") as temporary:
        source = Path(cache.environment()["VLLM_CACHE_ROOT"]).parent
        expected = Path(profile["values"]["VLLM_CACHE_ROOT"]).parent
        if source != expected:
            raise RuntimeError(f"Native cache identity mismatch: {source} != {expected}")
        staged = Path(temporary) / "cache"
        try:
            shutil.copytree(source, staged)
        finally:
            cache.close()
        shutil.copytree(staged, expected)
        files = [path for path in expected.rglob("*") if path.is_file()]
        print(
            json.dumps(
                {
                    "native_cache_restored": str(expected),
                    "python_executable": sys.executable,
                    "files": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
