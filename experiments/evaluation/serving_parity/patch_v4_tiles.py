# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply an idempotent, hash-guarded v4 tile experiment to a private uv cache."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PRISTINE_SHA256 = "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c"
OLD = "    match tpu_version:\n        case 5 | 6:\n"
NEW = """    match tpu_version:
        case 4:
            # Job-local v4 tiles validated by the Step12 v4-32 serving runs.
            bq_sz = bq_csz = 1 if case == RpaCase.DECODE else min(32, max_q)
            bkv_sz = min(512, max_kv)
            bkv_csz = min(128, max_kv)
        case 5 | 6:
"""


def patch(path: Path):
    source = path.read_text()
    pristine = source.replace(NEW, OLD)
    if hashlib.sha256(pristine.encode()).hexdigest() != PRISTINE_SHA256:
        raise ValueError("Unexpected kernel source; refusing experimental patch")
    expected = pristine.replace(OLD, NEW)
    if expected == pristine or pristine.count(OLD) != 1:
        raise ValueError("Expected exactly one tile selector")
    if source not in (pristine, expected):
        raise ValueError("Unexpected patched source")
    if source != expected:
        path.write_text(expected)
    return dict(
        kind="experimental_v4_tiles",
        arithmetic_changed=True,
        before_sha256=PRISTINE_SHA256,
        after_sha256=hashlib.sha256(expected.encode()).hexdigest(),
    )


def main():
    root = Path(importlib.util.find_spec("tpu_inference").origin).parent
    print(json.dumps(patch(root / "kernels/ragged_paged_attention/v3/kernel.py")), flush=True)
    command, *arguments = sys.argv[1:]
    if command == "python":
        os.execv(sys.executable, [sys.executable, *arguments])
    elif command == "vllm":
        os.execv(sys.executable, [sys.executable, str(Path(sys.executable).with_name("vllm")), *arguments])
    else:
        raise ValueError(f"Unsupported entrypoint {command}")


if __name__ == "__main__":
    main()
