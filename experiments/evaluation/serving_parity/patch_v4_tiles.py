"""Apply an idempotent, hash-guarded v4 tile experiment to a private uv cache."""

import argparse
import hashlib
from pathlib import Path

PRISTINE_SHA256 = "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c"
OLD = "    match tpu_version:\n        case 5 | 6:\n"
NEW = """    match tpu_version:
        case 4:
            # Task255 experimental v4 tiles: validate before serving weights.
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
    print("TASK255_KERNEL_SHA256", hashlib.sha256(expected.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("private_kernel", type=Path)
    args = parser.parse_args()
    patch(args.private_kernel)


if __name__ == "__main__":
    main()
