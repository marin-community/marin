# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the producer's Verifyit source inside the declared grader image."""

import hashlib
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    expected = sys.argv.pop(1)
    manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != expected:
        raise ValueError("Verifyit runtime manifest checksum mismatch")
    for name, checksum in json.loads(manifest).items():
        path = root / name
        if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Verifyit runtime file checksum mismatch: {name}")
    sys.path.insert(0, str(root))
    grade = importlib.import_module("verifyit.grade")
    if grade.__file__ is None or Path(grade.__file__).resolve() != root / "verifyit/grade.py":
        raise ValueError("Verifyit imported outside the staged runtime")
    return grade.main()


if __name__ == "__main__":
    raise SystemExit(main())
