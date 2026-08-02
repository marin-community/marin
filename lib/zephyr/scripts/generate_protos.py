#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Zephyr protobuf and Connect files, then fix package imports."""

import re
import subprocess
import sys
from pathlib import Path


def fix_imports(file_path: Path) -> None:
    content = file_path.read_text()
    content = re.sub(
        r"^import (\w+_pb2) as (\w+__pb2)$",
        r"from . import \1 as \2",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^import (\w+_pb2) as (_\w+_pb2)$",
        r"from . import \1 as \2",
        content,
        flags=re.MULTILINE,
    )
    file_path.write_text(content)


def main() -> None:
    root = Path(__file__).parent.parent
    result = subprocess.run(
        ["npx", "--yes", "@bufbuild/buf", "generate"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    rpc_dir = root / "src" / "zephyr" / "rpc"
    for path in [*rpc_dir.glob("*_pb2.py"), *rpc_dir.glob("*_pb2.pyi"), *rpc_dir.glob("*_connect.py")]:
        fix_imports(path)


if __name__ == "__main__":
    main()
