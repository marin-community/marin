#!/usr/bin/env python3
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

"""Generate protobuf and Connect files, then fix imports."""

import re
import subprocess
import sys
from pathlib import Path


def fix_imports(file_path: Path) -> None:
    """Fix imports in generated Python files to use relative imports."""
    content = file_path.read_text()

    # Pattern 1: import <name>_pb2 as <name>__pb2 (used in _pb2.py files)
    # Replace with: from . import <name>_pb2 as <name>__pb2
    pattern1 = r"^import (\w+_pb2) as (\w+__pb2)$"
    replacement1 = r"from . import \1 as \2"

    # Pattern 2: import <name>_pb2 as _<name>_pb2 (used in _pb2.pyi files)
    # Replace with: from . import <name>_pb2 as _<name>_pb2
    pattern2 = r"^import (\w+_pb2) as (_\w+_pb2)$"
    replacement2 = r"from . import \1 as \2"

    new_content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    new_content = re.sub(pattern2, replacement2, new_content, flags=re.MULTILINE)

    if new_content != content:
        file_path.write_text(new_content)
        print(f"✓ Fixed imports in {file_path.name}")
    else:
        print(f"✓ No changes needed in {file_path.name}")


def run_buf_generate(root_dir: Path) -> None:
    """Run buf generate."""
    print("Running buf generate...")
    result = subprocess.run(
        ["buf", "generate"],
        cwd=root_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error running buf generate:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("✓ buf generate completed successfully")


def main():
    """Generate protobuf files and fix imports."""
    root_dir = Path(__file__).parent.parent
    rpc_dir = root_dir / "src" / "iris" / "rpc"

    # Run buf generate
    run_buf_generate(root_dir)

    # Fix imports in all generated Python files (both _pb2.py and _connect.py)
    print("\nFixing imports in generated files...")
    for pb2_file in rpc_dir.glob("*_pb2.py"):
        fix_imports(pb2_file)
    for connect_file in rpc_dir.glob("*_connect.py"):
        fix_imports(connect_file)
    for pyi_file in rpc_dir.glob("*_pb2.pyi"):
        fix_imports(pyi_file)

    print("\n✓ Generation complete!")


if __name__ == "__main__":
    main()
