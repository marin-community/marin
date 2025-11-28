#!/usr/bin/env python3
"""Check which files from prune.txt are actually imported."""

import subprocess
import sys

with open("prune.txt") as f:
    files = [line.strip() for line in f if line.strip()]

imported = []
not_imported = []

for file_path in files:
    # Convert file path to module name
    module = file_path.replace(".py", "").replace("/", ".")

    # Search for imports
    result = subprocess.run(
        ["grep", "-r", f"from {module}\\|import {module}", "--include=*.py", "."],
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and result.stdout.strip():
        imported.append(file_path)
        print(f"IMPORTED: {file_path}")
    else:
        not_imported.append(file_path)

print(f"\n\n=== SUMMARY ===")
print(f"Imported (MUST RESTORE): {len(imported)}")
print(f"Not imported (CAN REMOVE): {len(not_imported)}")

print(f"\n\n=== FILES TO RESTORE ===")
for f in imported:
    print(f)
