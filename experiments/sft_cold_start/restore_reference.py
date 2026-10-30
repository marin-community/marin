# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore and verify the historical SFT source snapshots into an empty directory."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    reference = Path(__file__).parent / "reference"
    manifest = json.loads((reference / "manifest.json").read_text())
    for entry in manifest["files"]:
        source = reference / entry["snapshot"]
        content = source.read_bytes()
        if entry["encoding"] == "gzip":
            content = gzip.decompress(content)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Snapshot digest mismatch: {source}")
        destination = args.destination / entry["restore_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    print(f"Restored {len(manifest['files'])} verified source files to {args.destination}")


if __name__ == "__main__":
    main()
