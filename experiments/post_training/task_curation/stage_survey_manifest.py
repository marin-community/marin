# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a survey manifest's remote input parquets onto the local worker disk.

The backend survey reads local Parquet files and records their digests, so
manifest entries naming object-store paths are copied beside their storage
region before the census runs. Entries with only local paths pass through.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def stage_entry(entry: dict, stage_dir: Path) -> dict:
    """Copy one input's remote paths into stage_dir and return the local entry."""
    if all(not path.startswith(("s3://", "gs://")) for path in entry["paths"]):
        return entry
    key = hashlib.sha256(entry["source_id"].encode()).hexdigest()[:20]
    target_dir = stage_dir / key
    target_dir.mkdir(parents=True, exist_ok=True)
    local = []
    for index, path in enumerate(entry["paths"]):
        target = target_dir / f"{index}-{Path(path).name}"
        with StoragePath(path).open("rb") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        local.append(str(target))
    return {**entry, "paths": local}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = json.loads(args.manifest.read_text())
    inputs = [stage_entry(entry, args.stage_dir) for entry in manifest["inputs"]]
    args.output.write_text(json.dumps({**manifest, "inputs": inputs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
