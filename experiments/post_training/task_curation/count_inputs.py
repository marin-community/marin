# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count pinned, locally staged HF parquet files without reading their data pages."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parquet_counts(snapshot: Path, revision: str, pattern: str) -> dict[str, int]:
    """Return file row counts after checking HF download metadata against the pin.

    ``snapshot`` is an HF local download directory, including its
    ``.cache/huggingface/download`` metadata. These are whole-file input counts;
    a pipeline's row selector can reduce them further.
    """
    paths = sorted(snapshot.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet inputs match {pattern!r} in {snapshot}")
    counts = {}
    for path in paths:
        relative = path.relative_to(snapshot)
        metadata = snapshot / ".cache/huggingface/download" / f"{relative}.metadata"
        downloaded_revision = metadata.read_text().splitlines()[0]
        if downloaded_revision != revision:
            raise ValueError(f"{relative}: expected revision {revision}, found {downloaded_revision}")
        counts[relative.as_posix()] = pq.read_metadata(path).num_rows
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--files", required=True, help="Parquet glob relative to the local HF snapshot")
    args = parser.parse_args()
    print(json.dumps(parquet_counts(args.snapshot, args.revision, args.files), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
