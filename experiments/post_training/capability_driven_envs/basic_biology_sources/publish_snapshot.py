"""Publish a hash-verified, immutable private source snapshot; manifest last."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import fsspec


def manifest_files(root: Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Verify every declared HTML object before uploading anything."""
    data = (root / "manifest.json").read_bytes()
    manifest = json.loads(data)
    if manifest["status"] != "raw_official_html_not_cleared_for_model_input":
        raise ValueError("Only the explicitly unapproved raw-source archive may be published")
    rows = [*manifest["pages"], *manifest["rights_policy_snapshots"].values()]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate archive path")
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid archive path: {path}")
        content = (root / path).read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Archive object differs from manifest: {path}")
    return data, rows


def publish(root: Path, destination: str) -> None:
    """Copy source bytes only if absent/identical; commit manifest last."""
    if not destination.startswith("s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/us-gov-biology/"):
        raise ValueError("Destination must be the private, co-located government-biology source prefix")
    data, rows = manifest_files(root)
    base = destination.rstrip("/") + "/"
    for row in rows:
        uri = base + row["path"]
        fs, path = fsspec.core.url_to_fs(uri)
        if fs.exists(path):
            with fs.open(path, "rb") as source:
                existing = source.read()
            if hashlib.sha256(existing).hexdigest() != row["sha256"]:
                raise ValueError(f"Immutable archive collision: {uri}")
            continue
        with fs.open(path, "wb") as output:
            output.write((root / row["path"]).read_bytes())
    uri = base + "manifest.json"
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rb") as source:
            if source.read() != data:
                raise ValueError("Existing manifest differs: use a new snapshot prefix")
        return
    with fs.open(path, "wb") as output:
        output.write(data)
    print(f"Published {len(rows) - 2} source pages, 2 rights pages, and manifest to {base}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    publish(args.input, args.destination)


if __name__ == "__main__":
    main()
