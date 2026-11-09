"""Read the frozen PDBThink evaluation bundle without exposing gold in source control."""

import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

PDBTHINK_REVISION = "c00271fe049d60dc0d671b999d8c2fb595086166"
BUNDLE_PATH = f"Open-Athena/pdbthink/{PDBTHINK_REVISION}/evaluation.json.gz"
MANIFEST_PATH = Path(__file__).with_name("manifest.json")


@cache
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def load_bundle(cache_prefix: str) -> dict:
    """Verify the prepared bundle before returning evaluator-only records."""
    path = prefix_join(cache_prefix, BUNDLE_PATH)
    try:
        with open_url(path, "rb") as stream:
            compressed = stream.read()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"PDBThink has not been prepared at {path}. "
            "Run scripts/benchmarks/prepare_pdbthink.py as described in docs/explanations/pdbthink.md."
        ) from error
    expected = manifest()["bundle_sha256"]
    if hashlib.sha256(compressed).hexdigest() != expected:
        raise ValueError(f"PDBThink bundle checksum does not match the frozen manifest: {path}")
    return json.loads(gzip.decompress(compressed))
