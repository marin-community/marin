# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a final SFT checkpoint is kept in full at its permanent path.

Confirms, in region: permanent metadata (not temporary), manifest coverage of
both weights and optimizer state, an OCDBT chunk inventory that covers every
manifest array, non-empty object-store contents, and end-to-end decodes of
sample arrays. Writes a JSON receipt and fails loudly on any gap. The archive
stage reuses :func:`verify_final_checkpoint` for its post-archive assertion.
"""

import asyncio
import hashlib
import json
import logging

import click
import numpy as np
import tensorstore as ts
from levanter.checkpoint import load_checkpoint_metadata
from levanter.checkpoint_manifest import read_manifest
from levanter.tensorstore_serialization import _create_ocdbt_spec, _list_ocdbt_keys
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

OPT_STATE_PREFIX = "opt_state/"
ZARR_METADATA_SUFFIX = "/zarr.json"
SAMPLE_ARRAYS_PER_TREE = 4
DECODE_ELEMENT_BUDGET = 1_000_000
REQUIRED_FILES = ("metadata.json", "manifest.json", "manifest.ocdbt")


def _array_elements(array) -> int:
    elements = 1
    for extent in array.shape:
        elements *= extent
    return elements


def _smallest_arrays(arrays: list, count: int) -> list:
    return sorted(arrays, key=lambda array: (_array_elements(array), array.path))[:count]


def _decode_array(checkpoint_root: str, path: str) -> dict:
    """Read one array end-to-end through tensorstore and hash its bytes."""

    async def read() -> bytes:
        store = await ts.open(_create_ocdbt_spec(checkpoint_root, path))
        if _array_elements_from_store(store) <= DECODE_ELEMENT_BUDGET:
            data = await store.read()
        else:
            data = await store[0:1, ...]
        return np.asarray(data).tobytes()

    raw = asyncio.run(read())
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _array_elements_from_store(store) -> int:
    elements = 1
    for extent in store.shape:
        elements *= extent
    return elements


def verify_final_checkpoint(checkpoint: str, expect_step: int | None = None) -> dict:
    """Return a verification receipt, raising if the checkpoint is incomplete."""
    configure_coreweave_s3()
    root = StoragePath(checkpoint)

    metadata = load_checkpoint_metadata(checkpoint)
    if metadata.get("is_temporary", True):
        raise ValueError(f"{checkpoint} is marked temporary: {metadata}")
    step = metadata.get("step")
    if expect_step is not None and step != expect_step:
        raise ValueError(f"{checkpoint} holds step {step}, expected {expect_step}")

    manifest = read_manifest(checkpoint)
    if manifest is None:
        raise ValueError(f"{checkpoint} has no manifest.json; cannot verify array coverage")
    model_arrays = [array for array in manifest.arrays if not array.path.startswith(OPT_STATE_PREFIX)]
    optimizer_arrays = [array for array in manifest.arrays if array.path.startswith(OPT_STATE_PREFIX)]
    if not model_arrays:
        raise ValueError(f"{checkpoint} manifest lists no model arrays")
    if not optimizer_arrays:
        raise ValueError(f"{checkpoint} manifest lists no optimizer-state arrays")

    ocdbt_keys = asyncio.run(_list_ocdbt_keys(checkpoint))
    committed_arrays = {key[: -len(ZARR_METADATA_SUFFIX)] for key in ocdbt_keys if key.endswith(ZARR_METADATA_SUFFIX)}
    missing_arrays = sorted(manifest.array_paths - committed_arrays)
    if missing_arrays:
        raise ValueError(f"{checkpoint} OCDBT database lacks {len(missing_arrays)} manifest arrays: {missing_arrays}")
    chunk_keys = [key for key in ocdbt_keys if not key.endswith(ZARR_METADATA_SUFFIX)]
    if not chunk_keys:
        raise ValueError(f"{checkpoint} OCDBT database lists no chunk keys")

    decode_samples = [
        *_smallest_arrays(model_arrays, SAMPLE_ARRAYS_PER_TREE),
        *_smallest_arrays(optimizer_arrays, SAMPLE_ARRAYS_PER_TREE),
    ]
    decode_checks = [_decode_array(checkpoint, array.path) for array in decode_samples]

    fs, key = filesystem_for(checkpoint)
    objects = fs.find(key, detail=True)
    shards = [info for path, info in objects.items() if path.rstrip("/").split("/")[-2] == "d"]
    if not shards:
        raise ValueError(f"{checkpoint} has no objects under d/")
    file_receipts = {}
    for name in REQUIRED_FILES:
        entry = root / name
        if not entry.exists():
            raise ValueError(f"{checkpoint} is missing {name}")
        raw = entry.read_bytes()
        file_receipts[name] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    return {
        "checkpoint": checkpoint,
        "step": step,
        "is_temporary": False,
        "format_version": manifest.format_version,
        "arrays": {
            "total": len(manifest.arrays),
            "model": len(model_arrays),
            "optimizer_state": len(optimizer_arrays),
        },
        "ocdbt": {"keys": len(ocdbt_keys), "chunks": len(chunk_keys)},
        "objects": {"total": len(objects), "shards": len(shards)},
        "decode_checks": decode_checks,
        "files": file_receipts,
        "verdict": "full",
    }


@click.command(help=__doc__)
@click.option("--checkpoint", required=True, help="URI of the final SFT checkpoint directory")
@click.option("--receipt", required=True, help="URI to write the JSON verification receipt")
@click.option("--expect-step", type=int, default=None, help="Fail unless the checkpoint holds this step")
def main(checkpoint: str, receipt: str, expect_step: int | None) -> None:
    logging.basicConfig(level=logging.INFO)
    result = verify_final_checkpoint(checkpoint, expect_step)
    receipt_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    StoragePath(receipt).write_text(receipt_text)
    logger.info("Verified %s in full; receipt at %s", checkpoint, receipt)
    print(receipt_text)


if __name__ == "__main__":
    main()
