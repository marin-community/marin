# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the v0.1.1 identity delta to Snowball v0.1 on attention q/k/v/o projections only, at alpha 1.

Each slice tensor becomes the FP32 donor tensor cast to the anchor dtype (BF16, round to nearest even), the alpha 1
rule of the identity merge ladder. Every other tensor keeps the anchor's bytes. The anchor's shard layout is kept:
a shard without slice tensors is copied byte for byte and checked by sha256 against the anchor's shard; a shard with
slice tensors is rewritten with the same tensors and metadata. The index and metadata files are byte-equal copies.

``merge-receipt.json`` is written last. Without it the export is incomplete. It records per-tensor
``export_sha256`` (sha256 of each payload, read back from the written shards) for every tensor.
"""

import argparse
import hashlib
import json
import logging
import resource
import struct
import time
from dataclasses import asdict

import safetensors.torch
import torch
from marin.merging.checkpoint import INDEX_NAME, METADATA_NAMES, CheckpointReader, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.weight_merging.identity_restricted.weight_delta_audit import (
    CHUNK_BYTES,
    payload_sha256,
    shard_records,
)

logger = logging.getLogger(__name__)
RECEIPT_NAME = "merge-receipt.json"
ANCHOR = CheckpointSource(
    "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a",
    "adc0078a19a4383bc998fcdb43867b3360baf5f3",
)
ANCHOR_INDEX_SHA256 = "7737eed0ad5faa748e709ae9fb7dfb1baded1e5a18b2d811cf13f69776b4869d"
DONOR = CheckpointSource(
    "s3://marin-rno2a/tmp/ttl=14d/quick-serve-models/"
    "open-athena_Marin-Snowball-67B-A2B-v0.1.1_59852d6f07261722bbec1edb6ab148317156b58e-bbd0a975340c90c9",
    "59852d6f07261722bbec1edb6ab148317156b58e",
)
VARIANTS = {"attn-qkvo-l18-23": range(18, 24), "attn-qkvo-all": range(26)}


def slice_for(variant: str) -> list[str]:
    return [f"model.layers.{layer}.self_attn.{p}_proj.weight" for layer in VARIANTS[variant] for p in "qkvo"]


def file_sha256(fs, path: str) -> str:
    digest = hashlib.sha256()
    with fs.open(path, "rb", cache_type="none", block_size=CHUNK_BYTES) as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def shard_metadata(data: bytes) -> dict[str, str] | None:
    (length,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + length]).get("__metadata__")


def merge_shard(anchor_bytes: bytes, donor: CheckpointReader, members: list[str]) -> tuple[bytes, dict]:
    tensors = safetensors.torch.load(anchor_bytes)
    stats = {}
    for name in members:
        base, d = tensors[name], donor.tensor(name)
        if d.dtype != torch.float32 or d.shape != base.shape:
            raise ValueError(f"Donor tensor is not FP32 with the anchor shape: {name}")
        merged = d.to(base.dtype)
        if not torch.isfinite(merged).all():
            raise ValueError(f"Nonfinite merged tensor: {name}")
        delta = merged.double() - base.double()
        stats[name] = {
            "numel": base.numel(),
            "changed": int((merged != base).sum()),
            "delta_l2": float(delta.square().sum().sqrt()),
            "base_l2": float(base.double().square().sum().sqrt()),
        }
        tensors[name] = merged
    return safetensors.torch.save(tensors, metadata=shard_metadata(anchor_bytes)), stats


def build(
    variant: str,
    output: str,
    code_revision: str,
    anchor: CheckpointSource,
    donor: CheckpointSource,
    anchor_index_sha256: str,
) -> dict:
    start = time.monotonic()
    names = slice_for(variant)
    fs, out = filesystem_for(output)
    if fs.exists(out) and fs.ls(out):
        raise ValueError(f"Output must be empty: {output}")
    fs.makedirs(out, exist_ok=True)
    anchor_reader, donor_reader = CheckpointReader(anchor), CheckpointReader(donor)
    anchor_fs, anchor_root = anchor_reader.fs, anchor_reader.path
    index_bytes = anchor_fs.cat_file(prefix_join(anchor_root, INDEX_NAME))
    if hashlib.sha256(index_bytes).hexdigest() != anchor_index_sha256:
        raise ValueError("Anchor index differs from its pin")
    donor_meta = json.loads(donor_reader.fs.cat_file(prefix_join(donor_reader.path, ".cache_metadata.json")))
    if donor_meta.get("source_revision") != donor.revision:
        raise ValueError("Donor copy is not the pinned revision")
    if anchor_reader.weight_map.keys() != donor_reader.weight_map.keys():
        raise ValueError("Anchor and donor tensor names differ")
    missing = sorted(set(names) - anchor_reader.weight_map.keys())
    if missing:
        raise ValueError(f"Slice names not in the anchor: {missing}")

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(anchor_reader.weight_map[name], []).append(name)
    objects, slice_stats = {}, {}
    for shard in sorted(set(anchor_reader.weight_map.values())):
        source_path, target_path = prefix_join(anchor_root, shard), prefix_join(out, shard)
        if shard in by_shard:
            data, stats = merge_shard(anchor_fs.cat_file(source_path), donor_reader, sorted(by_shard[shard]))
            slice_stats.update(stats)
            fs.pipe_file(target_path, data)
            expected, kind = hashlib.sha256(data).hexdigest(), "rewritten"
            del data
        else:
            fs.copy(source_path, target_path)
            expected, kind = file_sha256(anchor_fs, source_path), "copied"
        actual = file_sha256(fs, target_path)
        if actual != expected:
            raise ValueError(f"Readback of {shard} differs: {actual} != {expected}")
        objects[shard] = {"kind": kind, "bytes": fs.size(target_path), "sha256": actual}
        logger.info("Verified %s shard %s", kind, shard)

    metadata_equal = {}
    for name in (*METADATA_NAMES, INDEX_NAME):
        source_path = prefix_join(anchor_root, name)
        if not anchor_fs.exists(source_path):
            continue
        data = anchor_fs.cat_file(source_path)
        fs.pipe_file(prefix_join(out, name), data)
        readback = fs.cat_file(prefix_join(out, name))
        metadata_equal[name] = readback == data
        objects[name] = {"kind": "metadata", "bytes": len(readback), "sha256": hashlib.sha256(readback).hexdigest()}
    if not all(metadata_equal.values()):
        raise ValueError(f"Metadata readback differs: {metadata_equal}")

    records = shard_records(output)
    dtypes = {}
    for record in records.values():
        dtypes[record["dtype"]] = dtypes.get(record["dtype"], 0) + 1
    if records.keys() != anchor_reader.weight_map.keys():
        raise ValueError("Written tensors differ from the anchor's")
    tensors = {
        name: {"dtype": r["dtype"], "shape": r["shape"], "export_sha256": payload_sha256(fs, r)}
        for name, r in sorted(records.items())
    }
    unchanged = sorted(n for n, s in slice_stats.items() if s["changed"] == 0)
    receipt = {
        "variant": variant,
        "alpha": 1,
        "rule": "slice tensor = FP32 donor cast to anchor dtype (round to nearest even); all other tensors anchor bytes",
        "anchor": asdict(anchor),
        "anchor_index_sha256": anchor_index_sha256,
        "donor": asdict(donor),
        "code_revision": code_revision,
        "slice": sorted(names),
        "slice_unchanged_after_cast": unchanged,
        "slice_stats": slice_stats,
        "header_counts": {"tensors": len(records), "dtypes": dict(sorted(dtypes.items()))},
        "metadata_byte_equal": metadata_equal,
        "objects": objects,
        "tensors": tensors,
        "wall_seconds": time.monotonic() - start,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    fs.pipe_file(prefix_join(out, RECEIPT_NAME), json.dumps(receipt, indent=1, sort_keys=True).encode())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    receipt = build(args.variant, args.output, args.code_revision, ANCHOR, DONOR, ANCHOR_INDEX_SHA256)
    logger.info(
        "VERIFIED %s: %s, slice unchanged after cast %s, %.0f s",
        args.variant,
        receipt["header_counts"],
        receipt["slice_unchanged_after_cast"],
        receipt["wall_seconds"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
