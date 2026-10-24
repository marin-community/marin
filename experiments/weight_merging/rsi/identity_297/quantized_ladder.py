# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one identity-specific quantized ladder arm with durable tensor audits."""

import argparse
import hashlib
import json
import logging
import resource
import time
from dataclasses import asdict, replace

import safetensors.torch
import torch
from marin.merging.checkpoint import METADATA_NAMES, CheckpointReader
from rigging.filesystem.buckets import filesystem_for

from experiments.weight_merging.rsi.identity_297.precision import ANCHOR, DONOR


def quantize(base: torch.Tensor, donor: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0:
        return base.clone()
    if alpha == 1:
        return donor.to(base.dtype)
    return (base.float() + alpha * (donor.float() - base.float())).to(base.dtype)


def audit(base: torch.Tensor, donor: torch.Tensor, merged: torch.Tensor, alpha: float) -> dict:
    result = {
        "numel": base.numel(),
        "changed": 0,
        "real_delta": 0,
        "erased": 0,
        "real_delta_l2_squared": 0.0,
        "actual_delta_l2_squared": 0.0,
        "error_l2_squared": 0.0,
    }
    for b, d, m in zip(
        base.reshape(-1).split(1048576), donor.reshape(-1).split(1048576), merged.reshape(-1).split(1048576), strict=True
    ):
        b, d, m = b.double(), d.double(), m.double()
        real = d != b
        result["real_delta"] += int(real.sum())
        result["changed"] += int((m != b).sum())
        result["erased"] += int((real & (m == b)).sum())
        result["real_delta_l2_squared"] += float((d - b).square().sum())
        result["actual_delta_l2_squared"] += float((m - b).square().sum())
        result["error_l2_squared"] += float((m - b - alpha * (d - b)).square().sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, choices=(0, 0.5, 0.75, 1), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--anchor-path", default=ANCHOR.path, help="Same pinned anchor, staged in the worker's region.")
    parser.add_argument("--donor-path", default=DONOR.path, help="Same pinned donor, staged in the worker's region.")
    args = parser.parse_args()
    torch.set_num_threads(16)
    start = time.monotonic()
    fs, prefix = filesystem_for(args.output.rstrip("/"))
    anchor_source = replace(ANCHOR, path=args.anchor_path)
    donor_source = replace(DONOR, path=args.donor_path)
    recipe = {
        "alpha": args.alpha,
        "anchor": asdict(anchor_source),
        "donor": asdict(donor_source),
        "code_revision": args.code_revision,
        "kind": "quantized ladder",
        "dtype": "anchor dtype: BF16 weights, persistent FP32 router biases",
        "endpoints": "alpha0 exact anchor; alpha1 exact donor cast to anchor dtype",
    }
    recipe_path = prefix + "/ladder_recipe.json"
    if fs.exists(recipe_path):
        assert json.loads(fs.cat_file(recipe_path)) == recipe, "Existing output has another recipe"
    else:
        assert not fs.exists(prefix) or not fs.ls(prefix), "Output must be empty"
        fs.makedirs(prefix, exist_ok=True)
        fs.pipe_file(recipe_path, json.dumps(recipe, indent=2).encode())
    if fs.exists(prefix + "/merge_manifest.json"):
        logging.info("Verified arm already complete")
        return
    fs.makedirs(prefix + "/audit_progress", exist_ok=True)
    anchor, donor = CheckpointReader(anchor_source), CheckpointReader(donor_source)
    assert anchor.weight_map.keys() == donor.weight_map.keys()
    names = sorted(anchor.weight_map, key=lambda name: (anchor.weight_map[name], name))
    objects, stats, weight_map = [], {}, {}
    total_size = 0

    def write_verified(name: str, data: bytes) -> dict:
        digest = hashlib.sha256(data).hexdigest()
        fs.pipe_file(prefix + "/" + name, data)
        check = fs.cat_file(prefix + "/" + name)
        assert len(check) == len(data) and hashlib.sha256(check).hexdigest() == digest, name
        return {"name": name, "bytes": len(data), "sha256": digest}

    for index, name in enumerate(names, 1):
        shard = f"model-{index:05d}-of-{len(names):05d}.safetensors"
        marker = prefix + f"/audit_progress/{index:05d}.json"
        if fs.exists(marker):
            entry = json.loads(fs.cat_file(marker))
            assert entry["key"] == name
            payload = fs.cat_file(prefix + "/" + shard)
            assert hashlib.sha256(payload).hexdigest() == entry["object"]["sha256"]
            del payload
        else:
            b, d = anchor.tensor(name), donor.tensor(name)
            assert b.shape == d.shape and d.dtype == torch.float32
            m = quantize(b, d, args.alpha)
            assert m.shape == b.shape and m.dtype == b.dtype and torch.isfinite(m).all()
            if args.alpha == 0:
                assert torch.equal(m, b)
            if args.alpha == 1:
                assert torch.equal(m, d.to(b.dtype))
            values = audit(b, d, m, args.alpha)
            entry = {
                "key": name,
                "stats": {**values, "shape": list(m.shape), "dtype": str(m.dtype)},
                "tensor_bytes": m.numel() * m.element_size(),
                "object": write_verified(shard, safetensors.torch.save({name: m}, metadata={"format": "pt"})),
            }
            fs.pipe_file(marker, json.dumps(entry).encode())
            del b, d, m
        objects.append(entry["object"])
        stats[name] = entry["stats"]
        total_size += entry["tensor_bytes"]
        weight_map[name] = shard
        logging.info("Verified tensor %d/%d %s", index, len(names), name)
    for name in METADATA_NAMES:
        source = anchor.path.rstrip("/") + "/" + name
        if anchor.fs.exists(source):
            objects.append(write_verified(name, anchor.fs.cat_file(source)))
    objects.append(
        write_verified(
            "model.safetensors.index.json",
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}).encode(),
        )
    )
    numeric_keys = (
        "numel",
        "changed",
        "real_delta",
        "erased",
        "real_delta_l2_squared",
        "actual_delta_l2_squared",
        "error_l2_squared",
    )
    totals = {key: sum(row[key] for row in stats.values()) for key in numeric_keys}
    manifest = {
        "recipe": recipe,
        "code_revision": args.code_revision,
        "verified": True,
        "objects": objects,
        "tensor_count": len(names),
        "total_size": total_size,
        "per_key_stats": stats,
        "totals": totals,
        "wall_seconds": time.monotonic() - start,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    fs.pipe_file(prefix + "/merge_manifest.json", json.dumps(manifest, indent=2, allow_nan=False).encode())
    logging.info("VERIFIED ladder alpha=%s totals=%s", args.alpha, totals)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
