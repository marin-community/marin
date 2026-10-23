# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run and read back one iteration-one merge on a same-region CPU worker."""

import argparse
import hashlib
import json
import logging
import random
import resource
import time
from pathlib import Path

import safetensors.torch
import torch
from marin.merging.checkpoint import (
    METADATA_NAMES,
    CheckpointReader,
    CheckpointSource,
    merge_checkpoint,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.weight_merging.merge import merge_parameters

logger = logging.getLogger(__name__)


def checkpoint_specs(reader: CheckpointReader) -> dict:
    specs = {}
    for shard in sorted(set(reader.weight_map.values())):
        path = prefix_join(reader.path, shard)
        length = int.from_bytes(reader.fs.cat_file(path, start=0, end=8), "little")
        header = json.loads(reader.fs.cat_file(path, start=8, end=8 + length))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            assert reader.weight_map[name] == shard, name
            specs[name] = {"shape": spec["shape"], "dtype": spec["dtype"]}
    assert specs.keys() == reader.weight_map.keys()
    return specs


def verify(recipe: dict, manifest: dict) -> dict:
    anchor = CheckpointReader(CheckpointSource(**recipe["anchor"]))
    output = CheckpointReader(CheckpointSource(recipe["output"], manifest["code_revision"]))
    assert anchor.weight_map.keys() == output.weight_map.keys(), "Output keys differ"
    specs = checkpoint_specs(anchor)
    assert checkpoint_specs(output) == specs, "Output shapes/dtypes differ"
    metadata = {}
    for name in METADATA_NAMES:
        path = prefix_join(anchor.path, name)
        if anchor.fs.exists(path):
            original = anchor.fs.cat_file(path)
            assert output.fs.cat_file(prefix_join(output.path, name)) == original, name
            metadata[name] = hashlib.sha256(original).hexdigest()
    sampled = []
    donor = None
    if recipe["parameters"]["method"] == "average" and recipe["parameters"]["coefficients"] == [0.5]:
        sampled = random.Random(19301).sample(sorted(specs), 3)
        donor = CheckpointReader(CheckpointSource(**recipe["donors"][0]))
    objects = {entry["name"]: entry for entry in manifest["objects"]}
    stats = {}
    for index, name in enumerate(sorted(specs), 1):
        shard = output.weight_map[name]
        payload = output.fs.cat_file(prefix_join(output.path, shard))
        assert hashlib.sha256(payload).hexdigest() == objects[shard]["sha256"], name
        tensors = safetensors.torch.load(payload)
        assert set(tensors) == {name}
        tensor = tensors[name]
        del payload
        assert torch.isfinite(tensor).all(), name
        # Chunk reductions avoid another full FP64 copy of a large expert tensor.
        total = squared = 0.0
        minimum, maximum = float("inf"), float("-inf")
        for chunk in tensor.reshape(-1).split(1_048_576):
            values = chunk.double()
            total += values.sum().item()
            squared += values.square().sum().item()
            minimum = min(minimum, values.min().item())
            maximum = max(maximum, values.max().item())
        stats[name] = {
            **specs[name],
            "numel": tensor.numel(),
            "min": minimum,
            "max": maximum,
            "mean": total / tensor.numel(),
            "rms": (squared / tensor.numel()) ** 0.5,
            "sha256": objects[shard]["sha256"],
        }
        if name in sampled:
            assert donor is not None
            expected = ((anchor.tensor(name).float() + donor.tensor(name).float()) * 0.5).to(tensor.dtype)
            assert torch.equal(tensor, expected), f"Mean mismatch: {name}"
            stats[name]["exact_mean_check"] = True
            del expected
        del tensor, tensors
        logger.info("Verified %d/%d: %s", index, len(specs), name)
    return {"per_key_stats": stats, "metadata_sha256": metadata, "mean_check_keys": sampled, "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    recipe = json.loads(args.recipe.read_text())
    started = time.monotonic()
    manifest = merge_checkpoint(
        CheckpointSource(**recipe["anchor"]),
        [CheckpointSource(**source) for source in recipe["donors"]],
        recipe["output"],
        merge_parameters(recipe["parameters"]),
        code_revision=args.code_revision,
        preserve_rows={},
        tensor_coefficients={},
        row_overrides={},
    )
    merge_seconds = time.monotonic() - started
    verification = verify(recipe, manifest)
    report = {
        **manifest,
        **verification,
        "recipe": recipe,
        "merge_seconds": merge_seconds,
        "merge_and_verify_seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    output = CheckpointReader(CheckpointSource(recipe["output"], args.code_revision))
    payload = json.dumps(report, indent=2, allow_nan=False).encode()
    path = prefix_join(output.path, "merge_manifest.json")
    output.fs.pipe_file(path, payload)
    assert output.fs.cat_file(path) == payload
    logger.info(
        "VERIFIED COMPLETE %s: merge %.1fs; total %.1fs; peak RSS %d bytes",
        recipe["output"],
        merge_seconds,
        report["merge_and_verify_seconds"],
        report["process_peak_rss_bytes"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
