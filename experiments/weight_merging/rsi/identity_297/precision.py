# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure scaled identity deltas before exporting full checkpoints."""

import argparse
import json
import logging
import math
import re
import resource
import time
from dataclasses import asdict
from pathlib import Path

import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, merge_tensor
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ANCHOR = CheckpointSource(
    "gs://marin-us-central2/scratch/karan/rsi-merge/parents-clean/"
    "v01--open-athena_Marin-Snowball-67B-A2B-v0.1@adc0078a19a4",
    "adc0078a19a4383bc998fcdb43867b3360baf5f3",
)
DONOR = CheckpointSource(
    "gs://marin-us-central2/scratch/karan/rsi-merge/parents-clean/"
    "v011--open-athena_Marin-Snowball-67B-A2B-v0.1.1@59852d6f0726",
    "59852d6f07261722bbec1edb6ab148317156b58e",
)


def measure_tensor(base: torch.Tensor, donor: torch.Tensor, chunk_elements: int) -> dict:
    assert base.shape == donor.shape
    assert base.dtype in (torch.bfloat16, torch.float32) and donor.dtype == torch.float32
    assert torch.isfinite(base).all() and torch.isfinite(donor).all()
    row = {"numel": base.numel(), "delta_nonzero": 0, "delta_l2_squared": 0.0, "base_l2_squared": 0.0}
    endpoints = {
        precision: {"different": 0, "over_one_ulp": 0, "max_abs_error": 0.0, "max_ulp_error": 0.0, "examples": []}
        for precision in ("fp32", "anchor_dtype")
    }
    reference_mismatches = 0
    arms = {
        f"{alpha:g}/{precision}": {
            "changed": 0,
            "erased": 0,
            "actual_delta_l2_squared": 0.0,
            "error_l2_squared": 0.0,
        }
        for alpha in ALPHAS
        for precision in ("fp32", "anchor_dtype")
    }
    for base_chunk, donor_chunk in zip(
        base.reshape(-1).split(chunk_elements), donor.reshape(-1).split(chunk_elements), strict=True
    ):
        b, d = base_chunk.float(), donor_chunk.float()
        delta = d.double() - b.double()
        real_delta = d != b
        row["delta_nonzero"] += torch.count_nonzero(real_delta).item()
        row["delta_l2_squared"] += delta.square().sum().item()
        row["base_l2_squared"] += b.double().square().sum().item()
        for alpha in ALPHAS:
            result = merge_tensor(
                b,
                [d],
                MergeParameters(MergeMethod.TASK_ARITHMETIC, (alpha,), 1, 1, 29701),
                tensor_name="precision_probe",
            )
            if alpha == 0:
                assert torch.equal(result, b), "FP32 alpha=0 changed the initializer"
            if alpha == 1:
                reference = (b.double() + delta.float().double()).float()
                reference_mismatches += torch.count_nonzero(result != reference).item()
            for precision, stored in (("fp32", result), ("anchor_dtype", result.to(base.dtype).float())):
                if alpha == 0:
                    assert torch.equal(stored, b), "Saved alpha=0 changed the initializer"
                if alpha == 1:
                    target = d if precision == "fp32" else d.to(base.dtype)
                    absolute = (stored.double() - target.double()).abs()
                    # Define one ULP as the spacing away from zero at the target.
                    magnitude = target.abs()
                    spacing = (
                        torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))).double()
                        - magnitude.double()
                    )
                    ulps = absolute / spacing
                    endpoint = endpoints[precision]
                    endpoint["different"] += torch.count_nonzero(absolute).item()
                    endpoint["over_one_ulp"] += torch.count_nonzero(ulps > 1).item()
                    endpoint["max_abs_error"] = max(endpoint["max_abs_error"], absolute.max().item())
                    endpoint["max_ulp_error"] = max(endpoint["max_ulp_error"], ulps.max().item())
                    remaining = 5 - len(endpoint["examples"])
                    if remaining:
                        for index in torch.nonzero(ulps > 1).flatten()[:remaining].tolist():
                            endpoint["examples"].append(
                                {
                                    "base": b[index].item(),
                                    "donor": d[index].item(),
                                    "stored": stored[index].item(),
                                    "target": target[index].item(),
                                    "abs_error": absolute[index].item(),
                                    "ulp_error": ulps[index].item(),
                                }
                            )
                metrics = arms[f"{alpha:g}/{precision}"]
                metrics["changed"] += torch.count_nonzero(stored != b).item()
                metrics["erased"] += torch.count_nonzero(real_delta & (stored == b)).item()
                actual_delta = stored.double() - b.double()
                metrics["actual_delta_l2_squared"] += actual_delta.square().sum().item()
                metrics["error_l2_squared"] += (actual_delta - alpha * delta).square().sum().item()
    return {
        **row,
        "arms": arms,
        "shape": list(base.shape),
        "dtype": str(base.dtype),
        "alpha_one_endpoints": endpoints,
        "fp32_reference_mismatches": reference_mismatches,
    }


def component(key: str) -> str:
    for marker, label in (
        (".self_attn.", "attention"),
        (".mlp.router.", "router"),
        (".mlp.experts.", "routed_experts"),
        (".shared_expert.", "shared_expert"),
        ("embed_tokens", "embeddings"),
        ("lm_head", "lm_head"),
    ):
        if marker in key:
            return label
    return "norm_and_gating"


def aggregate(rows: list[dict]) -> dict:
    totals = {
        field: sum(row[field] for row in rows)
        for field in ("numel", "delta_nonzero", "delta_l2_squared", "base_l2_squared")
    }
    totals["delta_nonzero_fraction"] = totals["delta_nonzero"] / totals["numel"]
    totals["delta_l2"] = math.sqrt(totals["delta_l2_squared"])
    arms = {}
    for name in rows[0]["arms"]:
        arm = {field: sum(row["arms"][name][field] for row in rows) for field in rows[0]["arms"][name]}
        alpha = float(name.split("/")[0])
        ideal_norm_squared = alpha**2 * totals["delta_l2_squared"]
        arm["fraction_changed"] = arm["changed"] / totals["numel"]
        arm["fraction_real_delta_erased"] = arm["erased"] / totals["delta_nonzero"] if totals["delta_nonzero"] else None
        arm["relative_l2_error"] = (
            math.sqrt(arm["error_l2_squared"] / ideal_norm_squared) if ideal_norm_squared else None
        )
        arm["actual_delta_l2"] = math.sqrt(arm["actual_delta_l2_squared"])
        arms[name] = arm
    endpoints = {}
    for precision in ("fp32", "anchor_dtype"):
        entries = [row["alpha_one_endpoints"][precision] for row in rows]
        endpoints[precision] = {
            **{field: sum(entry[field] for entry in entries) for field in ("different", "over_one_ulp")},
            **{field: max(entry[field] for entry in entries) for field in ("max_abs_error", "max_ulp_error")},
        }
    return {
        **totals,
        "arms": arms,
        "alpha_one_endpoints": endpoints,
        "fp32_reference_mismatches": sum(row["fp32_reference_mismatches"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    fs, output = filesystem_for(args.output)
    if fs.exists(output):
        raise FileExistsError(args.output)
    base, donor = CheckpointReader(ANCHOR), CheckpointReader(DONOR)
    assert base.weight_map.keys() == donor.weight_map.keys()
    donor_manifest = json.loads(donor.fs.cat_file(prefix_join(donor.path, "staging_manifest.json")))
    assert donor_manifest["revision"] == DONOR.revision
    assert donor.fs.exists(prefix_join(donor.path, ".cache_metadata.json"))
    specs = json.loads(Path(__file__).with_name("parent_headers.json").read_text())
    rows, groups = {}, {}
    for index, key in enumerate(sorted(base.weight_map, key=lambda k: (base.weight_map[k], k)), 1):
        row = measure_tensor(base.tensor(key), donor.tensor(key), args.chunk_elements)
        assert row["shape"] == specs["v0.1"]["specs"][key]["shape"]
        rows[key] = row
        if index == 1:
            fs.pipe_file(output + ".first-tensor.json", json.dumps({"key": key, **row}, allow_nan=False).encode())
        layer = re.search(r"model\.layers\.(\d+)\.", key)
        category = component(key)
        for group in ("component/" + category, "dtype/" + row["dtype"], "layer/" + (layer[1] if layer else "outside")):
            groups.setdefault(group, []).append(row)
        logger.info(
            "Measured %d/%d %s: %d/%d entries differ",
            index,
            len(base.weight_map),
            key,
            row["delta_nonzero"],
            row["numel"],
        )
    report = {
        "anchor": asdict(ANCHOR),
        "donor": asdict(DONOR),
        "code_revision": args.code_revision,
        "anchor_dtype_definition": "BF16 for 476 tensors; original FP32 retained for 26 router biases",
        "overall": aggregate(list(rows.values())),
        "groups": {name: aggregate(values) for name, values in groups.items()},
        "per_tensor": rows,
        "alpha_zero_exact_checks_passed": True,
        "alpha_one_policy": "Measure error against donor; do not accept or replace the arithmetic endpoint silently",
        "ulp_definition": "Spacing from abs(target) to its next representable value toward positive infinity",
        "wall_seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    payload = json.dumps(report, indent=2, allow_nan=False).encode()
    fs.pipe_file(output, payload)
    assert fs.cat_file(output) == payload
    logger.info("PRECISION AUDIT COMPLETE %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
