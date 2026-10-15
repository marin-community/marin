# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize learned or curvature-weighted updates inside an audited Base support."""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import (
    TensorSupport,
    export_constrained,
    finalize_constrained_export,
    validate_audit_location,
)
from marin.merging.curvature import ota_merge_tensor
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.weight_merging.calibration_model import coefficient_group


def read_artifact(url: str) -> tuple[dict, str]:
    fs, path = filesystem_for(url)
    raw = fs.cat_file(path)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def blend_slice(
    base: torch.Tensor, donors: list[torch.Tensor], coefficients: torch.Tensor, offset: int, total: int
) -> torch.Tensor:
    """Apply the same packed-support chunk partition used during coefficient fitting."""
    result = base.flatten().float().clone()
    anchor = base.flatten().float()
    donor_values = [x.flatten().float() for x in donors]
    for chunk in range(coefficients.shape[0]):
        start = max(offset, total * chunk // coefficients.shape[0]) - offset
        end = min(offset + base.numel(), total * (chunk + 1) // coefficients.shape[0]) - offset
        if end <= start:
            continue
        for index, donor in enumerate(donor_values):
            result[start:end].add_(coefficients[chunk, index + 1] * (donor[start:end] - anchor[start:end]))
    return result.reshape(base.shape).to(base.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    base = CheckpointReader(CheckpointSource(**recipe["base"]))
    donors = [CheckpointReader(CheckpointSource(**source)) for source in recipe["donors"]]
    support = tuple(TensorSupport(item["name"], tuple(item["rows"])) for item in recipe["support"])
    rows = {item.name: item.rows for item in support}
    method = recipe["parameters"]["method"]
    provenance = {}
    coefficients = {}
    moments = []
    moment_rows = []
    if method == "expertpp":
        trained, digest = read_artifact(recipe["calibration_artifact"])
        if trained["phase"] != "chunks" or trained["support"] != recipe["support"]:
            raise ValueError("Completed coefficient fit must use the exact export support")
        coefficients = {key: torch.tensor(value, dtype=torch.float32) for key, value in trained["coefficients"].items()}
        for value in coefficients.values():
            if (
                value.shape[1] != 3
                or not torch.isfinite(value).all()
                or (value < 0).any()
                or not torch.allclose(value.sum(-1), torch.ones(value.shape[0]), atol=1e-6, rtol=0)
            ):
                raise ValueError("Invalid convex mixing coefficients")
        provenance[recipe["calibration_artifact"]] = digest
    elif method == "ota_ffg":
        for source in recipe["second_moments"]:
            artifact = prefix_join(source["path"], "complete.json")
            completed, digest = read_artifact(artifact)
            provenance[artifact] = digest
            moments.append(CheckpointReader(CheckpointSource(**source)))
            moment_rows.append(
                {item["name"]: {row: index for index, row in enumerate(item["rows"])} for item in completed["support"]}
            )
    else:
        raise ValueError(f"Unknown calibrated merge: {method}")

    def transform(name, row, value):
        values = [reader.tensor(name)[row] for reader in donors]
        if method == "expertpp":
            position = rows[name].index(row)
            return blend_slice(
                value,
                values,
                coefficients[coefficient_group(name)],
                position * value.numel(),
                len(rows[name]) * value.numel(),
            )
        statistics = [
            reader.tensor(name)[selection[name][row]] for reader, selection in zip(moments, moment_rows, strict=True)
        ]
        return ota_merge_tensor(
            value, values, statistics, density=recipe["parameters"]["density"], epsilon=recipe["parameters"]["epsilon"]
        )

    audit_fs, audit_path = filesystem_for(recipe["audit"])
    validate_audit_location(recipe["audit"], recipe["output"])
    if audit_fs.exists(audit_path):
        raise FileExistsError(recipe["audit"])
    start = time.monotonic()
    audit = export_constrained(
        base,
        recipe["output"],
        support,
        recipe["support_cap"],
        recipe["change_cap"],
        transform,
    )
    audit.update(recipe=recipe, calibration_artifact_hashes=provenance, elapsed_seconds=time.monotonic() - start)
    finalize_constrained_export(audit, recipe["audit"], recipe["change_cap"])


if __name__ == "__main__":
    main()
