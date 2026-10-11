# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a bounded campaign merge and save its audit outside the model directory."""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, RamParameters, merge_tensor
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import TensorSupport, export_constrained
from marin.merging.isospectral import isospectral_merge
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    base = CheckpointReader(CheckpointSource(**recipe["base"]))
    donors = [CheckpointReader(CheckpointSource(**item)) for item in recipe["donors"]]
    support = tuple(TensorSupport(x["name"], None if x["rows"] is None else tuple(x["rows"])) for x in recipe["support"])
    params = recipe["parameters"].copy()
    iso = params["method"] == "iso"
    if iso:
        parameters = None
        if not torch.cuda.is_available():
            raise ValueError("Campaign ISO conversion requires its reserved CUDA device")
    else:
        params["method"] = MergeMethod(params["method"])
        params["coefficients"] = tuple(params["coefficients"])
        if params.get("ram") is not None:
            params["ram"] = RamParameters(**params["ram"])
        parameters = MergeParameters(**params)
    diagnostics = []
    measured_names = set()
    start = time.monotonic()
    audit_fs, audit_path = filesystem_for(recipe["audit"])
    if audit_fs.exists(audit_path):
        raise FileExistsError(recipe["audit"])
    if recipe["audit"].startswith(recipe["output"].rstrip("/") + "/"):
        raise ValueError("Audit must be outside the model files")

    def transform(name: str, row: int | None, value: torch.Tensor) -> torch.Tensor:
        values = [reader.tensor(name) for reader in donors]
        if row is not None:
            values = [tensor[row] for tensor in values]
        if "expert_assignment" in recipe:
            donor_index = recipe["expert_assignment"][name][str(row)]
            donor = values[donor_index]
            fraction = recipe["transfer_fraction"]
            if fraction == 1:
                return donor
            return (value.float() + fraction * (donor.float() - value.float())).to(value.dtype)
        if iso:
            device = torch.device("cuda:0")
            result, coefficients = isospectral_merge(value, values, params["keep"], device)
            diagnostic = {"tensor": name, "row": row, "coefficients": coefficients}
            if name not in measured_names:
                original = torch.linalg.svdvals(value.to(device=device, dtype=torch.float64))
                stored = torch.linalg.svdvals(result.to(device=device, dtype=torch.float64))
                diagnostic["stored_spectrum_relative_l2"] = (
                    torch.linalg.vector_norm(stored - original) / torch.linalg.vector_norm(original)
                ).item()
                measured_names.add(name)
            diagnostics.append(diagnostic)
            if len(diagnostics) % 32 == 0:
                logging.info("ISO matrices completed: %d", len(diagnostics))
            return result
        assert parameters is not None
        return merge_tensor(value, values, parameters, tensor_name=f"{name}:{row}")

    audit = export_constrained(base, recipe["output"], support, recipe["support_cap"], transform)
    if audit["changed_elements"] > int(recipe["change_cap"] * audit["total_elements"]):
        raise ValueError("Actual changed elements exceed the experiment cap")
    audit["recipe"] = recipe
    audit["parameters"] = params if iso else asdict(parameters)
    audit["elapsed_seconds"] = time.monotonic() - start
    if iso:
        audit["iso_diagnostics"] = diagnostics
    audit_fs.makedirs(str(StoragePath.parse(audit_path).parent), exist_ok=True)
    audit_fs.pipe_file(audit_path, json.dumps(audit, indent=2).encode())
    logging.info("Completed policy audit: %s; changed %.6f%%", recipe["audit"], audit["changed_percent"])


if __name__ == "__main__":
    main()
