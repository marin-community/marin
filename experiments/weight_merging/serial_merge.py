# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize both serial stages and audit each against the original recipient."""

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import TensorSupport, export_constrained, validate_audit_location
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def blend_serial_row(previous, donor, selected, fraction, name, row, base):
    current = previous.tensor(name)[row]
    if row not in selected.get(name, set()):
        return current
    value = donor.tensor(name)[row]
    if fraction == 1:
        return value
    return (current.float() + fraction * (value.float() - current.float())).to(base.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    original = CheckpointReader(CheckpointSource(**recipe["base"]))
    donors = [CheckpointReader(CheckpointSource(**source)) for source in recipe["donors"]]
    stages = recipe["stages"]
    support = tuple(TensorSupport(item["name"], tuple(item["rows"])) for item in recipe["support"])
    previous = original
    for index, stage in enumerate(stages):
        selected = {item["name"]: set(item["rows"]) for item in stage["support"]}

        transform = partial(blend_serial_row, previous, donors[stage["donor"]], selected, stage["fraction"])
        output = stage["output"]
        audit_fs, audit_path = filesystem_for(stage["audit"])
        validate_audit_location(stage["audit"], output)
        if audit_fs.exists(audit_path):
            raise ValueError("Require a fresh audit outside the model directory")
        start = time.monotonic()
        audit = export_constrained(original, output, support, recipe["support_cap"], transform)
        if audit["changed_elements"] > int(recipe["change_cap"] * audit["total_elements"]):
            raise ValueError("Serial stage exceeds original Base change cap")
        audit.update(recipe=recipe, serial_stage=index + 1, elapsed_seconds=time.monotonic() - start)
        audit_fs.makedirs(str(StoragePath.parse(audit_path).parent), exist_ok=True)
        audit_fs.pipe_file(audit_path, json.dumps(audit, indent=2).encode())
        previous = CheckpointReader(CheckpointSource(output, f"{recipe['id']}:stage-{index+1}"))
    if stages[-1]["output"] != recipe["output"] or stages[-1]["audit"] != recipe["audit"]:
        raise ValueError("Serial final output does not match experiment recipe")


if __name__ == "__main__":
    main()
