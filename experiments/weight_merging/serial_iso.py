# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply DARE-TIES then ISO, auditing both stages against the original Base."""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
from marin.merging.arithmetic import MergeParameters, merge_tensor
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.constrained import (
    TensorSupport,
)
from marin.merging.isospectral import isospectral_merge

from experiments.weight_merging.merge import merge_parameters
from experiments.weight_merging.serial_merge import materialize_serial_stage


def transform_row(
    previous: CheckpointReader,
    donor: CheckpointReader,
    parameters: dict,
    arithmetic: MergeParameters | None,
    device: torch.device,
    name: str,
    row: int | None,
    _value: torch.Tensor,
) -> torch.Tensor:
    assert row is not None
    current = previous.tensor(name)[row]
    specialist = donor.tensor(name)[row]
    if parameters["method"] == "iso":
        result, _ = isospectral_merge(current, [specialist], parameters["keep"], device)
        return result
    assert arithmetic is not None
    return merge_tensor(current, [specialist], arithmetic, tensor_name=f"{name}:{row}")


def run(recipe: dict, device: torch.device) -> None:
    stages = recipe["stages"]
    if not stages or stages[-1]["output"] != recipe["output"] or stages[-1]["audit"] != recipe["audit"]:
        raise ValueError("Serial final output does not match experiment recipe")
    original = CheckpointReader(CheckpointSource(**recipe["base"]))
    donor = CheckpointReader(CheckpointSource(**recipe["donors"][0]))
    support = tuple(TensorSupport(item["name"], tuple(item["rows"])) for item in recipe["support"])
    previous = original
    for stage_index, stage in enumerate(stages):
        parameters = stage["parameters"]
        method = parameters["method"]
        arithmetic = None
        if method != "iso":
            arithmetic = merge_parameters(parameters)
        transform = partial(transform_row, previous, donor, parameters, arithmetic, device)
        audit = materialize_serial_stage(original, recipe, stage, stage_index + 1, support, transform)
        previous = CheckpointReader(CheckpointSource(stage["output"], f"{recipe['id']}:stage-{stage_index + 1}"))
        logging.info("Verified serial stage %d: %.6f%% changed", stage_index + 1, audit["changed_percent"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise ValueError("Serial ISO requires a reserved CUDA device")
    run(recipe, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
