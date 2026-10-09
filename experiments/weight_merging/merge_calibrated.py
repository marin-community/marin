# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize calibrated coefficients or curvature statistics as a static checkpoint."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters
from marin.merging.checkpoint import (
    CheckpointReader,
    CheckpointSource,
    ChunkMerge,
    CurvatureMerge,
    merge_checkpoint,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.weight_merging.calibration_model import coefficient_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    recipe = json.loads(args.recipe.read_text())
    anchor = CheckpointSource(**recipe["anchor"])
    donors = [CheckpointSource(**source) for source in recipe["donors"]]
    if recipe["method"] == "expertpp":
        fs, path = filesystem_for(recipe["calibration_artifact"])
        payload = fs.cat_file(path)
        trained = json.loads(payload)
        if trained["phase"] != "chunks":
            raise ValueError("Require completed chunk calibration")
        groups = trained["coefficients"]
        names = CheckpointReader(anchor).weight_map
        coefficients = {
            name: tuple(tuple(row) for row in groups[coefficient_group(name)])
            for name in names
            if coefficient_group(name) is not None
        }
        calibration = ChunkMerge(
            coefficients, recipe["block_elements"], recipe["calibration_artifact"], hashlib.sha256(payload).hexdigest()
        )
    elif recipe["method"] == "ota_ffg":
        moments = tuple(CheckpointSource(**source) for source in recipe["second_moments"])
        for source in moments:
            fs, path = filesystem_for(source.path)
            if not fs.exists(prefix_join(path, "complete.json")):
                raise ValueError(f"Second-moment collection is incomplete: {source.path}")
        calibration = CurvatureMerge(moments, recipe["density"], recipe["epsilon"])
    else:
        raise ValueError(f"Unknown calibrated method: {recipe['method']}")
    merge_checkpoint(
        anchor,
        donors,
        recipe["output"],
        MergeParameters(
            method=MergeMethod.TASK_ARITHMETIC, coefficients=tuple(0.0 for _ in donors), density=1, scale=1, seed=42
        ),
        code_revision=args.code_revision,
        preserve_rows={name: tuple(rows) for name, rows in recipe["preserve_rows"].items()},
        tensor_coefficients={},
        row_overrides={},
        calibration=calibration,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
