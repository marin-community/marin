# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute an immutable Snowball merge recipe on a CPU worker."""

import argparse
import json
import logging
from pathlib import Path

import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters
from marin.merging.checkpoint import CheckpointSource, RowMerge, merge_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    recipe = json.loads(args.recipe.read_text())
    parameters = recipe["parameters"]
    parameters["method"] = MergeMethod(parameters["method"])
    parameters["coefficients"] = tuple(parameters["coefficients"])
    merge_checkpoint(
        CheckpointSource(**recipe["anchor"]),
        [CheckpointSource(**source) for source in recipe["donors"]],
        recipe["output"],
        MergeParameters(**parameters),
        code_revision=args.code_revision,
        preserve_rows={name: tuple(rows) for name, rows in recipe["preserve_rows"].items()},
        tensor_coefficients={name: tuple(values) for name, values in recipe["tensor_coefficients"].items()},
        row_overrides={
            name: tuple(RowMerge(row=entry["row"], coefficients=tuple(entry["coefficients"])) for entry in entries)
            for name, entries in recipe.get("row_overrides", {}).items()
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
