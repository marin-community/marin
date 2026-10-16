# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only check that every observation id of a dataset build is a run id of the public W&B project.

`uv run python -m experiments.domain_phase_mix.release.verify_wandb_ids [--build release_build]`

Reads the id columns (`checks.id_columns`) of each table that the build's MANIFEST.json lists, asks W&B for those run
ids in batches, prints each column's found and missing counts and the first missing ids, and exits 1 when any id is
missing. It reads W&B with the local credentials and changes nothing.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb

from experiments.domain_phase_mix.release import checks, paths, tables

PROJECT = "marin-community/marin"
BATCH = 100  # run ids per W&B query
SHOWN = 20  # missing ids printed per column


def build_ids(build: Path) -> dict[tuple[str, str], set[str]]:
    """The distinct ids of each id column of the build at `build`, keyed by (table, column)."""
    manifest = json.loads((build / "MANIFEST.json").read_text())
    ids = {}
    for entry in manifest["tables"]:
        columns = checks.id_columns(entry["columns"])
        if not columns:
            continue
        path = build / entry["path"]
        if entry["format"] == "parquet":
            frame = pd.read_parquet(path, columns=columns)
        else:
            frame = tables.read_csv(path, usecols=columns, dtype=str)
        for column in columns:
            ids[(entry["table"], column)] = set(frame[column].dropna())
    return ids


def wandb_run_ids(api: wandb.Api, candidates: set[str]) -> set[str]:
    """The members of `candidates` that are run ids of PROJECT; W&B's run filter calls the run id `name`."""
    ordered, found = sorted(candidates), set()
    for start in range(0, len(ordered), BATCH):
        batch = ordered[start : start + BATCH]
        found |= {run.id for run in api.runs(PROJECT, filters={"name": {"$in": batch}}, per_page=BATCH)}
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, default=paths.BUILD)
    args = parser.parse_args(argv)
    ids = build_ids(args.build)
    every = set().union(*ids.values())
    found = wandb_run_ids(wandb.Api(timeout=120), every)
    for (table, column), values in ids.items():
        missing = sorted(values - found)
        print(f"{table}.{column}: {len(values)} ids, {len(values) - len(missing)} found, {len(missing)} missing")
        for run_id in missing[:SHOWN]:
            print(f"  missing: {run_id}")
    print(f"{len(every)} distinct ids, {len(every & found)} found in {PROJECT}, {len(every - found)} missing")
    raise SystemExit(1 if every - found else 0)


if __name__ == "__main__":
    main()
