# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Refresh the two-phase many-domain nextgen export with the standalone Olmix baseline."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os

import fsspec
import pandas as pd

from experiments.domain_phase_mix.nextgen.merge_export import (
    ExportDatasetConfig,
    MERGED_RUNS_JSON,
    MERGED_TRAJ_PARQUET,
    _merge_runs,
    _load_trajectories,
    export_dataset,
)
from experiments.domain_phase_mix.nextgen.state_store import STATE_FILE, load_loop_state, write_loop_state
from experiments.domain_phase_mix.nextgen.utils import loop_root_path
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_OBJECTIVE_METRIC,
    OLMIX_LOGLINEAR_SOURCE_EXPERIMENT,
    create_olmix_loglinear_import_source,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOP_NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240"
DEFAULT_STATE_ROOT = "gs://marin-us-east5/domain_phase_mix/nextgen"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-name", default=DEFAULT_LOOP_NAME)
    parser.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    parser.add_argument("--local-run-id", type=int, default=240)
    parser.add_argument("--source-experiment", default=OLMIX_LOGLINEAR_SOURCE_EXPERIMENT)
    parser.add_argument("--objective-metric", default=OLMIX_LOGLINEAR_OBJECTIVE_METRIC)
    return parser.parse_args()


def _write_merged_artifacts(root: str, merged_runs, merged_traj: pd.DataFrame) -> None:
    merge_dir = os.path.join(root, "merge")
    fs, _, _ = fsspec.get_fs_token_paths(merge_dir)
    fs.makedirs(merge_dir, exist_ok=True)

    with fsspec.open(os.path.join(merge_dir, MERGED_RUNS_JSON), "w") as f:
        import json

        json.dump([dataclasses.asdict(run) for run in merged_runs], f, indent=2, sort_keys=True)

    with fsspec.open(os.path.join(merge_dir, MERGED_TRAJ_PARQUET), "wb") as f:
        merged_traj.to_parquet(f, index=False)


def _merge_trajectories(prior_traj: pd.DataFrame, imported_traj: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in (prior_traj, imported_traj) if not frame.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "wandb_run_id",
                "source_experiment",
                "local_run_id",
                "run_name",
                "step",
                "total_tokens",
                "metric_key",
                "metric_value",
            ]
        )

    merged = pd.concat(non_empty, ignore_index=True)
    return merged.drop_duplicates(
        subset=["wandb_run_id", "source_experiment", "step", "metric_key"],
        keep="last",
    )


def main() -> None:
    args = _parse_args()
    root = loop_root_path(args.state_root, args.loop_name)
    state_path = os.path.join(root, "state", STATE_FILE)

    prior_state = load_loop_state(state_path, loop_name=args.loop_name, objective_metric=args.objective_metric)
    source = create_olmix_loglinear_import_source(
        local_run_id=args.local_run_id,
        source_experiment=args.source_experiment,
    )

    imported_runs = source.collect_runs()
    imported_traj = source.collect_trajectories(args.objective_metric)
    merged_runs = _merge_runs(prior=prior_state.runs, imported=imported_runs, new_runs=[])
    prior_traj = _load_trajectories(os.path.join(root, "merge", MERGED_TRAJ_PARQUET))
    merged_traj = _merge_trajectories(prior_traj, imported_traj)

    updated_state = dataclasses.replace(
        prior_state,
        next_local_run_id=max((run.local_run_id if run.local_run_id is not None else -1) for run in merged_runs) + 1,
        runs=merged_runs,
    )
    write_loop_state(state_path, updated_state)
    _write_merged_artifacts(root, merged_runs, merged_traj)
    export_dataset(
        ExportDatasetConfig(
            output_path=os.path.join(root, "export"),
            merged_output_path=os.path.join(root, "merge"),
        )
    )

    logger.info(
        "Refreshed loop %s with %s as local_run_id=%d; loop now has %d runs",
        args.loop_name,
        source.run_name,
        args.local_run_id,
        len(merged_runs),
    )


if __name__ == "__main__":
    main()
