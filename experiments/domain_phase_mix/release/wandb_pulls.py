# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Uncheatable metrics and data seeds read once from W&B run summaries and configs, and frozen in the repository.

A frozen pull is a CSV that opens with `#` comment lines, then has rows of `training_wandb_run_id, metric, value,
pulled_at`. Each run has its seven Uncheatable components (`eval/uncheatable_eval/<corpus>/bpb`), its pooled Uncheatable
aggregate (POOLED) and the data seed of its W&B config (DATA_SEED).
"""

from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import naming, paths, tables

POOLED = "eval/uncheatable_eval/bpb"
DATA_SEED = "data_seed"


def read(path: Path) -> pd.DataFrame:
    """One row per pulled run, indexed by its training run name: `bpb__<task>` for the seven Uncheatable corpora in task
    order, `pooled` (the run's pooled Uncheatable aggregate) and `data_seed`.

    Raises when a run lacks a metric or repeats one, or when a metric is none of these.
    """
    long = tables.read_csv(paths.require(path), comment="#")
    wide = long.pivot(index="training_wandb_run_id", columns="metric", values="value")  # raises on a repeated pair
    pulls = wide[[POOLED, DATA_SEED]].set_axis(["pooled", "data_seed"], axis=1)
    components = wide.drop(columns=[POOLED, DATA_SEED])
    components.columns = [f"bpb__{naming.task_public(c)}" for c in components.columns]
    uncheatable = [f"bpb__{t}" for t in naming.TASKS if t.startswith("uncheatable/")]
    assert sorted(components.columns) == sorted(uncheatable), f"pulled metrics {sorted(components.columns)}"
    assert components.notna().all().all() and pulls.notna().all().all(), "every pulled run needs every metric"
    return pd.concat([components[uncheatable], pulls], axis=1)
