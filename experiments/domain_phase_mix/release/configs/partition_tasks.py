# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The `partition` (39 buckets) and `tasks` (58 evaluations) configs."""

import pandas as pd

from experiments.domain_phase_mix.release import naming, paths


def export(ctx) -> dict[str, pd.DataFrame]:
    buckets = pd.read_csv(paths.require(paths.MIXSEL / "data/buckets.csv"))  # bucket,pool_tokens,epochs_per_unit_weight
    buckets["bucket_id"] = buckets.bucket.map(naming.bucket_public)
    names = naming.bucket_table()
    partition = names.merge(buckets[["bucket_id", "pool_tokens"]], on="bucket_id", how="left")
    assert partition.pool_tokens.notna().all(), "every bucket needs a pool size"
    # The bundle's epochs_per_unit_weight is 0.905353 * total / P_i (proportional epochs rounded to six decimals),
    # 5.0e-7 above the D_tgt / P_i that simulated epoching used, so the exact ratio is published instead.
    partition["epochs_per_unit_weight"] = paths.D_TGT / partition.pool_tokens
    partition = partition[
        ["bucket_id", "paper_name", "pool", "quality_tier", "pool_tokens", "epochs_per_unit_weight", "internal_name"]
    ]
    partition["partition_revision"] = paths.PARTITION_REVISION

    tasks = naming.task_table().copy()
    weights = naming.task_weights()
    aggregate_weight = {**weights["uncheatable_fixed"], **weights["obe_easy"]}
    tasks["aggregate_weight"] = [aggregate_weight[t] for t in tasks.task_id]
    return {"partition": partition, "tasks": tasks}
