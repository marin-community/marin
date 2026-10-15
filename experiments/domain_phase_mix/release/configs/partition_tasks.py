# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The `partition` (39 buckets) and `tasks` (58 evaluations) configs."""

from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import naming, paths, tables

BUCKET_POOLS = paths.MIXSEL / "data/buckets.csv"  # bucket, pool_tokens, epochs_per_unit_weight
# The license that each pool's Hugging Face card declares (allenai/dolma3_pool, allenai/dolma3_dolmino_pool; read
# 2026-10-15, `unknown` for a card that declares none) and the paper's citation key for both pools, Olmo 3.
POOL_LICENSE = {"dolma3": "odc-by", "dolmino": "odc-by"}
POOL_CITATION_KEY = {"dolma3": "olmoOlmo32025", "dolmino": "olmoOlmo32025"}


def required_sources() -> list[Path]:
    """The files the export reads besides the context's (`context.required_sources`)."""
    return [BUCKET_POOLS]


def export(ctx) -> dict[str, pd.DataFrame]:
    buckets = tables.read_csv(paths.require(BUCKET_POOLS))
    buckets["bucket_id"] = buckets.bucket.map(naming.bucket_public)
    names = naming.bucket_table()
    partition = names.merge(buckets[["bucket_id", "pool_tokens"]], on="bucket_id", how="left")
    assert partition.pool_tokens.notna().all(), "every bucket needs a pool size"
    # The bundle's epochs_per_unit_weight is 0.905353 * total / P_i (proportional epochs rounded to six decimals),
    # 5.0e-7 above the D_tgt / P_i that simulated epoching used, so the exact ratio is published instead.
    partition["epochs_per_unit_weight"] = paths.D_TGT / partition.pool_tokens
    partition["pool_license"] = [POOL_LICENSE[pool] for pool in partition.pool]
    partition["pool_citation_key"] = [POOL_CITATION_KEY[pool] for pool in partition.pool]
    partition = partition[
        [
            "bucket_id",
            "paper_name",
            "pool",
            "pool_license",
            "pool_citation_key",
            "quality_tier",
            "pool_tokens",
            "epochs_per_unit_weight",
            "internal_name",
        ]
    ]
    partition["partition_revision"] = paths.PARTITION_REVISION

    tasks = naming.task_table().copy()
    weights = naming.task_weights()
    aggregate_weight = {**weights["uncheatable"], **weights["obe_easy"]}
    tasks["aggregate_weight"] = [aggregate_weight[t] for t in tasks.task_id]
    return {"partition": partition, "tasks": tasks}
