# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 MuonH Vizier LR/beta search with a 2x batch size."""

from __future__ import annotations

from dataclasses import replace

import experiments.grug.moe.muon_vizier_search as search

search.SWEEP = replace(
    search.SWEEP,
    experiment_name="moe-muonh-vizier-lr-beta-batch2x",
    search_space={
        "muonh_lr_multiplier": (0.5, 3.0),
        "adam_lr_multiplier": (0.5, 3.0),
        "momentum": (0.92, 0.99),
        "beta1": (0.70, 0.95),
        "beta2": (0.95, 0.999),
    },
    coefficient_type="quintic",
    base_train_tags=("moe", "muonh", "quintic", "vizier", "gate1-followup", "batch2x"),
    batch_multiplier=2,
    optimizer_family="muonh",
    matrix_lr_multiplier_name="muonh_lr_multiplier",
)


if __name__ == "__main__":
    search.executor_main(
        steps=[search._build_scale_chain(scale) for scale in search.SWEEP.scales],
        description="Grug MoE MuonH Vizier LR/beta search at exact gate-1 FLOP scales with 2x batch.",
    )
