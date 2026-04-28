# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 Muon Vizier LR/beta search with a 2x batch size."""

from __future__ import annotations

from dataclasses import replace

import experiments.grug.moe.muon_vizier_search as search

search.SWEEP = replace(
    search.SWEEP,
    experiment_name="moe-muon-vizier-lr-beta-batch2x",
    batch_multiplier=2,
    base_train_tags=("moe", "muon", "aol", "vizier", "gate1-followup", "batch2x"),
)


if __name__ == "__main__":
    search.executor_main(
        steps=[search._build_scale_chain(scale) for scale in search.SWEEP.scales],
        description="Grug MoE Muon Vizier LR/beta search at exact gate-1 FLOP scales with 2x batch.",
    )
