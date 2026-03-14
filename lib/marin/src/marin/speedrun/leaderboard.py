# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Leaderboard data formatting utilities.
"""

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderboardEntry:
    run_name: str
    model_size: int
    total_training_time: float
    total_training_flops: float
    submitted_by: str
    run_timestamp: datetime.datetime
    results_filepath: str
    wandb_link: str | None = None
    eval_paloma_c4_en_bpb: float | None = None
