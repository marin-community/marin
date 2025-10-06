# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Leaderboard data formatting utilities.
"""

import datetime
from dataclasses import dataclass

import humanfriendly


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


def format_leaderboard(entries: list[LeaderboardEntry]) -> str:
    """
    This formats the leaderboard in a table that can be printed; not really needed but
    keeping for testing/sanity checks.
    """
    if not entries:
        return "No entries found."

    # Sort by FLOPs used (lower is better)
    entries.sort(key=lambda x: x.total_training_flops, reverse=True)

    header = "| Rank | Run Name | Timestamp (UTC) | Model Size | Training Time | FLOPs Used | C4-EN BPB |"
    separator = "|------|----------|----------------|------------|-------------------|-------------|---------|"

    rows = []
    for i, entry in enumerate(entries, 1):
        model_size_str = humanfriendly.format_size(entry.model_size, binary=False).replace("bytes", "params")
        training_time = humanfriendly.format_timespan(entry.total_training_time)
        flops_str = humanfriendly.format_number(entry.total_training_flops)
        c4_bpb = f"{entry.eval_paloma_c4_en_bpb:.3f}" if entry.eval_paloma_c4_en_bpb is not None else "N/A"
        timestamp = entry.run_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        row = (
            f"| {i} | {entry.run_name} | {timestamp} | {model_size_str} | " f"{training_time} | {flops_str} | {c4_bpb} |"
        )
        rows.append(row)

    return "\n".join([header, separator, *rows])
