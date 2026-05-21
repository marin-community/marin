#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect recent W&B runs that match the saved run id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

STATE_PATH = Path(__file__).resolve().parents[2] / "scratch" / "grug_moe_ep8_ragged_48l_fix_state.json"
PROJECT_PATH = "marin-community/dial_moe"


def _default_run_name() -> str:
    with STATE_PATH.open() as f:
        payload = json.load(f)
    return payload["run_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    run_name = args.run_name or _default_run_name()
    api = wandb.Api()
    runs = api.runs(PROJECT_PATH, order="-created_at", per_page=200)

    matches = []
    for run in runs:
        display_name = getattr(run, "display_name", None) or run.name
        if display_name == run_name or run.name == run_name:
            matches.append(
                {
                    "id": run.id,
                    "name": display_name,
                    "state": run.state,
                    "url": run.url,
                    "step": run.summary.get("global_step"),
                    "loss": run.summary.get("train/loss"),
                }
            )
        if len(matches) >= args.limit:
            break

    print(json.dumps(matches, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
