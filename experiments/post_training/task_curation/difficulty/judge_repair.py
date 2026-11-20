# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify an existing pinned judge route without substituting models."""

import argparse
import json
from pathlib import Path

from finestore.rl.difficulty import Solver
from iris.rpc.proto_display import priority_band_value

from experiments.post_training.task_curation.difficulty.runtime import (
    GLM_MODEL,
    judge_serving,
    probe_judge_endpoint,
    probe_with_transient_retries,
)
from experiments.post_training.task_curation.difficulty.state import GapReason
from experiments.post_training.task_curation.difficulty_sampling import sampling_settings
from experiments.post_training.task_curation.judges import JUDGE_TIERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    tiers = [tier for tier, definition in JUDGE_TIERS.items() if definition.spec.model == args.model]
    if args.model != GLM_MODEL and len(tiers) != 1:
        args.result.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": GapReason.JUDGE_ROUTE_UNQUALIFIED,
                    "detail": "Original judge has no registered immutable serving route",
                    "model": args.model,
                }
            )
        )
        return
    tier = tiers[0] if tiers else None
    with judge_serving(tier, priority_band_value("interactive")) as (endpoint, api_model, key, jobs):
        if tier is None:
            probe = probe_with_transient_retries(
                Solver.HOSTED, endpoint, api_model, key, sampling_settings(Solver.HOSTED, [])
            )
        else:
            definition = JUDGE_TIERS[tier]
            probe = probe_judge_endpoint(endpoint, definition, definition.spec.temperature, key)
        args.result.write_text(
            json.dumps({"status": "ready", "model": args.model, "probe": probe, "serving_jobs": jobs}, sort_keys=True)
        )


if __name__ == "__main__":
    main()
