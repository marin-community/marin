# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate saved feedback coverage; diagnostic only, without model calls or new thresholds.

Promoted verbatim from the ad-hoc ``s0-v12-observation-census.py`` worker script. Runs in
region under SkyRL's frozen CPU environment:

    … cpu-submit --skyrl --cpu 8 --memory 64GB --disk 128GB -- \\
      python -m experiments.post_training.mochi_loop_guards observation-census \\
      --manifest <resolved-manifest.json> --pattern <run output eval jsonl glob> \\
      --output <report.json>
"""

import argparse
import collections
import hashlib
import json
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

DIAGNOSTIC_ONLY = (
    "Aggregate coverage, not an additional validity threshold. Context-summary helper turns or invalid "
    "agent output can have no terminal feedback; inspect exceptions before inferring loss."
)


def main(argv: list[str] | None = None) -> None:
    # The vendored SkyRL scripts ship in SkyRL's frozen CPU environment (cpu-submit
    # --skyrl); they are not Marin workspace dependencies.
    from vendor.skyrl.scripts.loop_audit import read_rows, trial_turns, verdict_passed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    manifest = json.loads(StoragePath(args.manifest).read_text())
    membership = {r["task_id"]: c for c, definition in manifest["cohorts"].items() for r in definition["rows"]}
    census = {c: collections.Counter() for c in manifest["cohorts"]}
    examples = {c: [] for c in manifest["cohorts"]}
    for row in read_rows(args.pattern):
        cohort = membership.get(row.get("task_id"))
        if cohort is None:
            continue
        counts = census[cohort]
        turns = trial_turns(row.get("messages") or [])
        observed = sum(bool(turn.observation) for turn in turns)
        graded = verdict_passed(row) is not None
        counts["rows"] += 1
        counts["graded_rows"] += graded
        counts["assistant_turns"] += len(turns)
        counts["observation_batches"] += observed
        if len(turns) < 2:
            counts["fewer_than_two_turns"] += 1
            continue
        counts["multiturn_rows"] += 1
        counts["graded_multiturn_rows"] += graded
        counts["multiturn_with_observations"] += observed > 0
        counts["graded_multiturn_with_observations"] += graded and observed > 0
        counts["at_least_one_observation_per_between_turn_gap"] += observed >= len(turns) - 1
        if not observed:
            counts["multiturn_without_observations"] += 1
            counts["graded_multiturn_without_observations"] += graded
            if len(examples[cohort]) < 3:
                examples[cohort].append(
                    {
                        "task_id_sha256": hashlib.sha256(row["task_id"].encode()).hexdigest(),
                        "uid": row.get("uid"),
                        "ordinal": row.get("ordinal"),
                        "turns": len(turns),
                        "graded": graded,
                    }
                )
    report = {
        "run_output_pattern": args.pattern,
        "cohorts": census,
        "zero_observation_examples": examples,
        "model_trials_created": 0,
        "diagnostic_only": True,
        "interpretation": DIAGNOSTIC_ONLY,
    }
    StoragePath(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
