# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional, streaming S0 counterfactual and bounded deterministic review export.

Promoted verbatim from the ad-hoc ``s0-counterfactual-regional.py`` worker script.
Run it in region through the generic CPU path; the vendored SkyRL helpers ship in
SkyRL's frozen CPU environment, not the Marin workspace:

    … cpu-submit --skyrl --cpu 16 --memory 64GB --disk 128GB -- \\
      python -m experiments.post_training.mochi_loop_guards counterfactual \\
      --manifest <resolved-manifest.json> --pattern <run output eval jsonl glob> \\
      --output <analysis prefix> [--replacement-input tb2=<stitched jsonl>]
"""

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from rolloutengine.models import ATTEMPT_SECONDS_METRIC

REVIEW_SAMPLE_SEED = 20261125
REVIEW_SAMPLE_SIZE = 30


def bounded(text: str, limit: int = 2400) -> str:
    """Redact and bound one free-text field to a centered excerpt."""
    clean = redact_string(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit // 2] + "\n[bounded excerpt]\n" + clean[-limit // 2 :]


def input_rows(
    rows: Iterable[Mapping], replacement_rows: Mapping[str, Iterable[Mapping]], membership: Mapping[str, str]
):
    """Yield original rows outside recorded replacements, then each replacement's rows.

    Every replacement row must belong to the frozen cohort it replaces.
    """
    for row in rows:
        if membership.get(row.get("task_id")) not in replacement_rows:
            yield row
    for cohort, replacement in replacement_rows.items():
        for row in replacement:
            if membership.get(row.get("task_id")) != cohort:
                raise ValueError("Replacement input contains a task outside its frozen cohort")
            yield row


def sample_rank(identity: list) -> str:
    """Rank one fired trial deterministically, independent of cut length and verdict."""
    return hashlib.sha256(json.dumps([REVIEW_SAMPLE_SEED, identity], separators=(",", ":")).encode()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    # The vendored SkyRL scripts ship in SkyRL's frozen CPU environment (cpu-submit
    # --skyrl); they are not Marin workspace dependencies.
    from vendor.skyrl.scripts.loop_audit import read_rows, trial_turns, verdict_passed
    from vendor.skyrl.scripts.repeat_stall_counterfactual import simulate_trial, summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument(
        "--replacement-input",
        action="append",
        default=[],
        help="COHORT=JSONL_URI for a recorded infrastructure recovery",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    manifest = json.loads(StoragePath(args.manifest).read_text())
    replacements = {}
    for value in args.replacement_input:
        cohort, pattern = value.split("=", 1)
        if cohort in replacements or cohort not in manifest["cohorts"]:
            raise ValueError("Duplicate or unknown replacement cohort")
        replacements[cohort] = pattern
    membership = {}
    for cohort, definition in manifest["cohorts"].items():
        for entry in definition["rows"]:
            task_id = entry["task_id"]
            if task_id in membership:
                raise ValueError("Task ID belongs to multiple cohorts")
            membership[task_id] = cohort
    trials = defaultdict(list)
    suppressed = defaultdict(int)
    timing = defaultdict(list)
    selected = []
    rows = input_rows(read_rows(args.pattern), {cohort: read_rows(pattern) for cohort, pattern in replacements.items()}, membership)
    for ordinal, row in enumerate(rows):
        cohort = membership.get(row.get("task_id"))
        if cohort is None:
            continue
        row = dict(row)
        if row.get("ordinal") is None:
            row["ordinal"] = ordinal
        simulation, waits = simulate_trial(row, 3)
        trials[cohort].append(simulation)
        suppressed[cohort] += waits
        duration = (row.get("env_metrics") or {}).get(ATTEMPT_SECONDS_METRIC)
        if verdict_passed(row) is not None and isinstance(duration, (int, float)):
            timing[cohort].append(float(duration))
        if not simulation["fired"]:
            continue
        identity = [cohort, *[simulation.get(key) for key in ("task_id", "attempt_id", "uid", "ordinal")]]
        rank = sample_rank(identity)
        if len(selected) == REVIEW_SAMPLE_SIZE and rank >= selected[-1]["sample_rank"]:
            continue
        turns = trial_turns(row.get("messages") or ())
        fire = simulation["fire_index"]
        window = []
        for index in range(max(0, fire - 3), min(len(turns), fire + 3)):
            turn = asdict(turns[index])
            turn["index"] = index
            for key in ("action", "observation", "message", "keystrokes"):
                value = turn[key]
                turn[key + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
                turn[key] = bounded(value)
            window.append(turn)
        selected.append({"cohort": cohort, "sample_rank": rank, **simulation, "window": window})
        selected.sort(key=lambda item: item["sample_rank"])
        selected = selected[:REVIEW_SAMPLE_SIZE]
    reports = []
    for cohort, definition in manifest["cohorts"].items():
        expected = len(definition["rows"]) * definition.get("trials_per_task", 1)
        if len(trials[cohort]) != expected:
            raise ValueError(f"{cohort}: expected {expected} saved trials, found {len(trials[cohort])}")
        reports.append(summarize(cohort, trials[cohort], suppressed[cohort], 3, 823))
    destination = StoragePath(args.output)
    destination.mkdirs()
    (destination / "counterfactual.json").write_text(json.dumps({"batches": 3, "cohorts": reports}, indent=2) + "\n")
    (destination / "review-sample.json").write_text(
        json.dumps(
            {
                "seed": REVIEW_SAMPLE_SEED,
                "selection": (
                    f"{REVIEW_SAMPLE_SIZE} smallest SHA256(seed, cohort, task_id, attempt_id, uid, ordinal), "
                    "independent of cut length and verdict"
                ),
                "population_fired": sum(report["fired_trials"] for report in reports),
                "sample_n": len(selected),
                "trials": selected,
            },
            indent=2,
        )
        + "\n"
    )
    statistics = {}
    for cohort, values in timing.items():
        statistics[cohort] = {
            "graded_timing_n": len(values),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }
    (destination / "graded-timing.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "reports": [
                    {key: value for key, value in report.items() if key != "hand_check_sample"} for report in reports
                ],
                "review_sample_n": len(selected),
                "timing": statistics,
            }
        )
    )


if __name__ == "__main__":
    main()
