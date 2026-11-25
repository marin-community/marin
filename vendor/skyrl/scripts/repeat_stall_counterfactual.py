# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline counterfactual for the I1 ``repeat_stall`` stop, over saved baseline traces.

Simulates the recorder's stop on each retained trial's turns with the same shared identity
module, and reports where the stop would fire, the fire rate, the recorded verdicts of the
trials it would have stopped, and a hand-check sample. Run it on S0's ``dumped_evals`` rows:

    uv run --project vendor/skyrl --with torch python vendor/skyrl/scripts/repeat_stall_counterfactual.py \\
        --input COHORT=JSONL_GLOB --output <json path> --sample 30
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from rolloutengine.repetition import ActionPair, detect_stall
from vendor.skyrl.scripts.loop_audit import trial_turns, verdict_passed

DEFAULT_BATCHES = 3


class StallSimulation:
    """The recorder's stall window, replayed turn by turn over one saved trial."""

    def __init__(self, batches: int):
        self.batches = batches
        self.window: list[ActionPair] = []
        self.fire_index: int | None = None
        self.wait_suppressed = 0
        """Windows that identical non-wait pairs would have stalled but a wait exempted."""

    def offer(self, index: int, turn: Mapping[str, Any]) -> bool:
        """Offer one completed (action, observation) turn; report whether the stop fires here.

        The recorder checks before sending the request that follows the candidate pair, so a
        fire at turn ``index`` cuts every later turn and their observations.
        """
        pair = ActionPair(turn["action"], turn["observation"], turn["wait"])
        self.window.append(pair)
        if detect_stall(self.window, self.batches):
            self.fire_index = index
            return True
        if self._identical_with_wait():
            self.wait_suppressed += 1
        self.window = self.window[-(self.batches - 1) :]
        return False

    def _identical_with_wait(self) -> bool:
        window = self.window[-self.batches :]
        if len(window) < self.batches:
            return False
        first = window[0]
        return all(
            pair.action == first.action and pair.observation == first.observation for pair in window[1:]
        ) and any(pair.wait for pair in window)


def simulate_trial(row: Mapping[str, Any], batches: int) -> tuple[dict[str, Any], int]:
    turns = trial_turns(row.get("messages") or ())
    simulation = StallSimulation(batches)
    for index, turn in enumerate(turns):
        if simulation.offer(index, {"action": turn.action, "observation": turn.observation, "wait": turn.wait}):
            break
    fired = simulation.fire_index is not None
    trial = {
        "task_id": row.get("task_id"),
        "turns": len(turns),
        "fired": fired,
        "fire_index": simulation.fire_index,
        "cut_turns": (len(turns) - 1 - simulation.fire_index) if fired else 0,
        "verdict": verdict_passed(row),
        "stop_reason": row.get("stop_reason"),
    }
    return trial, simulation.wait_suppressed


def read_rows(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(StoragePath(pattern).glob(), key=str):
        if path.name == "aggregated_results.jsonl":
            continue
        with path.open("r") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(cohort: str, simulations: Sequence[Mapping[str, Any]], wait_suppressed: int, batches: int, sample: int) -> dict[str, Any]:
    fired = [item for item in simulations if item["fired"]]
    fire_indices = [item["fire_index"] for item in fired]
    stopped_passes = [item for item in fired if item["verdict"]]
    at_risk = [item for item in stopped_passes if item["cut_turns"] > 0]
    sample_rows = sorted(fired, key=lambda item: (-item["cut_turns"], item["task_id"] or ""))[:sample]
    return {
        "cohort": cohort,
        "repeat_stall_batches": batches,
        "trials": len(simulations),
        "fired_trials": len(fired),
        "fire_rate": len(fired) / len(simulations) if simulations else 0.0,
        "fire_index_median": float(np.median(fire_indices)) if fire_indices else None,
        "fire_index_p90": float(np.percentile(fire_indices, 90)) if fire_indices else None,
        "cut_turns_median": float(np.median([item["cut_turns"] for item in fired])) if fired else None,
        "stopped_trials_with_passing_verdict": len(stopped_passes),
        "stopped_trials_pass_rate": len(stopped_passes) / len(fired) if fired else None,
        "at_risk_passes_cut_before_their_last_turn": len(at_risk),
        "wait_only_suppressed_windows": wait_suppressed,
        "hand_check_sample": [
            {key: item[key] for key in ("task_id", "fire_index", "turns", "cut_turns", "verdict", "stop_reason")}
            for item in sample_rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="COHORT=JSONL_GLOB; one saved session per cohort")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help="The stall window N (the recorder's repeat_stall_batches)")
    parser.add_argument("--sample", type=int, default=30, help="Hand-check sample size over fired trials")
    parser.add_argument("--output", required=True, help="JSON output URI; Markdown is written beside it")
    args = parser.parse_args()
    configure_coreweave_s3()
    reports = []
    for item in args.input:
        cohort, pattern = item.split("=", 1)
        rows = read_rows(pattern)
        simulations = []
        suppressed = 0
        for row in rows:
            trial, waits = simulate_trial(row, args.batches)
            simulations.append(trial)
            suppressed += waits
        reports.append(summarize(cohort, simulations, suppressed, args.batches, args.sample))
    document = {"batches": args.batches, "cohorts": reports}
    StoragePath(args.output).write_text(json.dumps(document, indent=2) + "\n")
    lines = ["| Cohort | Trials | Fired | Fire rate | Fire turn p50/p90 | Stopped pass rate | At-risk passes |", "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for report in reports:
        lines.append(
            f"| {report['cohort']} | {report['trials']} | {report['fired_trials']} | {report['fire_rate']:.3f} "
            f"| {report['fire_index_median']}/{report['fire_index_p90']} "
            f"| {report['stopped_trials_pass_rate'] if report['stopped_trials_pass_rate'] is not None else 'n/a'} "
            f"| {report['at_risk_passes_cut_before_their_last_turn']} |"
        )
    markdown = "\n".join(lines) + "\n\nHand-check the sample in the JSON output for precision (I1 passes at ≥ 90% genuine stalls).\n"
    StoragePath(args.output.removesuffix(".json") + ".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
