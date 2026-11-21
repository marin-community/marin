#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace sealed NUPA cells with verified fixed-grader replay results."""

import argparse
import fcntl
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

LOGGER = logging.getLogger(__name__)
BENCHMARK = "NUPA"
SCORE_CELL = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+)\)$")


@dataclass(frozen=True)
class PromotionSummary:
    """Tracker replacements and models that need fresh NUPA generations."""

    updates: list[dict]
    regenerate: list[str]


def read_json(uri: str) -> dict:
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "rb") as source:
        return json.load(source)


def result_uri(output_prefix: str, model: str) -> str:
    return prefix_join(output_prefix, f"results/nupa/{model.replace('/', '--')}.json")


def thinking_boundaries_preserved(model: str, raw_source: str) -> bool:
    """Require recorded decoding settings before promoting Grug/Snowball replays."""
    if not any(name in model.casefold() for name in ("grug", "snowball")):
        return True
    record_uri = str(StoragePath(raw_source).parent / "record.json")
    filesystem, key = filesystem_for(record_uri)
    if not filesystem.exists(key):
        return False
    config = read_json(record_uri)["model"].get("config")
    return config is not None and config["generation"]["extra_gen_kwargs"].get("skip_special_tokens") == "false"


def update_tracker(tracker: Path, output_prefix: str) -> PromotionSummary:
    lines = tracker.read_text().splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    nupa_index = header.index(BENCHMARK)
    updates = []
    regenerate = []
    sealed_cells = 0
    for line_index, line in enumerate(lines[2:], start=2):
        if not line.startswith("| "):
            continue
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) != len(header) or row[0] == "Metric":
            continue
        match = SCORE_CELL.fullmatch(row[nupa_index])
        if match is None:
            continue
        sealed_cells += 1
        prior_score = float(match.group(1))
        prior_source = match.group(2)
        uri = result_uri(output_prefix, row[0])
        result = read_json(uri)
        if not thinking_boundaries_preserved(row[0], result["raw_source"]):
            LOGGER.warning("Leave %s unchanged: thinking boundaries are unverified; regenerate NUPA", row[0])
            regenerate.append(row[0])
            continue
        trials = result["trials"]
        corrected_score = sum(float(trial["corrected_correct"]) for trial in trials) / len(trials)
        if prior_source == uri:
            if abs(prior_score - corrected_score) > 0.000500001:
                raise ValueError(f"current replay score differs from stored trials for {row[0]}")
            continue
        if (
            result["model"] != row[0]
            or result["benchmark"] != BENCHMARK
            or result["source"] != prior_source
            or abs(float(result["tracker_score"]) - prior_score) > 1e-12
            or len(trials) != int(result["num_trials"])
            or abs(float(result["corrected_score"]) - corrected_score) > 1e-12
        ):
            raise ValueError(f"invalid replay identity or aggregate for {row[0]}")
        row[nupa_index] = f"{corrected_score:.3f} ({uri})"
        lines[line_index] = "| " + " | ".join(row) + " |"
        updates.append(
            {
                "model": row[0],
                "prior_score": prior_score,
                "corrected_score": corrected_score,
                "changed_trials": int(result["changed_trials"]),
                "source": prior_source,
                "replay": uri,
            }
        )
    if not sealed_cells:
        raise ValueError("tracker has no sealed NUPA cells to update")
    if updates:
        temporary = tracker.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(tracker)
    return PromotionSummary(updates=updates, regenerate=regenerate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    with args.lock.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summary = update_tracker(args.tracker, args.output_prefix)
    args.summary.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
