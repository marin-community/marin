#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank SOTOPIA models by length-controlled normalized overall score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from harbor.metrics.sotopia import (
    AdjustedEpisodeScore,
    LengthControlFit,
    SotopiaCell,
    length_controlled_scores,
    read_episodes,
)
from rigging.filesystem.storage_path import prefix_join

SCORE = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+/results)\)$")


def tracker_cells(path: Path) -> list[SotopiaCell]:
    """Read every sealed SOTOPIA-hard cell from a campaign tracker."""
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = [part.strip() for part in lines[0].strip("|").split("|")]
    benchmark_index = header.index("SOTOPIA-hard")
    cells = []
    for line in lines[3:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        if match := SCORE.match(parts[benchmark_index]):
            cells.append(SotopiaCell(parts[0], match.group(2)))
    if len(cells) < 2:
        raise ValueError(f"fewer than two sealed SOTOPIA-hard cells in {path}")
    return cells


def write_results(
    output_dir: Path,
    fit: LengthControlFit,
    cells: list[SotopiaCell],
    tracker_hash: str,
    output_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = fit.scores
    raw_order = sorted(scores, key=lambda row: (-row.raw_overall, row.model))
    raw_ranks = {row.model: index + 1 for index, row in enumerate(raw_order)}
    rows = []
    for index, score in enumerate(scores, start=1):
        row = asdict(score)
        row.update(
            {
                "length_controlled_rank": index,
                "raw_rank": raw_ranks[score.model],
                "rank_change": raw_ranks[score.model] - index,
            }
        )
        rows.append(row)
    with (output_dir / "sotopia_length_controlled_ranking.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "tracker_sha256": tracker_hash,
        "models": [asdict(cell) for cell in cells],
        "method": {
            "outcome": "arithmetic mean of seven normalized SOTOPIA dimensions",
            "length": "evaluated-agent emitted characters over the full episode",
            "counterfactual": "zero difference from the per-task cohort median",
            "model": "fractional logistic GLM with model, task, and tanh length effects",
            "selected_regularization_c": fit.selected_regularization,
            "cross_validation": "five folds grouped by task",
            "cross_validation_losses": fit.cross_validation_losses,
        },
    }
    (output_dir / "sotopia_length_controlled_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    source_by_model = {cell.model: cell.results_path for cell in cells}
    episodes_by_model: dict[str, list[AdjustedEpisodeScore]] = defaultdict(list)
    for episode in fit.episodes:
        episodes_by_model[episode.model].append(episode)
    result_dir = output_dir / "results" / "sotopia-hard"
    result_dir.mkdir(parents=True, exist_ok=True)
    statistics = []
    updates = []
    for score in scores:
        model_episodes = episodes_by_model[score.model]
        values = np.asarray([episode.length_controlled_overall for episode in model_episodes])
        sem = float(values.std(ddof=1) / math.sqrt(values.size))
        relative = f"results/sotopia-hard/{score.model.replace('/', '--')}.json"
        result_uri = prefix_join(output_prefix, relative)
        payload = {
            "model": score.model,
            "benchmark": "SOTOPIA-hard",
            "source": source_by_model[score.model],
            "corrected_score": score.length_controlled_overall,
            "num_trials": len(model_episodes),
            "num_scored_failures": sum(episode.response_characters is None for episode in model_episodes),
            "method": metadata["method"],
            "trials": [asdict(episode) for episode in model_episodes],
        }
        (output_dir / relative).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        statistics.append(
            {
                "model": score.model,
                "benchmark": "SOTOPIA-hard",
                "score": score.length_controlled_overall,
                "trial_count": len(model_episodes),
                "raw_reward_mean": score.length_controlled_overall,
                "sem": sem,
                "adjusted_sem": sem / math.sqrt(0.9),
                "sem_basis": "length_controlled_episode_rewards",
                "source": result_uri,
                "statistics_version": "sotopia-length-controlled-overall-v1",
            }
        )
        updates.append(
            {
                "model": score.model,
                "old_source": source_by_model[score.model],
                "score": score.length_controlled_overall,
                "source": result_uri,
            }
        )
    for name, output_rows in (("cell_statistics.csv", statistics), ("tracker_updates.csv", updates)):
        with (output_dir / name).open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    cells = tracker_cells(args.tracker)
    episodes = [episode for cell in cells for episode in read_episodes(cell)]
    fit = length_controlled_scores(episodes)
    write_results(
        args.output_dir,
        fit,
        cells,
        hashlib.sha256(args.tracker.read_bytes()).hexdigest(),
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
