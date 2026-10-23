#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay sealed NUPA5K responses through the pinned NUPA Loose grader."""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
from importlib import import_module

from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

LOGGER = logging.getLogger(__name__)
SCORER = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")
EVALCHEMY_COMMIT = "2baaab1b9c0c9c8caffcd8f6df45355539aef436"
BENCHMARK = "NUPA"
BENCHMARK_SIZE = 5_000
SCORE_CELL = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+)\)$")
SAMPLE_COLUMNS = ["doc_id", "doc", "output", "extracted", "metrics"]


def read_bytes(uri: str) -> bytes:
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "rb") as source:
        return source.read()


def write_json(uri: str, payload: dict) -> None:
    filesystem, key = filesystem_for(uri)
    if filesystem.exists(key):
        raise FileExistsError(uri)
    with filesystem.open(key, "wb") as destination:
        destination.write((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def tracker_cells(tracker: str) -> list[dict]:
    lines = tracker.splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    nupa_index = header.index(BENCHMARK)
    cells = []
    for line in lines[2:]:
        if not line.startswith("| "):
            continue
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) != len(header) or row[0] == "Metric":
            continue
        match = SCORE_CELL.fullmatch(row[nupa_index])
        if match is None:
            continue
        cells.append({"model": row[0], "tracker_score": float(match.group(1)), "source": match.group(2)})
    if len({cell["model"] for cell in cells}) != len(cells):
        raise ValueError("tracker contains duplicate sealed NUPA models")
    return cells


def result_uri(output_prefix: str, model: str) -> str:
    return prefix_join(output_prefix, f"results/nupa/{model.replace('/', '--')}.json")


def replay_input(cell: dict) -> tuple[str, dict[int, dict] | None]:
    """Resolve a tracker cell to raw samples and its current per-trial grades."""
    source = cell["source"]
    if not source.endswith(".json"):
        return source, None

    prior = read_json(source)
    if prior["model"] != cell["model"] or prior["benchmark"] != BENCHMARK:
        raise ValueError(f"{cell['model']}: replay source identity does not match tracker")
    trials = {int(trial["doc_id"]): trial for trial in prior["trials"]}
    if set(trials) != set(range(BENCHMARK_SIZE)):
        raise ValueError(f"{cell['model']}: replay source has incomplete NUPA document IDs")
    prior_score = sum(bool(trial["corrected_correct"]) for trial in trials.values()) / BENCHMARK_SIZE
    if abs(prior_score - cell["tracker_score"]) > 0.000500001:
        raise ValueError(f"{cell['model']}: replay trial mean {prior_score} differs from tracker score")
    return prior.get("raw_source", prior["source"]), trials


def regrade(cell: dict) -> dict:
    raw_source, prior_trials = replay_input(cell)
    table = ReadView(raw_source).scan("samples", columns=SAMPLE_COLUMNS)
    if table is None or table.num_rows != BENCHMARK_SIZE:
        raise ValueError(f"{cell['model']}: expected {BENCHMARK_SIZE} stored trials")
    rows = table.to_pylist(maps_as_pydicts="strict")
    rows.sort(key=lambda row: int(row["doc_id"]))
    if [int(row["doc_id"]) for row in rows] != list(range(BENCHMARK_SIZE)):
        raise ValueError(f"{cell['model']}: incomplete NUPA document IDs")

    prior_score = (
        sum(
            (
                bool(prior_trials[int(row["doc_id"])]["corrected_correct"])
                if prior_trials is not None
                else float(row["metrics"]["exact_match"])
            )
            for row in rows
        )
        / BENCHMARK_SIZE
    )
    if abs(prior_score - cell["tracker_score"]) > 0.000500001:
        raise ValueError(f"{cell['model']}: stored trial mean {prior_score} differs from tracker score")

    trials = []
    metric_totals = {
        "exact_match": 0.0,
        "digit_match": 0.0,
        "dlength": 0.0,
        "format_valid_rate": 0.0,
        "no_answer_rate": 0.0,
    }
    changed_trials = 0
    for row in rows:
        doc = json.loads(row["doc"])
        scored = SCORER.score_prediction(row["output"], doc["answer"], doc["answer_format"])
        corrected = {
            "exact_match": scored.exact_match,
            "digit_match": scored.digit_match,
            "dlength": scored.dlength,
            "format_valid_rate": scored.format_valid,
            "no_answer_rate": scored.no_answer,
        }
        for metric, value in corrected.items():
            metric_totals[metric] += value
        prior_trial = prior_trials[int(row["doc_id"])] if prior_trials is not None else None
        prior_correct = (
            bool(prior_trial["corrected_correct"]) if prior_trial is not None else bool(row["metrics"]["exact_match"])
        )
        corrected_correct = bool(scored.exact_match)
        changed_trials += prior_correct != corrected_correct
        trials.append(
            {
                "doc_id": int(row["doc_id"]),
                "answer_format": doc["answer_format"],
                "reference_answer": doc["answer"],
                "prior_extracted": prior_trial["corrected_extracted"] if prior_trial is not None else row["extracted"],
                "corrected_extracted": scored.extracted_answer,
                "prior_correct": prior_correct,
                "corrected_correct": corrected_correct,
            }
        )

    aggregates = {metric: total / BENCHMARK_SIZE for metric, total in metric_totals.items()}
    return {
        "model": cell["model"],
        "benchmark": BENCHMARK,
        "source": cell["source"],
        "raw_source": raw_source,
        "tracker_score": cell["tracker_score"],
        "prior_trial_mean": prior_score,
        "corrected_score": aggregates["exact_match"],
        "corrected_metrics": aggregates,
        "changed_trials": changed_trials,
        "num_trials": BENCHMARK_SIZE,
        "evalchemy_commit": EVALCHEMY_COMMIT,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-uri", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()

    tracker_bytes = read_bytes(args.tracker_uri)
    tracker = tracker_bytes.decode()
    cells = tracker_cells(tracker)
    if args.model:
        requested = set(args.model)
        cells = [cell for cell in cells if cell["model"] in requested]
        if {cell["model"] for cell in cells} != requested:
            raise ValueError("requested model absent from sealed NUPA tracker cells")

    prefix = args.output_prefix
    provenance_uri = prefix_join(prefix, "provenance.json")
    filesystem, key = filesystem_for(provenance_uri)
    if not filesystem.exists(key):
        write_json(
            provenance_uri,
            {
                "benchmark": BENCHMARK,
                "evalchemy_commit": EVALCHEMY_COMMIT,
                "tracker_uri": args.tracker_uri,
                "tracker_sha256": hashlib.sha256(tracker_bytes).hexdigest(),
                "tracker": tracker,
                "cells": cells,
            },
        )

    def replay_cell(cell: dict) -> tuple[str, dict | None]:
        uri = result_uri(prefix, cell["model"])
        output_filesystem, output_key = filesystem_for(uri)
        if output_filesystem.exists(output_key):
            return uri, None
        result = regrade(cell)
        write_json(uri, result)
        return uri, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(replay_cell, cell): cell for cell in cells}
        for future in concurrent.futures.as_completed(futures):
            cell = futures[future]
            uri, result = future.result()
            if result is None:
                LOGGER.info("already regraded %s", cell["model"])
            else:
                LOGGER.info(
                    "%s: %.3f -> %.3f; changed=%d; %s",
                    cell["model"],
                    result["prior_trial_mean"],
                    result["corrected_score"],
                    result["changed_trials"],
                    uri,
                )


if __name__ == "__main__":
    main()
