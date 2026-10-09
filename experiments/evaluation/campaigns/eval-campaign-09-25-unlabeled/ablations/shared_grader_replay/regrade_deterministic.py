#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay sealed MATH500 and AIME24 samples through their pinned native graders."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import yaml
from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

LOGGER = logging.getLogger(__name__)
EVALCHEMY_COMMIT = "958cdb8019b7a4c8432fe85eb9572538ceb75950"
BENCHMARK_SIZES = {"math500": (500, 1), "aime24": (30, 10)}
SAMPLE_COLUMNS = ["doc_id", "trial_id", "doc", "extracted", "metrics"]


def write_json(uri: str, payload: dict) -> None:
    """Write a self-contained result to a new object-store key."""
    filesystem, key = filesystem_for(uri)
    if filesystem.exists(key):
        raise FileExistsError(uri)
    with filesystem.open(key, "wb") as destination:
        destination.write((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def read_cells(manifest: Path, benchmark: str) -> list[dict]:
    """Select frozen tracker cells for one benchmark."""
    rows = yaml.safe_load(manifest.read_text())["models"]
    cells = [
        {
            "model": row["model"],
            "tracker_score": float(row["benchmarks"][benchmark]["tracker_score"]),
            "source": row["benchmarks"][benchmark]["source"],
        }
        for row in rows
    ]
    if len(cells) != 21 or len({cell["model"] for cell in cells}) != 21:
        raise ValueError("expected 21 unique tracker models")
    return cells


def source_rows(cell: dict, benchmark: str) -> list[dict]:
    """Read and validate every sealed trial from its original FineStore archive."""
    questions, repeats = BENCHMARK_SIZES[benchmark]
    table = ReadView(cell["source"]).scan("samples", columns=SAMPLE_COLUMNS)
    if table is None or table.num_rows != questions * repeats:
        raise ValueError(f"{cell['model']}: expected {questions * repeats} trials")
    rows = table.to_pylist(maps_as_pydicts="strict")
    if benchmark == "math500":
        rows.sort(key=lambda row: int(row["doc_id"]))
        if [int(row["doc_id"]) for row in rows] != list(range(questions)):
            raise ValueError(f"{cell['model']}: incomplete MATH500 document IDs")
    else:
        rows.sort(key=lambda row: (int(row["doc_id"]), int(row["trial_id"])))
        positions = [(int(row["doc_id"]), int(row["trial_id"])) for row in rows]
        expected = [(document, repeat) for document in range(questions) for repeat in range(repeats)]
        if positions != expected:
            raise ValueError(f"{cell['model']}: incomplete AIME24 document/repeat IDs")
    old_mean = sum(float(row["metrics"]["accuracy"]) for row in rows) / len(rows)
    if abs(old_mean - cell["tracker_score"]) > 0.0005:
        raise ValueError(f"{cell['model']}: sealed trial mean {old_mean} differs from tracker score")
    return rows


def regrade(cell: dict, benchmark: str) -> dict:
    """Call the benchmark's evaluate_responses implementation on sealed extractions."""
    rows = source_rows(cell, benchmark)
    if benchmark == "math500":
        examples = []
        for row in rows:
            example = json.loads(row["doc"])
            example["model_answer"] = row["extracted"] or ""
            examples.append(example)
        scored = MATH500Benchmark().evaluate_responses({"examples": examples})
        new_scores = [float(example["sample_metrics"]["accuracy"]) for example in examples]
        native_score = float(scored["accuracy"])
    else:
        examples = []
        for document in range(30):
            document_rows = rows[document * 10 : (document + 1) * 10]
            example = json.loads(document_rows[0]["doc"])
            example["model_answers"] = [row["extracted"] or "" for row in document_rows]
            examples.append(example)
        scored = AIME24Benchmark().evaluate_responses({"examples": examples})
        new_scores = [
            float(examples[document]["sample_metrics_by_repeat"][repeat]["accuracy"])
            for document in range(30)
            for repeat in range(10)
        ]
        native_score = float(scored["accuracy_avg"])
    new_mean = sum(new_scores) / len(new_scores)
    if abs(new_mean - native_score) > 1e-9:
        raise ValueError(f"{cell['model']}: native aggregate differs from trial metrics")
    trials = [
        {
            "doc_id": int(row["doc_id"]),
            "trial_id": int(row["trial_id"]) if benchmark == "aime24" else None,
            "candidate_answer": row["extracted"] or "",
            "reference_answer": (
                json.loads(row["doc"])["answer"]
                if benchmark == "math500"
                else json.loads(row["doc"])["expected_answer"]
            ),
            "prior_correct": bool(row["metrics"]["accuracy"]),
            "corrected_correct": bool(correct),
        }
        for row, correct in zip(rows, new_scores, strict=True)
    ]
    return {
        "model": cell["model"],
        "benchmark": benchmark,
        "source": cell["source"],
        "tracker_score": cell["tracker_score"],
        "prior_trial_mean": sum(float(row["metrics"]["accuracy"]) for row in rows) / len(rows),
        "corrected_score": new_mean,
        "changed_trials": sum(trial["prior_correct"] != trial["corrected_correct"] for trial in trials),
        "num_trials": len(trials),
        "evalchemy_commit": EVALCHEMY_COMMIT,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_SIZES), required=True)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    cells = read_cells(args.sources, args.benchmark)
    if args.model:
        cells = [cell for cell in cells if cell["model"] in args.model]
        if len(cells) != len(args.model):
            raise ValueError("requested model absent from source manifest")
    prefix = args.output_prefix.rstrip("/")
    provenance_uri = f"{prefix}/provenance-{args.benchmark}.json"
    filesystem, key = filesystem_for(provenance_uri)
    if not filesystem.exists(key):
        write_json(
            provenance_uri,
            {
                "benchmark": args.benchmark,
                "evalchemy_commit": EVALCHEMY_COMMIT,
                "source_manifest_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
                "source_manifest": args.sources.read_text(),
            },
        )
    for cell in cells:
        result_uri = f"{prefix}/results/{args.benchmark}/{cell['model'].replace('/', '--')}.json"
        filesystem, key = filesystem_for(result_uri)
        if filesystem.exists(key):
            LOGGER.info("already regraded %s %s", cell["model"], args.benchmark)
            continue
        result = regrade(cell, args.benchmark)
        write_json(result_uri, result)
        LOGGER.info(
            "%s %s: %.3f -> %.3f; changed=%d",
            cell["model"],
            args.benchmark,
            result["prior_trial_mean"],
            result["corrected_score"],
            result["changed_trials"],
        )


if __name__ == "__main__":
    main()
