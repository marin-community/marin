#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay sealed GSM8K responses through the campaign's corrected grader."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

LOGGER = logging.getLogger(__name__)
BENCHMARK = "gsm8k-0shot"
BENCHMARK_SIZE = 1_319
FILTER = "flexible-extract"
SCORE_CELL = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+)\)$")
SAMPLE_COLUMNS = ["doc_id", "doc", "output", "extracted", "metrics", "filter"]


@dataclass(frozen=True)
class TrackerCell:
    model: str
    tracker_score: float
    source: str


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


def tracker_cells(tracker: str) -> list[TrackerCell]:
    """Return every sealed GSM8K cell in tracker order."""
    lines = tracker.splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    benchmark_index = header.index(BENCHMARK)
    cells = []
    for line in lines[2:]:
        if not line.startswith("| "):
            continue
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) != len(header) or row[0] == "Metric":
            continue
        match = SCORE_CELL.fullmatch(row[benchmark_index])
        if match is None:
            continue
        cells.append(TrackerCell(row[0], float(match.group(1)), match.group(2)))
    if len({cell.model for cell in cells}) != len(cells):
        raise ValueError("tracker contains duplicate sealed GSM8K models")
    return cells


def decode_doc(value: str | dict) -> dict:
    return json.loads(value) if isinstance(value, str) else value


def source_rows(cell: TrackerCell) -> list[dict]:
    """Read and validate the complete flexible-extract sample set."""
    table = ReadView(cell.source).scan("samples", columns=SAMPLE_COLUMNS)
    if table is None:
        raise ValueError(f"{cell.model}: no FineStore samples")
    rows = [row for row in table.to_pylist(maps_as_pydicts="strict") if row["filter"] == FILTER]
    if len(rows) != BENCHMARK_SIZE:
        raise ValueError(f"{cell.model}: expected {BENCHMARK_SIZE} {FILTER} samples, got {len(rows)}")
    rows.sort(key=lambda row: int(row["doc_id"]))
    if [int(row["doc_id"]) for row in rows] != list(range(BENCHMARK_SIZE)):
        raise ValueError(f"{cell.model}: incomplete or duplicate GSM8K document IDs")
    prior_score = sum(float(row["metrics"]["exact_match"]) for row in rows) / BENCHMARK_SIZE
    if abs(prior_score - cell.tracker_score) > 0.000500001:
        raise ValueError(
            f"{cell.model}: stored trial mean {prior_score:.9f} differs from tracker score {cell.tracker_score:.3f}"
        )
    return rows


def result_uri(output_prefix: str, model: str) -> str:
    return prefix_join(output_prefix, f"results/gsm8k-0shot/{model.replace('/', '--')}.json")


def regrade(
    cell: TrackerCell,
    evalchemy_revision: str,
    grader_sha256: str,
    extract_answer: Callable[[str], str],
    score_answer: Callable[[dict, list[str]], dict[str, float]],
) -> dict:
    """Re-extract and score all trials from one sealed GSM8K cell."""
    rows = source_rows(cell)
    trials = []
    changed_extractions = 0
    gains = 0
    regressions = 0
    prior_correct_count = 0
    corrected_correct_count = 0
    for row in rows:
        doc = decode_doc(row["doc"])
        prior_extracted = row.get("extracted") or ""
        corrected_extracted = extract_answer(row.get("output") or "")
        prior_correct = bool(row["metrics"]["exact_match"])
        corrected_correct = bool(score_answer(doc, [corrected_extracted])["exact_match"])
        prior_correct_count += prior_correct
        corrected_correct_count += corrected_correct
        changed_extractions += prior_extracted != corrected_extracted
        gains += corrected_correct and not prior_correct
        regressions += prior_correct and not corrected_correct
        trials.append(
            {
                "doc_id": int(row["doc_id"]),
                "question": doc["question"],
                "reference_answer": doc["answer"].rsplit("#### ", 1)[-1],
                "prior_extracted": prior_extracted,
                "corrected_extracted": corrected_extracted,
                "prior_correct": prior_correct,
                "corrected_correct": corrected_correct,
            }
        )
    return {
        "model": cell.model,
        "benchmark": BENCHMARK,
        "source": cell.source,
        "tracker_score": cell.tracker_score,
        "prior_trial_mean": prior_correct_count / BENCHMARK_SIZE,
        "corrected_score": corrected_correct_count / BENCHMARK_SIZE,
        "changed_extractions": changed_extractions,
        "changed_trials": gains + regressions,
        "gains": gains,
        "regressions": regressions,
        "num_trials": BENCHMARK_SIZE,
        "evalchemy_revision": evalchemy_revision,
        "grader_sha256": grader_sha256,
        "trials": trials,
    }


def write_local_artifacts(output_dir: Path, results: list[tuple[str, dict]], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    changed_trials = []
    for uri, result in results:
        summary.append(
            {
                "model": result["model"],
                "tracker_score": result["tracker_score"],
                "prior_score": result["prior_trial_mean"],
                "corrected_score": result["corrected_score"],
                "score_delta": result["corrected_score"] - result["prior_trial_mean"],
                "changed_extractions": result["changed_extractions"],
                "gains": result["gains"],
                "regressions": result["regressions"],
                "num_trials": result["num_trials"],
                "source": result["source"],
                "regrade_result": uri,
            }
        )
        changed_trials.extend(
            {"model": result["model"], **trial}
            for trial in result["trials"]
            if trial["prior_extracted"] != trial["corrected_extracted"]
            or trial["prior_correct"] != trial["corrected_correct"]
        )

    with (output_dir / "score_changes.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "score_changes.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "changed_samples.jsonl").open("w") as target:
        for trial in changed_trials:
            target.write(json.dumps(trial, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    changed_models = sum(bool(row["gains"] or row["regressions"]) for row in summary)
    total_gains = sum(row["gains"] for row in summary)
    total_regressions = sum(row["regressions"] for row in summary)
    lines = [
        "# GSM8K grader ablation",
        "",
        "## TL;DR",
        "",
        f"The corrected grader changed {changed_models} of {len(summary)} model scores. Across "
        f"{len(summary) * BENCHMARK_SIZE:,} sealed trials, it corrected {total_gains} false negatives and "
        f"{total_regressions} false positives.",
        "",
        "## Method",
        "",
        "The replay reads each tracker-linked FineStore archive and requires all 1,319 GSM8K test documents. "
        "It applies the shared final-box parser first. Unboxed responses retain the existing explicit-answer and "
        "last-number fallbacks. The scorer accepts scalar annotations such as currency, percentages, and units. "
        "For questions that ask for a time, an exact whole-hour clock answer can match the dataset's integer-hour "
        "reference. Non-whole-hour clock answers remain distinct.",
        "",
        "The replay uses the stored model outputs. It does not call a model or judge. Each durable result contains "
        "all per-document prior and corrected verdicts.",
        "",
        "## Scores",
        "",
        "| Model | Prior | Corrected | Delta | Gains | Regressions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['prior_score']:.3f} | {row['corrected_score']:.3f} | "
            f"{row['score_delta']:+.3f} | {row['gains']} | {row['regressions']} |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"- Evalchemy revision: `{metadata['evalchemy_revision']}`",
        f"- Grader source SHA-256: `{metadata['grader_sha256']}`",
        f"- Tracker SHA-256: `{metadata['tracker_sha256']}`",
        f"- Durable prefix: `{metadata['output_prefix']}`",
        "- Machine-readable summary: `score_changes.csv` and `score_changes.json`",
        "- Changed extraction or verdict rows: `changed_samples.jsonl`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--evalchemy-revision", required=True)
    parser.add_argument("--grader-source", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    grader = import_module("eval.lm_eval_tasks.gsm8k.utils")

    tracker_bytes = args.tracker.read_bytes()
    cells = tracker_cells(tracker_bytes.decode())
    if not cells:
        raise ValueError("tracker has no sealed GSM8K cells")
    grader_sha256 = hashlib.sha256(args.grader_source.read_bytes()).hexdigest()
    prefix = args.output_prefix.rstrip("/")
    metadata = {
        "benchmark": BENCHMARK,
        "evalchemy_revision": args.evalchemy_revision,
        "grader_sha256": grader_sha256,
        "output_prefix": prefix,
        "tracker_path": str(args.tracker),
        "tracker_sha256": hashlib.sha256(tracker_bytes).hexdigest(),
        "models": len(cells),
        "trials_per_model": BENCHMARK_SIZE,
    }
    provenance_uri = prefix_join(prefix, "provenance.json")
    provenance_filesystem, provenance_key = filesystem_for(provenance_uri)
    if not provenance_filesystem.exists(provenance_key):
        write_json(provenance_uri, {**metadata, "tracker": tracker_bytes.decode()})

    def replay_cell(cell: TrackerCell) -> tuple[str, dict]:
        uri = result_uri(prefix, cell.model)
        filesystem, key = filesystem_for(uri)
        if filesystem.exists(key):
            return uri, json.loads(read_bytes(uri))
        result = regrade(
            cell,
            args.evalchemy_revision,
            grader_sha256,
            grader.extract_gsm8k_flexible_answer,
            grader.process_results,
        )
        write_json(uri, result)
        return uri, result

    # Minerva's symbolic comparator enforces its timeout with SIGALRM. Keep
    # scoring on the main thread so expressions do not become false negatives.
    results = []
    for cell in cells:
        uri, result = replay_cell(cell)
        results.append((uri, result))
        LOGGER.info(
            "%s: %.3f -> %.3f; gains=%d regressions=%d",
            cell.model,
            result["prior_trial_mean"],
            result["corrected_score"],
            result["gains"],
            result["regressions"],
        )
    write_local_artifacts(args.local_output_dir, results, metadata)


if __name__ == "__main__":
    main()
