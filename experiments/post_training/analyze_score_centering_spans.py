# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit per-version coverage and matched A/B/C scores in schema-2 diagnostic records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.post_training.analyze_score_centering_mismatch import _read, _sources

FIELDS = (
    "step",
    "selected_tokens",
    "matched_tokens",
    "reference_tokens",
    "fresh_version_tokens",
    "other_version_tokens",
    "version_token_counts",
    "mixed_version_responses",
    "matched_mixed_version_responses",
    "age",
    "age_tokens",
    "engine_abs_mean",
    "stale_abs_mean",
    "combined_abs_mean",
    "opposite_sign_fraction",
    "tis_capped_fraction",
    "reconstruction_abs_max",
)


def _check_samples(record: dict, path: str) -> None:
    for sample in record["samples"]:
        positions = sample["positions"]
        scores = (sample[key] for key in ("A_inference", "B_generating_trainer", "C_consumer_trainer"))
        a, b, c = scores
        if not all(len(x) == len(positions) for x in (a, b, c, sample["B_source"])):
            raise ValueError(f"{path}: sampled score arrays have different lengths")
        for index, position in enumerate(positions):
            if not sample["loss_mask"][position]:
                raise ValueError(f"{path}: sampled token is outside the loss mask")
            segment = next(
                (
                    segment
                    for segment in sample["version_segments"]
                    if segment["start"] <= position < segment["start"] + segment["token_count"]
                ),
                None,
            )
            if segment is None:
                raise ValueError(f"{path}: sampled token has no generating version")
            expected_source = (
                "frozen_reference"
                if segment["policy_version"] == record["reference_version"]
                else "fresh_consuming_policy" if segment["policy_version"] == record["fresh_scored_version"] else None
            )
            if sample["B_source"][index] != expected_source:
                raise ValueError(f"{path}: sampled B is assigned to the wrong generating version")
            if abs((b[index] - a[index]) + (c[index] - b[index]) - (c[index] - a[index])) > 1e-10:
                raise ValueError(f"{path}: sampled A/B/C scores do not reconstruct")
            if expected_source == "fresh_consuming_policy" and b[index] != c[index]:
                raise ValueError(f"{path}: age-zero B does not equal consuming trainer C")


def summarize(prefix: str) -> list[dict]:
    sources = _sources(prefix)
    if not sources:
        raise ValueError(f"no diagnostic records in {prefix}")
    rows = []
    for step, fs, path in sources:
        record = _read(fs, path)
        if record["schema_version"] != 2 or record["consuming_step"] != step:
            raise ValueError(f"{path}: unexpected schema or consuming step")
        counts = record["selected_tokens_by_version"]
        if sum(counts.values()) != record["selected_tokens"]:
            raise ValueError(f"{path}: version counts do not cover selected tokens")
        if record["matched_tokens"] != record["reference_tokens"] + record["fresh_version_tokens"]:
            raise ValueError(f"{path}: matched token count disagrees with B sources")
        if record["selected_tokens"] != record["matched_tokens"] + record["other_version_tokens"]:
            raise ValueError(f"{path}: unmatched tokens are not accounted for")
        ages = [(key, value) for key, value in record["summaries"].items() if key.startswith("age_")]
        if sum(value["tokens"] for _, value in ages) != record["matched_tokens"]:
            raise ValueError(f"{path}: age summaries do not cover matched tokens")
        _check_samples(record, path)
        for age_key, item in ages or [(None, {"tokens": 0})]:
            if item["tokens"] and item["reconstruction_abs_max"] > 1e-10:
                raise ValueError(f"{path}: A/B/C terms do not reconstruct")
            row = {
                "step": step,
                "selected_tokens": record["selected_tokens"],
                "matched_tokens": record["matched_tokens"],
                "reference_tokens": record["reference_tokens"],
                "fresh_version_tokens": record["fresh_version_tokens"],
                "other_version_tokens": record["other_version_tokens"],
                "version_token_counts": json.dumps(counts, sort_keys=True, separators=(",", ":")),
                "mixed_version_responses": record["mixed_version_responses"],
                "matched_mixed_version_responses": record["matched_mixed_version_responses"],
                "age": int(age_key.removeprefix("age_")) if age_key else "",
                "age_tokens": item["tokens"],
            }
            if item["tokens"]:
                for component in ("engine", "stale", "combined"):
                    row[f"{component}_abs_mean"] = item[component]["log_ratio_abs_mean"]
                for key in ("opposite_sign_fraction", "tis_capped_fraction", "reconstruction_abs_max"):
                    row[key] = item[key]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local or S3 mismatch_decomposition export directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
