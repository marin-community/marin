# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the saved A/B/C token log-probability diagnostic by update and position.

Run with ``marin-env uv run --frozen --python 3.12 python``. The input is the
``mismatch_decomposition`` export directory, local or S3. Rows are computed
from saved aggregates; sampled raw tokens remain in each source JSON file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from rigging.filesystem.buckets import filesystem_for

FIELDS = (
    "step",
    "age",
    "position",
    "tokens",
    "engine_abs_mean",
    "stale_abs_mean",
    "combined_abs_mean",
    "engine_abs_p99",
    "stale_abs_p99",
    "combined_abs_p99",
    "engine_signed_mean",
    "stale_signed_mean",
    "combined_signed_mean",
    "opposite_sign_fraction",
    "canceled_absolute_mean",
    "tis_capped_fraction",
    "reconstruction_abs_max",
)


def _sources(prefix: str) -> list[tuple[int, object, str]]:
    if prefix.startswith("s3://"):
        fs, path = filesystem_for(prefix)
        names = fs.ls(path)
    else:
        fs = None
        names = [str(path) for path in Path(prefix).glob("global_step_*.json.gz")]
    result = []
    for name in names:
        match = re.fullmatch(r"global_step_(\d+)\.json\.gz", str(name).rsplit("/", 1)[-1])
        if match:
            result.append((int(match.group(1)), fs, str(name)))
    return sorted(result)


def _read(fs: object, path: str) -> dict:
    opener = fs.open if fs is not None else open
    with opener(path, "rb") as source:
        with gzip.GzipFile(fileobj=source) as compressed:
            return json.load(compressed)


def summarize(prefix: str) -> list[dict]:
    sources = _sources(prefix)
    if not sources:
        raise ValueError(f"no diagnostic records in {prefix}")
    if len({step for step, _, _ in sources}) != len(sources):
        raise ValueError("duplicate diagnostic step")
    rows = []
    for step, fs, path in sources:
        record = _read(fs, path)
        if record["schema_version"] not in (1, 2) or record["consuming_step"] != step:
            raise ValueError(f"unexpected diagnostic schema or step in {path}")
        if record["reference_version"] != 0 or record["other_version_tokens"]:
            raise ValueError(f"{path}: this fixed-version analysis requires only published version zero")
        if record["reference_tokens"] != record["selected_tokens"]:
            raise ValueError(f"{path}: selected tokens lack a matching-weight trainer score")
        if record["schema_version"] == 2 and record["matched_tokens"] != record["selected_tokens"]:
            raise ValueError(f"{path}: selected tokens lack a matched B score")
        expected_age = step - 1
        if record["summaries"]["all"] != record["summaries"][f"age_{expected_age}"]:
            raise ValueError(f"{path}: unexpected policy age")
        for position in ("all", "first_256", "middle", "last_256"):
            item = record["summaries"][position]
            if not item["tokens"]:
                continue
            if item["reconstruction_abs_max"] > 1e-10:
                raise ValueError(f"{path}: A/B/C terms do not reconstruct")
            row = {"step": step, "age": expected_age, "position": position, "tokens": item["tokens"]}
            for component in ("engine", "stale", "combined"):
                for source_key, output_key in (
                    ("log_ratio_abs_mean", "abs_mean"),
                    ("log_ratio_abs_p99", "abs_p99"),
                    ("log_ratio_mean", "signed_mean"),
                ):
                    row[f"{component}_{output_key}"] = item[component][source_key]
            for name in (
                "opposite_sign_fraction",
                "canceled_absolute_mean",
                "tis_capped_fraction",
                "reconstruction_abs_max",
            ):
                row[name] = item[name]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local or S3 mismatch_decomposition export directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", type=Path, help="Optional age-curve SVG")
    args = parser.parse_args()
    rows = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.figure is not None:
        age_rows = [row for row in rows if row["position"] == "all"]
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        for component, label in (
            ("engine", "B - A: matching-weight engine gap"),
            ("stale", "C - B: stale-weight drift"),
            ("combined", "C - A: combined gap"),
        ):
            ax.plot(
                [row["age"] for row in age_rows],
                [row[f"{component}_abs_mean"] for row in age_rows],
                marker="o",
                markersize=3,
                label=label,
            )
        ax.set(xlabel="Optimizer updates since generating version", ylabel="Mean absolute token log-probability gap")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure)
        plt.close(fig)
        if args.figure.suffix.lower() == ".svg":
            svg = args.figure.read_text()
            args.figure.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


if __name__ == "__main__":
    main()
