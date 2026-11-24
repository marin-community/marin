# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export completed reference-pipeline decontamination marks for inspection.

Run this command after the mark steps complete::

    uv run python experiments/datakit/decontam/viewer/export_reference_run.py \
        --minimum-sources 292 --label v4 --out <report-directory>
"""

import argparse
import html
import json
import logging
import time
from bisect import bisect_left
from collections import Counter

from marin.datakit.decon import DeconKeys
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import step_is_built
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.decontam.prepare_eval_corpus import AA_BENCHMARK_NAMES, EVAL_CORPUS_VERSION
from experiments.datakit.decontam.viewer.export_run import export_marks
from experiments.datakit.decontam.viewer.report import render_run
from experiments.datakit.reference_pipeline import decontamination_steps, select_sources

logger = logging.getLogger(__name__)

GALLERY_ROWS_PER_SOURCE = 5


def _wait_for_marks(all_marks: dict[str, StepSpec], minimum_sources: int, timeout: int) -> dict[str, StepSpec]:
    if minimum_sources > len(all_marks):
        raise ValueError(f"minimum sources {minimum_sources} exceeds the {len(all_marks)} configured marks")
    deadline = time.monotonic() + timeout
    while True:
        complete = {name: step for name, step in all_marks.items() if step_is_built(step)}
        logger.info("completed marks: %d/%d; gate=%d", len(complete), len(all_marks), minimum_sources)
        if len(complete) >= minimum_sources:
            return complete
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"only {len(complete)}/{len(all_marks)} marks completed before the timeout")
        time.sleep(min(30, remaining))


def _rate_histogram(rates: list[float]) -> str:
    bounds = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1]
    labels = [
        "0",
        "(0,1e-8]",
        "(1e-8,1e-7]",
        "(1e-7,1e-6]",
        "(1e-6,1e-5]",
        "(1e-5,1e-4]",
        "(1e-4,1e-3]",
        "(1e-3,1e-2]",
        "(1e-2,1]",
    ]
    counts = [0] * len(labels)
    for rate in rates:
        counts[bisect_left(bounds, rate)] += 1
    maximum = max(counts) or 1
    rows = "".join(
        f"<div style='display:grid;grid-template-columns:110px 1fr 45px;gap:8px;margin:3px 0'>"
        f"<span>{html.escape(label)}</span>"
        f"<span style='background:#26384b;width:{100 * count / maximum:.1f}%'>&nbsp;</span>"
        f"<span>{count}</span></div>"
        for label, count in zip(labels, counts, strict=True)
    )
    return f"<div style='max-width:620px'>{rows}</div>"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-sources", type=int, required=True, help="wait for this many successful source marks")
    parser.add_argument("--timeout", type=int, default=28_800, help="maximum wait in seconds")
    parser.add_argument("--samples-per-source", type=int, default=100, help="maximum flagged rows per source")
    parser.add_argument("--seed", type=int, default=20260815, help="seed for source shard order")
    parser.add_argument("--label", required=True, help="label for the exported run")
    parser.add_argument("--out", required=True, help="directory for reports and sample provenance")
    args = parser.parse_args()
    if args.minimum_sources <= 0:
        parser.error("--minimum-sources must be positive")
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative")
    if args.samples_per_source <= 0:
        parser.error("--samples-per-source must be positive")

    steps = decontamination_steps(select_sources(None))
    complete = _wait_for_marks(steps.marks, args.minimum_sources, args.timeout)
    items_dir = str(read_artifact(steps.keys.output_path, DeconKeys).items_dir)
    per_source = export_marks(complete, items_dir, args.samples_per_source, args.seed)
    aa_hits: Counter[str] = Counter()
    for source in per_source:
        aa_hits.update(
            eval_name for doc in source["samples"] for eval_name in doc["evals"] if eval_name in AA_BENCHMARK_NAMES
        )
    manifest_rows = [
        {"source": source["name"], "id": doc["id"], "max_score": doc["max_score"], "evals": doc["evals"]}
        for source in per_source
        for doc in source["samples"]
    ]
    run = {
        "label": args.label,
        "sources": [{**source, "samples": source["samples"][:GALLERY_ROWS_PER_SOURCE]} for source in per_source],
    }
    exact_docs = sum(source["docs"] for source in per_source)
    exact_flagged = sum(source["flagged"] for source in per_source)
    coverage = (
        f"{len(complete)}/{len(steps.marks)} completed sources; exact counters cover {exact_docs:,} documents. "
        f"The sample manifest has {len(manifest_rows):,} flagged rows from shuffled shard files with seed {args.seed}."
    )
    aa_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{aa_hits.get(name, 0):,}</td></tr>" for name in AA_BENCHMARK_NAMES
    )
    summary = (
        "<section style='margin:0 0 18px;padding:12px;background:#171a21;border:1px solid #262b34'>"
        f"<h2 style='font-size:16px'>Coverage and sampling</h2><p>{html.escape(coverage)}</p>"
        f"<p>Eval corpus: {html.escape(EVAL_CORPUS_VERSION)}. Exact flagged count: {exact_flagged:,}. "
        "AA counts below count the flagged records of the sample only.</p>"
        "<h3 style='font-size:14px'>Source contamination-rate distribution</h3>"
        f"{_rate_histogram([source['rate'] for source in per_source])}"
        "<h3 style='font-size:14px'>Mandatory AA benchmarks in the sample</h3>"
        f"<table style='max-width:620px'><thead><tr><th>benchmark</th><th>sampled flagged records</th></tr></thead>"
        f"<tbody>{aa_rows}</tbody></table></section>"
    )
    page = render_run(run, summary_html=summary)

    output = args.out
    StoragePath(output).mkdirs()
    with StoragePath(prefix_join(output, "report.json")).open("w") as fh:
        json.dump(run, fh)
    with StoragePath(prefix_join(output, "sample_manifest.jsonl")).open("w") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")
    sampling = {
        "label": args.label,
        "eval_corpus_version": EVAL_CORPUS_VERSION,
        "items_dir": items_dir,
        "completed_sources": len(complete),
        "total_sources": len(steps.marks),
        "samples_per_source": args.samples_per_source,
        "gallery_rows_per_source": GALLERY_ROWS_PER_SOURCE,
        "seed": args.seed,
        "source_outputs": {source: step.output_path for source, step in complete.items()},
    }
    with StoragePath(prefix_join(output, "sampling.json")).open("w") as fh:
        json.dump(sampling, fh, indent=2)
    with StoragePath(prefix_join(output, "report.html")).open("w") as fh:
        fh.write(page)
    logger.info("wrote %s with %d sources and %d sampled flagged rows", output, len(complete), len(manifest_rows))


if __name__ == "__main__":
    main()
