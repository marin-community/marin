# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a testbed decon run to a compact JSON for the decon viewer (:mod:`experiments.datakit.decontam.viewer.report`).

Runs where the testbed marks live. Rebuilds the testbed decon DAG (``--target-tokens-b`` and ``--exclude`` must
match the run) to locate each source's marks, then per source takes the flag rate from the mark counters and
samples flagged records from the mark-time flagged sample: the excerpt of the record's best pair, its score,
and the matched items with their texts from the item table.

    uv run iris --cluster=cw-rno2a job run -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/viewer/export_run.py \\
           --target-tokens-b 0.1 --exclude finetranslations ghalogs/public \\
           --label baseline --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/runs

The JSON is written to ``<out>/<label>.json``.
"""

import argparse
import json
import logging
import random
from collections import Counter

from marin.datakit.decon import DeconAttributes, DeconKeys
from marin.datakit.decon_tables import parquet_files, read_table
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.decontam.pair_text import ItemText, item_texts
from experiments.datakit.testbed.decon_arm import build_testbed_decon_steps

logger = logging.getLogger(__name__)

SAMPLES_PER_SOURCE = 60
MAX_MATCHED_ITEMS = 5
TOP_EVALS = 12
_FLAGGED_COLUMNS = ["id", "text", "max_score", "matched_items", "matched_evals"]


def sample_flagged(flagged_dir: str, limit: int, rng: random.Random) -> list[dict]:
    """Up to *limit* flagged records of one source, from its shuffled flagged-sample files."""
    files = parquet_files(flagged_dir)
    rng.shuffle(files)
    rows: list[dict] = []
    for path in files:
        rows += read_table(path, _FLAGGED_COLUMNS).to_pylist()
        if len(rows) >= limit:
            break
    return rows[:limit]


def item_evals(items_dir: str) -> dict[str, str]:
    """The eval of every item of the item table, by item id."""
    evals: dict[str, str] = {}
    for path in parquet_files(items_dir):
        columns = read_table(path, ["id", "eval"]).to_pydict()
        evals.update(zip(columns["id"], columns["eval"], strict=True))
    return evals


def flagged_doc(row: dict, texts: dict[str, ItemText], evals: dict[str, str]) -> dict:
    """One flagged record for the viewer: its excerpt, score, evals, and its matched items' texts."""
    items = list(row["matched_items"])[:MAX_MATCHED_ITEMS]
    return {
        "id": str(row["id"]),
        "max_score": float(row["max_score"]),
        "evals": list(row["matched_evals"]),
        # The viewer highlights the question of each matched item in both columns.
        "highlights": [texts[item].question for item in items if item in texts],
        "matched_items": [
            {"item": item, "eval": evals.get(item, ""), "text": texts[item].question if item in texts else ""}
            for item in items
        ],
        "text": str(row["text"]),
    }


def export_marks(marks: dict[str, StepSpec], items_dir: str, samples_per_source: int, seed: int) -> list[dict]:
    """Per source: record and flag counts from the mark counters, the most matched evals, and sampled flagged
    records (:func:`flagged_doc`), sorted by flag rate."""
    texts, evals_of = item_texts(items_dir), item_evals(items_dir)
    sources = []
    for name, step in sorted(marks.items()):
        attrs = read_artifact(step.output_path, DeconAttributes)
        contaminated = int(attrs.counters.get("decon/contaminated", 0))
        total = contaminated + int(attrs.counters.get("decon/clean", 0))
        rows = sample_flagged(str(attrs.flagged_output_dir), samples_per_source, random.Random(f"{seed}:{name}"))
        evals = Counter(eval_name for row in rows for eval_name in row["matched_evals"])
        sources.append(
            {
                "name": name,
                "docs": total,
                "flagged": contaminated,
                "rate": contaminated / total if total else 0.0,
                "top_evals": evals.most_common(TOP_EVALS),
                "samples": [flagged_doc(row, texts, evals_of) for row in rows],
            }
        )
        logger.info("source %s: %d/%d flagged", name, contaminated, total)
    return sorted(sources, key=lambda source: source["rate"], reverse=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens-b", type=float, required=True)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--label", required=True, help="run label (json basename)")
    parser.add_argument("--out", required=True, help="output dir (JSON written to <out>/<label>.json)")
    parser.add_argument("--only", nargs="*", default=None, help="restrict export to these sources")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SOURCE)
    parser.add_argument(
        "--sample-root", default=None, help="match the decon run's --sample-root (pre-materialized root)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    steps = build_testbed_decon_steps(
        target_total_tokens_b=args.target_tokens_b,
        only_sources=args.only,
        exclude_sources=frozenset(args.exclude or ()),
        sample_root=args.sample_root,
    )
    items_dir = str(read_artifact(steps.keys.output_path, DeconKeys).items_dir)
    run = {
        "label": args.label,
        "target_tokens_b": args.target_tokens_b,
        "exclude": args.exclude or [],
        "sources": export_marks(steps.marks, items_dir, args.samples, args.seed),
    }
    out_path = prefix_join(args.out, f"{args.label}.json")
    fs, _ = url_to_fs(out_path)
    with fs.open(out_path, "w") as f:
        json.dump(run, f)
    logger.info("wrote %s (%d sources)", out_path, len(run["sources"]))


if __name__ == "__main__":
    main()
