# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a GLM label set is fit to train on, before spending a training run.

A label set that is quietly broken still produces a model, and that model still
produces plausible metrics. Two failures on this project got that far:

* Documents were hard-cut at a character cap, so the grader saw text ending
  mid-token, marked it damaged, and assigned quality 1. Length and quality
  correlated at Spearman -0.25, and 85% of the bottom bucket sat at the cap.
* Prompts longer than the server's context were rejected outright with a 400.
  Those rejections were counted as ordinary dropped documents, so the *labeled*
  set looked healthy — it had simply lost its longest documents.

The second one is why this compares against the input set rather than only
describing the output. Selective loss is invisible from the survivors alone: drop
every long document and the remaining length distribution is still perfectly
well-behaved. Only the input/output comparison shows the hole.

Checks that fail here are stated as failures rather than warnings, because the
cost of training on a poisoned set is a wasted run plus the far larger cost of
believing its evaluation numbers.
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from scipy import stats

logger = logging.getLogger(__name__)

# The excerpting cap in sample_labels; text at or above it was shortened.
CAP_CHARS = 12_000
# Below this share of the input, the labeling lost too much to trust.
MIN_COVERAGE = 0.95
# A grader that never spends its top score cannot rank the top of the distribution,
# which is the end data selection uses.
MIN_TOP_SHARE = 0.08
# Corrupt/truncated/junk. Well above this and the excerpting is damaging documents.
MAX_INVALID = 0.10
# Below this a document is a stub, and grading it poorly is correct rather than an
# artifact: measured over 22k labels, documents under 200 characters are 41.5%
# invalid and average 1.66. Length correlation is therefore only meaningful above
# a floor — including the stub tail produced +0.33 overall against +0.17 without it.
MIN_JUDGED_CHARS = 500
# The poisoning signature is directional: long documents scoring *worse*, because
# the grader read a hard cut as damage. Quality rising with length is ordinary
# signal and is not a failure. The two bounds are therefore asymmetric.
MIN_LENGTH_CORR = -0.10
# Above this, length is standing in for judgment rather than correlating with it.
# The healthy set measures +0.17 over the same subset.
MAX_LENGTH_CORR = 0.35
# The longest decile must not be lost relative to the input (the 400-rejection
# signature). Expressed as a ratio of long-document share, output over input.
MIN_LONG_RETENTION = 0.90


@dataclass(frozen=True)
class Check:
    """One gate condition and what it measured."""

    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _read(path: str, columns: list[str]) -> dict[str, list]:
    """Read columns from a parquet file or a directory of shards."""
    target = path.rstrip("/")
    shards = (
        [target] if target.endswith(".parquet") else sorted(str(m) for m in StoragePath(f"{target}/*.parquet").glob())
    )
    if not shards:
        raise ValueError(f"no parquet under {path}")
    out: dict[str, list] = {c: [] for c in columns}
    for shard in shards:
        with StoragePath(shard).open("rb") as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        for c in columns:
            out[c].extend(table.column(c).to_pylist())
    return out


def gate(*, labels_path: str, label_set_path: str) -> list[Check]:
    """Run every gate condition and return what each one measured."""
    labels = _read(labels_path, ["id", "quality", "valid", "content_type", "text"])
    source = _read(label_set_path, ["id", "text"])

    n_in, n_out = len(source["id"]), len(labels["id"])
    quality = np.asarray(labels["quality"], dtype=float)
    chars = np.asarray([len(t or "") for t in labels["text"]], dtype=float)
    in_chars = np.asarray([len(t or "") for t in source["text"]], dtype=float)

    checks = [
        Check(
            "coverage",
            n_out >= n_in * MIN_COVERAGE,
            f"{n_out}/{n_in} labeled ({n_out / n_in:.1%}, floor {MIN_COVERAGE:.0%})",
        ),
        Check(
            "top-of-scale used",
            (top := float((quality == 5).mean())) >= MIN_TOP_SHARE,
            f"{top:.1%} scored 5 (floor {MIN_TOP_SHARE:.0%})",
        ),
        Check(
            "invalid rate",
            (inv := float(np.mean([not v for v in labels["valid"]]))) <= MAX_INVALID,
            f"{inv:.1%} marked invalid (ceiling {MAX_INVALID:.0%})",
        ),
    ]

    # Truncation poisoning: length driving the score. Measured above the stub floor,
    # where a low score reflects the document rather than its size.
    judged = chars >= MIN_JUDGED_CHARS
    corr = float(stats.spearmanr(chars[judged], quality[judged]).statistic)
    checks.append(
        Check(
            "length does not drive quality",
            MIN_LENGTH_CORR <= corr <= MAX_LENGTH_CORR,
            f"Spearman(chars, quality) = {corr:+.3f} over {int(judged.sum())} documents "
            f"≥{MIN_JUDGED_CHARS} chars (band {MIN_LENGTH_CORR:+.2f}..{MAX_LENGTH_CORR:+.2f})",
        )
    )

    # Selective loss of long documents, visible only against the input.
    long_cut = float(np.quantile(in_chars, 0.9))
    in_long = float((in_chars >= long_cut).mean())
    out_long = float((chars >= long_cut).mean())
    retention = out_long / in_long if in_long else 0.0
    checks.append(
        Check(
            "long documents retained",
            retention >= MIN_LONG_RETENTION,
            f"top-decile share {out_long:.1%} out vs {in_long:.1%} in "
            f"(retention {retention:.2f}, floor {MIN_LONG_RETENTION})",
        )
    )

    # Documents at the cap must not be scored differently from the rest; that was
    # the exact shape of the first poisoning.
    at_cap = chars >= CAP_CHARS
    if at_cap.any() and not at_cap.all():
        gap = float(quality[at_cap].mean() - quality[~at_cap].mean())
        checks.append(
            Check(
                "excerpted documents not penalised",
                abs(gap) <= 0.5,
                f"{at_cap.mean():.1%} at cap, mean quality {gap:+.2f} vs the rest (|gap| ceiling 0.5)",
            )
        )

    return checks


def report_parity(labels_path: str) -> None:
    """Log the per-type quality spread — the rubric's parity goal, not a gate."""
    labels = _read(labels_path, ["quality", "content_type"])
    quality = np.asarray(labels["quality"], dtype=float)
    types = np.asarray(labels["content_type"])
    logger.info("per-type quality (parity target: similar top-share across types)")
    for t in sorted(set(types.tolist())):
        m = types == t
        logger.info(
            "  %-14s n=%-7d mean=%.2f  top-share=%.1f%%",
            t,
            int(m.sum()),
            quality[m].mean(),
            100 * (quality[m] == 5).mean(),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="GLM label output (file or shard directory)")
    parser.add_argument("--label-set", required=True, help="the input set the labels were drawn from")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    checks = gate(labels_path=args.labels, label_set_path=args.label_set)
    for check in checks:
        logger.info("%s", check.line())
    report_parity(args.labels)
    failed = [c.name for c in checks if not c.passed]
    if failed:
        raise SystemExit(f"label gate FAILED: {', '.join(failed)} — do not train on this set")
    logger.info("label gate passed: fit to train on")


if __name__ == "__main__":
    main()
