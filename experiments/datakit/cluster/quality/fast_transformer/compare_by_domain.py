# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two quality scorers by how they treat each domain.

The store partitions on ``cluster=<C>/quality=<Q>``, so a quality score is only ever
read inside a domain. Two properties follow, and this reports both for each scorer
over the same documents:

**Parity.** Every domain contains its own good and bad documents, so a scorer that
grades documents rather than domains should promote a broadly similar share of each
domain to its top buckets. The spread of that share across domains is the measure —
the deployed scorer admits 61.3% of math and 0.0% of safety, which means selecting
"high quality" data currently means selecting some domains and discarding others.
Lower spread is better, and a scorer that collapses everything to one bucket also
scores a low spread, which is why the next measure is reported beside it.

**Discrimination.** Within a domain, the scores must actually separate documents. A
scorer that assigns a domain one near-constant score carries no information about it
however good its parity looks. Reported as the per-domain score standard deviation,
and as the share of domains below ``FLAT_STD`` — the variance gate the stage report
already uses to flag a source as uninformative.

Neither measure needs labels, so both run before any oracle comparison. They do not
establish that a scorer is *right* — that is what the held-out oracle agreement and
the intruder test are for — only whether it is usable per domain.
"""

import argparse
import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

# Buckets 3 and 4 of 0..4: what a selection pass keeps when it wants the good half.
TOP_BUCKETS = (3, 4)
# Below this within-domain standard deviation the scorer is not discriminating; the
# quality stage report uses the same threshold to flag a source "uninformative".
FLAT_STD = 0.03
GROUPINGS = ("domain_id", "cluster_5000", "source")


def _read_dir(path: str, columns: list[str]) -> dict[str, list]:
    """Concatenate ``columns`` across every parquet shard under ``path``."""
    out: dict[str, list] = {c: [] for c in columns}
    shards = sorted(str(m) for m in StoragePath(f"{path.rstrip('/')}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no parquet shards under {path}")
    for shard in shards:
        with StoragePath(shard).open("rb") as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        for c in columns:
            out[c].extend(table.column(c).to_pylist())
    return out


def _group_stats(groups: list, scores: np.ndarray, buckets: np.ndarray) -> dict[str, float]:
    """Parity and discrimination over one grouping."""
    keys = np.array([str(g) for g in groups])
    top_shares, stds = [], []
    for key in np.unique(keys):
        mask = keys == key
        if mask.sum() < 2:
            continue
        top_shares.append(float(np.isin(buckets[mask], TOP_BUCKETS).mean()))
        stds.append(float(scores[mask].std()))
    top = np.array(top_shares)
    sd = np.array(stds)
    return {
        "groups": float(len(top)),
        "top_share_mean": float(top.mean()),
        "top_share_std": float(top.std()),
        "top_share_min": float(top.min()),
        "top_share_max": float(top.max()),
        "top_share_range": float(top.max() - top.min()),
        "within_std_median": float(np.median(sd)),
        "flat_group_share": float((sd < FLAT_STD).mean()),
    }


def compare(*, docs_dir: str, scored: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    """Per-grouping stats for every named scorer, joined to the documents by id."""
    meta = _read_dir(docs_dir, ["id", *GROUPINGS])
    by_id = {doc_id: i for i, doc_id in enumerate(meta["id"])}
    logger.info("compare_by_domain: %d evaluation documents", len(by_id))

    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, path in scored.items():
        cols = _read_dir(f"{path.rstrip('/')}/outputs/main", ["id", "score", "quality_bucket"])
        rows = [
            (by_id[i], s, b)
            for i, s, b in zip(cols["id"], cols["score"], cols["quality_bucket"], strict=True)
            if i in by_id
        ]
        if not rows:
            raise ValueError(f"{name}: no scored ids matched the evaluation set")
        idx = np.array([r[0] for r in rows])
        scores = np.array([r[1] for r in rows], dtype=np.float64)
        buckets = np.array([r[2] for r in rows], dtype=np.int64)
        logger.info("compare_by_domain: %s matched %d/%d documents", name, len(rows), len(by_id))
        report[name] = {g: _group_stats([meta[g][i] for i in idx], scores, buckets) for g in GROUPINGS}
    return report


def _print_report(report: dict[str, dict[str, dict[str, float]]]) -> None:
    for grouping in GROUPINGS:
        print(f"\n=== grouped by {grouping} ===")
        header = f"{'scorer':16} {'n':>5} {'top%mean':>9} {'top%spread':>11}"
        print(f"{header} {'top%min':>8} {'top%max':>8} {'wstd':>7} {'flat%':>7}")
        for name, per_group in report.items():
            s = per_group[grouping]
            print(
                f"{name:16} {int(s['groups']):>5} {s['top_share_mean']:>8.1%} {s['top_share_std']:>10.3f} "
                f"{s['top_share_min']:>7.1%} {s['top_share_max']:>7.1%} {s['within_std_median']:>7.3f} "
                f"{s['flat_group_share']:>6.1%}"
            )
    print("\ntop%spread: lower is better parity. wstd: higher is better discrimination.")
    print(f"flat%: share of groups the scorer cannot discriminate within (std < {FLAT_STD}).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", required=True, help="evaluation set with id + domain_id + cluster_5000 + source")
    parser.add_argument(
        "--scored",
        required=True,
        action="append",
        metavar="NAME=PATH",
        help="a scorer's output root, repeatable (e.g. --scored v0=s3://... --scored v1=s3://...)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    scored = dict(pair.split("=", 1) for pair in args.scored)
    _print_report(compare(docs_dir=args.docs_dir, scored=scored))


if __name__ == "__main__":
    main()
