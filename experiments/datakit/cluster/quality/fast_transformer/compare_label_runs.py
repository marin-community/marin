# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-retest comparison of two window-label runs over the same windows.

The oracle's self-agreement bounds every downstream model: a scorer cannot
rank windows more consistently than the grader that labeled them. Grading the
same windows twice under the identical setup (model, rubric, temperature 0)
and correlating the two quality columns measures that noise ceiling directly.

Reports Spearman rho, exact-agreement rate, within-1 agreement, and the joint
quality distribution over windows labeled in both runs.
"""

import argparse
import json
import logging
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.metrics import spearman_rho

logger = logging.getLogger(__name__)

COLUMNS = ["id", "window", "quality", "content_type", "valid"]


def _load(path: str) -> dict[tuple[str, str], dict]:
    with StoragePath(path).open("rb") as fh:
        rows = pq.ParquetFile(fh).read(columns=COLUMNS).to_pylist()
    return {(r["id"], r["window"]): r for r in rows}


def compare(a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict]) -> dict:
    shared = sorted(set(a) & set(b))
    qa = [float(a[k]["quality"]) for k in shared]
    qb = [float(b[k]["quality"]) for k in shared]
    exact = sum(1 for x, y in zip(qa, qb, strict=True) if x == y)
    within1 = sum(1 for x, y in zip(qa, qb, strict=True) if abs(x - y) <= 1)
    type_agree = sum(1 for k in shared if a[k]["content_type"] == b[k]["content_type"])
    joint = Counter((int(x), int(y)) for x, y in zip(qa, qb, strict=True))
    return {
        "windows_in_both": len(shared),
        "quality_spearman": spearman_rho(qa, qb),
        "quality_exact_agreement": exact / len(shared),
        "quality_within_1": within1 / len(shared),
        "content_type_agreement": type_agree / len(shared),
        "joint_quality_counts": {f"{x}-{y}": n for (x, y), n in sorted(joint.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--out", default=None, help="optional JSON path for the comparison")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    result = compare(_load(args.run_a), _load(args.run_b))
    logger.info("compare_label_runs: %s", json.dumps(result, indent=2))
    if args.out:
        with StoragePath(args.out).open("w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
