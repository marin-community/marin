# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check H100 and TPU GSM8K scores on the same 100 items."""

import argparse
import json
import math

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def check_scores(h100: dict, tpu: dict, tolerance: float = 0.05) -> bool:
    """Print both scores; reject missing/different items instead of passing partial runs."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Tolerance must be between 0 and 1")
    if h100["hardware"] != "h100" or tpu["hardware"] not in ("v5p", "v4"):
        raise ValueError("Expected an H100 result and a v5p/v4 result")
    if (h100["model"], h100["revision"]) != (tpu["model"], tpu["revision"]):
        raise ValueError("Model checkpoints differ")
    scores, identities = [], []
    for result in (h100, tpu):
        items = result["items"]
        identity = {item["id"]: item["sha256"] for item in items}
        if len(items) != 100 or len(identity) != 100 or any(item["correct"] not in (0, 1) for item in items):
            raise ValueError("Expected 100 unique, scored GSM8K items")
        scores.append(sum(item["correct"] for item in items) / 100)
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError("The runs evaluated different GSM8K items")
    passed = abs(scores[0] - scores[1]) <= tolerance + 1e-12
    print(
        f"{'PASS' if passed else 'FAIL'}: H100={scores[0]:.3f}, TPU={scores[1]:.3f}, "
        f"tolerance={tolerance:.3f}, hardware={tpu['hardware']}"
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", required=True, help="H100 result.json path")
    parser.add_argument("--tpu", required=True, action="append", help="TPU result.json path; repeat for v5p and v4")
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="Absolute accuracy difference; default five points"
    )
    args = parser.parse_args()
    configure_coreweave_s3()
    h100 = json.loads(StoragePath(args.h100).read_text())
    passed = [check_scores(h100, json.loads(StoragePath(path).read_text()), args.tolerance) for path in args.tpu]
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
