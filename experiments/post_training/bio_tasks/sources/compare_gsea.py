# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete independently executed, pinned GSEA preparations."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

NUMERIC_COLUMNS = ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for name in ["gsea-results.tsv", "top5-absolute-nes.tsv", "top10-significant.tsv"]:
        tables = []
        for directory in [args.baseline, args.repeat]:
            with (directory / name).open() as stream:
                tables.append(list(csv.DictReader(stream, delimiter="\t")))
        left, right = tables
        assert len(left) == len(right) > 0
        assert set(left[0]) == set(right[0]) and set(NUMERIC_COLUMNS) <= left[0].keys()
        for table in tables:
            assert len({r["Term"] for r in table}) == len(table)
        if name != "gsea-results.tsv":
            assert [r["Term"] for r in left] == [r["Term"] for r in right], name
        left_by_id = {r["Term"]: r for r in left}
        right_by_id = {r["Term"]: r for r in right}
        assert left_by_id.keys() == right_by_id.keys()
        errors = dict.fromkeys(NUMERIC_COLUMNS, 0.0)
        for term, row in left_by_id.items():
            observed = right_by_id[term]
            for key in row:
                if key not in NUMERIC_COLUMNS:
                    assert row[key] == observed[key], (name, term, key)
                    continue
                a, b = float(row[key]), float(observed[key])
                assert math.isfinite(a) and math.isfinite(b)
                error = abs(a - b)
                assert error <= 1e-12, (name, term, key, a, b)
                errors[key] = max(errors[key], error)
        reports[name] = {"rows": len(left), "maximum_absolute_errors": errors}
    for name in ["ranking.tsv", "set-status.tsv"]:
        assert (args.baseline / name).read_bytes() == (args.repeat / name).read_bytes(), name
    gmt_hashes = []
    for directory in [args.baseline, args.repeat]:
        with gzip.open(directory / "go-bp.gmt.gz", "rb") as stream:
            gmt_hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    assert gmt_hashes[0] == gmt_hashes[1]
    report = {
        "status": "pinned-native-repeatability-passed",
        "tables": reports,
        "rank_and_library_identity": "Exact ranks, set eligibility and decompressed GMT matched.",
        "limits": "Two native runs with the same inputs, seed and package lock; no independent null-distribution audit.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
