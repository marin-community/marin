# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List the documents `kept.py` checks: those holding enough of some eval item (Gate 5).

Reads ``doc_item.parquet`` and ``items.parquet`` from an `aggregate.py` output
directory and writes ``[{"source", "path", "id"}]`` for every document that
holds at least ``MIN_COVERAGE`` of some item's features.

Usage: kept_requests.py AGG_DIR FILTERS_JSON OUT_JSON
"""

import json
import sys
from pathlib import Path

import duckdb

MIN_COVERAGE = 0.5


def main() -> None:
    agg_dir, filters_path, out_path = sys.argv[1:4]
    filters = json.loads(Path(filters_path).read_text())
    rows = duckdb.sql(
        f"""
        SELECT DISTINCT d.source, d.file, d.id
        FROM read_parquet('{agg_dir}/doc_item.parquet') d
        JOIN read_parquet('{agg_dir}/items.parquet') i USING (eval, eval_id)
        WHERE d.k / i.n >= {MIN_COVERAGE}
        ORDER BY ALL
        """
    ).fetchall()
    requests = [{"source": s, "path": f"{filters[s]['dir']}/{f}", "id": i} for s, f, i in rows]
    Path(out_path).write_text(json.dumps(requests))
    print(f"{len(requests)} documents in {len({r['path'] for r in requests})} files")


if __name__ == "__main__":
    main()
