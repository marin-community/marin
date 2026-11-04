# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List the (document, item) pairs `verbatim_copies.py` checks.

A pair is a candidate when the document holds at least the item bar of the
item's indexed 13-grams (``ITEM_BARS``), or at least ``QUESTION_SHARE`` of the
question's own 13-grams. The second test finds documents that copy only the
question of an item with a long answer, which stay under the item bar. ``k``
counts matches anywhere in the item, so the test applies only to questions with
at least ``QUESTION_MIN_NGRAMS`` 13-grams and without a quoted passage, where
the choices or the passage alone would pass it.

Both read ``doc_item.parquet`` and ``items.parquet`` from an `aggregate.py`
output directory and ITEMS_JSON from `verbatim_items.py`:

* ``kept``: write ``[{"source", "path", "id"}]`` for the candidate documents
  missing from KEPT_DIR, the input of `kept.py`.
* ``copies``: write ``[{"path", "id", "eval_id"}]`` for the candidate pairs whose
  document the store kept, per the `kept.py` outputs in KEPT_DIRS.

Usage:
    verbatim_requests.py kept AGG_DIR ITEMS_JSON FILTERS_JSON OUT_JSON KEPT_DIR
    verbatim_requests.py copies AGG_DIR ITEMS_JSON FILTERS_JSON OUT_JSON KEPT_DIR...
"""

import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

from experiments.contam_report.humaneval_copies import NGRAM_LENGTH

ITEM_BARS = {"gsm8k": 0.3, "humaneval": 0.3, "mmlu": 0.5, "mmlu_pro": 0.5}
QUESTION_SHARE = 0.6
QUESTION_MIN_NGRAMS = 10
PARAGRAPH = "\n\n"


def question_ngrams(question: str) -> int:
    """Count the question's distinct 13-grams with a letter, per paragraph, as the decon index does."""
    grams = set()
    for paragraph in question.split(PARAGRAPH):
        tokens = paragraph.split()
        for i in range(len(tokens) - NGRAM_LENGTH + 1):
            gram = tuple(tokens[i : i + NGRAM_LENGTH])
            if any(c.isalpha() for token in gram for c in token):
                grams.add(gram)
    return len(grams)


def candidates(con: duckdb.DuckDBPyConnection, agg_dir: str, items_path: str, filters_path: str) -> None:
    items = json.loads(Path(items_path).read_text())
    filters = json.loads(Path(filters_path).read_text())
    con.register(
        "question",
        pa.table(
            {
                "eval_id": list(items),
                "q_n": [question_ngrams(item["question"]) for item in items.values()],
                "has_passage": [item.get("has_passage", False) for item in items.values()],
            }
        ),
    )
    con.register("bar", pa.table({"eval": list(ITEM_BARS), "bar": list(ITEM_BARS.values())}))
    con.register("source_dir", pa.table({"source": list(filters), "dir": [f["dir"] for f in filters.values()]}))
    con.execute(
        f"""
        CREATE TABLE candidate AS
        SELECT d.source, s.dir || '/' || d.file AS path, d.id, d.eval_id
        FROM read_parquet('{agg_dir}/doc_item.parquet') d
        JOIN read_parquet('{agg_dir}/items.parquet') i USING (eval, eval_id)
        JOIN bar b USING (eval)
        JOIN question q USING (eval_id)
        JOIN source_dir s USING (source)
        WHERE d.k / i.n >= b.bar
           OR (q.q_n >= {QUESTION_MIN_NGRAMS} AND NOT q.has_passage AND d.k >= {QUESTION_SHARE} * q.q_n)
        """
    )


def kept_table(kept_dirs: list[str]) -> str:
    files = ", ".join(f"'{d}/*.parquet'" for d in kept_dirs)
    return f"(SELECT DISTINCT path, id, kept FROM read_parquet([{files}]))"


def main() -> None:
    command, agg_dir, items_path, filters_path, out_path, *kept_dirs = sys.argv[1:]
    con = duckdb.connect()
    candidates(con, agg_dir, items_path, filters_path)
    kept = kept_table(kept_dirs)
    if command == "kept":
        rows = con.execute(
            f"SELECT DISTINCT source, path, id FROM candidate c ANTI JOIN {kept} k USING (path, id) ORDER BY ALL"
        ).fetchall()
        requests = [{"source": s, "path": p, "id": i} for s, p, i in rows]
    elif command == "copies":
        rows = con.execute(
            f"SELECT c.path, c.id, c.eval_id FROM candidate c JOIN {kept} k USING (path, id) WHERE k.kept ORDER BY ALL"
        ).fetchall()
        requests = [{"path": p, "id": i, "eval_id": e} for p, i, e in rows]
    else:
        raise ValueError(f"unknown command {command!r}")
    Path(out_path).write_text(json.dumps(requests))
    print(f"{command}: {len(requests)} requests in {len({r['path'] for r in requests})} files")


if __name__ == "__main__":
    main()
