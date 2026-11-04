# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize `verbatim_copies.py` outputs: how often an eval item is copied verbatim.

For each item, takes its best kept document:

* ``qa``: the question and the answer verbatim in the same document;
* ``q_only``: the question verbatim, with no document holding both.

Also writes ``verbatim_docs.parquet``, one row per (document, item) pair with a
verbatim question: the training source (the ``normalized/<source>`` directory),
file, document id, and whether the answer is there too. It prints per eval how
many items each source copies.

A ``choice`` or ``near`` question counts as verbatim when its stem is found and is at least
``MIN_STEM_WORDS`` long, or is followed by at least half its distinctive choices,
since a short stem can occur by chance. ``qa_ngram`` also accepts a ``free``
answer at ``ANSWER_NGRAM_SHARE`` 13-gram coverage, for copies that reformat the
answer (GSM8K's calculator notes, line breaks).

Usage: verbatim_report.py ITEMS_JSON COPIES_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

MIN_STEM_WORDS = 10
ANSWER_NGRAM_SHARE = 0.8
SOURCE_PATTERN = "normalized/([^/]+)/"


def main() -> None:
    items_path, copies_dir, out_dir = sys.argv[1:4]
    items = json.loads(Path(items_path).read_text())
    con = duckdb.connect()
    con.register(
        "item",
        pa.table(
            {
                "eval_id": list(items),
                "eval": [item["eval"] for item in items.values()],
                "kind": [item["kind"] for item in items.values()],
            }
        ),
    )
    con.execute(
        f"""
        CREATE TABLE pair AS
        SELECT c.*, i.eval, i.kind,
               c.q_found AND (i.kind = 'free' OR c.q_words >= {MIN_STEM_WORDS}
                              OR (c.choices_total > 0 AND 2 * c.choices_found >= c.choices_total)) AS q_verbatim
        FROM read_parquet('{copies_dir}/*.parquet') c JOIN item i USING (eval_id)
        """
    )
    con.execute(
        f"""
        CREATE TABLE doc AS
        SELECT eval, eval_id, regexp_extract(path, '{SOURCE_PATTERN}', 1) AS source, path, id,
               a_found AS qa, a_found OR coalesce(a_ngram_coverage >= {ANSWER_NGRAM_SHARE}, false) AS qa_ngram
        FROM pair WHERE q_verbatim
        """
    )
    con.execute(
        f"""
        CREATE TABLE item_status AS
        SELECT i.eval, i.eval_id,
               coalesce(bool_or(p.q_verbatim AND p.a_found), false) AS qa,
               coalesce(bool_or(p.q_verbatim AND (p.a_found OR p.a_ngram_coverage >= {ANSWER_NGRAM_SHARE})), false)
                   AS qa_ngram,
               coalesce(bool_or(p.q_verbatim), false) AS q,
               count(DISTINCT p.path || p.id) FILTER (WHERE p.q_verbatim AND p.a_found) AS qa_pairs,
               count(DISTINCT p.path || p.id) FILTER (WHERE p.q_verbatim) AS q_pairs
        FROM item i LEFT JOIN pair p USING (eval_id)
        GROUP BY ALL
        """
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY item_status TO '{out_dir}/verbatim_item_status.parquet'")
    con.execute(f"COPY (SELECT * FROM doc ORDER BY ALL) TO '{out_dir}/verbatim_docs.parquet'")
    print(
        con.sql(
            """
            SELECT eval, count(*) AS items,
                   sum(qa::INT) AS qa, round(100 * avg(qa::INT), 1) AS qa_pct,
                   sum(qa_ngram::INT) AS qa_ngram,
                   sum((q AND NOT qa)::INT) AS q_only, round(100 * avg((q AND NOT qa)::INT), 1) AS q_only_pct,
                   sum(q::INT) AS q_any, round(100 * avg(q::INT), 1) AS q_any_pct,
                   sum(qa_pairs) AS qa_pairs, sum(q_pairs) AS q_pairs
            FROM item_status GROUP BY ALL ORDER BY eval
            """
        )
    )
    con.sql(
        """
            SELECT eval, source,
                   count(DISTINCT eval_id) FILTER (WHERE qa) AS qa_items,
                   count(DISTINCT eval_id) AS q_items,
                   count(DISTINCT path || id) AS docs
            FROM doc GROUP BY ALL ORDER BY eval, q_items DESC
            """
    ).show(max_rows=200)


if __name__ == "__main__":
    main()
