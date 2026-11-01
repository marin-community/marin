# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-eval contamination of what the hero run trained on (Gate 6).

Joins `aggregate.py` output with `kept.py` flags. An item counts as found in the
training data at threshold ``t`` when a document the store kept holds at least
``t`` of the item's features. Writes ``item_status.parquet`` (one row per item,
with its best kept coverage and that document's source) and prints the tables.

Usage: report_tables.py AGG_DIR KEPT_DIR OUT_DIR
"""

import sys

import duckdb

THRESHOLDS = (0.5, 0.8)


def main() -> None:
    agg_dir, kept_dir, out_dir = sys.argv[1:4]
    con = duckdb.connect()
    con.execute("SET threads TO 4; SET memory_limit = '8GB'")
    con.execute(f"CREATE VIEW items AS SELECT * FROM read_parquet('{agg_dir}/items.parquet')")
    con.execute(f"CREATE VIEW kept AS SELECT * FROM read_parquet('{kept_dir}/*.parquet')")
    con.execute(
        f"""
        CREATE TABLE cand AS
        SELECT d.source, d.file, d.id, d.eval, d.eval_id, d.k / i.n AS cov
        FROM read_parquet('{agg_dir}/doc_item.parquet') d JOIN items i USING (eval, eval_id)
        WHERE d.k / i.n >= {min(THRESHOLDS)}
        """
    )
    con.execute(
        """
        CREATE TABLE cand_kept AS
        SELECT c.*, k.kept, k.contaminated, k.exact_dup, k.fuzzy_dup
        FROM cand c JOIN kept k ON k.path LIKE '%/' || c.file AND k.id = c.id
        """
    )
    unmatched = con.execute("SELECT (SELECT count(*) FROM cand) - (SELECT count(*) FROM cand_kept)").fetchone()[0]
    if unmatched:
        raise ValueError(f"{unmatched} candidate rows have no kept flag")
    con.execute(
        f"""
        COPY (
          SELECT i.eval, i.eval_id, i.best_doc_coverage AS best_coverage_all,
                 coalesce(max(c.cov), 0) AS best_coverage_kept,
                 arg_max(c.source, c.cov) AS best_kept_source
          FROM items i LEFT JOIN cand_kept c ON c.eval_id = i.eval_id AND c.kept
          GROUP BY ALL
        ) TO '{out_dir}/item_status.parquet'
        """
    )
    status = f"read_parquet('{out_dir}/item_status.parquet')"
    print(
        con.sql(
            f"""
            SELECT eval, count(*) AS items,
                   round(100 * avg((best_coverage_all >= 0.5)::INT), 1) AS all_ge50,
                   round(100 * avg((best_coverage_kept >= 0.5)::INT), 1) AS kept_ge50,
                   round(100 * avg((best_coverage_kept >= 0.8)::INT), 1) AS kept_ge80
            FROM {status} GROUP BY eval ORDER BY eval
            """
        ).fetchdf().to_string(index=False)
    )
    print(
        con.sql(
            """
            SELECT count(DISTINCT (file, id)) AS docs,
                   count(DISTINCT (file, id)) FILTER (WHERE contaminated) AS decon_dropped,
                   count(DISTINCT (file, id)) FILTER (WHERE NOT contaminated AND fuzzy_dup) AS fuzzy_dropped,
                   count(DISTINCT (file, id)) FILTER (WHERE NOT contaminated AND NOT fuzzy_dup AND exact_dup)
                     AS exact_dropped,
                   count(DISTINCT (file, id)) FILTER (WHERE kept) AS kept
            FROM cand_kept
            """
        ).fetchdf().to_string(index=False)
    )
    print(
        con.sql(
            f"""
            SELECT eval, best_kept_source AS source, count(*) AS items
            FROM {status} WHERE best_coverage_kept >= 0.5
            GROUP BY eval, best_kept_source
            QUALIFY row_number() OVER (PARTITION BY eval ORDER BY count(*) DESC) <= 3
            ORDER BY eval, items DESC
            """
        ).fetchdf().to_string(index=False)
    )


if __name__ == "__main__":
    main()
