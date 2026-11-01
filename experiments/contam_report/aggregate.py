# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn scan hits into per-eval-item contamination (Gate 5, also used on the pilot).

For each eval item, with ``n`` its indexed features:

* ``best_doc_coverage``: the largest share of the item's features that one
  training document contains. 1.0 means one document holds the whole item.
* ``corpus_coverage``: the share of the item's features found anywhere.

Features that occur in more than ``--common-doc-limit`` hit documents are dropped
first: they are common phrases (licence text, boilerplate), not item leaks.

Usage: aggregate.py BLOOM_DIR OUT_DIR HITS_DIR... --common-doc-limit N
Each HITS_DIR holds ``<source>/hits/*.parquet`` as `scan.py` writes it.
"""

import argparse

import duckdb

# An item counts as contaminated when one document holds at least this share of it.
CONTAMINATED_COVERAGE = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bloom_dir")
    parser.add_argument("out_dir")
    parser.add_argument("hits_dirs", nargs="+")
    parser.add_argument("--common-doc-limit", type=int, required=True)
    args = parser.parse_args()
    bloom_dir, out_dir = args.bloom_dir, args.out_dir
    hits = " UNION ALL ".join(
        f"""
        SELECT regexp_extract(filename, '{d.rstrip("/")}/(.+)/hits/', 1) AS source,
               regexp_extract(filename, '/hits/([^/]+)$', 1) AS file, id, unnest(hits) AS hash, marked
        FROM read_parquet('{d.rstrip("/")}/**/hits/*.parquet', filename = true)
        """
        for d in args.hits_dirs
    )
    con = duckdb.connect()
    con.execute(
        f"SET threads TO 4; SET memory_limit = '12GB'; SET preserve_insertion_order = false;"
        f" SET temp_directory = '{out_dir}/duckdb_tmp'"
    )
    con.execute(
        f"""
        CREATE TABLE idx AS
        SELECT hash, eval_id, split_part(eval_id, '-test-', 1) AS eval
        FROM read_parquet('{bloom_dir}/_bloom/eval_hash_index.parquet')
        """
    )
    con.execute(
        "UPDATE idx SET eval = regexp_extract(eval_id, '^(uncheatable-[a-z0-9_]+)-\\d+$', 1) WHERE eval_id LIKE 'uncheatable-%'"
    )
    con.execute("CREATE TABLE item_n AS SELECT eval, eval_id, count(DISTINCT hash) AS n FROM idx GROUP BY ALL")
    # Each hit list holds distinct features, so a feature's row count is its document count.
    # Counting first keeps only the rare features' rows; common ones hold most of the hits.
    con.execute(f"CREATE TABLE hash_docs AS SELECT hash, count(*) AS docs FROM ({hits}) GROUP BY hash")
    con.execute(f"COPY hash_docs TO '{out_dir}/hash_docs.parquet'")
    con.execute(f"CREATE TABLE common AS SELECT hash FROM hash_docs WHERE docs > {args.common_doc_limit}")
    con.execute(f"CREATE TABLE doc_hash AS SELECT * FROM ({hits}) WHERE hash NOT IN (SELECT hash FROM common)")
    con.execute(
        """
        CREATE TABLE doc_item AS
        SELECT d.source, d.file, d.id, d.marked, i.eval, i.eval_id, count(DISTINCT d.hash) AS k
        FROM doc_hash d JOIN idx i USING (hash)
        GROUP BY ALL
        """
    )
    con.execute(
        """
        CREATE TABLE item AS
        SELECT n.eval, n.eval_id, n.n,
               coalesce(max(di.k), 0) / n.n AS best_doc_coverage,
               count(DISTINCT di.source || di.id) AS docs,
               arg_max(di.source, di.k) AS best_source,
               arg_max(di.id, di.k) AS best_doc,
               arg_max(di.file, di.k) AS best_file
        FROM item_n n LEFT JOIN doc_item di USING (eval, eval_id)
        GROUP BY n.eval, n.eval_id, n.n
        """
    )
    con.execute(
        """
        CREATE TABLE corpus AS
        SELECT i.eval_id, count(DISTINCT d.hash) AS found
        FROM idx i JOIN doc_hash d USING (hash)
        GROUP BY ALL
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT item.*, coalesce(corpus.found, 0) / item.n AS corpus_coverage
          FROM item LEFT JOIN corpus USING (eval_id)
        ) TO '{out_dir}/items.parquet'
        """
    )
    con.execute(f"COPY doc_item TO '{out_dir}/doc_item.parquet'")
    summary = con.execute(
        f"""
        SELECT eval, count(*) AS items,
               round(avg((best_doc_coverage >= {CONTAMINATED_COVERAGE})::INT) * 100, 2) AS pct_items_ge_50,
               round(avg((best_doc_coverage >= 0.8)::INT) * 100, 2) AS pct_items_ge_80,
               round(avg((best_doc_coverage > 0)::INT) * 100, 2) AS pct_items_any
        FROM read_parquet('{out_dir}/items.parquet') GROUP BY eval ORDER BY eval
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    print("common features dropped:", con.execute("SELECT count(*) FROM common").fetchone()[0])


if __name__ == "__main__":
    main()
