"""Turn scan hits into per-eval-item contamination (Gate 5, also used on the pilot).

For each eval item, with ``n`` its indexed features:

* ``best_doc_coverage``: the largest share of the item's features that one
  training document contains. 1.0 means one document holds the whole item.
* ``corpus_coverage``: the share of the item's features found anywhere.

Features that occur in more than ``COMMON_DOC_LIMIT`` hit documents are dropped
first: they are common phrases (licence text, boilerplate), not item leaks.

Usage: aggregate.py LOCAL_HITS_DIR BLOOM_DIR OUT_DIR
LOCAL_HITS_DIR holds ``<source>/hits/*.parquet`` as `scan.py` writes it.
"""

import sys

import duckdb

COMMON_DOC_LIMIT = 1000
# An item counts as contaminated when one document holds at least this share of it.
CONTAMINATED_COVERAGE = 0.5


def main() -> None:
    hits_dir, bloom_dir, out_dir = sys.argv[1:4]
    con = duckdb.connect()
    con.execute("SET threads TO 4; SET memory_limit = '12GB'")
    con.execute(
        f"""
        CREATE TABLE idx AS
        SELECT hash, eval_id, split_part(eval_id, '-test-', 1) AS eval
        FROM read_parquet('{bloom_dir}/_bloom/eval_hash_index.parquet')
        """
    )
    con.execute("UPDATE idx SET eval = regexp_extract(eval_id, '^(uncheatable-[a-z0-9_]+)-\\d+$', 1) WHERE eval_id LIKE 'uncheatable-%'")
    con.execute("CREATE TABLE item_n AS SELECT eval, eval_id, count(DISTINCT hash) AS n FROM idx GROUP BY ALL")
    con.execute(
        f"""
        CREATE TABLE doc_hash AS
        SELECT regexp_extract(filename, '{hits_dir}/(.+)/hits/', 1) AS source, id, unnest(hits) AS hash, marked
        FROM read_parquet('{hits_dir}/**/hits/*.parquet', filename = true)
        """
    )
    con.execute(
        f"""
        CREATE TABLE common AS
        SELECT hash FROM doc_hash GROUP BY hash HAVING count(*) > {COMMON_DOC_LIMIT}
        """
    )
    con.execute(
        """
        CREATE TABLE doc_item AS
        SELECT d.source, d.id, d.marked, i.eval, i.eval_id, count(DISTINCT d.hash) AS k
        FROM doc_hash d JOIN idx i USING (hash)
        WHERE d.hash NOT IN (SELECT hash FROM common)
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
               arg_max(di.id, di.k) AS best_doc
        FROM item_n n LEFT JOIN doc_item di USING (eval, eval_id)
        GROUP BY n.eval, n.eval_id, n.n
        """
    )
    con.execute(
        """
        CREATE TABLE corpus AS
        SELECT i.eval_id, count(DISTINCT d.hash) AS found
        FROM idx i JOIN doc_hash d USING (hash)
        WHERE d.hash NOT IN (SELECT hash FROM common)
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
