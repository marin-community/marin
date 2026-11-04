# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count agentic tasks whose instruction, and whose instruction with its solution, sit in one kept document.

The task evals of `targets_aaii_v5.py` and `targets_policy_agentic.py` index each
task as a full record plus one record per view (``<eval>-<task>-<view>``). The
scan scores each view on its own, so the question (the instruction) and the
answer (the reference solution, patch or rubric) can be checked apart:

* ``q``: a kept document holds at least ``VIEW_BAR`` of the instruction's 13-grams;
* ``qa``: the same document also holds at least ``VIEW_BAR`` of an answer view.

Reads ``doc_item.parquet`` and ``items.parquet`` from an `aggregate.py` output
directory and the `kept.py` outputs in KEPT_DIRS; only documents those checked
count. Writes ``task_views.parquet`` (one row per task) to OUT_DIR and prints per
eval: tasks, ``q``, ``qa``, ``q_only``, and how many tasks have their answer alone.

Usage: task_views.py AGG_DIR FILTERS_JSON OUT_DIR KEPT_DIR...
"""

import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

VIEW_BAR = 0.5
TASK_EVALS = (
    "aa_briefcase_lite",
    "automationbench",
    "terminal_bench_4",
    "terminal_bench_science",
    "swebench_verified",
    "deepswe",
    "ot_tblite",
    "ds1000",
    "bfcl_parity",
    "bixbench",
    "tau3",
    "sotopia_hard",
)
QUESTION_VIEW = "instruction"
ANSWER_VIEWS = ("solution", "patch", "rubric")
VIEW_PATTERN = "-(instruction|solution|patch|rubric|tests)$"


def main() -> None:
    agg_dir, filters_path, out_dir, *kept_dirs = sys.argv[1:]
    filters = json.loads(Path(filters_path).read_text())
    con = duckdb.connect()
    con.register("source_dir", pa.table({"source": list(filters), "dir": [f["dir"] for f in filters.values()]}))
    con.register("task_eval", pa.table({"eval_name": list(TASK_EVALS)}))
    kept_files = ", ".join(f"'{d}/*.parquet'" for d in kept_dirs)
    answer_views = ", ".join(f"'{v}'" for v in ANSWER_VIEWS)
    con.execute(
        f"""
        CREATE TABLE view AS
        SELECT t.eval_name AS eval, regexp_replace(i.eval_id, '{VIEW_PATTERN}', '') AS task,
               coalesce(regexp_extract(i.eval_id, '{VIEW_PATTERN}', 1), '') AS view, i.eval_id, i.n
        FROM read_parquet('{agg_dir}/items.parquet') i
        JOIN task_eval t ON starts_with(i.eval_id, t.eval_name || '-')
        """
    )
    con.execute(
        f"""
        CREATE TABLE hit AS
        SELECT v.eval, v.task, v.view, s.dir || '/' || d.file AS path, d.id
        FROM read_parquet('{agg_dir}/doc_item.parquet') d
        JOIN view v USING (eval_id)
        JOIN source_dir s USING (source)
        JOIN (SELECT DISTINCT path, id, kept FROM read_parquet([{kept_files}])) k
            ON k.path = s.dir || '/' || d.file AND k.id = d.id
        WHERE k.kept AND d.k / v.n >= {VIEW_BAR} AND v.view <> ''
        """
    )
    con.execute(
        f"""
        CREATE TABLE task_status AS
        WITH doc AS (
            SELECT eval, task, path, id,
                   bool_or(view = '{QUESTION_VIEW}') AS has_q,
                   bool_or(view IN ({answer_views})) AS has_a
            FROM hit GROUP BY ALL
        )
        SELECT t.eval, t.task,
               coalesce(bool_or(d.has_q), false) AS q,
               coalesce(bool_or(d.has_q AND d.has_a), false) AS qa,
               coalesce(bool_or(d.has_a), false) AS a_any
        FROM (SELECT DISTINCT eval, task FROM view) t LEFT JOIN doc d USING (eval, task)
        GROUP BY ALL
        """
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY task_status TO '{out_dir}/task_views.parquet'")
    con.sql(
        """
        SELECT eval, count(*) AS tasks, sum(q::INT) AS q, sum(qa::INT) AS qa,
               round(100 * avg(qa::INT), 1) AS qa_pct, sum((q AND NOT qa)::INT) AS q_only,
               round(100 * avg((q AND NOT qa)::INT), 1) AS q_only_pct, sum(a_any::INT) AS answer_any
        FROM task_status GROUP BY ALL ORDER BY eval
        """
    ).show(max_rows=50)


if __name__ == "__main__":
    main()
