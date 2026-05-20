# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every lm-eval-harness task referenced in ``task_configs.py`` loads.

Reflects exactly what ``prepare_eval_corpus.py`` will do on iris: applies
the same ``trust_remote_code`` monkey-patch, expands group names via
``flatten_task_dict``, and probes one doc from the first non-empty split.

For each unique ``EvalTaskConfig(name, ...)`` in
``experiments/evals/task_configs.py``:

1. ``get_task_dict([name])``  (lm-eval registry lookup)
2. ``flatten_task_dict`` -- group names like ``mmlu`` expand to N children;
   each leaf is verified independently.
3. iterate the first non-empty split (test -> validation -> training)
4. call ``doc_to_text`` + ``doc_to_target`` on the first doc

Reports SUCCESS / FAIL per *leaf task* (one row per group child) with a
tail of the extracted text. Runs serially -- lm-eval's task loader reads
YAML configs + cross-task utility scripts that aren't thread-safe.

Reads task names by regex over ``task_configs.py`` so we DON'T trigger
the heavy import chain (``experiments.evals.task_configs`` ->
``marin.evaluation`` -> ``levanter.eval_harness`` -> torch).

Run with the ``eval`` extra so lm-eval is importable. The
``ifeval`` extra pulls in ``langdetect`` for ``ifeval`` /
``leaderboard_ifeval``:

    uv run --with "lm-eval[math,api,ifeval]@git+https://github.com/stanford-crfm/\
lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab" \\
        python experiments/datakit/decontam/verify_lmh_tasks.py

Output: a summary table + a ``verify_lmh_tasks_report.tsv`` next to the
script with one row per leaf task (status, error, n_docs, split,
text_excerpt).
"""

import ast
import logging
import sys
import traceback
from pathlib import Path

from rigging.log_setup import configure_logging

from experiments.datakit.decontam.lmh_loader import (
    flatten_task_dict,
    materialize_first_nonempty_split,
    trust_remote_code_for_hf,
)

logger = logging.getLogger(__name__)


TASK_CONFIGS = Path(__file__).resolve().parents[2] / "evals" / "task_configs.py"
REPORT_PATH = Path(__file__).with_name("verify_lmh_tasks_report.tsv")
SAMPLE_TAIL_CHARS = 160


def _read_task_names() -> list[str]:
    """AST-parse every ``EvalTaskConfig(...)`` in task_configs.py, collect the ``name`` arg.

    Handles both positional (``EvalTaskConfig("mmlu", 5)``) and keyword
    (``EvalTaskConfig(name="mmlu", num_fewshot=0)``) call forms. AST avoids
    triggering the heavy ``experiments.evals.task_configs`` import chain
    (-> marin.evaluation -> levanter.eval_harness -> torch).
    """
    tree = ast.parse(TASK_CONFIGS.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "EvalTaskConfig"):
            continue
        # First positional arg, if it's a string literal.
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
            continue
        # Or `name=...` kwarg form.
        for kw in node.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                names.add(kw.value.value)
                break
    return sorted(names)


def _probe_leaf(leaf_name: str, task) -> dict:
    """Probe one leaf task: first non-empty split, doc[0] doc_to_text/target."""
    out: dict = {
        "name": leaf_name,
        "parent": "",
        "status": "FAIL",
        "split": "",
        "n_docs": 0,
        "excerpt": "",
        "error": "",
    }
    chosen = materialize_first_nonempty_split(task)
    if chosen is None:
        out["error"] = "no docs in any split"
        return out
    split, docs = chosen
    out["split"] = split
    out["n_docs"] = len(docs)
    try:
        first = docs[0]
        prompt = str(task.doc_to_text(first) or "")
        target = str(task.doc_to_target(first) or "")
    except Exception as exc:
        out["error"] = f"doc[0]: {type(exc).__name__}: {exc}"
        return out
    excerpt = (prompt + "  ||  " + target).replace("\n", " ").replace("\t", " ").strip()
    out["excerpt"] = excerpt[:SAMPLE_TAIL_CHARS]
    out["status"] = "SUCCESS"
    return out


def _verify_one_name(name: str) -> list[dict]:
    """Load ``name`` (a task or group), expand to leaves, probe each. Returns 1+ rows."""
    from lm_eval.tasks import get_task_dict

    try:
        task_dict = get_task_dict([name])
    except Exception as exc:
        return [
            {
                "name": name,
                "parent": "",
                "status": "FAIL",
                "split": "",
                "n_docs": 0,
                "excerpt": "",
                "error": f"load: {type(exc).__name__}: {exc}",
            }
        ]
    leaves = list(flatten_task_dict(task_dict))
    if not leaves:
        return [
            {
                "name": name,
                "parent": "",
                "status": "FAIL",
                "split": "",
                "n_docs": 0,
                "excerpt": "",
                "error": "no leaf tasks after flatten",
            }
        ]
    parent = name if len(leaves) > 1 else ""
    rows: list[dict] = []
    for leaf_name, task in leaves:
        row = _probe_leaf(leaf_name, task)
        row["parent"] = parent
        rows.append(row)
    return rows


def main() -> int:
    configure_logging(logging.INFO)
    trust_remote_code_for_hf()

    names = _read_task_names()
    logger.info("found %d unique task names in %s", len(names), TASK_CONFIGS)

    results: list[dict] = []
    for i, name in enumerate(names, 1):
        try:
            rows = _verify_one_name(name)
        except Exception:
            rows = [
                {
                    "name": name,
                    "parent": "",
                    "status": "FAIL",
                    "split": "",
                    "n_docs": 0,
                    "excerpt": "",
                    "error": f"executor: {traceback.format_exc().splitlines()[-1]}",
                }
            ]
        results.extend(rows)
        if i % 25 == 0 or i == len(names):
            ok = sum(1 for x in results if x["status"] == "SUCCESS")
            logger.info("progress: %d/%d parent names (ok leaves so far=%d)", i, len(names), ok)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("status\tname\tparent\tsplit\tn_docs\terror\texcerpt\n")
        for r in sorted(results, key=lambda x: (x["status"] != "SUCCESS", x["parent"], x["name"])):
            f.write(
                f"{r['status']}\t{r['name']}\t{r['parent']}\t{r['split']}\t{r['n_docs']}\t"
                f"{r['error'].replace(chr(9), ' ').replace(chr(10), ' ')}\t"
                f"{r['excerpt'].replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )

    ok = [r for r in results if r["status"] == "SUCCESS"]
    fail = [r for r in results if r["status"] != "SUCCESS"]
    print()
    print(f"summary: {len(ok)} ok / {len(fail)} fail / {len(results)} total leaf rows")
    print(f"  from {len(names)} parent names in task_configs.py")
    print(f"report: {REPORT_PATH}")
    if fail:
        print("\nfailure modes (top error patterns):")
        from collections import Counter

        patterns = Counter()
        for r in fail:
            key = r["error"].split(":")[0]
            patterns[key] += 1
        for pat, count in patterns.most_common(15):
            print(f"  {count:>4}  {pat}")

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
