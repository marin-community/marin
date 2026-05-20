# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect AA + lm-eval-harness eval schemas before running the full prep.

For each AA eval and a handful of lm-eval tasks, print:
- HF column names + dtypes
- First sample row (raw)
- Extracted ``text`` (what ``prepare_eval_corpus.py`` would write)
- Whether the per-eval field extraction matched (vs. fell back to concat)

Streams (``streaming=True``) so we don't download full datasets just to
peek at schemas. Local-runnable; needs HF token in env for any gated sets.

    uv run python experiments/datakit/decontam/ops/inspect_eval_corpus.py

Optional flags via env:
    INSPECT_LMH=1          also inspect a sample of lm-eval tasks
    INSPECT_AA_ONLY=mmlu_pro,hle   restrict AA inspection to specific subdirs
"""

import logging
import os
import sys
from typing import Any

from rigging.log_setup import configure_logging

from experiments.datakit.decontam.lmh_loader import materialize_first_nonempty_split
from experiments.datakit.decontam.prepare_eval_corpus import (
    AA_EVALS,
    _concat_strings,
    _extract_aa_text,
    _iter_aa_rows,
    _lmh_task_names,
)

logger = logging.getLogger(__name__)
SAMPLE_ROWS = 2
LMH_SAMPLE_TASKS = ("mmlu", "arc_easy", "hellaswag", "gsm8k", "humaneval")


def _truncate(s: str, n: int = 300) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "..."


def _row_columns(row: dict[str, Any]) -> str:
    """Compact column summary: name(type), with str-list vs str distinguished."""
    parts: list[str] = []
    for k in sorted(row.keys()):
        v = row[k]
        t = type(v).__name__
        if isinstance(v, list) and v and isinstance(v[0], str):
            t = f"list[str:{len(v)}]"
        elif isinstance(v, list):
            t = f"list:{len(v)}"
        elif isinstance(v, str):
            t = f"str:{len(v)}"
        parts.append(f"{k}({t})")
    return ", ".join(parts)


def _inspect_aa() -> None:
    selected = os.environ.get("INSPECT_AA_ONLY")
    selected_set = {s.strip() for s in selected.split(",")} if selected else None

    for cfg in AA_EVALS:
        if selected_set is not None and cfg.subdir not in selected_set:
            continue
        print()
        print(f"=== aa/{cfg.subdir}  ({cfg.hf_id} subset={cfg.subset} split={cfg.split}) ===")
        try:
            row_iter = _iter_aa_rows(cfg)
        except Exception as exc:
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        seen = 0
        for row in row_iter:
            row_dict = dict(row)
            print(f"  columns: {_row_columns(row_dict)}")
            extracted = _extract_aa_text(row_dict, cfg)
            generic = _concat_strings(row_dict)
            matched_via_config = bool(cfg.text_fields or cfg.list_fields) and (extracted != generic)
            print(f"  extractor: {'per-eval-fields' if matched_via_config else 'GENERIC CONCAT FALLBACK'}")
            print(f"  text ({len(extracted)} chars): {_truncate(extracted)}")
            if cfg.skip_if is not None:
                print(f"  skip_if matched: {cfg.skip_if(row_dict)}")
            seen += 1
            if seen >= SAMPLE_ROWS:
                break

        if seen == 0:
            print("  EMPTY -- streaming yielded no rows")


def _inspect_lmh_sample() -> None:
    from lm_eval.tasks import get_task_dict

    available_names = set(_lmh_task_names())
    names = [n for n in LMH_SAMPLE_TASKS if n in available_names]
    print()
    print(f"=== lm-eval-harness sample ({len(names)} tasks) ===")
    for name in names:
        print()
        print(f"  --- lmh/{name} ---")
        try:
            task = get_task_dict([name])[name]
        except Exception as exc:
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            continue
        chosen = materialize_first_nonempty_split(task)
        if chosen is None:
            print("  no docs in any split")
            continue
        split, docs = chosen
        print(f"  split: {split}  ({len(docs)} docs)")
        for i, doc in enumerate(docs[: min(SAMPLE_ROWS, len(docs))]):
            try:
                prompt = str(task.doc_to_text(doc) or "")
            except Exception as exc:
                prompt = f"<doc_to_text failed: {exc}>"
            try:
                target = str(task.doc_to_target(doc) or "")
            except Exception as exc:
                target = f"<doc_to_target failed: {exc}>"
            if isinstance(doc, dict):
                print(f"    doc[{i}] columns: {_row_columns(doc)}")
            print(f"    doc[{i}] prompt ({len(prompt)} chars): {_truncate(prompt)}")
            print(f"    doc[{i}] target ({len(target)} chars): {_truncate(target)}")


def main() -> None:
    configure_logging(logging.WARNING)
    _inspect_aa()
    if os.environ.get("INSPECT_LMH"):
        _inspect_lmh_sample()


if __name__ == "__main__":
    sys.exit(main())
