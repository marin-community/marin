# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Targeted retry of failed phase-2 judgments.

Reads phase2_<judge>/{va,vb}_judgments.jsonl, finds rows with `error` set, and
re-runs only those (statement, scenario, generator, variant) keys with the
*updated* call helpers (GLM bumped to 4000 max_tokens; Gemini with
safety_settings=BLOCK_NONE). Merges new judgments back into the structured
JSONL files in place. Preserves all original raw audit dirs; new raw responses
go to a fresh timestamped subdir under results/raw/e8_phase2_<judge>_retry/.

Usage:
    source .env2 && .venv/bin/python e8_phase2_retry_failures.py --judge glm
    source .env2 && .venv/bin/python e8_phase2_retry_failures.py --judge gemini
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    DIR,
    JUDGE_A_SYSTEM,
    JUDGE_B_SYSTEM,
    SPEC_PATH,
    TOGETHER_BASE_URL,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
    write_jsonl,
)
from e8_phase2_cross_model import call_glm_json, call_gemini_json, stage6_analyze


def find_failures(structured_path: Path) -> list[dict[str, Any]]:
    """Return list of (statement_id, scenario_idx, generator) keys that errored."""
    fails = []
    for r in load_jsonl(structured_path):
        if "error" in r:
            fails.append({
                "statement_id": r["statement_id"],
                "scenario_idx": r["scenario_idx"],
                "generator": r["generator"],
                "user_query": r.get("user_query", ""),
                "response": r.get("response", ""),
            })
    return fails


def merge_back(structured_path: Path, new_rows_by_key: dict[tuple, dict[str, Any]]) -> None:
    """Replace failed rows in-place with the new judgments. Keep all other rows."""
    rows = load_jsonl(structured_path)
    n_replaced = 0
    for i, r in enumerate(rows):
        k = (r["statement_id"], r["scenario_idx"], r["generator"])
        if k in new_rows_by_key:
            rows[i] = new_rows_by_key[k]
            n_replaced += 1
    write_jsonl(rows, structured_path)
    print(f"  merged {n_replaced} retried rows into {structured_path}")


def retry_with_judge(judge_label: str, judge_fn, max_workers: int = 4) -> None:
    """Retry both variant A and variant B failures for the given judge."""
    out_dir = DIR / f"phase2_{judge_label}"
    spec_by_id = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics_by_id = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics.jsonl") if "error" not in r}

    for variant_label, system, build_user_fn in [
        ("a", JUDGE_A_SYSTEM,
         lambda f: (
             f"STATEMENT TEXT:\n{spec_by_id[f['statement_id']]['text']}\n\n"
             f"SPEC EXAMPLES:\n{render_examples(get_examples(spec_by_id[f['statement_id']]))}\n\n"
             f"USER QUERY: {f['user_query']}\n\n"
             f"ASSISTANT RESPONSE: {f['response']}\n\n"
             "Score per the schema."
         )),
        ("b", JUDGE_B_SYSTEM,
         lambda f: (
             f"RUBRIC:\n{render_anchors(rubrics_by_id[f['statement_id']])}\n\n"
             f"USER QUERY: {f['user_query']}\n\n"
             f"ASSISTANT RESPONSE: {f['response']}\n\n"
             "Score per the schema."
         )),
    ]:
        structured_path = out_dir / f"v{variant_label}_judgments.jsonl"
        fails = find_failures(structured_path)
        if not fails:
            print(f"  variant {variant_label}: no failures to retry")
            continue
        print(f"  variant {variant_label}: retrying {len(fails)} failed keys")

        def retry(f):
            user = build_user_fn(f)
            key = {"statement_id": f["statement_id"], "scenario_idx": f["scenario_idx"], "generator": f["generator"]}
            try:
                data = judge_fn(role=f"judge_variant_{variant_label}_{judge_label}_retry", key=key,
                                system=system, user=user)
                base_row = {**key, "user_query": f["user_query"], "response": f["response"],
                            "score": data.get("score"), "reasoning": data.get("reasoning")}
                if variant_label == "a":
                    base_row["spec_quotes"] = data.get("spec_quotes") or []
                    base_row["example_refs"] = data.get("example_refs") or []
                else:
                    base_row["rubric_quotes"] = data.get("rubric_quotes") or []
                return key_tuple(key), base_row
            except Exception as exc:
                base_row = {**key, "user_query": f["user_query"], "response": f["response"],
                            "error": f"{type(exc).__name__}: {exc}"}
                return key_tuple(key), base_row

        new_rows = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(retry, f): f for f in fails}
            done = 0
            for fut in as_completed(futs):
                k, row = fut.result()
                new_rows[k] = row
                done += 1
                if done % 25 == 0 or done == len(fails):
                    n_recovered = sum(1 for r in new_rows.values() if "error" not in r)
                    print(f"    [{done}/{len(fails)}] recovered={n_recovered}")
        merge_back(structured_path, new_rows)


def key_tuple(k: dict[str, Any]) -> tuple:
    return (k["statement_id"], k["scenario_idx"], k["generator"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["glm", "gemini"], required=True)
    args = parser.parse_args()

    log = RawAPILogger(f"e8_phase2_{args.judge}_retry")
    print(f"phase 2 retry judge={args.judge}")
    print(f"  raw run dir: {log.run_dir}")

    if args.judge == "glm":
        glm = OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ["TOGETHER_API_KEY"])
        def judge_fn(*, role, key, system, user):
            return call_glm_json(log, glm, role, key, system, user)  # default 4000 tokens now
        max_workers = 3
    else:
        gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
        def judge_fn(*, role, key, system, user):
            return call_gemini_json(log, gem, role, key, system, user)  # safety_settings=BLOCK_NONE now
        max_workers = 6

    t0 = time.time()
    retry_with_judge(args.judge, judge_fn, max_workers=max_workers)

    out_dir = DIR / f"phase2_{args.judge}"
    print(f"\nre-running stage6 analysis after retries...")
    stage6_analyze(
        out_dir / "va_judgments.jsonl",
        out_dir / "vb_judgments.jsonl",
        DIR / "e8_rubrics.jsonl",
        out_dir / "per_statement.jsonl",
        out_dir / "decomposition.md",
        args.judge,
    )
    print(f"\nretry complete in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
