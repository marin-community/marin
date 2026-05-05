# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test the tightened COMPILER_SYSTEM prompt on the 5 statements that previously
produced nested-rationales schema errors. Verifies the new prompt yields top-level
`anchor_rationales` (not nested inside `score_anchors`) for all 5.

Run:
    source .env2 && .venv/bin/python experiments/posttrain/disagreement_primitive/e8_test_tighter_compiler.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    COMPILER_SYSTEM,
    SPEC_PATH,
    call_gpt_json,
    get_examples,
    make_clients,
    render_examples,
)

PRIOR_FAILURES = [
    "assume_best_intentions",
    "assume_objective_pov",
    "avoid_abuse",
    "avoid_errors",
    "be_empathetic",
]


def check_shape(rubric: dict) -> tuple[bool, str]:
    """Return (ok, diagnostic). New schema: single top-level 'anchors' map with 5
    entries; each entry is an object with {criterion, reasoning, spec_quotes, example_refs}."""
    if set(rubric.keys()) != {"anchors"}:
        return False, f"top-level keys = {sorted(rubric.keys())}; expected exactly {{'anchors'}}"
    anchors = rubric["anchors"]
    if sorted(anchors.keys()) != ["1", "2", "3", "4", "5"]:
        return False, f"anchors keys = {sorted(anchors.keys())}; expected ['1','2','3','4','5']"
    expected_fields = {"criterion", "reasoning", "spec_quotes", "example_refs"}
    for k in ("1", "2", "3", "4", "5"):
        ent = anchors[k]
        if not isinstance(ent, dict):
            return False, f"anchors[{k}] is {type(ent).__name__}, expected dict"
        if set(ent.keys()) != expected_fields:
            return False, f"anchors[{k}] fields = {sorted(ent.keys())}; expected {sorted(expected_fields)}"
        if not isinstance(ent["criterion"], str):
            return False, f"anchors[{k}].criterion is {type(ent['criterion']).__name__}, expected str"
        if not isinstance(ent["spec_quotes"], list):
            return False, f"anchors[{k}].spec_quotes is {type(ent['spec_quotes']).__name__}, expected list"
    return True, "OK"


def main() -> int:
    log = RawAPILogger("e8_test_tighter_compiler")
    print(f"raw run dir: {log.run_dir}\n")

    spec = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    oai, _, _ = make_clients()

    results = []
    for sid in PRIOR_FAILURES:
        stmt = spec[sid]
        examples = get_examples(stmt)
        user = (
            f"STATEMENT_ID: {sid}\n\n"
            f"STATEMENT TEXT:\n{stmt['text']}\n\n"
            f"EXAMPLES:\n{render_examples(examples)}\n\n"
            "Produce the anchored 1-5 rubric per the schema."
        )
        try:
            rubric = call_gpt_json(
                log, oai, "compiler", {"statement_id": sid},
                COMPILER_SYSTEM, user, max_tokens=4000,
            )
            ok, diag = check_shape(rubric)
            results.append((sid, ok, diag))
            tag = "PASS" if ok else "FAIL"
            print(f"  {tag}  {sid:40s}  {diag}")
        except Exception as exc:
            results.append((sid, False, f"{type(exc).__name__}: {exc}"))
            print(f"  ERR   {sid:40s}  {type(exc).__name__}: {exc}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_pass}/{len(results)} pass with tightened prompt")
    print(f"  raw dir: {log.run_dir}")

    if n_pass == len(results):
        print(f"\nALL PASS — tightened prompt is robust on prior-failure cases. Recommend regenerating all 46.")
        return 0
    else:
        print(f"\nNot all pass — schema issue persists. Consider restructuring schema (one map per anchor with criterion+rationale together).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
