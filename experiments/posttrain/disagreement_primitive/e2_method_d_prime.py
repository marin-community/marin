# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2 — Method D-prime for statements with <2 spec examples.

For each such statement: GPT-5.1 generates 4 plausible concrete examples
consistent with the statement text. Then run text-only and synthetic-examples-only
rubric compilation. Compare semantically.

Result: synthetic-examples-derived flag for statements that would otherwise have
no internal-consistency signal.
"""

from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "experiments/posttrain/disagreement_primitive")
import method_d_internal_consistency as md
import analyze_ambiguity as aa

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")

EXAMPLE_GEN_SYSTEM = """You are generating worked examples for a model spec statement. You will be given the statement text. Produce 4 distinct, plausible concrete examples that an alignment researcher might write to illustrate the statement's intended operational behavior.

OUTPUT FORMAT (single JSON, no markdown):
{
  "examples": [
    {"description": "<short label>", "user_query": "<concrete user message>", "good_response": "<assistant response that satisfies the statement>", "bad_response": "<assistant response that violates the statement>"},
    ...
  ]
}

REQUIREMENTS
- Exactly 4 examples.
- Each example must have all 4 fields populated with concrete (not abstract) content.
- Examples should span DIFFERENT scenarios — don't repeat the same kind of test case.
- good_response and bad_response should be substantively different (not just stylistic variants).
- Stay faithful to the spec text — don't add restrictions or permissions beyond what the text actually says.
- No commentary outside the JSON.
"""


def parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_gpt(client, system, user, temp=0):
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
        max_completion_tokens=2500,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def build_examples_only_prompt_synthetic(statement, examples):
    parts = [f"STATEMENT_ID: {statement['id']}\n", "EXAMPLES (statement text NOT shown):\n"]
    for idx, ex in enumerate(examples, 1):
        parts.append(f"Example {idx}:")
        parts.append(f"  description: {ex.get('description', '')}")
        parts.append(f"  user_query: {ex.get('user_query', '')}")
        parts.append(f"  good_response: {ex.get('good_response', '')}")
        parts.append(f"  bad_response: {ex.get('bad_response', '')}")
        parts.append("")
    parts.append("Infer the operational rule these examples encode and produce the rubric per the schema.")
    return "\n".join(parts)


def main() -> int:
    spec = [json.loads(l) for l in SPEC_PATH.open() if l.strip()]
    targets = [s for s in spec if len((s.get("metadata") or {}).get("examples") or []) < 2]
    print(f"target statements (<2 examples): {len(targets)}")
    for s in targets:
        n_ex = len((s.get("metadata") or {}).get("examples") or [])
        print(f"  {s['id']:35s} ({n_ex} examples)")

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rows = []

    def process(stmt):
        sid = stmt["id"]
        # Step 1: generate 4 examples
        gen_prompt = f"STATEMENT_ID: {sid}\n\nSTATEMENT TEXT:\n{stmt['text']}\n\nNow produce 4 examples per the schema."
        examples_data = call_gpt(oai, EXAMPLE_GEN_SYSTEM, gen_prompt)
        examples = examples_data.get("examples") or []
        if len(examples) < 2:
            return {"statement_id": sid, "error": "example_gen_failed", "n_examples": len(examples)}
        # Step 2: text-only rubric
        t_prompt = md.build_text_only_prompt(stmt)
        t_rubric = call_gpt(oai, md.SYSTEM_PROMPT, t_prompt)
        # Step 3: synthetic-examples-only rubric
        e_prompt = build_examples_only_prompt_synthetic(stmt, examples)
        e_rubric = call_gpt(oai, md.SYSTEM_PROMPT, e_prompt)
        # Step 4: equivalence
        equiv = call_gpt(oai, aa.EQUIV_SYSTEM, aa.build_equiv_prompt(stmt, t_rubric, e_rubric))
        return {
            "statement_id": sid,
            "n_synthetic_examples": len(examples),
            "synthetic_examples": examples,
            "t_rubric": t_rubric,
            "e_rubric": e_rubric,
            "good_equivalence": equiv.get("good_equivalence"),
            "bad_equivalence": equiv.get("bad_equivalence"),
            "overall_equivalence": equiv.get("overall_equivalence"),
            "materially_different": equiv.get("materially_different"),
            "disagreement_summary": equiv.get("disagreement_summary"),
        }

    print()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(process, s) for s in targets]):
            try:
                r = fut.result()
                rows.append(r)
                if "error" in r:
                    print(f"  ERR {r['statement_id']}: {r['error']}")
                else:
                    print(f"  {r['statement_id']:35s} D'_overall={r['overall_equivalence']}")
            except Exception as e:
                print(f"  WORKER FAIL: {e}")

    out = DIR / "e2_method_d_prime.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
