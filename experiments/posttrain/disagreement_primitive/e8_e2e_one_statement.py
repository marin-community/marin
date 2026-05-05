# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E8 end-to-end smoke test on a SINGLE statement.

Runs all six stages of the E8 pipeline on one statement (`do_not_lie` — interesting
because E7v2 flagged it with the largest behavioral divergence in the spec):
  1. compile rubric (anchored 1-5 with per-anchor rationale)
  2. generate 1 scenario (asking for 20, picking the first)
  3. generate response from all 3 generators (GPT-5.1, Qwen2.5-7B weak, Gemini-3-Flash; temp=1)
  4. judge variant A (raw spec + examples) on each response
  5. judge variant B (rubric only) on each response
  6. compare per-generator A vs B + verify all raw files saved with full text

Total: 11 API calls (~$0.05). Confirms every code path before launching the full run.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    COMPILER_SYSTEM,
    JUDGE_A_SYSTEM,
    JUDGE_B_SYSTEM,
    SCENARIO_GEN_SYSTEM,
    SPEC_PATH,
    call_gemini_text,
    call_gpt_json,
    call_gpt_text,
    call_weak_text,
    get_examples,
    make_clients,
    render_anchors,
    render_examples,
)

TEST_STATEMENT_ID = "do_not_lie"


def main() -> int:
    log = RawAPILogger("e8_e2e_one_statement")
    print(f"raw run dir: {log.run_dir}\n")

    spec = [json.loads(line) for line in SPEC_PATH.open() if line.strip()]
    stmt = next((s for s in spec if s["id"] == TEST_STATEMENT_ID), None)
    if not stmt:
        print(f"FAIL: statement {TEST_STATEMENT_ID} not found")
        return 1
    examples = get_examples(stmt)
    print(f"statement: {stmt['id']}")
    print(f"text head:  {stmt['text'][:160]}...")
    print(f"examples:   {len(examples)}\n")

    oai, weak, gem = make_clients()

    # --- Stage 1: compile rubric ---
    print("=== Stage 1: compile rubric (anchored 1-5 + rationales) ===")
    compile_user = (
        f"STATEMENT_ID: {stmt['id']}\n\n"
        f"STATEMENT TEXT:\n{stmt['text']}\n\n"
        f"EXAMPLES:\n{render_examples(examples)}\n\n"
        "Produce the anchored 1-5 rubric per the schema."
    )
    rubric = call_gpt_json(
        log, oai, "compiler", {"statement_id": stmt["id"]},
        COMPILER_SYSTEM, compile_user, max_tokens=4000,
    )
    anchors = rubric.get("anchors") or {}
    if sorted(anchors.keys()) != ["1", "2", "3", "4", "5"]:
        print(f"  FAIL: anchors keys = {sorted(anchors.keys())}, expected ['1','2','3','4','5']")
        return 1
    for k in ("1", "2", "3", "4", "5"):
        ent = anchors[k]
        if set(ent.keys()) != {"criterion", "reasoning", "spec_quotes", "example_refs"}:
            print(f"  FAIL: anchor[{k}] keys = {sorted(ent.keys())}, expected criterion/reasoning/spec_quotes/example_refs")
            return 1
    print(f"  all 5 anchors well-formed with criterion+reasoning+spec_quotes+example_refs")
    print(f"  anchor[1].criterion: {anchors['1']['criterion'][:120]}...")
    print(f"  anchor[5].criterion: {anchors['5']['criterion'][:120]}...")
    print(f"  anchor[1].spec_quotes: {anchors['1'].get('spec_quotes', [])[:2]}")
    print(f"  anchor[1].example_refs: {anchors['1'].get('example_refs', [])[:2]}")

    # --- Stage 2: generate scenarios (ask for 20, pick first) ---
    print("\n=== Stage 2: generate scenarios (ask 20, take first) ===")
    scen_user = f"STATEMENT TEXT:\n{stmt['text']}\n\nGenerate 20 borderline scenarios per the schema."
    scen_data = call_gpt_json(
        log, oai, "scenario_gen", {"statement_id": stmt["id"]},
        SCENARIO_GEN_SYSTEM, scen_user, max_tokens=4000,
    )
    scenarios = scen_data.get("scenarios") or []
    if len(scenarios) < 1:
        print(f"  FAIL: 0 scenarios")
        return 1
    user_q = scenarios[0].get("user_query", "")
    print(f"  scenarios returned: {len(scenarios)}")
    print(f"  scenario[0]: {user_q[:200]}")

    # --- Stage 3: generate responses from all 3 generators (temp=1) ---
    print("\n=== Stage 3: generate responses (3 generators, temp=1) ===")
    key = {"statement_id": stmt["id"], "scenario_idx": 0}
    gpt_resp = call_gpt_text(log, oai, "generator_gpt", key, user_q, max_tokens=1000, temp=1.0)
    print(f"  GPT-5.1     ({len(gpt_resp):4d} chars): {gpt_resp[:140]}...")
    weak_resp = call_weak_text(log, weak, "generator_weak", key, user_q, max_tokens=1000, temp=1.0)
    print(f"  Qwen2.5-7B  ({len(weak_resp):4d} chars): {weak_resp[:140]}...")
    gem_resp = call_gemini_text(log, gem, "generator_gemini", key, user_q, max_tokens=1000, temp=1.0)
    print(f"  Gemini      ({len(gem_resp):4d} chars): {gem_resp[:140]}...")
    if not (gpt_resp and weak_resp and gem_resp):
        print(f"  FAIL: empty response — gpt={bool(gpt_resp)} weak={bool(weak_resp)} gem={bool(gem_resp)}")
        return 1

    # --- Stage 4: judge variant A (raw spec + examples) ---
    print("\n=== Stage 4: judge variant A (raw spec + examples) ===")
    GENS = [("gpt-5.1", gpt_resp), ("Qwen/Qwen2.5-7B-Instruct-Turbo", weak_resp), ("gemini-3-flash-preview", gem_resp)]
    judgements_a: dict[str, dict] = {}
    for gen_label, response in GENS:
        ja_user = (
            f"STATEMENT TEXT:\n{stmt['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response}\n\n"
            "Score per the schema."
        )
        data = call_gpt_json(
            log, oai, "judge_variant_a",
            {"statement_id": stmt["id"], "scenario_idx": 0, "generator": gen_label},
            JUDGE_A_SYSTEM, ja_user, max_tokens=800,
        )
        judgements_a[gen_label] = data
        print(f"  {gen_label:30s} A_score={data.get('score')}  reasoning: {(data.get('reasoning') or '')[:120]}")
        print(f"  {' ':30s} spec_quotes={data.get('spec_quotes', [])[:2]}")

    # --- Stage 5: judge variant B (rubric only, NO spec text) ---
    print("\n=== Stage 5: judge variant B (rubric only) ===")
    judgements_b: dict[str, dict] = {}
    for gen_label, response in GENS:
        jb_user = (
            f"RUBRIC:\n{render_anchors(rubric)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response}\n\n"
            "Score per the schema."
        )
        data = call_gpt_json(
            log, oai, "judge_variant_b",
            {"statement_id": stmt["id"], "scenario_idx": 0, "generator": gen_label},
            JUDGE_B_SYSTEM, jb_user, max_tokens=600,
        )
        judgements_b[gen_label] = data
        print(f"  {gen_label:30s} B_score={data.get('score')}  reasoning: {(data.get('reasoning') or '')[:120]}")
        print(f"  {' ':30s} rubric_quotes={data.get('rubric_quotes', [])[:2]}")

    # --- Per-generator A vs B comparison ---
    print("\n=== Per-generator paired comparison ===")
    print(f"  {'generator':30s}  A   B   |Δ|")
    diffs = []
    for gen_label, _ in GENS:
        a = judgements_a[gen_label].get("score")
        b = judgements_b[gen_label].get("score")
        if a is not None and b is not None:
            diffs.append(abs(a - b))
            print(f"  {gen_label:30s}  {a}   {b}   {abs(a - b)}")
        else:
            print(f"  {gen_label:30s}  {a}   {b}   ?")
    if diffs:
        print(f"  mean |A-B| across 3 generators: {sum(diffs) / len(diffs):.2f}")

    # --- Verify saved files ---
    print(f"\n=== Verifying raw files in {log.run_dir} ===")
    files = sorted(p for p in log.run_dir.rglob("*.json") if p.name != "_manifest.json")
    expected = 1 + 1 + 3 + 3 + 3  # compile + scenario + 3 gens + 3 judge_a + 3 judge_b = 11
    if len(files) != expected:
        print(f"  FAIL: file count {len(files)} != expected {expected}")
        return 1
    by_role: dict[str, list[Path]] = {}
    for f in files:
        by_role.setdefault(f.parent.name, []).append(f)
    for role, fs in sorted(by_role.items()):
        print(f"  {role}/: {len(fs)} file(s)")

    # Round-trip check on a generator file: returned == saved
    gpt_file = by_role["generator_gpt"][0]
    gpt_rec = json.loads(gpt_file.read_text())
    gpt_saved = gpt_rec["response"]["choices"][0]["message"]["content"]
    if gpt_saved != gpt_resp:
        print(f"  FAIL: GPT saved {len(gpt_saved)} chars != returned {len(gpt_resp)} chars")
        return 1
    print(f"  OK GPT roundtrip: {len(gpt_saved)} chars match")

    # Round-trip check on judge_b file: parsed score matches what was returned
    jb_file = by_role["judge_variant_b"][0]
    jb_rec = json.loads(jb_file.read_text())
    saved_str = jb_rec["response"]["choices"][0]["message"]["content"]
    saved_parsed = json.loads(saved_str)
    if saved_parsed.get("score") not in {1, 2, 3, 4, 5}:
        print(f"  FAIL: judge_variant_b saved score {saved_parsed.get('score')} not in 1-5")
        return 1
    print(f"  OK judge_variant_b roundtrip: score={saved_parsed.get('score')} parses cleanly")

    # Verify rubric file has the structured fields
    comp_file = by_role["compiler"][0]
    comp_rec = json.loads(comp_file.read_text())
    comp_str = comp_rec["response"]["choices"][0]["message"]["content"]
    comp_parsed = json.loads(comp_str)
    comp_anchors = comp_parsed.get("anchors") or {}
    if sorted(comp_anchors.keys()) != ["1", "2", "3", "4", "5"]:
        print(f"  FAIL: compiler saved file missing anchors (keys: {sorted(comp_anchors.keys())})")
        return 1
    print(f"  OK compiler file has all 5 anchors with full per-anchor objects")

    print("\nE2E TEST PASS — all stages produced full content; pipeline verified.")
    print(f"  raw dir: {log.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
