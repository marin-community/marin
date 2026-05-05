# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E4 — Cross-judge equivalence audit.

Sample 30 random Method C compiler-rubric pairs (stratified across C-score buckets).
Re-judge each pair with BOTH GPT-5.1 (the original equivalence judge) and
Gemini-3-Flash (cross-family). Compare scores.

Goal: quantify any systematic GPT-5.1 self-evaluation bias.
"""

from __future__ import annotations
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any
from google import genai
from google.genai import types
from openai import OpenAI

sys.path.insert(0, "experiments/posttrain/disagreement_primitive")
import analyze_ambiguity as aa  # for EQUIV_SYSTEM and build_equiv_prompt

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")

GPT = "gpt-5.1"
GEMINI = "gemini-3-flash-preview"
SAMPLE_SIZE = 30
SEED = 42


def parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_gpt(client: OpenAI, system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=GPT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_completion_tokens=2000,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def call_gemini(client: genai.Client, system: str, user: str) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=2000,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(model=GEMINI, contents=user, config=config)
    return parse_json(resp.text or "")


def main() -> int:
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    c_rows = [json.loads(l) for l in (DIR / "method_c_rubrics.jsonl").open() if l.strip()]
    by_stmt: dict[str, dict[str, dict]] = {}
    for r in c_rows:
        by_stmt.setdefault(r["statement_id"], {})[r["compiler_model"]] = r

    pairs: list[tuple[str, str, str]] = []
    for sid, comps in by_stmt.items():
        for ma, mb in combinations(sorted(comps), 2):
            pairs.append((sid, ma, mb))
    print(f"total pairs available: {len(pairs)}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    sample = pairs[:SAMPLE_SIZE]
    print(f"sampling {len(sample)} pairs")

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)

    rows = []
    for sid, ma, mb in sample:
        if sid not in spec:
            continue
        prompt = aa.build_equiv_prompt(spec[sid], by_stmt[sid][ma], by_stmt[sid][mb])
        try:
            gpt_r = call_gpt(oai, aa.EQUIV_SYSTEM, prompt)
        except Exception as e:
            print(f"  GPT FAIL {sid}/{ma}~{mb}: {e}")
            continue
        try:
            gem_r = call_gemini(gem, aa.EQUIV_SYSTEM, prompt)
        except Exception as e:
            print(f"  GEMINI FAIL {sid}/{ma}~{mb}: {e}")
            continue
        rows.append({
            "statement_id": sid, "ma": ma, "mb": mb,
            "gpt_overall": gpt_r.get("overall_equivalence"),
            "gem_overall": gem_r.get("overall_equivalence"),
            "gpt_good": gpt_r.get("good_equivalence"),
            "gem_good": gem_r.get("good_equivalence"),
            "gpt_bad": gpt_r.get("bad_equivalence"),
            "gem_bad": gem_r.get("bad_equivalence"),
        })
        print(f"  {sid:30s} GPT={gpt_r.get('overall_equivalence')} GEM={gem_r.get('overall_equivalence')}")

    # Compute offset and correlation
    gpt_scores = [r["gpt_overall"] for r in rows if r["gpt_overall"] is not None and r["gem_overall"] is not None]
    gem_scores = [r["gem_overall"] for r in rows if r["gpt_overall"] is not None and r["gem_overall"] is not None]
    n = len(gpt_scores)
    if n:
        offset = sum(gem_scores) / n - sum(gpt_scores) / n
        # Pearson
        gx, gy = sum(gpt_scores) / n, sum(gem_scores) / n
        num = sum((a - gx) * (b - gy) for a, b in zip(gpt_scores, gem_scores))
        dx = (sum((a - gx) ** 2 for a in gpt_scores)) ** 0.5
        dy = (sum((b - gy) ** 2 for b in gem_scores)) ** 0.5
        pearson = num / (dx * dy) if dx * dy > 0 else None
        print(f"\nn={n}; mean offset (Gemini-GPT) = {offset:+.2f}; Pearson = {pearson:.3f}" if pearson is not None else f"\nn={n}; offset={offset:+.2f}")

    out = DIR / "e4_cross_judge_audit.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
