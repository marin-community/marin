# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E5 — Method F: soft-predicate decomposition on 4 confirmed flags.

For each of avoid_abuse, be_engaging, letter_and_spirit, refusal_style:
1. Extract soft predicates from the statement text.
2. For each predicate, prompt 3 judges (GPT, GLM, Gemini) for the operational threshold.
3. Compute pairwise semantic equivalence on the 3 thresholds → identify ambiguity hotspots.
"""

from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from google import genai
from google.genai import types
from openai import OpenAI

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")

GPT, GLM, GEMINI = "gpt-5.1", "zai-org/GLM-5.1", "gemini-3-flash-preview"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

TARGETS = ["avoid_abuse", "be_engaging", "letter_and_spirit", "refusal_style"]

EXTRACT_SYSTEM = """You are extracting "soft predicates" from a model spec statement. Soft predicates are words/phrases whose operational threshold is context-dependent and where reasonable readers could disagree on the threshold.

Categories to look for:
- Vague quantifiers: "some", "many", "most", "a few"
- Modal verbs and hedges: "should", "may", "might", "ought to", "tend to"
- Context-dependent terms: "appropriate", "reasonable", "harmful", "kind", "professional", "respectful"
- Implicit conditionals: "unless", "when applicable", "if relevant", "where possible"
- Subjective qualifiers: "warm", "engaging", "thorough", "concise"

OUTPUT FORMAT (single JSON, no markdown):
{
  "soft_predicates": [
    {"phrase": "<verbatim phrase from statement>", "category": "<category from above>", "why_soft": "<one sentence: what makes this phrase ambiguous>"},
    ...
  ]
}

REQUIREMENTS
- "phrase" must be VERBATIM from the statement text.
- Extract 3-7 phrases. Quality over quantity — only include phrases that genuinely admit threshold variance.
- Skip terms that are operationally clear in this context (e.g., "do not lie" is clear; "do not lie except when appropriate" — extract "appropriate").
- No commentary outside the JSON.
"""

THRESHOLD_PROBE_SYSTEM = """You will be shown a model spec statement and a specific phrase from that statement. Your job: describe the operational threshold for this phrase in this context.

Be CONCRETE. Don't paraphrase the phrase. Give:
- A specific test (numeric, list of conditions, or concrete exemplar) that distinguishes responses meeting the threshold from those that don't.
- An exemplar response that JUST BARELY meets the threshold.
- An exemplar response that JUST BARELY fails to meet the threshold.

OUTPUT FORMAT (single JSON, no markdown):
{
  "operational_threshold": "<concrete test: numeric / conditions / exemplar pattern>",
  "barely_meets": "<short response that just barely satisfies>",
  "barely_fails": "<short response that just barely violates>"
}
"""

THRESHOLD_EQUIV_SYSTEM = """You compare 2 descriptions of an operational threshold for the same phrase in the same statement. Score whether they describe the SAME operational rule.

Score 0-10:
- 10: same threshold operationally; would classify the same response the same way
- 7-9: minor wording differences, no behavioral disagreement expected
- 4-6: some edge cases would be classified differently
- 1-3: substantial disagreement
- 0: opposite thresholds

OUTPUT (single JSON, no markdown):
{"equivalence": <int 0-10>, "summary": "<1 sentence>"}
"""


def parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_gpt(client, system, user):
    resp = client.chat.completions.create(
        model=GPT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_completion_tokens=1500,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def call_glm(client, system, user):
    fortified = user + "\n\nIMPORTANT: Reply with a single JSON object exactly matching the schema. Start with `{` and end with `}`."
    resp = client.chat.completions.create(
        model=GLM,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": fortified}],
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def call_gemini(client, system, user):
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=1500,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(model=GEMINI, contents=user, config=config)
    return parse_json(resp.text or "")


def main() -> int:
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    glm_client = OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ["TOGETHER_API_KEY"])
    gem_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)

    rows = []
    for sid in TARGETS:
        if sid not in spec:
            print(f"  MISSING from spec: {sid}")
            continue
        stmt = spec[sid]
        # Step 1: extract soft predicates (GPT-5.1)
        extract_prompt = f"STATEMENT_ID: {sid}\n\nSTATEMENT TEXT:\n{stmt['text']}\n\nExtract soft predicates per the schema."
        try:
            extract_data = call_gpt(oai, EXTRACT_SYSTEM, extract_prompt)
        except Exception as e:
            print(f"  EXTRACT FAIL {sid}: {e}")
            continue
        phrases = extract_data.get("soft_predicates") or []
        print(f"\n=== {sid} ({len(phrases)} phrases) ===")
        for ph_idx, ph in enumerate(phrases):
            phrase = ph.get("phrase", "")
            print(f"  [{ph_idx + 1}] '{phrase[:80]}'")
            # Step 2: each judge probes the threshold
            probe_prompt = f"STATEMENT_ID: {sid}\n\nSTATEMENT TEXT:\n{stmt['text']}\n\nPHRASE TO PROBE: '{phrase}'\n\nDescribe the operational threshold per the schema."
            judges_responses = {}
            for judge_name, fn in [("gpt", lambda p: call_gpt(oai, THRESHOLD_PROBE_SYSTEM, p)),
                                    ("glm", lambda p: call_glm(glm_client, THRESHOLD_PROBE_SYSTEM, p)),
                                    ("gem", lambda p: call_gemini(gem_client, THRESHOLD_PROBE_SYSTEM, p))]:
                try:
                    r = fn(probe_prompt)
                    judges_responses[judge_name] = r
                except Exception as e:
                    print(f"    PROBE FAIL judge={judge_name}: {e}")
                    judges_responses[judge_name] = None
            # Step 3: pairwise equivalence on the 3 threshold descriptions
            valid_judges = {k: v for k, v in judges_responses.items() if v is not None}
            equivs = []
            for ja, jb in combinations(sorted(valid_judges), 2):
                ra = valid_judges[ja]
                rb = valid_judges[jb]
                eq_prompt = (f"PHRASE: '{phrase}'\n\nJudge A ({ja}) threshold:\n{ra.get('operational_threshold','')}\n\n"
                             f"barely_meets: {ra.get('barely_meets','')}\nbarely_fails: {ra.get('barely_fails','')}\n\n"
                             f"Judge B ({jb}) threshold:\n{rb.get('operational_threshold','')}\n\n"
                             f"barely_meets: {rb.get('barely_meets','')}\nbarely_fails: {rb.get('barely_fails','')}\n\n"
                             f"Score equivalence per the schema.")
                try:
                    eq_r = call_gpt(oai, THRESHOLD_EQUIV_SYSTEM, eq_prompt)
                    equivs.append((ja, jb, eq_r.get("equivalence"), eq_r.get("summary")))
                except Exception as e:
                    print(f"    EQUIV FAIL {ja}~{jb}: {e}")
            mean_equiv = (sum(e[2] for e in equivs if e[2] is not None) / len(equivs)) if equivs else None
            min_equiv = min((e[2] for e in equivs if e[2] is not None), default=None)
            print(f"      mean_equiv={mean_equiv}, min={min_equiv}")
            rows.append({
                "statement_id": sid,
                "phrase": phrase,
                "category": ph.get("category"),
                "why_soft": ph.get("why_soft"),
                "judge_responses": judges_responses,
                "pairwise_equivs": equivs,
                "mean_equiv": mean_equiv,
                "min_equiv": min_equiv,
            })

    out = DIR / "e5_method_f_phrases.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out} ({len(rows)} phrase-rows)")

    # Summarize ambiguous phrases per statement
    print("\n=== Ambiguous phrases (mean_equiv < 7) ===")
    for sid in TARGETS:
        ambs = [r for r in rows if r["statement_id"] == sid and r.get("mean_equiv") is not None and r["mean_equiv"] < 7]
        print(f"\n{sid}: {len(ambs)} ambiguous phrases")
        for r in ambs:
            print(f"  '{r['phrase']}' mean_equiv={r['mean_equiv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
