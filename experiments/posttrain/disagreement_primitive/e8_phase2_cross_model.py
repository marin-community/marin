# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E8 phase 2 — cross-model judges.

Reuses phase-1 artifacts (rubrics, scenarios, responses) and re-runs only stages
4 (variant A) and 5 (variant B) with a NON-GPT judge to test whether the GPT-5.1
compiled rubric is faithful enough that a different model's prior can apply it
consistently.

Usage:
    source .env2 && .venv/bin/python e8_phase2_cross_model.py --judge glm
    source .env2 && .venv/bin/python e8_phase2_cross_model.py --judge gemini

Outputs land in `phase2_<judge>/` (NOT the phase-1 paths). Raw responses go to
`results/raw/e8_phase2_<judge>/<UTC-ts>/`. Independent of phase 1 — both can run
in parallel.
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
from google.genai import types
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    DIR,
    GEMINI,
    JUDGE_A_SYSTEM,
    JUDGE_B_SYSTEM,
    SPEC_PATH,
    TOGETHER_BASE_URL,
    get_examples,
    load_jsonl,
    parse_json,
    render_anchors,
    render_examples,
    spearman,
    write_jsonl,
)

GLM_MODEL = "zai-org/GLM-5.1"


# -------------------- Provider-specific JSON judges (always wrapped) --------------------


def call_glm_json(log: RawAPILogger, glm: OpenAI, role: str, key: dict[str, Any],
                  system: str, user: str, max_tokens: int = 4000) -> dict[str, Any]:
    """JSON-mode GLM-5.1 via Together. Temp=0. Uses response_format=json_object.

    GLM-5.1 has hidden chain-of-thought consumption that eats the token budget
    BEFORE producing visible JSON output. At max_tokens=1500 we observed 6.7%
    of judge calls returning empty content (finish_reason=length, all 1500
    tokens consumed by reasoning, 0 chars of JSON emitted) — same pattern
    documented in the post-mortem for the generator stage. Default bumped to
    4000 to give the JSON output room after GLM finishes thinking.
    """
    raw = log.call(
        role=role,
        key=key,
        fn=lambda: glm.chat.completions.create(
            model=GLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        ),
    )
    return parse_json(raw.choices[0].message.content or "")


# Gemini safety thresholds: BLOCK_NONE on every category. The judge sees
# (spec text + scenario + response); for spec statements about CSAM, abuse, or
# explicit content, the JUDGE INPUT alone triggers Gemini's safety filter and
# the API returns prompt_feedback.block_reason='PROHIBITED_CONTENT' with no
# candidates. This is the wrong default for an alignment-research judge — the
# judge isn't producing harmful content, it's scoring already-generated text.
_GEMINI_SAFETY_BLOCK_NONE = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
]


def call_gemini_json(log: RawAPILogger, gem: genai.Client, role: str, key: dict[str, Any],
                     system: str, user: str, max_tokens: int = 1500) -> dict[str, Any]:
    """JSON-mode Gemini-Flash via google-genai. Temp=0. Safety thresholds set to
    BLOCK_NONE for the judge role (see comment above) so the judge can read
    spec content about abuse/explicit edge cases without prompt-blocking.
    Note: Gemini still hard-refuses some categories (e.g., CSAM) even at
    BLOCK_NONE — those will continue to fail and must be excluded.
    """
    # Per dart.md Gotcha 17: Flash supports thinking_level="minimal" (Pro doesn't).
    # Use minimal for judge calls — high throughput, classification-style.
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        response_mime_type="application/json",
        safety_settings=_GEMINI_SAFETY_BLOCK_NONE,
    )
    raw = log.call(
        role=role,
        key=key,
        fn=lambda: gem.models.generate_content(model=GEMINI, contents=user, config=config),
    )
    return parse_json(raw.text or "")


# -------------------- Judging stages (judge-agnostic) --------------------


def judge_variant_a(log: RawAPILogger, judge_fn, responses_path: Path, spec_by_id: dict[str, Any],
                    out_path: Path, judge_label: str, max_workers: int = 4) -> None:
    if out_path.exists():
        print(f"  skip variant_a (exists: {out_path})")
        return
    GEN_KEYS = [("gpt-5.1", "response_gpt"),
                ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
                ("gemini-3-flash-preview", "response_gemini")]
    resp_rows = load_jsonl(responses_path)
    tasks = [
        (r["statement_id"], r["scenario_idx"], r["user_query"], gen_label, r[col])
        for r in resp_rows
        for gen_label, col in GEN_KEYS
        if r.get(col)
    ]
    print(f"  judge_variant_a ({judge_label}): {len(tasks)} judgments")

    def judge(sid: str, idx: int, user_q: str, gen: str, response: str) -> dict[str, Any]:
        stmt = spec_by_id[sid]
        examples = get_examples(stmt)
        user = (
            f"STATEMENT TEXT:\n{stmt['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response}\n\n"
            "Score per the schema."
        )
        key = {"statement_id": sid, "scenario_idx": idx, "generator": gen}
        try:
            data = judge_fn(role=f"judge_variant_a_{judge_label}", key=key,
                            system=JUDGE_A_SYSTEM, user=user)
            return {**key, "user_query": user_q, "response": response,
                    "score": data.get("score"), "reasoning": data.get("reasoning"),
                    "spec_quotes": data.get("spec_quotes") or [],
                    "example_refs": data.get("example_refs") or []}
        except Exception as exc:
            return {**key, "user_query": user_q, "response": response,
                    "error": f"{type(exc).__name__}: {exc}"}

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge, *t): (t[0], t[1], t[3]) for t in tasks}
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"    [{done}/{len(tasks)}]")
    write_jsonl(sorted(rows, key=lambda r: (r["statement_id"], r["scenario_idx"], r["generator"])), out_path)
    print(f"  wrote {out_path}")


def judge_variant_b(log: RawAPILogger, judge_fn, responses_path: Path, rubrics_path: Path,
                    out_path: Path, judge_label: str, max_workers: int = 4) -> None:
    if out_path.exists():
        print(f"  skip variant_b (exists: {out_path})")
        return
    rubrics_by_id = {r["statement_id"]: r["rubric"] for r in load_jsonl(rubrics_path) if "error" not in r}
    GEN_KEYS = [("gpt-5.1", "response_gpt"),
                ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
                ("gemini-3-flash-preview", "response_gemini")]
    resp_rows = load_jsonl(responses_path)
    tasks = [
        (r["statement_id"], r["scenario_idx"], r["user_query"], gen_label, r[col])
        for r in resp_rows
        for gen_label, col in GEN_KEYS
        if r.get(col) and r["statement_id"] in rubrics_by_id
    ]
    print(f"  judge_variant_b ({judge_label}): {len(tasks)} judgments")

    def judge(sid: str, idx: int, user_q: str, gen: str, response: str) -> dict[str, Any]:
        rubric = rubrics_by_id[sid]
        user = (
            f"RUBRIC:\n{render_anchors(rubric)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response}\n\n"
            "Score per the schema."
        )
        key = {"statement_id": sid, "scenario_idx": idx, "generator": gen}
        try:
            data = judge_fn(role=f"judge_variant_b_{judge_label}", key=key,
                            system=JUDGE_B_SYSTEM, user=user)
            return {**key, "user_query": user_q, "response": response,
                    "score": data.get("score"), "reasoning": data.get("reasoning"),
                    "rubric_quotes": data.get("rubric_quotes") or []}
        except Exception as exc:
            return {**key, "user_query": user_q, "response": response,
                    "error": f"{type(exc).__name__}: {exc}"}

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge, *t): (t[0], t[1], t[3]) for t in tasks}
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"    [{done}/{len(tasks)}]")
    write_jsonl(sorted(rows, key=lambda r: (r["statement_id"], r["scenario_idx"], r["generator"])), out_path)
    print(f"  wrote {out_path}")


# -------------------- Stage 6: analyze --------------------


def stage6_analyze(va_path: Path, vb_path: Path, rubrics_path: Path,
                   per_stmt_path: Path, decomp_path: Path, judge_label: str) -> None:
    print(f"  stage6 analyze ({judge_label})")
    va = {(r["statement_id"], r["scenario_idx"], r["generator"]): r for r in load_jsonl(va_path) if "error" not in r}
    vb = {(r["statement_id"], r["scenario_idx"], r["generator"]): r for r in load_jsonl(vb_path) if "error" not in r}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(rubrics_path) if "error" not in r}

    keys = sorted(set(va.keys()) & set(vb.keys()))
    by_stmt: dict[str, list[tuple]] = {}
    for k in keys:
        sid, idx, gen = k
        ra = va[k]; rb = vb[k]
        if ra.get("score") is None or rb.get("score") is None:
            continue
        by_stmt.setdefault(sid, []).append((idx, gen, ra, rb))

    rows = []
    for sid, items in sorted(by_stmt.items()):
        pairs = [(it[2]["score"], it[3]["score"]) for it in items]
        rho = spearman(pairs)
        mean_abs = sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None
        exact = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None
        signed = sum(b - a for a, b in pairs) / len(pairs) if pairs else None
        rows.append({
            "statement_id": sid,
            "n_pairs": len(pairs),
            "spearman_a_b": round(rho, 4) if rho is not None else None,
            "mean_abs_diff": round(mean_abs, 3) if mean_abs is not None else None,
            "pct_exact_agreement": round(exact, 3) if exact is not None else None,
            "signed_bias_b_minus_a": round(signed, 3) if signed is not None else None,
            "mean_a": round(sum(a for a, _ in pairs) / len(pairs), 3),
            "mean_b": round(sum(b for _, b in pairs) / len(pairs), 3),
        })
    write_jsonl(rows, per_stmt_path)
    print(f"  wrote {per_stmt_path}")

    # Decomposition
    disag = []
    for sid, items in by_stmt.items():
        for idx, gen, ra, rb in items:
            d = abs(ra["score"] - rb["score"])
            if d >= 2:
                disag.append((sid, idx, gen, ra, rb, d))
    disag.sort(key=lambda x: -x[5])

    md = [f"# E8 phase 2 — paired indirection analysis (judge={judge_label})", ""]
    md.append("## Per-statement summary")
    md.append("")
    md.append("| statement | n | Spearman(A,B) | mean \\|A−B\\| | % exact | signed bias (B−A) | mean A | mean B |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        md.append(f"| `{r['statement_id']}` | {r['n_pairs']} | {r['spearman_a_b']} | {r['mean_abs_diff']} | {r['pct_exact_agreement']} | {r['signed_bias_b_minus_a']:+} | {r['mean_a']} | {r['mean_b']} |")
    md.append("")
    finite = [r for r in rows if r["spearman_a_b"] is not None]
    if finite:
        rhos = sorted(r["spearman_a_b"] for r in finite)
        median = rhos[len(rhos) // 2]
        below_04 = sum(1 for x in rhos if x < 0.4)
        above_07 = sum(1 for x in rhos if x >= 0.7)
        md.append(f"**Distribution of per-statement Spearman(A,B):** n={len(rhos)}, median={median:.3f}, IQR=[{rhos[len(rhos)//4]:.3f}, {rhos[3*len(rhos)//4]:.3f}], below 0.4: {below_04}, above 0.7: {above_07}")
        md.append("")

    md.append(f"## Top disagreements (|A−B| ≥ 2): {len(disag)} cases")
    md.append("")
    for sid, idx, gen, ra, rb, d in disag[:30]:
        rubric = rubrics.get(sid, {})
        anchor_rats = (rubric.get("anchors") or {}).get(str(ra["score"]), {})
        md.append(f"### `{sid}` / scenario {idx} / generator `{gen}` — A={ra['score']} B={rb['score']} (Δ={d})")
        md.append("")
        md.append(f"- **User query:** {ra.get('user_query','')[:300]}")
        md.append(f"- **Response:** {(ra.get('response') or '')[:400]}")
        md.append(f"- **A rationale:** {ra.get('reasoning','')}")
        md.append(f"  - A spec quotes: `{ra.get('spec_quotes', [])}`")
        md.append(f"  - A example refs: `{ra.get('example_refs', [])}`")
        md.append(f"- **B rationale:** {rb.get('reasoning','')}")
        md.append(f"  - B rubric quotes: `{rb.get('rubric_quotes', [])}`")
        md.append(f"- **Compiler's anchor for score={ra['score']} reasoning:** {anchor_rats.get('reasoning','')}")
        md.append(f"  - Compiler spec quotes: `{anchor_rats.get('spec_quotes', [])}`")
        md.append("")
    decomp_path.parent.mkdir(parents=True, exist_ok=True)
    decomp_path.write_text("\n".join(md))
    print(f"  wrote {decomp_path}")


# -------------------- main --------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["glm", "gemini"], required=True,
                        help="Which judge model to use for variants A and B")
    args = parser.parse_args()

    out_dir = DIR / f"phase2_{args.judge}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = RawAPILogger(f"e8_phase2_{args.judge}")
    print(f"phase 2 judge={args.judge}")
    print(f"  raw run dir: {log.run_dir}")
    print(f"  output dir:  {out_dir}")

    spec_by_id = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}

    # Phase-1 inputs (reused)
    rubrics_path = DIR / "e8_rubrics.jsonl"
    responses_path = DIR / "e8_responses.jsonl"
    for p in (rubrics_path, responses_path):
        if not p.exists():
            print(f"  ERROR: phase-1 artifact missing: {p}")
            return 1

    # Phase-2 outputs
    va_path = out_dir / "va_judgments.jsonl"
    vb_path = out_dir / "vb_judgments.jsonl"
    per_stmt_path = out_dir / "per_statement.jsonl"
    decomp_path = out_dir / "decomposition.md"

    # Bind judge_fn
    if args.judge == "glm":
        glm = OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ["TOGETHER_API_KEY"])
        def judge_fn(*, role, key, system, user):
            return call_glm_json(log, glm, role, key, system, user)
        max_workers = 3  # Together rate limit
    else:  # gemini
        gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
        def judge_fn(*, role, key, system, user):
            return call_gemini_json(log, gem, role, key, system, user)
        max_workers = 6

    t0 = time.time()
    judge_variant_a(log, judge_fn, responses_path, spec_by_id, va_path, args.judge, max_workers=max_workers)
    judge_variant_b(log, judge_fn, responses_path, rubrics_path, vb_path, args.judge, max_workers=max_workers)
    stage6_analyze(va_path, vb_path, rubrics_path, per_stmt_path, decomp_path, args.judge)
    print(f"\nphase 2 ({args.judge}) elapsed: {time.time() - t0:.1f}s")
    print(f"  output dir: {out_dir}")
    print(f"  raw dir:    {log.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
