# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E1 — Method C K=5 reruns on all 46 statements.

For each statement, K=5 independent runs of the 3-compiler rubric writer
(GPT-5.1, GLM-5.1, Gemini-Flash) at temp=0. For each run, compute the 3-pair
semantic equivalence, take the mean. Aggregate K-rerun-means per statement
to [mean, stdev, min, max] for confidence-interval reporting.
"""

from __future__ import annotations
import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from google import genai
from google.genai import types
from openai import OpenAI

sys.path.insert(0, "experiments/posttrain/disagreement_primitive")
import method_c_multi_compiler as mc
import analyze_ambiguity as aa

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")
K = 5


def parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


# Override the temperature in method_c_multi_compiler to 0
mc.DEFAULT_TEMPERATURE = 0


def call_gpt_judge(client, system, user):
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_completion_tokens=2000,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def main() -> int:
    spec = [json.loads(l) for l in SPEC_PATH.open() if l.strip()]
    print(f"statements: {len(spec)}")
    oai = mc.make_openai_client()
    glm = mc.make_together_client()
    gem = mc.make_gemini_client()

    def one_run(stmt):
        """Generate 3 rubrics + compute 3-pair equivalence mean for one rerun."""
        prompt = mc.build_user_prompt(stmt)
        rubrics = {}
        for name, fn, model in [
            ("gpt", lambda p: mc.call_openai(oai, p), mc.GPT_5_1_MODEL),
            ("glm", lambda p: mc.call_together(glm, p), mc.GLM_5_1_MODEL),
            ("gem", lambda p: mc.call_gemini(gem, p), mc.GEMINI_FLASH_MODEL),
        ]:
            try:
                r = fn(prompt)
                rubrics[model] = r["parsed"]
            except Exception as e:
                # Skip this compiler if it fails
                pass
        if len(rubrics) < 2:
            return None
        # 3-pair equivalence
        pair_scores = []
        for ma, mb in combinations(sorted(rubrics), 2):
            try:
                eq = call_gpt_judge(oai, aa.EQUIV_SYSTEM, aa.build_equiv_prompt(stmt, rubrics[ma], rubrics[mb]))
                pair_scores.append(eq.get("overall_equivalence"))
            except Exception as e:
                pass
        valid = [s for s in pair_scores if s is not None]
        if not valid:
            return None
        return sum(valid) / len(valid)

    def process(stmt):
        sid = stmt["id"]
        rerun_means = []
        for k in range(K):
            try:
                m = one_run(stmt)
                if m is not None:
                    rerun_means.append(m)
            except Exception as e:
                print(f"  {sid} run {k}: FAIL {e}")
        if not rerun_means:
            return {"statement_id": sid, "error": "no successful reruns"}
        mean = sum(rerun_means) / len(rerun_means)
        sd = statistics.stdev(rerun_means) if len(rerun_means) > 1 else 0.0
        return {
            "statement_id": sid,
            "k": len(rerun_means),
            "rerun_means": [round(m, 2) for m in rerun_means],
            "mean": round(mean, 2),
            "stdev": round(sd, 2),
            "min": round(min(rerun_means), 2),
            "max": round(max(rerun_means), 2),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process, s): s["id"] for s in spec}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)
            done += 1
            if "error" in r:
                print(f"  [{done}/{total}] ERR {r['statement_id']}")
            else:
                print(f"  [{done}/{total}] {r['statement_id']:35s} mean={r['mean']} stdev={r['stdev']} range=[{r['min']},{r['max']}]")

    out = DIR / "e1_method_c_k5.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
