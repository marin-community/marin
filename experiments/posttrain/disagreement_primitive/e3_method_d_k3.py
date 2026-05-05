# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E3 — Method D K=3 reruns on 35 evaluable statements.

For each statement with ≥2 spec examples, run Method D 3 independent times.
Each run: fresh text-only rubric + fresh examples-only rubric + equivalence.
Aggregate to per-statement [mean, stdev, 95% CI].
"""

from __future__ import annotations
import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "experiments/posttrain/disagreement_primitive")
import method_d_internal_consistency as md
import analyze_ambiguity as aa

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")
K = 3


def parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_gpt(client, system, user):
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
    targets = [s for s in spec if len((s.get("metadata") or {}).get("examples") or []) >= 2]
    print(f"target statements (≥2 examples): {len(targets)}")
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def one_run(stmt):
        t = call_gpt(oai, md.SYSTEM_PROMPT, md.build_text_only_prompt(stmt))
        e = call_gpt(oai, md.SYSTEM_PROMPT, md.build_examples_only_prompt(stmt))
        eq = call_gpt(oai, aa.EQUIV_SYSTEM, aa.build_equiv_prompt(stmt, t, e))
        return eq.get("overall_equivalence")

    def process(stmt):
        sid = stmt["id"]
        scores = []
        for k in range(K):
            try:
                s = one_run(stmt)
                if s is not None:
                    scores.append(s)
            except Exception as e:
                print(f"  {sid} run {k}: FAIL {e}")
        if not scores:
            return {"statement_id": sid, "error": "no scores"}
        mean = sum(scores) / len(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return {"statement_id": sid, "scores": scores, "mean": round(mean, 2), "stdev": round(sd, 2), "min": min(scores), "max": max(scores)}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(process, s): s["id"] for s in targets}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                rows.append(r)
                if "error" in r:
                    print(f"  ERR {r['statement_id']}")
                else:
                    print(f"  {r['statement_id']:35s} mean={r['mean']} stdev={r['stdev']} range=[{r['min']},{r['max']}]")
            except Exception as e:
                print(f"  worker fail: {e}")

    out = DIR / "e3_method_d_k3.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")
    # Print flagged (mean<6)
    flagged = [r for r in rows if r.get("mean") is not None and r["mean"] < 6]
    print(f"\nFlagged (mean<6): {len(flagged)}")
    for r in flagged:
        print(f"  {r['statement_id']:35s} mean={r['mean']} stdev={r['stdev']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
