"""Fetch GPT-5.1 baseline fill batches and append to per_judgment_opposite.jsonl."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

from openai import OpenAI

DIR = Path("experiments/posttrain/disagreement_primitive")
TRACKER = DIR / "gpt_baseline_fill_batches.json"
OUT_PATH = DIR / "per_judgment_opposite.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def parse_json_strict(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            first = s[:nl].strip("`").strip()
            if first == "" or first.lower() == "json":
                s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    return json.loads(s)


def main():
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    info = json.loads(TRACKER.read_text())
    cmap = json.loads(Path(info["custom_id_map_path"]).read_text())

    existing = load_jsonl(OUT_PATH)
    existing_keys = {(r["statement_id"], r["scenario_idx"], r["generator"], r["condition"], r.get("judge"))
                     for r in existing}

    new_rows = []
    n_parse_err = 0
    n_score_null = 0
    for cond, batch_meta in info["batches"].items():
        b = oai.batches.retrieve(batch_meta["batch_id"])
        if b.status != "completed":
            print(f"  {cond}: not completed (status={b.status})")
            continue
        # Download output file
        content = oai.files.content(b.output_file_id).read()
        # Save raw output for debugging
        out_raw = Path(info["job_dir"]) / f"output_{cond}.jsonl"
        out_raw.write_bytes(content)
        # Parse line by line
        n_lines = 0
        for line in content.decode().splitlines():
            if not line.strip(): continue
            n_lines += 1
            entry = json.loads(line)
            cid = entry.get("custom_id")
            mapping = cmap.get(cid)
            if mapping is None:
                continue
            sid, condition_from_map, scen, gen = mapping
            if (sid, scen, gen, cond, "gpt") in existing_keys:
                continue
            response = entry.get("response", {})
            if response.get("status_code") != 200:
                n_parse_err += 1
                new_rows.append({
                    "judge": "gpt", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond,
                    "error": f"http_{response.get('status_code')}",
                })
                continue
            body = response.get("body", {})
            try:
                msg = body["choices"][0]["message"]["content"]
                data = parse_json_strict(msg)
            except (KeyError, json.JSONDecodeError, IndexError) as e:
                n_parse_err += 1
                new_rows.append({
                    "judge": "gpt", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond,
                    "error": f"parse:{type(e).__name__}",
                })
                continue
            score = data.get("score")
            try:
                score = int(score) if score is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
            except (TypeError, ValueError):
                score = None
            if score is None:
                n_score_null += 1
            new_rows.append({
                "judge": "gpt",
                "statement_id": sid, "scenario_idx": scen,
                "generator": gen, "condition": cond,
                "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
                "example_refs": data.get("example_refs") or [],
            })
        print(f"  {cond}: parsed {n_lines} lines, accumulated {len(new_rows)} rows so far")

    with OUT_PATH.open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    n_scored = sum(1 for r in new_rows if r.get("score") is not None)
    print(f"\nAppended {len(new_rows)} new GPT rows: {n_scored} scored, {n_score_null} null, {n_parse_err} parse/http errors")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
