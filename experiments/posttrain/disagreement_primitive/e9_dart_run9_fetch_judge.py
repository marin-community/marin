"""Fetch Run 9 judge batches (GPT + Claude) and merge with Gemini sync output.
Writes per_judgment_run9_r{N}.jsonl with all 3 judges combined.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from openai import OpenAI

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_DIR = DIR / "dart_run9"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    state = json.loads((OUT_DIR / f"run9_judge_r{args.round}_batches.json").read_text())
    job_dir = Path(state["job_dir"])

    # GPT
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gpt_meta = state["batches"]["gpt"]
    content = oai.files.content(oai.batches.retrieve(gpt_meta["batch_id"]).output_file_id).read()
    raw = job_dir / "output_gpt.jsonl"
    raw.write_bytes(content)
    cmap_gpt = json.loads((job_dir / "custom_id_map_gpt.json").read_text())
    gpt_rows = []
    for line in content.decode().splitlines():
        if not line.strip(): continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        m = cmap_gpt.get(cid)
        if not m: continue
        sid, cond_name, scen, gen, v9_cond = m
        body = entry.get("response", {}).get("body", {})
        try:
            data = parse_json_strict(body["choices"][0]["message"]["content"])
        except Exception as e:
            gpt_rows.append({"judge": "gpt", "statement_id": sid, "scenario_idx": scen, "generator": gen,
                             "condition": cond_name, "v9_condition": v9_cond, "error": f"parse:{type(e).__name__}"})
            continue
        score = data.get("score")
        try:
            score = int(score) if score is not None else None
            if score is not None and not 1 <= score <= 5: score = None
        except (TypeError, ValueError): score = None
        gpt_rows.append({"judge": "gpt", "statement_id": sid, "scenario_idx": scen, "generator": gen,
                         "condition": cond_name, "v9_condition": v9_cond, "score": score,
                         "reasoning": data.get("reasoning")})
    print(f"GPT: {len(gpt_rows)} rows")

    # Claude
    cla_meta = state["batches"]["claude"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    entries = ba.collect(api_key, job_dir, name=cla_meta["name"])
    cmap_cla = json.loads((job_dir / "custom_id_map_cla.json").read_text())
    cla_rows = []
    for entry in entries:
        cid = entry.get("custom_id")
        m = cmap_cla.get(cid)
        if not m: continue
        sid, cond_name, scen, gen, v9_cond = m
        args_dict = ba.extract_tool_args(entry, tool_name="submit_judgment")
        if args_dict is None:
            cla_rows.append({"judge": "claude", "statement_id": sid, "scenario_idx": scen, "generator": gen,
                             "condition": cond_name, "v9_condition": v9_cond, "error": "no_tool_args"})
            continue
        score = args_dict.get("score")
        try:
            score = int(score) if score is not None else None
            if score is not None and not 1 <= score <= 5: score = None
        except (TypeError, ValueError): score = None
        cla_rows.append({"judge": "claude", "statement_id": sid, "scenario_idx": scen, "generator": gen,
                         "condition": cond_name, "v9_condition": v9_cond, "score": score,
                         "reasoning": args_dict.get("reasoning")})
    print(f"Claude: {len(cla_rows)} rows")

    # Gemini already on disk
    gem_rows = [json.loads(l) for l in (OUT_DIR / f"per_judgment_run9_r{args.round}_gem.jsonl").open() if l.strip()]
    print(f"Gemini: {len(gem_rows)} rows (preexisting)")

    # Merge
    out_path = OUT_DIR / f"per_judgment_run9_r{args.round}.jsonl"
    with out_path.open("w") as f:
        for r in gpt_rows + gem_rows + cla_rows:
            f.write(json.dumps(r) + "\n")
    n_scored = sum(1 for r in gpt_rows + gem_rows + cla_rows if r.get("score") is not None)
    print(f"\nWrote {out_path}: {len(gpt_rows) + len(gem_rows) + len(cla_rows)} rows, {n_scored} scored")


if __name__ == "__main__":
    main()
