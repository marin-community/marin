"""Fetch GPT and Claude compiler batches from Run 9 Phase 1.
Writes diagnoses_gpt.jsonl and diagnoses_cla.jsonl in dart_run9/.
"""
from __future__ import annotations
import json, os, sys
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
    state = json.loads((OUT_DIR / "run9_batches.json").read_text())

    # GPT
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gpt_meta = state["batches"]["gpt"]
    gpt_b = oai.batches.retrieve(gpt_meta["batch_id"])
    if gpt_b.status != "completed":
        raise SystemExit(f"GPT batch not completed: {gpt_b.status}")
    content = oai.files.content(gpt_b.output_file_id).read()
    raw_path = Path(state["job_dir"]) / "output_gpt.jsonl"
    raw_path.write_bytes(content)
    gpt_cmap = gpt_meta["custom_id_map"]
    gpt_rows = []
    for line in content.decode().splitlines():
        if not line.strip(): continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        sid = gpt_cmap.get(cid)
        if not sid: continue
        body = entry.get("response", {}).get("body", {})
        try:
            msg_text = body["choices"][0]["message"]["content"]
            data = parse_json_strict(msg_text)
        except Exception as e:
            gpt_rows.append({"statement_id": sid, "error": f"parse:{type(e).__name__}"})
            continue
        gpt_rows.append({"statement_id": sid, **data})
    out_gpt = OUT_DIR / "diagnoses_gpt.jsonl"
    out_gpt.write_text("\n".join(json.dumps(r) for r in gpt_rows))
    print(f"  GPT: {len(gpt_rows)} rows → {out_gpt}")

    # Claude
    cla_meta = state["batches"]["claude"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    job_dir = Path(state["job_dir"])
    entries = ba.collect(api_key, job_dir, name=cla_meta["name"])
    cla_cmap = cla_meta["custom_id_map"]
    cla_rows = []
    for entry in entries:
        cid = entry.get("custom_id")
        sid = cla_cmap.get(cid)
        if not sid: continue
        args = ba.extract_tool_args(entry, tool_name="submit_dart_diagnosis")
        if args is None:
            cla_rows.append({"statement_id": sid, "error": "no_tool_args"})
            continue
        cla_rows.append({"statement_id": sid, **args})
    out_cla = OUT_DIR / "diagnoses_cla.jsonl"
    out_cla.write_text("\n".join(json.dumps(r) for r in cla_rows))
    print(f"  Claude: {len(cla_rows)} rows → {out_cla}")

    # Summary
    print("\n=== Diagnoses summary ===")
    print(f"{'sid':40s}  {'GPT':35s}  {'Pro':35s}  {'Claude':35s}")
    print("-" * 150)
    gpt_d = {r["statement_id"]: r for r in gpt_rows}
    cla_d = {r["statement_id"]: r for r in cla_rows}
    gem_d = {}
    for line in (OUT_DIR / "diagnoses_gem.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gem_d[r["statement_id"]] = r
    sids = sorted(set(gpt_d) | set(gem_d) | set(cla_d))
    for sid in sids:
        g = gpt_d.get(sid, {}).get("diagnosis", "?")
        p = gem_d.get(sid, {}).get("diagnosis", "?")
        c = cla_d.get(sid, {}).get("diagnosis", "?")
        print(f"{sid:40s}  {g:35s}  {p:35s}  {c:35s}")


if __name__ == "__main__":
    main()
