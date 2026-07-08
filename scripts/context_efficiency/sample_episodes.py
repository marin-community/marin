#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Build goal-directed tool-call *episodes* from raw transcripts and draw a
token-cost-weighted sample for semantic labeling.

An **episode** is a contiguous run of information-gathering tool calls (Read,
Grep, Glob, Bash, WebFetch, ...) inside one session that serve a single sub-goal.
It is bounded by a human message, a mutating edit (Write/Edit), or a length cap.
Each episode carries the governing user request, the assistant's stated intent
before each call, a summary of each call, and a truncated result preview — enough
for a sub-agent to judge *why* the calls happened and whether a wiki / semantic
index / persistent memory / better tool / result compaction would have served the
same need with a smaller answer.

Sampling is **PPS (probability proportional to size)** on each episode's amplified
carry cost, with the heavy tail taken as certainty units. This makes the labeled
coverage fractions map directly onto the ground-truth tool budget: a sampled
episode represents a known slice of token cost, so
`population_saved = Σ certainty cost_i·frac_i + (remainder_total/n_rem)·Σ_rem frac_i`.

Outputs:
- `_data/episode_batches/batch_NNN.jsonl` — self-contained episodes for labelers.
- `_data/episodes_sampled.parquet` — per-episode metadata + sampling weight for the
  analysis join.
"""
import argparse
import glob
import json
import os
import random

import pandas as pd

GATHER_TOOLS = {
    "Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch",
    "NotebookRead", "ToolSearch", "TaskGet", "TaskList", "TaskOutput", "LS",
}  # fmt: skip
MUTATE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
P_WRITE, P_READ = 1.25, 0.10
MAX_STEPS = 12
REQ_CHARS, INTENT_CHARS, INPUT_CHARS, PREVIEW_CHARS = 700, 180, 200, 260


def block_list(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return c
    if isinstance(c, str):
        return [{"type": "human_text", "text": c}]
    return []


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def input_summary(tool, inp):
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "pattern", "file_path", "path", "url", "query", "prompt"):
        if inp.get(key):
            return f"{key}={str(inp[key])[:INPUT_CHARS]}"
    return json.dumps(inp)[:INPUT_CHARS]


def parse_session(path, sid):
    """Yield episodes as dicts (without sampling metadata)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("isSidechain"):  # sub-agent transcript, not the main loop
                continue
            if d.get("type") in ("user", "assistant"):
                rows.append(d)

    user_request = ""
    pending_intent = ""
    ep_steps = []
    open_calls = {}  # tool_use_id -> step dict
    episodes = []
    seq = 0

    def flush():
        nonlocal ep_steps, seq
        steps = [s for s in ep_steps if s.get("result_chars", 0) >= 0 and s["tool"] in GATHER_TOOLS]
        if steps:
            episodes.append(
                {
                    "episode_id": f"{sid[:8]}-{seq:04d}",
                    "session_id": sid,
                    "user_request": user_request[:REQ_CHARS],
                    "n_steps": len(steps),
                    "tools": [s["tool"] for s in steps],
                    "steps": steps,
                    "total_result_chars": sum(s.get("result_chars", 0) for s in steps),
                }
            )
            seq += 1
        ep_steps = []

    for d in rows:
        msg = d.get("message")
        if not isinstance(msg, dict):
            continue
        role = d.get("type")
        blocks = block_list(msg)
        is_tool_result_msg = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)

        if role == "user" and not is_tool_result_msg:
            # a real human message -> episode boundary + new goal context
            flush()
            user_request = text_of(msg.get("content"))
            pending_intent = ""
            continue

        if role == "user" and is_tool_result_msg:
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    step = open_calls.get(b.get("tool_use_id"))
                    if step is not None:
                        rc = text_of(b.get("content"))
                        step["result_chars"] = len(rc)
                        step["result_preview"] = rc[:PREVIEW_CHARS]
            continue

        if role == "assistant":
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    pending_intent = (pending_intent + " " + b.get("text", "")).strip()[:INTENT_CHARS]
                elif b.get("type") == "tool_use":
                    name = b.get("name")
                    if name in MUTATE_TOOLS:
                        flush()  # gathering ended in an edit; the goal shifts
                        pending_intent = ""
                        continue
                    step = {
                        "tool": name,
                        "intent": pending_intent,
                        "input": input_summary(name, b.get("input")),
                        "result_chars": 0,
                        "result_preview": "",
                    }
                    open_calls[b.get("id")] = step
                    if name in GATHER_TOOLS:
                        ep_steps.append(step)
                    if len(ep_steps) >= MAX_STEPS:
                        flush()
                    pending_intent = ""
    flush()
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--n", type=int, default=2000, help="target labeled episodes")
    ap.add_argument("--batch", type=int, default=15, help="episodes per labeling shard")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    amp = pd.read_parquet(os.path.join(args.data, "session_amplifier.parquet")).set_index("session_id")
    amp_map = amp["observed_amplifier"].to_dict()
    repo_map = {}
    b = pd.read_parquet(os.path.join(args.data, "blocks.parquet"))
    for sid, r in b.groupby("session_id").repo.first().items():
        repo_map[sid] = r

    all_eps = []
    files = glob.glob(os.path.join(args.projects, "*", "*.jsonl"))
    for i, path in enumerate(files):
        sid = os.path.basename(path)[:-6]
        if sid not in amp_map:
            continue
        try:
            eps = parse_session(path, sid)
        except (OSError, UnicodeDecodeError):
            continue
        a = float(amp_map.get(sid, 1.0))
        for e in eps:
            e["repo"] = repo_map.get(sid, "?")
            e["amplifier"] = round(a, 2)
            tok = e["total_result_chars"] / 4.0
            e["result_tokens"] = int(tok)
            e["cost"] = tok * (P_WRITE + P_READ * max(a, 1.0))
            all_eps.append(e)
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{len(files)} files, {len(all_eps)} episodes")

    total_cost = sum(e["cost"] for e in all_eps)
    n_sess = len({e["session_id"] for e in all_eps})
    print(f"\n{len(all_eps)} episodes across {n_sess} sessions, total cost {total_cost/1e6:.0f}M")

    # PPS-systematic with certainty units for the heavy tail
    rng = random.Random(args.seed)
    rng.shuffle(all_eps)
    threshold = total_cost / args.n
    certainty = [e for e in all_eps if e["cost"] >= threshold]
    remainder = [e for e in all_eps if e["cost"] < threshold]
    n_rem = max(args.n - len(certainty), 1)
    rem_total = sum(e["cost"] for e in remainder)
    step = rem_total / n_rem
    # systematic PPS over the remainder
    sampled = list(certainty)
    for e in certainty:
        e["weight_cost"] = e["cost"]
    start = rng.uniform(0, step)
    cum, targets, ti = 0.0, [start + k * step for k in range(n_rem)], 0
    for e in remainder:
        cum += e["cost"]
        while ti < len(targets) and targets[ti] <= cum:
            e2 = dict(e)
            e2["weight_cost"] = step  # each remainder pick represents one interval of cost
            sampled.append(e2)
            ti += 1
    print(f"sampled {len(sampled)} ({len(certainty)} certainty + {len(sampled)-len(certainty)} PPS)")

    # write labeling shards (compact, self-contained)
    bdir = os.path.join(args.data, "episode_batches")
    os.makedirs(bdir, exist_ok=True)
    for old in glob.glob(os.path.join(bdir, "*.jsonl")):
        os.remove(old)
    meta = []
    for j, e in enumerate(sampled):
        e["batch"] = j // args.batch
        payload = {
            "episode_id": e["episode_id"],
            "repo": e["repo"],
            "user_request": e["user_request"],
            "steps": [
                {"i": k, "tool": s["tool"], "intent": s["intent"], "call": s["input"],
                 "result_tokens": int(s["result_chars"] / 4), "result_preview": s["result_preview"]}
                for k, s in enumerate(e["steps"])
            ],
        }  # fmt: skip
        with open(os.path.join(bdir, f"batch_{e['batch']:03d}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        meta.append(
            {
                "episode_id": e["episode_id"],
                "session_id": e["session_id"],
                "repo": e["repo"],
                "batch": e["batch"],
                "n_steps": e["n_steps"],
                "dominant_tool": max(set(e["tools"]), key=e["tools"].count),
                "result_tokens": e["result_tokens"],
                "amplifier": e["amplifier"],
                "cost": e["cost"],
                "weight_cost": e["weight_cost"],
                "is_certainty": e["cost"] >= threshold,
            }
        )
    n_batches = (len(sampled) + args.batch - 1) // args.batch
    pd.DataFrame(meta).to_parquet(os.path.join(args.data, "episodes_sampled.parquet"), index=False)
    with open(os.path.join(args.data, "episodes_meta.json"), "w") as fh:
        json.dump(
            {
                "n_episodes_population": len(all_eps),
                "total_cost_population": total_cost,
                "n_sampled": len(sampled),
                "n_certainty": len(certainty),
                "remainder_interval_cost": step,
                "n_batches": n_batches,
                "batch_size": args.batch,
            },
            fh,
            indent=2,
        )
    print(f"wrote {n_batches} batches to {bdir}")
    print("wrote episodes_sampled.parquet + episodes_meta.json")


if __name__ == "__main__":
    main()
