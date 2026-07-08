# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build goal-directed tool-call *episodes* from raw transcripts and draw a
token-cost-weighted sample for semantic labeling.

An **episode** is a contiguous run of information-gathering tool calls (Read, Grep,
Glob, Bash, WebFetch, ...) inside one session that serve a single sub-goal. It is
bounded by a human message, a mutating edit (Write/Edit), or a length cap. Each
episode carries the governing user request, the assistant's stated intent before
each call, a summary of each call, and a truncated result preview — enough for a
labeler to judge *why* the calls happened and whether a wiki / semantic index /
persistent memory / better tool / result compaction would have served the same need
with a smaller answer.

Sampling is **PPS (probability proportional to size)** on each episode's amplified
carry cost, with the heavy tail taken as certainty units. This makes the labeled
coverage fractions map directly onto the ground-truth tool budget: a sampled episode
represents a known slice of token cost.

Outputs (under the step's output directory):

- ``episode_batches/batch_NNN.jsonl`` — self-contained episodes for labelers.
- ``episodes_sampled.parquet`` — per-episode metadata + sampling weight for the join.
- ``episodes_meta.json`` — population totals and sampling parameters.
"""

import glob
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from rigging.filesystem import StoragePath, prefix_join

from experiments.context_efficiency.accounting import P_READ, P_WRITE
from experiments.context_efficiency.transcripts import CHARS_PER_TOK, iter_records

logger = logging.getLogger(__name__)

GATHER_TOOLS = {
    "Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch",
    "NotebookRead", "ToolSearch", "TaskGet", "TaskList", "TaskOutput", "LS",
}  # fmt: skip
MUTATE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
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


def input_summary(inp):
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "pattern", "file_path", "path", "url", "query", "prompt"):
        if inp.get(key):
            return f"{key}={str(inp[key])[:INPUT_CHARS]}"
    return json.dumps(inp)[:INPUT_CHARS]


def build_episodes(path, sid):
    """Yield episodes (dicts, without sampling metadata) for one session file."""
    rows = [
        d
        for d in iter_records(path)
        if d.get("type") in ("user", "assistant") and not d.get("isSidechain")  # skip sub-agent transcripts
    ]

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
                        "input": input_summary(b.get("input")),
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


class PpsSample(NamedTuple):
    sampled: list[dict]
    total_cost: float
    threshold: float  # cost above which an episode is a certainty unit
    interval_cost: float  # cost each remainder pick represents
    n_certainty: int


def _pps_sample(all_eps, n, seed) -> PpsSample:
    """PPS-systematic sample with the heavy tail taken as certainty units.

    A high-cost episode above the ``total/n`` threshold is always included and
    represents exactly its own cost; each remainder pick represents one interval of
    remainder cost. So ``weight_cost`` is the slice of tool budget each labeled
    episode stands for.
    """
    total_cost = sum(e["cost"] for e in all_eps)
    rng = random.Random(seed)
    rng.shuffle(all_eps)
    threshold = total_cost / n
    certainty = [e for e in all_eps if e["cost"] >= threshold]
    remainder = [e for e in all_eps if e["cost"] < threshold]
    n_rem = max(n - len(certainty), 1)
    rem_total = sum(e["cost"] for e in remainder)
    step = rem_total / n_rem
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
    return PpsSample(sampled, total_cost, threshold, step, len(certainty))


@dataclass(frozen=True)
class EpisodesConfig:
    sessions_path: str
    accounting_path: str
    projects_dir: str
    session_glob: str
    n: int
    batch: int
    seed: int
    output_path: str


def run_sampling(cfg: EpisodesConfig) -> None:
    amp = pd.read_parquet(prefix_join(cfg.accounting_path, "session_amplifier.parquet")).set_index("session_id")
    amp_map = amp["observed_amplifier"].to_dict()
    b = pd.read_parquet(prefix_join(cfg.sessions_path, "blocks.parquet"))
    repo_map = b.groupby("session_id").repo.first().to_dict()

    all_eps = []
    skipped = 0
    # sorted so the seeded sample is reproducible: unsorted glob order would make the
    # shuffle pick a different sample each run and silently invalidate prior labels.
    files = sorted(glob.glob(os.path.join(cfg.projects_dir, cfg.session_glob, "*.jsonl")))
    for i, path in enumerate(files):
        sid = os.path.basename(path)[:-6]
        if sid not in amp_map:
            continue
        try:
            eps = build_episodes(path, sid)
        except (OSError, UnicodeDecodeError):
            skipped += 1
            continue
        a = float(amp_map.get(sid, 1.0))
        for e in eps:
            e["repo"] = repo_map.get(sid, "?")
            e["amplifier"] = round(a, 2)
            tok = e["total_result_chars"] / CHARS_PER_TOK
            e["result_tokens"] = int(tok)
            e["cost"] = tok * (P_WRITE + P_READ * max(a, 1.0))
            all_eps.append(e)
        if (i + 1) % 500 == 0:
            logger.info("  scanned %d/%d files, %d episodes", i + 1, len(files), len(all_eps))

    if not all_eps:
        raise ValueError(f"no episodes built from {cfg.projects_dir}/{cfg.session_glob} — is the amplifier table empty?")
    total_cost = sum(e["cost"] for e in all_eps)
    n_sess = len({e["session_id"] for e in all_eps})
    logger.info(
        "%d episodes / %d sessions, cost %.0fM, %d files unreadable", len(all_eps), n_sess, total_cost / 1e6, skipped
    )

    pps = _pps_sample(all_eps, cfg.n, cfg.seed)
    logger.info(
        "sampled %d (%d certainty + %d PPS)", len(pps.sampled), pps.n_certainty, len(pps.sampled) - pps.n_certainty
    )

    # group sampled episodes into labeling shards and write each batch once. Clear any
    # prior shards so a re-run with different parameters cannot leave stale batches behind.
    StoragePath(cfg.output_path).mkdirs()
    bdir = prefix_join(cfg.output_path, "episode_batches")
    bsp = StoragePath(bdir)
    if bsp.exists():
        bsp.rmtree()
    bsp.mkdirs()
    batches: dict[int, list[dict]] = {}
    meta = []
    for j, e in enumerate(pps.sampled):
        bi = j // cfg.batch
        payload = {
            "episode_id": e["episode_id"],
            "repo": e["repo"],
            "user_request": e["user_request"],
            "steps": [
                {"i": k, "tool": s["tool"], "intent": s["intent"], "call": s["input"],
                 "result_tokens": int(s["result_chars"] / CHARS_PER_TOK), "result_preview": s["result_preview"]}
                for k, s in enumerate(e["steps"])
            ],
        }  # fmt: skip
        batches.setdefault(bi, []).append(payload)
        meta.append(
            {
                "episode_id": e["episode_id"],
                "session_id": e["session_id"],
                "repo": e["repo"],
                "batch": bi,
                "n_steps": e["n_steps"],
                # sorted() breaks count ties deterministically (set iteration order is hash-seeded)
                "dominant_tool": max(sorted(set(e["tools"])), key=e["tools"].count),
                "result_tokens": e["result_tokens"],
                "amplifier": e["amplifier"],
                "cost": e["cost"],
                "weight_cost": e["weight_cost"],
                "is_certainty": e["cost"] >= pps.threshold,
            }
        )
    for bi, payloads in batches.items():
        text = "\n".join(json.dumps(p) for p in payloads) + "\n"
        StoragePath(prefix_join(bdir, f"batch_{bi:03d}.jsonl")).write_text(text)

    pd.DataFrame(meta).to_parquet(prefix_join(cfg.output_path, "episodes_sampled.parquet"), index=False)
    with StoragePath(prefix_join(cfg.output_path, "episodes_meta.json")).open("w") as fh:
        json.dump(
            {
                "n_episodes_population": len(all_eps),
                "total_cost_population": total_cost,
                "n_sampled": len(pps.sampled),
                "n_certainty": pps.n_certainty,
                "remainder_interval_cost": pps.interval_cost,
                "n_batches": len(batches),
                "batch_size": cfg.batch,
            },
            fh,
            indent=2,
        )
    logger.info("wrote %d batches to %s", len(batches), bdir)
