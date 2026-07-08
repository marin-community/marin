# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize ``~/.claude`` session transcripts into two Parquet tables.

Reads every ``<projects>/<glob>/*.jsonl`` transcript and emits, under the step's
output directory:

- ``blocks.parquet``: one row per content block (assistant text/thinking/tool_use,
  user text, tool_result). Carries the normalized ``target``, a content hash, an
  estimated token count, and — for tool_results — ``first_paid_turn``, the
  assistant turn on which the result first becomes billed input.
- ``turns.parquet``: one row per assistant turn with the four ``usage`` fields, the
  ephemeral 5m/1h cache split, and the inter-turn wall-clock gap (for TTL reasoning).

Ground-truth billed tokens come from ``usage``; ``est_tokens = chars /
CHARS_PER_TOK`` is the content proxy, calibrated against ``usage`` downstream. The
stored transcript is not the billed stream (tool results truncated in storage,
thinking stripped, images ~0 chars), so magnitudes anchor on ``usage``.
"""

import glob
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from rigging.filesystem import StoragePath, prefix_join

logger = logging.getLogger(__name__)

CHARS_PER_TOK = 4.0

# A session's git repo, if the cwd sits under ~/code/<repo>[/...].
REPO_RE = re.compile(r"/home/[^/]+/code/([^/]+?)(?:[-.]|/|$)")
# Strip worktree/agent-worktree prefixes to a repo-relative path key.
WORKTREE_RE = re.compile(r"^/home/[^/]+/code/[^/]+?" r"(?:/\.worktrees/[^/]+|/\.claude-worktrees/[^/]+|/--?[^/]+)?/")
# Tools whose first argument is a subcommand, so the "verb" is verb+subcommand.
MULTIVERB = {
    "git", "gh", "gcloud", "kubectl", "docker", "weaver", "cargo",
    "npm", "uv", "pip", "python", "python3", "bq", "aws",
}  # fmt: skip
# Prefixes that wrap the operative command (`cd x && rg ...`, `export A=B; cmd`).
WRAP_HEADS = {
    "cd",
    "export",
    ":",
    "true",
    "set",
    "source",
    ".",
    "sudo",
    "time",
    "nice",
    "env",
    "timeout",
    "then",
    "do",
    "{",
}


def repo_of(cwd: str) -> str:
    m = REPO_RE.search(cwd or "")
    return m.group(1) if m else "?"


def norm_path(p: str) -> str | None:
    """Absolute path -> repo-relative-ish key (worktree/agent prefixes removed)."""
    if not p:
        return None
    p = p.strip()
    p2 = WORKTREE_RE.sub("", p)
    if p2 == p and p.startswith("/home/"):
        # generic home path: drop the /home/<user>/code/<repo>/ prefix if present
        p2 = re.sub(r"^/home/[^/]+/code/[^/]+/", "", p)
    return p2


def _shape_of_segment(seg: str):
    toks = seg.split()
    while toks and (toks[0].split("/")[-1] in {"sudo", "time", "nice", "env", "timeout"}):
        # env/timeout take args before the real command; skip the wrapper + its flag args
        toks = toks[1:]
        while toks and (toks[0].startswith("-") or "=" in toks[0] or toks[0].isdigit()):
            toks = toks[1:]
    if not toks:
        return None, None
    head = toks[0].split("/")[-1]
    sub = toks[1] if len(toks) > 1 and not toks[1].startswith("-") else ""
    shape = f"{head} {sub}".strip() if head in MULTIVERB else head
    rest = toks[2:] if (head in MULTIVERB and sub) else toks[1:]
    arg = next((t for t in rest if not t.startswith("-")), "")
    return shape, arg[:120]


def bash_shape(cmd: str):
    """Return ``(shape, normalized_arg)`` for the operative command in a bash line.

    Splits on ``&&``/``;``/``|``/newlines and skips wrapper segments (``cd``,
    ``export``, ``timeout``, ...) so ``cd repo && rg foo`` yields ``(rg, foo)`` not
    ``(cd, repo)``. ``shape`` is the verb (verb+subcommand for multi-verb tools like
    ``git log``); ``normalized_arg`` keeps the memory-relevant argument.
    """
    if not cmd:
        return None, None
    segs = [s.strip() for s in re.split(r"&&|\|\||[|;]|\n", cmd.strip()) if s.strip()]
    for seg in segs:
        head = seg.split()[0].split("/")[-1] if seg.split() else ""
        if head in WRAP_HEADS or "=" in head:
            continue
        shape, arg = _shape_of_segment(seg)
        if shape:
            return shape, arg
    # everything was a wrapper; fall back to the last segment
    return _shape_of_segment(segs[-1]) if segs else (None, None)


def content_hash(text: str) -> str:
    """Short content fingerprint for a tool result (MD5, non-cryptographic)."""
    return hashlib.md5((text or "").encode("utf-8", "replace")).hexdigest()[:12]


def result_text(block) -> str:
    c = block.get("content")
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def parse_ts(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def iter_records(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tool_target(name: str, inp: dict) -> str | None:
    """Normalized (memory-relevant) target of a tool call: a repo-relative path for
    file tools, or a ``verb\targ`` shape for Bash/Grep/Glob so ``target.split('\t')[0]``
    is always the tool/verb."""
    if name in ("Read", "Write", "Edit"):
        return norm_path(inp.get("file_path"))
    if name == "Bash":
        shp, arg = bash_shape(inp.get("command"))
        return f"{shp}\t{arg}" if shp else None
    if name in ("Grep", "Glob"):
        pat = inp.get("pattern") or inp.get("glob") or inp.get("path")
        verb = "grep" if name == "Grep" else "glob"
        return f"{verb}\t{pat}" if pat else None
    return None


def parse_session(path, block_rows, turn_rows):
    """Append this session's block and turn rows to the two accumulators."""
    session_id = os.path.basename(path)[:-6]
    recs = list(iter_records(path))
    if not recs:
        return
    # Walk records in order, counting assistant turns. n_turns is the session length;
    # a tool_result's first paid turn is the next assistant turn.
    turn_idx = 0
    n_turns = sum(1 for d in recs if d.get("type") == "assistant")
    if n_turns == 0:
        return
    prev_ts = None
    id_to_tool = {}  # tool_use_id -> (tool_name, target)
    for d in recs:
        t = d.get("type")
        ts = parse_ts(d.get("timestamp", ""))
        proj = os.path.basename(os.path.dirname(path))
        cwd = d.get("cwd", "")
        common = {
            "session_id": session_id,
            "project": proj,
            "repo": repo_of(cwd),
            "git_branch": d.get("gitBranch"),
            "version": d.get("version"),
            "ts": ts,
        }
        if t == "assistant":
            turn_idx += 1
            m = d.get("message", {})
            u = m.get("usage", {}) or {}
            cc = u.get("cache_creation", {}) or {}
            gap = (ts - prev_ts) if (prev_ts and ts) else 0.0
            prev_ts = ts or prev_ts
            turn_rows.append(
                {
                    "session_id": session_id,
                    "project": proj,
                    "turn_idx": turn_idx,
                    "n_turns": n_turns,
                    "model": m.get("model"),
                    "input_tokens": u.get("input_tokens", 0) or 0,
                    "cache_creation": u.get("cache_creation_input_tokens", 0) or 0,
                    "cache_read": u.get("cache_read_input_tokens", 0) or 0,
                    "output_tokens": u.get("output_tokens", 0) or 0,
                    "eph_5m": cc.get("ephemeral_5m_input_tokens", 0) or 0,
                    "eph_1h": cc.get("ephemeral_1h_input_tokens", 0) or 0,
                    "gap": gap,
                }
            )
            for b in m.get("content", []):
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "tool_use":
                    name = b.get("name")
                    inp = b.get("input", {}) or {}
                    target = _tool_target(name, inp)
                    id_to_tool[b.get("id")] = (name, target)
                    txt = json.dumps(inp)
                    block_rows.append(
                        {
                            **common,
                            "turn_idx": turn_idx,
                            "role": "assistant",
                            "block_type": "tool_use",
                            "tool_name": name,
                            "target": target,
                            "content_chars": len(txt),
                            "est_tokens": len(txt) / CHARS_PER_TOK,
                            "content_hash": None,
                            "first_paid_turn": turn_idx,
                        }
                    )
                elif bt in ("text", "thinking"):
                    txt = b.get(bt, "") or ""
                    block_rows.append(
                        {
                            **common,
                            "turn_idx": turn_idx,
                            "role": "assistant",
                            "block_type": bt,
                            "tool_name": None,
                            "target": None,
                            "content_chars": len(txt),
                            "est_tokens": len(txt) / CHARS_PER_TOK,
                            "content_hash": None,
                            "first_paid_turn": turn_idx,
                        }
                    )
        elif t == "user":
            m = d.get("message", {})
            c = m.get("content")
            # a tool_result first becomes paid input on the NEXT assistant turn
            first_paid = turn_idx + 1
            if isinstance(c, list):
                for b in c:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        continue
                    name, target = id_to_tool.get(b.get("tool_use_id"), (None, None))
                    txt = result_text(b)
                    block_rows.append(
                        {
                            **common,
                            "turn_idx": turn_idx,
                            "role": "tool_result",
                            "block_type": "tool_result",
                            "tool_name": name,
                            "target": target,
                            "content_chars": len(txt),
                            "est_tokens": len(txt) / CHARS_PER_TOK,
                            "content_hash": content_hash(txt),
                            "first_paid_turn": first_paid,
                        }
                    )
            elif isinstance(c, str):
                block_rows.append(
                    {
                        **common,
                        "turn_idx": turn_idx,
                        "role": "user",
                        "block_type": "user_text",
                        "tool_name": None,
                        "target": None,
                        "content_chars": len(c),
                        "est_tokens": len(c) / CHARS_PER_TOK,
                        "content_hash": None,
                        "first_paid_turn": turn_idx + 1,
                    }
                )


@dataclass(frozen=True)
class ParseConfig:
    """Where the raw transcripts live and where the normalized tables go."""

    projects_dir: str
    session_glob: str
    limit: int
    output_path: str


def run_parse(cfg: ParseConfig) -> None:
    files = sorted(glob.glob(os.path.join(cfg.projects_dir, cfg.session_glob, "*.jsonl")))
    if cfg.limit:
        files = files[: cfg.limit]
    logger.info("parsing %d session files from %s", len(files), cfg.projects_dir)

    block_rows: list[dict] = []
    turn_rows: list[dict] = []
    for i, f in enumerate(files):
        parse_session(f, block_rows, turn_rows)
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d files, %d blocks", i + 1, len(files), len(block_rows))

    blocks = pd.DataFrame(block_rows)
    turns = pd.DataFrame(turn_rows)
    StoragePath(cfg.output_path).mkdirs()
    blocks.to_parquet(prefix_join(cfg.output_path, "blocks.parquet"), index=False)
    turns.to_parquet(prefix_join(cfg.output_path, "turns.parquet"), index=False)
    logger.info("wrote %d blocks, %d turns, %d sessions", len(blocks), len(turns), turns.session_id.nunique())
