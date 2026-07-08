# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a code-search benchmark from real agent sessions.

Each benchmark item is a *navigation moment* mined from ``~/.claude`` transcripts: a
run of Grep/Glob/Read calls that ended when the agent read or edited a specific file.
That file is the **gold** answer; the agent's stated intent becomes the query. A
headless agent (``claude -p``) then rewrites each raw intent into a clean
natural-language question and drops the moments that were not genuine
repository-location needs (running tests, inspecting live git state, reading the
agent's own output). This grounds the benchmark in needs that actually arose while
working in the repo, and gives a gold file for exact recall without hand-labeling.

The gold file is the one the agent settled on, so it is verified to still exist in
``repo_root``; moments whose target has since been deleted or renamed are dropped so
recall is measurable against the current tree.

Output (under the step's output directory):

- ``benchmark.jsonl`` — one ``{query_id, query, gold_files, session_id, ...}`` per line.
- ``benchmark_meta.json`` — population counts and derivation parameters.
"""

import glob
import json
import logging
import os
import random
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from rigging.filesystem import StoragePath, prefix_join

from experiments.code_search_eval.common import read_jsonl, write_jsonl
from experiments.context_efficiency.episodes import block_list, text_of
from experiments.context_efficiency.labeling import extract_json, run_agent
from experiments.context_efficiency.transcripts import iter_records, norm_path

logger = logging.getLogger(__name__)

SEARCH_TOOLS = {"Grep", "Glob"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
MAX_WIN_STEPS = 12  # cap a navigation window before forcing a boundary
MIN_QUERY_CHARS = 15
INTENT_CHARS = 220

# Worktree-internal prefixes that sit between the repo root and the canonical path.
WORKTREE_INNER_RE = re.compile(r"^(?:\.claude/worktrees|\.worktrees|\.claude-worktrees)/[^/]+/")

CLEAN_PROMPT = """\
You are turning AI coding-agent navigation moments into clean natural-language code-search \
queries for a retrieval benchmark. Each item is a moment where the agent went looking for \
something in a code repository: you get the agent's stated intent and the raw search terms \
it used.

For EACH item decide:
- is_navigation: true if this was a genuine "find where X is / locate the code for Y" need \
that a code search engine could serve; false if it was running tests, inspecting live \
git/PR/CI state, reading the agent's own just-written output, or otherwise not a \
repository-location need.
- query: a single clear natural-language question describing what code the agent was trying \
to find (e.g. "where is the autoscaler demand gap computed", "how does iris register a task \
endpoint proxy"). Do NOT include file paths or line numbers. If is_navigation is false, use "".

Return one JSON object and nothing else:
{"results": [ {"cand_id": "<copied>", "is_navigation": true, "query": "..."} ]}

Items (one JSON object per line):
"""


def _rel_gold(path: str, repo_root: str) -> str | None:
    """Canonical repo-relative form of a tool's file_path, or None if it is not a file
    that lives in ``repo_root`` (scratch/tmp output, or a since-deleted target)."""
    rel = norm_path(path)
    if not rel:
        return None
    rel = WORKTREE_INNER_RE.sub("", rel)  # map a worktree copy back onto the canonical path
    if rel.startswith("/") or rel.startswith(".."):  # escaped the repo (tmp/scratch/home) — not indexable
        return None
    return rel if os.path.exists(os.path.join(repo_root, rel)) else None


def _is_boilerplate(query: str) -> bool:
    """A systematic role/system prompt, not a human navigation need."""
    q = query.lstrip()
    return q.startswith("##") or "READ-ONLY reviewer" in query or "You are a" in query[:40]


def extract_candidates(path: str, repo_root: str) -> list[dict]:
    """Mine navigation moments (raw query + gold files) from one session transcript."""
    recs = [d for d in iter_records(path) if d.get("type") in ("user", "assistant") and not d.get("isSidechain")]
    sid = os.path.basename(path)[:-6]
    out: list[dict] = []
    user_request = ""
    first_intent = ""
    pending = ""
    reads: list[str] = []
    searches: list[str] = []
    edits: list[str] = []
    seq = 0

    def flush():
        nonlocal reads, searches, edits, first_intent, seq
        gold: list[str] = []
        for p in edits + reads:  # the edited/settled file is the strongest answer, list it first
            g = _rel_gold(p, repo_root)
            if g and g not in gold:
                gold.append(g)
        query = (first_intent or user_request).strip()
        if gold and (searches or len(reads) >= 2) and len(query) >= MIN_QUERY_CHARS and not _is_boilerplate(query):
            out.append(
                {
                    "cand_id": f"{sid[:8]}-{seq:03d}",
                    "session_id": sid,
                    "raw_query": query[:600],
                    "search_terms": searches[:6],
                    "gold_files": gold[:5],
                }
            )
            seq += 1
        reads, searches, edits, first_intent = [], [], [], ""

    for d in recs:
        msg = d.get("message")
        if not isinstance(msg, dict):
            continue
        role = d.get("type")
        blocks = block_list(msg)
        is_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)
        if role == "user" and not is_tool_result:
            flush()  # a human message ends the current goal
            user_request = text_of(msg.get("content"))
            pending = ""
            continue
        if role == "user" and is_tool_result:
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                pending = (pending + " " + b.get("text", "")).strip()[:INTENT_CHARS]
            elif b.get("type") == "tool_use":
                name = b.get("name")
                inp = b.get("input")
                inp = inp if isinstance(inp, dict) else {}
                if name in EDIT_TOOLS:
                    fp = inp.get("file_path")
                    if fp:
                        edits.append(fp)
                    flush()  # searching ended in an edit; that file is the target
                elif name in SEARCH_TOOLS:
                    pat = inp.get("pattern") or inp.get("glob") or inp.get("path")
                    if pat:
                        searches.append(str(pat)[:120])
                    if not first_intent:
                        first_intent = pending
                elif name == "Read":
                    fp = inp.get("file_path")
                    if fp:
                        reads.append(fp)
                    if not first_intent:
                        first_intent = pending
                if len(reads) + len(searches) >= MAX_WIN_STEPS:
                    flush()
                pending = ""
    flush()
    return out


@dataclass(frozen=True)
class BenchmarkConfig:
    projects_dir: str
    session_glob: str
    repo_root: str
    limit: int  # cap sessions scanned (0 = all)
    max_candidates: int  # cap raw candidates fed to the cleaner (seeded subsample)
    max_queries: int  # final benchmark size
    seed: int
    agent_command: str
    concurrency: int
    timeout: int
    batch: int  # candidates per agent call
    output_path: str


def _clean_batch(cands: list[dict], agent_cmd: list[str], timeout: int) -> list[dict]:
    """Ask the agent to clean one batch of candidates; return kept navigation queries."""
    ids = {c["cand_id"] for c in cands}
    items = [{"cand_id": c["cand_id"], "intent": c["raw_query"], "search_terms": c["search_terms"]} for c in cands]
    prompt = CLEAN_PROMPT + "\n".join(json.dumps(it) for it in items)
    stdout = run_agent(agent_cmd, prompt, timeout)
    parsed = extract_json(stdout) if stdout else None
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        return []
    return [
        r
        for r in results
        if isinstance(r, dict) and r.get("cand_id") in ids and r.get("is_navigation") and (r.get("query") or "").strip()
    ]


def run_benchmark(cfg: BenchmarkConfig) -> None:
    files = sorted(glob.glob(os.path.join(cfg.projects_dir, cfg.session_glob, "*.jsonl")))
    if cfg.limit:
        files = files[: cfg.limit]
    logger.info("mining navigation moments from %d session files", len(files))

    candidates: list[dict] = []
    for path in files:
        try:
            candidates.extend(extract_candidates(path, cfg.repo_root))
        except (OSError, UnicodeDecodeError):
            continue
    logger.info("%d raw navigation candidates with in-repo gold", len(candidates))
    if not candidates:
        raise ValueError(f"no navigation candidates from {cfg.projects_dir}/{cfg.session_glob}; check repo_root")

    rng = random.Random(cfg.seed)
    rng.shuffle(candidates)
    pool = candidates[: cfg.max_candidates]
    by_id = {c["cand_id"]: c for c in pool}

    agent_cmd = shlex.split(cfg.agent_command)
    batches = [pool[i : i + cfg.batch] for i in range(0, len(pool), cfg.batch)]
    cleaned: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futures = [ex.submit(_clean_batch, b, agent_cmd, cfg.timeout) for b in batches]
        for fut in as_completed(futures):
            cleaned.extend(fut.result())
    logger.info("agent kept %d/%d candidates as navigation queries", len(cleaned), len(pool))

    # Join the cleaned query back onto its gold, dedup by gold set, cap to the target size.
    rows: list[dict] = []
    seen_gold: set[tuple[str, ...]] = set()
    for r in cleaned:
        c = by_id.get(r["cand_id"])
        if not c:
            continue
        key = tuple(sorted(c["gold_files"]))
        if key in seen_gold:
            continue
        seen_gold.add(key)
        rows.append(
            {
                "query_id": c["cand_id"],
                "query": r["query"].strip(),
                "gold_files": c["gold_files"],
                "session_id": c["session_id"],
                "raw_query": c["raw_query"],
                "search_terms": c["search_terms"],
            }
        )
    rows = rows[: cfg.max_queries]
    if not rows:
        raise ValueError("agent rejected every candidate as non-navigation; check the agent command")

    StoragePath(cfg.output_path).mkdirs()
    write_jsonl(prefix_join(cfg.output_path, "benchmark.jsonl"), rows)
    with StoragePath(prefix_join(cfg.output_path, "benchmark_meta.json")).open("w") as fh:
        json.dump(
            {
                "n_sessions_scanned": len(files),
                "n_raw_candidates": len(candidates),
                "n_cleaned_pool": len(pool),
                "n_kept_navigation": len(cleaned),
                "n_benchmark_queries": len(rows),
                "repo_root": cfg.repo_root,
                "seed": cfg.seed,
            },
            fh,
            indent=2,
        )
    logger.info("wrote %d benchmark queries to %s", len(rows), cfg.output_path)


def load_benchmark(output_path: str) -> list[dict]:
    return read_jsonl(prefix_join(output_path, "benchmark.jsonl"))
