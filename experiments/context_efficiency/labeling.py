# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic labeling and topic clustering via a headless agent (``claude -p`` or any
other headless CLI, e.g. ``codex exec``).

The labeling step is where the analysis becomes *semantic*: for each sampled
episode, a sub-agent judges why the tool calls happened, what they sought, and
whether a wiki / semantic index / persistent memory / better tool / result
compaction would have served the same need with a smaller answer. The agent is
invoked once per batch, reading its prompt from stdin and returning a JSON object on
stdout — the same agent-agnostic contract the lint ``--review`` uses.

Two steps:

- :func:`run_labeling` labels every batch (or a deterministic ``fraction`` of them,
  for a stronger-model validation pass) into ``batch_NNN.json`` files. Batch-level
  idempotent: a re-run relabels only the batches still missing a valid file.
- :func:`run_clustering` collapses the fragmented per-episode ``wiki_topic_slug``s
  into the maintainable-doc topics a team would actually keep, so recurrence is
  later measured per doc rather than per raw slug.
"""

import hashlib
import json
import logging
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from rigging.filesystem import StoragePath, prefix_join

from experiments.context_efficiency.schema import ANSWER_KINDS, INTENT_CATEGORIES, SUBSTITUTES, SUFFICIENCY

logger = logging.getLogger(__name__)

RUBRIC = f"""\
You are labeling tool-call *episodes* from AI coding-agent sessions to quantify how \
much token budget a memory/wiki/index/tooling change could save. Each episode is a \
run of information-gathering tool calls (Read, Grep, Glob, Bash, ...) serving one \
sub-goal. You are given the user request that governs the episode, and each call's \
stated intent, the call itself, its result size in tokens, and a short result preview.

For EACH episode, decide what — if anything — would have let the agent skip these \
calls or get the same information in a much smaller answer. The economic test that \
matters: a substitute only helps if BOTH its coverage is sufficient (it actually \
answers the question) AND its answer is smaller than the tool results it replaces \
(a one-paragraph article beats re-grepping; a 200-line file dump it merely points to \
does not). Judge the substitute against the tokens actually consumed.

Return a single JSON object, and nothing else:
{{"labels": [ {{one object per episode, echoing episode_id}} ]}}

Each label object has these fields:
- episode_id: string, copied from the episode.
- intent_category: one of {INTENT_CATEGORIES}
- answer_kind: one of {ANSWER_KINDS}
- best_substitute: one of {SUBSTITUTES}
    none                = irreducible (must run these calls live: git/PR state, test runs,
                          reading the agent's own just-written output, genuinely novel code).
    semantic-code-index = a code-aware search/index (RAG over the repo) would locate it directly.
    better-tool-or-flag = a better tool default or flag would return a far smaller result
                          (e.g. head/-n limits, structured query instead of full-file read).
    result-compaction   = the call is fine but its result should be summarized/truncated on the way in.
    repo-map-or-docs    = a per-repo map or structured docs (one artifact per repo) would answer it.
    shared-wiki         = a durable shared article about a recurring fact/how-to would forestall it.
    persistent-memory   = a per-agent remembered fact from a prior session would forestall it.
- substitute_sufficient: one of {SUFFICIENCY} — would the substitute actually cover the need?
- substitute_size_ratio: number in [0,1] = (substitute answer tokens + any residual live lookups)
    / (original episode result tokens). 1.0 = no saving; 0.1 = the substitute is ~10x smaller.
    Be honest about residual lookups — most substitutes still need one confirming call.
- wiki_topic_slug: a kebab-case topic slug IF a durable article/memory could serve this episode
    (e.g. "iris-cluster-restart", "levanter-checkpoint-format"), else "". Use the SAME slug for
    episodes about the same topic so recurrence can be measured.

Be conservative: when unsure, prefer best_substitute="none" and substitute_size_ratio=1.0. \
A wrong "a wiki would fix this" over-counts the saving.

Episodes to label (one JSON object per line):
"""

CLUSTER_PROMPT = """\
Below is a list of raw topic slugs that were each proposed as a candidate wiki/doc \
topic while labeling AI coding-agent sessions. Many are near-duplicates or overly \
specific. Group them into the maintainable documentation topics a team would \
actually keep — one cluster per doc a human would own and update (a subsystem, a \
recurring how-to, an architecture area). Prefer coarser clusters: a good target is \
one cluster per ~4-8 slugs, not one per slug.

Return a single JSON object and nothing else:
{"assignments": [ {"slug": "<raw slug>", "cluster": "<cluster slug>"}, ... ]}

Every input slug must appear exactly once. Cluster slugs are kebab-case.

Slugs:
"""


def extract_json(stdout: str) -> dict | None:
    """Pull the JSON object out of an agent's stdout.

    Handles a raw JSON object, a Claude ``--output-format json`` envelope (the
    payload lands in ``structured_output`` or as text in ``result``), a fenced
    ```json block, and prose wrapped around the object. Returns None if nothing
    parses.
    """
    text = stdout.strip()
    if not text:
        return None
    # Claude --output-format json envelope
    try:
        env = json.loads(text)
        if isinstance(env, dict) and ("labels" in env or "assignments" in env):
            return env
        if isinstance(env, dict) and isinstance(env.get("structured_output"), dict):
            return env["structured_output"]
        if isinstance(env, dict) and isinstance(env.get("result"), str):
            text = env["result"]
    except (json.JSONDecodeError, ValueError):
        pass
    # first balanced object in the (possibly prose/fenced) text
    start = text.find("{")
    while start != -1:
        depth, instr, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find("{", start + 1)
    return None


def run_agent(agent_cmd: list[str], prompt: str, timeout: int) -> str | None:
    """Invoke the headless agent with the prompt on stdin; return stdout or None."""
    try:
        proc = subprocess.run(agent_cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("agent timed out after %ds", timeout)
        return None
    if proc.returncode != 0:
        logger.warning("agent exited %s: %s", proc.returncode, (proc.stderr or "").strip()[:200])
        return None
    return proc.stdout


def _label_one_batch(batch_path: str, out_path: str, agent_cmd: list[str], timeout: int, retries: int) -> int:
    """Label one batch file, writing ``out_path``. Returns the number of labels, 0 on failure."""
    episodes = [json.loads(line) for line in StoragePath(batch_path).read_text().splitlines() if line.strip()]
    ids = {e["episode_id"] for e in episodes}
    prompt = RUBRIC + "\n".join(json.dumps(e) for e in episodes)
    for attempt in range(retries + 1):
        stdout = run_agent(agent_cmd, prompt, timeout)
        parsed = extract_json(stdout) if stdout else None
        labels = parsed.get("labels") if isinstance(parsed, dict) else None
        if isinstance(labels, list):
            kept = [x for x in labels if isinstance(x, dict) and x.get("episode_id") in ids]
            if kept:
                StoragePath(out_path).write_text(json.dumps({"labels": kept}, indent=2))
                return len(kept)
        logger.warning(
            "batch %s: no usable labels (attempt %d/%d)", os.path.basename(batch_path), attempt + 1, retries + 1
        )
    return 0


def _select_batches(bdir: str, fraction: float) -> list[str]:
    """The batch files to label — a deterministic prefix ``fraction`` of them.

    ``fraction >= 1`` labels all batches; ``0 < fraction < 1`` a prefix (the
    stronger-model validation subset); ``fraction <= 0`` disables the step.
    """
    if fraction <= 0:
        return []
    batches = sorted(str(p) for p in StoragePath(prefix_join(bdir, "batch_*.jsonl")).glob())
    if fraction >= 1.0:
        return batches
    k = max(1, round(len(batches) * fraction))
    return batches[:k]


@dataclass(frozen=True)
class LabelConfig:
    episodes_path: str
    output_path: str
    agent_command: str
    fraction: float
    concurrency: int
    timeout: int
    retries: int
    min_coverage: float


def run_labeling(cfg: LabelConfig) -> None:
    agent_cmd = shlex.split(cfg.agent_command)
    bdir = prefix_join(cfg.episodes_path, "episode_batches")
    selected = _select_batches(bdir, cfg.fraction)
    StoragePath(cfg.output_path).mkdirs()
    if not selected:
        logger.info("labeling disabled (fraction=%s); nothing to label", cfg.fraction)
        return

    def out_for(batch_path: str) -> str:
        # Name the label file by the batch's content hash, not its index, so a re-run
        # after the sample changed relabels the new content instead of silently reusing
        # a stale label file that happens to share the batch index.
        bhash = hashlib.sha256(StoragePath(batch_path).read_text().encode()).hexdigest()[:12]
        return prefix_join(cfg.output_path, f"labels_{bhash}.json")

    todo = [b for b in selected if not StoragePath(out_for(b)).exists()]
    logger.info(
        "labeling %d/%d batches with '%s' (%d already done)",
        len(todo),
        len(selected),
        cfg.agent_command,
        len(selected) - len(todo),
    )

    done = len(selected) - len(todo)
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futures = {ex.submit(_label_one_batch, b, out_for(b), agent_cmd, cfg.timeout, cfg.retries): b for b in todo}
        for fut in as_completed(futures):
            n = fut.result()
            if n:
                done += 1
            if done % 10 == 0:
                logger.info("  labeled %d/%d batches", done, len(selected))

    coverage = done / max(len(selected), 1)
    logger.info("labeled %d/%d batches (%.0f%% coverage)", done, len(selected), 100 * coverage)
    if coverage < cfg.min_coverage:
        raise ValueError(
            f"only {done}/{len(selected)} batches labeled ({coverage:.0%} < {cfg.min_coverage:.0%} floor); "
            f"check the agent command '{cfg.agent_command}'. Re-run to relabel the missing batches."
        )


@dataclass(frozen=True)
class ClusterConfig:
    labels_path: str
    output_path: str
    agent_command: str
    timeout: int


def run_clustering(cfg: ClusterConfig) -> None:
    slugs = set()
    for fp in sorted(str(p) for p in StoragePath(prefix_join(cfg.labels_path, "*.json")).glob()):
        try:
            d = json.loads(StoragePath(fp).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for lab in d.get("labels", []):
            s = (lab.get("wiki_topic_slug") or "").strip()
            if s:
                slugs.add(s)
    slugs = sorted(slugs)
    StoragePath(cfg.output_path).mkdirs()
    out = prefix_join(cfg.output_path, "topic_clusters.json")
    if not slugs:
        StoragePath(out).write_text(json.dumps({"assignments": []}, indent=2))
        logger.info("no wiki topic slugs to cluster")
        return

    prompt = CLUSTER_PROMPT + "\n".join(f"- {s}" for s in slugs)
    stdout = run_agent(shlex.split(cfg.agent_command), prompt, cfg.timeout)
    parsed = extract_json(stdout) if stdout else None
    assignments = parsed.get("assignments") if isinstance(parsed, dict) else None
    if not isinstance(assignments, list):
        # Degrade to identity clustering (each slug its own cluster) rather than fail
        # the pipeline; the analysis then reports the per-slug (lower) bound only.
        logger.warning("clustering agent returned no assignments; falling back to per-slug identity")
        assignments = [{"slug": s, "cluster": s} for s in slugs]
    StoragePath(out).write_text(json.dumps({"assignments": assignments}, indent=2))
    logger.info("clustered %d slugs into %d clusters", len(slugs), len({a.get("cluster") for a in assignments}))
