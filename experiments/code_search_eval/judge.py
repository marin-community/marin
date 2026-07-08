# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge each engine's results with a headless agent (``claude -p``).

For every query, the judge sees the engine's ranked snippets (file + line range +
text, extracted uniformly from the repo) and returns the first rank that actually
answers the need, plus a full/partial/none verdict. Judging once at the full K and
recording that rank lets the scorer read off ``judge_pass@k`` for every k by
comparison — the same "run once, truncate" idea as recall@k.

The judge never sees the gold file, so ``judge_pass@k`` is an independent semantic
measure of answer quality, complementing the deterministic gold ``recall@k``.

Output: ``<engine>_judge.jsonl`` — one ``{query_id, best_rank, verdict}`` per query.
"""

import json
import logging
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from rigging.filesystem import StoragePath, prefix_join

from experiments.code_search_eval.common import read_jsonl, snippet, write_jsonl
from experiments.context_efficiency.labeling import agent_json_list

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 1200  # cap per-snippet text shown to the judge

JUDGE_PROMPT = """\
You are evaluating a code search engine for a retrieval benchmark. For each query you \
get the developer's search need and the engine's top ranked code snippets (each with its \
file path, line range, and text). Decide how well the ranked list answers the need.

Return one JSON object and nothing else:
{"results": [ {"query_id": "<copied>", "best_rank": <int or null>, "verdict": "full"|"partial"|"none"} ]}

- "full": some snippet in the list contains the actual code the query asks for (the \
definition / implementation / location sought).
- "partial": a snippet points to the right file or area but not the exact code — you'd \
still open the file and look around.
- "none": nothing in the list answers the query.
- best_rank: the 1-indexed position of the first snippet you count as full; if there is no \
full match, the first partial; null if none.

Judge only from the snippets shown; do not assume unshown context. Be strict: ranking a \
loosely-related file is "partial" at best.

Queries and ranked results (one JSON object per line):
"""


@dataclass(frozen=True)
class JudgeConfig:
    engine: str
    benchmark_path: str
    hits_path: str  # dir holding <engine>_hits.jsonl
    repo_root: str
    output_path: str
    agent_command: str
    concurrency: int
    timeout: int
    batch: int


def _judge_item(query: dict, hits: list[dict], repo_root: str) -> dict:
    """Build the judge payload for one query: its text + ranked snippets."""
    snippets = []
    for rank, h in enumerate(hits, start=1):
        text = snippet(repo_root, h["file"], h["start_line"], h["end_line"])[:SNIPPET_CHARS]
        snippets.append({"rank": rank, "file": h["file"], "lines": f"{h['start_line']}-{h['end_line']}", "code": text})
    return {"query_id": query["query_id"], "query": query["query"], "results": snippets}


def _judge_batch(items: list[dict], agent_cmd: list[str], timeout: int) -> list[dict]:
    ids = {it["query_id"] for it in items}
    prompt = JUDGE_PROMPT + "\n".join(json.dumps(it) for it in items)
    return [r for r in agent_json_list(agent_cmd, prompt, timeout, "results") if r.get("query_id") in ids]


def run_judge(cfg: JudgeConfig) -> None:
    benchmark = {q["query_id"]: q for q in read_jsonl(prefix_join(cfg.benchmark_path, "benchmark.jsonl"))}
    hits_by_query = {
        h["query_id"]: h["hits"] for h in read_jsonl(prefix_join(cfg.hits_path, f"{cfg.engine}_hits.jsonl"))
    }

    items = [
        _judge_item(benchmark[qid], hits_by_query.get(qid, []), cfg.repo_root)
        for qid in benchmark
        if qid in hits_by_query
    ]
    batches = [items[i : i + cfg.batch] for i in range(0, len(items), cfg.batch)]
    agent_cmd = shlex.split(cfg.agent_command)
    verdicts: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futures = [ex.submit(_judge_batch, b, agent_cmd, cfg.timeout) for b in batches]
        for fut in as_completed(futures):
            verdicts.extend(fut.result())

    coverage = len(verdicts) / max(len(items), 1)
    logger.info("judged %s: %d/%d queries verdicted (%.0f%%)", cfg.engine, len(verdicts), len(items), 100 * coverage)
    StoragePath(cfg.output_path).mkdirs()
    write_jsonl(prefix_join(cfg.output_path, f"{cfg.engine}_judge.jsonl"), verdicts)
