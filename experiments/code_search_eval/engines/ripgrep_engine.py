# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""ripgrep baseline: extract identifier-like keywords from the query and rank files
by how many distinct keywords they contain. No index — this is the current default
an agent already has, and the bar every semantic engine must clear.

Each hit is the best-matching line widened to a small context window (what an agent
reading with ``rg -A/-B`` would actually see), so its answer size is comparable to
the chunk-returning engines.
"""

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])

from experiments.code_search_eval.common import Hit, run_engine_cli

RG_BEFORE, RG_AFTER = 3, 12  # context window around the matched line
MAX_KEYWORDS = 8
# Query filler that is not a code identifier; drop before searching.
STOPWORDS = frozenset(
    """the a an is are was where how does do did what which who whom this that these those
    let me check look read verify find locate current state relevant all any code file function
    method class module for and or not with without into onto from to of in on at by we i you it
    its their there here now then also just still see get set use used using make made get show
    whether if else when while about around across over under via can could would should will
    need want going want confirm ensure surrounding context two one more full exact""".split()
)


def keywords(query: str) -> list[str]:
    """Identifier-like tokens from the query, snake/camel/backtick terms preferred."""
    quoted = re.findall(r"[`\"']([A-Za-z_][\w./]{2,})[`\"']", query)
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    ranked = []
    for tok in quoted + idents:
        low = tok.lower()
        if low in STOPWORDS or tok in ranked:
            continue
        ranked.append(tok)
    # identifiers carrying _ or internal caps are far more selective than plain words
    ranked.sort(key=lambda t: ("_" not in t and t.islower(), len(t) < 5))
    return ranked[:MAX_KEYWORDS]


def build_index(repo_root: str, index_dir: str) -> None:
    """ripgrep needs no index."""


def _search(repo_root: str, kws: list[str]) -> dict[str, dict[int, set[str]]]:
    """file -> {line_number -> set(matched keywords)} via one ripgrep JSON pass."""
    if not kws:
        return {}
    cmd = ["rg", "--json", "-i", "--max-count", "40"]
    for k in kws:
        cmd += ["-e", re.escape(k)]
    cmd.append(repo_root)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    kws_low = [k.lower() for k in kws]
    hits: dict[str, dict[int, set[str]]] = {}
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        d = obj["data"]
        rel = d["path"]["text"]
        lineno = d["line_number"]
        text = (d.get("lines", {}).get("text") or "").lower()
        matched = {k for k, kl in zip(kws, kws_low, strict=False) if kl in text}
        if matched:
            hits.setdefault(rel, {}).setdefault(lineno, set()).update(matched)
    return hits


def query_index(repo_root: str, index_dir: str, queries: list[dict], k: int) -> list[dict]:
    results = []
    for q in queries:
        kws = keywords(q["query"])
        per_file = _search(repo_root, kws)
        scored = []
        for rel, lines in per_file.items():
            distinct = set().union(*lines.values())
            total = sum(len(v) for v in lines.values())
            best_line = max(lines, key=lambda ln: (len(lines[ln]), -ln))
            scored.append((len(distinct), total, rel, best_line))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        hits = []
        for distinct, _total, rel, line in scored[:k]:
            path = rel if os.path.isabs(rel) else os.path.join(repo_root, rel)
            relpath = os.path.relpath(path, repo_root)
            hits.append(Hit(relpath, max(1, line - RG_BEFORE), line + RG_AFTER, float(distinct)).to_json())
        results.append({"query_id": q["query_id"], "hits": hits})
    return results


if __name__ == "__main__":
    run_engine_cli(build_index, query_index)
