# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["seagoat>=0.54"]
#
# [tool.uv]
# # chromadb>=1.4 (pulled in by seagoat>=0.54.5) depends on pre-release opentelemetry
# # packages; --no-project skips this repo's [tool.uv] prerelease="explicit" setting.
# prerelease = "allow"
# ///

"""SeaGOAT semantic search: a local ChromaDB + MiniLM embedding index over per-line
code chunks, driven directly through SeaGOAT's ``Engine`` library API (no server
process, no network calls beyond the one-time embedding-model download).

SeaGOAT expects ``repo_root`` to be a git repository's top level: it lists files
with ``rg --files`` relative to ``repo_root`` but reads history with ``git log``,
whose paths are relative to the enclosing repository's root. Inside a monorepo
checkout those disagree, and every file silently drops out of the index. To keep
this adapter usable on a monorepo subtree, ``build_index`` mirrors the indexable
files into a single-commit shadow git repository under ``index_dir`` and points
SeaGOAT at that instead; relative paths are unchanged, so results still map back
onto ``repo_root``.

SeaGOAT's own cache (ChromaDB data, ripgrep mmap, analyzed-chunk bookkeeping) also
lives outside ``index_dir`` by default, under an ``appdirs`` user-cache directory.
``RUNNER_TEMP`` is the one environment variable SeaGOAT honors to relocate that
root, so both build and query set it to ``index_dir`` before touching the engine.
"""

import os
import shutil
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])

from seagoat.engine import Engine

from experiments.code_search_eval.common import Hit, iter_source_files, run_engine_cli

CONTEXT_ABOVE, CONTEXT_AFTER = 3, 12  # mirrors ripgrep_engine's RG_BEFORE/RG_AFTER context window
SHADOW_DIRNAME = "shadow_repo"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "seagoat-adapter",
    "GIT_AUTHOR_EMAIL": "seagoat-adapter@localhost",
    "GIT_COMMITTER_NAME": "seagoat-adapter",
    "GIT_COMMITTER_EMAIL": "seagoat-adapter@localhost",
}


def _shadow_repo_path(repo_root: str, index_dir: str) -> str:
    """Single-commit git mirror of ``repo_root``'s indexable files, built once and reused."""
    shadow = os.path.join(index_dir, SHADOW_DIRNAME)
    if os.path.isdir(os.path.join(shadow, ".git")):
        return shadow
    for rel in iter_source_files(repo_root):
        dst = os.path.join(shadow, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(repo_root, rel), dst)
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q"], cwd=shadow, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=shadow, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "snapshot"], cwd=shadow, check=True, env=env)
    return shadow


def _engine(repo_root: str, index_dir: str) -> Engine:
    os.environ["RUNNER_TEMP"] = index_dir  # redirects seagoat's chroma/ripgrep/pickle cache here
    return Engine(_shadow_repo_path(repo_root, index_dir))


def build_index(repo_root: str, index_dir: str) -> None:
    engine = _engine(repo_root, index_dir)
    # Engine.process_chunk pickles its full analyzed-chunk-id cache to disk after every
    # single chunk, which is fine interactively but O(n^2) over a from-scratch index of
    # 10^5 line-level chunks. Persist once at the end instead of after each chunk.
    real_persist = engine.cache.persist
    engine.cache.persist = lambda: None
    try:
        while engine.analyze_codebase():
            pass
    finally:
        engine.cache.persist = real_persist
        engine.cache.persist()


def query_index(repo_root: str, index_dir: str, queries: list[dict], k: int) -> list[dict]:
    engine = _engine(repo_root, index_dir)
    results = []
    for q in queries:
        matches = engine.query_sync(
            q["query"], limit_clue=max(k * 4, 20), context_above=CONTEXT_ABOVE, context_below=CONTEXT_AFTER
        )
        blocks = []
        for result in matches:
            result_json = result.to_json()
            for block in result_json["blocks"]:
                lines = [ln["line"] for ln in block["lines"]]
                blocks.append((block["score"], result_json["path"], min(lines), max(lines)))
        blocks.sort(key=lambda b: b[0])  # SeaGOAT block score is a vector distance: lower is better
        hits = [Hit(path, start, end, -score).to_json() for score, path, start, end in blocks[:k]]
        results.append({"query_id": q["query_id"], "hits": hits})
    return results


if __name__ == "__main__":
    run_engine_cli(build_index, query_index)
