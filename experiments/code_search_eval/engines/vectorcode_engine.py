# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["vectorcode>=0.7", "chromadb==0.6.3"]
#
# [tool.uv]
# prerelease = "allow"
# ///

"""VectorCode (https://github.com/Davidyz/VectorCode) adapter: a local ChromaDB-backed
code index driven through the `vectorcode` CLI. Embeddings come from chromadb's
`SentenceTransformerEmbeddingFunction` (all-MiniLM-L6-v2, downloaded once from the
Hugging Face Hub and cached like every other engine's model weights) — no cloud calls,
no API keys.

VectorCode ties its ChromaDB path and collection identity to `--project_root`, so this
adapter points `--project_root` at `index_dir` itself (never at `repo_root`) and writes
a project-local `.vectorcode/config.json` there pinning `db_path`/`db_log_path` under
`index_dir`. Each `vectorcode` invocation starts and tears down its own bundled,
localhost-only ChromaDB server subprocess against that on-disk database, so no state
lands in the repo or in the caller's home directory.

`chromadb` is pinned explicitly because an unconstrained resolve of VectorCode's
`chromadb<=0.6.3` picks a version too old to expose the async client API VectorCode
imports; that pin in turn needs `prerelease = "allow"` because chromadb 0.6.3 depends
on a beta-versioned `opentelemetry-instrumentation-fastapi` release.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])

from experiments.code_search_eval.common import Hit, iter_source_files, read_text, run_engine_cli

VECTORCODE_TIMEOUT = 3600  # first run downloads the sentence-transformers model
FALLBACK_WINDOW = 40  # line window used only if a hit somehow lacks chunk line spans


def _configure(index_dir: str) -> None:
    """Write a project-local VectorCode config pinning db_path/db_log_path under `index_dir`."""
    db_path = os.path.join(index_dir, "chromadb")
    db_log_path = os.path.join(index_dir, "vc_logs")
    vc_dir = os.path.join(index_dir, ".vectorcode")
    os.makedirs(db_path, exist_ok=True)
    os.makedirs(db_log_path, exist_ok=True)
    os.makedirs(vc_dir, exist_ok=True)
    config = {
        "db_path": db_path,
        "db_log_path": db_log_path,
        "embedding_function": "SentenceTransformerEmbeddingFunction",
        "embedding_params": {"model_name": "all-MiniLM-L6-v2"},
    }
    with open(os.path.join(vc_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)


def _run(args: list[str], index_dir: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, ANONYMIZED_TELEMETRY="False")
    return subprocess.run(
        ["vectorcode", "--project_root", index_dir, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=VECTORCODE_TIMEOUT,
    )


def build_index(repo_root: str, index_dir: str) -> None:
    _configure(index_dir)
    files = [os.path.join(repo_root, rel) for rel in iter_source_files(repo_root)]
    if not files:
        raise ValueError(f"no indexable files under {repo_root}")
    # --force: skip .gitignore-based exclusion (there is none at project_root=index_dir anyway;
    # this just makes the behavior explicit rather than relying on that absence).
    proc = _run(["vectorise", "--pipe", "--force", *files], index_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"vectorcode vectorise failed: {proc.stderr}")
    stats = json.loads(proc.stdout)
    if stats.get("failed"):
        raise RuntimeError(f"vectorcode failed to vectorise {stats['failed']} file(s): {proc.stderr}")


def query_index(repo_root: str, index_dir: str, queries: list[dict], k: int) -> list[dict]:
    results = []
    for q in queries:
        # --absolute: VectorCode's own relative-path mode is relative to --project_root
        # (index_dir), not repo_root, so we resolve repo-relative paths ourselves below.
        # --include chunk: return line-range chunks instead of whole-file documents.
        proc = _run(
            ["query", "--pipe", "--absolute", "--include", "chunk", "-n", str(k), q["query"]],
            index_dir,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"vectorcode query failed: {proc.stderr}")
        raw = json.loads(proc.stdout)
        hits = []
        # VectorCode's --pipe output already lists chunks best-first (it does not expose
        # the underlying ChromaDB distance through this interface), so rank position is
        # the only ranking signal available; turn it into a descending synthetic score.
        for rank, item in enumerate(raw[:k]):
            relpath = os.path.relpath(os.path.expanduser(item["path"]), repo_root)
            start, end = item.get("start_line"), item.get("end_line")
            if start is None or end is None:
                start = 1
                end = min(len(read_text(repo_root, relpath).splitlines()), FALLBACK_WINDOW) or 1
            hits.append(Hit(relpath, start, end, float(len(raw) - rank)).to_json())
        results.append({"query_id": q["query_id"], "hits": hits})
    return results


if __name__ == "__main__":
    run_engine_cli(build_index, query_index)
