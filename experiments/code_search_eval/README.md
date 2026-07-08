# Code search evaluation

Benchmarks local semantic code-search engines against a ripgrep baseline, on queries
mined from real agent sessions, scored by exact gold recall **and** an agent judge.

This is the empirical follow-up to the context-efficiency analysis
(`experiments/context_efficiency/`), whose top recommendation was an *automatic
semantic code index*. That analysis assumed such an index would locate code with a
smaller answer than repeated grepping. This experiment tests whether that holds and
which engine delivers it.

## The question

For a realistic navigation need ("where is the autoscaler demand gap computed"), which
engine gets the answer into the smallest top-k? We report every metric at k ∈ {1,3,5,10}
by running each engine once at K=10 and truncating — so "answer size" is just pass@k at
small k (the "you can always `head -n1`" argument): an engine that needs k=10 to surface
the file costs far more context than one that nails it at k=1.

## Pipeline

A single `ArtifactStep` DAG (`pipeline.py`):

```
benchmark ─┬─────────────────────────────────────────────┐
           │  per engine:  build ─→ query ─→ judge ───────┤
           └─────────────────────────────────────────────┴─ score (results.json/.md)
```

- **benchmark** (`benchmark.py`) — mines *navigation moments* from `~/.claude`
  transcripts (a run of Grep/Glob/Read that ended when the agent read or edited a file;
  that file is the **gold**), then uses a headless agent to rewrite each into a clean
  natural-language query and drop non-navigation moments. Gold files are verified to
  still exist in the repo.
- **build / query** — each engine is a standalone adapter under `engines/` with its own
  isolated `uv` dependencies; the pipeline shells out to it. Indexes are cached
  independently of the benchmark.
- **judge** (`judge.py`) — a headless agent scores each engine's ranked snippets for
  whether they answer the need (full / partial / none) and at what rank. It never sees
  the gold, so `judge_hit@k` is independent of exact `recall@k`.
- **score** (`scoring.py`) — `recall@k`, `mrr`, `judge_hit@k`, `judge_full@k`, answer
  `tokens@k` (extracted uniformly so engines are comparable), plus index build
  time/size and query latency. Writes `results.json` and a `results.md` leaderboard.

## Engines

| engine | kind | index | deps |
|---|---|---|---|
| `ripgrep` | lexical regex over query keywords (baseline) | none | none |
| `bm25` | sparse ranked lexical over line-window chunks | build | `bm25s` |
| `dense` | local embedding + cosine (the recommended index) | build | `fastembed` |
| `vectorcode` | off-the-shelf ChromaDB code RAG | build | `vectorcode`, `chromadb` |
| `seagoat` | off-the-shelf local semantic search | build | `seagoat` |

`dense` is our reference implementation of the recommended index: line-window chunking,
a local ONNX embedding model via `fastembed` (default `BAAI/bge-small-en-v1.5`, set with
`--embed-model`), cosine ranking. No GPU, no cloud.

## Running it

```bash
MARIN_PREFIX=~/scratch/cse python -m experiments.code_search_eval.pipeline \
    --glob='-home-you-code-marin*' \
    --repo /home/you/code/marin \
    --engines ripgrep,bm25,dense,vectorcode \
    --agent-command 'claude -p' \
    --judge-agent-command 'claude -p'
```

Outputs land under `$MARIN_PREFIX/code_search_eval/`; the scorecard is
`score/dev/results.md`. Use any headless agent for `--agent-command` (`codex exec`
works). Pass a leading-dash glob with `=` (`--glob='-home-...'`) so argparse does not
read it as a flag.

Notes:
- The `dense` index build is the slow, one-time step (it embeds the whole repo on CPU);
  it is cached, so re-runs with a changed benchmark reuse it. For a quick pass, index a
  subtree or use a smaller `--embed-model`.
- Engine adapters run in isolated `uv` environments, so their heavy deps never touch the
  marin environment. They import the shared `common.py` over `PYTHONPATH`.
- Standalone scripts carry no pytest suite (repo policy); they are validated by running
  against the real repo.
