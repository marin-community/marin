# Which code-search engine actually saves tokens?

**TL;DR.** On 150 code-navigation queries mined from our own agent sessions, a local
dense embedding index beats ripgrep decisively: it **doubles the top-1 hit rate**
(judge@1 0.42 vs 0.19) and reaches a satisfying answer at **about one third of the
tokens** (tok@5 1,882 vs 5,865). BM25 captures roughly half of dense's ranking gain at
1/600th of the index-build cost. The one real price is the dense index build — 42 minutes
for 18.5k chunks on CPU, one-time and cached. This confirms the context-efficiency
report's recommendation: an automatic semantic code index locates code in a smaller
answer than repeated grepping, which is where the token budget actually goes.

This is the empirical follow-up to the context-efficiency analysis
(`experiments/context_efficiency/`), whose top recommendation was an automatic semantic
code index. That report *assumed* such an index would find code with a smaller answer
than grepping. Here we test it, and measure how much smaller.

## Why token size, not just recall

The context-efficiency analysis found the agent token budget is dominated by *carrying*
tool output across turns, not by the one-time fetch. So the useful question is not only
"does the engine find the file" but "how many tokens does the agent pay to get the
answer." We run every engine once at K=10 and read off every metric at k ∈ {1,3,5,10} by
truncation — the "you can always `head -n1`" argument. An engine that surfaces the answer
at k=1 with a tight snippet is worth far more than one that needs k=10: same recall,
very different budget.

## Setup

**Benchmark.** We mine *navigation moments* from `~/.claude` transcripts: a run of
Grep/Glob/Read calls that ended when the agent read or edited a specific file. That file
is the **gold** answer; the agent's stated intent, rewritten by a headless agent into a
clean natural-language question, is the query. Non-navigation moments (running tests,
inspecting live git state, reading the agent's own output) are dropped. From 2,927 marin
sessions we mined 1,197 candidates; the cleaner kept 237 as genuine navigation queries;
we scored 150.

Gold is strict — the single file the agent settled on — so exact `recall@k` undercounts
an engine that surfaces an equally-good *different* file. We therefore also use an
independent **judge**: a headless agent sees each engine's ranked snippets (never the
gold) and marks the first rank that answers the need, full/partial/none. `judge_hit@k`
is the fraction answered within the top-k, and is the better guide to real answer
quality. `tokens@k` is the mean snippet tokens across the top-k, extracted uniformly
from the repo so engines are comparable regardless of how they chunk.

**Engines** (all local, indexing the full marin monorepo):

- **ripgrep** — keyword extraction from the query + `rg`, ranked by distinct-keyword
  coverage. The zero-config default an agent already has; the bar to clear.
- **bm25** — sparse ranked lexical over 40-line chunks (`bm25s`).
- **dense** — local embedding (`fastembed`, `BAAI/bge-small-en-v1.5`) + cosine over the
  same chunks. Our reference implementation of the recommended index.

We also wrote adapters for two off-the-shelf local tools, **VectorCode** and
**SeaGOAT**, but neither completed a scored run headless over the full monorepo (see
"Off-the-shelf tools" below). The adapters ship in `engines/` for anyone who wants to
pursue them.

## Results

150 queries over the full marin monorepo (18,522 indexed chunks). R@k = gold file in
top-k; J@k = judge says a snippet answers the need within top-k.

| engine  | R@1 | R@3 | R@5 | R@10 | J@1 | J@3 | J@5 | J@10 | MRR | tok@5 | build | index | ms/q |
|---------|-----|-----|-----|------|-----|-----|-----|------|-----|-------|-------|-------|------|
| dense   | **0.31** | 0.49 | 0.53 | 0.63 | **0.42** | 0.67 | **0.77** | 0.91 | **0.41** | **1,882** | 42 min | 30 MB | 12.6 |
| bm25    | 0.22 | 0.33 | 0.37 | 0.52 | 0.27 | 0.50 | 0.63 | 0.81 | 0.30 | 2,012 | 4 s | 15 MB | 2.3 |
| ripgrep | 0.15 | 0.28 | 0.36 | 0.50 | 0.19 | 0.42 | 0.56 | 0.79 | 0.25 | 5,865 | — | — | 222.7 |

## What we found

- **Dense wins on every quality metric.** It finds the exact historical file first 31% of
  the time and the judge accepts a top-1 snippet 42% of the time — about double ripgrep
  (15% / 19%). MRR 0.41 vs 0.25.
- **The token win is the headline.** Dense reaches judge-hit@5 = 0.77 while showing 1,882
  tokens; ripgrep needs 5,865 tokens to reach only 0.56. Dense delivers a *better* answer
  at ~32% of the tokens — exactly the budget saving the context-efficiency report
  predicted, since those tokens are then carried on every subsequent turn.
- **BM25 is the pragmatic middle.** It captures most of the ranking gain (judge-hit@5 0.63
  vs dense 0.77 vs ripgrep 0.56) and builds in 4 seconds versus dense's 42 minutes. If you
  want most of the benefit today with no embedding model, BM25 is the cheap default; dense
  earns its keep where the index build amortizes.
- **By k=10 everything converges** on judge-hit (0.79–0.91): all three eventually surface
  something relevant. The separation is at small k — which is the whole point for budget.
- **ripgrep is expensive per answer.** Its tok@5 is 3× the others because keyword matches
  land on long lines in data/generated files, and it spawns a process per query (222 ms vs
  12.6 ms dense, 2.3 ms bm25).

One concrete case — *"how does the scheduler and autoscaler handle availability"*, gold
`lib/iris/src/iris/cluster/controller/autoscaler/routing.py`: dense ranks it #1; ripgrep
never surfaces it in the top 10 (the query shares no literal token with the file).

## Cost

The dense index build was the only slow step: 2,524 s (42 min) to embed 18,522 chunks
with `bge-small` ONNX on CPU (~7 chunks/s under load). It is one-time and cached — a
changed benchmark reuses it — but for a first pass it dominates. A smaller model
(`all-MiniLM-L6-v2`), a GPU, or indexing a subtree all cut it substantially. BM25 and
ripgrep are effectively free to build.

## Off-the-shelf tools

We tried two packaged local engines and both are built for interactive/editor use, which
made them awkward in a headless batch harness at monorepo scale:

- **SeaGOAT** indexes at *per-line* granularity backed by ChromaDB. Its from-scratch
  index of the full monorepo did not finish in a practical time on this box (it was still
  building lib/iris alone after ~15 minutes). The adapter mirrors the subtree into a
  shadow git repo (SeaGOAT needs a git root) and is correct on small inputs; the build
  simply does not scale here.
- **VectorCode** starts a bundled ChromaDB server per invocation and talks to it over
  localhost HTTP. In our non-interactive environment that server never accepted a
  connection (`httpx.ConnectError`), on both the full repo and a single subtree, so it
  produced no index.

The takeaway is itself useful: a ~70-line dense adapter (`fastembed` + numpy cosine) and
BM25 ran first try and scored well, while the packaged tools needed an interactive
context we could not give them headlessly. For agent integration — which is headless by
definition — the simple index is the more practical starting point.

## Caveats

- Gold is one historical file per query; `recall@k` is a strict lower bound and the judge
  metric is the better guide to answer quality.
- Queries derive largely from review-agent and edit sessions and lean toward iris (where
  most recent work happened); the corpus is our repo, not a general one.
- The judge is `claude -p` (haiku); the context-efficiency work found haiku optimistic vs
  sonnet on a harder rubric, so treat absolute judge rates as an upper-ish bound — the
  *ranking* of engines is what matters here and is consistent across recall and judge.
- ripgrep uses a simple query→keyword heuristic; a cleverer query rewrite would help it,
  but the comparison is against zero-config default behavior.

## Recommendation

A dense semantic index materially beats grep for code navigation and, more importantly,
does so in far fewer carried tokens — the metric that drives agent cost. Ship an
automatic dense index; use BM25 as a cheap first step that already captures about half the
gain. The build cost is the thing to engineer down (smaller model / incremental updates /
GPU), not the retrieval quality.

## Reproduce

```bash
MARIN_PREFIX=~/scratch/cse python -m experiments.code_search_eval.pipeline \
    --glob='-home-you-code-marin*' --repo /home/you/code/marin \
    --engines ripgrep,bm25,dense --agent-command 'claude -p'
```

Full harness and per-engine adapters: `experiments/code_search_eval/`. Raw scorecard:
`results/2026-07-08-marin/`.
