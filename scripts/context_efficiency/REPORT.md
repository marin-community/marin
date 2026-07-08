# How much token budget could agent "supermemory" actually save? A measurement over 3,251 Claude Code sessions

## TL;DR

We parsed 3,251 Claude Code session transcripts (174,668 assistant turns, ~2 GB
from `~/.claude`, mostly work on the Marin monorepo) to measure how much token
budget a shared wiki/memory, RAG code search, or better docs would have saved.
The answer is smaller than the vendor pitch, and the reason is structural.

- **The total budget is dominated by carrying context, not by fetching it.**
  Under Anthropic prompt-cache pricing, `cache_read` is **72%** of the base-price
  input budget. Each distinct context token is re-read **34×** on average
  (aggregate), because every turn re-reads the whole window.
- **The surface a memory system can touch is small.** All tool-result plus
  tool-call content — everything a wiki/RAG/docs could shrink — is **7.8%** of the
  budget. The other ~92% is the prelude and the re-read conversation.
- **Net uplift from a shared wiki/memory is ~0.2–0.5% of the budget** (likely
  0.28%), grounded in an audit of the hottest re-read files. **RAG on repeated
  code search saves ~0%** here: agents almost never re-run the identical
  stable-output search across sessions. Better docs deliver the same small pool
  as the wiki summary, not an additional one.
- **The largest single memory-shaped cost is the always-on prelude.** The fixed
  prompt prefix — tool schemas, skill catalog, `AGENTS.md`, and the `MEMORY.md`
  index — is ~28K tokens and is re-read every turn, costing **14.8%** of the
  budget. An always-on wiki lives in exactly this prefix, so growing it costs
  multiples of what lazy lookups save. **How memory is integrated matters more
  than whether it exists.**
- **Recommendation:** build a retrieval-gated shared wiki for correctness and
  onboarding, not for token savings; do not concatenate it into every prompt.
  The real token levers are trimming/lazy-loading the prelude and managing long
  sessions (compaction, sub-agent offload) — 59% of the addressable cost sits in
  the heaviest 100 of 3,251 sessions.

Tools, data schema, and the full result JSONs are in
[`scripts/context_efficiency/`](.); the method was peer-reviewed by `codex`
before implementation ([`CODEX_REVIEW.md`](CODEX_REVIEW.md)) and the literature
backing is in [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md).

## Setup

Each Claude Code session is one JSONL transcript under `~/.claude/projects/`.
Ground-truth billed tokens are in `assistant.message.usage`: `input_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`. Tool
outputs that fill context are `tool_result` blocks in user records, joined to
their `tool_use` by id. We normalized every transcript into two Parquet tables
(one row per content block, one per assistant turn) with
[`parse_sessions.py`](parse_sessions.py), then ran the analyzers over them.

Corpus: 3,251 non-empty sessions across 288 project directories, 174,668
assistant turns, spanning roughly three months. Most sessions are agent work on
the Marin monorepo (Iris, Zephyr, Marin, Levanter) plus the `loom`/`weaver`
tooling. These are our own traces on one codebase; the percentages are
indicative for this workflow, not universal.

### Pricing and the cost unit

We express budget and savings in **base-price input-equivalents** using
Anthropic's published prompt-cache multipliers: cache write 1.25×, cache read
0.10×, uncached input 1.0×, output 5.0× (output reported separately). A byte
change anywhere in the cached prefix invalidates it, and output is never cached.

The counterfactual cost of loading a content chunk of `C` tokens is its creation
plus every downstream re-read: `1.25·C + 0.10·C·(reads)`. We do **not** assume a
chunk survives to the end of the session. The naive full-retention version
(`reads = turns − t`) overcounts the observed re-read rate by ~12× (it implies
each token is read 410× vs the observed 34×), because it ignores compaction and
1-hour cache-TTL eviction. It does rank sessions correctly (Spearman 0.84 between
modeled and observed amplification over the 1,993 sessions with ≥5 turns), so we
keep the per-session shape but **price each saved chunk with its own session's
observed amplifier** `A_S = cache_read/cache_creation`, capped by turns
remaining. Full derivation and validation output:
[`token_accounting.py`](token_accounting.py).

## Result 1: the budget is carry, not fetch

| Component (all sessions) | Raw tokens | Base-price input-equiv | Share |
|---|---:|---:|---:|
| cache_read | 27.65B | 2,765M | 72% |
| cache_creation | 813M | 1,016M | 27% |
| uncached input | 45M | 45M | 1% |
| **Base-price budget** | | **3,827M** | 100% |
| output (separate) | 240M | 1,200M | — |

The amplifier — `cache_read / cache_creation` — is **34× in aggregate** but
**2.0× at the median session** (mean 6.3, p90 11.2). The aggregate is
token-weighted and dominated by a few very long sessions (turns per session:
median 8, mean 49, max 5,983). A token loaded early in a long session is re-read
on every later turn; a token loaded in a median 8-turn session is not. Savings
therefore concentrate in long sessions, and any estimate that applies a flat 34×
everywhere is wrong by ~5×.

## Result 2: the addressable ceiling is 7.8%

A memory/RAG/docs system can only act on content the agent fetches: tool results
and tool calls. Priced with the cost model above, that entire surface is **300M
input-equivalents, 7.8% of the budget.**

| Tool | Cost (input-equiv) | Share of budget |
|---|---:|---:|
| Bash | 132M | 3.45% |
| Read | 99M | 2.58% |
| Edit | 34M | 0.90% |
| Write | 17M | 0.45% |
| Agent (sub-agent returns) | 14M | 0.36% |
| Grep + Glob | 1.3M | 0.03% |

Edit and Write are the agent's own output, not something memory can supply. The
memory-addressable part is Bash discovery plus Read, ~6% of the budget — and most
of that is not redundant across sessions (see Result 3). The 7.8% ceiling, not
the hit-rate, is the binding constraint: no memory system, however good, saves
more than this.

## Result 3: what is actually redundant

**File reads (the wiki/cache lever).** Of 14,531 Read results over 3,232 paths,
9,082 reads (920 paths) are cross-session redundant — a prior session already
read that path. Priced by session amplifier, that pool is **62.1M input-equiv
gross**. The stability split is the story:

| Cross-session redundant reads | Input-equiv | Share |
|---|---:|---:|
| byte-identical (pure cache win) | 3.4M | 5% |
| changed content (needs a summary) | 58.8M | 95% |

Only 5% is a file read again with identical bytes. The other 95% is re-reading a
file that changed between reads. A byte cache banks the 5%; the 95% needs a
structural summary that survives edits, and re-reading an evolved file is often
legitimate. (A caveat inflates the "changed" share: `Read` with different
offset/limit returns different bytes for the same unchanged file, so
byte-identical undercounts the unchanged pool. The audit in Result 4 corrects for
this by looking at files directly.) Hot re-read paths:
`controller/service.py` (188 reads / 49 sessions),
`controller/controller.py` (332 / 78), `execution/lazy.py` (99 / 84),
`experiment/data.py` (85 / 80).

**Code search (the RAG lever).** After classifying Bash/Grep/Glob by
answer-stability — an allowlist of stable-answer shapes (`rg`, `grep`, `find`,
`cat`, `git log`, `git show`, `gh … view`), excluding volatile self-inspection —
the exact-repeat stable-output discovery pool across sessions is **97K
input-equiv, essentially zero**. Agents rarely re-run the identical search with
the same output in a different session. The large discovery costs are volatile:
`git diff` alone is **32M input-equiv** (7,924 calls, 1,736 sessions), inspecting
the current working tree — not a fact any wiki can serve. Real code-search
redundancy is semantic (differently phrased searches for the same fact), which an
exact key cannot bank and a cache cannot serve; capturing it needs semantic RAG,
whose accuracy the literature disputes (retrieval below closed-book baseline in
*Lost in the Middle*; Anthropic dropped RAG from Claude Code citing staleness).
Detail: [`redundancy.py`](redundancy.py).

## Result 4: the audit grounds the hit-rate

We audited the 15 hottest re-read files — opening the code, checking size and
`git log` churn — to estimate what fraction of re-reads a maintained wiki entry
could actually replace. The decisive signal is reads-per-session: files read
~once per session are cross-session orientation reads a summary can replace
(`lazy.py` 1.18, `data.py` 1.06); files read 3–5× per session are within-session
edit-tracing (`controller.py` 4.3, `service.py` 3.8) that a summary cannot
replace — you cannot safely mutate `_control_tick` from a summary, so the read
still happens. Token-weighting pulls the replaceable fraction to **~0.17**,
because the token budget is dominated by the large, volatile, edit-traced files
whose fraction is ~0.10.

Two findings from the audit matter for the recommendation:

- **Staleness is already realized.** `execution/executor.py` was read 76 times
  but no longer exists — the layer was refactored into `lazy.py`/`step_runner.py`.
  A wiki entry an agent half-trusts here misleads rather than saves.
- **The best-documented files are still re-read.** `lazy.py` and `data.py` carry
  module docstrings that already function as wiki entries, yet are read once per
  session — because the summary lives *inside* the file. A wiki's value is
  surfacing that orientation without opening the file.

## Result 5: net uplift

Combining each pool with audit-grounded hit-rate bands (byte-identical 0.60/0.80/
0.95; changed-content 0.10/0.17/0.30 from the audit; RAG 0.20/0.45/0.70), netting
a 15% replacement-read cost, and unioning overlapping levers rather than summing
them ([`uplift.py`](uplift.py)):

| Intervention | Conservative | Likely | Optimistic | Primary value |
|---|---:|---:|---:|---|
| Shared wiki / memory | 0.18% | 0.28% | 0.46% | correctness/onboarding |
| RAG semantic search | 0.00% | 0.001% | 0.002% | (not for tokens) |
| Better docs / repo-map | 0.13% | 0.22% | 0.39% | same pool as wiki summary |
| Combined (non-double-counted) | 0.18% | 0.28% | 0.46% | — |

The headline is robust to the one swing variable. Even at a 50% changed-content
hit-rate the wiki lever is 0.71% of the budget; at 0% it is 0.06% (the pure-cache
floor). Better docs and the wiki summary address the same evolving-file pool, so
they are unioned, not added.

## Result 6: the prelude is the memory-shaped cost that dominates

The fixed prompt prefix — tool schemas, the skill catalog, `AGENTS.md`/`CLAUDE.md`,
and the `MEMORY.md` index — has a median size of **27,762 tokens** and is re-read
on every turn. Carrying it costs **567M input-equiv, 14.8% of the budget** (about
20% of all `cache_read`), and 65% of that sits in the heaviest 100 sessions.

An always-on shared wiki lives in this prefix. Every entry added is re-read every
turn and taxed by the amplifier, so growing an always-on memory to capture the
~0.3% redundant-read saving would cost several times that. The prelude carry
(14.8%) is nearly twice the entire addressable-content ceiling (7.8%). The lever
here is to **trim and lazy-load the prefix**, not grow it: defer tool schemas
(the harness already does this via on-demand tool search), keep `AGENTS.md` lean,
and lazy-load skills and memory topic files (Claude Code already caps `MEMORY.md`
at 200 lines with lazy topic files). A shared wiki must be retrieval-gated for the
same reason — concatenating it into every prompt would add to this 14.8%, not
subtract from it.

## Recommendations and anticipated uplift

1. **Trim and lazy-load the prelude — the biggest token lever.** The prelude
   carry is 14.8% of the budget. A leaner always-on prefix cuts it proportionally:
   removing ~30% of the 28K-token prefix is worth roughly 4% of the budget, an
   order of magnitude more than lazy wiki lookups return. Concretely: keep growing
   the memory index off the always-loaded path (retrieve entries on demand), and
   audit which tool schemas and skill descriptions must be resident.

2. **Manage long sessions — where the tokens are.** 59% of addressable cost and
   65% of prelude carry are in the top 100 of 3,251 sessions. Aggressive
   compaction and delegating exploration to sub-agents that return short summaries
   attack this concentrated cost directly. Anthropic reports 84% token reduction
   on a 100-turn eval from context-editing plus a memory tool; the concentration
   here says the same lever applies.

3. **Build a retrieval-gated shared wiki (OKF-style) for correctness, not
   tokens.** Anticipated token uplift is 0.2–0.5%. The real payoff is avoiding
   stale re-derivation (the `executor.py` case) and giving cold agents the right
   orientation — which our measurement cannot price, because thinking tokens are
   not persisted in the transcript and whole-session avoidance is out of frame.
   Constraints from the data: it must be **lazy** (never in the always-on prompt),
   **fresh** (staleness kills it), and **scoped** to stable orientation facts
   (module purpose, key signatures, invariants, "where things live"), not volatile
   internals. The Open Knowledge Format — markdown files with YAML frontmatter and
   cross-links — is a reasonable schema and is near-identical to the existing
   `MEMORY.md` format; adopting it as a format is low-risk, but the format is not
   the source of savings.

4. **Do not adopt RAG semantic code search for token savings.** Exact-repeat
   exploration is ~0% of the budget, and the semantic case is unproven and
   literature-disputed. If adopted, justify it on retrieval quality, not budget.

5. **Separately, attack `git diff` self-inspection.** It is the single largest
   discovery cost (32M input-equiv) and is volatile, so it is a harness concern
   (a compact-diff surface, or less frequent re-diffing), not a memory one. It is
   out of scope for this study but larger than the memory lever.

## Threats to validity

- **chars/4 token proxy.** Content size is estimated as characters/4 (k=1.0, band
  0.9–1.2). Budget totals use ground-truth `usage`; only content attribution uses
  the proxy.
- **Thinking is not persisted.** Anchoring the proxy on `output_tokens` implies
  k≈13, which is not a tokenization factor — thinking blocks are largely absent
  from the transcript though the model was billed for them. This means output-side
  re-derivation savings (an agent re-reasoning through what a prior session solved)
  are under-observed. The measured memory uplift is a lower bound on total value.
- **Whole-session avoidance is out of frame.** The unit is the existing session.
  A memory that prevents an entire redundant investigation saves at the session
  level, which a block-level analysis does not capture (SWE-Effi reports failed
  runs cost 4.5–13× successes).
- **Sub-agent transcripts are separate files** (~3,700 turns) not folded into the
  main session unit; `isSidechain` is 0% in the top-level transcripts.
- **Redundancy definition.** "Cross-session redundant" = a prior session read the
  same normalized path. `Read` offset/limit makes byte-identical undercount the
  unchanged pool; the audit corrects the hit-rate for this.
- **Survivorship and scope.** These are our own retained transcripts on one
  monorepo. The percentages describe this workflow.

## Reproduction

```bash
cd scripts/context_efficiency
uv run parse_sessions.py            # ~/.claude/projects/*/*.jsonl -> _data/{blocks,turns}.parquet
uv run token_accounting.py          # denominators, cost-model validation, prelude residual
uv run redundancy.py                # read + exploration pools, stability tiers
uv run uplift.py                    # per-intervention net uplift, ablation, sensitivity
```

Result JSONs land in `_data/`. The milestone-by-milestone record, including the
two probes that caught classification bugs and the cost-model correction, is in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
