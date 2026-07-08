# Plan: Quantifying the token-budget uplift of agent memory/RAG/doc systems

**Issue:** weaver #405. **Author:** analyze-context-efficiency session.
**Status:** draft for codex peer review.

## 1. Question

We run thousands of Claude Code agent sessions against this repo. Each session
re-derives a lot of what earlier sessions already knew: it re-reads the same
files, re-runs the same discovery commands, re-loads the same docs. We want a
defensible answer to: **how much token budget would we save by adding
"supermemory"-style infrastructure** — a shared wiki/memory, RAG code search,
better generated docs — and *which* of those is worth building first?

The deliverable is (a) a set of analysis tools over `~/.claude` transcripts and
(b) a report with per-intervention estimated uplift and a recommendation.

## 2. Data

`~/.claude/projects/*/*.jsonl`, one file per session. Census already taken (see
`EXPERIMENT_LOG.md` M0): **3,338 sessions, 178,407 assistant turns, 2.0 GB**.
Ground-truth billed tokens are in `assistant.message.usage`
(`input`/`cache_creation`/`cache_read`/`output`). Tool outputs that fill context
are `user.message.content[].tool_result`, joined to their `tool_use` by id.

Three census facts drive the design:

1. **~34× cache-read amplifier.** Summed `cache_read` (28.2B) ÷ `cache_creation`
   (830M) ≈ 34. Each distinct context token is re-read ~34× because every turn
   re-reads the whole window. **Cost of a context token ≈ its size × turns
   remaining.** Redundant loads early in a session are the expensive ones.
2. **Bash + Read = 94% of tool-result content** (55.5% / 38.9%). This is the
   variable surface memory/RAG/docs can shrink.
3. **Fixed prelude dominates cache_creation.** 830M created vs ~94M attributable
   body content ⇒ most created cache is the re-created prelude (system prompt,
   tool schemas, skill catalog, AGENTS.md/CLAUDE.md, memory index),
   ~249K tok/session.

## 3. Cost model (the counterfactual unit)

We convert "content tokens avoided" into "billed tokens saved" with an
amortization factor derived from the transcripts themselves, not assumed.

When a chunk of `C` content tokens enters context at turn `t` of a `T`-turn
session, its lifetime billing is approximately:

```
billed(C, t, T) = 1.25·C            # cache_creation write (5m TTL multiplier)
                + 0.10·C·(T − t)     # cache_read on every later turn
```

The read term dominates for early/large chunks. Savings are computed
**per-session with the chunk's actual turn index**, never by a flat global
factor — because the amplifier is **heavy-tailed**: the aggregate
`Σcache_read/Σcache_creation ≈ 34×` is token-weighted and dominated by a few
enormous sessions (max 5,983 turns), while the **median session amplifier is
2.0×** (mean 6.3, p90 11.2, median 8 turns). Applying 34× everywhere would
overcount by ~5×. Per-session amortization makes the total savings correctly
concentrate in long high-turn sessions where bloat compounds. Price multipliers
(1.25 write / 0.10 read / 1.0 base / ~5 output) are Anthropic's published
prompt-cache ratios; we parameterize them so the report can show sensitivity.

"Budget saved" for removing a redundant chunk = its `billed(C, t, T)`. This is
the honest counterfactual: you avoid the write *and* every downstream re-read.

**The caching correction (load-bearing).** Prompt caching already serves re-sent
context at ~10% of list price, so we must not claim base-price savings for
tokens that were being cache-read anyway. Two pools with very different value:

- **Cross-session cold reads (the prize).** Session N re-reads a file / re-runs a
  search that session <N already resolved. N's cache is cold (5m/1h TTL, separate
  session), so N pays *full* `cache_creation` for it. A shared wiki/memory/RAG
  lets N read a compact entry instead. Savings = `billed(C_full − C_entry, t, T)`
  in session N — real, full-price savings.
- **Within-session re-reads (low value).** Re-reading inside one session is
  already mostly `cache_read` at 0.1×; attacking it is context-hygiene/compaction,
  not a memory problem. We report it but do **not** bank it as memory uplift.

Because output is never cached, a distinct secondary prize is **output tokens
spent re-deriving conclusions** a memory could have stated — we surface this
qualitatively (hard to attribute precisely) rather than banking a number.

## 4. Tools (`scripts/context_efficiency/`)

Standalone `uv` scripts (repo convention: no pytest for scripts; validate on
real inputs). Pipeline: parse once to a compact table, then analyzers read that.

### 4.1 `parse_sessions.py` — normalizer
Stream every JSONL → two Parquet tables (via pandas/pyarrow, already in repo):

- **blocks**: one row per content block — `session_id, project, cwd, git_branch,
  version, ts, turn_idx, role, is_sidechain, model, block_type, tool_name,
  tool_input (raw json), target (normalized: file_path | command-shape |
  pattern), content_chars, est_tokens`.
- **turns**: one row per assistant turn — the four `usage` fields + turn_idx +
  session join key.

`est_tokens = chars / 4`, **calibrated**: fit a scalar so that
Σ(est body tokens + estimated prelude) matches Σ(real usage) per session; report
the residual. `target` normalization: absolute→repo-relative paths; Bash
commands reduced to a "shape" (first token + subcommand, args/paths stripped,
e.g. `gh pr view`, `git log`, `rg`, `uv run pytest`); Grep/Glob keep pattern.

### 4.2 `token_accounting.py` — where do the tokens go
From the tables: prelude vs body split; tool-result tokens by tool; per-session
and global amplifier distribution; calibration residual. Output: a stacked
breakdown + the amplifier histogram. Establishes the denominator everything else
is a fraction of.

### 4.3 `read_redundancy.py` — the wiki/cache lever
Group `Read` blocks by normalized `target`. Per path: read count, distinct
sessions, distinct git branches, est tokens, amortized billed tokens. A read is
**redundant** if the same path was already read (in the same session, or in any
prior session for the cross-session view). Split by content-stability: if we can
hash the returned content (we have the tool_result text), classify
same-content (pure wiki/cache win) vs changed (a summary/diff still helps but
less). Bucket targets: source vs docs (`*.md`, `AGENTS.md`, `OPS.md`) vs config
vs generated. Output: redundant-read tokens, top re-read files, savings if a
shared read-through cache / wiki summary served repeats.

### 4.4 `exploration_redundancy.py` — the RAG/memory lever
Normalize Bash to command-shapes and classify by **answer stability**, not just
read-only. The M0.5 probe proved this matters: read-only `git diff` alone is
17.0M tokens across 1,736 sessions but is *volatile* self-inspection of the
working tree — **not** memory-addressable. Three buckets:
- **stable-answer discovery** (memory/RAG-addressable): `rg`/`grep`, `find`,
  `cat <known file>`, `git log <path>`, a *specific* `gh pr/issue view #N` —
  "where is X / what is X" whose answer is stable across sessions.
- **volatile self-inspection** (NOT addressable): `git diff`, `git show`,
  `git status`, `ls` of a churning dir — the agent inspecting its own current work.
- **mutating/build** (out of scope): `uv run pytest`, `git commit`, edits.
Only the first bucket enters the RAG/memory savings pool. Metrics: repeated
stable-discovery tokens across sessions; recurring Grep/Glob patterns (same
pattern re-run in N sessions = a fact that should have been retrievable). We
also **report** the volatile `git diff` mass separately as a distinct
(non-memory) optimization opportunity, explicitly out of scope for uplift.

### 4.5 `doc_loading.py` — the docs / prelude lever
Track (a) reads/greps of documentation files and (b) the fixed prelude
re-creation cost per session (approx = cache_creation not attributable to body).
Quantify: how often each doc is re-read; how much prelude re-creation costs in
aggregate; what a leaner prelude or a cached/shared prelude would save. Ties to
the `ToolSearch` deferral already in the harness (tool schemas loaded on demand).

### 4.6 `uplift.py` — headline estimate
For each intervention, combine the relevant redundancy pool with a **hit-rate**
(fraction of that redundancy the intervention actually captures) at three
levels — conservative / likely / optimistic — and **net out the intervention's
own cost** (a memory/wiki entry must be read too; RAG returns tokens; stale
entries cost a re-derivation). Output: table of `% of total budget saved` per
intervention with ranges, plus a combined (non-double-counted) figure.

Interventions → levers:
- **Shared wiki/memory** → cross-session read redundancy (4.3) + repeated
  discovery (4.4).
- **RAG semantic code search** → replaces multi-command *locate-the-code*
  exploration (4.4) and failed greps; some overlap with wiki.
- **Better generated docs / repo map** → first-time exploration (4.4/4.5),
  less redundancy-driven, more onboarding-tax.
- **Prelude discipline** (lean AGENTS.md, deferred tool schemas) → 4.5.
- **(baseline) compaction / sub-agent offload** → intra-session amplifier (4.2);
  included as the "already have it" comparison point.

## 5. Threats to validity (call these out in the report)
- **chars/4 proxy** — calibrated but imperfect; report residual and repeat key
  numbers with a ±band.
- **Attribution** — usage is per-turn, not per-block; prelude is inferred as a
  residual, not measured directly.
- **"Redundant" ≠ "wasteful"** — re-reading a *changed* file is legitimate work;
  we separate same-content from changed-content and never count the latter as
  free savings.
- **Hit-rate is an assumption** — hence the 3-level range, not a point estimate.
- **Memory has its own costs** — read cost, staleness, retrieval distraction;
  netted out, and covered qualitatively from the research brief's counter-evidence.
- **Survivorship / self-analysis** — these are our own traces on one repo; the %s
  are indicative for *this* workflow, not universal.

## 6. Prior art (from the research brief; full cites in `RESEARCH_BRIEF.md`)
- **Prompt caching** (Anthropic): read 0.1×, write 1.25×/2×, output never cached,
  any-byte-change invalidation — the mechanism behind the caching correction in §3.
  Worked cases −53%/−86%/−90% cost.
- **Memory systems**: mem0 (>90%/~73% token cut via write-time fact compression),
  MemGPT/Letta (OS-style paging, 93.4% vs 35.3%), supermemory (recall@15 95% at
  ~720 tok), Reflexion (episodic lessons; but a **3–5× "reflection tax"**). Claude
  Code's own `MEMORY.md`+lazy topic files is the closest analog to what we'd build.
- **RAG vs agentic search**: Anthropic **deliberately dropped RAG** for agentic
  search citing **staleness/security/reliability**; Sourcegraph/Continue similarly
  retreated from embeddings-first; Cursor keeps it (+12.5%). IND accuracy caveats:
  Lost-in-the-Middle (retrieval < closed-book), distractor drops 6–11pp, agentic
  RAG 3.3× tokens. => RAG is **not** a free win; freshness is the core risk.
- **Docs/repo maps**: Aider's tree-sitter+PageRank repo map to a 1K-token budget;
  llms.txt (weak adoption); AGENTS.md. Anthropic quantifies a codebase-overview
  skill + pre-tool grep hook cutting "tens of thousands of tokens to hundreds."
- **Compaction/offload**: context-editing + memory tool −84% on a 100-turn eval;
  sub-agents return 1–2K-token summaries but cost ~15× (and Anthropic warns coding
  has "fewer truly parallelizable tasks than research").
- **OKF**: markdown+YAML-frontmatter wiki spec — **near-identical to the existing
  marin auto-memory format**; low-risk as a *format*, not itself a token saver.
- **Measurement precedent**: ClawTrace/CostCraft (redundant tool-call clusters,
  32% cut), SWE-Effi (token-snowball, expensive-failure 4.5–13×).
- **Counter-evidence to bank honestly**: memory poisoning/staleness (STALE: 55.2%
  at spotting invalidated memory), compaction losing critical context, CLAUDE.md
  >200 lines *reducing adherence*, agents ignoring a better tool 58% of the time.

## 7. Deliverables & milestones
1. `parse_sessions.py` + calibrated tables. (M1)
2. `token_accounting.py` — the denominator + amplifier. (M2)
3. `read_redundancy.py`, `exploration_redundancy.py`, `doc_loading.py`. (M3)
4. `uplift.py` — per-intervention ranges. (M4)
5. Marin-style report with recommendation + figures; PR; close #405. (M5)

Experiment log updated at each milestone; weaver status kept current.

## 8. Revisions after codex peer review (v2 — supersedes conflicting text above)

Codex reviewed v1 (`CODEX_REVIEW.md`, verdict PROCEED-WITH-FIXES). Adopted:

- **Three denominators, one named headline.** Report every saving against (a) raw
  context tokens, (b) base-price-input-equivalent, (c) full dollar-equivalent
  (incl. 5× output). Headline uses (b), the price-weighted input-equivalent.
- **Cost model is validated, not assumed.** `token_accounting.py` reconstructs
  modeled per-session cost from `billed(C,t,T)` and compares to actual
  `cache_creation/read/input`; uses the `ephemeral_5m/1h` split + inter-turn
  timestamp gaps to model TTL re-creation. The chars/4 factor is fit here. A
  large modeled-vs-actual gap invalidates the model — reported, not hidden.
- **Prelude claim is earned by reconciliation.** Do per-turn reconciliation
  before attributing any cache_creation residual to "fixed prelude"; inspect
  high-residual sessions. Treat the residual as *unexplained* until shown.
- **Turn lifetime, not record index.** A tool_result's first *paid* turn is the
  next assistant request; assistant output is paid input only on later turns;
  `summary`/compaction records **censor** a chunk's lifetime. Compute
  `first_in_prompt_turn`/`last_in_prompt_turn` and cap `(T−t)` at the censor.
- **Three savings pools, one bankable.** cross-session (memory — bankable),
  within-session hygiene, compaction-avoidance. Rank all three; only cross-session
  counts as "shared memory" uplift.
- **Warm-start measurement.** Use first-turn `cache_read/cache_creation` per
  session to measure how often sessions start warm (prelude cache sharing) vs cold.
- **Richer redundancy key + tiers.** Key reads by (repo, branch/commit, path,
  content-hash); tier byte-identical / structure-stable / changed. Bash stability
  judged by **output-hash across sessions**, not command name (a `git show <ref>`
  is stable; a `gh pr view #N` may not be); preserve normalized args/patterns.
- **Hit-rates are grounded, not guessed.** `uplift.py` derives hit-rate ranges +
  bootstrap CIs from a **labeled audit of 200–500 high-weight redundancy events**
  (replaceable? by which intervention? replacement cost? verification needed?
  stale risk?). The audit — not priors — sets the ranges.
- **Intervention masking.** Every redundancy event gets an intervention mask;
  combined savings use priority allocation (union, not sum) with the
  replacement-entry cost charged once per session.
- **Falsification test (trust anchor).** Top-K ablation: hand/LLM-classify the top
  80% of modeled savings in the heaviest ~100 sessions; if manual-replaceable ≪
  modeled, scale the entire headline down by the observed ratio.
- **Expanded threats:** sub-agent/sidechain transcripts not captured;
  retained-transcript survivorship; chars/4 error by content type; unmeasured
  output re-derivation; cache concurrency/scope; parser-missed record types;
  Claude Code version drift; repo state drift over time.
