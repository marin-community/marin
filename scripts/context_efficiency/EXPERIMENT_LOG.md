# Context-Efficiency Analysis — Experiment Log

Tracking issue: weaver #405. Goal: quantify how much agent **token budget** could
be saved by "supermemory"-style systems (shared wiki/memory, RAG code search,
better docs) by analyzing our own `~/.claude` session transcripts.

Append-only. Newest entries at the bottom.

---

## 2026-07-08 — M0: Dataset recon

Mapped the `~/.claude/projects/*/*.jsonl` transcript schema and took a first
census. Numbers below are the ground truth the plan is built on.

**Schema.** One JSONL file per session. Record `type`s: `user`, `assistant`,
`attachment`, `ai-title`, `queue-operation`, `last-prompt` (+ `summary`/`system`
in some). The load-bearing fields:

- `assistant.message.usage` — real billed tokens per turn: `input_tokens`,
  `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`
  (+ `cache_creation.ephemeral_5m/1h` TTL split).
- `assistant.message.content[]` — blocks: `text`, `thinking`, `tool_use`
  (`name`, `input`, `id`).
- `user.message.content[]` — `tool_result` (`tool_use_id`, `content`) = the
  file/command output that fills context.
- Context fields on every record: `cwd`, `gitBranch`, `sessionId`, `timestamp`,
  `version`, `isSidechain` (sub-agent turns).

**Census (all 288 projects, 3,338 non-empty sessions, 178,407 assistant turns):**

| usage field (summed)          | tokens          |
|-------------------------------|-----------------|
| cache_read_input_tokens       | 28,185,227,612  |
| cache_creation_input_tokens   |    830,440,852  |
| output_tokens                 |    245,593,192  |
| input_tokens (uncached)       |     45,931,622  |

Marin project alone: 2,929 sessions; cache_read 22.2B, cache_creation 708M,
output 195M.

**Three findings that shape everything downstream:**

1. **The ~34× cache-read amplifier.** cache_read / cache_creation ≈ 28.2B / 830M
   ≈ 34. Every distinct context token is re-read ~34× on average, because each
   turn re-reads the whole accumulated window. => The cost of a context token ≈
   its size × (turns remaining in the session). Bloating context *early* is the
   expensive mistake; this is the multiplier that makes redundant reads hurt.

2. **Bash + Read = 94% of tool-result content** (est. tokens, chars/4 proxy):
   Bash 55.5%, Read 38.9%, Agent 2.8%, Edit 1.3%, Grep 0.9%. This is the
   *variable* context — the surface a wiki/RAG/memory system can shrink.
   Attributable message content totals ~94M tok (tool_result 52.5M, user_text
   24M, tool_use_input 13.8M, assistant text 3.8M, thinking 0.5M).

3. **The fixed prelude dominates cache_creation.** 830M cache_creation vs ~94M
   attributable body content => the bulk of created cache is the re-created
   *prelude*: system prompt + tool schemas + skill catalog + CLAUDE.md/AGENTS.md
   + memory index, paid once per session (more if the 5m/1h TTL expires mid-run).
   ~249K created tokens/session; a large chunk is fixed harness overhead.

**Tokenization.** No public exact Claude tokenizer; `tiktoken` absent,
`transformers` present. Use chars/4 as the content proxy and **calibrate it
against the ground-truth `usage` totals** to derive a correction factor.

Tooling will live in `scripts/context_efficiency/` (standalone uv scripts, no
pytest per repo convention for scripts).

Next: research brief (background agent) → plan artifact → codex review → build.

---

## 2026-07-08 — M0.5: Redundancy teaser (validates hypothesis, exposes a trap)

Ran a fast cross-session probe on marin sessions to check the premise before
building. It validated the core hypothesis **and** exposed a classification
trap that reshapes the tool design.

**Read redundancy.** 2,512 distinct paths; 16.7M Read tool_result tokens.
Reads-beyond-first-per-path = **73.2% of Read tokens are re-reads**. BUT only
**16.8% of that is same-content** — the rest are re-reads of files that *changed*
between reads (evolving code across weeks/branches). Implication: a naive
"cache the file bytes" wiki cleanly captures only the same-content slice; the
larger slice needs a *structural summary / repo-map* (API, signatures, layout)
that stays useful as the body changes. 708 paths were read in >1 session. Hot
re-read files: `controller.py` 350 reads / 80 sess, `lazy.py` 123 / 85,
`experiment/data.py` 97 / 81, `experiment/train.py` 79 / 66 — the wiki/summary
short-list writes itself.

**Bash discovery redundancy — the trap.** Naively, 99.9% of read-only-discovery
tokens are "redundant". But that number is dominated by **`git diff`: 17.0M
tokens, 7,810 calls across 1,736 sessions.** `git diff` is read-only yet its
answer is *volatile* (the working tree, different every call) — it is NOT
memory/wiki/RAG-addressable. Same for `git show`/`status`/`ls`. So
**read-only ≠ stable-answer.** The memory-addressable pool is specifically
*stable-answer discovery* — `grep`/`rg` (569K+75K tok, locate-the-code), `find`
(29K), `cat <known file>` — a fraction of the raw 19M. The tools must classify
Bash by **answer stability**, not just read-only, or they will wildly overcount.

Two consequences folded into the plan: (1) `read_redundancy` splits
same-content vs changed-content and treats them as different interventions;
(2) `exploration_redundancy` gates on a *stable-answer* command allowlist and
explicitly excludes volatile self-inspection (`git diff` et al.). The 17M-token
`git diff` habit is noted as a separate, non-memory optimization opportunity
(compact-diff surface / less diff re-inspection), out of scope for this study.

**Amplifier is heavy-tailed (cost-model correction).** Per-session
`cache_read/cache_creation`: **median 2.0×, mean 6.3×, p90 11.2×** — far below
the 34× aggregate. The aggregate is token-weighted and dominated by a few giant
sessions (turns/session median 8, mean 49, **max 5,983**). Lesson: never apply
a flat 34× to estimate savings — amortize **per session using each chunk's
actual turn index**; total savings then correctly concentrate in the long,
high-turn sessions where context bloat compounds. Also observed: `isSidechain`
is 0% here (sub-agent internal turns are stored/summarized elsewhere), and there
are ~14 record types beyond user/assistant (`system`, `file-history-snapshot`,
`pr-link`, `mode`, …) the parser ignores for token accounting.

---

## 2026-07-08 — M1: Research brief + plan published

Background research agent returned a dense cited brief (saved to
`RESEARCH_BRIEF.md`). The load-bearing finding: **prompt caching already serves
re-sent context at ~10% of list price**, so removing N already-cached tokens
saves ~`0.1·N`, not `N`. The prize is (a) *cross-session cold reads* (cache
misses) and (b) *output tokens spent re-deriving conclusions* (output is never
cached). This validates the `0.10·C·(T−t)` read term and forces the
cross-session vs within-session split in the cost model. Brief also supplies the
honest counter-evidence (Anthropic dropped RAG for staleness/security; RAG can
score below closed-book; memory poisoning; the 3–5× "reflection tax"; CLAUDE.md
>200 lines *reduces* adherence). OKF is markdown+YAML-frontmatter — near-identical
to the existing marin auto-memory format; low-risk as a format, not a token saver.

Plan written to `PLAN.md` and published as weaver artifact `plan`.

## 2026-07-08 — M1.5: Codex peer review → plan v2

`codex exec` reviewed the plan (read-only). Verdict **PROCEED-WITH-FIXES**, 12
prioritized issues (full text in `CODEX_REVIEW.md`). Adopted into the build:

1. **Multiple denominators.** Report savings against three: raw context tokens,
   base-price-input-equivalent, and full dollar-equivalent (incl. 5× output).
   Headline names one explicitly. (Was mixing raw M0 totals with price-weighted
   savings.)
2. **Validate the cost model against actuals.** Reconstruct modeled per-session
   cost from `billed(C,t,T)` and compare to real `cache_creation/read/input`;
   use the `ephemeral_5m/1h` split + timestamp gaps for TTL re-creation. This is
   also the calibration for chars/4. If modeled ≠ actual, the model is wrong.
3. **Don't assert "prelude dominates cache_creation" — reconcile per turn.** The
   residual (cache_creation − visible body) may be missed assistant/thinking/
   tool content or TTL re-creation, not just prelude. Per-turn reconciliation +
   inspect high-residual sessions before banking any docs/prelude pool.
4. **Turn index is subtle.** A tool_result's first *paid* turn is the next
   assistant request; assistant output becomes paid input only later; compaction
   censors lifetime. Use `first_in_prompt_turn`/`last_in_prompt_turn`, treat
   `summary`/compaction records as censoring points.
5. **Within-session rereads are discounted, not worthless.** Keep three pools —
   cross-session (memory), within-session hygiene, compaction-avoidance — rank
   all three but only bank cross-session as "shared memory."
6. **Warm-start check.** Body reads almost never share cache across sessions, but
   the fixed prelude might (concurrent/prewarmed). Use observed first-turn
   `cache_read/cache_creation` to measure how often sessions start warm.
7. **Redundancy key must be richer.** Key reads by (repo, branch/commit if
   available, path, content-hash); tier: byte-identical / structure-stable /
   changed. Human-label a stratified sample.
8. **Bash stability by output-hash, not command name.** Hash tool_result outputs;
   preserve normalized patterns/paths/commit-refs; `git show <ref>` is stable,
   `gh pr view #N` may not be.
9. **Ground hit-rates with a labeled audit** (200–500 high-weight redundancy
   events): replaceable? by which intervention? replacement token cost?
   verification still needed? stale risk? Derive hit-rate ranges + bootstrap CIs
   from the audit — not from priors. **This is the credibility crux.**
10. **Intervention masking** to avoid double-counting overlapping wiki/RAG/docs;
    union savings via priority allocation, replacement-entry cost once/session.
11. **Expand threats to validity** (sub-agent transcripts, survivorship, chars/4
    error by content type, unmeasured output re-derivation, cache concurrency
    scope, parser-missed record types, Claude Code version drift).
12. **Falsification test.** Top-K ablation: hand-classify the top 80% of modeled
    savings in the heaviest ~100 sessions; if manual replaceable ≪ modeled, scale
    the whole estimate down. Ship this as the trust anchor.

Codex "what's solid": using real `usage` as anchor, the heavy-tail correction,
the same/changed-content split, and the `git diff` volatility trap.

Next: build `parse_sessions.py` with these fixes baked in (M2).

---

## 2026-07-08 — M2: Parser + token accounting (cost model validated & corrected)

Built `parse_sessions.py` (→ `blocks.parquet` 260,784 rows, `turns.parquet`
174,668 rows, 3,251 sessions) and `token_accounting.py`. Two intended fixes from
the codex review paid off immediately by catching my own errors:

**Denominators (all 3, codex #1).** Headline uses base-price input-equivalents.
- raw distinct input: **857M** tokens
- **base-price input-equiv: 3,826M** ⟵ headline denominator
- full dollar-equiv (incl 5× output): 5,026M
Composition of the base-price budget: **cache_read = 72%** (0.1 × 27.65B),
cache_creation 27% (1.25 × 813M), input 1%. Re-reading context *is* the budget.

**Cost-model validation caught a 12× overcount.** Full-retention
`1.25C+0.10C(T−t)` implies each created token is read **410×**; observed is
**34×**. The model assumes every chunk survives to session end — false under
compaction + 1h-TTL eviction. Spearman(modeled, observed ratio) = **0.842** over
1,993 ≥5-turn sessions ⇒ the model *ranks* sessions right but its magnitude is
wrong. **Fix applied:** the uplift model prices a saved chunk with its session's
*observed* amplifier `A_S` (exported to `session_amplifier.parquet`), capped by
remaining turns — not with `(T−t)`.

**Calibration exposed missing thinking.** Anchoring chars→tokens on
`output_tokens` gives k≈**13.4**, which is impossible as a tokenization factor.
Cause: **thinking blocks are largely not persisted in the transcript** though the
model was billed for them (est visible generated 17.9M tok vs real output 240M).
So (a) we use k=1.0 (chars/4) with a 0.9–1.2 band, and (b) **output-token
re-derivation savings are under-observed and cannot be measured here** — logged
as a threat, not banked.

**Prelude residual — reported, not over-claimed (codex #3).** cache_creation 813M
vs calibrated visible body **91M** ⇒ **89% (722M) is prelude + TTL re-creation +
hidden thinking**, ~222K/session. BUT `eph_1h` = **99.6%** of writes and **85% of
sessions start warm** (first turn already cache_reads) ⇒ the fixed prelude is
written long-TTL and **already shared across most session starts by the harness**.
So the residual is dominated by growing-prefix re-creation, not a fresh
per-session prelude we could trim. We do **not** bank it as a docs/prelude pool.

**Addressable surface.** Visible body ≈ **94M** distinct tokens; tool_results 52M
(Bash 55.6% / Read 38.9% / Agent 2.8%), user_text 24M, tool_use 14M. Memory/RAG/
docs act here. This is the pool the redundancy analyzers carve up.

Next (M3): `read_redundancy.py` + `exploration_redundancy.py`, keyed by
content-hash with stability tiers.

---

## 2026-07-08 — M3: Redundancy pools (the hypothesis gets tempered)

Built `redundancy.py`. Two parser/classification bugs surfaced and were fixed
(the analyzers earning their keep): (a) `cd repo && rg foo` was keyed as `cd`
(40M of phantom "savings") — fixed `bash_shape` to skip `cd/export/timeout`
wrappers and compound separators; (b) `git diff <sha>` leaked into the
"addressable" pool via output-hash churn alone — added an allowlist of
stable-answer shapes (rg/grep/find/cat/git log/git show/gh view) so working-tree
self-inspection is excluded by name, not just churn.

Pricing uses each session's **observed amplifier** (M2), base-price input-equiv.

**Read redundancy (wiki/cache lever).** 14,531 Read results over 3,232 paths.
Cross-session redundant (a prior session already read the path): **9,082 reads
over 920 paths, 62.1M input-equiv gross.** Stability split is the story:
**byte-identical only 5% (3.4M)**; **changed-content 95% (58.8M).** Buckets:
source 50.4M, docs 5.9M, other 5.0M, config 0.8M. A raw byte cache banks only the
5%; the 95% is re-reading *evolved* files — a structural summary could help but is
staleness-prone (exactly the memory failure mode the research flagged). Hot paths:
`controller/service.py` (188 reads/49 sess), `execution/lazy.py` (99/84),
`experiment/data.py`, `config.py` — the wiki short-list. Within-session hygiene
re-reads: 2,217, worth 10.5M at the 0.1× cache-read rate only (not bankable as
shared memory).

**Exploration redundancy (RAG lever) — near zero.** After correct
classification, **stable-answer discovery that exactly repeats across sessions =
12 targets, ~60K input-equiv.** Agents almost never re-run the *identical*
stable-output command in another session. The real exploration redundancy is
*semantic* (different greps, same underlying fact), which exact-keying cannot
bank and a naive cache cannot serve — capturing it needs semantic RAG, whose
accuracy the literature disputes (retrieval < closed-book; Anthropic dropped it).
The large discovery costs are `git diff` (32M, volatile self-inspection — a
harness/compact-diff optimization, not memory), `echo`/`sed`/`uv run` (mutating).

**Takeaway shaping the report:** the token prize is NOT exact-repeat exploration.
It is (1) the small pure-cache win on hot unchanged files, (2) a *bounded,
staleness-limited* structural-summary win on hot evolving files, and (3) the
large but non-memory `git diff` self-inspection habit. The uplift model must bank
these conservatively and net replacement + staleness cost.

Next (M4): `uplift.py` — stability-grounded hit-rates, replacement cost,
intervention masking, and an LLM-assisted audit + top-K ablation (codex #9/#12).

---

## 2026-07-08 — M4: Uplift model → the counter-to-hype headline

Built `uplift.py`. Base-price budget = **3,827M input-equiv**. Three layers:

**1. Addressable ceiling = 7.84% of budget.** All tool-result + tool-use content
combined (Bash 3.45% + Read 2.58% + Edit 0.90% + Write 0.45% + Agent 0.36% + …)
is the *entire* surface memory/RAG/docs can touch. The other **~92% is prelude +
the cache_read carry of accumulated conversation** (thinking, prior turns, system
prompt) — untouchable by any memory system. This ceiling, not the hit-rate, is
the headline: even a perfect memory can't save more than ~8%, and most of that
8% (Edit/Write = agent's own output; non-redundant first reads) isn't
redundant.

**2. Net uplift (grounded hit-rates × pools, net replacement, masked):**
- shared wiki/memory: **0.11% / 0.30% / 0.53%** (cons/likely/opt) of budget
- RAG semantic search: **~0.001%** (exact-repeat pool is 60K; semantic
  generalization unmeasurable and literature-disputed)
- better docs/repo-map: 0.07% / 0.24% / 0.46% (same changed-file pool as
  wiki-summary, unioned not summed)
- combined non-double-counted: **0.11% / 0.30% / 0.53%**

**3. Robustness.** Sensitivity: even at an optimistic 50% changed-content
hit-rate the wiki lever is 0.71% of budget; at 0% it's 0.06% (the pure-cache
floor). Ablation: **top-100 of 3,251 sessions hold 59% of tool-result cost**
(top 1% hold 29%) — savings concentrate in a few giant sessions, so the
high-leverage lever is long-session hygiene, not memory.

**Honest headline: shared memory/wiki/RAG/docs save ~0.1–0.5% of the measurable
token budget; the ceiling is ~8%.** Vendor ">90%" claims are on conversational-
memory benchmarks, not a coding workflow dominated by (a) reading current code
you must read anyway and (b) the cache_read tax on long sessions.

**Two big caveats that keep the report honest (and pro-wiki):**
- **Output-side re-derivation is unmeasurable here** — thinking isn't persisted
  (M2), so tokens spent *reasoning through* what a prior session already figured
  out don't appear. The `controller.py`-read-in-78-sessions pattern means 78
  agents re-derived how the controller works; a memory could shortcut the
  *reasoning*, which is output, which we can't see. Real saving > measured.
- **Whole-session avoidance is out of frame.** Our unit is the existing session.
  If a shared memory prevents an entire redundant investigation (SWE-Effi: failed
  runs cost 4.5–13×), that saving is at the session level, not the block level.

Caveat on the stability split: `Read` with different offset/limit yields
different bytes for the *same unchanged file*, so byte-identical (5%) undercounts
the unchanged pool; the LLM audit calibrates the true changed-file hit-rate.

So the wiki's real case is **correctness + onboarding + cross-agent knowledge**,
not raw token savings. Recommendation framing follows from this.

---

## 2026-07-08 — M4.5: The prelude-carry finding (the twist)

Measured the cost of carrying the **fixed prelude** — the initial prompt prefix
(tool schemas + skill catalog + AGENTS.md/CLAUDE.md + the **MEMORY.md index** +
first user message), which is re-read on every turn via cache_read. Median
initial prefix = **27,762 tokens**; carrying it costs **566.9M input-equiv =
14.8% of the budget** (≈20% of all cache_read), 65% of it in the top-100 sessions.

The twist for the "supermemory" question: **an always-on shared wiki/memory lives
in exactly this prefix**, so every entry added is re-read every turn and taxed by
the amplifier. Growing an always-on memory to capture the ~0.3% redundant-read
saving would cost multiples of it. **The prelude carry (14.8%) is >2× the entire
addressable content ceiling for redundant work.** => The highest-leverage
"memory/docs" move is to **trim / lazy-load the prelude** (defer tool schemas —
ToolSearch already does this; keep AGENTS.md lean; lazy-load skills and memory
topic files — Claude Code already caps MEMORY.md at 200 lines with lazy topics),
NOT to add always-on content. A shared wiki must be **retrieval-gated**, never
concatenated into every prompt. The data validates Claude Code's existing lazy-
memory design and argues for pushing it further.

This is the report's central, non-obvious conclusion: *how* memory is integrated
dominates *whether* it exists.

Next (M5): fold LLM audit result; write the marin-style report; PR; close #405.
