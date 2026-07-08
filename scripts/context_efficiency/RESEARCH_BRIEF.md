# Token-Budget Savings from "Supermemory"-Style Systems — Cited Research Brief

Prepared for weaver #405 (agent context-efficiency analysis). Vendor-self-reported
(VSR) vs. independent (IND) flagged throughout. Figures that could not be traced
to a primary source are marked **[UNVERIFIED]**.

## Key numbers (each with source)

- **Prompt cache pricing:** cache *read* ≈ **10%** of base input price; cache *write* =
  **1.25×** (5-min TTL) or **2×** (1-hour TTL). Default TTL **5 minutes**. Min cacheable
  prefix **4096 tokens** (Opus). (Anthropic, VSR) — claude.com/blog/prompt-caching
- **Prompt-caching worked examples:** 100K-cached book chat **−90% cost / −79% latency**;
  many-shot (10K) **−86% / −31%**; 10-turn chat w/ long system prompt **−53% / −75%**.
  (Anthropic, VSR)
- **mem0:** ">90% token cost" and "91% lower p95 latency" vs full-context; ~**7K tokens**
  vs **26,031** full-context per query (table implies ~73%, not >90%). (VSR) — arxiv 2504.19413
- **MemGPT:** **93.4%** vs **35.3%** baseline on Deep Memory Retrieval. — arxiv 2310.08560
- **supermemory:** LongMemEval recall@15 **95% at ~720 tokens** ("99.4% context reduction");
  81.6%-vs-Zep figure is inconsistent VSR. — supermemory.ai/research
- **Anthropic multi-agent:** single agents ~**4×**, multi-agent ~**15×** chat tokens; token
  usage explains **80%** of performance variance; subagents return **1–2K-token** summaries.
  (VSR) — anthropic.com/engineering/multi-agent-research-system
- **Context editing + memory tool:** **−84%** tokens on a 100-turn eval; context-editing
  alone **+29%** score, with memory **+39%**. (Anthropic, VSR) — claude.com/blog/context-management
- **Lost in the Middle:** answer mid-list of 20 docs scores **53.8% < 56.1%** closed-book
  baseline — retrieval can be *worse than nothing*. (IND) — arxiv 2307.03172
- **Grep vs vector (inline delivery):** grep **93.1%** vs vector **83.6%** (Opus); ranking
  inverts under file-based delivery. (IND) — arxiv 2605.15184v1
- **Agentic RAG cost:** **3.3×** input tokens, **1.9×** output, **1.5×** slower vs fixed RAG;
  53% cross-round evidence overlap. (IND) — arxiv 2601.07711v2
- **SWE-Effi "token snowball":** one scaffold **8.1M** input tokens vs **440K** for another;
  failed runs cost **4.5–13×** more than successes. (IND) — arxiv 2509.09853v2
- **ClawTrace/CostCraft:** transferable prune patches cut median session cost **32%**;
  flags redundant tool-call clusters (same tool, ≥80% arg similarity). (IND) — arxiv 2604.23853
- **Cursor semantic search:** **+12.5%** avg accuracy; time-to-first-query **4.03h → 21s**
  on large repos. (VSR) — cursor.com/blog/semsearch
- **Claude Code spend anchor:** ~**$13/dev/active day**, **$150–250/dev/month**. (VSR) —
  code.claude.com/docs/en/costs

## The caching correction (load-bearing for our uplift model)

Prompt caching is a **prefix match**: the rendered prompt (`tools`→`system`→`messages`)
is hashed to each `cache_control` breakpoint; a hit skips re-processing. Read ≈ 0.1×,
write 1.25×/2×, **output never discounted**, any byte change in the prefix invalidates.
The three `usage` fields decompose the prompt exactly: `cache_creation` (1.25×) +
`cache_read` (0.1×) + `input` (1×) = total prompt.

**Consequence:** in a running session, re-sent context is already served at ~10% of list
price. Removing N already-cached tokens saves ~`0.1·N`, **not** `N`. The prize is therefore
(1) context that never gets a cache hit — *fresh reads and cross-session cold starts* —
and (2) **output tokens spent re-deriving conclusions**. Anthropic runs cache-hit-rate as
an SLO ("declare SEVs if too low"). The circulated "92% Claude Code cache hit rate" is
**[UNVERIFIED]**.

## Memory systems
- **Claude Code CLAUDE.md + auto-memory**: concatenated every session, re-injected after
  `/compact`; `MEMORY.md` always-loaded (≤200 lines/25KB), topic files lazy-loaded. Cost:
  moderate CLAUDE.md + a few MCP servers can start a session at **20–30K tokens consumed**;
  files >200 lines "reduce adherence" (Anthropic docs) — more memory can lower quality.
- **MemGPT/Letta**: OS-style paging between in-window and external store; 93.4% vs 35.3%.
- **mem0**: write-time compression to structured facts; >90%/73% token cut; graph variant
  costs 2× tokens for ~2 pts (vendor-admitted diminishing return).
- **supermemory**: evolving-fact tracking w/ contradiction resolution + time-expiry;
  recall@15 95% at ~720 tok.
- **Generative Agents / Reflection**: recency(0.995/hr)·importance·relevance retrieval;
  reflection compresses observations into reusable insights so reasoning isn't re-derived.
- **Reflexion**: episodic verbal lessons re-fed on retry (HumanEval 91% vs 80%); IND
  "reflection tax" **3–5×** single-shot cost, round 3 net-negative.

## RAG vs agentic search
- **Anthropic dropped RAG** (Boris Cherny): "agentic search generally works better…
  doesn't have the same issues around security, privacy, staleness, and reliability." The
  mechanism is **freshness** — an index "returns a function the team renamed two weeks ago";
  "no embedding pipeline or centralized index to maintain." Admitted price: "at the cost of
  latency and tokens" — prompt caching makes that affordable.
- **Vendors that keep RAG**: Cursor (+12.5%), Windsurf. **Continue.dev deprecated** its
  embeddings provider; **Sourcegraph Cody abandoned embeddings-first for BM25/keyword**
  citing cost/privacy/scale past ~100K repos — independently echoing Anthropic.
- **Accuracy tradeoffs (IND)**: Lost-in-the-Middle (retrieval < closed-book); Distracting
  Effect (**6–11pp** drops; stronger retrievers → more convincing near-misses); grep>vector
  under inline delivery but harness-dependent; agentic RAG 3.3× tokens, often re-fetches
  same evidence. Milvus "40%+ savings" is **[UNVERIFIED vendor marketing]**.

## Docs / repo maps
- **Aider repo map**: tree-sitter symbol signatures + PageRank over the reference graph
  (×10 mentioned, ×50 open files, ×0.1 private), binary-searched to a **1,000-token** budget;
  recomputed live to avoid staleness. No published token delta.
- **llms.txt**: curated markdown index; weak adoption (**84/62,100** bot requests hit it;
  Google says no AI system uses it) — cautionary for any new convention.
- **AGENTS.md**: tool-agnostic agent README; mechanism (avoid rediscovery) is narrative.
  Closest primary quantification is Anthropic's: a "codebase-overview" skill avoids "spending
  tokens reading multiple files"; a pre-tool grep hook cut "tens of thousands of tokens to
  hundreds."

## Compaction / sub-agent offload
- **Compaction**: clears old tool outputs, then summarizes; API `compact_20260112` default
  trigger **150K** input tokens. Anthropic: "overly aggressive compaction can result in the
  loss of subtle but critical context." **Context editing** (`clear_tool_uses`) + memory tool
  cut tokens **84%** on a 100-turn eval.
- **Sub-agents** return 1–2K-token summaries from tens of thousands explored; cost ~15×
  chat tokens; Anthropic warns "most coding tasks involve fewer truly parallelizable tasks
  than research" — a caveat against fan-out for coding.

## OKF (Open Knowledge Format) — skeptical verdict
Google Cloud (2026-06-12): "formalizes the **LLM-wiki pattern**" — a directory of markdown
files with YAML frontmatter (only required field `type`; also `title/description/resource/
tags/timestamp`), cross-linked into a graph; optional `index.md`/`log.md`. It is essentially
**AGENTS.md/CLAUDE.md + a light metadata schema + a cross-link convention**; no runtime,
no SDK. Examples are data-analytics-centric (BigQuery tables/metrics), not code navigation;
improvements are **narrative, unmeasured**; v0.1, days old, zero adoption evidence; inherits
every memory-store failure mode (staleness, poisoning, drift) with no freshness/verification
mechanism. **Verdict: low-risk to adopt as a format, but not itself a source of measurable
token savings** — savings come from the wiki *behavior* (write-once, read-selectively),
obtainable from any markdown tree. The `timestamp`/`log.md` give a staleness hook raw
CLAUDE.md lacks. Note: it is near-identical to the existing marin auto-memory format
(`MEMORY.md` index + per-fact frontmatter files).

## Measurement precedent
- **ClawTrace/CostCraft**: per-session TraceCards; redundant tool-call clusters; 32% cut.
- **SWE-Effi**: token-budget effectiveness = AUC of resolve-rate under caps; "token snowball"
  and "expensive failures" (4.5–13×).
- **Cost-columned benchmarks**: HAL SWE-bench Verified Mini ($/agent, no caching),
  SWE-rebench ("Tokens per Problem"). Terminal-Bench reports pass-rate only.
- **Context rot (Chroma)**: accuracy degrades non-uniformly with length; a single distractor
  lowers accuracy; coherent haystacks can retrieve *worse* than shuffled — naive
  summarization isn't guaranteed to help.

## Counter-evidence (memory/RAG/compaction making agents worse)
- **Memory poisoning/persistence**: a hallucinated/malicious fact "can persist across
  sessions and later steer agent behavior"; STALE benchmark — best model **55.2%** at
  recognizing invalidated memory.
- **Compaction data loss**: Anthropic-acknowledged; reports of access-control rules and
  CLAUDE.md rules summarized away.
- **RAG distraction/staleness**: retrieval below closed-book; better retrievers → more
  harmful near-misses.
- **Multi-agent fragility (Cognition)**: parallel subagents make conflicting implicit
  decisions; favors single-thread + compression.
- **Behavioral bottleneck**: given a superior retrieval tool, agents ignored it **58%** of
  the time absent explicit prompting — availability ≠ use.
- **Adherence**: CLAUDE.md >200 lines reduces instruction adherence.

## Sources
See inline citations. Full list in the M1 experiment-log entry / final report bibliography.
Flagged unverified: supermemory 81.6%; "92% Claude Code cache hit rate"; Milvus "40%+";
mem0 ">90%" vs table-implied ~73%.
