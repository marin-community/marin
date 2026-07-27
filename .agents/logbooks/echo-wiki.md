---
topic: echo-wiki
description: Echo wiki, hybrid retrieval, and browser UI
author: user
---

# Echo Wiki: Task Logbook

## Scope

- Goal: Add searchable agent-authored wiki notes, improve Echo retrieval, and add a Vue UI.
- Primary metrics: Exact-term relevance, semantic-query relevance, API behavior, and UI build health.
- Constraints: Preserve the existing BGE passage embedding space; avoid large corpus transfers; deploy only through the existing Echo stack.
- Coordinating work item: Weaver issue #170.

## Current TL;DR

- Production baseline supplied by the user: `grafana` returns unrelated MoE comments at cosine distances 0.421–0.435.
- Root cause: vector-only ranking plus use of the generic document encoder for queries.
- Decision: BGE query encoding plus PostgreSQL full-text candidates fused with vector candidates through RRF.

## Entry Log

### 2026-07-27 17:45 UTC - Architecture and access baseline

- Hypothesis: Exact-term failures can be fixed without re-embedding by correcting query encoding and adding lexical candidates.
- Commit Hash: Uncommitted design checkpoint on `origin/main`.
- Commands: `weaver summary`; repository inspection with `rg`; Echo CLI semantic/literal probes; Cloud SQL and IAP access probes.
- Config: Production `echo-api`, `BAAI/bge-small-en-v1.5`, current 384-dimensional corpus embeddings.
- Result: FastEmbed provides `query_embed()` and `passage_embed()`. Production probes are blocked because `loom-vm@hai-gcp-models.iam.gserviceaccount.com` lacks Cloud SQL connect and IAP access.
- Interpretation: Use the user-provided `grafana` output as production baseline and local deterministic fixtures during implementation. No corpus re-embedding is justified yet.
- Next action: Implement the migration, hybrid query layer, wiki API, and dashboard, then validate locally and retry authorized deployment probes.

### 2026-07-27 18:02 UTC - Real PostgreSQL hybrid-ranking probe

- Hypothesis: Weighted lexical RRF will put exact terms above unrelated semantic candidates while retaining paraphrase retrieval.
- Commit Hash: Uncommitted implementation worktree.
- Command: Temporary `pgvector/pgvector:pg16` plus a Python 3.12 probe running migrations `m0001` through `m0004`, BGE passage/query encoding, inserts, and the production SQL statements.
- Config: Five synthetic 384-dimensional activity chunks, one wiki note, RRF `k=60`, lexical weight `2`, candidate limit `40`.
- Result:

  | Query | Top result | Score | Next result |
  | --- | --- | ---: | --- |
  | `grafana` | `Grafana dashboards` | 0.04918 | unrelated MoE, 0.01613 |
  | `where can I see training charts and alerts?` | `Training observability` | 0.01639 | `Grafana dashboards`, 0.01613 |
  | `ragged_all_to_all` | `Collective kernels` | 0.04918 | unrelated Zephyr, 0.01613 |
  | wiki `TPU vLLM` | `TPU serving` | top | n/a |

- Interpretation: The migration chain, generated full-text columns, pgvector queries, and RRF statements execute together. Exact lexical matches receive roughly 3x the semantic-only top score; a no-exact-match paraphrase remains semantically ranked.
- Next action: Tighten tests/docs, build the complete API image, run lint/type checks, then deploy if credentials allow.
