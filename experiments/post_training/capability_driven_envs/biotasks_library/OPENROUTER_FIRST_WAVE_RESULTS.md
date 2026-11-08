# OpenRouter first-wave BioTasks diagnostic (2026-11-06)

**Collection complete; IRT interpretation not ready.** Twelve provider-pinned
models each attempted 247 provisional v2 TaskSpecs (2 corrected references,
245 only confirmed by a GLM reference-auditor). The 58 disputed references were
excluded. Frozen bare-JSON grading, identical OpenAI chat/ShellSim tools and
limits, and one original attempt per model/item were used. Only upstream 429
infrastructure failures were selectively retried at four workers; both the
original attempts and the immutable retry files remain available. The bank is
not independently validated beyond the two deterministic fixes.

Private immutable snapshot:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/combined/v2/calibration/openrouter-first-wave-v1/
    attempts/<model-slug>/results.jsonl
    attempts/<model-slug>/rate-limit-recovery1.jsonl  # where needed
    attempts/<model-slug>/route-metadata-backfill*.jsonl  # eventual-consistency lookups
    snapshots/complete-1/item-responses.jsonl
    snapshots/complete-1/manifest.json
```

`item-responses.jsonl` SHA-256:
`db308dba8e62eac2ef06b728675eedb62903b209863049640689f0d013afd9c7`.
It has exactly 2,964 unique model/item pairs, the selected-attempt reference,
original 429 status, frozen TaskSpec digest, model/provider, route evidence,
stop reason, score and missingness. Raw trajectories and request IDs are retained
in immutable attempt files; the manifest pins their URIs and checksums.
The 12-route manifest SHA-256 is
`7d60e3c370928e66fa6165e4c9a6b31a04c1ef901d4ee7a99b8fd7c24e98246e`.

| Model | Exact correct | Graded zero | Ungraded | Missing infrastructure |
|---|---:|---:|---:|---:|
| Qwen3.5-122B-A10B | 101 | 94 | 51 | 1 |
| DeepSeek V3.2 | 2 | 110 | 135 | 0 |
| Mistral Medium 3.5 | 59 | 76 | 112 | 0 |
| Llama 4 Maverick | 0 | 161 | 86 | 0 |
| Gemini 2.5 Flash | 16 | 156 | 58 | 17 |
| GPT-4.1 mini | 5 | 214 | 28 | 0 |
| Claude Haiku 4.5 | 32 | 182 | 33 | 0 |
| Kimi K2.5 | 152 | 55 | 40 | 0 |
| GLM-5.3 (OpenRouter) | 173 | 74 | 0 | 0 |
| Grok 4.3 | 159 | 63 | 25 | 0 |
| Nemotron 3 Ultra | 108 | 29 | 23 | 87 |
| Claude Sonnet 4.6 | 6 | 234 | 7 | 0 |
| **Total** | **813** | **1,448** | **598** | **105** |

These are **frozen-policy scores, not model capability rankings**. Most of
Sonnet's and many other models' zeros are bare-JSON format violations. This
may reflect tool-use/submit-contract friction; independently inspect traces
before interpreting accuracy. Ungraded and missing infrastructure are not
quietly converted to zeros in the snapshot. There is no Rasch/IRT fit yet.

Upstream providers returned 372 initial 429 interruptions (Qwen 102, Mistral
54, Llama 53, Nemotron 163). Selective four-worker retries resolved 284;
**88 still hit 429** (Qwen 1, Nemotron 87). Gemini had another 17
infrastructure-missing trials because its endpoint omitted token usage. Repeated
Nemotron 429s even at four workers argue against immediate repeated retries on
that route. Model/provider mismatches were **not** observed. Seven delayed
OpenRouter generation-metadata lookups were verified later against the same
pinned providers in separate immutable backfill files, without rerunning the
models.

Observed key spend for this campaign (excluding the earlier one-task pilot):
**$43.34**; key remaining **$206.31** at final check. This is well below the
$200 campaign breaker and the key's $250 hard limit. It includes the selective
recovery calls, not infrastructure/cluster costs.

Iris jobs:

- `/zack/biotasks-openrouter-first-wave-v1` — first seven model batches, then
  stopped on generation-metadata propagation (terminal failure; trials retained).
- `/zack/biotasks-openrouter-first-wave-v1-resume1` — resumed without repeating
  completed model batches; completed all 12.
- `/zack/biotasks-openrouter-first-wave-rate-limit-recovery1` — completed
  selective 429 retries.
- `/zack/biotasks-openrouter-first-wave-finalize` — completed snapshot.

Next: inspect format/tool traces, independently validate references (including
items with model disagreements), decide a treatment for model-level ungraded
outcomes and remaining infrastructure missingness, then fit *regularized*
Rasch/1PL if the item/runner quality suffices. Twelve respondents/item cannot
support credible item-level 2PL. Do not expand the panel before this review.
