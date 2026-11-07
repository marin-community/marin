# BioTasks v3 OpenRouter diagnostic results (2026-11-07)

The versioned reference/path/submission-contract wave **completed**. This is
**not** an independently verified task bank or IRT analysis. The first-wave v2
scores, frozen TaskSpecs and raw trajectories were never overwritten.

Private immutable artifacts:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v3/
  snapshots/reference-path-review-1/manifest.json
  calibration/openrouter-first-wave-v3/attempts/<model>/smoke.jsonl
  calibration/openrouter-first-wave-v3/attempts/<model>/results.jsonl
  calibration/openrouter-first-wave-v3/attempts/anthropic--claude-sonnet-4.6/results-part{1,2,3}.jsonl
  calibration/openrouter-first-wave-v3/snapshots/complete-1/manifest.json
  calibration/openrouter-first-wave-v3/snapshots/complete-1/item-responses.jsonl
```

The response snapshot SHA-256 is
`acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda`.
Its manifest names/hashes all contributing raw attempt files and delayed
provider-metadata overlays. V3 TaskSpecs SHA-256 is
`b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6`.

**Coverage:** 230 provisional tasks × 12 pinned model/provider routes =
2,760 unique model/task pairs. **690** original v2 generations were proven
identical in model-visible instructions, tools, files, sandbox settings,
provider and generation limits, then rescored under v3's reference-blind
single-JSON-object extractor. **2,070** pairs needed fresh attempts (170 changed
tasks/model plus 30 v2 infrastructure-missing pairs). The twelve v3 smoke
responses were reused by the main run; no completed pair was sampled again.
The offline reuse audit found 66 grades changed without any new generation.

| Model | Correct | Scored zero | Ungraded | Missing infrastructure |
|---|---:|---:|---:|---:|
| Qwen3.5-122B-A10B | 164 | 47 | 19 | 0 |
| DeepSeek V3.2 | 127 | 36 | 67 | 0 |
| Mistral Medium 3.5 | 105 | 62 | 63 | 0 |
| Llama 4 Maverick | 0 | 153 | 77 | 0 |
| Gemini 2.5 Flash | 59 | 125 | 25 | 21 |
| GPT-4.1 mini | 60 | 110 | 60 | 0 |
| Claude Haiku 4.5 | 138 | 85 | 7 | 0 |
| Kimi K2.5 | 178 | 38 | 13 | 1 |
| GLM-5.3 (OpenRouter) | 167 | 62 | 1 | 0 |
| Grok 4.3 | 168 | 53 | 9 | 0 |
| Nemotron 3 Ultra / BaseTen | 155 | 30 | 17 | 28 |
| Claude Sonnet 4.6 | 174 | 54 | 2 | 0 |
| **Total** | **1,495** | **855** | **360** | **50** |

The v3 counts **must not be compared as an ability gain** against v2's 813
correct: the cohort, references, path behavior and answer extraction changed.
Notably Sonnet's formerly malformed prose-embedded JSON can now receive a
score if there is exactly one parseable object; this is a different frozen
grading contract. No invalid submission or ungraded result was silently made
correct by referencing the answer. Llama still shows pervasive tool/format
problems, so its zeros are not interpretable as biology competence.
Infrastructure remains incomplete, especially Gemini's usage-reporting
endpoint and BaseTen's upstream 429s; do **not** cast the 50 missing responses
as zeros. Quarantined references (58 original plus 17 first-wave consensus
leads), the remaining provisional tasks, task instructions lacking exact JSON
key spellings, and an independent reference survey still require review before
IRT or release. Twelve respondents are insufficient for credible item-level
2PL/3PL estimates.

OpenRouter key spend since the immutable v3 budget baseline, including all
smokes and the premium-anchor batch: **$26.219**, below the $50 pre-batch gate;
remaining key credit at final check: **$180.095**. This is API usage only.
The whole-batch premium reserve triggered a safe stop *before* any Sonnet
batch; the resumed run partitioned Sonnet into three disjoint immutable
batches, each budget-gated under the same ceiling.

Iris jobs:

- `/zack/biotasks-v3-freeze`: published immutable v3 tasks.
- `/zack/biotasks-openrouter-v3-smoke`: 12 model-route smokes.
- `/zack/biotasks-v3-offline-reuse-audit`: validated 690 cached grades offline.
- `/zack/biotasks-openrouter-v3-first-wave`: eleven model batches, then
  stopped safely at the premium reserve gate.
- `/zack/biotasks-openrouter-v3-first-wave-resume1`: completed three Sonnet
  partitions without rerunning existing batches.
- `/zack/biotasks-openrouter-v3-finalize`: verified and published the
  complete response matrix.
