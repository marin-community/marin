# Phase 4 diagnostics

Pure analysis on existing Phase 4 outputs. No API calls.

## Ungrounded judges (Phase 4 first cut)

### Per-judge score stats (Ungrounded judges (Phase 4 first cut))

| judge | n | mean | std | min | max |
|---|--:|--:|--:|--:|--:|
| `gemini-3-flash-preview` | 574 | 9.58 | 1.60 | 0 | 10 |
| `gpt-5.1` | 578 | 9.29 | 1.64 | 0 | 10 |
| `zai-org/GLM-5.1` | 578 | 8.90 | 1.47 | 0 | 10 |

### Pass-rate sweep (Ungrounded judges (Phase 4 first cut))

Fraction of all judge calls returning compliance_score >= threshold.

| judge | t=6 | t=6.5 | t=7 | t=7.5 | t=8 |
|---|--:|--:|--:|--:|--:|
| `gemini-3-flash-preview` | 94.8% | 94.6% | 94.6% | 92.9% | 92.9% |
| `gpt-5.1` | 95.2% | 94.1% | 94.1% | 89.6% | 89.6% |
| `zai-org/GLM-5.1` | 94.6% | 93.9% | 93.9% | 89.3% | 89.3% |

### Highest / lowest scorer distribution (Ungrounded judges (Phase 4 first cut), 574 complete triples)

Tied scores break alphabetically by judge model name.

| judge | times highest | times lowest |
|---|--:|--:|
| `gpt-5.1` | 203 | 192 |
| `gemini-3-flash-preview` | 361 | 130 |
| `zai-org/GLM-5.1` | 10 | 252 |

### Pairwise Cohen κ on pass/fail @ 7 (Ungrounded judges (Phase 4 first cut))

| pair | n common | Cohen κ |
|---|--:|--:|
| `gemini-3-flash-preview` × `gpt-5.1` | 574 | 0.299 |
| `gemini-3-flash-preview` × `zai-org/GLM-5.1` | 574 | 0.389 |
| `gpt-5.1` × `zai-org/GLM-5.1` | 578 | 0.430 |

### Verbatim-audit per judge (Ungrounded judges (Phase 4 first cut))

Fraction of `cited_spec_clauses` entries that are exact substrings of statement text or example fields.

| judge | verbatim | total | rate |
|---|--:|--:|--:|
| `gemini-3-flash-preview` | 1685 | 1759 | 95.8% |
| `gpt-5.1` | 2232 | 2285 | 97.7% |
| `zai-org/GLM-5.1` | 1725 | 1869 | 92.3% |

## Grounded judges (per-pair rubric)

### Per-judge score stats (Grounded judges (per-pair rubric))

| judge | n | mean | std | min | max |
|---|--:|--:|--:|--:|--:|
| `gemini-3-flash-preview` | 572 | 9.58 | 1.61 | 0 | 10 |
| `gpt-5.1` | 578 | 9.20 | 1.66 | 0 | 10 |
| `zai-org/GLM-5.1` | 578 | 8.67 | 1.77 | 0 | 10 |

### Pass-rate sweep (Grounded judges (per-pair rubric))

Fraction of all judge calls returning compliance_score >= threshold.

| judge | t=6 | t=6.5 | t=7 | t=7.5 | t=8 |
|---|--:|--:|--:|--:|--:|
| `gemini-3-flash-preview` | 95.1% | 94.8% | 94.8% | 92.7% | 92.7% |
| `gpt-5.1` | 95.0% | 92.7% | 92.7% | 88.8% | 88.8% |
| `zai-org/GLM-5.1` | 90.5% | 89.3% | 89.3% | 86.5% | 86.5% |

### Highest / lowest scorer distribution (Grounded judges (per-pair rubric), 572 complete triples)

Tied scores break alphabetically by judge model name.

| judge | times highest | times lowest |
|---|--:|--:|
| `gpt-5.1` | 188 | 187 |
| `gemini-3-flash-preview` | 378 | 118 |
| `zai-org/GLM-5.1` | 6 | 267 |

### Pairwise Cohen κ on pass/fail @ 7 (Grounded judges (per-pair rubric))

| pair | n common | Cohen κ |
|---|--:|--:|
| `gemini-3-flash-preview` × `gpt-5.1` | 572 | 0.438 |
| `gemini-3-flash-preview` × `zai-org/GLM-5.1` | 572 | 0.368 |
| `gpt-5.1` × `zai-org/GLM-5.1` | 578 | 0.537 |

### Verbatim-audit per judge (Grounded judges (per-pair rubric))

Fraction of `cited_spec_clauses` entries that are exact substrings of statement text or example fields.

| judge | verbatim | total | rate |
|---|--:|--:|--:|
| `gemini-3-flash-preview` | 1798 | 1843 | 97.6% |
| `gpt-5.1` | 2485 | 2516 | 98.8% |
| `zai-org/GLM-5.1` | 1935 | 2047 | 94.5% |

## Grounded vs ungrounded judge κ comparison

How much does adding a per-pair rubric change pass/fail agreement among the 3 judges?

- Ungrounded mean pairwise Cohen κ (n_pairs=3): **0.373**
- Grounded mean pairwise Cohen κ (n_pairs=3): **0.448**
- Δ (grounded − ungrounded): **+0.075**

