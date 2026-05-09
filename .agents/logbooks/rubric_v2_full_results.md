Loading v1 (existing) judgments...
  v1 cells: 7338
Loading v2 phase_4 judgments...
  v2 phase_4 cells: 396

# Agreement on 3 unfixed statements: v1 vs v2_partial vs v2_full

| statement | n | α (v1) | α (v2_partial: GPT only) | α (v2_full: all 3) | Δ (v2_full − v1) |
|---|--:|--:|--:|--:|--:|
| no_agenda | 80 | +0.761 | +0.756 | +0.980 | +0.219 |
| comply_with_laws | 80 | -0.034 | -0.055 | -0.039 | -0.005 |
| sexual_content_involving_minors | 73 | +0.101 | +0.084 | +0.221 | +0.120 |

## Fleiss κ binary (k2) — same comparison

| statement | k2 (v1) | k2 (v2_partial) | k2 (v2_full) |
|---|--:|--:|--:|
| no_agenda | +0.742 | +0.742 | +0.980 |
| comply_with_laws | +0.178 | +0.137 | +0.059 |
| sexual_content_involving_minors | -0.024 | -0.019 | -0.005 |

## Fleiss κ 3-way (k3) — same comparison

| statement | k3 (v1) | k3 (v2_partial) | k3 (v2_full) |
|---|--:|--:|--:|
| no_agenda | +0.725 | +0.725 | +0.960 |
| comply_with_laws | +0.076 | +0.062 | +0.047 |
| sexual_content_involving_minors | +0.007 | +0.053 | +0.239 |

# Per-judge: did Gemini and Claude actually shift under v2?

Counts of cells where each judge's score shifted v1 → v2 (rubric only changed; bare judgments unchanged).

| statement | judge | n shifted | n unchanged | mean Δscore | range Δscore |
|---|---|--:|--:|--:|---|
| no_agenda | gpt | 3 | 77 | +0.33 | [-1, 1] |
| no_agenda | gemini | 12 | 68 | -3.75 | [-4, -2] |
| no_agenda | claude | 13 | 65 | +0.00 | [-4, 3] |
| comply_with_laws | gpt | 34 | 46 | +0.21 | [-2, 2] |
| comply_with_laws | gemini | 2 | 73 | +0.00 | [-1, 1] |
| comply_with_laws | claude | 25 | 55 | -0.08 | [-2, 1] |
| sexual_content_involving_minors | gpt | 5 | 73 | +0.20 | [-1, 1] |
| sexual_content_involving_minors | gemini | 1 | 67 | +1.00 | [1, 1] |
| sexual_content_involving_minors | claude | 16 | 62 | +1.75 | [1, 3] |

# Sanity check — bare condition (should be identical, v2 only affects phase_4)

| statement | α_bare (v1) | α_bare (v2_full) |
|---|--:|--:|
| no_agenda | +0.914 | +0.914 |
| comply_with_laws | +0.113 | +0.113 |
| sexual_content_involving_minors | +0.211 | +0.211 |
