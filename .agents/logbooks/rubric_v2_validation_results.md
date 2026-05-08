Loading v1 (existing) judgments...
  v1 cells: 7338
Loading v2 GPT-5.1 phase_4 judgments...
  v2 GPT phase_4 rows: 396

# v1 vs v2 ensemble agreement on the 5 affected statements

| statement | n_p4 | α_p4 (v1) | α_p4 (v2) | Δα (v2−v1) | k3_p4 (v1) | k3_p4 (v2) | k2_p4 (v1) | k2_p4 (v2) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| refusal_style | 78 | +0.213 | +0.330 | +0.117 | +0.038 | +0.196 | +0.094 | +0.264 |
| no_agenda | 80 | +0.761 | +0.756 | -0.005 | +0.725 | +0.725 | +0.742 | +0.742 |
| comply_with_laws | 80 | -0.034 | -0.055 | -0.021 | +0.076 | +0.062 | +0.178 | +0.137 |
| avoid_abuse | 75 | -0.138 | +0.647 | +0.785 | -0.136 | +0.412 | -0.191 | +0.321 |
| sexual_content_involving_minors | 71 | +0.101 | +0.084 | -0.016 | +0.007 | +0.053 | -0.024 | -0.019 |

Mean Δα (phase_4, v2−v1) across 5 statements: +0.172

# Sanity check — bare-condition agreement (should be identical, since v2 only affects phase_4)

| statement | n | α_bare (v1) | α_bare (v2) |
|---|--:|--:|--:|
| refusal_style | 78 | +0.382 | +0.382 |
| no_agenda | 80 | +0.914 | +0.914 |
| comply_with_laws | 80 | +0.113 | +0.113 |
| avoid_abuse | 78 | -0.022 | -0.022 |
| sexual_content_involving_minors | 66 | +0.211 | +0.211 |

# Top-K rubric-poison cells under v1 — does Δpwv drop under v2?

| statement | total Δpwv (v1 top-12) | total Δpwv (same cells, v2) | drop |
|---|--:|--:|--:|
| refusal_style | +48 | +40 | +17% |
| no_agenda | +282 | +282 | +0% |
| comply_with_laws | +102 | +88 | +14% |
| avoid_abuse | +318 | +4 | +99% |
| sexual_content_involving_minors | +62 | +60 | +3% |

# GPT outlier rate within top-K poison cells — v1 vs v2 (rubric condition)

| statement | gpt outlier rate (v1) | gpt outlier rate (v2) |
|---|--:|--:|
| refusal_style | 4/12 (33%) | 2/2 (100%) |
| no_agenda | 10/12 (83%) | 10/10 (100%) |
| comply_with_laws | 8/12 (67%) | 6/12 (50%) |
| avoid_abuse | 12/12 (100%) | 2/2 (100%) |
| sexual_content_involving_minors | 4/12 (33%) | 4/6 (67%) |
