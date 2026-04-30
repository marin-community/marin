# Within-judge reproducibility

Sampled 30 scenarios. Re-judged each with all 3 judges × 3 reps at temperature=0.2.

Per-judge within-rep statistics:

| judge | n scenarios | mean within-rep std | mean within-rep spread | scenarios with std=0 |
|---|--:|--:|--:|--:|
| `gpt-5.1` | 30 | 0.00 | 0.00 | 30/30 |
| `zai-org/GLM-5.1` | 30 | 0.31 | 0.53 | 23/30 |
| `gemini-3-flash-preview` | 30 | 0.00 | 0.00 | 30/30 |

Interpretation: low within-rep std (e.g. < 0.5) means the judge is largely deterministic at temp=0.2 and the cross-judge κ=0.32 is mostly between-judge anchoring drift. High within-rep std means a substantial chunk of the disagreement is single-judge noise.
