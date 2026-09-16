# Proportional versus UniMax-8 at 1e21 FLOPs

Both final step-22056 checkpoints completed the same accuracy evaluation. UniMax-8
has lower native Table-9 BPB but lower accuracy on 8 of the 11 evaluated families.
The native BPB improvement is concentrated in math, code and some basic skills,
which this accuracy suite does not test. This is a performance trade-off, not a
uniform improvement or evidence that lower gold-answer BPB guarantees accuracy.

## Accuracy

Values are percentages; changes are UniMax minus Proportional in percentage points.
ARC Easy, ARC Challenge, HellaSwag and PIQA use length-normalized accuracy;
the others use raw accuracy. All are five-shot except zero-shot LAMBADA.

| Family | Proportional | UniMax-8 | Change |
| --- | ---: | ---: | ---: |
| MMLU | 34.51 | 33.71 | -0.80 |
| ARC Easy | 69.15 | 68.64 | -0.51 |
| ARC Challenge | 38.05 | 37.80 | -0.26 |
| CommonsenseQA | 27.44 | 25.72 | -1.72 |
| HellaSwag | 63.13 | 58.47 | -4.66 |
| WinoGrande | 62.27 | 58.33 | -3.95 |
| SocialIQA | 45.75 | 45.85 | +0.10 |
| PIQA | 74.65 | 74.59 | -0.05 |
| SciQ | 95.00 | 95.10 | +0.10 |
| LAMBADA | 60.33 | 56.32 | -4.02 |
| MedMCQA | 29.48 | 31.20 | +1.72 |
| Equal-family mean | 54.52 | 53.25 | -1.28 |

The mean is descriptive, not a previously selected headline metric. MMLU has one
family weight, rather than 57 of 67 equal leaf weights. The full subject and family
tables are in `leaf_comparison.csv` and `family_comparison.csv`.

## BPB Decomposition

Native Table-9 macro BPB improves from 0.673091 to 0.625261, a 7.11% reduction.
The checkpoint identities match the fresh accuracy reports.

| Native Table-9 group | Components | Proportional BPB | UniMax-8 BPB | UniMax BPB wins |
| --- | ---: | ---: | ---: | ---: |
| Math | 7 | 0.632127 | 0.517607 | 7/7 |
| Code | 19 | 0.521671 | 0.459017 | 19/19 |
| Basic skills | 6 | 0.743475 | 0.629460 | 3/6 |
| Remaining QA | 19 | 0.817375 | 0.829842 | 5/19 |

The first three groups contribute -0.052474 to the 51-component macro change;
the QA group offsets this by +0.004644. Math/code generation accuracy was not
evaluated, so BPB gains do not establish better problem-solving success rates.

## Correlation

These are correlations across tasks of BPB improvement (Proportional minus UniMax)
with accuracy improvement (UniMax minus Proportional), not correlations of absolute
task difficulty or correlations estimated across independently trained models.

| Comparison | Tasks | Pearson | Spearman | Matching improvement direction |
| --- | ---: | ---: | ---: | ---: |
| Same-harness BPB, families | 7 | -0.006 | 0.071 | 5/7 |
| Native BPB, families | 11 | 0.029 | 0.136 | 6/11 |
| Same-harness BPB, leaves | 63 | 0.556 | 0.517 | 42/63, with 4 ties |
| Native BPB, leaves | 67 | 0.260 | 0.239 | 34/67, with 4 ties |

The moderate same-harness leaf correlation is dominated by the 57 MMLU subjects;
it does not imply a reliable relationship across benchmark families. Native MMLU
family BPB is subject-size-weighted to match the accuracy aggregation for this
comparison; Table-9 itself uses four separately weighted MMLU components.

The native and fresh protocols differ: native gold-answer/reading-comprehension
requests versus standard lm-eval prompts, 8192 versus 4096 maximum context, and
FP32 versus BF16. Within the fresh harness, lower BPB still need not increase
accuracy: CommonsenseQA BPB improves from 2.929897 to 2.853116 while accuracy falls
1.72 points. Accuracy depends on the gold answer beating the alternatives, not
only on its absolute probability.

This is one trained checkpoint per mixture, with different data seeds
(660704 and 660705). These are descriptive findings, not replicated policy effects
or significance claims. No new training, evaluation, or paper edits were made.

## Provenance And Reproduction

- Frozen plan: `2ec4492fd7cd0363284dc2ba24eb1dbca48aa25af34de6d6b1c1ebb730c99a7c`.
- Both reports cover all 67 task leaves and 44,248 documents with identical task configs, runtime and evaluation seed.
- Iris parent `/calvinxu/mariner-ladder-accuracy-1e21-v5p-cpu-staging-20260914` succeeded; its final stage verifies all artifacts and full document coverage.
- This analysis independently verifies completion identities and hashes of bounded summary/provenance files; full per-document outputs remain in regional GCS.
- Native BPB summaries were read from W&B runs [Proportional](https://wandb.ai/marin-community/marin-eval/runs/cjsscu9h) and [UniMax-8](https://wandb.ai/marin-community/marin-eval/runs/66hcnt1c), with checkpoint paths checked against the fresh reports.
- `native_wandb_summaries.json` preserves those source summaries; `native_51_components.csv` records every component.

From the repository root:

```bash
uv run .agents/projects/mariner_ladder_accuracy_20260914/compare_accuracy_bpb.py \
  --native-summary .agents/projects/mariner_ladder_accuracy_20260914/comparison/native_wandb_summaries.json \
  --output .agents/projects/mariner_ladder_accuracy_20260914/comparison
```
