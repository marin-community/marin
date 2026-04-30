# Debugging log for benchmark aggregate double counting

## Initial status

The expanded all-suite benchmark aggregate fit looked suspiciously worse than
the OLMoBase easy-overlap fit. OLMoBase easy-overlap accuracy fit at about
`0.636` OOF Spearman, while the expanded suite-level mean only fit at about
`0.274-0.285`.

## Hypothesis 1: optimizer settings caused the drop

I reran OLMoBase-only and all-suite targets under the same fast diagnostic
optimizer:

```bash
uv run --with torch python - <<'PY'
# imports fit_grp_no_l2_benchmark_aggregates.py and runs _fit_objective
# with method=L-BFGS-B, random_starts=4, coarse_top_k=2.
PY
```

Results:

| target | optimizer | OOF Spearman |
|:--|:--|--:|
| OLMoBase easy-overlap accuracy | L-BFGS-B 4 starts | 0.637 |
| all-suite accuracy | L-BFGS-B 4 starts | 0.274 |
| all-suite accuracy | L-BFGS-B 12 starts | 0.255 |
| all-suite accuracy logit | L-BFGS-B 12 starts | 0.260 |

The optimizer is not the main cause. OLMoBase remains strong under the same
optimizer settings, while the expanded target remains weak.

## Hypothesis 2: the expanded target is dominated by added noisy metrics

Target correlations showed that the all-suite mean was more aligned with MMLU
than with OLMoBase:

| component | Spearman vs all-suite |
|:--|--:|
| OLMoBase easy-overlap accuracy | 0.466 |
| MMLU accuracy | 0.718 |
| GSM8K accuracy | 0.332 |
| HumanEval pass@1 | 0.473 |

Component ablations:

| target | OOF Spearman |
|:--|--:|
| OLMoBase + MMLU | 0.276 |
| OLMoBase + GSM8K | 0.394 |
| OLMoBase + HumanEval | 0.481 |
| OLMoBase + GSM8K + HumanEval | 0.398 |
| OLMoBase + MMLU + GSM8K + HumanEval | 0.274 |

MMLU is the largest source of the drop.

## Root cause: MMLU was included twice

`OLMO_BASE_EASY_OVERLAP_ACCURACY_COLUMNS` already included
`lm_eval/mmlu_5shot/acc`. The all-suite target then averaged:

- OLMoBase easy-overlap mean, already containing MMLU,
- MMLU accuracy again,
- GSM8K accuracy,
- HumanEval pass@1.

This double-counted the noisiest added component.

## Fix

Updated
`experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_no_l2_benchmark_aggregates.py`
to add:

- `olmo_base_easy_overlap_no_mmlu_accuracy_mean`
- `olmo_base_easy_overlap_no_mmlu_bpb_mean`
- corrected `all_suite_accuracy_mean`
- corrected `all_suite_bpb_mean`

The corrected all-suite accuracy target uses:

```text
mean(OLMoBase easy-overlap excluding MMLU, MMLU, GSM8K, HumanEval)
```

The corrected BPB target uses:

```text
mean(MMLU BPB, available non-MMLU OLMoBase easy-overlap BPB tasks)
```

## Results

Corrected diagnostic fit:

| target | OOF RMSE | OOF Spearman |
|:--|--:|--:|
| OLMoBase no-MMLU accuracy raw | 0.003304 | 0.628 |
| OLMoBase no-MMLU accuracy logit | 0.003309 | 0.628 |
| all-suite no-duplicate accuracy raw | 0.002365 | 0.324 |
| all-suite no-duplicate accuracy logit | 0.002368 | 0.300 |
| all-suite no-duplicate BPB | 0.052643 | 0.328 |

The fix improves the expanded aggregate from about `0.274-0.285` to about
`0.300-0.324`, but the target remains much less fitable than OLMoBase alone.
The remaining drop is real signal/noise conflict from MMLU, GSM8K, and
HumanEval, not just an optimizer issue.

## Future Work

- [ ] Consider reporting OLMoBase easy-overlap excluding MMLU as the default
      local aggregate when MMLU is modeled separately.
- [ ] Consider suite-level rank/z-score aggregation, but the initial z-score
      check did not recover OLMoBase-level fit quality.
- [ ] Rerun candidate aggregate objectives with the full Powell multistart
      optimizer before promoting any benchmark-aggregate optimizer.

