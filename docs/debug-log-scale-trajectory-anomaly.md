# Debugging log for scale trajectory anomaly

Investigate why the complete scale trajectory plot shows `130M` as smaller than `60M` by actual non-embedding parameters, and why several mixtures have worse BPB at the historical 60M point than at the 130M replay point.

## Initial status

The plot `complete_1x_through_520m_actual_params.png` orders the 1.0x points as `130M -> 60M -> 300M -> 520M` on the actual non-embedding parameter axis. Six of the twelve mixtures have higher BPB at the 60M point than at the 130M point, which looks inconsistent if the points are interpreted as a clean same-mixture scaling ladder.

## Hypothesis 1: actual parameter metadata is wrong

The apparent reversal could come from a bad hard-coded parameter map in the plotting/eval code.

## Results

The reversal is real under the non-embedding parameter convention:

- `regmix_60m_proxy`: hidden 768, MLP 1536, 10 layers, 8 heads, 8 KV heads, tied embeddings. Non-embedding params: `58,998,528`.
- `regmix_130m_proxy`: repo `llama_150m` geometry with hidden 512, MLP 1792, 6 layers, 8 heads, 8 KV heads, tied embeddings. Non-embedding params: `22,813,184`.
- `regmix_300m_proxy`: non-embedding params: `102,648,576`.
- `regmix_520m_proxy`: non-embedding params: `339,788,800`.
- `regmix_1_2b_proxy`: non-embedding params: `906,037,248`.

So the `130M` label is a nominal/historical rung name, not a reliable actual non-embedding parameter count. The 60M RegMix proxy is larger than the 130M rung under the scaling-law `N` convention.

## Hypothesis 2: the 60M and 130M mixtures are mismatched

The 60M-vs-130M nonmonotonic rows could come from stale/misaligned phase weights.

## Results

For the 12 plotted mixtures, the phase weights in `nd_scale_runs.csv` match exactly between the 60M historical rows and the 130M qsplit replay rows:

- phase-column L1 difference: `0.0` for every checked run.
- phase 0 and phase 1 both sum to `1.0` for both scales.

This is not the stale packet-weight bug.

## Hypothesis 3: the 60M point is not apples-to-apples with replay scales

The plot might be mixing historical 60M rows with new strong-tier replay rows.

## Results

Confirmed. In `complete_1x_through_520m.csv`:

- `60m_1p2b` rows come from `legacy_swarm_60m`, mainly `two_phase_many_all_60m_1p2b.csv`.
- `130m_2p6b`, `300m_6b`, and `520m_10p4b` rows come from qsplit replay / strong-tier runs.

The 60M rows have historical metadata:

- `experiment_budget`: `1.2B`
- `realized_train_tokens`: `1,199,833,088`
- `final_checkpoint_step`: `4576`
- `model_family`: missing in packet rows

The 130M replay rows have:

- `experiment_budget`: `2.6B`
- `realized_train_tokens`: `2,599,944,192`
- `final_checkpoint_step`: `9917`
- `model_family`: `regmix_130m_proxy`
- MuonH optimizer config through the strong-tier scaling recipe

Within the registry-native replay scales, all 12 mixtures are monotone:

`130M BPB > 300M BPB > 520M BPB`

The only nonmonotonicity is the historical 60M point relative to the 130M replay point.

## Conclusion

There is no evidence of a plotting bug or a mixture-weight alignment bug in this specific plot. The misleading part is the data contract: it combines a historical 60M swarm point with newer replay/strong-tier scales and places them on an actual non-embedding parameter axis where the nominal `130M` rung is actually smaller than `60M`.

Treat the 60M point as legacy context, not as a clean rung in the new scaling ladder. For clean scaling diagnostics, use the replay-native 130M/300M/520M points until a replayed 60M rung exists or the old 60M checkpoints are re-evaluated under the same eval/launcher contract.

## Future Work

- [ ] Add a canonical actual-parameter metadata table to the packet instead of hard-coding it in evaluation scripts.
- [ ] Rename plots to distinguish nominal labels from actual non-embedding parameter counts.
- [ ] Add a registry-native scaling plot that excludes historical 60M rows by default.
- [ ] If 60M is still important for modeling, re-run or re-evaluate the 60M mixtures under the same qsplit replay/expanded-eval contract.
