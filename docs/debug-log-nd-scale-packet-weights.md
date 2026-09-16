# Debugging log for N/D scale packet weights

The goal is to determine whether 60M baseline mixtures can be compared apples-to-apples against the qsplit replay baselines at 130M, 300M, 520M, and 1.2B.

## Initial Status

The baseline scaling plot showed 60M points as "legacy-swarm context" rather than direct apples-to-apples comparisons. Calvin pointed out that proportional and unimax should have unchanged weights if the domain set did not change.

## Hypothesis 1: 60M Source Weights Differ From Qsplit Replay Weights

The original concern was that 60M swarm rows might use different mixture domains or stale source weights, preventing direct comparison.

## Results

The source CSVs agree: `two_phase_many.csv`, `two_phase_many_all_60m_1p2b.csv`, and the packet `sixtym_swarm_reference.csv` have matching phase weights for `baseline_proportional` and `baseline_unimax` on the 39 packet domains. Conceptually, the 60M points are valid apples-to-apples inputs for the baseline scaling plot.

## Hypothesis 2: Packet NPZ Weights Are Stale Or Misaligned

The next check compared `chatgpt_pro_hybrid_data_mixing_packet_v28/data/nd_scale_packet.npz` `weights` against the canonical phase-weight columns in `data/nd_scale_runs.csv`.

## Results

The NPZ `weights` tensor is stale/misaligned for 293 of 643 packet rows:

- `60m_1p2b | legacy_swarm_60m`: 289 mismatched rows
- `1_2b_24b | qsplit_baselines3_holdout`: 3 mismatched rows
- `300m_6b | stratified`: 1 mismatched row

Concrete baseline mismatches:

- `baseline_proportional` 60M: phase TV `[0.203723, 0.203723]` between CSV-derived weights and NPZ weights
- `baseline_unimax` 60M: phase TV `[0.228534, 0.502267]`
- `baseline_proportional` 1.2B holdout: phase TV `[0.203723, 0.203723]`
- `baseline_unimax` 1.2B holdout: phase TV `[0.296358, 0.296358]`

Within the CSV source of truth, baseline weights do match across scales. The blocker was not the experiment design; it was the packet loader trusting stale NPZ weights.

## Changes To Make

Patch `chatgpt_pro_hybrid_data_mixing_packet_v28/code/nd_scale_packet.py` to reconstruct `weights` from `nd_scale_runs.csv` phase columns at load time.

Patch the local 1.2B holdout evaluator to report that loaded packet weights match the CSV-derived weights instead of applying a narrow 1.2B-only baseline patch.

## Results After Patch

The patched loader audit reports:

- `mismatched_rows_after_load`: `0`
- `max_phase_tv_after_load`: `0.0`

Rerunning the compact q/support 1.2B holdout with all packet weights corrected gives:

- train rows: `626`
- held-out 1.2B rows: `2`
- train RMSE: `0.026808`
- 60M train RMSE: `0.037242`
- 60M train slope: `0.243332`
- 1.2B holdout RMSE: `0.118990`

The broader weight fix makes the current compact q/support law look worse than the earlier narrow 1.2B-only baseline patch. The model was partly fitting incorrect 60M mixture features before this correction.

The corrected baseline scaling plot includes 60M as apples-to-apples:

- `baseline_unimax`: `60M 1.083430`, `130M 1.103117`, `300M 0.981139`, `520M 0.888729`, `1.2B 0.827538`
- `baseline_proportional`: `60M 1.091835`, `130M 1.112409`, `300M 0.990733`, `520M 0.895344`, `1.2B 0.832938`

The 130M 1.0x point remains anomalously worse than 60M for both baselines. This is no longer explainable by mixture weights; it is a remaining cross-scale/protocol sanity issue.

## Hypothesis 3: Nominal Scale Names Do Not Match Transformer Capacity

The 60M and 130M rungs are not monotone in RegMix-style non-embedding parameter count:

- `60m_1p2b` uses `regmix_60m_proxy`: `hidden_dim=768`, `intermediate_dim=1536`, `num_layers=10`, approximately `59.0M` non-embedding parameters
- `130m_2p6b` uses `regmix_130m_proxy` / `llama_150m`: `hidden_dim=512`, `intermediate_dim=1792`, `num_layers=6`, approximately `22.8M` non-embedding parameters
- `300m_6b`: approximately `102.6M` non-embedding parameters
- `520m_10p4b`: approximately `339.8M` non-embedding parameters
- `1_2b_24b`: approximately `906.0M` non-embedding parameters

This explains why the 130M rung can be worse than 60M despite matching mixture weights. The `model_size` labels in the packet are nominal scale labels, not a clean monotone capacity axis. Any scaling-law fit that treats `130m_2p6b` as larger than `60m_1p2b` is semantically wrong under the non-embedding-parameter convention.

## Impact

Two issues affect recent packet-based modeling:

- Stale NPZ weights affected any script that consumed `nd_scale_packet.npz["weights"]` directly or through the unpatched packet loader.
- Nominal `model_size` labels affected any joint N/D scaling law that treated `60M < 130M < 300M` as actual capacity.

Fixed-scale 520M calibration probes are less directly affected because the 520M qsplit rows had correct weights and a constant scale label, but model fitting still used corrupted 60M rows before the loader/NPZ repair.

## Corrected q/support Refit

Added and ran:

`experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_qsupport_corrected_data_20260423.py`

This evaluator refits the compact q/support architecture with:

- packet weights reconstructed from `nd_scale_runs.csv`
- actual non-embedding parameter counts used as the scale axis
- refreshed registry labels applied before fitting

Seed-7 split metrics:

- train RMSE: `0.028409`
- seed-7 holdout RMSE: `0.026121`
- fixed-520M RMSE: `0.020842`
- fixed-520M slope: `0.474835`
- fixed-520M std ratio: `0.536017`

Fixed-520M multiplier means:

- `0.5x`: actual `0.909023`, predicted `0.883004`, residual `-0.026019`
- `1.0x`: actual `0.885388`, predicted `0.870593`, residual `-0.014795`
- `2.0x`: actual `0.862621`, predicted `0.862268`, residual `-0.000353`

1.2B explicit holdout:

- holdout RMSE: `0.113156`
- `baseline_unimax`: actual `0.827538`, predicted `0.987523`
- `baseline_proportional`: actual `0.832938`, predicted `0.836612`

Interpretation:

- the compact q/support model is not robust after correcting both mixture weights and the scale axis
- the fixed-520M shape problem reappears strongly; predictions are too low at 0.5x and still compressed
- the 1.2B failure is asymmetric: proportional is close, but unimax is badly overpredicted
- earlier q/support rankings should be considered invalid until rerun from scratch on corrected weights and actual parameter metadata

## Future Work

- [x] Regenerate the baseline scaling plot with the corrected packet weights and include 60M as an apples-to-apples point.
- [x] Rerun the q/support 1.2B holdout evaluation with all packet weights corrected, not only the 1.2B baseline rows.
- [ ] Fix the packet builder so future `nd_scale_packet.npz` files cannot diverge from `nd_scale_runs.csv`.
- [x] Investigate why 130M 1.0x baselines are worse than the matching 60M baselines despite identical source weights.
- [ ] Add actual non-embedding parameter counts to the packet and prefer them over nominal `model_size` for scaling-law fits.
