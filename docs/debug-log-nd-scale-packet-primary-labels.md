# Debugging log for nd-scale packet primary labels

Investigate why the matched-mixture budget plot showed `baseline_unimax` spiking upward at `520M x 1x`, and fix the packet/plot so only trustworthy strong-tier primary labels are used.

## Initial status

- The plot in `reference_outputs/figures/matched_mixture_budget_trajectories.png` showed `baseline_unimax` jumping to roughly `3.84` BPB at `520M x 1x`.
- That behavior was not scientifically plausible and looked inconsistent with the rest of the strong-tier data.

## Hypothesis 1

The plotting code is reading a bad packet label for `baseline_unimax` at `520M x 1x`, likely from an incomplete or running strong-tier row that should not have been treated as a valid primary label.

## Changes to make

- Inspect the packet row provenance for `baseline_unimax` at `520M x 1x`.
- Compare packet `primary_y` against the canonical run registry.
- If the packet is admitting untrusted strong-tier labels, patch the packet builder to gate strong-tier primary labels on registry trust.

## Future Work

- [ ] Regenerate other packet-local artifacts that may have been affected by the old strong-tier label selection.
- [ ] Consider centralizing this trust policy so duplicated packet builders cannot drift again.

## Results

- The packet row for `baseline_unimax` at `520M x 1x` was a `running` pilot row from `ngd3dm2_qsplit240_520m_chinchilla`, not a perplexity-ready strong-tier row.
- The packet had `primary_y = 3.837783` for that row because `build_packet.py` read the last checkpoint `eval_metrics.jsonl` record and treated any finite `eval/uncheatable_eval/bpb` as a valid primary label.
- The canonical run registry did **not** consider that row perplexity-ready. The registry-side logical row for the same cell had `is_perplexity_ready = False`.
- There was also a mismatch between the raw checkpoint blob (`3.837783`) and the registry’s vetted `objective_metric_value` (`1.041495`) for that same `checkpoint_root`, which confirmed the packet should not trust the raw last-eval blob here.
- Fix implemented:
  - strong-tier rows now carry `objective_metric_value` and `is_perplexity_ready` into the packet builder,
  - strong-tier `PRIMARY_METRIC` is masked unless `is_perplexity_ready`,
  - when trusted, `PRIMARY_METRIC` is overwritten with the registry’s `objective_metric_value`.
- A reproducible plot script was added at `chatgpt_pro_swarm_transfer_packet/code/plot_matched_mixture_budget_trajectories.py`.

## Hypothesis 2

After rebuilding the packet with trustworthy strong-tier primary labels, the suspicious upward jump should disappear, and the matched-mixture trajectory plot should show either sensible `520M` values or explicit missing data where the strong-tier cell is not yet ready.

## Changes to make

- Rebuild the packet data from the patched builder.
- Regenerate the matched-mixture trajectory CSVs and plot from the corrected packet.

## Results

- Source fix implemented in both packet builders:
  - `chatgpt_pro_swarm_transfer_packet/build_packet.py`
  - `chatgpt_pro_hybrid_data_mixing_packet_v20/build_packet.py`
- Strong-tier primary labels are now trusted only when `is_perplexity_ready`.
- Trusted strong-tier primary labels now use the registry’s vetted `objective_metric_value` instead of the raw last `eval_metrics.jsonl` record under `checkpoint_root`.
- Because the packet rebuild was interrupted after clearing generated directories, the current packet data was restored from the v20 packet copy and patched in-place with the same trust rule.
- Packet correction summary:
  - `primary_mask_count_before = 638`
  - `primary_mask_count_after = 604`
  - `strong_tier_rows_total = 121`
  - `strong_tier_rows_masked = 39`
  - `strong_tier_rows_overridden = 82`
- Corrected row counts now show:
  - `520M x 0.5x = 3`
  - `520M x 1.0x = 1`
  - `520M x 2.0x = 0`
- The corrected matched-mixture trajectory plot was regenerated and no longer shows the `baseline_unimax` `520M x 1x` spike.

## Hypothesis 3

Many of the remaining untrusted `520M` rows may simply be incomplete strong-tier runs that wrote intermediate `eval/uncheatable_eval/bpb` blobs before reaching the target checkpoint step. If so, the right trust boundary is the run registry's `reached_target_step` / `is_perplexity_ready`, not whether any eval metric exists.

## Changes to make

- Audit all `520m_10p4b` strong-tier logical rows in the run registry.
- Separate:
  - rows that reached target and are perplexity-ready,
  - rows with partial evals only,
  - rows with no usable final eval.

## Results

- The `520M` strong-tier registry has `39` logical rows total.
- Exactly `4/39` are trustworthy (`is_perplexity_ready = True`), and all four also have `reached_target_step = True`.
- The remaining `35/39` are **not** hidden finished runs:
  - all `35` have `has_objective_metric_value = True`,
  - all `35` have `reached_target_step = False`,
  - none of them are `reached_target_step = True` while still failing `is_perplexity_ready`.
- So the bad `520M` labels are almost entirely **partial-training evals**, not missing final labels that the packet failed to pick up.
- Cell-level summary:
  - `qsplit_representative12, 0.5x`: `12` rows total, `3` perplexity-ready, `12` with some eval
  - `qsplit_representative12, 1.0x`: `12` rows total, `1` perplexity-ready, `12` with some eval
  - `qsplit_representative12, 2.0x`: `12` rows total, `0` perplexity-ready, `12` with some eval
  - `stratified, 0.5x`: `1` row total, `0` perplexity-ready, `1` with some eval
  - `stratified, 1.0x`: `1` row total, `0` perplexity-ready, `1` with some eval
  - `stratified, 2.0x`: `1` row total, `0` perplexity-ready, `1` with some eval
- The trustworthy `520M` rows are:
  - `baseline_unimax` at `0.5x`
  - `run_00018` at `0.5x`
  - `run_00213` at `0.5x`
  - `run_00090` at `1.0x`
- This explains why so many `520M` numbers looked available in raw checkpoint files but were untrustworthy in the packet:
  - intermittent evaluation ran during training,
  - the old packet builder treated those intermediate metrics as final labels,
  - but the registry correctly marks them unusable until target step is reached and perplexity is ready.
