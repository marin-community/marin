# Figure 3: residual shape near 60% StarCoder

The corrected target curve has a small flattening near 60% StarCoder. Its loss still decreases from 55% through 70%. The 60% point is 0.001093014189 BPB above the straight line between its 55% and 65% neighbors, or 0.1421% of the 60% loss. This deviation is present in the saved token loss and survives the common BPB conversion. The audit found no residual metric, point-mapping, evaluation-population, or checked configuration error that explains it. Its cause remains unresolved.

All 102 archived raw metric files pass their recorded hashes and frozen run identities. Reconstructing their normalized losses independently reproduces all 60 plotted curve means exactly. The paper input is byte-identical to the corrected analysis, and the current PDF and PNG match the September 12 correction receipts. No training, checkpoint evaluation, cloud payload reads, or changes to canonical data, the paper, or Fieldbook were made.

## Target measurements

BPB is total prediction-loss bits divided by total scored bytes. Every row uses the same factor:

$$
\mathrm{BPB}=\mathrm{token\ loss}\frac{11{,}612{,}631}{28{,}741{,}166\ln 2}
=0.582909028651567\,\mathrm{token\ loss}.
$$

The frozen PALOMA programming-language validation population contains 5,673 examples. The counts have canonical SHA-256 `7c7258e80208b230394cbee4ce1f41436d6c248d5d02e4ad86d24d05b5c32f4a`.

| StarCoder fraction | Saved token loss | Consistent BPB | Change from previous row | Original BPB schema |
|---:|---:|---:|---:|---|
| 0.50 | 1.3306833505630493 | 0.7756673393195197 | — | Legacy |
| 0.55 | 1.3211923837661743 | 0.7701349690829891 | −0.0055323702365306 | 2 |
| 0.60 | 1.3199977874755860 | 0.7694386281196114 | −0.0006963409633777 | 2 |
| 0.65 | 1.3150529861450195 | 0.7665562587791359 | −0.0028823693404755 | 2 |
| 0.70 | 1.3133516311645508 | 0.7655645236000794 | −0.0009917351790565 | Legacy |

There is no schema boundary inside the 0.55–0.65 neighborhood. The original schema-2 values differ from reconstruction by 4.32e-8, 1.23e-7, and 1.15e-7 BPB at 0.55, 0.60, and 0.65. These discrepancies are roughly four orders of magnitude below the 0.001093 BPB shape deviation. Across all five schema-2 endpoints, the largest discrepancy is 1.851067024106e-7 BPB. The three previously completed checkpoint audits share the counts hash and reconstruct corrected BPB within 1.224764398344e-6. Those checkpoint audits cover other coordinates; they do not freshly rescore the four target neighbors.

The reference line between neighbors is a descriptive diagnostic. The experiment does not establish that the true response should be locally linear, so its residual is not by itself a model error or an outlier test. The sampled target minimum remains 70%; it is supported by one training seed.

## Proxy measurements

The unmatched proxy and target use the same 92,928-sequence StarCoder parent pool. Each matched proxy uses a 5,120-sequence subset to match the target's StarCoder epoch exposure at the proxy's smaller token budget. The three subset labels below are the seeds used to draw those distinct subsets; each plotted curve averages trainer seeds 20260910 and 20260911. Both seeds of the unmatched proxy have positive 60% chord residuals. One seed actually worsens from 55% to 60%. The plotted two-seed mean only flattens. The matched subset means have smaller positive residuals, while their individual seed residuals have both signs.

| Plotted curve | BPB at 0.55 | BPB at 0.60 | BPB at 0.65 | BPB at 0.70 | 0.60 above 0.55–0.65 line |
|---|---:|---:|---:|---:|---:|
| Unmatched, two-seed mean | 1.085302391312 | 1.084557165420 | 1.072046237128 | 1.063628196863 | +0.005882851200 |
| Matched subset 20260912, two-seed mean | 1.133708997909 | 1.136383701850 | 1.138079803876 | 1.136091365114 | +0.000489300957 |
| Matched subset 20260913, two-seed mean | 1.129211897191 | 1.132684915986 | 1.132118691624 | 1.134608661261 | +0.002019621579 |
| Matched subset 20260914, two-seed mean | 1.127520381384 | 1.129585569831 | 1.130750052604 | 1.130761865593 | +0.000450352837 |

For unmatched seed 20260910, the 55%→60% change is −0.003916631280 BPB and the chord residual is +0.003468050391. For seed 20260911, the change is +0.002426179497 and the residual is +0.008297652009. The matched individual residuals range from −0.003978649473 to +0.005780269287 BPB. Every proxy endpoint in this neighborhood originally used the legacy schema and now receives the same reconstruction factor. The small features therefore do not track a proxy schema transition.

These curves share the fixed data seed, parent data source, and mixture construction; the matched curves use distinct subsets of that parent. Their coincident shape is not independent evidence of random training noise, nor does the two-seed comparison identify a mechanism. Per-run values and deltas are retained in [audit.json](audit.json).

## Training history and realized allocations

The target 60% chord residual, computed from token losses at common saved evaluation steps, changes during training:

| Saved evaluation step | 60% residual in reconstructed BPB |
|---:|---:|
| 2,298 | +0.006575978335 |
| 4,596 | +0.001329795132 |
| 6,894 | +0.004844819527 |
| 9,192 | −0.002767053720 |
| 11,490 | +0.001093014189 |

The feature exists before the 60% run's interactive recovery from checkpoint `step-6958`. At step 2,298, the 65% run has duplicate records on the two BPB schemas with exactly identical token loss, 1.6138192415237427. The older evaluator therefore already measures the early positive residual. This excludes the BPB correction as its origin. The changing sign also cautions against treating the final residual as a fixed offset through training.

The archived realized allocations were recomputed locally with `sequence_allocation` for 55%, 60%, 65%, and 70%, without reading token payloads. All component counts match exactly. Actual StarCoder fractions are:

| Nominal fraction | Target realized fraction | Unmatched proxy realized fraction | Target epochs |
|---:|---:|---:|---:|
| 0.55 | 0.551761296885 | 0.551873518957 | 8.733180526860 |
| 0.60 | 0.601563179880 | 0.601698262243 | 9.521435950413 |
| 0.65 | 0.650878948742 | 0.650979956556 | 10.301997245179 |
| 0.70 | 0.701165586111 | 0.701298380727 | 11.097925275482 |

The target's 55%→60% and 60%→65% realized increments are 0.049801882995 and 0.049315768863. Using these realized fractions for the chord gives a residual of 0.001101789929 BPB, leaving the feature essentially unchanged. The unmatched proxy remains below one StarCoder epoch. Matching errors between target and matched-proxy epochs are at most 0.004612% in these four rows. Realized allocation totals rule out a large mixture-fraction mislabel here. The configuration comparison separately finds the same target support size and cache path; neither check audits every sample in the training stream or independently establishes unchanged cache payloads.

## Source and checkpoint provenance

The paper builder reads normalized values directly, plots sorted measured coordinates, and joins them with straight segments. It performs no fitted smoothing. The current PNG was visually inspected: the target and unmatched-proxy flattening is visible, and neither local feature comes from a break in the line or cropped labels. The paper's normalized input, builder, and allocation file all match the hashes in its figure receipt. Current PDF SHA-256 is `7d4d01d2688f9481fc17bc032df836691082804d2cb93e4674d2f68f2b148b8f`.

Independent recursive comparisons of the retained W&B `model`, `optimizer`, `trainer`, and `data` subtrees for 55%, 60%, and 65% against 70% reproduce exactly 12 intended differences each: seven mixture weights and five run-specific IDs or output paths. Shared settings include the model, optimizer, schedule, PALOMA cache, full evaluation population (`max_eval_batches=null`), and evaluation parallelism. All four targets have final step 11,490, non-temporary checkpoint metadata, seed 20260910, 128-sequence batches, and 3,012,296,704 training tokens. Raw file values agree with the archived final W&B summaries; the only numerical discrepancy in the wider five-row table is 2.22e-16 token loss at 50%, from decimal serialization.

The retained pilot and original refinement ZIPs were compared member-by-member using full bytes. Their only three changed files are unrelated to training or evaluation. All three interactive target child receipts name the same recovery bundle, `b66dcc0d2406e598a2ad0335bda2934ccf80da4c86e7f158e1bfdbee3990bcbe`. The archived extracted interactive evaluator matches its saved SHA-256. The evaluator diff preserves model-loss execution and token-loss accumulation; it changes BPB aggregation to total loss bits divided by total scored bytes. No mixture-fraction branch appears in that diff.

The existing interactive bundle inventory compares CRC32 and member size across the remaining library files; this audit did not freshly read every interactive bundle member. The runtime guard pins 27 selected files and four package versions. Its historical omission of `eval.py` allowed the original metric drift. Top-level W&B config values outside the four saved subtrees are not retained, even where their names appear in `config_keys`; the common data seed is supported by pinned launcher source. The guard and archived metadata are useful provenance, but they are not a complete runtime image attestation.

Archived restore logs identify the correct own-run checkpoints: 55% at `step-10759`, 60% at `step-6958`, and 65% at `step-2283`. Checkpoint generation/hash receipts report preserved objects through the migration. In the verified source, a checkpoint name denotes the last completed step: `StepInfo.step` is `state.step - 1`, and the restored data loader starts at `state.step`. The 60% source therefore implies continuation with batch 6,959. No reset of the data stream or schedule was found in the checked code. Checkpoint tensors, optimizer/RNG contents, and uninterrupted-equivalent resume trajectories were not replayed, so the audit does not rule out every possible resume or runtime effect.

Key evidence:

- [Canonical normalization and raw source manifest](../../consistent_bpb_20260912/metric_provenance.json); [archived target history](../../target_jump_audit_20260912/tpp10_target_p060_s20260910_eval_metrics.jsonl).
- [Current normalization implementation](../../../../../../experiments/domain_phase_mix/plot_starcoder_tpp10_refinement.py), especially `normalize_final_metrics` and `summarize`; [allocation implementation](../../../../../../experiments/domain_phase_mix/starcoder_tpp10.py), `sequence_allocation`.
- [Checkpoint completion receipts](../../completion_20260912/verified_endpoints.json); [pilot checkpoint metadata](../../pilot_results/target_checkpoint_validation.json).
- [Config comparison](../../target_jump_audit_20260912/config_wandb_target_diffs.json); [pilot/refinement ZIP comparison](../../target_jump_audit_20260912/config_bundle_comparison.json); [interactive bundle evidence](../../target_jump_audit_20260912/config_interactive_bundle_comparison.json).
- [Interactive evaluator](../../target_jump_audit_20260912/interactive_source/lib/levanter/src/levanter/eval.py), lines 559–624; [p60 child receipt](../../interactive_migration_20260912/child_449bcb9c_raw.json), line 87.
- [Restore logs](../../heartbeat_20260912_2237/checkpoint_logs.txt), lines 46–53; [p65 restore](../../heartbeat_20260912_2237/target_p65_restore.txt), line 5; [pre-migration checkpoint receipts](../../interactive_migration_20260912/pre_migration_checkpoints.json), line 71.

## Scope and reproduction

The 0.60 feature should remain an observed finite-panel deviation with an unresolved cause. One target seed cannot establish whether it reflects seed variation, the particular data order/support, a systematic response, or a runtime effect beyond the checked evidence. Removing the measurement, silently smoothing it away, or declaring it definitively training noise would exceed this audit. A fitted companion can retain every point and identify its curve as a descriptive fit.

Run `uv run .agents/projects/starcoder_tpp10/live/figure3_mariner_20260913/audit/audit_p060.py` from the Marin checkout to reproduce [audit.json](audit.json). The script uses the standard library, verifies archived hashes and identities, and writes only inside this audit directory. The separate [allocation recheck](allocation_recheck.json) records the local allocation computation. The code/config subaudit independently checked bundle files, retained configuration trees, and resume source. Fieldbook was inspected read-only before analysis; prior-work search through Echo returned HTTP 403. No external incident publication was attempted within this read-only audit scope.
