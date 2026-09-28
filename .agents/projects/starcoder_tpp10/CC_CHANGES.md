# TPP10 design and implementation handoff — 9 September 2026

The new experiment has been built locally. No GCS caches, CPU preparation jobs or TPU jobs have been created. The earlier 20-run pilot/refinement and its adverse matched-proxy selection result remain intact. The manuscript and outline have not been edited during this task.

## Scientific changes

- Match **total trainable parameter TPP** at approximately 10: 16,587,008 parameters / 165,937,152 tokens for both proxies; 301,241,344 / 3,012,296,704 for the target. Counts include tied embeddings once and Qwen3 Q/K normalization. Nonembedding TPP is disclosed as 19.766 versus 11.220.
- Bundle TinyLlama's 32k tokenizer at a pinned revision. No pretrained weights are used. Fresh bounded regional caches replace the previous Llama3-tokenized input only for this new experiment.
- Use a 190,316,544-token finite StarCoder parent for target and unmatched; each matched subset contains 10,485,760 tokens. Unmatched reaches at most 0.872 epochs, matched 15.825, target 15.828. Actual allocator mismatch is below 0.051% across the dense grid; web does not repeat.
- Construct the parent from equal token quotas across all 49 raw shards, globally permute packed sequences, and sample three independently seeded nested subsets. Keep membership fixed across fractions and trainer seeds. This mitigates shard-order bias; shard coverage does not establish representative content.
- Use proxy batch 32 (2,532 steps), target batch 128 (11,491 steps). Before a target run, an eight-run p=0.5 screen compares batches 32/128 at the same proxy horizon. Only the mean unmatched loss enters its 0.01-BPB rejection rule. Four runs are extra; four belong to the main grid.
- Pilot has seven common coordinates and 57 primary artifacts; dense has 21 and 183. Three subsets cross two proxy trainer seeds; target has one seed. The entire staged program has 187 distinct artifacts including calibration. Estimated dense primary training is 1.440e20 FLOPs, plus 9.964e16 for the four extra calibration runs; chip-hours await timing.
- Primary selection averages trainer seeds within each subset and scores the observed optimum on the common target grid. Report all three paired regret differences, their descriptive mean, and all individual seed selections. Pooling subset information is secondary. No favorable-result or interior-optimum gate exists.

## Implementation and checks

Four repository modules implement the frozen design/allocator audit, bounded preparation, resumable training and endpoint collection, and analysis/plots. Assets pin source generations and tokenizer files. Preparation receipts bind exact quotas, written token hashes and index identities. Regional preparation publishes an immutable audit URI. Training releases bind a concrete plan and current cache audit, and the child verifies code, assets and the locked runtime before entering Levanter.

The launcher now enforces the batch-calibration loss rule before target training, in addition to verifying all preceding-stage endpoints. Local tests exercise the actual cache writer/reader and native `train_sets`, rather than only comparing configuration keys. They reject incomplete runs, nonfinal metrics, stale runtime receipts and changed cache recipes. The native zero-weight audit covers p=0/.05, all three matched subset handles, and the surviving StarCoder stream at p=.05/1. Mixture interleaving is allowed to change with the mixture weights.

Local evidence: [allocator audit](allocation_audit.json), [parent order audit](parent_order_audit.json), [zero-weight stream audit](zero_weight_stream_audit.json), and [cache consolidation check](cache_consolidation_check.json). The synthetic plot check is clearly labeled and contains no training measurements. The final [validation record](validation.json) records 12 passing tests, passing repository lint and type checks, regional command validation, consistent manifests, and verified source-snapshot hashes.

## Review disposition

The two independent CC passes use Opus 5 through `claude -p`, with the API-key environment removed and the configured subscription account checked. Tools are limited to read-only repository access; neither review can submit or edit.

The [scientific review](review_science.md) and [follow-up](review_science_followup.md) motivated the batch screen, parent permutation, explicit per-subset selection, scale qualifications and removal of the vague promotion gate. Its remaining request for a durable native zero-weight-stream audit has now been implemented. [CC's closure](review_science_closure.md) reports no remaining offline scientific objection to regional preparation and calibration once authorized. Both index audits have been refreshed to the same frozen design. Two wording cautions remain in interpreting the review: observed overlap is **253, 260 and 252** sequences, and all-shard coverage does not prove composition bias has disappeared. The proposed extra arm and second evaluation domain are outside this three-curve experiment.

The [independent implementation review](review_implementation.md) and [closure](review_implementation_followup.md) report no remaining offline blocker. Its fixes and dispositions are:

- The manifest writer now refuses to replace an existing different design; a test verifies preserved bytes after attempted budget drift.
- Collection uses archived output paths, fingerprints and runtime receipts without rebuilding today's recipe or overwriting the saved plan. The real CLI is tested against historical design/code pins. Strict source/data/lock identity remains intentional for further training, and CC accepted that policy after the recovery fix. The selected-source archive is byte-checked against current pins; retain the full Iris workspace bundle at submission and record it in Fieldbook before the next stage.
- Actual tokenizer path resolution and vocabulary size are checked on both preparation/parent and TPU-child paths. The child check does not initialize a JAX backend.
- All remote wrappers explicitly receive the central1 prefix. The review's original claim that the prefix would not be inherited was not established: Iris inherits parent job environment variables, as verified in its client source. The explicit values make the requirement clearer without relaxing regional checks.
- Cache audits compare recipe and index digests with the frozen design. Raw and subset jobs reject the wrong design before processing. Training-data artifact paths are namespaced while runtime source names and the primary metric stay stable.
- Both proxy and target web allocations are checked; web budgets derive from the computed target horizon. Collection skips blank lines but rejects malformed records. Concurrency defaults are four preparation artifacts and eight training artifacts.
- The upstream consolidation-order concern was checked against Zephyr's ordered shard regrouping and ordered executor map, and the real local consolidation/native-cache-reader path passed. No library patch was needed.

The source pin intentionally changed while completing this pre-submission review; all eight final audit/plan files now reference design `72d14a845033b901d73397c74c724a668cbdec5fd72cec41178626b993f889a2`. The earlier reviewed manifest is retained only as review provenance. The scientific settings and numerical budgets did not change during the implementation fixes.

## Suggested outline entry

> Planned simulated-epoching experiment: a new three-curve StarCoder/web comparison at total-parameter TPP ≈10, using 16.6M-parameter proxies and a 301.2M target. The unmatched proxy does not repeat; three nested matched subsets reproduce the target's ≈15.83 maximum epochs. Primary evidence will be observed-grid target regret, with crossed proxy trainer/subset seeds and a single target seed. This is a separate setting from the completed fixed-model pilot, which favored the unmatched proxy. Regional preparation, calibration and target validation have not yet run. No result or curve-shape claim should enter the paper yet.

Fieldbook experiment: `exp_01m23ddmn78breygyvkkpzyrq8`. [The runbook](README.md) gives the staged commands and authorization boundaries. The next executable step, after release, is bounded regional CPU preparation and its cache audit.

## 9 September 2026: preparation and calibration submitted

The user released the experiment. The preparation parent `/calvinxu/starcoder-tpp10-prepare` was accepted at 20:16:31 UTC. It runs in central1-a and schedules all 12 reviewed cache artifacts, respecting dependencies. Evaluation preparation has completed; the StarCoder and web caches are progressing.

The calibration parent `/calvinxu/starcoder-tpp10-calibration` was accepted at 20:29:14 UTC. A small coordinator, `experiments/domain_phase_mix/await_starcoder_tpp10_calibration.py`, waits for successful preparation, verifies every finished cache plus the immutable audit, records the precise audit hash in the user-authorized calibration release, and calls the unchanged reviewed launcher for all eight runs. It releases no later stages.

CC reviewed the coordinator through subscription-safe read-only Opus. All three operational findings were addressed before submission: a 12-hour preparation wait inside the 48-hour parent timeout, no automatic parent failure/preemption retries, and durable calibration endpoint rows plus the batch-screen summary in GCS. See `live/coordinator_review.md` and `live/coordinator_review_disposition.md`. Manual recovery must inspect existing descendants first; only completed matching artifacts are automatically reused.

The original 27-file identity and calibration plan remain unchanged. The added coordinator passed repository lint, pyrefly, module import, plan-identity, and regional-command checks. Live TPU execution remains to be observed. Both full submitted workspace bundles are retained and linked in Fieldbook. The eight calibration datapoints are linked to the acknowledged parent. Exact commands, acknowledgments, source hashes, bundle receipts, and current release state are under `live/`. No manuscript or scientific design changes were made in this submission round.

## 9 September 2026: calibration completed

Preparation succeeded at 15:23:54 PDT and calibration at 16:18:03 PDT. All eight child runs succeeded. A fresh archived-plan collection verified all final metrics, fingerprints, and runtime receipts. Unmatched mean loss was 1.126364 BPB at batch 32 and 1.121061 at batch 128; the +0.005304 gap passes the prespecified 0.010 limit. Matched diagnostic means were 1.158346 and 1.184079. Retain batch 32. Once scheduled, child jobs took approximately 4–6 minutes including setup.

Results and the immutable release were downloaded to `live/calibration_results.json` and `live/calibration_release.json`; final metrics are also in `live/calibration_metrics.csv`. Fieldbook now marks all ten parent/child jobs and all eight planned training links successful. Optimizer-trace review and the three p=1 canaries remain the next stage. This status check submitted no new jobs and changes no scientific specification.

## 9 September 2026: optimizer review and p=1 canary release

The user authorized the optimizer-trace review and the three p=1 canaries. The [review](live/calibration_optimizer_review.md) covers all eight calibration runs, with 12,644 training-loss/LR records and 1,264 paired gradient/parameter-norm records from archived W&B histories. The scanned values are finite, loss decreases, transient gradient spikes recover, and both learning rates follow the frozen schedule. Batch 32 is retained under the original 0.010-BPB screen. The [diagnostic plot](live/calibration_optimizer_traces.png) was visually checked.

Iris accepted `/calvinxu/starcoder-tpp10-canary` at 18:17:38 PDT. It releases only the target, unmatched proxy and matched proxy at p=1, with all three selected children scheduled concurrently. Parent and children use central1-a; automatic parent failure/preemption retries are disabled. The frozen canary plan and all 27 source/asset pins are unchanged. The prior calibration workspace bundle was confirmed retained, and the new full workspace bundle is recorded in `live/canary_workspace_receipt.json` and Fieldbook.

Fieldbook contains the optimizer validation, release, three new datapoints, parent and training links. The release is `releases/canary.json`; the exact region-validated command is `live/submitted_canary.txt`. Current child state is recorded in `live/launch_state.json`. Pilot and dense-grid stages remain unreleased. No training-code, scientific-design, manuscript or outline changes were made; the new evidence supports optimizer feasibility at proxy p=0.5, not an epoch-matching result.

## 10 September 2026: canary recovery and parent-placement correction

The target and unmatched canaries completed with final BPB 0.820746 and 1.061356. Their frozen fingerprints, runtime receipts, final-step metrics and checkpoint metadata pass verification. The matched proxy never started: the original parent lost its worker at 06:24:03 UTC, and Iris killed the queued child. The underlying host failure is unconfirmed. The complete old job tree was terminal and the output lease inactive before recovery.

The first recovery reused both successes and selected exactly one pending artifact. Its 2-CPU/8-GB parent again landed on a preemptible TPU worker, while its child queued for worker memory. This request exceeds Iris's automatic non-preemptible coordinator heuristic (1 CPU/4 GiB). The queued recovery was canceled before training started and replaced by `/calvinxu/starcoder-tpp10-canary-recovery-ondemand-20260910`, accepted at 09:07:16 UTC with explicit `--no-preemptible`. The TPU child retains the frozen resources and runtime: the parent's preemptibility constraint is not inherited. No cluster changes or training-code changes were made.

The two future command templates and runbook now require explicit non-preemptible parents and disable automatic parent retries. Actual submitted-command receipts remain intact. Fieldbook tracks both recovery attempts and their lineage; `live/launch_state.json` gives the latest state. The new command selects only the missing matched canary, preserving the original three-run release and all 27 source/asset pins.

The completed canaries' loss, gradient and LR traces were checked and plotted in `live/canary_partial_optimizer_traces.png`. The local history reader now accepts identical repeated W&B export rows while rejecting conflicting values at a repeated step; raw histories are preserved. These repeats occur at multiples of 1,000 steps in this export. This analysis-only correction changes no training identity or endpoint. Pilot and dense sweeps remain unreleased.

Echo incident publication was attempted but returned HTTP 403. Recovery evidence and the access failure are retained in Fieldbook; the temporary incident draft is outside the repository.

## 10 September 2026: all canaries completed and reviewed

The on-demand recovery finished at 05:25:56 PDT, completing the missing matched proxy while reusing the target and unmatched successes. All three final endpoints pass a fresh archived-plan collection: target 0.820746 BPB, unmatched 1.061356, matched 1.314493. The matched training trace is stable, its logged loss and norms remain finite, and both learning rates follow the frozen schedule. The full [canary review](live/canary_optimizer_review.md), [endpoint table](live/canary_metrics.csv) and [visually checked trace plot](live/canary_optimizer_traces.png) are archived.

The calibration and canary gates are complete. The next stage is the seven-coordinate pilot, which remains unreleased along with the dense sweep. No additional jobs or scientific changes were made during this status check. These p=1 measurements establish feasibility; the curve-shape and selection comparisons require the sweep. Fieldbook and the launch-state record have been updated; there is no manuscript or outline result to propagate yet.

## 10 September 2026: seven-coordinate pilot released and submitted

The user released the pilot after the canary review. Iris accepted `/calvinxu/starcoder-tpp10-pilot` at 15:53:05 PDT. The frozen grid is p = 0, 0.1, 0.3, 0.5, 0.7, 0.9 and 1.0, with 57 cumulative primary artifacts. Seven successful artifacts are reused; the remaining 50 comprise six target, 11 unmatched-proxy and 33 matched-proxy runs. Submission concurrency is 50, leaving accelerator scheduling to Iris.

Fresh preflight checks reproduce the reviewed pilot manifest, all 27 source pins and the completed cache audit. All calibration and canary endpoints verify again, the batch screen still passes, and the full prior workspace bundle remains retained. The new release is [releases/pilot.json](releases/pilot.json); the exact region-validated command is [live/submitted_pilot.txt](live/submitted_pilot.txt). The CPU parent explicitly requests non-preemptible central1-a placement and disables automatic failure and preemption retries. TPU settings, data membership and output fingerprints are unchanged.

Fieldbook records all 57 datapoints and links the 50 new runs to the acknowledged parent. [Live launch state](live/launch_state.json) records subsequent progress and the workspace receipt. The dense stage remains unreleased. No manuscript or outline result has changed; the pilot must finish before the common-grid selection analysis.

## 10 September 2026 evening: proxy curves complete; target sweep running

A fresh check verifies 51 of 57 pilot endpoints, including all 50 proxy runs and the reused p=1 target. The 44 newly submitted proxy jobs have succeeded; the six new target jobs are running with no reported failures or remaining queue. Each completed endpoint passes the SUCCESS, archived fingerprint, exact code/runtime receipt, finite final-step BPB and final checkpoint metadata checks.

On the seven-point grid, both unmatched trainer seeds select p=1.0. Each of the three matched subsets selects p=0.5, with both trainer seeds agreeing within every subset. The mean minimum losses are 1.061733 BPB for unmatched and 1.158346, 1.161024 and 1.156841 for the matched subsets. These are proxy-only selections; target regret and transfer benefit remain unmeasured until the target sweep finishes.

The [completed endpoint table](live/pilot_partial_metrics.csv), [proxy summaries](live/pilot_proxy_results.json) and [visually checked curves](live/pilot_proxy_curves.png) are archived. At 20:05 PDT, the six target runs were 38–92% through their training updates; the slowest had approximately 89 minutes left at its run-average rate. That estimate assumes uninterrupted throughput. Fieldbook carries the latest job states, verified measurements and remaining analysis action. No additional jobs, training changes, manuscript edits or dense release were made.

## 10 September 2026 night: complete pilot and target selection results

The pilot parent succeeded at 21:34:02 PDT, with all 50 new child jobs successful and seven prior artifacts reused. Collection against the archived pilot plan verifies all 57 endpoints, including exact fingerprints, source/runtime receipts and finite final-step BPB. All seven target final checkpoint metadata records also have step 11490 and are permanent. The frozen plan and training settings are unchanged.

The target's observed grid minimum is p=0.7 at 0.784388 BPB. Both unmatched trainer seeds select p=1.0, whose target loss is 0.820746 and target regret is 0.036357 BPB. Each matched subset selects p=0.5, with both trainer seeds agreeing; its target loss is 0.794241 and regret is 0.009852. Epoch matching therefore improves the selected target loss by 0.026505 BPB and reduces regret by about 73% in this pilot. It predicts an interior optimum but places it earlier than the target and overstates the high-repetition penalty. Excess-curve RMSE is 0.083–0.088 BPB for the matched subsets versus 0.129 for unmatched; rank correlations are 0.786–0.857 versus 0.679.

These are observed minima on the seven-point common grid, conditional on one target trainer seed and one finite parent corpus. The three subset comparisons share the target and unmatched curves; they do not provide three independent target replications. The [report](live/pilot_results/report.md), [machine-readable analysis](live/pilot_results/analysis.json), [full endpoint CSV](live/pilot_metrics.csv) and [visually checked plot](live/pilot_results/curves.png) preserve the measurements and scope. Fieldbook records completion and the analysis. The dense stage awaits review and release; no manuscript, outline or training-code edits were made.

## 10 September 2026 late evening: descriptive MARINER fits

At the user's request, the native MARINER registry entry was fitted to each completed pilot curve. Fitted continuous minima are 62.82% StarCoder for the target, 100% for unmatched, and 55.57% for the pooled matched curve. The three matched subset means individually select 54.53%, 55.28% and 50.85%. All fourteen fits, including individual-seed diagnostics, are in the [fit report](live/pilot_results/mariner_fit/README.md) and [JSON](live/pilot_results/mariner_fit/summary.json); the [plot](live/pilot_results/mariner_fit/mariner_fits.png) was visually checked.

The fit uses seven distinct coordinates, trainer means within subsets, leave-one-mixture-out tuning, physical nominal epochs and the existing StarCoder training-median floor fallback because this pilot lacks a proportional calibration run. These are descriptive predictions. In particular, the target fit misses the measured 70% loss by +0.00769 BPB, comparable to the measured 50%-to-70% gap. The exact continuous locations therefore need new measurements. The original observed-grid selections and regrets remain the primary pilot analysis.

The new offline script `experiments/domain_phase_mix/exploratory/two_phase_many/fit_starcoder_tpp10_mariner_20260911.py` reuses the registry and head implementation, records source hashes/runtime versions and passes targeted repository lint and type checks. No model implementation, training recipe, release, manuscript or outline changed.

## 11 September 2026: five-fraction refinement and measured figure in paper

The user released 40%, 55%, 60%, 65%, and 80% after reviewing the pilot and descriptive MARINER fits. Iris accepted `/calvinxu/starcoder-tpp10-refinement` at 00:08:42 PDT. It selects 45 unchanged recipes from the frozen dense grid: five target, ten unmatched and thirty matched runs (3.431586854e19 additional estimated training FLOPs). The remaining dense coordinates stay unreleased. This is adaptive refinement using the original trainer seeds, subsets, model/token pairs and optimizer.

`launch_starcoder_tpp10_refinement.py` is a parent-only selection wrapper; it changes none of the 27 frozen pins or child fingerprints. It checks the pilot/dense plan hashes, requires the complete verified pilot, and delegates cache/runtime/allocation validation, successful-artifact reuse, execution and collection to the original launcher. The exact regional command and bundle passed preflight; the parent is explicitly nonpreemptible in central1-a with zero automatic retries and all 45 selected jobs submitted concurrently. Repository lint and pyrefly pass for the new wrapper. `live/refinement_plan.json`, `releases/refinement.json` and `live/submitted_refinement.txt` record the release.

Figure 5 on page 7 of the paper now contains the pilot's raw and excess-loss panels with measured points/segments and observed-minimum stars, without fitted curves. Section 4.2 reports the grid selections and target regret. Appendix B.1 gives the TPP10 protocol and preserves the earlier unfavorable fixed-model comparison. The outline and manuscript revision log are synchronized; the static-cap experiment remains a placeholder. The paper builds cleanly (43 pages, references still on page 10), and affected pages were visually checked. Full manuscript changes: `revision_notes/20260911_tpp10_measured_curves/CHANGES.md` in the paper directory. No push was made.

After completion, collect only the 45 refinement endpoints against their archived plan, then explicitly extend the analysis common grid to the twelve coordinates before merging with the 57 verified pilot artifacts. Do not bypass per-plan metric verification or use the complete dense-stage label. The paper still shows only the seven completed pilot coordinates.

## 11 September 2026: Uncheatable companion to Figure 5

The user requested the existing three curves under a common evaluation. The 57 seven-grid pilot checkpoints have only PALOMA programming-language scores, so a post-hoc evaluation was submitted as `/calvinxu/starcoder-tpp10-uncheatable`. It restores the frozen final checkpoints and evaluates the same seven Uncheatable components used in the paper, with equal component weights. The frozen TinyLlama tokenizer is applied to the existing central1 raw evaluation objects; no training data, mixtures, horizons, seeds, subsets, or training jobs change. Every restored checkpoint must reproduce its recorded PALOMA endpoint within 0.00005 BPB before its Uncheatable result is accepted. Two model-size batches reuse compilation and immutable per-checkpoint results. The original figure and manuscript remain unchanged.

The specification, exact command, and preflight are in `live/uncheatable_spec.json`, `live/submitted_uncheatable.txt`, and `live/uncheatable_preflight.json`. The new evaluator/collector is `experiments/domain_phase_mix/evaluate_starcoder_tpp10_uncheatable.py`; the original plot helper now accepts an optional y-axis label while keeping its prior default. All 14 TPP10 tests, repository lint, and evaluator type checking pass. Fieldbook links the evaluation job to the 57 existing datapoints. The scientific result and companion plot await evaluation.

### Evaluation runtime follow-up

The active Uncheatable evaluation is /calvinxu/starcoder-tpp10-uncheatable-ram32. Both v5p-8 children are queued for occupied TPUs; the coordinator is running, and all seven central1 caches are finished (7,810,444 tokens before per-component packing). No Uncheatable checkpoint result exists yet. The final frozen specification is live/uncheatable_spec_ram32.json, hash 9417770a83fe5a1b30811e6f3c10cc91295e8d984df306a65a128de876312dca. It keeps all 57 Figure 5 pilot checkpoints, seven component scores and their equal mean, original trainer/subset averaging, and the PALOMA restoration control.

The first coordinator was canceled before assignment to use the supported two-core CPU class. The second completed caches but both TPU children stayed queued behind the 224-GiB default host-memory request. That tree was canceled before evaluation; the final job reuses its caches with explicit 8 CPU/32 GiB per v5p-8 evaluator. Cache preparation now writes each one-file set locally without distributed merge workers. Parent-tag numpy metrics are converted to Python scalars for JSON receipts. No training run was changed or duplicated.

The final preflight and handoff are `live/uncheatable_ram32_preflight.json` and `live/uncheatable_handoff.md`. Fourteen TPP10 tests and repository lint pass; the evaluator type check has zero errors. The paper figure and outline have not been changed.

## 11 September 2026: Wikipedia and FineMath setup reviewed

The proposed domain extension received a read-only CC Opus 5 review and a final scope check. [The specification](domain_extension_review/SPEC.md), [final review](domain_extension_review/cc_target_scope.md), and [disposition](domain_extension_review/CC_DISPOSITION.md) preserve the settings, evidence and decisions. CC found no scientific-setting blockers for an exploratory survey of measured mixture minima. Source metadata, model/token counts, epoch allocation and compute estimates were independently checked. The prose now describes conditional mixture preferences and reports epochs at the selected fraction.

The user excluded unmatched controls and multiple matched sweeps, then requested initial estimates in both target and matched settings and inclusion of Uncheatable. The plan now uses one target curve and one matched-proxy curve for each of Wikipedia and FineMath-3+, one trainer seed, and one matched subset. Each new curve uses p = 0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70 and 1.00. Reusing each scale's web-only p=0 checkpoint leaves 14 target and 14 matched-proxy runs, estimated at 9.36434e19 training FLOPs. Existing StarCoder curves are reused on their measured grid; no StarCoder points, unmatched arms, replicate subsets or automatic refinement are added.

Uncheatable's fixed seven-component mean and all component scores are required at every endpoint of both scales, alongside existing PALOMA validation. A common four-component sensitivity reuses these scores. Report observed grid minima and neighboring losses, target loss at the proxy choice, and target-grid regret; disclose different grid resolution for StarCoder. The first survey need not establish significance or precisely locate a shallow continuous minimum.

This round submitted no preparation, training or evaluation jobs and did not change manuscript or training code. The new regional builders, bounded FineMath parquet reader, domain launcher and analyzer remain to be implemented and validated. FineMath needs a fresh TinyLlama-tokenized cache identity because the existing dataset helper pins a legacy cache. Keep frozen StarCoder modules/assets/lock untouched, verify p=0 stream reuse and nonzero allocation/shuffle parity, and include immutable cache and evaluation receipts before submission. Fieldbook records this reviewed scope; the earlier unmatched-controls recommendation is retired.

## 12 September, 22:39 UTC: 43/45 refinements verified

The interactive replacement has three new verified completions: target p55 (0.7701349258422852 BPB), target p60 (0.7694385051727295 BPB), and matched p80 with trainer 20260911/subset 20260914 (1.1448901891708374 BPB). Each passed exact frozen fingerprint, permanent final checkpoint, frozen child runtime, finite final PALOMA metric and finished W&B parity. The previously verified 40 outputs remain reused.

Only target p65 and p80 remain. Their temporary checkpoints advanced from steps 2283 to 7112 and 728 to 5572 of 11491. All four targets' logs show restoration from the saved full-state checkpoint paths. Current parent is `/calvinxu/starcoder-tpp10-refinement-interactive`; it is running without failures or preemptions. Its actual start is 12 September 21:34:54.158 UTC, so the 172800-second deadline is 14 September 21:34:54.158 UTC. No recovery, new launch or priority change was needed.

Evidence: `live/heartbeat_20260912_2237/remaining_artifact_audit.json`, `progress_receipt.json` and filtered runtime logs. Fieldbook records the three final BPB values and their validations plus current child states. Keep the existing heartbeat active. After the last two pass, use the complete-45 collector and 12-coordinate common-grid procedure in `live/completion_20260912/README.md`, then update the approved plot, caption/outline facts and compile/visually verify the paper. No partial-grid plot update was made in this check. FineMath's already completed work was not recollected.

## 12 September, 23:34 UTC: 44/45 refinements verified

Target p65 completed at final step 11490 with PALOMA programming-languages BPB 0.7665561437606812. Its frozen fingerprint, permanent checkpoint metadata, original runtime and finished W&B metric parity all pass. The new result and checks are recorded in Fieldbook. Evidence: live/heartbeat_20260912_2333/remaining_artifact_audit.json.

Only target p80 remains under /calvinxu/starcoder-tpp10-refinement-interactive/verified_training-869fe9a0. It is running normally; the latest committed checkpoint was step 9697 at 23:29:42 UTC and live training reached approximately 10100 by 23:33:49 UTC. No final metadata exists yet. No intervention, recollection of completed work, or plot update was made. Keep the interactive parent and heartbeat active. Its execution deadline remains 14 September 21:34:54.158 UTC.

After p80 passes, run the all-45 collector and complete 12-coordinate analysis from the 57 pilot plus 45 refinement identities. The approved plot, caption/outline facts, PDF build/visual checks and CC/Fieldbook records are still required before pausing the heartbeat. Follow live/completion_20260912/README.md; preserve the frozen plans and averaging. FineMath is already complete and should not be recollected.

## 12 September: all 45 refinements and final plot complete

All 45 refinement runs are verified complete. The interactive parent `/calvinxu/starcoder-tpp10-refinement-interactive` and all five replacement children succeeded without failures or preemptions. The original 40 successes were reused; the four interrupted targets restored their saved full training state. No new recipes or additional training were submitted during finalization.

The authoritative collector checks all 45 refinement fingerprints, permanent final checkpoints, frozen runtime receipts, finite final-step PALOMA metrics and finished W&B parity. Combined with the verified 57 initial measurements, it accounts for all 102 distinct training artifacts on the twelve-point grid `[0,10,30,40,50,55,60,65,70,80,90,100]`. Every point has its original seed/subset coverage; missing and verified-but-unplotted lists are empty.

The unmatched mean selects 100% StarCoder. Each of the three independently drawn matched subsets selects 50% after averaging its two trainer seeds. The target's observed minimum moves from 70% on the initial grid to 65% on the complete grid, at 0.7665561438 BPB. Target regret is 0.0541894436 BPB without simulated epoching and 0.0276846290 BPB with it, a 48.9114% reduction. The absolute target-loss advantage remains 0.0265048146 BPB; only the lower reference minimum changes the percentage from the old 72.9%. Individual matched trainer-seed selections are retained separately in the analysis and are not all 50%; the primary estimator remains the prespecified two-seed mean per subset.

The figure retains the approved two-panel layout, absolute  BPB axes, categorical colors, measured points and connecting segments, stacked brackets and actual epoch labels. The legend now says three subsets; the right-axis limits accommodate the lower target minimum and both full brackets. Colored annotations are +3.6116% and +7.0692% relative to the target minimum; the gray bracket is 48.9114% of unmatched excess avoided. No fitted curve is overlaid. This plot is currently Figure 3, on PDF page 5.

Updated `sections/simulated_epoching.tex`, Appendix B.1 and current outline facts/caption. The main prose rounds the regret reduction to 49%; the caption reports 48.9%. The incomplete-refinement placeholders are removed. The appendix defines selection and regret on the combined twelve-point grid and retains the adaptive-refinement and single-target-seed limitations. Clearly dated initial-grid history remains in the outline. The header figure and unrelated prose/figures were not edited.

Analysis code changes: `analyze_starcoder_tpp10.py` shares its existing estimator with the refinement collector; `plot_starcoder_tpp10_refinement.py` exposes `complete_common_grid_analysis` only for the verified 102-artifact complete grid. Both frozen plan hashes and per-subset statistics are retained. Pilot/dense outputs match their pre-edit values exactly. Five focused behavioral tests and required targeted lint passed. A separate Python-standard-library recomputation from the two metric CSVs confirms all means, selected fractions, regrets and plotted percentages.

Visual checks passed for the standalone figure and compiled pages 5,16,17. A cold reader correctly inferred the proxy-selection/target-regret argument and all percentage meanings; the subset-count legend addresses its curve-count ambiguity. The PDF remains 43 pages, references start on page 10, and the build has no undefined references/citations, warnings or overfull boxes. No commit or push was performed.

The `data/` directory retains the complete analysis, verified refinement snapshot, both original plans, both metric CSVs, per-curve plotted points, allocation audit and figure receipt. Original paper sources, figure and PDF are in `before/`. Canonical experiment records are in `.agents/projects/starcoder_tpp10/live/completion_20260912/` in the Marin repository. Fieldbook experiment: `exp_01m23ddmn78breygyvkkpzyrq8`.


## 12 September: target-curve metric schema audit supersedes the complete-grid figure

The published jump at target p70 is caused by mixed BPB definitions: 97 legacy endpoints and five schema2 endpoints from the interactive recovery. The recovery bundle includes the corrected evaluator, but eval.py was omitted from training code pins and collectors did not check metric schema. Original training settings and saved token losses remain comparable. Consistent reconstruction from the audited PALOMA token/byte counts gives target minimum70%, matched selection50%, unmatched100%; target regret reduction71.1795% (replacing48.9%). The current paper figure and its65% minimum/48.9% claims require correction; this audit has not edited them. No retraining is required for metric normalization. Full evidence and standalone diagnostic: live/target_jump_audit_20260912/AUDIT.md and target_metric_schema_audit.png. All102 raw final metrics, actual recovery-bundle evaluator, prior live checkpoint validation and independent calculations were checked.


## 12 September: consistent BPB correction applied

# CC handoff: consistent BPB in the StarCoder figure

The target-curve jump was caused by combining 97 legacy BPB endpoints with five corrected endpoints after interactive recovery. All 102 plotted endpoints now use total loss bits divided by total scored bytes, reconstructed from the saved final token-average losses and the audited PALOMA population. Original run artifacts, raw CSVs, the verified endpoint snapshot and frozen training plans are preserved.

The complete twelve-point grid now has its target minimum at 70% StarCoder (0.7655645236 BPB). Each of the three matched subset means still selects 50%; unmatched selects 100%. Target losses at those selections are 0.7756673393 and 0.8006188044 BPB. Excess losses are 0.0101028157 and 0.0350542808 BPB, so simulated epoching avoids 71.1795% of unmatched selection regret. Figure brackets show +1.32%, +4.58%, and 71.2% less excess loss. One individual matched replicate changes its selected fraction from 80% to 55%; the corresponding two-seed subset mean still selects 50%.

The figure retains its approved two-panel layout, colors, measured points and segments, epoch annotations, and absolute BPB axes. Both proxy and target curves are corrected; the proxy limits expand downward to retain the corrected endpoint. No fitted curve was added. The main paragraph reports 71%, the caption 71.2%, and Appendix B.1 defines BPB as total prediction loss bits divided by total scored bytes. Current outline facts and caption agree; earlier incorrect values are explicitly marked superseded. The header figure and unrelated plots are untouched.

The canonical refinement plotter now requires the audited metric manifest and pinned population counts. It verifies all raw JSONL hashes, run identities, final steps, original reported BPB, finite token loss, known schema and agreement of schema 2 values with reconstruction. Its output records the common metric definition and preserves each raw value and schema separately in metric_provenance.json. Both plotters refuse unnormalized input. The paper builder was tested against the original mixed analysis and rejects it before rendering. Frozen training launchers and their original code pins were not changed.

All 60 plotted curve means and five selections agree with the independent audit to within 1e-12. The 12 focused regression tests and required targeted lint pass. The standalone main figure, diagnostic curves and compiled pages 5,16,17 were visually checked; an independent image-only reader understood the selection/regret comparison and found no overlap or clipping. The paper remains 43 pages, references begin on page 10, and there are no warnings, undefined references or overfull boxes.

Canonical corrected output is .agents/projects/starcoder_tpp10/live/consistent_bpb_20260912/ in the Marin repository. Its README carries the offline regeneration command. The paper input is revision_notes/20260912_outline_figures/data/epoch_matching_analysis.json. This revision's data/ retains the normalized analysis, per-run metric provenance, plotted points, exact population counts, independent audit and final figure receipt. before/ retains the previous paper inputs and outputs. Fieldbook experiment exp_01m23ddmn78breygyvkkpzyrq8 records the correction. No new training, checkpoint evaluation, commit or push was performed; the completed monitor stays paused.
