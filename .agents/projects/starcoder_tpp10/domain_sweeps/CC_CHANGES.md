# Changes for CC and the outline — 11 September 2026

- Implemented and submitted the reviewed 28-run Wikipedia/FineMath target and matched-proxy survey. Scientific scope is unchanged from SPEC.md: one seed and subset, common grid, shared p=0, both measured scales, Uncheatable components and mean plus PALOMA. No unmatched controls, extra replicates, or new StarCoder runs.
- New regional builders preserve the original tokenizer, exact190.32M/10.49M-token geometry, packed-sequence permutation/subset indices, internal focus source alias and web ordering. FineMath reads only pinned 3+ parquet objects, with byte and row-group memory limits; resumed parts require matching receipts.
- Added pre-training checks for identical evaluation populations/caches, actual finite tags, source/code pins and allocation. Two included matched p=100 points gate the remaining 26. Collection rejects incomplete checkpoints or missing/conflicting component metrics.
- CC's implementation follow-up cleared submission after the evaluation and memory checks. The initial suggestion to require non-preemptible TPU workers was withdrawn; the available central1 v5p pool is preemptible. CPU coordinator/preparation placement is explicitly on-demand; new survey priority is interactive.18 relevant tests, lint and Pyrefly pass.
- Per the user's latest instruction, moved only the unfinished StarCoder 45-point refinement to batch. The old tree was stopped after a saved checkpoint was verified; all 45 fingerprints, seeds, subsets, and outputs remain identical. 11 completed artifacts are reused; 34 unfinished points remain. Target p=55 resumes its latest checkpoint. Both parent IDs and exact commands are in HANDOFF.md.
- Fieldbook records the 28 new datapoints, submissions, review/release/bundle artifacts and migration lineage. A 15-minute thread monitor tracks actionable changes and recovery. No manuscript, numerical table, figure, outline, frozen StarCoder file, or uv.lock was changed. Source token yield and new-run performance await regional execution.

## Coordinator recovery at 11:18 UTC

The first all-batch replacement coordinator was scheduler-preempted by the interactive Wikipedia preparation job before dispatching training. It is superseded. Current refinement parent: `/calvinxu/starcoder-tpp10-refinement-batch-recovery`, submitted 11:18:27 UTC, Fieldbook job `job_01m28317ycr6k3vmhzm5xksec6`. The CPU coordinator is interactive; every TPU child is explicitly BATCH. This avoids sacrificing the entire sweep when individual batch training should yield capacity.

The new `resume_starcoder_tpp10_refinement_batch.py` wrapper changes only `JobRequest.priority` at dispatch via the native current-client context. It calls the frozen refinement launcher unchanged. The wrapper is separately pinned to SHA-256 `b99a6277374bb0da97a1df91b0fd7dad5a7f8f1004e33e224c5cf4bf571b1e84`; recovery bundle `3655dd72d95b64c91b0b39341b27308cead8f5a24402128694032f1390067246` verifies the original 28 pins plus the wrapper. CC found no blocker; lint and Pyrefly passed. All dependencies were already verified successful, so no CPU prep job should reach the wrapper's TPU-only dispatch guard. Region, parent resources, zero automatic retries, 45 scientific fingerprints and checkpoint paths are unchanged. Verify actual child priority from controller receipts once dispatch begins.

Use `submitted_refinement_batch_recovery.txt` for any needed recovery, with a new documented parent attempt only after this tree is terminal and leases expired. Do not restart either superseded coordinator. Both original and first batch parents are terminal. Eleven completed training artifacts remain retained; the unfinished p=55 target can resume its saved checkpoint. The monitor now targets this recovery parent and the new domain survey.

## Monitor checkpoint, 11:38–11:43 UTC

Wikipedia raw tokenization completed at 11:38 UTC. Independent checks confirm the expected fingerprint 5e2ac160, frozen recipe/source generations and tokenizer metadata, a finished cache ledger, and exactly 190,316,544 tokens (95,158,272 from each source shard). Parent permutation and matched subset remain to be materialized. FineMath raw preparation started and initialized JAX 0.11.1. The refinement recovery coordinator still waits for CPU capacity; all 11 previous successful training fingerprints remain verified, with 34 unfinished. No new-domain training has completed. Zephyr cache-probe children shut down normally after successful probes; no repair or resubmission is needed. Evidence: domain_sweeps/monitor_20260911T1138/.

## Monitor checkpoint, 11:59–12:04 UTC

FineMath did not hang: Iris reports the job RUNNING, but its sole task is PENDING after TaskRetryScheduled at 11:42:49 UTC, preempted by the Wikipedia parent-materialization job. Automatic retry remains queued for CPU capacity; no duplicate submission or manual restart was performed. Independently verified three finished FineMath parts against their exact recipes/source generations, tokenizer metadata and completed cache ledgers: 11,894,784 tokens each, 35,684,352 total. The fourth part is incomplete and will be rebuilt by the existing resumable writer. Wikipedia parent construction is active and wrote fresh cache objects at 12:00 UTC. The refinement recovery coordinator is still CPU-capacity queued. No new-domain training has started. Echo incident publication is unavailable: the required prior-work search returns HTTP403; this monitoring checkpoint records the evidence in Fieldbook without changing cluster state.

## 11 September, 12:26 UTC: Wikipedia caches and FineMath raw pool verified

Wikipedia parent permutation and matched subset completed. Independent checks confirm 190,316,544 parent tokens and 10,485,760 matched tokens, frozen index hashes, exact recipe and design identities, tokenizer metadata, completed cache ledgers and artifact fingerprints. FineMath automatically resumed after its scheduler preemption and completed all 16 raw parts (190,316,544 tokens); each part receipt matches its pinned source generation and quota. Its parent materialization is running. No new-domain training has started, and no manual restart or change to the experiment was needed. Refinement remains CPU-capacity queued with its 11 retained successes; actual child BATCH priority remains to be checked after dispatch. Evidence: monitor_20260911T1220/cache_validation.json and status.json.

## 11 September, 12:48 UTC: batch dispatch verified; FineMath parent ready

The refinement coordinator started and released only the 34 unfinished recipes. Every child controller request has BATCH priority, the reviewed bundle and central1-a placement. Each serialized callable hash matches its original training job, allowing an exact mapping to the original Fieldbook run; the resulting set is exactly the 34 unfinished plan rows. All 11 successful artifacts remain valid and were skipped. These children are waiting for TPU capacity. FineMath parent preparation completed and independently passed the token-count, recipe, tokenizer, fingerprint and permutation checks; its matched subset is waiting for CPU capacity. New-domain training has not started. No manual resubmission, scientific change or cluster change was made. Evidence: monitor_20260911T1243/.

## 11 September, 13:08 UTC: target p=55 resumes its saved checkpoint

The first batch refinement child acquired a TPU. Runtime logs confirm a 2.37-GiB checkpoint restore from saved step 3010, resumption at step 3011, and completion of that step at 13:07:24 UTC. Its original W&B run is reused. W&B had logged through step 3402 before the old parent was stopped; it suppresses repeated step logging while the trainer catches up from the last saved checkpoint. This is expected and does not require changing run IDs or discarding the checkpoint. Current refinement coverage: 11 complete recipes, one training, 33 awaiting TPU capacity. FineMath matched-subset preparation remains CPU-capacity queued. No new-domain training result, code change, or manual resubmission. Evidence: monitor_20260911T1305/refinement_resume_validation.json and runtime logs.

## 11 September, 13:29 UTC: three p=55 endpoints verified

Refinement coverage is 14/45 complete. New verified endpoints: tpp10_matched_p055_s20260910_d20260913: 1.163977 BPB; tpp10_matched_p055_s20260910_d20260914: 1.165215 BPB; tpp10_unmatched_p055_s20260911: 1.114053 BPB. Each has its permanent final step-2531 checkpoint, exact frozen runtime and artifact fingerprint, consistent finite final-step PALOMA record, and an identical finished W&B summary. Four other batch children were scheduler-preempted and automatically requeued at 13:20:07 UTC; no manual retry is needed. Iris reports their jobs RUNNING but their tasks PENDING. The p=55 target saved checkpoint 3748 before preemption; normal checkpoint retention replaced 3010. The remaining 31 recipes await TPU capacity. FineMath matched-subset preparation is still CPU-capacity queued; no new-domain training has started. These are partial measurements, with no new optimum claim. Evidence: monitor_20260911T1325/.

## 11 September, 13:50 UTC: refinement reaches 16/45

Two additional matched-proxy p=55 endpoints passed all completion checks: tpp10_matched_p055_s20260911_d20260912: 1.164136 BPB; tpp10_matched_p055_s20260911_d20260914: 1.153962 BPB. Verified permanent final step2531, frozen runtime and fingerprints, consistent final PALOMA records and matching finished W&B summaries. Three tasks are training with fresh logs and zero failures: targets p=55 and p=60, and the remaining trainer20260911/subset20260913 matched p=55 run. The other26 await TPU capacity. Native retries preserve the target checkpoint and original run identities. FineMath matched-subset preparation remains CPU-capacity queued; no new-domain training has started. No manual retry or scientific change. Evidence: monitor_20260911T1346/.

## 11 September, 14:10 UTC: refinement reaches 19/45

Three additional endpoints passed permanent final-checkpoint, frozen runtime/fingerprint, final PALOMA and finished W&B checks: tpp10_matched_p055_s20260911_d20260913: 1.159597 BPB; tpp10_unmatched_p060_s20260910: 1.117194 BPB; tpp10_matched_p060_s20260910_d20260914: 1.158234 BPB. Targets p=55 and p=60 are training; saved checkpoints 4482 and 716 are independently verified. Two proxy tasks were automatically requeued after preemption. Current coverage: 19 complete, 2 training, 24 awaiting TPU capacity. FineMath matched-subset preparation still awaits CPU capacity; no new-domain training has started. No manual resubmission or scientific change. Evidence: monitor_20260911T1406/.

## 11 September, 15:08 UTC: refinement reaches 32/45

Thirteen more p=60/p=65 proxy endpoints passed permanent final-checkpoint, exact frozen runtime and artifact-fingerprint, consistent finite final PALOMA, and matching finished W&B checks. Their values and checkpoint hashes are recorded in monitor_20260911T1503/new_refinement_endpoints.json and Fieldbook. Current coverage is 32 complete, eight training and five awaiting TPU capacity. All four remaining target coordinates are now training; p=65 and p=80 are in startup. FineMath matched-subset preparation still awaits CPU capacity, with no new-domain training yet. No code/scientific change or manual resubmission. No new optimum claim from this partial batch.


## 11 September, 15:33 UTC — refinement reaches 40/45

Eight additional proxy endpoints passed final checkpoint, artifact fingerprint, frozen runtime receipt, finite final-step PALOMA and matching finished W&B checks. Values and checkpoint hashes are in `monitor_20260911T1527/new_refinement_endpoints.json`. The remaining four targets and one matched proxy are queued after native preemption, with zero runtime failures. Target temporary checkpoints were verified at steps9216/5406/728/728 for p55/p60/p65/p80; no temporary checkpoint exists for the short remaining proxy. Both CPU coordinators remain live. FineMath matched-subset prep continues to await CPU capacity, so no new-domain training has begun. Fieldbook records the eight completions and current per-task queue states. No manuscript, scientific settings or submission changes.


## 11 September, figure refinement and CPU reservation diagnosis

Figure 5 now includes the completed StarCoder refinement. The original colors, straight segments and observed-minimum stars are preserved; no fitted curve is overlaid.

- Reverified 40/45 refinement endpoints against the archived plan: artifact fingerprints, exact child runtime receipts, final permanent checkpoint metadata, final PALOMA BPB and finished W&B summaries. Original pilot metrics are unchanged. The verified snapshot lists all 45 requests, including the five incomplete runs.
- The figure uses 96 distinct completed artifacts (57 pilot + 39 refinement). One completed matched run at 80% is retained in the data but withheld from the plot because its second trainer seed is incomplete. Each plotted proxy point still averages both seeds. Target coverage is eight coordinates; unmatched twelve; matched subsets twelve/twelve/eleven.
- Observed minima remain target 70%, unmatched 100%, and all matched subsets 50%. The quoted 0.0364 versus 0.0099 BPB target regrets remain explicitly tied to the original common seven-point grid. Four target refinements at 55/60/65/80% must finish before reporting refined common-grid regret.
- The Figure 5 caption and Appendix B.1 use `\placeholder{}` for missing results. Appendix B.1 records that refinement coordinates were chosen after looking at the pilot and descriptive fits. The governing outline and figure-caption inventory match these edits.
- Added `experiments/domain_phase_mix/plot_starcoder_tpp10_refinement.py`, which verifies both original plans and requires complete seed pairs per plotted point. It has a GCS/W&B refresh mode and can regenerate the figure from the archived verification snapshot. Frozen training code, plan hashes, seeds, subset identities, and outputs are unchanged.

Validation: lint, formatting, AST checks and focused Pyrefly all pass. The original seven-point analysis reproduces exactly. `./build.sh` succeeds, with no undefined references or overfull boxes; 43 pages, references still on page 10. The figure and affected pages 7 and 16 were rendered and visually checked. `before/` preserves the overwritten manuscript/figure/PDF; `data/` holds both plans, metrics, endpoint verification, plotted values, analysis, and the plotter. No commit or Overleaf push.

## Operational finding for CC

The Wikipedia/FineMath survey remains at 0/28 training runs: five of six preparation caches are complete, and the final FineMath subset is blocked by CPU/RAM reservations in central1-a. The on-demand pool is at six VMs. No worker has both 2 CPU and 8 GiB free. Thirty-seven active attempts occupy these workers, including 24 runner Zephyr coordinators; the two TPP10 coordinators reserve 8 GiB each while peaking below 0.8 GiB.

The corresponding Wikipedia subset materializer peaked at 1.35 GiB. A 2 CPU/4 GiB prep request would fit currently available capacity; this is the recommended next operational change, with exact scientific recipes and completed caches preserved. FineMath's actual peak is unmeasured. No job cancellations, resubmissions, resource changes, or cluster changes were made. The user's priority budget is healthy. Echo publication was attempted but denied with HTTP 403; safe controller and Finelog evidence is recorded in Fieldbook and `domain_sweeps/monitor_20260911Tplot_refresh/`.


## 11 September, 21:24 UTC: approved subset-memory recovery

The user authorized reducing the final FineMath matched-subset worker from 2 CPU / 8 GiB to 2 CPU / 4 GiB. The new dispatch wrapper leaves the original recipe and callable unchanged, preserving all scientific fingerprints, seeds, source generations, output paths and TPU requests. The five completed data caches were reverified and retained. Parent resources stay 2 CPU / 8 GiB on-demand, interactive, central1-a.

Validation: four survey tests, repository lint and focused Pyrefly passed; native Fray-to-Iris wire comparison confirms that CPU RAM is the only change and every TPU request is identical. All 35 original frozen pins plus the wrapper pass bundle preflight. CC review `f3b8b8a7-5835-483c-8cad-271f12389b1f` found no blocker.

Canceled only the old survey coordinator and its unfinished child. Verified the whole old tree terminal, no unfinished worker attempts, and all data leases absent before resubmitting as `/calvinxu/tpp10-domain-sweeps-cpu4g` at 21:24:43 UTC. Fieldbook retry `job_01m295bxqrsneazvbk5f8f95nk` links to the original parent. Initial runtime verification is in progress; the StarCoder refinement parent remains unchanged. No manuscript, outline, scientific design, budget, cluster or cross-region changes. Evidence and exact command: `cpu4g_recovery/`.


### Live recovery verification at 21:30 UTC

The final FineMath subset completed successfully; all six data caches now pass their original recipe, tokenizer, token-count, index-membership and artifact-fingerprint checks. The matched FineMath cache remains exactly 10,485,760 tokens with fingerprint `6d286890`. Finelog measured a 1.37 GiB peak over 11 samples; the live request is 2 CPU / 4 GiB. All five earlier cache identities were retained.

Both included p100 training checks are dispatched with the unchanged TPU resources, frozen plan and interactive central1-a placement. Wikipedia acquired a TPU and loaded its first batch; FineMath awaits TPU capacity. The remaining26 wait for both execution checks to finish. No survey training result is claimed yet. Fieldbook and the durable monitoring state carry the replacement root, prep completion and first two child jobs. The existing heartbeat's job identity was updated; its pre-existing PAUSED status was preserved.


## 11 September, 23:34 UTC: survey progress and validation failures

Current check at 23:28–23:34 UTC: the domain survey has eight TPU training jobs finished, one Wikipedia target training and nineteen training jobs queued for TPU capacity. All seven nonzero Wikipedia matched points and the FineMath matched p100 point have permanent final checkpoint files, matching frozen runtime/domain receipts and finite final metrics. Only the two p100 checks have finalized executor artifacts; the other six retain stale RUNNING status without records. The coordinator logs `RuntimeError: cannot schedule new futures after shutdown`, following both shared p0 evaluation failures. PALOMA control errors are 0.00595093 target and 0.000258923 proxy (tolerance0.00005). Seven W&B summaries match their final metrics; Wikipedia matched p50 differs. These failures require a recovery pass before final collection. Do not relax the evaluator gate or repeat completed training. No cluster/job/recipe changes were made during this status check. StarCoder refinement remains 40/45; all five unfinished tasks are queued with zero runtime failures. The heartbeat remains paused. Evidence: `monitor_20260911T2328/`.

Saved metrics put the Wikipedia matched-proxy minimum at p30 (4.76836epochs, macroBPB1.531399) among the seven nonzero points, ahead of p20 (1.534229). This is provisional: shared p0 evaluations failed and artifact finalization is incomplete. No manuscript or figure update is made from this partial audit.

## 11 September: PALOMA discrepancy traced to batch-dependent BPB aggregation

A read-only audit found a batch-dependent BPB aggregation bug in the native Levanter evaluator. In lib/levanter/src/levanter/eval.py:583-590, each batch's loss is divided by its byte count, but RunningMean weights the resulting BPB by token count. Dataset BPB should be total loss in bits divided by total bytes; the existing average of batch ratios changes with batch boundaries.

A local reproduction through the actual TaggedEvaluator used identical one-bit losses for two examples with byte counts 1 and 4. PALOMA alone reported 0.40000000596 BPB. Prepending one unrelated example, with batch size two, changed PALOMA to 0.625 BPB. Token loss stayed exactly 0.69314718246. Correct global BPB is 2/5 = 0.4. This establishes the aggregation bug independently of model precision or checkpoint restoration.

In the real expanded evaluation, seven Uncheatable caches contain 3,810 examples before PALOMA's 5,673 examples. The PALOMA offset is 2 modulo the proxy batch size 32 and 98 modulo the target batch size 128, leaving 30 PALOMA examples in the first batch in either case. Original evaluation used PALOMA alone. Thus the expanded evaluation changes the batch boundaries of unchanged PALOMA examples.

This is a strong explanation for the observed control errors (0.005950927734375 target and 0.0002589225769042969 proxy), not yet an exact attribution: no actual checkpoint has been re-evaluated in this audit. The original token losses are 2.06310725212 and 3.03879451752; restored logs agree only to their displayed three decimal places. Next verify original versus expanded evaluation on the same saved checkpoint and accumulate loss bits and bytes explicitly. Do not widen the 5e-5 gate or mix corrected and legacy scores. The native aggregation bug predates these jobs; no source, tolerance, frozen recipe, checkpoint, or live job was changed.

Evidence: evaluation_bpb_audit_20260911/{native_evaluator_reproduction,original_control_metrics,evaluation_batch_boundaries}.json. Checkpoint integrity, stale executor completion records, and the Wikipedia p50 W&B discrepancy remain separate checks. This finding has not resolved those issues. Echo publication was denied with HTTP 403 earlier in this investigation; the incident draft remains at /tmp/tpp10_control_evaluation_incident_20260911.md.

## 11 September: BPB correction, coordinator recovery, and W&B repair

The native Levanter evaluator now divides total prediction loss in bits by total scored bytes. It no longer averages batch BPB with token weights. Hierarchical and labeled evaluation retain zero-byte special-token loss, and both log `bpb_schema_version=2`. Regression tests cover batch size, added domains, masking, zero-byte special tokens and hierarchical tags. Six cases fail before the fix; all 17 evaluator tests now pass, as do 17 recovery/domain tests and eight BPB cases on a four-device local mesh. Repository lint and focused type checking pass. The broad safe runner stops on a pre-existing import of removed `experiments.defaults.default_tokenize` in the legacy determinism tests.

CC's first review confirmed the arithmetic and identified recovery gaps. The survey launcher was restored byte for byte to preserve all frozen retry identities; its live bundle remains unchanged. Settling every independent branch before propagating an exception is implemented in the new repair coordinator. Original materialized training records are archived and match the two pre-existing p100 artifacts exactly.

Wikipedia matched p50 W&B was missing the final evaluation: its summary matched step2024, whereas the permanent checkpoint, saved metrics and TPU finish logs reached step2531. After backing up the old summary and history, all 30 final metrics were appended at step2531. Every value reads back exactly and W&B remains finished. No training output was overwritten. A subsequent check of all 21 completed survey runs found every saved final metric equal to its W&B summary.

The separately pinned regional repair requires successful children, matching original runtime/domain receipts, permanent checkpoints, every expected loss and BPB metric, and exact W&B agreement. A live parent prevents any artifact lease or record mutation. The broken coordinator can be retired only after all 28 training children succeed and every other descendant is terminal. Record recovery then rechecks the parent and outputs under the native lease. The repair never dispatches training. Result snapshots explicitly identify completeness, missing runs and outstanding artifact records.

Two TPU workers will audit three saved checkpoints: p0 proxy, p0 target and Wikipedia p50. The pinned historical evaluator must reproduce its previous values and batching discrepancy; the corrected evaluator must preserve PALOMA across population layouts and match reconstruction from saved token-average loss with exact CPU counts. The original 5e-5 gates remain unchanged. Corrected metrics get a separate namespace and immutable result snapshots. Both CC reviews are complete, their findings are addressed, and the recovery was submitted at 04:29:10 UTC on 12 September as `/calvinxu/tpp10-bpb-repair`. Its 1-CPU/4-GiB regional coordinator has verified 21 completed endpoints and counted all eight eval sets. Both v5p-8 audit workers await TPU capacity; numerical verification and artifact finalization remain pending. The original training jobs continue. Exact code, plans, review, commands and validation receipts are in `repairs_20260911/`.


## 12 September, 06:02 UTC: audited partial domain-sweep plots

All three saved-checkpoint BPB audits passed unchanged 5e-5 gates; both audit workers succeeded. The controls reproduce the original PALOMA batching discrepancies exactly, and corrected ratio-of-totals BPB can now be used across the survey. Native artifact-record recovery still awaits completion of the four remaining FineMath target jobs.

Archived a consistent 24/28-point snapshot plus the two shared p0 controls in `plots_20260912/`. The new `plot_tpp10_domain_sweeps.py` writes full-range and valley-detail Uncheatable plots, a seven-component diagnostic, CSV and exact minimum/neighbor tables. The source hash, audit receipts and actual exposure coordinates accompany the figures. Lint and focused Pyrefly pass; both PNGs were visually inspected and label overlaps corrected.

Wikipedia proxy and target both minimize at p30 (4.77 epochs) on the observed grid. FineMath proxy also minimizes there. Wikipedia target p20 differs by only 0.000505 BPB, so its valley is shallow. FineMath target has reached p20 and continues improving; p30/p50/p70/p100 remain unfinished. These points do not yet support widely separated domain optima under the common Uncheatable objective. Component trade-offs differ, but should not replace the prespecified aggregate comparison. See `plots_20260912/README.md` for interpretation and reproduction. Manuscript, outline, frozen experiment identities and training settings remain unchanged.


## 12 September: proxy optima across all recorded BPB evaluations

The complete proxy grids have different observed minima on six of eight component evaluations. PALOMA Programming Languages gives Wikipedia p30 (4.77 epochs) versus FineMath p50 (7.94 epochs), with opposite-choice penalties of 0.025175 and 0.007624 BPB. GitHub C++ has the same coordinates but FineMath p50 beats p30 by only 0.001349 BPB. AO3 and arXiv CS have roughly fourfold epoch-coordinate differences, with flatter valleys. The Uncheatable mean, GitHub Python and arXiv physics optima coincide. Wikipedia-English and BBC choose zero FineMath weight, so those differences do not establish positive-repetition turnover.

The analysis preserves all nine reported metrics (eight components plus the Uncheatable mean), both complete grids including p100, runner-up gaps and opposite-choice penalties. `eval_optima_20260912/` holds the source receipt, table, exploratory plot and interpretation; reproduce with `analyze_tpp10_domain_eval_optima.py`. Repository lint and focused Pyrefly pass; the plot was visually checked. Wikipedia-specific evaluation exists and selects p50; a math-specific evaluation has not been recorded for these checkpoints. No training, evaluation jobs, manuscript changes or replacement of the prespecified aggregate objective.


## 12 September: proxy-first scope correction and regional math inventory

User reaffirmed the original proxy-first staging: establish domain-optimum separation on completed proxies before any additional target release. The submitted survey included 14 proxies and 14 targets after the later request for initial optima in both settings; that scope expansion was not reconciled with the original gate. No new target submission or evaluation job was made during this audit. The four unfinished FineMath targets are currently queued after worker preemption, with checkpoints at steps 7965/6426/7219/6432 of final step 11490. A cancel-versus-continue decision was requested; no cancellation has been performed. Checkpoints currently reside under the region-local 14-day temporary-checkpoint prefix.

Estimated training FLOPs: proxy-only survey 3.48724e17; submitted target portion 9.32946e19. Completed targets plus retained partial progress account for at least 8.29036e19 target FLOPs; 1.03911e19 remain from the saved steps. This excludes evaluation, compilation, preparation and lost/replayed work, and is not a billing estimate.

Live central1 inventory confirms MATH-500 raw evaluation data with 500 rows and problem/solution/answer/subject/level fields, plus GSM8K raw evaluation data with 1,319 rows and problem/prompt/solution/ground_truth fields. Object generations, sizes, schemas and existing data provenance are archived in scope_math_audit_20260912. The GSM8K cached prompt uses five examples (seed 1234); any new scoring format must be explicitly fixed. These can be formatted with the TPP10 tokenizer and scored on existing proxy checkpoints without retraining or cross-region data movement. No math metric has yet been measured for these proxy checkpoints.


## 12 September, 06:39 UTC: four FineMath targets canceled; checkpoints preserved

The user chose to cancel the queued FineMath targets at 30%, 50%, 70% and 100%. All four are KILLED in Iris. Their latest saved checkpoints at steps 7965/6426/7219/6432 were preserved outside the 14-day temporary prefix with same-bucket server-side copies. All 28 objects (10,164,798,257 bytes) match source size, CRC32C and MD5, and the complete relative key sets and metadata agree. The preservation receipt records exact durable central1 paths and both generations; no checkpoint was deleted or downloaded locally.

All 14 proxies and 10 completed targets are retained. After every child was terminal, both idle coordinators received stop requests; the survey parent ended FAILED and the repair parent KILLED. No job in either tree remains active or queued. Numerical audit receipts and corrected result snapshots remain valid. Native artifact-record closeout remains incomplete because the old repair plan requires all 28 successes; do not rerun that plan unchanged or retry the intentionally canceled targets.

Further target training requires explicit user promotion after reviewing the proxies. HANDOFF.md now places that restriction above the historical recovery instructions. Fieldbook records the cancellation decision, live terminal states and preservation receipt. This supersedes the pending decision recorded in the preceding entry. No new training, evaluation, figure, manuscript or outline change was made. Evidence: scope_math_audit_20260912/{decision_status,checkpoint_preservation_receipt}.json and the two post-cancellation Iris inventories.


## 12 September: FineMath math likelihood and candidate screening

The user requested math perplexity on existing FineMath proxy checkpoints, then discussion with CC and the user before any new domain training. The release is exactly eight completed proxies (shared p0 plus FineMath p5/10/20/30/50/70/100); no targets or new training. MATH-500 reference-solution likelihood is primary and GSM8K secondary. Raw object generations, tokenization, scored populations, checkpoint metadata, code and tokenizer are pinned. Each endpoint must reproduce its saved PALOMA token loss before math results are accepted.

The first evaluation failed before accepting results because native PALOMA and math examples had incompatible attention-mask trees. The repaired retry `/calvinxu/tpp10-finemath-math-eval-retry` supplies one all-zero segment per math example, preserving causal attention. Eight tests, focused type checking, lint, native bundle checks and regional launch safety pass. The exact same eight checkpoints and token populations are retained. Diagnostic receipts precede validation; the separate collector preserves all points, provenance, observed minima, boundary flags and neighboring gaps. First-attempt artifacts remain in math_eval_20260912/attempt1/.

CC completed two read-only reviews through the verified subscription account using Opus 5. The follow-up accepts two evidence corrections: archived synthetic-math/code/thinking minima at 14.486 epochs are interior because 28.971 was also measured and worse; the regional Nemotron Math Textbooks source has plain text. Both CC and Codex recommend discussing Nemotron Math Textbooks and Stack-Edu Python, with arXiv as backup, after FineMath's math results. No archived domain establishes >10 epochs on common Uncheatable. No candidate training is released. Full protocol, shortlist, reviews, disposition and Fieldbook receipts are in math_eval_20260912/.


## 12 September, 07:19 UTC: math evaluation complete

All eight FineMath/shared-p0 math evaluations are verified; the retry parent and TPU worker both succeeded. MATH-500 and GSM8K both minimize at p70 = 11.098 realized epochs. At p50/p70/p100, MATH-500 PPL is 7.8880/7.5340/8.1328 and GSM8K PPL is 12.0467/11.4399/12.9085. FineMath's common Uncheatable minimum remains p30 = 4.768 epochs. The high-epoch result is therefore established on these math diagnostics, not on the common Uncheatable objective. All raw metrics and identities reproduce independently; maximum restored PALOMA loss delta is 1.19209e-6 against the unchanged 5e-5 gate. The full plot is visually checked.

CC reviewed the complete results and agrees that the next proposed step is scoring existing Wikipedia proxies on the same math metrics before any new math-domain training. This evaluation is not yet released. If further training is agreed, discuss Stack-Edu Python, arXiv and Nemotron Textbooks, with proxy-only scope. The historical target cancellations remain in force. No manuscript or outline change; propagate these measured results from math_eval_20260912/RESULTS.md when deciding the illustrative figure.


## 12 September: Wikipedia-English curves at both scales

Plotted both complete Wikipedia sweeps on the separate Wikipedia-English evaluation corpus. Proxy minimum: p50, 7.93789 epochs, 1.3571734 BPB. Target minimum: p70, 11.09793 epochs, 0.99084579 BPB; p50 is only 0.00335984 BPB worse. The earlier 7.94-epoch reference applied to the proxy. Figure, numerical table, provenance receipt and interpretation are in wikipedia_eval_20260912/. Source hashes and minima were checked independently and the plot visually inspected. No training, evaluation, manuscript or outline changes.


## 12 September: Wikipedia proxies scored on the same math protocol

The user approved the proposed existing-checkpoint evaluation. Seven Wikipedia matched proxies were scored on the identical frozen MATH-500/GSM8K protocol used for FineMath, reusing the verified shared p0 measurement. The central1 parent `/calvinxu/tpp10-wikipedia-math-eval` and its single TPU worker both succeeded. No training or target evaluation was submitted.

The complete curves now show separation on a common evaluation: MATH-500 selects Wikipedia p30 = 4.768 epochs versus FineMath p70 = 11.098; GSM8K selects Wikipedia p20 = 3.162 versus FineMath p70 = 11.098. On the primary MATH-500 metric, adopting the other domain's preferred epoch count raises PPL by 21.2% for Wikipedia and 17.6% for FineMath. These exact-choice penalties do not measure a cap's cost: a loose upper bound can admit both minima. The common-Uncheatable minima remain equal at 4.768 epochs.

Independent live receipt review reproduced all 16 curve points and every minimum and opposite-choice penalty, verified all seven new checkpoint metadata hashes, and confirmed identical scored populations and shared-p0 lineage. Maximum Wikipedia PALOMA restore discrepancy is 3.81470e-6, below 5e-5. The full-range and explicitly restricted valley-detail plots retain measured points without fitted curves. They show the p100 failure separately from the valleys.

Protocol, complete table, limitations, reproducer, plot/receipt artifacts and Fieldbook completion are in `wikipedia_math_eval_20260912/`. The new evaluator wrapper reuses the unchanged FineMath scorer and pins its own code and baseline receipt; local scoring tests, independent release review, bundle verification and region checks passed before submission. `analyze_tpp10_wikipedia_math.py` performs the read-only collection and plotting. This comparison was exploratory, after the broad-evaluation survey, and uses one trainer seed and matched subset per domain. No manuscript or outline edits; propagate from `RESULTS.md` when selecting the illustrative figure. The canceled FineMath targets stay canceled.


## 12 September: full FineMath completion authorized

The user explicitly reversed the four-target cancellation and requested completion of the full FineMath sweep plus math evaluation of every completed checkpoint. Recovery is limited to the original p30/p50/p70/p100 targets, with full optimizer-state resume and unchanged data, seeds, horizons and output identities. All proxy points are already complete. The first target math release scores the eleven existing final target checkpoints (Wikipedia7, FineMath3, shared targetp0); a follow-up release will score the four resumed final checkpoints with the same frozen math scoring. No additional domains or unmatched controls.

The four-target recovery is now running as `/calvinxu/tpp10-finemath-target-completion`. Live logs confirm full-state restoration from steps 7965/6426/7219/6432 and subsequent training progress. The original training fingerprints and permanent output paths are unchanged. The narrow resume wrapper, frozen recovery plan, bundle verification and live receipts are in `target_completion_20260912/`.

The first eleven target math evaluations have completed and passed independent review. All 11 unique receipts (12 displayed points, with shared p0) reproduce their permanent checkpoint metadata, full scored populations and PALOMA controls; maximum token-loss discrepancy is 9.54e-7. Wikipedia target minimizes MATH-500 at p50 = 7.937 epochs / PPL 7.8320 and GSM8K at p20 = 3.161 epochs / PPL 7.6110. The MATH-500 valley is shallow: p20 is only 0.00706 NLL worse. FineMath target still improves through its last completed p20 point; its minimum is not yet established. The proxy-target figure retains measured points and marks the incomplete FineMath target track. Results and independent review are in `target_math_eval_20260912/`.

`evaluate_tpp10_target_math_complete.py` and the full-grid collector have passed independent review. The final evaluation will reuse the 11 completed receipts by exact hash and score only the 4 new permanent target checkpoints after training finishes. The MATH-500/GSM8K protocol, source populations, solution mask, tokenizer, evaluation batch size and checkpoint control threshold are identical across scales and domains. No manuscript or outline edits.

The existing thread heartbeat was updated and reactivated for this completion only. It will verify the four training outputs, run the reviewed final evaluation with Fieldbook and bundle checks, produce the complete plots and pause once the authorized work is finished. The final evaluation has not been submitted while training is still running.

The user then requested pausing monitoring to work on another task. The heartbeat is PAUSED; the four submitted training continuations remain running. Final checkpoint verification and math evaluation will resume when the user checks back. No training was canceled.


## 12 September, 21:12 UTC: final math evaluation submitted; refinement capacity checked

The user requested completion of both existing experiments and their plots, superseding the earlier monitoring pause. All four resumed FineMath targets are successful with permanent final checkpoints and native component metrics verified. The final math evaluation `/calvinxu/tpp10-target-math-completion` was submitted after immutable-spec, source, checkpoint, bundle and regional checks. It reuses eleven accepted target receipts and scores only four new checkpoints. Fieldbook job `job_01m2bq8ta6wg2w5d3bgjhkkxxs` links to the four existing training runs as an evaluation; no training point was added. Release evidence is in `target_math_complete_20260912/`.

A fresh StarCoder audit verifies 40/45 final artifacts against fingerprints, permanent checkpoints, frozen runtime and final W&B metrics. The five remaining BATCH jobs comprise four targets at p55/p60/p65/p80 and one matched p80 replicate. The four target checkpoints retain steps 10759/6958/2283/728; their parent is healthy and central1 v5p capacity is exhausted. They were not resubmitted. Exact recovery boundaries and the 13 September 12:38 UTC parent deadline are recorded in the monitoring state. Plot completion remains conditional on accepted final outputs.


The CPU-only native closeout succeeded: all 28 corrected BPB rows are now published, `complete=true`, with no omitted points or artifact recoveries pending. It reused the existing numerical audits and changed only the job-source mapping for four resumed targets. The final math evaluator's TPU worker is queued for capacity under the running coordinator. The existing thread heartbeat is active every 15 minutes for both math completion and StarCoder refinement, including checkpoint-preserving recovery after a terminal failure or timeout. No scientific settings, manuscript prose or header figure changed.


The complete native plots are in `native_closeout_20260912/plots/`. On common Uncheatable BPB, both domains at both scales have their lowest measured loss at p30, about 4.768 epochs. FineMath target is 1.084206923 BPB there; p50 is higher by 0.001421914. The prior 24 rows, exact scored populations and shared controls are unchanged. These results preserve the coincident-optimum finding on Uncheatable; the final math-specific comparison still awaits the four new target evaluations. No manuscript claims were changed.


## 12 September, 21:23 UTC: final math results verified and priority clarified

The final four target math evaluations succeeded; all 15 unique targets and 15 proxies now verify, with identical populations and maximum PALOMA discrepancy 3.81e-6 below 5e-5. FineMath target MATH-500 still improves through the p100 boundary (15.828 epochs, PPL 3.55485); its GSM8K minimum is p70 (11.098 epochs, PPL 4.03641). FineMath proxy selects p70 for both metrics. Wikipedia target selects p50/p20 for MATH-500/GSM8K; its proxy selects p30/p20. Complete plots and reproducible tables live in `target_math_complete_20260912/`. No fitted curve or continuous-optimum claim was added.

The remaining five StarCoder jobs are explicitly BATCH. At 21:23 UTC all 30 ready v5p-8 workers were occupied by other users; regional quota blocked additional slices. Finishing other jobs reduced the user's active interactive charge to 45/75000, restoring interactive eligibility. This does not promote explicit BATCH jobs. FineMath's interactive evaluation obtained capacity and completed at 21:18 UTC. Priority, allocation, job identities and checkpoints were unchanged during this explanation.


## 12 September, 21:34 UTC: remaining StarCoder jobs returned to interactive

The user explicitly requested interactive priority for the five remaining StarCoder refinements. Iris has no supported in-place priority update, so the old batch-recovery tree was stopped, all jobs verified terminal and all five output leases allowed to expire. All 40 completed artifacts, scientific fingerprints, permanent checkpoints and the four pending target checkpoint object sets were unchanged. The replacement `/calvinxu/starcoder-tpp10-refinement-interactive` calls the original frozen launcher directly, omitting the batch wrapper. It retains the original release, plan and output paths, skips the 40 completed recipes and dispatches only the five unfinished recipes. Region/native-bundle checks and an independent route review passed. No scientific settings or manuscript content changed. Exact command, pre/post-migration receipts and new Fieldbook lineage are in `../live/interactive_migration_20260912/`. The heartbeat now monitors the interactive parent.


Live dispatch at 21:37 UTC confirms exactly the five unfinished children, each with INTERACTIVE requests and the unchanged central1-a TPU resources. All five received workers and entered BUILDING. This verifies the priority migration; checkpoint restoration and final metrics remain under the completion monitor.

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
