# Proportional Perturbation Scale-Transfer: Research Logbook

## Scope
- Goal: test whether interpretable exposure increases around `baseline_proportional` change the BPB improvement from `60M/1.2B` to `100M/6B`.
- Primary metric: `delta_bpb = eval/uncheatable_eval/bpb@60M - eval/uncheatable_eval/bpb@100M`.
- Primary contrast: `effect_vs_proportional = delta_bpb(intervention) - delta_bpb(baseline_proportional)`.
- Constraints: run on `v5p-8` in `us-east5-a`, use region-local runtime caches, skip lm-eval harness, preserve validation/perplexity metrics, final checkpoints, and HF export.
- Experiment issue: https://github.com/marin-community/marin/issues/5521

## Baseline
- Base mixture: `baseline_proportional`, using constant `phase_0 == phase_1` top-level Dolma 3 + Dolmino weights.
- Baseline source experiment: `pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240`.
- Existing baseline rows at `60M/1.2B` and `100M/6B` are reused as the anchor; no fresh baseline reruns in Stage 1.
- Noise context: prior SNR tables estimate `eval/uncheatable_eval/bpb` run noise around `0.00140` at 60M and `0.00073` at 100M, so a paired delta has about `0.0016` BPB independent noise.

## Experiment Log

### 2026-05-09 - Conceptual framing: local perturbations as directional derivatives
- The one-at-a-time perturbations around `baseline_proportional` are best interpreted as finite
  directional-derivative measurements on the mixture simplex, not as unconstrained partial derivatives.
  For an expected objective `F(w)` at a fixed scale, a domain bump estimates
  `(F(w0 + epsilon * v_i) - F(w0)) / epsilon`, where `v_i` is defined by the chosen renormalizer:
  increase the target mass and remove mass proportionally from the donor pool.
- In an infinite-compute thought experiment, repeatedly estimating these local directional derivatives
  and taking projected or mirror-descent steps on the simplex would be a very strong mixture optimizer.
  It would still be local and could miss disconnected optima or sharp phase-transition regions, but it is
  conceptually cleaner than relying on one global surrogate fit from a fixed random design.
- The practical blocker is cost and latency: every gradient step needs many paired perturbation runs,
  and enough repeats or scale to separate true marginal effects from trainer/data-subset noise.
- The current surrogate work can be framed as amortizing this expensive adaptive optimization. GRP, MCT,
  and IRT-style aggregate targets should be judged partly by whether they reproduce the measured local
  gradient around proportional and whether the direction from proportional to their predicted optimum is
  aligned with the perturbation-derived descent direction.
- A useful next diagnostic is therefore: compare the measured local perturbation effects, the
  surrogate-predicted local effects at proportional, and the surrogate's proposed move from proportional.
  Strong alignment supports trust-region validation of the surrogate optimum; disagreement argues for a
  narrower local model or additional interventions that break incidental correlations in the random swarm.

### 2026-05-10 07:42 UTC - Domain-only gradient-step candidate generation
- Goal: turn the domain-only proportional perturbation effects into one-step candidate mixtures for validating
  the finite-difference-as-local-gradient idea.
- Command:
  ```bash
  uv run python experiments/domain_phase_mix/exploratory/two_phase_many/build_proportional_gradient_step_candidates.py
  ```
- Inputs:
  - `paired_bpb_effects.csv` from the Stage 1 perturbation analysis.
  - `intervention_manifest.csv` for the exact domain-bump displacement vectors and proportional base weights.
- Output directory:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/gradient_step_candidates_domain_only/`.
- Generated artifacts:
  - `candidate_summary.csv`: 41 domain-only candidates and local predicted BPB effects.
  - `candidate_weights.csv`: launchable `phase_0 == phase_1` weights for every candidate.
  - `candidate_domain_changes.csv`: long-form domain deltas versus proportional.
  - `domain_local_gradients.csv`: fitted local ridge-gradient coefficients.
  - `local_linear_diagnostics.csv`: in-sample and leave-one-domain diagnostics.
  - `recommended_candidates.csv`: short list for launch discussion.
  - `candidate_predicted_effects.html` and `top_100m_candidate_domain_deltas.html`.
- Key observed finite-difference structure:
  - Domain bump effects are strongly aligned across scales: Pearson `0.938`, Spearman `0.742`.
  - Top helpful domains at both scales are `dolma3_stack_edu`, `dolmino_stack_edu_fim`,
    `dolmino_synth_code`, and `dolma3_arxiv`.
- Main caveat:
  - Ridge-gradient candidates have the strongest predicted BPB gains, but the 39-domain local linear
    model is high-leverage. With ridge `1e-5`, leave-one-domain RMSE is `0.0539` BPB at 60M and
    `0.0287` BPB at 100M, far larger than in-sample RMSE. This means fitted-gradient optima should be
    treated as aggressive diagnostics, not the safest first validation.
- Recommended conservative validation candidates:
  - `60m_observed_good_all_tv0.050`: TV `0.050`, predicted `-0.0200` BPB at 60M and `-0.0206` at 100M.
  - `100m_observed_good_all_tv0.050`: TV `0.050`, predicted `-0.0168` BPB at 60M and `-0.0183` at 100M.
  - These are convex combinations of observed helpful domain-bump directions, so they stay close to the
    measured perturbation radius and measured intervention cone.
- Balanced good/bad candidates:
  - `60m_balanced_all_tv0.050`: TV `0.050`, predicted `-0.0216` BPB at 60M and `-0.0207` at 100M.
  - `100m_balanced_all_tv0.050`: TV `0.050`, predicted `-0.0184` BPB at 60M and `-0.0189` at 100M.
  - `100m_balanced_top8_tv0.050`: TV `0.050`, predicted `-0.0200` BPB at 60M and `-0.0206` at 100M.
  - The balanced candidates explicitly remove from harmful-effect domains instead of only adding helpful
    observed directions with proportional removal.
- Aggressive fitted-gradient alternative:
  - `avg_ridge_mirror_tv0.050`: TV `0.050`, predicted `-0.0206` BPB at 60M and `-0.0191` at 100M.
- Interpretation:
  - The safest scale-matched launch is `60m_observed_good_all_tv0.050` at 60M and
    `100m_observed_good_all_tv0.050` at 100M. If using four slots, add `60m_balanced_all_tv0.050`
    at 60M and `100m_balanced_top8_tv0.050` at 100M. If using eight slots, run the four candidate
    mixtures at both scales to directly test scale transfer.

### 2026-05-10 08:00 UTC - GRP no-L2 scoring of gradient-step candidates
- Goal: check whether the validated 60M GRP no-L2 uncheatable-BPB surrogate agrees with the finite-difference
  candidate rankings, especially for higher-TV candidates.
- Commands:
  ```bash
  uv run python experiments/domain_phase_mix/exploratory/two_phase_many/build_proportional_gradient_step_candidates.py
  uv run python experiments/domain_phase_mix/exploratory/two_phase_many/score_proportional_gradient_candidates_with_grp.py
  ```
- Bug fixed before scoring:
  - The first balanced-good/bad construction applied domain effects in intervention-row order rather than
    domain-column order. Observed-cone and ridge-gradient candidates were unaffected, but balanced candidates
    were regenerated after aligning effects by domain name.
  - Balanced removal now uses capped proportional removal so the requested TV radius is exactly respected
    without negative weights.
- Output:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/gradient_step_candidates_domain_only/candidate_summary_with_grp_no_l2.csv`.
- GRP no-L2 predictions for requested TV `0.05` candidates, all as BPB deltas versus proportional:
  - `60m_observed_good_all_tv0.050`: `-0.0228`.
  - `100m_observed_good_all_tv0.050`: `-0.0198`.
  - `60m_balanced_all_tv0.050`: `-0.0158`.
  - `100m_balanced_all_tv0.050`: `-0.0158`.
  - `100m_balanced_top8_tv0.050`: `-0.0236`.
  - `avg_ridge_mirror_tv0.050`: `-0.0214`.
- GRP no-L2 does not penalize high-TV candidates in this local family. Median predicted improvement grows
  with TV: about `-0.0246` at TV `0.05`, `-0.0327` at TV `0.075`, and `-0.0386` at TV `0.10`.
- Interpretation:
  - GRP agrees that these candidates are better than proportional and is especially favorable to the
    ridge-gradient and top8/observed high-TV variants.
  - This does not mean we should jump to TV `0.10` first; the perturbation-derived finite differences were
    measured at TV `0.05`, so TV `0.05` remains the clean validation of the local-gradient idea.

### 2026-05-10 - DSP agreement with domain perturbations
- Goal: test whether the canonical DSP surrogate agrees with the actual one-at-a-time domain-bump
  perturbations around proportional.
- Implementation: added a DSP agreement section to
  `experiments/domain_phase_mix/exploratory/two_phase_many/proportional_perturbation_scale_transfer_analysis.py`.
- Outputs:
  - `reference_outputs/proportional_perturbation_scale_transfer_20260507/dsp_domain_perturbation_agreement.csv`
  - `reference_outputs/proportional_perturbation_scale_transfer_20260507/dsp_domain_perturbation_agreement_summary.csv`
  - `reference_outputs/proportional_perturbation_scale_transfer_20260507/img/dsp_domain_finite_prediction_agreement.html`
  - `reference_outputs/proportional_perturbation_scale_transfer_20260507/img/dsp_domain_local_gradient_agreement.html`
- Method:
  - finite prediction compares `DSP(w_bump) - DSP(w_proportional)` against the observed 100M/6B BPB effect;
  - local directional effect estimates DSP's derivative at proportional along the exact bump direction using a
    small interpolation step.
- Canonical DSP (`dsp_phase_benefit_penalty_nnls`) finite-prediction agreement with 39 domain bumps:
  Pearson `0.899`, Spearman `0.640`, sign agreement `0.872`, calibration slope `1.106`, top-8 helpful
  overlap `7/8`, top-8 harmful overlap `2/8`.
- Effective-exposure DSP comparator finite-prediction agreement:
  Pearson `0.907`, Spearman `0.644`, sign agreement `0.923`, calibration slope `1.078`, top-8 helpful
  overlap `7/8`, top-8 harmful overlap `2/8`.
- Local-gradient agreement is weaker than finite-prediction agreement. For canonical DSP, local Pearson is
  `0.823`, Spearman `0.571`, sign agreement `0.718`, and top-8 helpful overlap `5/8`.
- Interpretation:
  DSP captures the finite 0.05-TV domain perturbation effects around proportional surprisingly well,
  especially the helpful directions. The local derivative is less reliable, so the perturbation radius is
  large enough that curvature matters. For validation and candidate generation, finite trust-region
  scoring is safer than pure infinitesimal-gradient descent.

### 2026-05-10 - Scale-specific DSP prediction-vs-actual plot
- Goal: visualize scale-specific DSP predictions for each domain perturbation versus actual measured effects,
  with sign disagreements highlighted.
- Implementation: added a scale-specific DSP prediction cell to
  `experiments/domain_phase_mix/exploratory/two_phase_many/proportional_perturbation_scale_transfer_analysis.py`.
- Additional model artifact:
  `reference_outputs/dsp_canonical_variants_60m_20260510/`, fitting canonical
  `dsp_phase_benefit_penalty_nnls` on the 60M fit swarm.
- 60M canonical DSP fit: CV RMSE `0.009969`, OOF Spearman `0.850313`.
- Output table:
  `reference_outputs/proportional_perturbation_scale_transfer_20260507/dsp_scale_specific_domain_perturbation_predictions.csv`.
- Output plot:
  `reference_outputs/proportional_perturbation_scale_transfer_20260507/img/dsp_scale_specific_domain_prediction_vs_actual.html`.
- Results:
  - 60M/1.2B: Pearson `0.877`, Spearman `0.696`, sign agreement `36/39`.
  - 100M/6B: Pearson `0.899`, Spearman `0.640`, sign agreement `34/39`.
- Sign disagreements are mostly small measured effects near zero. 60M disagreements are
  `dolma3_cc/education_and_jobs_high`, `dolma3_cc/science_math_and_technology_low`, and
  `dolma3_cc/history_and_geography_high`. 100M disagreements are
  `dolma3_cc/science_math_and_technology_low`, `dolma3_cc/history_and_geography_high`,
  `dolma3_finemath_3plus`, `dolmino_synth_qa`, and `dolmino_synth_thinking`.

### 2026-05-07 14:11 UTC - Stage 1 launcher implementation
- Hypothesis: if dense/domain-specific data has scale-dependent marginal value, a perturbation around proportional can change the 60M→100M BPB drop in a domain- or family-specific way.
- Command: implementation in `experiments/domain_phase_mix/launch_proportional_perturbation_scale_transfer.py`.
- Config:
  - 39 domain bumps: add `+0.05` absolute mass to one domain and remove proportionally from all non-target domains.
  - 3 GRP-family bumps: add `+0.05` absolute mass to `broad_text`, `tech_code`, or `reasoning`, preserve within-family ratios, and remove proportionally outside the family.
  - 13 CC quality swaps: move `50%` of the low-quality split mass to the paired high-quality split, preserving parent-topic mass.
  - Phase mode: `both_phases`, with `phase_0 == phase_1` for every perturbed mixture.
  - Scales: `60m_1p2b` and `300m_6b`, displayed as `60M/1.2B` and `100M/6B`.
  - Training children: `55 * 2 = 110`.
- Expected artifacts:
  - Local dry-run manifests under `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/`.
  - Per-scale executor manifests, result CSVs, and fit datasets under the executor output prefix.
- Next action: run dry-run validation, create the experiment issue, submit through Iris, and hand off monitoring instructions to CC.

### 2026-05-07 14:19 UTC - Dry-run validation and experiment issue
- Command:
  ```bash
  .venv/bin/python experiments/domain_phase_mix/launch_proportional_perturbation_scale_transfer.py \
    --dry-run \
    --tpu-region us-east5 \
    --tpu-zone us-east5-a \
    --max-concurrent 256
  ```
- Local artifact directory:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/`.
- Validation results:
  - `55` interventions and `110` training rows.
  - Intervention counts: `39` domain bumps, `3` family bumps, `13` quality swaps.
  - Scale coverage: `55` rows at `60m_1p2b` and `55` rows at `300m_6b`.
  - No duplicate intervention IDs or `(scale, run_name)` keys.
  - Target final checkpoint steps: `4576` for `60m_1p2b`, `22887` for `300m_6b`.
  - Phase columns: `39` domains in each phase.
  - Max phase-sum error: `3.33e-16`.
  - Max `phase_0` vs `phase_1` delta: `0.0`.
  - Minimum phase weight: `4.77e-4`.
  - Domain/family bump and quality-swap renormalizer checks passed.
- GitHub issue: https://github.com/marin-community/marin/issues/5521.
- Next action: submit the live Iris job.

### 2026-05-07 14:20-14:38 UTC - Live submission debugging
- First parent:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-142000`.
  - Outcome: failed during worker build before child dispatch.
  - Root cause: `uv.lock` referenced `torch==2.10.0+cu128` from `d3rlpy==2.8.1`, but the lockfile had no corresponding locked package block.
  - Fix: restored the missing locked `torch 2.10.0+cu128` package block and verified `uv lock --locked`.
- Second parent:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-142321`.
  - Outcome: failed before child dispatch.
  - Root cause: default Iris parent resources (`1 GiB` memory) were insufficient for this 497-step executor graph; the parent was OOM-killed while writing executor metadata.
  - Fix: resubmit the parent with `--enable-extra-resources --cpu 1 --memory 16GB --disk 20GB`.
- Third parent:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143206`.
  - Outcome: failed before child dispatch.
  - Root cause: the parent ran in inherited `us-central1`, so executor region inference rejected east5-local dependency steps such as `tokenized/finemath_3_plus`.
  - Fix: resubmit the parent itself with `--region us-east5 --zone us-east5-a`.

### 2026-05-07 14:38 UTC - Live experiment submitted
- Live parent:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728`.
- Submission command:
  ```bash
  .venv/bin/iris --cluster marin job run \
    --no-wait \
    --enable-extra-resources \
    --cpu 1 \
    --memory 16GB \
    --disk 20GB \
    --region us-east5 \
    --zone us-east5-a \
    --job-name "dm-proportional-perturbation-scale-transfer-$(date -u +%Y%m%d-%H%M%S)" \
    -- python experiments/domain_phase_mix/launch_proportional_perturbation_scale_transfer.py \
      --tpu-region us-east5 \
      --tpu-zone us-east5-a \
      --max-concurrent 256
  ```
- Health check:
  - Parent state: running.
  - Child state after dispatch: `110` pending training jobs, `0` failed, `0` succeeded.
  - Child resources: `v5p-8`, `32 cpu`, `128 GiB`, `50 GiB disk`, pinned to `us-east5-a`.
  - Initial pending reason: TPU capacity unavailable (`need 4, available 0`), expected scheduler behavior rather than a code bug.
- Monitor prefix:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728`.

### 2026-05-07 19:28 UTC - 60M partial results
- Live state:
  - `60m_1p2b`: `54` children marked succeeded, `1` child marked failed.
  - `300m_6b`: all `55` children still running.
- The failed Iris child is:
  `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728/train_lm_60m_1p2b_ppert_domain_dolma3_cc_food_and_dining_low`.
  Iris reports SIGSEGV, but the target artifacts exist:
  - `checkpoints/eval_metrics.jsonl`
  - `checkpoints/step-4576/metadata.json`
  - `hf/step-4576/model.safetensors`
- Retry policy for this child: do not blindly retry. Treat it as artifact-complete unless registry ingestion later rejects it as not target-ready.
- Partial 60M result CSV:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/partial_60m_perturbation_results.csv`.
- Baseline comparison:
  - Reused `baseline_proportional` at `60m_1p2b`, target-step `eval/uncheatable_eval/bpb = 1.0918355`.
  - All `55/55` 60M perturbations have target-step BPB at step `4576`.
- Best 60M perturbations by uncheatable BPB delta vs proportional:
  - `family_tech_code`: `-0.023664` BPB.
  - `domain_dolma3_stack_edu`: `-0.022480` BPB.
  - `domain_dolmino_stack_edu_fim`: `-0.021701` BPB.
  - `domain_dolmino_synth_code`: `-0.013186` BPB.
  - `domain_dolma3_arxiv`: `-0.004867` BPB.
- Worst 60M perturbations:
  - `family_broad_text`: `+0.077631` BPB.
  - `domain_dolma3_wikipedia`: `+0.016062` BPB.
  - `domain_dolmino_stem_heavy_crawl`: `+0.014528` BPB.
  - `domain_dolma3_cc_food_and_dining_low`: `+0.009362` BPB.
  - `domain_dolma3_cc_industrial_low`: `+0.006831` BPB.
- By intervention family, mean delta vs proportional:
  - Quality swaps: `+0.000416` BPB, range `[-0.002270, +0.003579]`.
  - Domain bumps: `+0.002048` BPB, range `[-0.022480, +0.016062]`.
  - Family bumps: `+0.019407` BPB, range `[-0.023664, +0.077631]`.
- Interpretation:
  - At 60M only, proportional appears underweight on code/technical sources for `eval/uncheatable_eval/bpb`.
  - Broad-text and several low-quality CC domain bumps are clearly harmful at 60M.
  - Quality swaps are small and mostly near the noise floor.
  - These results do not yet answer the scale-transfer question; the decisive metric is the paired `60M -> 100M` delta after the 100M jobs finish.

### 2026-05-08 - Paired analysis and downstream-eval completion workflow
- Parent job status:
  - `/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728` succeeded with `1/1` parent task complete, `0` parent failures, and `3` preemptions.
  - Direct checkpoint scan found `110/110` perturbation children with target-step `eval/uncheatable_eval/bpb`.
  - All `55` interventions are complete at both `60m_1p2b` and `300m_6b`.
- Registry work:
  - Added registry families `proportional_perturbation_60m_1p2b` and `proportional_perturbation_300m_6b`.
  - Registry rows preserve intervention metadata, target/family/domain fields, TV distance, renormalizer, donor pool, phase weights, target steps, checkpoint roots, and HF paths.
  - Metric-registry ingestion now hydrates checkpoint-backed `eval/*` metrics for these perturbation families.
- Analysis workflow:
  - Added Marimo notebook:
    `experiments/domain_phase_mix/exploratory/two_phase_many/proportional_perturbation_scale_transfer_analysis.py`.
  - The notebook writes paired BPB summaries and plots under:
    `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/`.
  - BPB sign convention: negative perturbation effect helps; negative `effect_100_bpb - effect_60_bpb` means the perturbation helps more at 100M/6B.
- Downstream-eval workflow:
  - Added candidate builder:
    `experiments/domain_phase_mix/build_proportional_perturbation_eval_candidates.py`.
  - Candidate population is the `110` perturbation checkpoints plus proportional baseline anchors at `60m_1p2b` and `300m_6b`.
  - The candidate CSV is intended for existing eval launchers through `MARIN_EXTRA_EVAL_CANDIDATES_CSVS`.
  - Updated parity and agentic-coding eval launchers to accept perturbation and proportional-anchor panels.
- Downstream-eval submission:
  - Candidate CSV:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/proportional_perturbation_eval_candidates.csv`.
  - Candidate counts: `112` rows total, `55` perturbations per scale plus `2` proportional anchors.
  - Dry-run state counts:
    - GSM8K/HumanEval: `110` launch steps, `2` baseline-anchor skips.
    - English-lite: `111` launch steps over `112` candidate checkpoints.
    - Generative smooth proxies: `111` launch steps over `112` candidate checkpoints.
    - MCQ smooth proxies: `111` launch steps over `112` candidate checkpoints.
    - Noise/parity aliases: `112` launch steps.
    - Agentic-coding BPB: `111` launch steps over `112` candidate checkpoints.
  - Submitted parents:
    - `/calvinxu/dm-ppert-gsmhe-20260508-130207`
    - `/calvinxu/dm-ppert-english-lite-20260508-130235`
    - `/calvinxu/dm-ppert-gensmooth-20260508-130255`
    - `/calvinxu/dm-ppert-mcq-smooth-20260508-130320`
    - `/calvinxu/dm-ppert-noise-parity-20260508-130339`
    - `/calvinxu/dm-ppert-agentic-bpb-20260508-130408`
  - Initial health check: all parents submitted in `us-east5-a`; noise-parity parent completed immediately; the other five parents were running with zero parent failures at first check.
- Issue #5416 aggregate:
  - Added implementation helper:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/issue5416_aggregate.py`.
  - The implementation follows the latest collaborator gist: select task columns from the 300M raw metric matrix, use MCQ `choice_logprob` with positive sign when available, use non-MCQ `bpb` sign-flipped, drop meta aggregates and per-MMLU-subject columns, choose factor count by Horn parallel analysis, and use the uniform mean posterior factor score as the aggregate projection.
  - The notebook fits this projection on the current 300M matrix plus variable-subset noise, then applies the fixed projection to perturbation rows once downstream eval columns are complete.

### 2026-05-08 - Downstream-eval correction
- The first downstream-eval parents listed above completed, but their result CSVs contained only the default
  `fixed_seed_noise_300m_6b` and `variable_subset_noise_300m_6b` panels.
- Root cause: `MARIN_EXTRA_EVAL_CANDIDATES_CSVS` and `MARIN_300M_CANDIDATE_PANELS` were set in the local
  shell but were not injected into the remote Iris parent environment. The parent therefore built the
  default noise-baseline DAG instead of the perturbation DAG.
- A second attempt injected the env vars, but used a local generated CSV path. Those parents failed before
  dispatch with `FileNotFoundError` because generated reference-output CSVs are not available inside the
  Iris workspace bundle.
- Fix:
  - Uploaded the candidate CSV to:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer_20260507/proportional_perturbation_eval_candidates.csv`.
  - Resubmitted ppert3 parents with Iris `-e` environment injection and the GCS candidate CSV.
  - Moved the noise-only local result CSVs aside with `.noise_only_wrong_20260508.csv` suffixes so the
    metric registry cannot accidentally ingest them as perturbation results.
- ppert3 parents:
  - `/calvinxu/dm-ppert3-gsmhe-20260509-011821`
  - `/calvinxu/dm-ppert3-english-lite-20260509-011821`
  - `/calvinxu/dm-ppert3-gensmooth-20260509-011821`
  - `/calvinxu/dm-ppert3-mcq-smooth-20260509-011821`
  - `/calvinxu/dm-ppert3-noise-parity-20260509-011821`
  - `/calvinxu/dm-ppert3-agentic-bpb-20260509-011821`
- Health check: all ppert3 parents stayed running past the previous failure point and began dispatching
  child eval steps for the perturbation panels.
- Metric-registry preparation:
  - Added mixed-scale local downstream result sources for the perturbation eval result CSVs.
  - Added `teacher_forced/` and `mcq_smooth/` metric-prefix ingestion.
  - Added mixed-scale source handling so collected CSVs can preserve per-row `60m_1p2b` vs `300m_6b`
    provenance.

### 2026-05-09 - ppert3 downstream recovery
- Parent status:
  - Five ppert3 downstream parents failed after partial child completion:
    - `/calvinxu/dm-ppert3-gsmhe-20260509-011821`: `50` failed children.
    - `/calvinxu/dm-ppert3-english-lite-20260509-011821`: `23` failed children.
    - `/calvinxu/dm-ppert3-gensmooth-20260509-011821`: `38` failed children.
    - `/calvinxu/dm-ppert3-mcq-smooth-20260509-011821`: `47` failed children.
    - `/calvinxu/dm-ppert3-agentic-bpb-20260509-011821`: `48` failed children.
  - `/calvinxu/dm-ppert3-noise-parity-20260509-011821` was still running with `0` parent failures at
    recovery time.
- Failure diagnosis:
  - GSM8K/HumanEval failures were local vLLM port collisions (`Address already in use` on port `8000`).
  - English-lite and generative-smooth failures were mostly HuggingFace Hub rate limits.
  - MCQ-smooth failures included TPU startup/SIGSEGV and transient GCS/HF-checkpoint reads.
  - Agentic-coding failures included HuggingFace Hub rate limits. Background W&B artifact warnings about
    `scored_documents.parquet` were observed but were not the primary fatal error for completed rows.
- Collection correction:
  - The first collection attempt was unsafe because local retry state enumeration no longer matched the
    remote ppert3 executor enumeration. Numeric `eval_key` matching could attach the wrong GCS result to a
    local row after baseline-anchor skips shifted row IDs.
  - Correct collection must match executor outputs by stable `(panel, run_name)`, not by the numeric
    `eval_key` prefix.
  - Rebuilt the local result CSVs by scanning GCS executor-status paths and joining to
    `proportional_perturbation_eval_candidates.csv` on `(panel, run_name)`.
- Corrected local collection counts:
  - `ppert_gsm8k_humaneval_eval_results.csv`: `62` success, `50` failed.
  - `ppert_english_lite_eval_results.csv`: `89` success, `23` failed.
  - `ppert_generative_smooth_proxy_eval_results.csv`: `74` success, `38` failed.
  - `ppert_mcq_smooth_proxy_eval_results.csv`: `65` success, `47` failed.
  - `ppert_agentic_coding_bpb_results.csv`: `64` success, `48` failed.
- Wrote failure-only retry states locally and uploaded them to:
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer_20260507/retry_states/`.
- Retry parents submitted:
  - `/calvinxu/dm-ppert3-retry-gensmooth-20260509-085737`
  - `/calvinxu/dm-ppert3-retry-mcq-smooth-20260509-085737`
  - `/calvinxu/dm-ppert3-retry-gsmhe-20260509-090533`
  - `/calvinxu/dm-ppert3-retry-english-lite-20260509-090722`
  - `/calvinxu/dm-ppert3-retry-agentic-bpb-20260509-090806`
- Retry launch policy:
  - Used lower concurrency for HF/vLLM-heavy suites to avoid immediately reproducing the observed
    rate-limit and port-collision failures: GSM8K/HumanEval `32`, English-lite `16`, generative-smooth
    `16`, agentic BPB `16`.
  - Used MCQ-smooth `64` because its original failures were mostly TPU-side rather than HF-side.
  - Do not regenerate a full ppert3 DAG for recovery. Continue using failure-only retry states and
    stable `(panel, run_name)` collection.
- Retry follow-up:
  - MCQ-smooth and generative-smooth retries started before a Levanter retry-classifier patch and still
    showed transient GCS/HF checkpoint-read failures while loading tokenizer/config metadata.
  - Patched `lib/levanter/src/levanter/compat/hf_checkpoints.py` so `_hf_hub_retry` treats the observed
    gcsfs transient (`TypeError: the JSON object must be str, bytes or bytearray, not NoneType`) and
    wrapped `gs://...` tokenizer/config load failures as retryable.
  - The running retry parents will not pick up this code change. Salvage their successes first, then run a
    second failure-only retry for any remaining failed rows using the patched code and lower concurrency
    if needed.
  - First MCQ retry salvaged `16/46` evaluation rows. Built
    `ppert_mcq_smooth_proxy_retry2_state.csv` for the remaining `30` rows and uploaded it to the same GCS
    retry-state directory.
  - Submitted patched low-concurrency MCQ retry:
    `/calvinxu/dm-ppert3-retry2-mcq-smooth-20260509-092123`, with `--max-concurrent 8`.
  - First generative-smooth retry salvaged `24/38` rows. Built
    `ppert_generative_smooth_proxy_retry2_state.csv` for the remaining `14` rows and uploaded it to the
    GCS retry-state directory.
  - Submitted patched low-concurrency generative-smooth retry:
    `/calvinxu/dm-ppert3-retry2-gensmooth-20260509-092642`, with `--max-concurrent 8`. The submitter
    timed out once and then reported `JobAlreadyExists`, but the job exists and is running.

### 2026-05-09 - conceptual interpretation of local perturbations
- One-at-a-time perturbations around `baseline_proportional` can be interpreted as noisy finite-difference
  estimates of local directional derivatives on the mixture simplex, not as global domain Shapley values.
- The clean thought experiment is projected gradient descent on the simplex:
  1. Pick a base mixture.
  2. Estimate local marginal effects by small mass transfers that preserve simplex constraints.
  3. Move in the descent direction for BPB or ascent direction for the aggregate task score.
  4. Re-measure at the new mixture and repeat with an adaptive step size.
- With infinite compute and wall-clock time, this is a strong optimization method because it directly
  queries the deployed training objective at the target scale. It is weaker than exhaustive uniform
  coverage of the full simplex only in the sense that it can follow a local basin unless paired with
  restarts or global exploration.
- Practical limits are exactly the expensive parts of the thought experiment: metric noise, finite
  perturbation size, local curvature, simplex-boundary behavior, and cost/latency of repeated training.
- The surrogate-model workflow is a compute-efficient approximation to that ideal:
  - randomized swarm rows estimate the broader response surface,
  - perturbation rows provide local derivative checks near a strategically important baseline,
  - noise baselines determine which apparent gradients are likely reliable,
  - follow-up targeted interventions can break incidental correlations in the original design.

### 2026-05-09 - ppert3 retry recovery status after GCS-state fix
- The first retry3 submission attempt used local `--state-csv` paths. Iris did not bundle those untracked
  local CSVs, so the affected parents failed immediately with `FileNotFoundError`.
- Fixed by uploading retry states to:
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer_20260507/retry_states/`
  and resubmitting with GCS `--state-csv` paths.
- Corrected retry parents:
  - `/calvinxu/dm-ppert3-retry3-gsmhe-20260510-055827`
  - `/calvinxu/dm-ppert3-retry3-english-lite-20260510-055906`
  - `/calvinxu/dm-ppert3-retry3-mcq-smooth-20260510-055944`
  - `/calvinxu/dm-ppert3-retry2-agentic-bpb-20260510-060028`
  - `/calvinxu/dm-ppert3-retry-noise-parity-20260510-060108`
- Local overlays merged after successful retries:
  - `ppert_gsm8k_humaneval_eval_results.csv`: `112/112` collected.
  - `ppert_english_lite_eval_results.csv`: `112/112` collected.
  - `ppert_mcq_smooth_proxy_eval_results.csv`: `112/112` collected.
  - `ppert_agentic_coding_bpb_results.csv`: `112/112` collected.
- Remaining active job:
  - `/calvinxu/dm-ppert3-retry-noise-parity-20260510-060108`
  - It is healthy and running with `--max-concurrent 16`.
  - The active children are long-running loglikelihood jobs of about `219M` tokens each, with roughly
    `45` minutes remaining on the first wave at the last check.

### 2026-05-11 - downstream collection and maximin diagnostic
- Collected ppert downstream result CSVs after retry completion.
  - GSM8K/HumanEval, English-lite, generative smooth proxies, MCQ smooth proxies, and agentic BPB are
    complete at `112/112` local rows.
  - Noise/parity was recollected from both original and retry prefixes:
    `ngd3dm2_ppert3_noise_parity_20260509` and `ngd3dm2_ppert3_retry_noise_parity_20260509`.
  - Recollection improved noise/parity coverage from `55/112` successful rows to `97/112`; `15` rows
    remain `executor_not_success` and need one more failure-only retry before aggregate scoring is fully
    complete for all perturbations.
  - Wrote the remaining failure-only local retry state:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_retry4_state.csv`.
- Proportional variable-subset noise training finished at both scales.
  - `20/20` rows succeeded, all with final HF checkpoints.
  - Local training-metric collection wrote:
    `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_variable_subset_noise_collected_20260511.csv`.
  - `eval/uncheatable_eval/bpb` variable-subset standard deviation:
    `0.001140` at `60m_1p2b`, `0.001188` at `300m_6b`.
- Added maximin/Pareto diagnostic:
  `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_issue5416_maximin_300m.py`.
  - Output directory:
    `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/issue5416_maximin_300m_20260511/`.
  - On the `242` signal rows and `26` issue #5416 selected items, no non-proportional observed mixture
    strictly or weakly improves over proportional on all items.
  - Best non-proportional row by number of improved items is `baseline_stratified`: improves `13/26`
    items, worsens `13/26`, and improves the issue #5416 aggregate by `+0.2193`.
  - Best non-proportional row by worst signal-scaled item delta is `run_00075`, but it still worsens
    `17/26` items and has lower aggregate than proportional.
  - Convex-hull diagnostic over observed non-proportional metric vectors still has negative maximin
    margin, so the observed cloud does not contain an all-item Pareto-improving direction over
    proportional under this item set.

### 2026-05-11 - effective-exposure DSP perturbation agreement
- Updated `experiments/domain_phase_mix/exploratory/two_phase_many/proportional_perturbation_scale_transfer_analysis.py`
  so the perturbation-agreement diagnostics use `dsp_effective_exposure_penalty_nnls` as the primary DSP
  model.
- Regenerated:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/proportional_perturbation_scale_transfer_analysis.html`.
- Effective-exposure DSP agreement with observed 100M domain-bump BPB effects:
  - finite perturbation Pearson `0.9069`, Spearman `0.6435`, sign agreement `36/39`.
  - local directional Pearson `0.6779`, Spearman `0.4933`, sign agreement `29/39`.
- Scale-specific finite perturbation predictions:
  - `60m_1p2b`: `37/39` signs agree.
  - `300m_6b`: `36/39` signs agree.

### 2026-05-16 - parity backfill and raw-PPL SNR overlay
- Completed the remaining perturbation noise/parity backfill with failure-only retry:
  `/calvinxu/dm-ppert-noise-parity-retry6-20260516-060847`.
  - Collected from original prefix `ngd3dm2_ppert3_noise_parity_20260509` plus retry prefix
    `ngd3dm2_ppert3_noise_parity_retry6_20260515`.
  - Local overlay:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_eval_results.csv`.
  - Result: `15/15` retry rows collected, `0` remaining missing rows, `863` `lm_eval/` metric columns.
- Added David/#5005 raw-PPL priority eval coverage for the 300M signal/noise matrix.
  - Canary succeeded: `/calvinxu/dm-300m-raw-ppl-canary-20260516-061950`.
  - First full wave `/calvinxu/dm-300m-raw-ppl-firstwave-20260516-062925` produced `66` valid
    summaries before failing on FineWeb2 HF rate limits and W&B artifact finalization.
  - Recovered with priority-only retry
    `/calvinxu/dm-300m-raw-ppl-priority-retry1-20260516-073839`, collecting the remaining `196`
    rows with a fresh executor prefix.
  - Local overlay:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_raw_ppl_completion/300m_raw_ppl_eval_results.csv`.
  - Result: `262/262` rows collected over `242` signal + `10` fixed noise + `10` variable noise,
    with `40` selected raw-PPL BPB datasets fully non-null.
- Deferred raw-PPL slices from this priority pass:
  - `formal_methods/tptp`: current materializer OOMs before byte cap.
  - `bio_chem/refseq/refseq_viral_fasta` and `bio_chem/refseq/refseq_viral_gff`: stale NCBI viral GFF
    source URL affects the shared RefSeq step.
  - `gh_archive_structured_output/WorkflowRunEvent`, `hardware_rtl/rtl_coder`, and `hardware_rtl/rtl_repo`:
    current bounded sample surfaces produce no scored metric rows, so they are not required for
    failure-only retry completeness.
- Rebuilt local 300M artifacts:
  - `raw_metric_matrix_300m.csv`: `242` rows, `40` raw-PPL BPB columns complete.
  - `noise_baseline_run00097_fixed_subset_300m.csv`: `10` rows, `40` raw-PPL BPB columns complete.
  - `noise_baseline_run00097_variable_subset_300m.csv`: `10` rows, `40` raw-PPL BPB columns complete.
  - `raw_metric_matrix_300m_with_noise.csv`: `262` rows, `40` raw-PPL BPB columns complete.
  - `metric_registry/metrics_wide.csv`: local registry refresh with `969` runs, `1,279` canonical
    metrics, `486,157` canonical metric facts, and `0` conflicts.
  - `paper_plots/img/metric_snr_summary.html/.png/.pdf`: refreshed with `40` raw-PPL items included.
- SNR highlights from `eval_signal_to_noise_all_metrics_300m_current.csv`:
  - `signal_rows=242`, `noise_rows=10`, `shared_metric_count=1311`, `noise_subset_mode=variable`.
  - Top raw-PPL item: `asr_ocr_noisy_ppl/rtm_sgt_ocr_v1_train/clean` with SNR `18.40`.
  - Other high raw-PPL items include `bio_chem/rnacentral/rnacentral_active_fasta` with SNR `6.57`,
    `asr_ocr_noisy_ppl/hypr_librispeech_without_lm_test_clean/clean` with SNR `5.89`, and
    `formal_methods/coqgym` with SNR `5.73`.

### 2026-05-18 - MCQ smooth-proxy perturbation controllability
- Goal: use the completed 300M proportional perturbation MCQ smooth-proxy evals to inspect local
  directional finite differences around proportional.
- Notebook:
  `experiments/domain_phase_mix/exploratory/two_phase_many/perturbation_mcq_smooth_controllability.py`.
- Output directory:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_mcq_smooth_controllability_20260518/`.
- Data coverage:
  - `ppert_mcq_smooth_proxy_eval_results.csv` has all `55/55` 300M perturbations for
    `medmcqa_5shot`, `sciq_5shot`, `swag_0shot`, `truthfulqa_mc1_0shot`, and
    `truthfulqa_mc2_0shot`.
  - Domain-effect table has `39` domain bumps x `5` tasks x `7` metric leaves = `1248`
    complete rows.
- Method:
  - Treat domain bumps as directional finite differences, not unconstrained partial derivatives.
  - Orient metrics so positive means better: probabilities/logprobs positive, BPB/NLL negative.
  - Noise-scale effects by the fixed-subset `run_00097` 300M noise baseline. Exported CSVs also
    include slopes using intervention TV as the denominator (`0.05` for current domain/family bumps).
  - Keep all metric leaves in CSVs, but headline plots omit redundant `logprob`/`nll` columns and
    show `bpb` instead.
- Findings:
  - `medmcqa_5shot` is strongly locally controllable for BPB/logprob-style proxies. `bpb` has
    median `|z|=2.18`, max `|z|=5.69`, and `56%` of domain bumps exceed `|z|>=2`.
  - `swag_0shot` is strongly locally controllable for normalized-choice proxies. `choice_prob_norm`
    has median `|z|=2.29`, max `|z|=5.10`, and `56%` of domain bumps exceed `|z|>=2`.
  - `sciq_5shot` is effectively flat at this perturbation scale: `choice_prob_norm` max `|z|=0.71`
    and BPB max `|z|=0.34`.
  - `truthfulqa_mc1_0shot` and `truthfulqa_mc2_0shot` are weak under these domain bumps; most smooth
    proxies have median `|z|<1` and only isolated domains exceed `|z|>=2`.
  - Highest-potency domains across displayed MCQ smooth proxies include `dolmino_olmocr_pdfs_hq`,
    `dolma3_cc/games_low`, `dolmino_stem_heavy_crawl`, `dolma3_cc/literature_high`, and
    `dolmino_synth_instruction`; signs are not uniformly helpful.
- CC review:
  - Invoked Claude Code Opus 4.7 Max with `env -u ANTHROPIC_API_KEY`; account preflight showed
    `plambdafour@proton.me` / `stripe_subscription`.
  - Review agreed with the directional finite-difference framing and highlighted three issues:
    slope fallback should use `tv_distance`, redundant BPB/NLL/logprob should not be triple-counted
    in headline plots, and non-domain bars should not stack multiple metrics. All three were fixed.

### 2026-05-20 - packet coverage completion pass
- Goal: complete perturbation metric coverage before preparing the ChatGPT Pro packet for SNR and
  projected-controllability analysis.
- Local registry fixes:
  - Added checkpoint-root metadata repair in
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_metric_registry.py`
    so retry rows with missing `scale`/intervention metadata join against
    `proportional_perturbation_eval_candidates.csv` instead of creating `scale=nan` registry keys.
  - Added
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/collect_proportional_perturbation_training_eval_metrics.py`.
  - Collected `ppert_training_eval_metrics.csv`: `112/112` rows and `60` training-time `eval/*`
    columns from checkpoint `eval_metrics.jsonl`/`tracker_metrics.jsonl`.
  - Local issue #5416 coverage check after the training-eval overlay: all `26` selected aggregate
    columns are present; `15/110` perturbation rows are complete before the full parity rerun. The
    remaining missing columns are the parity aliases `lm_eval/piqa/choice_logprob`,
    `lm_eval/arc_easy/choice_logprob`, `lm_eval/mmlu_sl_verb_5shot/choice_logprob`, and
    `lm_eval/hellaswag_0shot/choice_logprob`.
- Guardrail: two direct local launcher attempts were stopped before treating them as real submissions:
  - `ngd3dm2_ppert_noise_parity_full_20260520` fell back to `LocalClient` and began local lm-eval
    processes, so PIDs `71093`/`71094` were killed.
  - `ngd3dm2_ppert_raw_ppl_full_20260520` was materializing large HF-backed raw eval datasets locally
    and had not reached a valid Iris parent submission, so PIDs `61701`/`61702` were killed.
  - Ignore both prefixes for collection unless a later manual audit proves they contain intentional
    complete outputs.
- Correct full parity alias backfill submission:
  - Intended prefix:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry1_20260520`.
  - Shape: `112` candidates (`55` 60M perturbations, `55` 300M perturbations, `2` proportional anchors).
  - Tasks: `mmlu_5shot`, `mmlu_sl_verb_5shot`, `mmlu_pro_5shot`, `arc_easy` 10-shot, `piqa` 10-shot,
    `sciq_0shot`, and `hellaswag_0shot`.
  - Submission command wraps the launcher in `uv run iris --cluster marin job run --no-wait` with
    parent resources `cpu=1`, `memory=16GB`, `disk=20GB`, `region=us-east5`, and `zone=us-east5-a`.
  - Status at note time: Iris controller RPCs were timing out/resetting during `LaunchJob`; the submit
    client was left retrying for `/calvinxu/dm-ppert-noise-parity-full-retry1-20260521-024744` rather
    than starting duplicate parents. A read-only `iris query` for the same job prefix also timed out on
    `ExecuteRawQuery`, so there was no confirmed controller-side parent job at this checkpoint.
- Correct raw-PPL perturbation completion plan:
  - Intended fresh prefix:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry1_20260520`.
  - Shape: `112` candidates and `55` raw-PPL datasets (`priority` plus `fineweb2-representative`).
  - Do not start this until Iris controller RPCs are healthy enough to submit the parent job through
    `uv run iris --cluster marin job run`; direct local launcher execution is not a valid live path.

### 2026-05-21 - perturbation coverage submission after Iris recovery
- Fieldbook tracking:
  - Experiment: `exp_01ks48kds4x1s1gbqhg0skatdc`
    (`Proportional perturbation coverage completion`).
  - Ledger: `.experiments/ledger.sqlite`.
- Shared candidate CSV:
  - Uploaded local
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/proportional_perturbation_eval_candidates.csv`
    to
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer/proportional_perturbation_eval_candidates_20260520.csv`.
  - Verified shape: `112` rows (`55` 60M perturbations, `55` 300M perturbations, and `2`
    proportional baseline anchors).
- Retry history:
  - `dm-ppert-noise-parity-full-retry1-20260521-024744`: parent was created, but failed during Iris
    bundle staging with a bundle fetch connection reset.
  - `dm-ppert-noise-parity-full-retry2-20260521-034135`: parent succeeded as a no-op because the
    perturbation candidate env was not passed to the remote parent, so it saw the default noise panels.
  - `dm-ppert-raw-ppl-full-retry1-20260521-034249` and
    `dm-ppert-noise-parity-full-retry3-20260521-034507`: failed because the remote parent could not
    read the local-only perturbation candidate CSV path.
- Correct live submissions:
  - Parity aliases:
    `/calvinxu/dm-ppert-noise-parity-full-retry4-20260521-034828`
    with prefix
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry4_20260520`.
  - Raw-PPL:
    `/calvinxu/dm-ppert-raw-ppl-full-retry2-20260521-034828`
    with prefix
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry2_20260520`.
  - Both pass `MARIN_EXTRA_EVAL_CANDIDATES_CSVS` as the east5 GCS CSV URI and
    `MARIN_300M_CANDIDATE_PANELS` as
    `proportional_perturbation_60m_1p2b,proportional_perturbation_300m_6b,proportional_baseline_anchor_60m_1p2b,proportional_baseline_anchor_300m_6b`.
- Current state at submission check:
  - Parity retry4 parent is running and has created `113` child/cache jobs:
    `102` pending, `11` running, `1` succeeded. Pending children are waiting on `v5p-8`
    capacity in `us-east5-a`.
  - Raw-PPL retry2 parent is running and has created `65` child jobs so far, all pending behind the
    same `v5p-8` capacity constraint; the parent is still active.
- Next action:
  - After children finish, collect from retry4/retry2 prefixes, rebuild the perturbation coverage
    overlays, and use a failure-only retry if individual children fail transiently.

### 2026-05-22 - perturbation eval collection and transient-only retry
- Active Fieldbook experiment: `exp_01ks48kds4x1s1gbqhg0skatdc`
  (`Proportional perturbation coverage completion`).
- Collection from existing retry prefixes:
  - Parity aliases from retry4 + retry5:
    `/tmp/ppert_parity_collected_retry4_retry5.csv` has `39` rows:
    `30` collected and `9` executor-not-success.
  - Raw-PPL from retry2 + retry3:
    `/tmp/ppert_raw_ppl_collected_retry2_retry3.csv` has `68` retry3 rows:
    `32` collected and `36` missing summaries.
- Failure triage:
  - Parity retry5 failures split into `2` transient `SIGSEGV/add_port.cc` failures and `7`
    missing-HF-config failures.
  - Raw-PPL retry3 failures split into `19` transient failures (`16` GCS/HF rate-limit or east5
    egress quota, `3` `SIGSEGV/add_port.cc`) and `17` missing-HF-config failures.
  - Missing-HF-config rows are not eval-retryable; they need HF export/checkpoint repair before more
    eval attempts.
- Local code repair:
  - Restored `experiments/evals/long_tail_ppl_runnable.py` as a compatibility registry for the
    historical `long_tail_ppl_runnable/...` dataset keys used by raw-PPL state/result files.
  - Verified `uv run python -m py_compile experiments/domain_phase_mix/launch_300m_raw_ppl_evals.py
    experiments/evals/long_tail_ppl_runnable.py`.
- Transient-only retry state files:
  - Local parity state:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_noise_parity_completion/ppert_noise_parity_retry6_transient_state.csv`
    (`2` rows).
  - Local raw-PPL state:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_raw_ppl_completion/ppert_raw_ppl_retry4_transient_state.csv`
    (`19` rows).
  - Uploaded GCS copies:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer/ppert_noise_parity_retry6_transient_state_20260522.csv`
    and
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/proportional_perturbation_scale_transfer/ppert_raw_ppl_retry4_transient_state_20260522.csv`.
- Guardrail: two attempted local/remote submissions were superseded and should be ignored:
  - `ngd3dm2_ppert_noise_parity_full_retry6_20260522` fell back to `LocalClient`; the local process
    was killed.
  - Iris parents `dm-ppert-noise-parity-full-retry6b-20260522-0222` and
    `dm-ppert-raw-ppl-full-retry4b-20260522-0223` used local-only state CSV paths and failed before
    creating useful eval children.
- Correct active transient retries:
  - Parity: `/calvinxu/dm-ppert-noise-parity-full-retry6c-20260522-0224`, prefix
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry6c_20260522`.
    Immediate check: parent running, cache step succeeded, both eval children pending on `v5p-8`
    capacity in `us-east5-a`.
  - Raw-PPL: `/calvinxu/dm-ppert-raw-ppl-full-retry4c-20260522-0225`, prefix
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry4c_20260522`.
    Immediate check: parent running with `4` raw-PPL children pending on `v5p-8` capacity in
    `us-east5-a` (`max_concurrent=4` to reduce east5 egress-quota pressure).
- Next action:
  - Monitor retry6c/retry4c to terminal state.
  - Collect from retry4/retry5/retry6c for parity and retry2/retry3/retry4c for raw-PPL.
  - Separately plan HF export/checkpoint repair for the missing-HF-config rows before claiming full
    perturbation downstream coverage.

### 2026-05-22 - perturbation eval collection after transient retries
- Iris terminal-state check:
  - `/calvinxu/dm-ppert-noise-parity-full-retry6c-20260522-0224`: `4` succeeded cells, `0` failed.
  - `/calvinxu/dm-ppert-raw-ppl-full-retry4c-20260522-0225`: `20` succeeded cells, `0` failed.
- Collection commands:
  - Parity retry5-state collection from retry4/retry5/retry6c wrote
    `/tmp/ppert_parity_collected_retry4_retry5_retry6c.csv`: `30/39` collected.
  - Parity transient-state collection from retry6c wrote `/tmp/ppert_parity_collected_retry6c.csv`:
    `2/2` collected.
  - Raw-PPL retry3-state collection from retry2/retry3 wrote
    `/tmp/ppert_raw_ppl_collected_retry2_retry3.csv`: `32/68` collected.
  - Raw-PPL transient-state collection from retry4c wrote `/tmp/ppert_raw_ppl_collected_retry4c.csv`:
    `19/19` collected.
- Merged local overlays:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_eval_results.csv`:
    `112` rows total, `105` collected and `7` remaining failed rows.
  - `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_raw_ppl_eval_results.csv`:
    `112` rows total, `95` collected and `17` remaining missing-summary rows.
- Registry refresh:
  - Ran
    `uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry --no-gcs`.
  - Output summary: `983` runs, `1,354` canonical metrics, `595,622` canonical metric facts,
    `0` conflicts.
- Coverage after refresh:
  - Issue #5416 selected aggregate columns are complete for `54/55` 300M perturbations and
    `49/55` 60M perturbations.
  - Raw-PPL BPB columns are complete for `55/55` 300M perturbations and `38/55` 60M perturbations.
  - Remaining issue #5416 parity gaps are `ppert_qswap_crime_and_law` at 300M and six 60M
    perturbation rows; remaining raw-PPL gaps are `17` 60M rows.
- Interpretation:
  - Transient retries are fully collected.
  - The remaining gaps are consistent with missing HF export/config rows and should be handled as
    checkpoint/HF repair before launching further eval retries.

### 2026-05-23 - remaining perturbation eval retries after HF audit
- Current local coverage before retry:
  - `ppert_noise_parity_eval_results.csv`: `105/112` rows collected; `7` rows had
    `collection_status=executor_not_success`.
  - `ppert_raw_ppl_eval_results.csv`: `95/112` rows collected; `17` rows were missing all raw-PPL
    metric columns.
- HF audit:
  - Rebuilt state from the current missing rows and checked `hf_checkpoint_latest/config.json` for
    all unique missing-row checkpoints.
  - Result: `23` unique HF checkpoint paths, `0` missing `config.json`.
  - Interpretation: the remaining rows are retryable now; do not treat the stale missing-HF diagnosis
    from 2026-05-22 as still current without rechecking.
- Retry state files:
  - Local parity state:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/retry_states_20260523/ppert_noise_parity_retry7_state.csv`
    (`7` rows).
  - Local raw-PPL state:
    `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/retry_states_20260523/ppert_raw_ppl_retry5_state.csv`
    (`17` rows).
  - GCS parity state:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_noise_parity_retry7_state.csv`.
  - GCS raw-PPL state:
    `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_raw_ppl_retry5_state.csv`.
- Dry-runs:
  - Parity command prepared exactly `7` eval steps over `7` candidate checkpoints.
  - Raw-PPL command prepared exactly `17` eval steps over `17` candidate checkpoints and selected
    `55` raw-PPL datasets from `priority+fineweb2-representative`.
- Claude Code review:
  - Reviewed with `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`.
  - Preflight showed `plambdafour@proton.me`, `stripe_subscription`, and no inherited
    `ANTHROPIC_API_KEY`.
  - Verdict: no blockers. Review specifically checked state scoping, GCS state readability, east5
    checkpoint locality, and executor prefixes.
- Submitted:
  - Parity retry7:
    `/calvinxu/dm-ppert-noise-parity-full-retry7-20260523-1630`.

    Command:
    `uv run iris --cluster=marin job run --no-wait --job-name dm-ppert-noise-parity-full-retry7-20260523-1630 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_300m_noise_parity_evals --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry7_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry7_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_noise_parity_retry7_state.csv --max-concurrent 7`
  - Raw-PPL retry5:
    `/calvinxu/dm-ppert-raw-ppl-full-retry5-20260523-1630`.

    Command:
    `uv run iris --cluster=marin job run --no-wait --job-name dm-ppert-raw-ppl-full-retry5-20260523-1630 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_raw_ppl_retry5_state.csv --bundle priority --bundle fineweb2-representative --allow-partial --max-concurrent 17`
- Fieldbook:
  - Added running jobs `job_01ksbjtnqxynxnv26at22drjex` and
    `job_01ksbjtns6k0ytphs5wwf2thcq` to experiment `exp_01ks48kds4x1s1gbqhg0skatdc`.
  - Added debug note `note_01ksbjv0w7d9sb9s3aqgqdaghq` with the retry scope and HF audit result.
- Next action:
  - Monitor both parents to terminal state.
  - Collect parity from retry4/retry5/retry6c/retry7 and raw-PPL from retry2/retry3/retry4c/retry5,
    then rebuild the metric registry.

### 2026-05-23 - retry7b/retry5b nonpreemptible parent replacement
- Issue:
  A simultaneous Grug eval retry showed a child killed by `Parent task preempted`. The just-submitted
  perturbation retry7/retry5 parents were also preemptible CPU parents, and their eval children were
  still only pending, so replacing them before TPU work began avoided likely parent-preemption loss.
- Claude Code review:
  Reviewed the stop/relaunch decision with `env -u ANTHROPIC_API_KEY`, Opus 4.7, max effort, session
  `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: stop the preemptible parents now, relaunch with
  `--no-preemptible`, and use fresh executor prefixes to avoid stale locks/status files.
- Action:
  Stopped:
  - `/calvinxu/dm-ppert-noise-parity-full-retry7-20260523-1630`
  - `/calvinxu/dm-ppert-raw-ppl-full-retry5-20260523-1630`
- Submitted:
  - Parity retry7b:
    `/calvinxu/dm-ppert-noise-parity-full-retry7b-20260523-1638`

    Command:
    `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-ppert-noise-parity-full-retry7b-20260523-1638 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_300m_noise_parity_evals --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry7b_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry7b_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_noise_parity_retry7_state.csv --max-concurrent 7`
  - Raw-PPL retry5b:
    `/calvinxu/dm-ppert-raw-ppl-full-retry5b-20260523-1638`

    Command:
    `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-ppert-raw-ppl-full-retry5b-20260523-1638 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5b_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5b_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_raw_ppl_retry5_state.csv --bundle priority --bundle fineweb2-representative --allow-partial --max-concurrent 17`
- Fieldbook:
  Marked retry7/retry5 jobs `killed`, added retry7b/retry5b jobs
  `job_01ksbkbj94f9jcpmwxp0874fy0` and `job_01ksbkbjbr65js7mptxa1y12wb`, and replaced the current
  next-action note with the retry7b/retry5b collection plan.
- Next action:
  - Monitor retry7b/retry5b to terminal state.
  - Collect parity from retry4/retry5/retry6c/retry7b and raw-PPL from
    retry2/retry3/retry4c/retry5b.

### 2026-05-23 - retry7c/retry5c parent capacity recovery
- Issue:
  retry7b/retry5b had correct nonpreemptible semantics but were pinned to
  `us-east5-a` for the parent and remained pending on parent CPU capacity. No
  useful eval child work had started.
- Claude Code review:
  Reviewed the capacity recovery with `env -u ANTHROPIC_API_KEY`, Opus 4.7,
  max effort, session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: stop
  retry7b/retry5b and relaunch with nonpreemptible placement-unconstrained CPU
  parents; retain child locality through launcher resources and checkpoint
  paths.
- Action:
  Stopped:
  - `/calvinxu/dm-ppert-noise-parity-full-retry7b-20260523-1638`
  - `/calvinxu/dm-ppert-raw-ppl-full-retry5b-20260523-1638`

  Submitted:
  - `/calvinxu/dm-ppert-noise-parity-full-retry7c-20260523-1645`
  - `/calvinxu/dm-ppert-raw-ppl-full-retry5c-20260523-1645`

  Both parents use `--no-preemptible --cpu 1 --memory 2GB --disk 10GB` and no
  parent region/zone pin. The parity retry still targets `7` rows; raw-PPL
  still targets `17` rows with the `priority+fineweb2-representative` bundle.
- Current status:
  retry7c is running, has cached eval datasets, and has `7` parity eval
  children pending on v5p-8 east5-a capacity. retry5c is running and has
  active raw-PPL materialization/eval work; the killed Zephyr worker rows seen
  under the prefix are cleanup/coordination artifacts while the parent remains
  active, not final failed eval cells yet.
- Fieldbook:
  Marked retry7b/retry5b `killed`, added running retry7c/retry5c jobs
  `job_01ksbkr57aemsp51fy0rs6n69e` and
  `job_01ksbkr54rntj1kgr9wpm7kvzs`, and updated the current next-action note.
- Next action:
  - Monitor retry7c/retry5c to terminal state.
  - Collect parity from retry4/retry5/retry6c/retry7c and raw-PPL from
    retry2/retry3/retry4c/retry5c.

### 2026-05-23 - raw-PPL retry5d after parent OOM
- Status:
  `/calvinxu/dm-ppert-raw-ppl-full-retry5c-20260523-1645` failed at the parent
  with exit `137` while materializing raw-PPL datasets. Its child jobs were
  Zephyr cache/materialization workers only; no raw-PPL eval cells had launched.
- Interpretation:
  The earlier 2GB placement-unconstrained parent was too small for
  `priority+fineweb2-representative` raw-PPL dataset materialization.
- Claude Code review:
  Reviewed the fix with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort,
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: no blockers; use a
  fresh executor prefix and restore parent memory to `16GB` with `20GB` disk.
- Dry-run:
  `uv run python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals --dry-run --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5d_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5d_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_raw_ppl_retry5_state.csv --bundle priority --bundle fineweb2-representative --allow-partial --max-concurrent 17`
  prepared `17` raw-PPL eval steps over `17` candidate checkpoints.
- Submitted:
  `/calvinxu/dm-ppert-raw-ppl-full-retry5d-20260523-2228`

  Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-ppert-raw-ppl-full-retry5d-20260523-2228 --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals --name-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5d_20260523 --executor-prefix pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5d_20260523 --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260523/ppert_raw_ppl_retry5_state.csv --bundle priority --bundle fineweb2-representative --allow-partial --max-concurrent 17`
- Fieldbook:
  Marked retry5c failed and added running retry5d as
  `job_01ksc73ez4dz0pbsknkkemc2vx`.

### 2026-05-24 - cross-region egress incident

Rohith's egress report flagged perturbation eval retries, especially
`/calvinxu/dm-ppert-raw-ppl-full-retry5d-20260523-2228` and smaller
retry7c activity. The problematic retry5d command restored parent memory but
also omitted explicit parent `--region us-east5 --zone us-east5-a` while using
east5 state CSV and executor paths. That parent placement pattern is no longer
acceptable.

Immediate status/actions:

- Stop request for
  `/calvinxu/dm-ppert-raw-ppl-full-retry5d-20260523-2228` found no running
  jobs; the prefix was already terminal with `16` failed and `2` succeeded.
- `/calvinxu/dm-ppert-noise-parity-full-retry7c-20260523-1645` was also
  terminal with `6` failed and `3` succeeded.
- Marked Fieldbook jobs `job_01ksc73ez4dz0pbsknkkemc2vx` and
  `job_01ksbkr57aemsp51fy0rs6n69e` as failed.

CC review:

- Invoked with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, resumed Marin
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`; preflight showed
  `plambdafour@proton.me`, `stripe_subscription`, and no inherited
  `ANTHROPIC_API_KEY`.
- Diagnosis: missing parent placement is the dominant root cause. A central or
  unconstrained parent can repeatedly read/write east5 GCS state, status, and
  materialization artifacts even when child eval jobs are east5-a.

Policy from this point:

- No perturbation eval retry without explicit parent
  `--region us-east5 --zone us-east5-a`.
- No placement-unconstrained CPU parent for east5 eval/materialization work,
  even for capacity recovery.
- State CSVs, executor prefixes, checkpoint roots, and eval caches must all use
  `gs://marin-us-east5`.
- Run CC review before every live retry submission.

### 2026-05-25 - collected partial retry7c/retry5d successes

Read-only collection from the latest terminal retry prefixes found additional
usable outputs:

- Parity retry7c
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry7c_20260523`
  contributed `2/7` rows:
  - `proportional_perturbation_300m_6b:ppert_qswap_crime_and_law`
  - `proportional_perturbation_60m_1p2b:ppert_domain_dolma3_cc_electronics_and_hardware_high`
- Raw-PPL retry5d
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry5d_20260523`
  contributed `2/17` rows:
  - `proportional_perturbation_60m_1p2b:ppert_domain_dolmino_synth_code`
  - `proportional_perturbation_60m_1p2b:ppert_qswap_health`

Actions:

- Merged the new collected rows into:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_eval_results.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_raw_ppl_eval_results.csv`
- Rebuilt the local metric registry with:
  `uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry --no-gcs`
- Refreshed Fieldbook artifacts and validation counts for experiment
  `exp_01ks48kds4x1s1gbqhg0skatdc`.

Current coverage:

- Parity: `107/112` collected. Remaining `5` rows are all 60M and have
  `executor_not_success`.
- Raw-PPL: `97/112` collected. Remaining `15` rows are all 60M and have
  `missing_summary`.
- The previous 300M issue #5416 parity gap
  `ppert_qswap_crime_and_law` is now collected.

Next action:
build a fresh failure-only retry state from the updated canonical CSVs, not from
the stale retry5/retry7 state files. Submit only after CC review and only with
explicit east5 parent placement plus east5-a child TPU placement.

### 2026-05-25 - submitted retry8/retry6 after CC review

Fresh state files were generated from the updated canonical result CSVs and
uploaded to:

- `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260525/ppert_noise_parity_retry8_state.csv`
- `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260525/ppert_raw_ppl_retry6_state.csv`

Dry-runs prepared exactly `5` parity eval steps and `15` raw-PPL eval steps.
CC reviewed the submission plan with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max
effort, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, and found
no blockers.

The first two parity parent attempts dispatched no children and were stopped:

- `/calvinxu/dm-ppert-noise-parity-full-retry8-20260525-1850`: pinned
  `us-east5-a`, blocked on east5-a CPU/highmem capacity.
- `/calvinxu/dm-ppert-noise-parity-full-retry8-region-20260525-1901`:
  region-only east5 parent, still blocked on the 16GB highmem parent shape.

After a focused CC review, the parity parent was resubmitted as a lightweight
coordination-only parent:

- `/calvinxu/dm-ppert-noise-parity-full-retry8-region-lite-20260525-1909`
- parent: `--region us-east5`, `--cpu 0.5`, `--memory 8GB`, `--disk 20GB`
- children: `--tpu-region us-east5 --tpu-zone us-east5-a`
- executor prefix:
  `pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry8_region_lite_20260525`

This parent started, completed the eval-cache step, and dispatched all `5`
parity eval children. The eval children are currently waiting for east5-a TPU
capacity.

Raw-PPL retry6 was submitted after parity dispatch was proven:

- `/calvinxu/dm-ppert-raw-ppl-full-retry6-region-20260525-1913`
- parent: `--region us-east5`, `--cpu 1`, `--memory 16GB`, `--disk 20GB`
- children: `--tpu-region us-east5 --tpu-zone us-east5-a`
- executor prefix:
  `pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry6_region_20260525`

The raw-PPL parent is currently waiting for the east5 highmem CPU worker scale-up.
Keep the 16GB parent memory for raw-PPL because retry5c previously OOMed during
raw-PPL dataset materialization. The updated operational rule is: use explicit
east5 placement for parent and children; a region-only east5 parent is acceptable
after CC review when a zone-pinned CPU parent is capacity-blocked, because it
keeps all GCS traffic intra-region while allowing the CPU parent to land in
another east5 zone.

### 2026-05-25 - parity retry8 collected

Live Iris check showed
`/calvinxu/dm-ppert-noise-parity-full-retry8-region-lite-20260525-1909`
finished successfully. Collected the five retry8 rows with:

`uv run python experiments/domain_phase_mix/launch_300m_noise_parity_evals.py --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260525/ppert_noise_parity_retry8_state.csv --collect-from-prefix gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_noise_parity_full_retry8_region_lite_20260525 --collect-output-csv /tmp/ppert_parity_collected_retry8.csv`

Result:

- Retry8 collection status: `5/5` collected.
- Merged those rows into
  `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_eval_results.csv`.
- Rebuilt the local registry with
  `uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry --no-gcs`.
- Updated Fieldbook job `job_01ksh0xsded45vs31g4gyab25s` to `succeeded`.
- Updated Fieldbook validation `perturbation.parity_collection` to pass:
  `112/112` collected.

Remaining gap:

- Raw-PPL remains `97/112`; retry6 is pending on east5 CPU capacity.

### 2026-05-26 - raw-PPL retry7 submitted after retry6 parent capacity stall

Raw-PPL retry6
`/calvinxu/dm-ppert-raw-ppl-full-retry6-region-20260525-1913`
remained pending for the east5 highmem CPU parent shape with no children
dispatched. I asked CC to review a replacement using the same 15-row GCS state
CSV but with a lighter coordination parent. CC was invoked with
`env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, and resumed Marin session
`d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`; subscription preflight showed
`plambdafour@proton.me`, `stripe_subscription`, and no inherited API key. CC
recommended stopping retry6 and using an 8GB parent rather than the 16GB
highmem shape.

Actions:

- Stopped retry6 before dispatch.
- Submitted `/calvinxu/dm-ppert-raw-ppl-full-retry7-region-lite-20260526-0555`.
- Parent placement: `--region us-east5`, `--cpu 0.5`, `--memory 8GB`,
  `--disk 20GB`.
- Child eval placement remains `us-east5-a`; explicit GCS inputs use
  `gs://marin-us-east5`.
- Safety gate passed with `--allow-region-only-parent`.

The first live submit attempt failed before job creation because Iris tried to
bundle large untracked local analysis artifacts. I fixed this with local-only
`.git/info/exclude` entries for reference-output and ChatGPT packet artifacts;
the bundle dropped to `15.1MB`. CC reviewed this focused bundle fix and found no
blockers before retry7 submission.

Post-submit status:

- Retry7 parent is running.
- All `15` raw-PPL children are dispatched.
- Latest check: all `15` children are running in east5-a.

Fieldbook:

- Marked retry6 as `killed`.
- Added retry7 as a running retry of retry6.
- Archived stale retry6 live validations.
- Added a passing retry7 dispatch validation and a new next-action to monitor
  retry7 to terminal, then collect raw-PPL outputs.

### 2026-05-26 - raw-PPL retry7 collected; perturbation coverage complete

Live check showed
`/calvinxu/dm-ppert-raw-ppl-full-retry7-region-lite-20260526-0555`
finished successfully: parent and all `15` raw-PPL eval children succeeded.

Collection command:

`uv run python -m experiments.domain_phase_mix.launch_300m_raw_ppl_evals --state-csv gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ppert_recovery_state_20260525/ppert_raw_ppl_retry6_state.csv --bundle priority --bundle fineweb2-representative --allow-partial --collect-from-prefix gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_ppert_raw_ppl_full_retry7_region_lite_20260526 --collect-output-csv /tmp/ppert_raw_ppl_collected_retry7.csv`

Result:

- Collected `15/15` retry7 raw-PPL rows.
- Merged them into
  `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_raw_ppl_eval_results.csv`.
- Rebuilt the local metric registry with
  `uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry --no-gcs`.
- Registry rebuild summary: `983` runs, `1,354` canonical metrics,
  `606,254` canonical metric facts, `0` conflicts.

Coverage:

- Raw-PPL perturbation coverage is now `112/112`.
- Parity perturbation coverage was already `112/112`.

Fieldbook:

- Marked retry7 as `succeeded`.
- Archived stale retry7 dispatch and raw-PPL collection warnings.
- Added a passing `perturbation.raw_ppl_collection` validation and a research
  note recording the final coverage.

### 2026-05-26 - preliminary SNR heteroskedasticity analysis

Compared the canonical global 300M SNR table against a proportional-noise
variant to test whether the noise denominator changes with the mixture anchor.
Both tables use the same 242-row global 300M signal numerator; the only intended
difference is the 10-row noise denominator:

- `run00097` variable-subset noise.
- `baseline_proportional` variable-subset noise.

Artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/snr_global_run00097_variable_noise.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/snr_global_proportional_variable_noise.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/snr_noise_anchor_comparison.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/raw_ppl_bpb_noise_anchor_comparison.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/ppert_raw_ppl_local_response_not_global_snr.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/run00097_vs_proportional_snr_notebook.py`

Headline results:

- Shared global SNR metrics: `1,142`.
- Raw-PPL BPB metrics with both noise anchors: `40`.
- Median proportional/run00097 noise-scale ratio across all shared metrics:
  `0.966`.
- Across all shared metrics, `28.6%` have proportional noise more than `1.25x`
  run00097 noise, and `32.6%` have proportional noise less than `0.8x`
  run00097 noise.
- Raw-PPL BPB has stronger anchor dependence: median ratio `0.786`, with
  `27.5%` above `1.25x` and `50.0%` below `0.8x`.
- Median global SNR is nearly unchanged overall (`1.390` run00097 vs `1.390`
  proportional), but individual metrics move substantially.

Interpretation:

- This is preliminary evidence that SNR is heteroskedastic: the noise scale is
  not just a metric property, but depends on the mixture/noise anchor.
- The result is not uniformly in one direction. Proportional is quieter for many
  metrics, but noisier for others.
- Ppert local signal-to-noise should not be compared directly to the global SNR
  tables because its numerator is a finite perturbation-response spread rather
  than the global swarm signal spread, and its denominator is the proportional
  noise proxy rather than a matched denominator per perturbation. The artifact is
  retained as a local effect-to-noise diagnostic, not an apples-to-apples SNR.

Coverage caveat:

- Ppert raw-PPL contains `55` BPB metrics. The two-noise-anchor SNR comparison
  covers `40` of them; the missing `15` are FineWeb2 representative language
  slices that do not currently have matching proportional-noise denominator
  coverage in the global SNR table.

Visualization notebook:

- Added a Marimo notebook that plots run00097's mixture weights against the
  proportional mixture, plus per-metric SNR under the run00097 and proportional
  noise denominators. The notebook also includes SNR distribution histograms
  and proportional/run00097 SNR-ratio and noise-scale-ratio distributions.
- Launch with:
  `uv run --with marimo --with pandas --with plotly --with numpy marimo edit experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/ppert_raw_ppl_snr_heteroskedasticity_20260526/run00097_vs_proportional_snr_notebook.py`
