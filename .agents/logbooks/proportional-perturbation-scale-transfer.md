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
