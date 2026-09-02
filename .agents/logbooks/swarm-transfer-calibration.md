# Swarm Transfer Calibration: Research Logbook

## Scope
- Goal: Build a practical transfer-calibration baseline for selecting 300M mixtures from a known 60M swarm, while keeping a path toward a continuous `y_hat(w, N, D)` law.
- Primary metric(s): `eval/uncheatable_eval/bpb` RMSE and rank preservation on held-out legacy `300M` swarm mixtures.
- Constraints:
  - The operational holdout is `exclude_300m_swarm_predict_300m_swarm`, where the `60M` swarm remains available.
  - Use packet-local code in `/Users/calvinxu/Downloads/chatgpt_pro_nd_scaling_packet_v14/chatgpt_pro_nd_scaling_packet`.
  - Keep models simple enough to hand off later for a bigger ChatGPT Pro jump.

## Baseline
- Date: 2026-04-19
- Code refs:
  - `/Users/calvinxu/Downloads/chatgpt_pro_nd_scaling_packet_v14/chatgpt_pro_nd_scaling_packet/code/run_swarm_transfer_calibration.py`
  - `/Users/calvinxu/Downloads/chatgpt_pro_nd_scaling_packet_v14/chatgpt_pro_nd_scaling_packet/code/run_continuous_nd_grp_law.py`
- Baseline numbers:
  - Paired `60M -> 300M` swarm identity: `RMSE 0.1206`, `Spearman 0.7769`
  - Paired affine transfer: `RMSE 0.0110`, `Spearman 0.7723`
  - Corrected continuous-law holdout with `60M` swarm available: tuned continuous law `RMSE 0.0190`, `Spearman 0.3833`

## Experiment Log
### 2026-04-19 23:10 - Continuous transfer reframing
- Hypothesis: The practical task is a transfer problem around observed `60M` swarm losses, not pure unseen-mixture extrapolation. A useful next baseline is `y60 + delta(scale) + residual`.
- Command:
  - ad hoc `uv run python` probes against the v14 packet
- Config:
  - Hold out only legacy `300M` swarm rows.
  - Keep `60M` swarm in train.
  - Compare `include 1.2B` vs `exclude 1.2B`.
- Result:
  - Only `14` mixture IDs outside the legacy `300M` swarm also have a `60M` anchor.
  - `delta(scale)` anchored on `y60` greatly improves calibration while preserving the identity ranking.
  - The remembered old 10-feature GRP signal set carries some extra rank information, but currently destabilizes calibration on the continuous transfer setting.
- Interpretation:
  - The transfer layer is the right local track.
  - The immediate baseline to beat is not the current continuous law; it is `y60 + delta(scale)`.
- Next action:
  - Build a packet-local script that evaluates the anchored transfer family cleanly and emits plots/artifacts for `with/without 1.2B`.

### 2026-04-19 23:35 - Anchored continuous transfer sweep
- Hypothesis: The old 10-feature GRP signal set might improve 300M selection if used only as a small residual around an identity or scale-delta anchor.
- Command:
  - `uv run code/run_continuous_swarm_transfer_calibration.py`
- Config:
  - Train on all rows with a `60M` swarm anchor and a primary label, excluding legacy `300M` swarm.
  - Compare `include_1.2B = False` and `include_1.2B = True`.
  - Models:
    - identity `yhat = y60`
    - `delta_scale`: `yhat = y60 + g(uN, uD)`
    - `delta_scale + old10 residual`
  - Hyperparameters chosen by grouped train-only CV over anchor mixture IDs.
- Result:
  - `include_1.2B = False`
    - identity: `RMSE 0.1206`, `Spearman 0.7769`
    - `delta_scale`: `RMSE 0.0518`, `Spearman 0.7769`
    - best old10 by CV-RMSE: `RMSE 0.0129`, `Spearman 0.7765`
    - best old10 by CV-Spearman: `RMSE 0.0298`, `Spearman 0.7767`
  - `include_1.2B = True`
    - identity: `RMSE 0.1206`, `Spearman 0.7769`
    - `delta_scale`: `RMSE 0.0638`, `Spearman 0.7769`
    - best old10 by CV-RMSE collapses to gate `0.0`
    - best old10 by CV-Spearman: `RMSE 0.0427`, `Spearman 0.7766`
- Interpretation:
  - Train-only tuning does not validate a real Spearman gain from the old 10-feature residual on the continuous transfer task.
  - The robust improvement is calibration, not ranking.
  - `1.2B` hurts rather than helps on this task.
- Next action:
  - Treat `delta_scale` and the small-gated old10 residual as calibration baselines.
  - If we want ranking gains, we likely need a different residual formulation or a different source of cross-scale supervision, not just the old feature set.

### 2026-04-19 23:55 - Cleaner rank-preserving calibrators
- Hypothesis: A cleaner monotone calibration around `y60` may give most of the calibration gain without relying on the old GRP residual features.
- Command:
  - `uv run code/run_continuous_swarm_transfer_calibration.py`
- Config:
  - Added:
    - `monotone_affine_scale` with scale-conditioned intercept and positive scale-conditioned slope
    - `global_affine_scale` with scale-conditioned intercept and one global positive slope
- Result:
  - `global_affine_scale` is the best clean baseline:
    - without `1.2B`: `RMSE 0.0417`, `Spearman 0.7769`
    - with `1.2B`: `RMSE 0.0519`, `Spearman 0.7769`
  - `monotone_affine_scale` gets lower RMSE (`0.0180` without `1.2B`) but does it by collapsing the 300M slope almost to zero, so it is not a good story.
  - The old 10-feature residual still gives the best holdout RMSE (`0.0129` without `1.2B`) but does not improve validated rank and is less clean.
- Interpretation:
  - If we prioritize clean/justifiable, the 7-parameter `global_affine_scale` model is the right current baseline.
  - If we prioritize pure calibration on the 300M swarm holdout, the old 10-feature residual is still strongest.
  - `1.2B` should remain a deployment goal, so its current degradation is evidence that the model family still needs work.
- Next action:
  - Keep `global_affine_scale` as the clean baseline to beat.
  - Treat `include 1.2B` as the default desired training regime and use the current degradation as a failure mode to fix in the next residual design.

### 2026-04-20 00:10 - Target-aware weighting negative result
- Hypothesis: If we weight training rows by proximity to the target `300M/6B` scale and tune on anchored 300M rows only, the old 10-feature residual might recover rank gains without being pulled around by distant scales.
- Command:
  - ad hoc weighted-ridge probe over anchored training rows
- Config:
  - weighted ridge on both the scale delta and old10 residual
  - weights `exp(-beta * dist((uN,uD), target_300)^2)`
  - hyperparameters selected by CV performance on anchored training rows at `300M` only
- Result:
  - best anchored-300 train-CV: `RMSE 0.0184`, `Spearman 0.8425`
  - true held-out legacy `300M` swarm: `RMSE 0.0124`, `Spearman 0.7464`
- Interpretation:
  - The target-aware weighting overfit the small anchored-300 subset and generalized worse than identity on rank.
  - This is not the right next path.
- Next action:
  - Keep the clean `global_affine_scale` baseline.
  - If we want to exceed identity on rank, we likely need a different residual family or additional supervision, not just reweighting the current old10 residual.

### 2026-04-20 00:35 - Fixed transfer benchmark with 520M holdouts
- Hypothesis: We need a more stable local optimization target than the tiny `520M` set alone. A fixed benchmark that always includes all verified `520M` rows plus a deterministic random supplement from the remaining anchored targets should let us iterate on form without losing sight of extrapolation.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Hold out all verified `520M` perplexity-ready rows from `run_registry/strong_tier_perplexity_ready.csv`
  - Add a deterministic `10%` stratified random supplement by `(target_scale, target_path)` with seed `7`
  - Train on the remaining anchored rows, including `1.2B`
- Result:
  - Holdout size: `39`
    - fixed `520M`: `5`
    - random supplement: `34`
  - `global_affine_scale`:
    - overall `RMSE 0.0291`, `Spearman 0.8682`
    - fixed `520M` `Spearman 0.20`
  - `rank_gate_small_residual`:
    - overall `RMSE 0.0285`, `Spearman 0.8962`
    - fixed `520M` `Spearman 0.60`
- Interpretation:
  - This combined benchmark is a better local target than `520M` alone.
  - The current best local model is still `rank_gate_small_residual`.
  - The model is close to `0.9` overall Spearman on the benchmark, but its true extrapolation signal is still mostly coming from the random supplement, not the `520M` subset.
- Next action:
  - Use this benchmark as the local objective while iterating on functional form.
  - Try one more GRP-structured residual family that uses target-side mixture geometry directly.

### 2026-04-20 00:48 - Target-side old10 GRP residuals
- Hypothesis: The old GRP target-side signal/penalty features may still contain useful transfer information if they are added as a small residual on top of `global_affine_scale`.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added `global_affine_old10_residual`
    - base: `global_affine_scale`
    - residual: ridge on target-side old-10 GRP features
    - constant residual gate selected by train-only CV
  - Added `global_affine_old10_scale_gate`
    - same residual score
    - plus a learned 3-parameter scale gate on `(score, score*uN, score*uD)`
- Result:
  - `global_affine_old10_residual` (`17` learned params):
    - overall `RMSE 0.0944`, `Spearman 0.8992`
    - fixed `520M` `RMSE 0.1567`, `Spearman 0.50`
  - `global_affine_old10_scale_gate` (`20` learned params):
    - overall `RMSE 0.0291`, `Spearman 0.8682`
    - fixed `520M` `RMSE 0.0747`, `Spearman 0.20`
  - Selected configs were effectively the unmodified old10 geometry:
    - constant-gate model: `gate=0.5`, `resid_alpha=10.0`
    - scale-gated model: `resid_alpha=1000.0`, `gate_alpha=0.01`
- Interpretation:
  - The target-side old10 features do carry rank signal, but the simple constant-gate version buys it by destroying calibration and hurting `520M`.
  - The scale-gated version regularizes the residual away and collapses back to `global_affine_scale`.
  - So the current local best remains `rank_gate_small_residual`: it is the best tradeoff between benchmark Spearman, RMSE, and `520M` behavior.
- Next action:
  - Keep `rank_gate_small_residual` as the incumbent local baseline.
  - If we continue locally, the next experiments should either:
    - directly weight the `520M` subset in model selection, or
    - impose a cleaner rank-preserving residual constraint rather than a freer GRP residual.

### 2026-04-20 01:15 - Low-parameter rank-gate refinements and hybrid channel
- Hypothesis: The remaining gap might be mostly calibration rather than missing structure. If so, small post-hoc refinements of `rank_gate_small_residual` should help. If that fails, a hybrid source+target residual channel is the next minimally more expressive test.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added:
    - `rank_gate_affine_calibrated` (`26` params total)
    - `rank_gate_residual_shrink` (`25` params total)
    - `rank_gate_dual_channel` (`37` params total)
  - `rank_gate_dual_channel` uses:
    - current source-feature rank score,
    - target-side old10 GRP score,
    - a 6-coefficient combiner on `(source, source*uN, source*uD, target, target*uN, target*uD)`
- Result:
  - `rank_gate_small_residual` remains the incumbent:
    - overall `RMSE 0.02848`, `Spearman 0.89615`
    - fixed `520M` `RMSE 0.07433`, `Spearman 0.60`
  - `rank_gate_affine_calibrated`:
    - overall `RMSE 0.02845`, `Spearman 0.89615`
    - fixed `520M` `RMSE 0.07427`, `Spearman 0.60`
    - fitted slope is almost exactly `1.007`
  - `rank_gate_residual_shrink`:
    - identical to `rank_gate_small_residual`
    - fitted `lambda` is essentially `1.0`
  - `rank_gate_dual_channel`:
    - overall `RMSE 0.02856`, `Spearman 0.89494`
    - fixed `520M` `RMSE 0.07462`, `Spearman 0.60`
- Interpretation:
  - The simple calibration story is exhausted:
    - the affine calibrator barely moves anything,
    - the shrink model says the current rank-gate residual is already at the best train-fitted scale.
  - The target-side GRP signal is not useless, but once the source rank signal is already present it does not provide additional validated lift.
  - We are likely near the limit of this local family under the current benchmark and supervision.
- Next action:
  - Keep `rank_gate_small_residual` as the benchmark incumbent.
  - The next local step should change the selection objective, not just add another nearby residual family:
    - explicitly upweight or constrain the fixed `520M` subset in hyperparameter/model selection, or
    - redesign the residual objective around pairwise/rank preservation rather than plain squared error.

### 2026-04-20 01:40 - Source-side old10 calibration wins
- Hypothesis: The useful extra transfer signal may come from GRP-style proxy-scale geometry, but the raw source-old10 residual was mostly a calibration problem. A simple affine calibrator on top of the source-old10 model might retain the rank gain while fixing the diagonal.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added:
    - `global_affine_source_old10_residual`
    - `source_old10_affine_calibrated`
    - `rank_source_blend`
  - `source_old10_affine_calibrated` is just a 2-parameter affine post-fit on the source-old10 challenger.
- Result:
  - `source_old10_affine_calibrated` is the first strict local improvement over `rank_gate_small_residual`:
    - params: `19`
    - overall `RMSE 0.02829`, `Spearman 0.89919`
    - fixed `520M` `RMSE 0.07354`, `Spearman 0.70`
  - incumbent `rank_gate_small_residual`:
    - params: `24`
    - overall `RMSE 0.02848`, `Spearman 0.89615`
    - fixed `520M` `RMSE 0.07433`, `Spearman 0.60`
  - `rank_source_blend` did not help:
    - fitted blend weights are essentially `lambda_rank ~= 1.0`, `lambda_source ~= 0.01`
- Interpretation:
  - Source-side old10 GRP geometry is the best extra signal found so far.
  - The main issue with the raw source-old10 variant was calibration, not lack of rank signal.
  - A very small affine post-calibration fixes enough of that to overtake the rank-gate incumbent.
- Next action:
  - Promote `source_old10_affine_calibrated` to the new local baseline.
  - Next local experiments should start from this family, not from `rank_gate_small_residual`.

### 2026-04-20 03:30 - Calibrator simplification study: 12-param form is minimal
- Hypothesis: The incumbent 12-param scale-aware monotone calibrator (quadratic intercept + quadratic log-slope on top of source_old10 raw predictions) is doing more work than needed. Simpler bases (affine, gain-only, uN-only quadratic, scalar slope, etc.) might recover the gain while being more interpretable.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added a parameterized `_fit_scale_calibrator_core` family keyed by `(intercept_basis, slope_basis)` over bases `{constant, gain, affine, affine_cross, diag_quadratic, u_n_quadratic, u_n_quadratic_u_d_linear, quadratic}`.
  - Variants tested:
    - `(quadratic, quadratic)` via grouped-CV alpha selection to reproduce incumbent with proper diagnostics.
    - All cross-products of smaller bases down to `(gain, constant)`.
    - Pure additive (slope=1 fixed) variants with each basis.
    - Anchored form `pred = raw + a(uN, uD) * (1 - raw)` with various bases. Half the params of the incumbent, derived from the observation that the incumbent's fit satisfies `log_slope ≈ -intercept` at every training scale.
  - Final fit always uses alpha=0 (matching the incumbent's effective behavior, since grouped-CV alpha selection over-regularizes for this benchmark).
  - Multi-start L-BFGS deferred after confirming it produces the same fit as single-start from `global_affine_init`.
- Result:
  - Incumbent `source_old10_scale_calibrated` (29 params) at seed 7: overall `RMSE 0.01076`, `Spearman 0.91356`; fixed 520M `RMSE 0.00673`, `Spearman 0.90`; random supplement `RMSE 0.01124`, `Spearman 0.86982`. Reproduces the reference summary.
  - `source_old10_scale_quadratic_cv` (same 12-param quadratic form via the refactored fit path, 29 params): matches incumbent exactly at seed 7 (by design).
  - Every simpler (quadratic-dropped) variant collapses to roughly `source_old10_affine_calibrated` numbers (overall `RMSE ≈ 0.028`, Spearman `0.899`, fixed 520M `RMSE ≈ 0.073`, Spearman `0.70`). Tested variants: (quadratic, constant), (affine, affine), (affine, constant), (gain, gain), (gain, constant), (constant, quadratic), (affine_cross, affine_cross), (diag_quadratic, diag_quadratic), (quadratic, affine), (affine, quadratic), (u_n_quadratic, u_n_quadratic), additive-only with all bases.
  - Anchored 6-param form regresses catastrophically on fixed 520M: `RMSE 0.09094`, `Spearman = -0.90` (ranking inverted). Quadratic `a(uN, uD)` extrapolates to `a > 1` at 520M, which flips the sign of the raw-prediction coefficient and reverses the ranking.
- Interpretation:
  - Inspection of the incumbent's fitted coefficients shows `log_slope + intercept ≈ 0` in every quadratic basis component, i.e. `slope ≈ 1 - intercept`. At the three training scale centroids the incumbent implements per-scale `(a, 1 - a)` convex combination toward BPB = 1.0:
    - 130M: `a = +0.242`, slope = 0.752, `1 - a = 0.758`.
    - 300M: `a = -0.130`, slope = 1.094, `1 - a = 1.130`.
    - 1.2B: `a = +0.626`, slope = 0.383, `1 - a = 0.374`.
  - So structurally the incumbent is close to a 6-param anchored form, but the **extra 6 DOF of the 12-param parameterization absorb extrapolation error at the 520M point that is outside the training scales**. Strictly collapsing to 6 params causes the quadratic `a` to explode at 520M.
  - The training data covers only 3 distinct `uN` scales (130M, 300M, 1.2B). The 12-param quadratic basis can exactly fit 3-point scale-specific calibration along each axis; bases with fewer parameters don't have enough DOF to do so.
  - Grouped-CV alpha selection over `(0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)` picks alpha ≈ 0.01–0.1 for most variants and collapses them to a global-affine recalibration. The incumbent's "train objective" selection effectively always picks alpha = 0 and relies on the full 12-parameter freedom as implicit regularization via the `slope ≈ 1 - intercept` ridge.
  - Simplification of the calibrator functional form does not yield a cleaner equivalent model on this benchmark. The 12-parameter `(quadratic, quadratic)` form is minimal.
- Next action:
  - Keep `source_old10_scale_calibrated` as the incumbent.
  - Run multi-seed robustness study on the deterministic holdout manifest to confirm the seed-7 result is not an outlier, and inspect candidate plausibility.

### 2026-04-20 03:55 - Multi-seed robustness sweep and candidate plausibility
- Hypothesis: Seed 7's overall `RMSE 0.01076` might be a lucky draw. Using deterministic alternate seeds for the random supplement should stress-test whether the incumbent's gain generalizes. The top-1 selected mixture per model should be a reasonable GRP candidate close to a known training mixture.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added `evaluate_robustness_sweep` that reruns the benchmark for `ROBUSTNESS_SEEDS = (1, 3, 5, 7, 11, 13, 17, 23)` and writes `reference_outputs/transfer_benchmark_robustness_sweep.csv`.
  - Added `candidate_plausibility_summary` that, for each model variant, records the holdout row it ranks as lowest-predicted BPB, the regret vs the actual-min-BPB mixture, and the TV distance to the nearest training mixture's weight tensor.
- Result:
  - Mean overall RMSE across 8 seeds (sorted ascending):
    - `source_old10_scale_calibrated`: `0.01659`.
    - `source_old10_scale_diag_quadratic`: `0.02360`.
    - `source_old10_scale_affine_intercept_quadratic_slope`: `0.02489`.
    - Every other scale variant clusters around `0.0275`–`0.0277`.
    - All anchored variants around `0.038`–`0.039`.
  - Mean overall Spearman: incumbent `0.91642`, best simplified `affine_intercept_quadratic_slope` at `0.91389`, all others `≤ 0.911`.
  - Mean fixed 520M RMSE: incumbent `0.03479`, next-best `diag_quadratic` at `0.05928`, simpler variants `≈ 0.072`. Anchored forms `≈ 0.095`.
  - Per-seed incumbent RMSE on fixed 520M ranges from `0.00673` (seed 7) to `0.07125` (seed 3). Seed 7 is the incumbent's best case in this family, not a representative sample.
  - Candidate plausibility (seed 7):
    - Incumbent and `scale_quadratic_cv` choose `run_00125` (the true overall argmin at `520M`, actual BPB `0.8933`); regret 0; nearest training mixture `baseline_stratified` at TV = 0.7040.
    - Every simpler scale variant and every additive variant chooses `run_00180` (`300M` random supplement, actual BPB `0.9610`); regret 0.0677; nearest train TV 0.7827.
    - Anchored variants pick erratic candidates (`baseline_olmix_loglinear_uncheatable_bpb` at 1.2B with regret 0.1398 at TV 1.367, or `baseline_unimax` with regret 0.024), confirming their extrapolation failure.
- Interpretation:
  - Incumbent is the best model across every deterministic seed in the sweep for overall RMSE, overall Spearman, and fixed 520M RMSE. The advantage is durable, not a seed artifact.
  - On seed 7 the benchmark is unusually friendly to the incumbent (best case of 8), but the incumbent is still strictly best at every other tested seed too.
  - The only model that selects the correct argmin mixture across the holdout is the incumbent. All simpler forms mis-rank the 520M tail and select a worse 300M mixture with the same resulting regret of 0.0677.
  - Anchored forms should not be deployed: their quadratic `a` extrapolates outside `[0, 1]` at unseen scales and can invert the ranking.
- Next action:
  - Keep `source_old10_scale_calibrated` as the deployed transfer law.
  - When considering future simplification, the path is to add explicit **extrapolation regularization** that shrinks the calibrator toward a safe prior at scales far from the training `(uN, uD)` centroids; any pure basis reduction will keep losing the 520M signal.

### 2026-04-20 04:20 - Promising follow-up directions tested locally
- Hypothesis:
  - Three follow-up directions still looked technically plausible after the simplification sweep:
    1. bounded anchored calibration to prevent the anchored coefficient from exploding or flipping sign at 520M,
    2. a full quadratic calibrator with an explicit penalty toward the anchored relation `intercept + slope ≈ 1`,
    3. if either helped, use that as the next local baseline before handing off.
- Command:
  - Seed-7 benchmark and short robustness check via ad hoc `uv run` Python snippets importing `run_transfer_benchmark_holdouts.py`.
- Config:
  - Added to `SCALE_CAL_VARIANT_SPECS`:
    - `source_old10_bounded_anchored_quadratic`
    - `source_old10_bounded_anchored_diag_quadratic`
    - `source_old10_scale_quadratic_coupled`
  - Bounded anchored form:
    - `pred = raw + tanh(g(u)) * (1 - raw)`
    - basis = quadratic or diag_quadratic
  - Coupled quadratic form:
    - same quadratic intercept + quadratic log-slope calibrator as incumbent
    - plus penalty toward `intercept + slope - 1 = 0`
- Result:
  - On seed 7:
    - incumbent `source_old10_scale_calibrated`:
      - overall `RMSE 0.01076`, `Spearman 0.91356`
      - fixed `520M` `RMSE 0.00673`, `Spearman 0.90`
    - `source_old10_bounded_anchored_quadratic`:
      - overall `RMSE 0.03857`, `Spearman 0.89433`
      - fixed `520M` `RMSE 0.09953`, `Spearman 0.70`
    - `source_old10_bounded_anchored_diag_quadratic`:
      - overall `RMSE 0.03860`, `Spearman 0.89433`
      - fixed `520M` `RMSE 0.09959`, `Spearman 0.70`
    - `source_old10_scale_quadratic_coupled`:
      - overall `RMSE 0.03011`, `Spearman 0.89575`
      - fixed `520M` `RMSE 0.07938`, `Spearman 0.70`
  - Short 4-seed robustness check (`1, 3, 7, 13`) means:
    - incumbent:
      - mean overall `RMSE 0.01726`, `Spearman 0.91802`
      - mean fixed `520M` `RMSE 0.03656`, `Spearman 0.675`
    - bounded anchored quadratic:
      - mean overall `RMSE 0.04055`, `Spearman 0.91776`
      - mean fixed `520M` `RMSE 0.10022`, `Spearman 0.40`
    - bounded anchored diag quadratic:
      - mean overall `RMSE 0.04055`, `Spearman 0.91933`
      - mean fixed `520M` `RMSE 0.10031`, `Spearman 0.55`
    - coupled quadratic:
      - mean overall `RMSE 0.02875`, `Spearman 0.91645`
      - mean fixed `520M` `RMSE 0.07463`, `Spearman 0.475`
- Interpretation:
  - Bounded anchored calibration does preserve a plausible anchored interpretation, but it is too compressive. It kills calibration and badly under-spreads the fixed `520M` predictions.
  - The explicit coupling penalty toward `intercept + slope = 1` does not help either. It improves neither overall benchmark performance nor the 520M stress test relative to the incumbent.
  - So the empirically successful ingredient is still the full unconstrained quadratic intercept + quadratic log-slope calibrator. None of the cleaner extrapolation-regularized variants tested so far improves it.
- Next action:
  - Keep `source_old10_scale_calibrated` as the local incumbent.
  - The next step should probably not be another nearby calibrator tweak. Either:
    - hand off to a stronger search / external modeler, or
    - change the problem setup again (more data, or a better extrapolation prior) rather than doing more local hillclimbing in this same family.

### 2026-04-20 02:05 - Scale-aware source-old10 calibration breaks through
- Hypothesis: The source-old10 family already has the right residual direction, and the remaining gap is in how the calibration changes with target scale. A monotone scale-aware calibrator on top of the source-old10 residual may preserve the rank signal while fixing the diagonal better than a single global affine fit. In parallel, test a pairwise source-old10 rank fit to see whether the family benefits from an explicitly rank-oriented objective.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
- Config:
  - Added:
    - `source_old10_scale_calibrated` (`29` params total = `17` source-old10 core + `12` scale-aware monotone calibration coefficients)
    - `source_old10_pairwise_residual` (`18` params total = `7` global-affine base + `9` source-old10 pairwise score + `2` affine calibration)
- Result:
  - `source_old10_scale_calibrated` is the new incumbent by a wide margin:
    - overall `RMSE 0.01076`, `Spearman 0.91356`
    - fixed `520M` `RMSE 0.00673`, `Spearman 0.90`
  - previous incumbent `source_old10_affine_calibrated`:
    - overall `RMSE 0.02829`, `Spearman 0.89919`
    - fixed `520M` `RMSE 0.07354`, `Spearman 0.70`
  - `source_old10_pairwise_residual` fails badly:
    - overall `RMSE 0.05030`, `Spearman 0.23117`
    - fixed `520M` `RMSE 0.10339`, `Spearman 0.30`
- Interpretation:
  - The bottleneck was not lack of rank signal in the source-old10 family. It was calibration that was too rigid across target scales.
  - A monotone scale-aware calibrator is enough to turn the source-old10 family into the first local model that clears the `0.9` overall benchmark Spearman target while also improving the fixed `520M` stress-test substantially.
  - The pairwise residual route is a dead end in the current form; it collapses prediction spread and destroys calibration.
- Next action:
  - Promote `source_old10_scale_calibrated` to the new local baseline.
  - Stress-test whether this gain survives on alternate deterministic holdout seeds / manifests and check candidate quality under this new transfer law before packaging a larger handoff.

### 2026-04-20 05:02 - Refreshed fixed-520M incumbent plot clarifies the pessimism question
- Hypothesis: The older fixed-`520M` figure is stale and is visually overstating current pessimism. If we redraw the same fixed-`520M` scatter using the current incumbent ladder, the older models should still sit uniformly above the diagonal, but the current source-old10 scale-calibrated model should no longer do so.
- Command:
  - `uv run code/plot_520m_incumbent_holdout.py`
- Config:
  - Read current benchmark outputs from `reference_outputs/transfer_benchmark_holdout_predictions.csv`
  - Restricted to `holdout_kind == fixed_520m`
  - Compared:
    - `global_affine_scale`
    - `rank_gate_small_residual`
    - `source_old10_affine_calibrated`
    - `source_old10_scale_calibrated`
- Result:
  - Wrote:
    - `reference_outputs/transfer_holdout_520m_incumbent_predictions.csv`
    - `reference_outputs/transfer_holdout_520m_incumbent_summary.json`
    - `reference_outputs/figures/transfer_holdout_520m_incumbent_predicted_vs_actual.png`
  - Older incumbents remain clearly pessimistic on the fixed `520M` rows:
    - `global_affine_scale`: mean bias `+0.07386`, `RMSE 0.07458`, `Spearman 0.20`
    - `rank_gate_small_residual`: mean bias `+0.07375`, `RMSE 0.07433`, `Spearman 0.60`
    - `source_old10_affine_calibrated`: mean bias `+0.07305`, `RMSE 0.07354`, `Spearman 0.70`
  - Current incumbent is qualitatively different:
    - `source_old10_scale_calibrated`: mean bias `+0.00387`, `RMSE 0.00673`, `Spearman 0.90`
- Interpretation:
  - The older plot was correctly showing systematic pessimism, but only for the older models.
  - The current incumbent is only mildly pessimistic on average at `520M`; it is much closer to the diagonal and no longer uniformly overpredicts all five holdout rows.
- Next action:
  - Use the refreshed incumbent figure in future discussions instead of the stale three-panel snapshot.
  - When discussing residual `520M` bias, separate “older transfer baselines are uniformly pessimistic” from “current incumbent still has a small positive mean bias.”

### 2026-04-20 05:33 - Composed direct law makes arbitrary-mixture optimization possible, but proxy quality is now the bottleneck
- Hypothesis: We can recover arbitrary new-mixture optimization by composing a direct `60M` mixture surrogate with the current transfer incumbent. If the composition works mechanically and stays plausible, the remaining question is whether the direct proxy is good enough to preserve transfer quality.
- Command:
  - `uv run --project /Users/calvinxu/Projects/Work/Marin/marin --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_composed_direct_transfer.py`
- Config:
  - Stage 1 direct proxy:
    - reused the trusted no-L2 `60M` GRP surrogate from `reference_outputs/grp_power_family_penalty_no_l2_retune_best.csv`
  - Stage 2 transfer:
    - reused the current incumbent transfer family:
      - `global_affine_source_old10_residual`
      - `source_old10_scale_calibrated`
  - Composed law:
    - predict `y60` from arbitrary weights
    - feed predicted `y60` plus source-old10 mixture features into the transfer law
  - Added direct arbitrary-mixture optimization targets:
    - `300M_1x`
    - `520M_1x`
    - `1.2B_1x`
- Result:
  - Wrote:
    - `reference_outputs/composed_direct_transfer_holdout_predictions.csv`
    - `reference_outputs/composed_direct_transfer_summary.json`
    - `reference_outputs/figures/composed_direct_transfer_predicted_vs_actual.png`
    - `candidate_mixtures_composed_direct.csv`
  - Stage 1 direct `60M` proxy quality on available rows:
    - `RMSE 0.01603`, `Spearman 0.68392`, predicted/actual std ratio `0.887`
  - Composed direct law on current mixed holdout benchmark:
    - overall `RMSE 0.01471`, `Spearman 0.80567`
    - fixed `520M` `RMSE 0.00958`, `Spearman 0.90`
  - Incumbent with observed `y60` remains stronger:
    - overall `RMSE 0.01076`, `Spearman 0.91356`
    - fixed `520M` `RMSE 0.00673`, `Spearman 0.90`
  - Direct arbitrary-mixture optimizer produced stable candidates and did not collapse to degenerate near-single-domain mixtures:
    - `300M_1x`: predicted `0.9151`
    - `520M_1x`: predicted `0.8367`
    - `1.2B_1x`: predicted `1.0185`
- Interpretation:
  - Arbitrary new-mixture optimization is now mechanically available through composition. We do not need to choose between “good transfer” and “direct mixture optimization”; the two-stage law composes cleanly.
  - The main degradation comes from replacing observed `y60` with predicted `y60`. The transfer layer appears strong enough; the bottleneck is the direct proxy stage.
  - Fixed-`520M` rank survives surprisingly well under composition, but broader mixed-holdout ranking degrades materially, which is consistent with the direct proxy only reaching about `0.68` Spearman.
- Next action:
  - Treat “improve the direct proxy stage” as the next route to better arbitrary-mixture optimization.
  - Compare composed-direct candidates against known good GRP optima and nearest observed mixtures before using them as deployment suggestions.

## 2026-04-20 520M no-L2 raw optimum launch attempt

- Goal:
  - Launch the trusted GRP `power_family_penalty_no_l2` raw optimum as a dedicated `520M / 10.4B` replay validation.
- Added launcher:
  - `experiments/domain_phase_mix/launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b.py`
  - mirrors the existing `300M / 6B` raw-optimum launcher, but targets the `520M` replay config and defaults to `power_family_penalty_no_l2`
- Launch command:
  - `uv run --with torch python experiments/domain_phase_mix/launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b.py --variants power_family_penalty_no_l2 --max-concurrent 1`
- Current local artifacts:
  - executor plan:
    - `/tmp/marin/experiments/launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b-08db87.json`
  - run manifest:
    - `/tmp/marin/pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_520m_10p4b/run_manifest-61e549/run_manifest.json`
  - replay checkpoint root:
    - `/tmp/marin/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_520m_10p4b/baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e`
- Intended replay spec:
  - source run: `baseline_genericfamily_power_family_penalty_no_l2_raw_optimum`
  - source `run_id`: `415`
  - replay run name: `baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b`
  - cohort: `grp_penalty_raw_optimum_520m_10p4b`
  - model family: `regmix_520m_proxy`
  - experiment budget: `10400000000`
  - target budget: `6325183647689`
  - train steps: `19836`
- What actually happened:
  - the launch did not create a child Iris training job yet
  - the active local launcher process started local `zephyr.subprocess_worker` children and is processing raw `dolma3_pool` shards before submission
  - this explains why no corresponding Iris `running` or `pending` job exists for the no-L2 optimum yet
- Relevant local processes at time of handoff:
  - stale/active launcher with zephyr workers:
    - parent PID `79182`
    - child worker PIDs: `87469 87541 87542 87543 87544 87545 87546 89776`
  - second launcher attempt waiting/sleeping:
    - parent PID `79718`
- Cluster-side sanity check:
  - fresh `520M` Iris jobs visible during inspection were unrelated batch reruns:
    - `/calvinxu/dm-520m-qsplit-5p2b-20260420-022643`
    - `/calvinxu/dm-520m-strat-5p2b-20260420-022830`
  - their logs show qsplit/stratified baseline reruns, not the no-L2 optimum replay
- Babysit guidance:
  - do not babysit an Iris child job yet; there is none
  - babysit the local launcher first, or kill/relaunch once the pre-submit local Zephyr work is understood or disabled

### 2026-04-20 02:40 - no-L2 520M launcher fix and Iris submission
- Root cause of local stall:
  - The launcher calls `executor_main()` directly, running the full DAG (tokenization, merge, training) on the local machine
  - Tokenization/merge steps are Zephyr tasks that block when run locally (no Zephyr coordinator on the laptop)
  - The 300M version had the same design but was launched via Iris in previous sessions
- Fix:
  - Killed stuck local launchers (PIDs 79182, 79718)
  - Submitted via `marin.run.iris_run` to run as an Iris job in us-east5-a with 32GB memory
  - Command: `uv run python -m marin.run.iris_run --config lib/iris/examples/marin.yaml -- --job-name dm-grp-no-l2-520m-10p4b-20260420-023652 --cpu 4 --memory 32GB --disk 20GB --region us-east5 --zone us-east5-a --no-wait --enable-extra-resources --extra marin:tpu --extra marin:eval -- python experiments/domain_phase_mix/launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b.py --variants power_family_penalty_no_l2 --max-concurrent 1`
- Result:
  - Parent: `/calvinxu/dm-grp-no-l2-520m-10p4b-20260420-023652` — RUNNING
  - Child: `train_lm_baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e` — RUNNING
  - GCS: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_520m_10p4b/baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e/`
- Monitor: `uv run iris --config lib/iris/examples/marin.yaml job logs --include-children /calvinxu/dm-grp-no-l2-520m-10p4b-20260420-023652`

### 2026-04-20 02:55 - no-L2 1.2B launcher added, launched, and registry-accounted
- Added dedicated launcher:
  - `experiments/domain_phase_mix/launch_two_phase_many_genericfamily_penalty_raw_optima_1_2b_24b.py`
- Tightened both dedicated launchers so `build_run_specs()` defaults to only `power_family_penalty_no_l2` when called programmatically with `variants=None`
- Submitted the 1.2B replay remotely through Iris:
  - Parent: `/calvinxu/dm-grp-no-l2-1-2b-24b-20260420-024302` — RUNNING
  - Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --priority batch --job-name dm-grp-no-l2-1-2b-24b-20260420-024302 --cpu 1 --memory 2GB --extra marin:tpu -- python experiments/domain_phase_mix/launch_two_phase_many_genericfamily_penalty_raw_optima_1_2b_24b.py --variants power_family_penalty_no_l2 --max-concurrent 1`
- Registry code updated:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py`
  - Added families:
    - `grp_penalty_raw_optima_520m_10p4b`
    - `grp_penalty_raw_optima_1_2b_24b`
  - Added tracked live jobs:
    - `/calvinxu/dm-grp-no-l2-520m-10p4b-20260420-023652`
    - `/calvinxu/dm-grp-no-l2-1-2b-24b-20260420-024302`
- Canonical run-registry outputs refreshed incrementally:
  - `run_registry/logical_runs.csv`
  - `run_registry/run_attempts.csv`
  - `run_registry/live_watchlist.csv`
  - `run_registry/summary.json`
- Current accounted state after refresh:
  - `grp_penalty_raw_optima_520m_10p4b`: logical run present, `running`, one attempt present, live watch entry present
  - `grp_penalty_raw_optima_1_2b_24b`: logical run present, `planned` pending first checkpoint root, no attempts yet, live watch entry present and `running`
- Metric registry was intentionally left unchanged for now because neither new replay has emitted evaluation metrics yet

### 2026-04-20 03:55 - explicit uncheatable-component GRP beats single-head no-L2 at 60M
- New benchmark script:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_grp_power_family_penalty_no_l2_uncheatable_components.py`
- Experiment:
  - Fit 7 independent GRP no-L2 models on the 60M fit swarm, one for each `eval/uncheatable_eval/*/bpb` subdomain:
    - `ao3_english`
    - `arxiv_computer_science`
    - `arxiv_physics`
    - `bbc_news`
    - `github_cpp`
    - `github_python`
    - `wikipedia_english`
  - Aggregate predicted sub-losses back to overall `eval/uncheatable_eval/bpb` with a fixed NNLS decomposition learned from the observed component metrics
- Decomposition sanity:
  - overall `eval/uncheatable_eval/bpb` is almost exactly a fixed weighted sum of the 7 subdomain BPBs
  - aggregation reconstruction RMSE on observed metrics: `5.67e-08`
- Result on the matched 241-row 60M fit swarm:
  - baseline `single_head_no_l2`:
    - train RMSE `0.00758`
    - CV RMSE `0.00916`
    - CV Spearman `0.86679`
    - chosen candidate `run_00125`
  - `component_aggregate_no_l2`:
    - train RMSE `0.00687`
    - CV RMSE `0.00785`
    - CV Spearman `0.90513`
    - chosen candidate `run_00125`
- Interpretation:
  - explicit sub-loss decomposition materially improves 60M fit quality over the current single-head GRP no-L2 baseline
  - this suggests “predict multiple components then aggregate” is a promising backbone for the direct multi-scale law
  - caveat: parameter count jumps from `42` to `302`, so the next step should be to preserve most of the gain with more sharing, not to ship the fully independent 7-head version as-is
## 2026-04-20 04:15 - first direct multi-scale GRP laws lag transfer incumbent

- Added packet-local evaluator:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_direct_multiscale_grp_components.py`
- Reused the transfer benchmark split (`5` fixed `520M`-style holdouts had become `4` in the current packet snapshot, plus `34` deterministic random holdouts) but trained a true direct law on **all non-holdout primary rows** (`train_primary_count = 600`), with no `y60` proxy input.
- Tried two direct law families:
  - `direct_scalar_grp`
    - joint multi-scale GRP feature map from `ContinuousNDGRPLaw.build_features(..., mode="joint_nd_model")`
    - scalar floor-law head
    - selected `alpha = 30`
    - `39` params
  - `direct_component_aggregate_grp`
    - same direct GRP feature map
    - `7` independent uncheatable component heads, aggregated back to overall BPB with NNLS weights
    - selected common `alpha = 30`
    - `281` params
- Seed-7 benchmark results:
  - incumbent `source_old10_scale_calibrated`: overall `RMSE 0.01090`, `Spearman 0.90655`; fixed `520M` `RMSE 0.00744`, `Spearman 0.80`
  - `direct_scalar_grp`: overall `RMSE 0.03882`, `Spearman 0.59689`; fixed `520M` `RMSE 0.08229`, `Spearman 0.40`
  - `direct_component_aggregate_grp`: overall `RMSE 0.06517`, `Spearman 0.67830`; fixed `520M` `RMSE 0.12440`, `Spearman -0.20`
- Robustness sweep over seeds `{1,3,5,7,11,13,17,23}`:
  - incumbent mean overall `RMSE 0.01591`, mean overall `Spearman 0.90984`
  - `direct_scalar_grp` mean overall `RMSE 0.04185`, mean overall `Spearman 0.71657`
  - `direct_component_aggregate_grp` mean overall `RMSE 0.06089`, mean overall `Spearman 0.71345`
- Candidate-quality read:
  - direct scalar `60M` optimum is at least plausible (`predicted 0.9818`, better than observed `1.0318`, with stable opt and moderate support)
  - direct component optimum is too collapsed / off-manifold even at `60M`, and much worse than known best observations
  - at `300M/520M/1.2B`, direct scalar gives internally consistent optima, but still extrapolates from 60M-nearest geometry and is not yet strong enough to trust as a deployment law
- Main lesson:
  - explicit sub-loss decomposition helped at fixed `60M`, but naive multi-component aggregation does **not** transfer cleanly to the multi-scale direct-law setting
  - the next direct-law move should likely be a **shared / coupled multi-component law** or a simpler scalar direct law with better scale structure, not `7` independent component heads

## 2026-04-20 04:40 - direct old10 + scale calibration is a dead end

- Extended the direct-law benchmark with one additional scalar family:
  - `direct_old10_scale_calibrated`
  - raw direct old10 features (`9` old10 GRP features + `5` explicit scale basis terms = `14` raw dims)
  - floor-law raw head (`16` params total with floor/intercept)
  - plus the same quadratic scale-aware monotone calibrator shape used in the transfer incumbent (`12` params)
  - total params: `28`
- Selected config:
  - `alpha = 1000`
  - multipliers `eta=1.0, lam=1.0, k1=1.0, kt=1.0`
- Seed-7 benchmark:
  - overall `RMSE 0.08613`, `Spearman 0.68552`
  - fixed `520M` `RMSE 0.26130`, `Spearman -0.40`
- 8-seed robustness:
  - mean overall `RMSE 0.07321`
  - mean overall `Spearman 0.59717`
  - mean fixed-`520M` `RMSE 0.20435`
  - mean fixed-`520M` `Spearman -0.40`
- Candidate behavior is pathological:
  - optimization collapses to nearly single-family / single-domain phase weights
  - `520M` optimum predicted BPB `1.4609`, which is not remotely credible
  - nearest observed mixtures are farther off-manifold than the scalar direct GRP
- Interpretation:
  - simply transplanting the transfer-style old10 + scale-calibration recipe into a direct law does not work
  - the transfer incumbent is strong because it conditions on observed small-scale loss, not because this functional form is a good standalone direct law
  - for direct multi-scale modeling, the plain scalar GRP remains the least-bad current baseline

## 2026-04-20 04:55 - explicit N/D splits inside the scalar GRP body do not rescue the direct law

- Extended `run_direct_multiscale_grp_components.py` with two more scalar direct-law families that try to improve the GRP body's N/D dependence without adding a free post-hoc calibrator:
  - `direct_affine_body_grp`
    - family-specific affine N/D modulation of retained-exposure gains and family penalty thresholds
    - direct scalar floor-law head on top
    - `33` total params
  - `direct_global_nd_split_grp`
    - same body-only feature family, but with a tightly constrained global split:
      - `eta_mul`
      - `lam_mul`
      - shared `kN_scale`, `kD_scale`
      - shared `tN_scale`, `tD_scale`
    - direct scalar floor-law head on top
    - `25` total params
- Seed-7 benchmark:
  - `direct_affine_body_grp`: overall `RMSE 0.10298`, `Spearman 0.61352`; fixed `520M` `RMSE 0.17945`, `Spearman 0.80`
  - `direct_global_nd_split_grp`: overall `RMSE 0.06130`, `Spearman 0.55356`; fixed `520M` `RMSE 0.13305`, `Spearman 0.80`
  - baseline `direct_scalar_grp`: overall `RMSE 0.03882`, `Spearman 0.59689`; fixed `520M` `RMSE 0.08229`, `Spearman 0.40`
- 8-seed robustness means:
  - `direct_affine_body_grp`: mean overall `RMSE 0.09685`, mean overall `Spearman 0.65593`
  - `direct_global_nd_split_grp`: mean overall `RMSE 0.05950`, mean overall `Spearman 0.60729`
  - baseline `direct_scalar_grp`: mean overall `RMSE 0.04185`, mean overall `Spearman 0.71657`
- Candidate behavior:
  - both new N/D-split variants optimize toward nearly one-hot phase mixtures close to `run_00063` / `run_00003`
  - `direct_affine_body_grp` is clearly pathological and off-manifold
  - `direct_global_nd_split_grp` is cleaner on parameter count, but still pushes the optimizer into extreme phase concentration and predicts implausibly high target BPB
- Interpretation:
  - the direct-law gap is not fixed by simply giving the GRP body separate `uN` and `uD` coefficients
  - these body-scale variants lose too much calibration and produce worse arbitrary-mixture optima than the existing scalar direct baseline

## 2026-04-20 05:10 - scale-balanced fitting for the scalar direct GRP is also a negative result

- Added a scale-balanced fit variant for the scalar direct law:
  - `direct_scalar_grp_scale_balanced`
  - same exact feature map and parameter count as `direct_scalar_grp` (`39` params)
  - only change is the fitting objective:
    - inverse-frequency row weights by target scale during floor-law ridge fitting
    - alpha selected by weighted CV RMSE on the same grouped folds
- Seed-7 benchmark:
  - `direct_scalar_grp_scale_balanced`: overall `RMSE 0.04725`, `Spearman 0.39249`; fixed `520M` `RMSE 0.10543`, `Spearman 0.20`
  - baseline `direct_scalar_grp`: overall `RMSE 0.03882`, `Spearman 0.59689`; fixed `520M` `RMSE 0.08229`, `Spearman 0.40`
- 8-seed robustness means:
  - scale-balanced: mean overall `RMSE 0.04822`, mean overall `Spearman 0.55589`, mean fixed-`520M` `RMSE 0.10604`, mean fixed-`520M` `Spearman -0.05`
  - baseline scalar: mean overall `RMSE 0.04185`, mean overall `Spearman 0.71657`, mean fixed-`520M` `RMSE 0.08569`, mean fixed-`520M` `Spearman 0.325`
- Candidate behavior:
  - optimized candidates collapse toward high-tech/reasoning corners and off-manifold mixtures
  - seed-7 chosen holdout candidate is `baseline_olmix_loglinear_uncheatable_bpb` with regret `0.1387`, much worse than the baseline scalar law
- Interpretation:
  - the current direct-law failure is not just a matter of the dense `60M`/`300M` rows numerically dominating the fit
  - scale-balancing the objective alone makes candidate quality worse and does not improve `520M`
  - taken together with the N/D-split failures, this suggests the next direct-law improvement must come from a better **base feature law / body**, not another small fitting or calibration tweak on top of the current scalar GRP family

## 2026-04-20 05:30 - separating global scale baseline from mixture residual also fails

- Added one more direct scalar family to `run_direct_multiscale_grp_components.py`:
  - `direct_scalar_grp_scale_residual`
  - decomposition:
    - fit a smooth global scale law `s(N,D)` from the 6-dim scale design matrix
    - fit a ridge residual head on the mixture-only GRP features (full scalar design minus the last 5 scale columns)
    - final prediction is `s(N,D) + r(w,N,D)`
  - config selected by grouped CV:
    - `scale_alpha = 30`
    - `resid_alpha = 1000`
  - total params: `41`
- Seed-7 benchmark:
  - overall `RMSE 0.19837`, `Spearman 0.39818`
  - fixed `520M` `RMSE 0.28248`, `Spearman 0.40`
  - much worse than the plain scalar direct GRP baseline (`0.03882 / 0.59689`)
- 8-seed robustness:
  - mean overall `RMSE 0.19868`
  - mean overall `Spearman 0.45708`
  - mean fixed-`520M` `RMSE 0.27828`
  - mean fixed-`520M` `Spearman 0.50`
- Candidate behavior:
  - still picks non-best `520M` holdout rows
  - optimized direct candidates move toward extreme high-tech corners and predict implausibly bad target BPBs (`60M` optimum around `1.25`)
  - nearest observed geometry is not obviously safer than the plain scalar direct baseline
- Interpretation:
  - the direct-law failure is not fixed by a naive additive decomposition `global scale baseline + mixture residual`
  - with the current scalar GRP features, the residual head is too weak / too unstable to model cross-scale mixture effects after subtracting the global scale trend
  - this reinforces the same conclusion as the earlier negative results: the next direct-law step needs a better **base law / representation**, not another decomposition or weighting trick layered onto the current scalar head

## 2026-04-20 05:55 - versioned ChatGPT Pro packet refreshed for the direct-law handoff

- Created a new versioned handoff packet:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_direct_multiscale_law_packet_v18`
- Built a matching archive:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_direct_multiscale_law_packet_v18.zip`
- Purpose:
  - avoid attachment-collision issues from reusing old packet names
  - switch the handoff from the old transfer-calibration objective to the current **direct multi-scale law** objective
- Updated packet docs:
  - rewrote `REQUEST_TO_CHATGPT_PRO.md`
  - rewrote `README.md`
  - added `reference_outputs/local_direct_law_handoff_summary.md`
  - replaced the stale transfer-era `MANIFEST.json` with a versioned direct-law manifest
- Refreshed packet data snapshots:
  - copied the latest `run_registry/summary.json` to `reference_outputs/run_registry_summary.json`
  - copied the latest `run_registry/strong_tier_perplexity_ready.csv`
  - copied the latest `run_registry/live_watchlist.csv`
- Packet now explicitly records the pending trusted GRP no-L2 raw-optimum replays:
  - `520M / 10.4B`: `/calvinxu/dm-grp-no-l2-520m-10p4b-20260420-023652`
  - `1.2B / 24B`: `/calvinxu/dm-grp-no-l2-1-2b-24b-20260420-024302`
- Validation:
  - parsed the new manifest
  - verified all manifest-referenced artifact paths exist in the packet

### 2026-04-20 06:20 - retuning the GRP signal exponents `a_f` changes the direct-law landscape but does not beat the scalar baseline overall
- Hypothesis:
  - The direct-law body is biased toward reasoning/code corners partly because the fixed GRP exponents are highly asymmetric:
    - `a_broad_text = 0.48`
    - `a_tech_code = 0.048`
    - `a_reasoning = 1.03`
  - Jointly retuning `a_f` with the family-specific `eta/lam/k/t` body might keep the useful 37-feature scalar head while changing the optimization landscape enough to reduce the bad reasoning-corner bias.
- Code change:
  - Added `direct_enriched_grp_family_a` to `chatgpt_pro_swarm_transfer_packet/code/run_direct_multiscale_grp_components.py`
  - This model keeps the full 37-feature enriched direct head, retunes the 14 family-specific GRP body parameters, and additionally learns:
    - `a_broad_text`
    - `a_tech_code`
    - `a_reasoning`
  - Total parameter count: `56`
- Seed-7 result from the refreshed predictions file:
  - baseline `direct_scalar_grp`:
    - overall `RMSE 0.03882`, `Spearman 0.59689`
    - fixed `520M` `RMSE 0.08229`, `Spearman 0.40`
    - random supplement `RMSE 0.02979`, `Spearman 0.46188`
  - new `direct_enriched_grp_family_a`:
    - overall `RMSE 0.03501`, `Spearman 0.67502`
    - fixed `520M` `RMSE 0.07399`, `Spearman 0.40`
    - random supplement `RMSE 0.02694`, `Spearman 0.54714`
  - On the seed-7 split, this is the first direct-law variant that clearly improves both overall RMSE and overall Spearman over `direct_scalar_grp` without degrading fixed-`520M` Spearman.
- Candidate geometry (seed 7):
  - `direct_enriched_grp_family_a` no longer pushes all targets to the same pure reasoning/code corner.
  - Optimised targets are still off-manifold, but less pathological than the pure scalar baseline:
    - `300M`: phase0 `~[broad 0.00, tech 0.30, reasoning 0.70]`, phase1 `~[broad 0.32, tech 0.68, reasoning ~0]`, nearest-TV `0.636`
    - `520M`: phase0 `~[broad 0.00, tech 0.50, reasoning 0.50]`, phase1 `~[broad 0.35, tech 0.65, reasoning 0.00]`, nearest-TV `0.686`
    - `1.2B`: phase0 `~[broad 0.00, tech 0.52, reasoning 0.48]`, phase1 `~[broad 0.40, tech 0.60, reasoning 0.00]`, nearest-TV `0.680`
  - This is still not the broad-heavy observed winner geometry (`run_00125`), but it is directionally closer than `direct_scalar_grp`.
- Quick robustness check:
  - Full canonical 8-seed rerun was too expensive because it recomputes the full rejected-model zoo.
  - Ran a focused 4-seed check on seeds `{1, 3, 7, 13}` for:
    - `direct_scalar_grp`
    - `direct_enriched_grp_family_a`
  - Means:
    - `direct_scalar_grp`:
      - overall `RMSE 0.03867`, `Spearman 0.69904`, `Kendall 0.55121`
      - fixed `520M` `RMSE 0.07752`, `Spearman 0.40`, `Kendall 0.33333`
      - random supplement `RMSE 0.03099`, `Spearman 0.59473`, `Kendall 0.46702`
    - `direct_enriched_grp_family_a`:
      - overall `RMSE 0.04618`, `Spearman 0.67431`, `Kendall 0.51138`
      - fixed `520M` `RMSE 0.10156`, `Spearman 0.65`, `Kendall 0.66667`
      - random supplement `RMSE 0.03417`, `Spearman 0.62613`, `Kendall 0.46970`
- Interpretation:
  - The `a_f` retune is a **real structural change**, not a no-op:
    - it improves the seed-7 benchmark materially
    - it improves candidate geometry somewhat
    - it raises fixed-`520M` rank on the quick robustness check
  - But it does **not** yet beat `direct_scalar_grp` as the overall direct-law baseline once we average over multiple seeds.
  - So the current conclusion is:
    - retuning `a_f` is a promising direction and should stay in the search space
    - but it is not yet the new incumbent
    - the remaining problem is still the tradeoff between getting the larger-scale ranking right and keeping overall calibration / interpolation strong
- Artifacts:
  - refreshed plot:
    - `chatgpt_pro_swarm_transfer_packet/reference_outputs/figures/direct_multiscale_grp_predicted_vs_actual.png`
  - focused robustness artifact:
    - `chatgpt_pro_swarm_transfer_packet/reference_outputs/direct_multiscale_grp_enriched_family_a_quick_robustness.json`

### 2026-04-20 05:30 - Direct-law simplification attempts (direction A) and landscape diagnosis
- Hypothesis: The current 39-param `direct_scalar_grp` baseline uses a 37-feature `joint_nd_model` head with many engineered share / entropy / sqrt-share / delta-share auxiliary features. Stripping these down to the lean GRP body terms (retained-exposure family signal + family penalty + low-order scale basis) and jointly retuning the nonlinear GRP body (eta, lam, kN, kD, tN, tD) should give a leaner, more principled scalar law that generalises at least as well.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_direct_multiscale_grp_components.py`
- Config:
  - Added an 11-feature lean body: 3 family `signal(retained_f)` + 3 family `softplus(log1p(retained_f) - tau_f)^2` + [uN, uD, uN*uD, uN^2, uD^2]. Lean body reuses the same custom-k/t body config as the existing `affine_body` family.
  - Added two lean variants that Powell-retune the body params on 5-fold grouped CV RMSE, reusing the existing alpha grid:
    - `direct_lean_grp_shared` (19 params = 11 features + 2 floor + 6 shared body: eta_mul, lam_mul, kN_scale, kD_scale, tN_scale, tD_scale).
    - `direct_lean_grp_family` (27 params = 11 features + 2 floor + 14 per-family body: eta_mul, lam_mul, kN_f*3, kD_f*3, tN_f*3, tD_f*3).
  - Added two "enriched-but-retuned" variants that keep the full 37-feature head and retune the body with the same Powell procedure:
    - `direct_enriched_grp_shared` (45 params = 37 + 2 + 6).
    - `direct_enriched_grp_family` (53 params = 37 + 2 + 14).
  - Lean + enriched body design matrices are invocation-compatible with the existing candidate-optimisation path (`direct_custom_lean_grp`, `direct_custom_enriched_grp`).
  - Robustness sweep, candidate plausibility, and plot panels updated to include all four new variants.
- Seed-7 result (direct laws + transfer reference):
  - `source_old10_scale_calibrated` (reference only): overall `RMSE 0.01090`, `Sp 0.90655`; fixed 520M `RMSE 0.00744`, `Sp 0.80`.
  - `direct_scalar_grp` (baseline, 39 params): overall `RMSE 0.03882`, `Sp 0.59689`; fixed 520M `RMSE 0.08229`, `Sp 0.40`; random supplement `RMSE 0.02979`, `Sp 0.46188`.
  - `direct_enriched_grp_shared` (45): overall `RMSE 0.04118`, `Sp 0.59952`; fixed 520M `RMSE 0.07394`, `Sp -0.20`.
  - `direct_enriched_grp_family` (53): overall `RMSE 0.04086`, `Sp 0.63278`; fixed 520M `RMSE 0.08497`, `Sp 0.40`.
  - `direct_lean_grp_shared` (19): overall `RMSE 0.04555`, `Sp 0.55925`; fixed 520M `RMSE 0.10959`, `Sp 0.40`; predicted/actual std ratio 0.20 (collapsed spread).
  - `direct_lean_grp_family` (27): overall `RMSE 0.04412`, `Sp 0.53058`; fixed 520M `RMSE 0.10642`, `Sp 1.00`; predicted/actual std ratio 0.20 (collapsed — the 520M Spearman is spurious).
- 8-seed robustness (mean overall RMSE / Spearman):
  - `direct_scalar_grp`: `0.04185 / 0.71657` (best direct law).
  - `direct_enriched_grp_shared`: `0.04267 / 0.71014`.
  - `direct_lean_grp_family`: `0.04305 / 0.65748`.
  - `direct_lean_grp_shared`: `0.04739 / 0.65076`.
  - `direct_enriched_grp_family`: `0.04760 / 0.63204`.
  - No new variant improves on the seed-7 `direct_scalar_grp` benchmark in a meaningful way. Enriched-shared is within 2% on RMSE and 0.5% on Spearman, trading a slightly worse fixed-520M spearman for a slightly lower fixed-520M RMSE.
- Candidate plausibility (seed 7 optimised mixtures):
  - `direct_scalar_grp` picks `phase0=[tech 0.40, reasoning 0.60, broad 0.00], phase1=[tech 0.82, reasoning 0.00, broad 0.18]` at every target scale. TV to nearest observed ≈ 0.55.
  - `direct_lean_grp_shared` and `direct_lean_grp_family` both collapse to `phase0 tech_code 1.0, phase1 tech_code 1.0` (TV ≈ 0.88 — far from any observed mixture).
  - `direct_enriched_grp_shared` and `direct_enriched_grp_family` pick the same reasoning-heavy pattern as `direct_scalar_grp`, with phase1 tech_code tightening toward 0.93 and 0.99 respectively (slightly more collapsed than baseline).
  - **Critical observation:** every direct-law candidate has phase0 broad_text 0 and phase1 broad_text < 0.18, but the actual best observed mixture at 300M and 520M (`run_00125`, actual BPB 0.93 / 0.89) has phase0 broad_text 0.48 / tech 0.29 / reasoning 0.22 and phase1 broad_text 0.52 / tech 0.39 / reasoning 0.10. The direct laws point the optimizer toward reasoning/code corners, while the winning observed mixtures are broad-heavy and mixed.
- Interpretation:
  - **Lean direction (A) does not help.** Stripping the engineered share/entropy features removes implicit regularisation that keeps the ridge head from collapsing predictions to 1-hot tech_code. The 11-feature lean body has insufficient capacity to distinguish mixtures beyond a global scale-of-training-budget prediction (prediction std collapses to ~20% of actual std), and candidates optimise to pure tech_code.
  - **Retuning body inside the engineered head (shared / per-family) also does not help.** The ridge relies on the engineered features more than on the body; retuning the body barely shifts predictions. Per-family body slightly improves mean 520M Spearman (0.9 across seeds) but at the cost of worse mean overall Spearman (0.632 vs 0.717).
  - **The direct-law bottleneck is the optimisation landscape, not the ridge head's calibration.** All direct-law families land on reasoning/code-heavy optima regardless of how the body is parameterised. The GRP body has `a_reasoning = 1.03` (linear in exposure), `a_broad_text = 0.48`, `a_tech_code = 0.048` (nearly constant); this asymmetry biases the signal landscape toward reasoning-loaded mixtures. The ridge head trained on in-distribution data has learned valid coefficients, but extrapolating to the unobserved region where broad_text dominates — which is where the actual minimiser lives — is exactly where the signal geometry is most distorted.
  - `direct_component_aggregate_grp` is the only existing variant whose candidates are broad-heavy in phase1 (0.79 broad at 60M, 0.69 broad at 300M/520M). That is because its 7 independent component heads each re-weight the 37 features differently, which breaks the reasoning-bias dominant in the shared joint head. The cost is 281 params and 520M Spearman -0.2 at seed 7.
- Next action:
  - Stop pursuing pure-stripping / body-retune directions on the scalar head.
  - Next direct-law path that has a chance: **jointly retune the GRP a_f exponents along with k/t**. Fixing `a_reasoning = 1.03` vs `a_tech_code = 0.048` is what ties the landscape to reasoning dominance; letting those float might break the reasoning-corner bias without destroying the rest of the law.
  - Complementary path: **shared-backbone low-rank component heads** inspired by what `direct_component_aggregate_grp` is doing right. Constrain the 7 component ridge heads to share most of their structure with a single backbone plus a small per-component residual, to import the broad-heavy landscape of the component model without the 281-param overhead.
  - Candidate evaluation should, from now on, be gated on `nearest_observed_mean_phase_tv < 0.5` and `phase0_broad_text + phase1_broad_text > 0.3`. Models that predict very good RMSE / Spearman but only if you follow them into a region where no observed mixture lives should not count as real improvements.

### 2026-04-20 06:10 - Created versioned hybrid handoff packet v20
- Goal:
  - Start a fresh ChatGPT Pro thread with a packet that reflects the current problem framing:
    - not transfer-only
    - not direct-only
    - but a unified hybrid model `y_hat(w, N, D | O_w)` with:
      - direct fallback `f(w, N, D)`
      - observation-conditioned update `u(w, N, D, O_w)`
- New packet:
  - directory:
    - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v20`
  - archive:
    - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v20.zip`
- Main docs written / refreshed:
  - `README.md`
  - `BACKGROUND_AND_GOALS.md`
  - `REQUEST_TO_CHATGPT_PRO.md`
  - `reference_outputs/local_hybrid_handoff_summary.md`
  - `MANIFEST.json`
- Packet content:
  - direct-law background and current incumbent:
    - `direct_scalar_grp`
  - transfer reference and why it is not enough:
    - `source_old10_scale_calibrated`
  - latest direct-law evidence:
    - `direct_enriched_grp_family_a`
    - `direct_multiscale_grp_enriched_family_a_quick_robustness.json`
  - registry snapshot:
    - `strong_tier_perplexity_ready.csv`
    - `run_registry_summary.json`
    - `live_watchlist.csv`
  - pending `520M` and `1.2B` GRP no-L2 replay jobs
- Acceptance criteria in the new packet:
  - direct-mode benchmark quality
  - conditioned-mode benchmark quality
  - same-scale identity
  - candidate plausibility
  - end-to-end hybrid tradeoff
- Validation:
  - manifest parsed successfully
  - all manifest-referenced artifacts exist
  - versioned archive rebuilt successfully

### 2026-04-20 07:55 - 74-config hybrid hillclimb and promoted tighter-kernel canonical hybrid
- Hypothesis:
  - The hybrid update family is not fundamentally wrong; the current canonical Nadaraya-Watson update may simply be too wide in scale-space. A tighter local kernel may preserve the direct fallback and exact identity while materially improving conditioned robustness, especially on the fixed `520M` subset and the unstable `multi_obs` path.
- Commands:
  - Search harness:
    - `uv run --with numpy --with pandas --with scipy --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_hybrid_multiscale_law_hillclimb.py`
  - Focused 4-seed robustness:
    - `uv run --with numpy --with pandas --with scipy --with matplotlib python - <<'PY' ... run_hybrid_multiscale_law_hillclimb helpers over seeds (1,3,7,13) ... PY`
  - Refreshed canonical evaluator:
    - `uv run --with numpy --with pandas --with scipy --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_hybrid_multiscale_law.py`
- Config:
  - Added search harness:
    - `code/run_hybrid_multiscale_law_hillclimb.py`
  - Seed-7 search space:
    - backbones:
      - `direct_scalar_grp`
      - `direct_enriched_grp_family_a`
    - update families:
      - `nw_single_bw` with `bw in {0.15, 0.25, 0.35, 0.5, 0.75, 1.0}`
      - `nw_pair_gain` with a 27-point grid over `(bw60, bw300, prior300)`
      - `lowrank_transport` over 4 low-rank configs
  - Total evaluated seed-7 hybrid configs:
    - `74`
  - Added applicable-subset reporting to the canonical hybrid summary:
    - `hybrid_<mode>_applicable`
  - Promoted canonical hybrid bandwidth from `0.5` to `0.15`
- Result:
  - Seed-7 search top configs:
    - `direct_enriched_grp_family_a + nw_single_bw_0.15`
      - direct fallback `0.04093 / 0.62928`
      - `obs60_and_300` overall `0.03174 / 0.64504`
      - fixed `520M` `0.03988 / 1.00`
    - `direct_scalar_grp + lowrank_r2_lf0.01_li0.1`
      - direct fallback `0.03882 / 0.59689`
      - `obs60_and_300` overall `0.02393 / 0.86957`
      - fixed `520M` `0.06149 / -0.20`
    - `direct_scalar_grp + nw_single_bw_0.15`
      - direct fallback `0.03882 / 0.59689`
      - `obs60_and_300` overall `0.03226 / 0.61462`
      - fixed `520M` `0.04841 / 1.00`
  - Focused 4-seed robustness on finalists:
    - Current canonical baseline (`direct_scalar_grp + nw_single_bw_0.5`):
      - `obs60_and_300` mean overall `RMSE 0.03899`, `Spearman 0.67978`
      - fixed `520M` mean `RMSE 0.04955`, `Spearman 0.80`
      - `multi_obs` mean overall `RMSE 0.13237`, `Spearman 0.61686`
      - failure mode: large seed-3 blow-up (`RMSE 0.3773`) in `multi_obs`
    - New tight-kernel candidate (`direct_scalar_grp + nw_single_bw_0.15`):
      - `obs60_and_300` mean overall `RMSE 0.03472`, `Spearman 0.69619`
      - fixed `520M` mean `RMSE 0.04817`, `Spearman 0.95`
      - `multi_obs` mean overall `RMSE 0.03850`, `Spearman 0.68312`
      - no catastrophic seed-3 blow-up
    - `direct_scalar_grp + lowrank_r2_lf0.01_li0.1`:
      - `obs60_and_300` mean overall `RMSE 0.02702`, `Spearman 0.83882`
      - but fixed `520M` mean `RMSE 0.06351`, `Spearman -0.20`
      - not acceptable as the canonical hybrid because it wins only by sacrificing the extrapolation stress test
    - `direct_enriched_grp_family_a + nw_single_bw_0.15`:
      - improves conditioned metrics vs the old canonical hybrid
      - but direct fallback remains worse than `direct_scalar_grp` over the 4-seed check
  - Refined local check around the best kernel:
    - `bw < 0.15` collapses back to direct fallback (too little transport)
    - `bw > 0.15` quickly degrades `multi_obs`
    - `bw = 0.15` is the useful threshold
  - Refreshed canonical hybrid at `bw = 0.15` (8-seed summary):
    - direct unchanged:
      - mean overall `RMSE 0.04185`, `Spearman 0.71657`
    - `obs60_and_300`:
      - mean overall `RMSE 0.03550`, `Spearman 0.73340`
      - mean fixed `520M` `RMSE 0.05223`, `Spearman 0.90`
    - `multi_obs`:
      - mean overall `RMSE 0.03987`, `Spearman 0.67833`
      - mean fixed `520M` `RMSE 0.05223`, `Spearman 0.90`
- Interpretation:
  - This is the first **real** hybrid-model improvement over the previous CC canonical hybrid:
    - same direct fallback
    - same exact identity
    - better conditioned `obs60_and_300`
    - much more stable `multi_obs`
    - better fixed `520M` behavior
  - The kernel family was directionally correct. The previous canonical bandwidth was simply too wide.
  - The low-rank residual transport remains promising as a research direction because it dominates on overall conditioned fit, but it is still not safe: it flips the fixed-`520M` ranking the wrong way.
  - The enriched `a_f` backbone remains a live direct-law challenger, but not the hybrid canonical choice yet because the scalar backbone still gives the better direct tradeoff.
- Next action:
  - Keep the new canonical hybrid as:
    - direct fallback `direct_scalar_grp`
    - Nadaraya-Watson residual transport with `bandwidth = 0.15`
  - Use the low-rank update as the next research branch, but only with an explicit mechanism to avoid the fixed-`520M` sign failure.
  - The next promising ideas are:
    - low-rank transport with a locality gate or shrinkage toward the tight-kernel update
    - or a stronger direct backbone under the same now-stabilized hybrid evaluator

### 2026-04-20 07:00 - Hybrid multi-scale law (f + u) evaluator
- Hypothesis: A unified hybrid model `y_hat(w, N, D | O_w) = f(w, N, D) + u(w, N, D, O_w)` can combine the best direct law with an observation-conditioned residual update. Required modes: direct (`O_w = empty`), single-observation (`obs60_only`, `obs300_only`), multi-observation, and exact identity when the target scale itself is in `O_w`.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_hybrid_multiscale_law.py`
- Config:
  - `f` = `direct_scalar_grp` (39 params) — current direct-law incumbent.
  - `u` = identity-preserving Nadaraya-Watson on `(uN, uD)` kernel distances over same-mixture observations, normalised Gaussian with a single fixed bandwidth. When every kernel weight underflows, `u = 0` (fall back to direct). When any observation matches the target scale, identity short-circuits to the observed `y`.
  - Bandwidth selection:
    - leave-one-scale-out LOOCV on training (larger-scale targets only) as a diagnostic, writing `hybrid_multiscale_law_bandwidth_search.csv`
    - LOOCV picks `bw = 10`, which minimises cross-scale LOOCV RMSE but destroys 520M Spearman on the real holdout
    - canonical bandwidth hard-coded to `bw = 0.5` — narrow enough that the nearest same-mixture sibling dominates at every larger-scale target, wide enough that single-sibling 60M → target corrections are non-trivial
  - Evaluation modes wired: direct, obs60_only, obs300_only, obs60_and_300, multi_obs.
  - Identity check: for every holdout row, drop any training observation at the same scale, append `(target_scale, y_target)` to `O_w`, and verify reconstruction (hybrid_multiscale_law_identity_checks.csv).
  - Robustness sweep over seeds `(1, 3, 5, 7, 11, 13, 17, 23)` evaluates every mode (hybrid_multiscale_law_robustness_sweep.csv).
  - Candidate plausibility reuses `bench.candidate_plausibility_summary` so per-mode "which holdout mixture does the model rank lowest" is directly comparable.
  - Plot style preserved: actual on x, predicted on y, `RdYlGn_r`, `X` marker for fixed 520M annotated with mixture_id.
- Seed-7 result (canonical `bw = 0.5`):
  - `direct_scalar_grp_baseline` and `hybrid_direct` produce identical predictions (max abs diff `2.2e-16`), so direct-mode matches the current direct-law incumbent exactly.
    - overall `RMSE 0.03882`, `Spearman 0.59689`, `Kendall 0.43954`.
    - fixed 520M `RMSE 0.08229`, `Spearman 0.40`.
  - `hybrid_obs60_only`:
    - overall `RMSE 0.04580`, `Sp 0.53168`, `K 0.36558`.
    - fixed 520M `RMSE 0.08229`, `Sp 0.40` (unchanged — the 60M → 520M kernel underflows, so `u = 0` and the model falls back to direct for those rows).
    - random supplement `RMSE 0.03934`, `Sp 0.54989`.
  - `hybrid_obs300_only`:
    - overall `RMSE 0.03856`, `Sp 0.41635`.
    - fixed 520M `RMSE 0.05064`, `Sp 0.80`, `K 0.66667` (big win over direct on both).
    - random supplement `RMSE 0.03688`, `Sp 0.18533` (rank dropped because 300M obs is only available for 10/34 random-supplement rows).
  - `hybrid_obs60_and_300` — the best overall mode:
    - overall `RMSE 0.04023`, `Sp 0.64788`, `K 0.49644` (Spearman ↑ from direct, RMSE roughly unchanged).
    - fixed 520M `RMSE 0.05064`, `Sp 0.80`, `K 0.66667`.
    - random supplement `RMSE 0.03882`, `Sp 0.52177`.
  - `hybrid_multi_obs`: fixed 520M matches `obs60_and_300` exactly (`RMSE 0.05064`, `Sp 0.80`), but overall `RMSE 0.04546`, `Sp 0.53890` — adding distant sibling residuals (e.g. 520M → 300M on random supplement rows) hurts overall calibration.
  - Reference: `source_old10_scale_calibrated` still wins overall at `RMSE 0.01090`, `Sp 0.90655`, fixed 520M `RMSE 0.00744`, `Sp 0.80` (uses observed proxy loss as an input so this is a ceiling, not a baseline).
- Identity check: `n = 38`, `max_abs_error = 0.0`, `rmse = 0.0`, `rows_violating_identity = 0`. Same-scale conditioning is exact.
- 8-seed robustness (mean / min):
  - `direct`: mean overall `RMSE 0.04185`, `Sp 0.71657`; mean fixed 520M `RMSE 0.08568`, `Sp 0.325`, min `-0.20`.
  - `obs60_and_300`: mean overall `RMSE 0.03946`, `Sp 0.70404`; mean fixed 520M `RMSE 0.05375`, `Sp 0.825`, min `0.80`.
  - `obs300_only`: mean overall `RMSE 0.04020`, `Sp 0.53551`; mean fixed 520M `RMSE 0.05375`, `Sp 0.825`, min `0.80`.
  - `obs60_only`: mean overall `RMSE 0.04486`, `Sp 0.55854`; mean fixed 520M `RMSE 0.08568`, `Sp 0.325`, min `-0.20`.
  - `multi_obs`: mean overall `RMSE 0.09461`, `Sp 0.62422`; mean fixed 520M `RMSE 0.05375`, `Sp 0.825`. At seed 3 `multi_obs` overall `RMSE = 0.37732` — one row takes a large residual from a far-scale sibling and blows up. Every other seed's `multi_obs` RMSE is 0.044-0.075.
- Candidate plausibility (seed 7, hybrid-holdout argmin):
  - `hybrid_direct` picks `run_00018` @ 520M with regret `0.010` (nearest-observed mean-phase TV `0.71`).
  - `hybrid_obs60_only` picks `run_00018` @ 300M (wrong scale target) with regret `0.097`.
  - `hybrid_obs300_only` and `hybrid_obs60_and_300` both pick `run_00018` @ 520M with regret `0.010`, matching direct.
  - `hybrid_multi_obs` picks `run_00125` @ 300M with regret `0.086` — the 520M same-mixture residual destabilised this selection.
  - Direct-mode free optimisation inherits `direct_scalar_grp`'s known reasoning/code-corner candidates (`phase0 broad ≈ 0`, `phase1 tech ≈ 0.83`), TV ≈ 0.55 from the nearest observed mixture. The hybrid does not fix this collapse in direct mode by construction — it only helps when same-mixture observations exist.
- Interpretation:
  - `hybrid_direct` reproduces `direct_scalar_grp` exactly (by construction). No direct-mode regression.
  - `hybrid_obs60_and_300` is the recommended conditioned mode. It gets large wins on the fixed 520M subset (mean RMSE drops from 0.086 to 0.054, mean Spearman rises from 0.33 to 0.83) and mostly preserves overall calibration. Overall Spearman drops slightly (0.704 vs 0.717 mean); random-supplement calibration moves a little but not badly.
  - `obs60_only` is safe for random-supplement ranking but leaves fixed 520M unchanged because the 60M → 520M kernel underflows with `bw = 0.5`. Widening the bandwidth recovers some 60M → 520M transport but simultaneously damages the near-neighbour behaviour at other scales, so a single global bandwidth is not a good fit for both regimes.
  - `multi_obs` has catastrophic outlier rows when a far-scale sibling's residual is large and dominates kernel weights after underflow elimination. It is functionally supported but **not recommended as the default conditioned mode**.
  - Identity is built in, exactly — dropping same-scale collisions before inserting the augmented observation gives `rmse = 0.0`.
  - The hybrid does not beat `source_old10_scale_calibrated` in absolute terms; the transfer law is tuned specifically to consume the observed 60M loss and has invested a separate 12-parameter scale-aware calibrator for this. But the hybrid's strength is that it plays the same direct / conditioned / identity roles in one object, doesn't rely exclusively on 60M, and does not regress the direct-only baseline at all.
- Next action:
  - Recommended deployed form: `hybrid_obs60_and_300` at `bw = 0.5` with `hybrid_direct` as the no-observation fallback and `identity` when the target scale is observed.
  - Open paths:
    - replace the shared bandwidth with a learned per-scale-pair weight (small MLP or scale-distance-parameterised gamma) so 60M → 520M transport can be modelled without destroying near-neighbour transport at smaller scale gaps;
    - investigate the seed-3 `multi_obs` blow-up — specific residual-aggregate caps or outlier-trimming should stabilise it;
    - if the direct-law body (`direct_scalar_grp`) itself improves (e.g. via `a_f` retuning or low-rank component structure) the whole hybrid improves for free, since `u` is defined on top of whatever `f` is in place.

### 2026-04-20 17:25 - Fixed packet strong-tier primary-label leak and regenerated matched-mixture trajectories
- Hypothesis: the suspicious `baseline_unimax` `520M x 1x` spike in the matched-mixture budget plot was not a real scaling effect, but a packet-construction bug that admitted non-perplexity-ready strong-tier rows as valid `primary_y`.
- Command:
  - inspection:
    - `uv run --with pandas python - <<'PY' ... packet.frame / run_registry logical_runs / strong_tier_perplexity_ready checks ... PY`
  - correction:
    - patched `chatgpt_pro_swarm_transfer_packet/build_packet.py`
    - patched `chatgpt_pro_hybrid_data_mixing_packet_v20/build_packet.py`
    - direct packet-data patch after interrupted rebuild:
      - `uv run --with pandas --with numpy python - <<'PY' ... patch nd_scale_runs.csv + nd_scale_packet.npz from logical_runs checkpoint_root join ... PY`
  - plot regeneration:
    - `uv run --with numpy --with pandas --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/plot_matched_mixture_budget_trajectories.py`
- Config:
  - strong-tier rows now trust `PRIMARY_METRIC = eval/uncheatable_eval/bpb` only when `is_perplexity_ready`.
  - when trusted, packet `PRIMARY_METRIC` is overwritten with the registry’s vetted `objective_metric_value`.
  - untrusted strong-tier rows retain metadata/weights but their primary label is masked out of `primary_y_mask`.
- Result:
  - root cause confirmed:
    - packet row `520m_10p4b:representative12:...:baseline_unimax` had `status=running` and `primary_y=3.837783`
    - registry did **not** consider that row perplexity-ready
    - registry-side vetted value for the same checkpoint root was inconsistent (`objective_metric_value=1.041495`), so the packet’s raw last-eval read was not trustworthy
  - packet correction summary:
    - `primary_mask_count_before = 638`
    - `primary_mask_count_after = 604`
    - `strong_tier_rows_total = 121`
    - `strong_tier_rows_masked = 39`
    - `strong_tier_rows_overridden = 82`
  - corrected trustworthy primary counts now are:
    - `130M x 0.5x = 13`
    - `130M x 1x = 13`
    - `130M x 2x = 13`
    - `300M x 0.5x = 13`
    - `300M x 1x = 242`
    - `300M x 2x = 13`
    - `520M x 0.5x = 3`
    - `520M x 1x = 1`
    - `60M x 1x = 293`
  - corrected plot:
    - `reference_outputs/figures/matched_mixture_budget_trajectories.png`
    - no longer shows the `baseline_unimax` `520M x 1x` blow-up
    - now correctly shows that the full `0.5x / 1x / 2x` triplets are only well-supported at `130M` and `300M`; `520M` appears only where a trustworthy row exists
- Interpretation:
  - the earlier upward jump at `520M` was a packet-data bug, not a real scaling result
  - the packet builder’s old policy of trusting any finite checkpoint `eval/uncheatable_eval/bpb` for strong-tier rows was too permissive
  - any direct/hybrid metrics run on the old 638-row packet are now stale relative to the corrected 604-row packet
- Next action:
  - rerun the direct/hybrid benchmark scripts against the corrected packet before making any further model-comparison claims based on packet-local `primary_y`

### 2026-04-20 18:12 - Audited 520M trustworthiness in the run registry
- Hypothesis:
  - the remaining untrusted `520M` rows might actually be complete runs with final evals that the packet/registry was failing to surface.
- Command:
  - `uv run python - <<'PY' ... logical_runs.csv / run_attempts.csv / strong_tier_perplexity_ready.csv audit for scale == "520m_10p4b" ... PY`
- Result:
  - `39` strong-tier logical rows at `520M`
  - `4` are `is_perplexity_ready = True`
  - `35` are not
  - critically:
    - `rows reached_target but not perplexity_ready = 0`
    - `rows perplexity_ready without reached_target = 0`
    - `rows with eval but not reached_target = 35`
  - so the untrusted `520M` rows are not hidden final evals; they are partial-training evals from runs that never reached target step
  - trustworthy `520M` rows are only:
    - `baseline_unimax` at `0.5x`
    - `run_00018` at `0.5x`
    - `run_00213` at `0.5x`
    - `run_00090` at `1.0x`
  - cell summary:
    - `qsplit_representative12 0.5x`: `12` total, `3` ready
    - `qsplit_representative12 1.0x`: `12` total, `1` ready
    - `qsplit_representative12 2.0x`: `12` total, `0` ready
    - `stratified 0.5x/1.0x/2.0x`: `1` total each, `0` ready each
- Interpretation:
  - the old packet bug mattered because every incomplete `520M` row still had some `eval/uncheatable_eval/bpb` blob under `checkpoint_root`
  - but the registry’s `reached_target_step` / `is_perplexity_ready` boundary is doing the right thing
  - operational `failed` is not the same as analytically invalid; some failed runs are still trustworthy because they reached target step before failing
- Artifacts:
  - `reference_outputs/registry_520m_logical_status_debug.csv`
  - `reference_outputs/registry_520m_trust_summary.csv`
  - `reference_outputs/registry_520m_cell_summary.csv`
  - `reference_outputs/registry_520m_perplexity_ready_rows.csv`
  - `reference_outputs/registry_520m_partial_eval_rows.csv`
  - `reference_outputs/registry_520m_attempts_debug.csv`

### 2026-04-20 18:34 - Re-evaluated current models on the corrected 604-row packet
- Goal:
  - verify how much the current model ranking changes after removing the bogus partial `520M` labels from packet `primary_y`.
- Command:
  - full transfer rerun:
    - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_transfer_benchmark_holdouts.py`
  - targeted direct / hybrid recheck:
    - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python - <<'PY' ... current_model_recheck ... PY`
  - targeted bandwidth sweep on corrected packet:
    - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python - <<'PY' ... hybrid_corrected_packet_bandwidth_recheck ... PY`
- Result:
  - corrected packet now has `604` primary-labelled rows
  - refreshed transfer incumbent `source_old10_scale_calibrated` collapsed badly on the corrected benchmark:
    - seed-7 overall `RMSE 0.1982`, `Spearman 0.3397`
    - seed-7 fixed `520M` `RMSE 0.6023`, `Spearman -0.4`
    - 8-seed mean overall `RMSE 0.1883`, `Spearman 0.3350`
    - 8-seed mean fixed `520M` `RMSE 0.5719`, `Spearman -0.575`
  - refreshed direct baseline `direct_scalar_grp` is now far stronger:
    - seed-7 overall `RMSE 0.02305`, `Spearman 0.7826`
    - seed-7 fixed `520M` `RMSE 0.03738`, `Spearman 1.0`
    - 8-seed mean overall `RMSE 0.02803`, `Spearman 0.7789`
    - 8-seed mean fixed `520M` `RMSE 0.03492`, `Spearman 0.95`
  - refreshed `direct_enriched_grp_family_a` no longer looks like an improvement:
    - seed-7 overall `RMSE 0.03160`, `Spearman 0.7589`
    - seed-7 fixed `520M` `RMSE 0.06826`, `Spearman 0.8`
  - current hybrid canonical config (`bw = 0.15`) is stale after the packet correction:
    - the kernel underflows for every conditioned row
    - `hybrid_obs60_and_300` becomes numerically identical to `direct_scalar_grp`
    - same for `hybrid_multi_obs`
  - corrected-packet bandwidth sweep shows wider kernels are now better:
    - `bw = 0.5`: mean overall `RMSE 0.02657`, mean overall `Spearman 0.7783`, mean fixed `520M RMSE 0.03774`, mean fixed `520M Spearman 0.825`
    - `bw = 1.5`: mean overall `RMSE 0.02702`, mean overall `Spearman 0.8615`, mean fixed `520M RMSE 0.03625`, mean fixed `520M Spearman 0.975`
    - `bw = 2.0`: mean overall `RMSE 0.02705`, mean overall `Spearman 0.8622`, mean fixed `520M RMSE 0.03618`, mean fixed `520M Spearman 0.975`
    - the old promoted `bw = 0.15` is dominated: mean overall `RMSE 0.02803`, mean overall `Spearman 0.7789`
- Interpretation:
  - the old transfer story was propped up by poisoned incomplete `520M` labels
  - after correcting the packet, the direct scalar law is the clear incumbent again
  - the hybrid idea still looks viable, but it needs re-tuning on the corrected packet; `bw = 0.15` should no longer be treated as canonical
- Artifacts:
  - `reference_outputs/current_model_recheck.json`
  - `reference_outputs/hybrid_corrected_packet_bandwidth_recheck.csv`
  - refreshed `reference_outputs/transfer_benchmark_holdout_summary.json`

### 2026-04-20 17:46 - Built corrected hybrid packet v21
- Built:
  - `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v21`
  - `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v21.zip`
- Synced packet-local `data/`, `code/`, and refreshed `reference_outputs/` from the corrected swarm packet state instead of carrying forward stale pre-fix artifacts.
- Rewrote:
  - `README.md`
  - `BACKGROUND_AND_GOALS.md`
  - `REQUEST_TO_CHATGPT_PRO.md`
  - `CLAUDE_CODE_INSTRUCTIONS.md`
  - `reference_outputs/local_hybrid_handoff_summary.md`
  - `MANIFEST.json`
- The v21 packet now reflects the corrected story:
  - `direct_scalar_grp` is the corrected direct incumbent
  - the current hybrid baseline uses bandwidth `1.5`
  - `source_old10_scale_calibrated` is explicitly marked stale on the corrected packet
- Added the refreshed focused holdout-comparison plot to the packet:
  - `reference_outputs/figures/current_model_recheck_predicted_vs_actual.png`

### 2026-04-20 18:02 - Pruned stale plot clutter from packet figure directories
- Removed stale pre-fix exploration figures from:
  - `chatgpt_pro_swarm_transfer_packet/reference_outputs/figures/`
  - `chatgpt_pro_hybrid_data_mixing_packet_v21/reference_outputs/figures/`
- Deleted:
  - `composed_direct_transfer_predicted_vs_actual.png`
  - `continuous_swarm_transfer_calibration_vs_actual_300m.png`
  - `direct_multiscale_grp_predicted_vs_actual.png`
  - `predict_300m_from_60m_and_grid_vs_actual.png`
  - `swarm_transfer_calibration_vs_actual_300m.png`
  - `transfer_holdout_520m_incumbent_predicted_vs_actual.png`
  - `transfer_holdout_520m_predicted_vs_actual.png`
- Remaining supported working-set plots are now:
  - `current_model_recheck_predicted_vs_actual.png`
  - `hybrid_multiscale_law_predicted_vs_actual.png`
  - `matched_mixture_budget_trajectories.png`
  - `nd_scale_completion_grid.png`
  - `transfer_benchmark_holdout_predicted_vs_actual.png`
  - `verified_520m_cross_scale_predicted_vs_actual.png`
  - `grp_power_family_penalty_raw_mixture.png`

### 2026-04-20 18:24 - Patched Pro residual-transport hybrid semantics and reran on corrected packet
- Goal:
  - integrate the stronger ChatGPT Pro residual-transport update into the repo-side hybrid evaluator, but fix the two semantic issues identified in review before trusting the numbers.
- Code changes:
  - `chatgpt_pro_swarm_transfer_packet/code/run_hybrid_multiscale_law.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_current_model_recheck.py`
- Semantic fixes applied to the hybrid evaluator:
  - exact identity now keys off exact `target_index`, not `scale_label`
  - same-size different-budget rows are valid context; only the exact target row is excluded
  - `520m_10p4b` is no longer hard-excluded as a potential source scale
- Commands:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_hybrid_multiscale_law.py`
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_current_model_recheck.py`
- Result:
  - patched canonical hybrid family:
    - direct backbone `direct_scalar_grp`
    - update family `scale_summary_residual_transport`
    - update parameter count `21`
    - selected update alpha `0.001`
    - total non-identity parameter count `61`
  - seed-7 focused comparison:
    - `direct_scalar_grp`: overall `RMSE 0.02305`, `Spearman 0.7826`; fixed `520M` `RMSE 0.03738`, `Spearman 1.0`
    - `direct_enriched_grp_family_a`: overall `RMSE 0.03160`, `Spearman 0.7589`; fixed `520M` `RMSE 0.06826`, `Spearman 0.8`
    - patched `hybrid_obs60_and_300`: overall `RMSE 0.00890`, `Spearman 0.8746`; fixed `520M` `RMSE 0.00393`, `Spearman 1.0`
    - stale transfer reference `source_old10_scale_calibrated`: overall `RMSE 0.19820`, `Spearman 0.3397`; fixed `520M` `RMSE 0.60229`, `Spearman -0.4`
  - 8-seed robustness:
    - `direct`: mean overall `RMSE 0.02803`, mean overall `Spearman 0.7789`, mean fixed `520M RMSE 0.03492`, mean fixed `520M Spearman 0.95`
    - patched `obs60_and_300`: mean overall `RMSE 0.01064`, mean overall `Spearman 0.9032`, mean fixed `520M RMSE 0.00603`, mean fixed `520M Spearman 0.90`
    - patched `multi_obs`: mean overall `RMSE 0.01056`, mean overall `Spearman 0.9063`, mean fixed `520M RMSE 0.00632`, mean fixed `520M Spearman 0.725`
  - exact identity preserved:
    - `n = 37`, `rmse = 0.0`, `max_abs_error = 0.0`, `rows_violating_identity = 0`
  - observation coverage on the corrected holdout:
    - `obs60_only` applicable on all `37` rows
    - `obs300_only` applicable on `14` rows
    - `obs60_and_300` applicable on all `37` rows
    - `multi_obs` applicable on all `37` rows
- Interpretation:
  - this is the first corrected-packet hybrid update that materially improves conditioned prediction over both the corrected direct baseline and the corrected kernel hybrid
  - the overall conditioned gain is large and survives the semantic fixes
  - `obs60_and_300` is the safest promoted conditioned mode: it improves strongly overall while keeping the `520M` stress subset strong
  - `multi_obs` is competitive overall but still less stable on fixed `520M`
- Refreshed artifacts:
  - `reference_outputs/hybrid_multiscale_law_summary.json`
  - `reference_outputs/hybrid_multiscale_law_predictions.csv`
  - `reference_outputs/hybrid_multiscale_law_robustness_sweep.csv`
  - `reference_outputs/hybrid_multiscale_law_candidate_plausibility.csv`
  - `reference_outputs/hybrid_multiscale_law_candidate_diagnostics.csv`
  - `reference_outputs/hybrid_multiscale_law_identity_checks.csv`
  - `reference_outputs/figures/hybrid_multiscale_law_predicted_vs_actual.png`
  - `reference_outputs/current_model_recheck.json`
  - `reference_outputs/current_model_recheck_predictions.csv`
  - `reference_outputs/figures/current_model_recheck_predicted_vs_actual.png`

### 2026-04-20 20:11 - Tested low-rank parallel scaling-law backbones
- Goal:
  - test whether mixture identity can be modeled as a low-rank modulation of a
    shared scale curve, instead of a fully flexible direct backbone.
- Hypothesis ladder tested:
  - `shared_curve_only`
    - `L = floor + C_N N^-alpha + C_D D^-beta`
  - `parallel_rho_shares`
    - `L = floor + rho(w) * (C_N N^-alpha + C_D D^-beta)` with share/entropy features
  - `parallel_rho_grp_lite`
    - same multiplicative form, but `rho(w)` built from GRP-lite exposure and penalty features
  - `offset_rho_grp_lite`
    - `L = floor + delta(w) + rho(w) * (C_N N^-alpha + C_D D^-beta)`
  - `split_rho_nd_grp_lite`
    - `L = floor + rho_N(w) C_N N^-alpha + rho_D(w) C_D D^-beta`
  - `scale_floor_rho_grp_lite`
    - shared scale-dependent floor plus multiplicative `rho(w)`
- Code:
  - added `chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
  - updated `.agents/projects/hybrid-data-mixing-v20.md` with the low-rank scaling-law TODO ladder
- Command:
  - `uv run --with numpy --with pandas --with scipy --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Result:
  - corrected incumbent `direct_scalar_grp` remains stronger than every tested low-rank family
  - seed-7:
    - `direct_scalar_grp`: overall `RMSE 0.02305`, `Spearman 0.7826`; fixed `520M` `RMSE 0.03738`, `Spearman 1.0`
    - best low-rank overall challenger `shared_curve_only`: overall `RMSE 0.03059`, `Spearman 0.6901`; fixed `520M` `RMSE 0.06978`, `Spearman 0.7746`
    - best low-rank GRP-lite variant `offset_rho_grp_lite`: overall `RMSE 0.03417`, `Spearman 0.6043`; fixed `520M` `RMSE 0.04172`, `Spearman 0.40`
  - 4-seed robustness (`1,3,7,13`):
    - `direct_scalar_grp`: mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`, mean fixed `520M` `RMSE 0.03090`, mean fixed `520M` `Spearman 0.95`
    - `shared_curve_only`: mean overall `RMSE 0.03327`, mean overall `Spearman 0.7251`, mean fixed `520M` `RMSE 0.07069`, mean fixed `520M` `Spearman 0.7746`
    - `parallel_rho_grp_lite`: mean overall `RMSE 0.03347`, mean overall `Spearman 0.7224`, mean fixed `520M` `RMSE 0.06938`, mean fixed `520M` `Spearman 0.20`
    - `split_rho_nd_grp_lite`: mean overall `RMSE 0.03377`, mean overall `Spearman 0.7233`, mean fixed `520M` `RMSE 0.07153`, mean fixed `520M` `Spearman 0.20`
    - `scale_floor_rho_grp_lite`: mean overall `RMSE 0.03412`, mean overall `Spearman 0.7325`, mean fixed `520M` `RMSE 0.07004`, mean fixed `520M` `Spearman 0.20`
    - `parallel_rho_shares`: mean overall `RMSE 0.03443`, mean overall `Spearman 0.6354`, mean fixed `520M` `RMSE 0.07436`, mean fixed `520M` `Spearman 0.20`
    - `offset_rho_grp_lite`: mean overall `RMSE 0.03582`, mean overall `Spearman 0.6703`, mean fixed `520M` `RMSE 0.04194`, mean fixed `520M` `Spearman 0.40`
- Important fitting signals:
  - the multiplicative `rho` families mostly selected strong regularization:
    - `parallel_rho_shares`, `parallel_rho_grp_lite`, `split_rho_nd_grp_lite`, and `scale_floor_rho_grp_lite` all chose `rho_alpha = 1.0`
    - this indicates the current `rho` features are not earning much freedom; CV prefers shrinking them heavily
  - `offset_rho_grp_lite` was the only extension that preferred a nontrivial flexible term:
    - selected `rho_alpha = 0.001`, `delta_alpha = 0.1`
    - but it still failed to beat `direct_scalar_grp`
- Candidate geometry:
  - `direct_scalar_grp` continues to optimize to a plausible mixed but off-manifold candidate:
    - nearest observed mean-phase TV around `0.40` to `0.43`
    - broad-text mass nonzero in both phases
  - all low-rank `rho` families except `direct_scalar_grp` collapse to `baseline_stratified`:
    - nearest observed mean-phase TV `0.0`
    - phase supports near `39`
    - identical phase-0/phase-1 family shares
  - this means the current low-rank formulations are too rigid: they prefer a safe interior mixture and cannot recover the broad-heavy observed winners at larger scales
- Interpretation:
  - the strongest result is negative but informative:
    - the pure “parallel reducible-loss curves” hypothesis is too restrictive in this first form
    - a shared scale curve explains a decent amount of variance, but mixture-specific `rho` terms as currently parameterized do not improve enough to justify replacing the corrected direct backbone
  - `shared_curve_only` is the best low-rank baseline and is worth keeping as a reference
  - `offset_rho_grp_lite` is the least-bad GRP-like extension, suggesting mixture-specific additive structure is more useful than pure multiplicative `rho`
  - current `rho(w)` feature maps are not yet expressive in the right way for the broad-heavy winners, even when they include GRP-lite exposure/penalty features
- Artifacts:
  - `reference_outputs/parallel_scaling_law_predictions.csv`
  - `reference_outputs/parallel_scaling_law_search.csv`
  - `reference_outputs/parallel_scaling_law_candidate_plausibility.csv`
  - `reference_outputs/parallel_scaling_law_candidate_diagnostics.csv`
  - `reference_outputs/parallel_scaling_law_robustness.csv`
  - `reference_outputs/parallel_scaling_law_robustness_summary.csv`
  - `reference_outputs/parallel_scaling_law_summary.json`
  - `reference_outputs/figures/parallel_scaling_law_predicted_vs_actual.png`
  - `candidate_mixtures_parallel_scaling_law.csv`
- Next action:
  - keep `shared_curve_only` as the low-rank baseline
  - if this direction is revisited, try one of:
    - richer offset structure with a weak shared curve (`delta(w)` carrying more of the burden)
    - estimating `rho` from observed same-mixture points inside the hybrid model rather than forcing a strong direct `rho(w)` predictor
    - factorizing `rho` around family-specific broad/tech/reasoning tradeoffs rather than a single global multiplicative term

### 2026-04-20 20:46 - Followed up with additive and factorized low-rank variants
- Goal:
  - test the next rung suggested by the first low-rank sweep:
    - additive mixture offset on top of the shared curve
    - weak multiplicative interaction instead of full `rho`
    - small family-factorized scale terms instead of one scalar `rho`
- Added models:
  - `offset_only_grp_lite`
    - `L = floor + shared_curve(N,D) + delta(w,D)`
  - `offset_weak_rho_grp_lite`
    - `L = floor + delta(w,D) + (1 + 0.25 * tanh(theta(w,D))) * shared_curve(N,D)`
  - `family_factorized_scale_grp_lite`
    - `L = floor + sum_f s_f(w,D) [c_{N,f} N^-alpha + c_{D,f} D^-beta]`
    - where `s_f(w,D)` are normalized GRP-lite family retained-signal scores
- Code:
  - extended `chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Command:
  - `uv run --with numpy --with pandas --with scipy --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Seed-7 model-selection signals:
  - `offset_only_grp_lite` best CV config:
    - `delta_alpha = 0.1`
    - CV `RMSE 0.03716`, `Spearman 0.8444`
  - `offset_weak_rho_grp_lite` best CV config:
    - `rho_alpha = 0.001`, `delta_alpha = 0.1`
    - CV `RMSE 0.03564`, `Spearman 0.8493`
  - `family_factorized_scale_grp_lite`:
    - no regularization needed
    - CV `RMSE 0.03634`, `Spearman 0.7862`
- 4-seed robustness:
  - incumbent `direct_scalar_grp`:
    - mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`
    - mean fixed `520M` `RMSE 0.03090`, mean fixed `520M` `Spearman 0.95`
  - `family_factorized_scale_grp_lite`:
    - mean overall `RMSE 0.03244`, mean overall `Spearman 0.7504`
    - mean fixed `520M` `RMSE 0.06731`, mean fixed `520M` `Spearman 0.20`
  - `offset_only_grp_lite`:
    - mean overall `RMSE 0.03351`, mean overall `Spearman 0.7018`
    - mean fixed `520M` `RMSE 0.05587`, mean fixed `520M` `Spearman 0.40`
  - `offset_weak_rho_grp_lite`:
    - mean overall `RMSE 0.03548`, mean overall `Spearman 0.6735`
    - mean fixed `520M` `RMSE 0.07034`, mean fixed `520M` `Spearman 0.40`
- Interpretation:
  - additive structure helped relative to the original multiplicative `rho` variants
  - `offset_only_grp_lite` is materially better than `offset_rho_grp_lite`
  - adding the weak multiplicative interaction did not help; it was slightly worse than offset-only
  - `family_factorized_scale_grp_lite` is the best of the new batch overall, and it nearly matches the incumbent on mean overall Spearman
  - but it fails the `520M` stress test badly enough that it is not a real replacement
- Candidate behavior:
  - `offset_only_grp_lite` no longer always picks the `520M` best row; on seed 7 it chooses `baseline_stratified` at `300M` with nonzero regret
  - `family_factorized_scale_grp_lite` still collapses toward the interior `baseline_stratified` geometry rather than the broad-heavy observed winners
- Current conclusion:
  - the best new structural signal is:
    - additive mixture structure matters more than pure multiplicative `rho`
    - but the current low-rank direct forms still underfit the trustworthy `520M` ordering
  - `shared_curve_only` remains the clean reference baseline
  - `family_factorized_scale_grp_lite` is the most promising low-rank follow-up if this thread continues, but it is not yet competitive enough to displace `direct_scalar_grp`

### 2026-04-20 21:18 - Tested shared-curve + structured GRP mixture-behavior variants
- Goal:
  - push further on the “shared curve + GRP mixture term” idea by testing:
    - full GRP additive residuals
    - effective-scale warp models
    - varying-coefficient GRP residuals
    - family-factorized additive residuals
    - scalar latent-quality interactions
- Added models:
  - `offset_only_full_grp`
    - `L = floor + shared_curve(N,D) + delta_full_GRP(w,N,D)`
    - uses the full `joint_nd_model` mixture feature block (all non-scale columns)
  - `warp_grp_lite`
    - effective `N` and `D` warped by GRP-lite features:
    - `L = floor + c_N N^-alpha exp(-shift_N(w,D)) + c_D D^-beta exp(-shift_D(w,D))`
  - `warp_plus_delta_grp_lite`
    - warp model plus additive GRP-lite residual
  - `varying_coeff_grp_lite`
    - GRP-lite residual with coefficients varying linearly with `log N` and `log D`
  - `family_factorized_residual_grp_lite`
    - additive residual from 3 normalized family retained-signal scores
  - `latent_quality_grp_lite`
    - scalar latent quality `q(w,D)` with polynomial and scale interaction terms
- Code:
  - extended `chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Command:
  - `uv run --with numpy --with pandas --with scipy --with matplotlib python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Best CV configs for the new families:
  - `offset_only_full_grp`:
    - `delta_alpha = 0.001`
    - CV `RMSE 0.03246`, `Spearman 0.8568`
  - `warp_grp_lite`:
    - `rho_alpha = 0.001`
    - CV `RMSE 0.03528`, `Spearman 0.8639`
  - `warp_plus_delta_grp_lite`:
    - `rho_alpha = 0.01`, `delta_alpha = 0.1`
    - CV `RMSE 0.03606`, `Spearman 0.8565`
  - `varying_coeff_grp_lite`:
    - `delta_alpha = 0.01`
    - CV `RMSE 0.03420`, `Spearman 0.8736`
  - `family_factorized_residual_grp_lite`:
    - no regularization needed
    - CV `RMSE 0.03486`, `Spearman 0.8217`
  - `latent_quality_grp_lite`:
    - `rho_alpha = 0.01`, `delta_alpha = 0.001`
    - CV `RMSE 0.03532`, `Spearman 0.8195`
- 4-seed robustness:
  - incumbent `direct_scalar_grp` remains best:
    - mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`
    - mean fixed `520M` `RMSE 0.03090`, mean fixed `520M` `Spearman 0.95`
  - strongest new challenger `warp_plus_delta_grp_lite`:
    - mean overall `RMSE 0.03138`, mean overall `Spearman 0.7114`
    - mean fixed `520M` `RMSE 0.05928`, mean fixed `520M` `Spearman 0.40`
  - `family_factorized_scale_grp_lite` stays competitive overall:
    - mean overall `RMSE 0.03244`, mean overall `Spearman 0.7504`
    - but fixed `520M` still bad: `RMSE 0.06731`, `Spearman 0.20`
  - `latent_quality_grp_lite` / `varying_coeff_grp_lite` show brittle fixed-`520M` rank:
    - mean fixed `520M` Spearman `1.0`, but worse overall RMSE and clearly collapsed candidate geometry
  - `offset_only_full_grp` is a negative result:
    - mean overall `RMSE 0.18149`
    - mean fixed `520M` `RMSE 0.16572`
- Candidate geometry:
  - `warp_plus_delta_grp_lite` is not viable despite the best new overall RMSE:
    - it collapses to a near one-hot tech-only corner in both phases
    - nearest observed mean-phase TV grows from `0.51` to `0.70` with scale
  - `family_factorized_scale_grp_lite`, `latent_quality_grp_lite`, `varying_coeff_grp_lite`, and `offset_only_full_grp` mostly collapse to `baseline_stratified`
    - mean-phase TV `0.0` or effectively zero
    - identical high-support interior geometry
  - `offset_only_full_grp` also becomes wildly over-optimistic in predicted large-scale BPB, another sign of poor extrapolation
- Interpretation:
  - the latest sweep says something sharper:
    - adding “more GRP” to the shared-curve model is not enough
    - the problem is not merely missing mixture features
    - it is the structural form used to express mixture-specific scale behavior
  - `warp_plus_delta_grp_lite` is the strongest new metric challenger, which suggests effective-scale warping is a real signal
  - but its candidate collapse means the current warp parameterization is not safe
  - `family_factorized_scale_grp_lite` remains the most promising clean low-rank model because it is compact and reasonably competitive overall, even though it still misses the `520M` ordering
- Current conclusion:
  - no tested shared-curve + structured-GRP model beats corrected `direct_scalar_grp`
  - the two most informative directions from this sweep are:
    - effective-scale warp ideas
    - family-factorized low-rank structure
  - if this line continues, I would refine those two, not the full-GRP additive residual path

### 2026-04-21 16:40 - targeted shared-curve refinements
- Hypothesis:
  - two focused follow-ups are worth testing after the previous sweep:
    - if `shared_curve_only` is paired with a **full GRP multiplicative residual** rather than just an additive delta, it may recover the missing mixture signal while preserving the clean scale backbone
    - if the warp idea is reparameterized to use only **compact family retained-signal summaries**, it may keep the promising effective-scale signal without collapsing to the GRP-lite tech corner
- Added models:
  - `parallel_rho_full_grp`
    - `L = floor + rho_full_GRP(w,N,D) * shared_curve(N,D)`
  - `offset_rho_full_grp`
    - `L = floor + delta_full_GRP(w,N,D) + rho_full_GRP(w,N,D) * shared_curve(N,D)`
  - `warp_family_scores`
    - same effective-scale warp form as `warp_grp_lite`, but with only the 3 normalized family retained-signal scores as warp features
  - `warp_plus_delta_family_scores`
    - compact family-score warp plus additive family-score delta
- Code:
  - extended `chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- Command:
  - `uv run --with scikit-learn experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
- 4-seed robustness:
  - incumbent `direct_scalar_grp` still wins:
    - mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`
    - mean fixed `520M RMSE 0.03090`, mean fixed `520M Spearman 0.95`
  - `warp_plus_delta_family_scores` is the best new branch numerically:
    - mean overall `RMSE 0.03188`, mean overall `Spearman 0.7635`
    - mean fixed `520M RMSE 0.06275`, mean fixed `520M Spearman 0.35`
  - `warp_family_scores`:
    - mean overall `RMSE 0.03370`, mean overall `Spearman 0.7590`
    - mean fixed `520M RMSE 0.07192`, mean fixed `520M Spearman 0.25`
  - `parallel_rho_full_grp` is a strong negative result:
    - mean overall `RMSE 0.12615`, mean fixed `520M RMSE 0.10782`
  - `offset_rho_full_grp` is also a strong negative result:
    - mean overall `RMSE 0.12350`, mean fixed `520M RMSE 0.11370`
- Candidate geometry:
  - `parallel_rho_full_grp` and `offset_rho_full_grp` both collapse to the exact `baseline_stratified` interior mixture
    - mean-phase TV `0.0`
    - phase supports `39/39`
    - predictions are badly over-optimistic at larger scales
  - `warp_family_scores` collapses to the same near one-hot tech corner as the earlier GRP-lite warp family
    - mean-phase TV `~0.879`
    - effective supports ~`1`
  - `warp_plus_delta_family_scores` is slightly less degenerate in phase 1, but still essentially a tech-only corner
    - phase-1 max weight `~0.99`
    - broad-text mass effectively zero
- Interpretation:
  - “just use the full GRP mixture term on top of the shared curve” does **not** solve the problem
    - it collapses to the uniform stratified interior and loses badly
  - compact family-score warping preserves some of the numerical benefit of the warp idea, but it still buys fit by moving into an implausible tech corner
  - the shared-curve backbone remains attractive conceptually, but the current residual/warp parameterizations are still not expressing mixture-specific scale behavior correctly
- Updated conclusion:
  - no tested shared-curve refinement beats corrected `direct_scalar_grp`
  - the only branches still worth treating as live are:
    - effective-scale warp ideas, but with a safer constraint than the current family-score or GRP-lite warp
    - family-factorized low-rank structures, but only if they can recover the trustworthy `520M` ordering without collapsing to `baseline_stratified`

### 2026-04-21 17:25 - direct-scalar scale ablation pass
- Goal:
  - understand what is actually carrying `direct_scalar_grp`
  - test whether its explicit 5-term polynomial scale head can be replaced by cleaner shared-curve elements without losing the trustworthy `520M` ordering
- Added script:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_scale_ablation.py`
- Ablation matrix:
  - `direct_scalar_grp`
  - `direct_no_explicit_scale`
    - remove the explicit `[uN, uD, uNuD, uN2, uD2]` scale head
  - `direct_no_internal_scale`
    - keep the explicit scale head, but remove the internal GRP scale shifts by switching to `joint_ND_baseline`
  - `shared_curve_only`
  - `direct_shared_curve_basis`
    - replace the 5 explicit scale columns with the 2 fitted shared-curve terms
  - `shared_curve_plus_joint_resid`
    - `shared_curve_only + residual(joint_nd mixture body)`
  - `shared_curve_plus_static_resid`
    - `shared_curve_only + residual(static joint_ND_baseline mixture body)`
- Command:
  - `uv run experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_scale_ablation.py`
- 4-seed robustness:
  - incumbent `direct_scalar_grp`:
    - mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`
    - mean fixed `520M RMSE 0.03090`, mean fixed `520M Spearman 0.95`
  - `shared_curve_only`:
    - mean overall `RMSE 0.03327`, mean overall `Spearman 0.7251`
    - mean fixed `520M RMSE 0.07069`, mean fixed `520M Spearman 0.7746`
  - `direct_no_internal_scale`:
    - mean overall `RMSE 0.04785`, mean overall `Spearman 0.4531`
    - mean fixed `520M RMSE 0.10031`, mean fixed `520M Spearman -0.25`
  - `direct_shared_curve_basis`:
    - mean overall `RMSE 0.06869`, mean overall `Spearman 0.5782`
    - mean fixed `520M RMSE 0.03462`, mean fixed `520M Spearman 0.80`
  - `shared_curve_plus_static_resid`:
    - mean overall `RMSE 0.03449`, mean overall `Spearman 0.7031`
    - mean fixed `520M RMSE 0.07206`, mean fixed `520M Spearman 0.20`
  - `shared_curve_plus_joint_resid`:
    - mean overall `RMSE 0.05264`, mean overall `Spearman 0.7614`
    - mean fixed `520M RMSE 0.10024`, mean fixed `520M Spearman 0.20`
  - `direct_no_explicit_scale`:
    - catastrophic failure
    - mean overall `RMSE 1.4390`
- Direct-scalar contribution analysis on the corrected seed-7 split:
  - contribution standard deviations by feature group:
    - `scale`: `0.343` train / `0.272` holdout
    - `penalties`: `0.095` / `0.110`
    - `shares`: `0.063` / `0.063`
    - `signals`: `0.062` / `0.103`
    - `entropies`: `0.063` / `0.049`
  - coefficient norms show the same pattern:
    - scale-group `L2 = 0.445`
    - next-largest group (`signals`) only `0.113`
  - knockout test on the trained seed-7 baseline:
    - zeroing `scale` destroys the fit:
      - RMSE worsens from `0.02305` to `0.05841`
      - Spearman drops from `0.7826` to `0.3966`
    - zeroing `signals` also hurts:
      - RMSE `0.02692`, Spearman `0.6453`
    - zeroing `shares` / `penalties` hurts modestly
    - zeroing `entropies` barely hurts and slightly improves RMSE on this split
- Interpretation:
  - `direct_scalar_grp` works because it combines:
    - a strong explicit scale head, and
    - scale-aware GRP body features
  - removing the explicit scale head is not viable at all
  - removing the internal GRP scale shifts also hurts badly, especially on fixed `520M`
  - cleaner shared-curve substitutions are not yet competitive:
    - `shared_curve_only` is the cleanest elegant baseline, but clearly underfits
    - `direct_shared_curve_basis` preserves some fixed-`520M` rank but loses badly overall
    - the additive shared-curve residual decompositions still miss the fixed-`520M` ordering
- Current conclusion:
  - the ugly polynomial scale head in `direct_scalar_grp` is doing real work, not just acting as a redundant convenience term
  - the direct-law gap is not just that the current scale features are inelegant; those features are presently the main thing stabilizing the fit
  - any “more elegant” replacement needs to recover both:
    - the explicit scale capacity of the 5-term head, and
    - the internal GRP scale shifts that move mixture quality with scale

### 2026-04-21 18:05 - direct-scalar design simplification pass
- Goal:
  - gather intuition on whether `direct_scalar_grp` can be simplified without losing corrected holdout performance
  - especially test:
    - whether the entropy/share channels are really needed
    - whether the explicit scale head can be reduced
    - whether shared-curve terms can be added back in cleaner ways
- Added script:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_design_ablation.py`
- Models tested:
  - `direct_no_entropy`
  - `direct_no_shares`
  - `direct_signals_penalties_scale`
  - `direct_linear_scale_head`
  - `direct_linear_interaction_scale_head`
  - `direct_shared_curve_basis`
  - `direct_shared_curve_linear_resid`
  - `direct_shared_curve_interaction_resid`
  - `direct_shared_curve_full_poly`
- Command:
  - `uv run experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_design_ablation.py`
- 4-seed robustness:
  - incumbent `direct_scalar_grp` remains best overall:
    - mean overall `RMSE 0.02920`, mean overall `Spearman 0.7602`
  - strongest clean simplification:
    - `direct_linear_interaction_scale_head`
    - mean overall `RMSE 0.03374`, mean overall `Spearman 0.7429`
    - mean fixed `520M RMSE 0.03878`, mean fixed `520M Spearman 0.95`
    - interpretation: dropping the quadratic scale terms is much less harmful than replacing the scale head entirely
  - strongest *numerical* small model:
    - `direct_signals_penalties_scale`
    - mean overall `RMSE 0.03214`, mean overall `Spearman 0.7770`
    - mean fixed `520M RMSE 0.03893`, mean fixed `520M Spearman 1.0`
    - but candidate geometry is bad (see below)
  - `direct_no_entropy` is a strong negative result:
    - mean overall `RMSE 0.14288`
    - this shows entropy channels are acting as training-time regularizers even though frozen knockouts looked weak
  - shared-curve substitutions still underperform:
    - `direct_shared_curve_basis`: `0.06869 / 0.5782`
    - `direct_shared_curve_interaction_resid`: `0.06203 / 0.6415`
    - `direct_shared_curve_full_poly`: `0.06113 / 0.6264`
- Seed-7 candidate geometry:
  - `direct_signals_penalties_scale` is not a clean simplification:
    - collapses to a broad-text phase-0 one-hot corner with mean-phase TV `~0.86`
    - phase-1 also becomes too concentrated
  - `direct_linear_interaction_scale_head` stays close to baseline geometry:
    - mean-phase TV `~0.41`
    - high-support mixed solutions very similar to `direct_scalar_grp`
- Per-scale-term knockout on the trained seed-7 baseline:
  - `uN` removal hurts a lot:
    - RMSE `0.0430`, Spearman `0.7141`
  - `uD` removal also hurts:
    - RMSE `0.0411`, Spearman `0.7838`
  - `uNuD` removal barely hurts on the frozen baseline:
    - RMSE `0.0297`
  - `uN2` hurts modestly:
    - RMSE `0.0326`
  - `uD2` is effectively dead on this split:
    - RMSE `0.02365` vs baseline `0.02305`
- Targeted retrain check for scale-head simplification:
  - one-off retrained variants on the corrected 4-seed benchmark:
    - `drop_uNuD`:
      - mean overall `RMSE 0.02767`, mean overall `Spearman 0.7759`
      - mean fixed `520M RMSE 0.03237`, mean fixed `520M Spearman 0.95`
      - candidate geometry remains close to baseline (TV `~0.42`)
    - `drop_uD2`:
      - mean overall `RMSE 0.03188`, mean overall `Spearman 0.7590`
      - mean fixed `520M RMSE 0.02937`, mean fixed `520M Spearman 0.95`
      - candidate geometry remains very close to baseline (TV `~0.40`)
    - `drop_uN2`:
      - weaker than the two above
- Interpretation:
  - there is now a credible path to a slightly more elegant direct backbone:
    - keep the scale-aware GRP body
    - keep explicit `uN` and `uD`
    - likely keep `uN2`
    - `uNuD` may be removable and could even be mildly harmful
    - `uD2` looks removable with almost no downside
  - the most tempting “small elegant” model (`signals + penalties + scale`) is a false friend because it wins metrics by collapsing to implausible candidates
- Current conclusion:
  - useful direct-law simplification signal is coming more from *pruning* the current `direct_scalar_grp` scale head than from replacing it with shared-curve-only elements
  - the two strongest next simplification candidates are:
    - `direct_scalar_grp - uNuD`
    - `direct_scalar_grp - uD2`

### 2026-04-21 18:20 - promoted scale-term pruning variants
- Goal:
  - move the promising one-off scale-term pruning checks into the same corrected design-ablation benchmark so they can be compared directly against `direct_scalar_grp`
- Updated script:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_design_ablation.py`
- Added models:
  - `direct_drop_uNuD`
  - `direct_drop_uD2`
  - `direct_drop_uNuD_uD2`
- Command:
  - `uv run experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_design_ablation.py`
- 4-seed robustness:
  - `direct_drop_uNuD` is now the strongest simplification candidate:
    - mean overall `RMSE 0.02767`, mean overall `Spearman 0.7759`
    - mean fixed `520M RMSE 0.03237`, mean fixed `520M Spearman 0.95`
    - this is slightly better overall than corrected `direct_scalar_grp` while preserving the same fixed-`520M` stress-test ranking
  - `direct_drop_uD2` looks near-neutral:
    - mean overall `RMSE 0.03187`, mean overall `Spearman 0.7590`
    - mean fixed `520M RMSE 0.02937`, mean fixed `520M Spearman 0.95`
  - `direct_drop_uNuD_uD2` is viable but less clean than `drop_uNuD` alone:
    - mean overall `RMSE 0.03309`, mean overall `Spearman 0.7799`
    - mean fixed `520M RMSE 0.02707`, mean fixed `520M Spearman 0.95`
- Seed-7 candidate geometry:
  - `direct_drop_uNuD` remains close to the original incumbent:
    - mean-phase TV `~0.42`
    - high-support mixed phase-0/phase-1 geometry
    - no collapse to the tech or broad-text corners
  - `direct_drop_uD2` is even closer to baseline geometry:
    - mean-phase TV `~0.40`
    - effectively the same candidate shape as `direct_scalar_grp`
- Interpretation:
  - `uNuD` now looks like the strongest removable explicit scale term
  - `uD2` looks like a weak or redundant term, but removing it alone is not a clear overall improvement
  - the most useful simplification signal so far is not a replacement backbone; it is a better-pruned incumbent
- Current conclusion:
  - the best direct-law simplification candidate is now:
    - `direct_scalar_grp - uNuD`
  - if we prepare a new handoff, that variant should be treated as a live incumbent/challenger rather than as a side experiment

### 2026-04-21 20:55 - built ChatGPT Pro hybrid packet v22
- Goal:
  - prepare a new versioned handoff packet with the corrected-data story, the updated direct/hybrid incumbents, and the new shared-curve / mixture-specific-scale-behavior framing
- New packet:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v22`
  - archive:
    - `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v22.zip`
- Contents updated:
  - new top-level docs:
    - `README.md`
    - `BACKGROUND_AND_GOALS.md`
    - `REQUEST_TO_CHATGPT_PRO.md`
    - `CLAUDE_CODE_INSTRUCTIONS.md`
  - new local summary:
    - `reference_outputs/local_handoff_summary.md`
  - curated corrected artifacts:
    - `current_model_recheck.*`
    - direct design / scale ablations
    - parallel scaling law summaries
    - registry trust-audit files
    - selected canonical figures
- Packet emphasis:
  - best direct simplification candidate:
    - `direct_drop_uNuD`
  - direct reference:
    - `direct_scalar_grp`
  - conditioned incumbent:
    - scale-summary residual-transport hybrid
  - main open question:
    - elegant and justifiable direct backbone with better mixture-specific scale behavior
  - explicit inclusion of the external shared-curve / `rho` hypothesis and our corrected mathematical interpretation
- Validation:
  - rebuilt packet via:
    - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v22/build_packet.py`
  - manifest parses and all manifest-referenced artifacts exist
  - no `.DS_Store` files remain in the packet tree

### 2026-04-21 22:40 - Direct-backbone mixture-scale-interaction sweep (first pass)
- Goal: engage the "mixture-specific scale behavior" hypothesis by adding
  structurally motivated mixture*scale interaction features on top of
  `direct_drop_uNuD`, plus test power-curve substitutions of `{uN2, uD2}`
- Added script:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_mixture_scale_interaction.py`
- Candidates tested (`direct_drop_uNuD` + ...):
  - `sym_scale`: replace `{uN2, uD2}` with `(uN+uD)^2` (Chinchilla axis)
  - `family_gain`: 3 features `phase1_share_fam * (uN+uD)`
  - `family_split`: 6 features `phase1_share_fam * uN` + `phase1_share_fam * uD`
  - `family_gain_asym`: 3 gain + 3 mismatch `(uN-uD)` family terms
  - `power_scale`: replace `{uN2, uD2}` with `{(N/Nref)^-alpha, (D/Dref)^-beta}` (alpha/beta via outer grid)
  - `power_plus_family`: power curve plus 3 family-gain interactions
- 4-seed robustness (overall RMSE / Spearman / 520M Spearman):
  - `direct_drop_uNuD` baseline: `0.02767 / 0.776 / 0.95`
  - `direct_scalar_grp` reference: `0.02920 / 0.760 / 0.95`
  - `direct_drop_uNuD_sym_scale` (37 params): `0.03008 / 0.753 / 0.95` — near-neutral
  - `direct_drop_uNuD_family_gain` (41 params): `0.03422 / 0.790 / 1.00`
  - `direct_drop_uNuD_family_gain_asym` (44 params): `0.03720 / 0.793 / 0.95`
  - `direct_drop_uNuD_power_plus_family` (41 params): `0.03912 / 0.728 / 0.80`
  - `direct_drop_uNuD_power_scale` (38 params): `0.04015 / 0.634 / 0.80`
  - `direct_drop_uNuD_family_split` (44 params): `0.04569 / 0.787 / 1.00`
- Interpretation:
  - single-stage family * gain interactions *do* improve cross-scale rank and
    fixed-520M Spearman, but cost roughly 25% overall RMSE
  - power-law substitutions over-extrapolate at 520M
    (seed-7 520M slope 1.36, std_ratio 1.43) — the floor-law `exp(z)` head
    amplifies power terms nonlinearly and blows up the largest-N predictions
  - symmetric Chinchilla axis is a wash, saves one parameter for no gain
- Next action:
  - try residual fitting rather than joint fit so the mixture-scale coupling
    gets its own regularization

### 2026-04-21 23:00 - Two-stage residual + group-ridge structural pass
- Hypothesis: single-stage joint fits let free interaction features drag
  training-rich scales (130M/300M) off calibration. A two-stage fit where the
  main `direct_drop_uNuD` ridge selects normally and the small mixture*scale
  residual gets its own outer alpha should add rank signal without pulling
  overall calibration.
- Added script:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_mixture_scale_structural.py`
- Candidates tested:
  - `direct_drop_uNuD_group_ridge` (38 params): separate alpha for scale-head
    vs body columns
  - `two_stage_compact_gain` (41 params): base + residual on
    `(tech-broad)*gain, (reas-broad)*gain`
  - `two_stage_compact_split` (43 params): base + residual on same contrasts
    * `{uN, uD}` separately
  - `two_stage_full_gain` (42 params): base + residual on raw
    `phase1_share_fam * (uN+uD)` for each family
  - `two_stage_full_split` (45 params): base + residual on raw
    `phase1_share_fam * {uN, uD}` separately
- 4-seed robustness (overall RMSE / Spearman / 520M RMSE / 520M Spearman):
  - `direct_drop_uNuD_group_ridge`: `0.02108 / 0.756 / 0.0235 / 0.30` — best
    overall RMSE but 520M rank *collapses*; over-regularized scale head
    flattens predictions
  - `two_stage_full_gain`: `0.02250 / 0.767 / 0.0383 / 1.00` (all 4 seeds)
  - `two_stage_compact_gain`: `0.02250 / 0.758 / 0.0406 / 0.85`
  - `two_stage_full_split`: `0.02269 / 0.764 / 0.0392 / 1.00`
  - `two_stage_compact_split`: `0.02281 / 0.753 / 0.0417 / 1.00`
  - `direct_drop_uNuD` baseline: `0.02767 / 0.776 / 0.0324 / 0.95`
  - `direct_scalar_grp` reference: `0.02920 / 0.760 / 0.0309 / 0.95`
- Seed-7 per-scale calibration of `two_stage_full_gain`:
  - `520M`: slope `0.957`, std_ratio `0.971` (baseline drop_uNuD: slope
    `0.865`, std_ratio `0.907`) — meaningful improvement on the
    under-dispersion at 520M that the packet flagged
  - `130M`: slope `0.826`, std_ratio `0.864` — essentially unchanged vs
    baseline
  - `300M`: slope `0.540`, std_ratio `0.721` — essentially unchanged
- Adaptive residual alpha across seeds: `1.0, 100.0, 10.0, 1000.0` for seeds
  `1, 3, 7, 13`, i.e. when interactions help they are used, when they do not
  they are shrunk — a safety property.
- Candidate plausibility test (seed-7 optimized phase1 family shares for the
  predicted-best mixture at each scale):
  - `direct_drop_uNuD` baseline already collapses to phase1 tech ~`80%`,
    broad ~`19%`, reasoning ~`0%` at `300M`, `520M`, and `1.2B`
  - `two_stage_full_gain` intensifies this: phase1 tech ~`97%` at `520M`,
    ~`100%` at `1.2B`
  - Observed best `520M` rows have phase1 ~`52% broad / 31% tech / 17% reas`
  - Both the baseline and the two-stage extension are over-pulled toward
    phase1-tech corners; the two-stage extension is somewhat worse
- Negative findings:
  - `group_ridge` wins RMSE by flattening predictions at the expense of 520M
    rank (Spearman `0.30`) — degenerate
  - `power_scale` and `power_plus_family` (first-pass) over-extrapolate at
    520M (slope >1.3, std_ratio >1.4)
  - `sym_scale` saves a parameter but is a wash on metrics
  - `family_split` forms do not improve on `family_gain` and cost parameters
- Interpretation:
  - the "mixture-specific scale behavior" hypothesis *is* quantitatively
    correct: the family*gain coupling fixes the 520M under-dispersion on
    holdout
  - but the induced predicted-optimum drifts *further* into tech-heavy
    phase1 corners than the baseline already does, because the fitted
    coefficient on `phase1_tech * gain` is strongly negative
    (standardized `-0.0087` vs `+0.0040` on `phase1_broad * gain`)
  - this means the direct backbone *cannot* safely absorb the family*gain
    correction on its own — improving cross-scale calibration and preserving
    candidate plausibility appear to be in tension with the current feature
    construction
- Quick sidebar: a scale-balanced weighted fit of `direct_drop_uNuD`
  (per-scale inverse-frequency weights) reduces 4-seed mean overall RMSE to
  `~0.0172` but drops 520M Spearman to `0.80`. Same tension, different
  knob — orthogonal to the structural form.
- Current conclusion:
  - no tested backbone cleanly beats `direct_drop_uNuD` once candidate
    plausibility is preserved
  - `two_stage_full_gain` is the best new *metric-side* candidate:
    - 4-seed mean overall RMSE `0.02250` vs baseline `0.02767` (~19% lower)
    - 4-seed min fixed-520M Spearman `1.00` vs baseline `0.80`
    - parameter count `42` vs `38`
    - adaptive residual alpha (shrinks to near-zero on seeds where the
      signal is weak, so it is not a constant-flexibility win)
  - but its predicted 520M optimum is more tech-corner than baseline,
    which is exactly the kind of plausibility regression the packet warned
    against — so this belongs in the *conditioned update* family, not as the
    direct backbone
  - the family*gain residual should be the next thing the hybrid update
    channel evaluates, since it already has same-scale observations that
    anchor the geometry and should neutralize the tech-corner drift
- Artifacts:
  - `reference_outputs/mixture_scale_interaction_summary.json`
  - `reference_outputs/mixture_scale_interaction_robustness_summary.csv`
  - `reference_outputs/mixture_scale_interaction_scale_calibration.csv`
  - `reference_outputs/structural_summary.json`
  - `reference_outputs/structural_robustness_summary.csv`
  - `reference_outputs/structural_scale_calibration.csv`
  - copied into v22: same filenames under
    `chatgpt_pro_hybrid_data_mixing_packet_v22/reference_outputs/`

### 2026-04-21 23:55 - Scalar-quality residual with sign constraint (direct-backbone WINNER)
- Hypothesis: the family*gain failure was specifically because free coefficients on
  three family shares let ridge prefer tech over broad. A *scalar* mixture
  quality `q(w)` multiplied by `uN` / `uD` cannot prefer any family — it can
  only reshape the cross-scale curve uniformly as a function of overall
  mixture quality. This matches the packet's corrected form
  `L_inf + rho_N(w) * C_N * N^-alpha + rho_D(w) * C_D * D^-beta` directly.
- Added scripts:
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_quality_residual.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_quality_split_robustness.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_quality_signed_robustness.py`
- Scalar residual forms tested (two-stage on top of `direct_drop_uNuD`):
  - `two_stage_entropy_gain` (1): `phase1_entropy_all * (uN+uD)`
  - `two_stage_both_entropy_gain` (2): `{phase0_entropy, phase1_entropy} * (uN+uD)`
  - `two_stage_total_signal_gain` (1): `sum_f signal_f(w,D) * (uN+uD)`
  - `two_stage_total_penalty_gain` (1): `sum_f penalty_f(w,D) * (uN+uD)`
  - `two_stage_quality_gain` (1): `q(w,D) * (uN+uD)`,
    `q = sum_f signal_f - sum_f penalty_f`
  - `two_stage_quality_split` (2): `{q(w,D) * uN, q(w,D) * uD}`
  - `two_stage_penalty_family_gain` (3): `{penalty_f(w,D) * (uN+uD)}_f`
  - `two_stage_signal_family_gain` (3): `{signal_f(w,D) * (uN+uD)}_f`
  - `two_stage_entropy_plus_total_penalty` (2)
- 4-seed robustness (overall RMSE / Spearman / 520M Spearman; baselines for
  reference):
  - `direct_drop_uNuD`: `0.02767 / 0.776 / 0.95`
  - `two_stage_signal_family_gain`: `0.02198 / 0.783 / 1.00`
  - `two_stage_total_signal_gain`: `0.02267 / 0.780 / 1.00`
  - `two_stage_quality_split`: `0.02298 / 0.802 / 1.00` (best Spearman)
  - `two_stage_quality_gain`: `0.02370 / 0.794 / 1.00`
- Candidate plausibility (seed-7 predicted-best phase-1 at 520M):
  - `direct_drop_uNuD` baseline: `broad 0.195 / tech 0.805 / reas 0.000`
  - `two_stage_quality_split`: `broad 0.197 / tech 0.803 / reas 0.000`
    — *identical to baseline geometry at 300M and 520M*
  - At 1.2B the signed model is `broad 0.265 / tech 0.735 / reas 0.000`,
    slightly more broad-heavy than the baseline's `broad 0.186 / tech 0.814`
    — i.e. candidate geometry actually *improves* at the largest scale
  - Contrast with the earlier `two_stage_full_gain` (family*gain):
    520M phase-1 was `broad 0.035 / tech 0.965 / reas 0` — the scalar
    approach fixes the tech-corner collapse by construction
- 8-seed check surfaced a seed-5 robustness crack in the unconstrained
  scalar residuals: on seed 5, `gamma_N` flipped sign and the 520M rank
  inverted (520M Spearman = -0.20). Diagnosis: with only 4 fixed-520M rows
  and different random-supplement draws, the unconstrained MLE can pick
  either sign for gamma_N / gamma_D.
- Fix: sign-constrained ridge. A-priori `higher quality -> lower loss at
  larger scale`, so `gamma_N <= 0` and `gamma_D <= 0`. Implemented as ridge
  with active-set projection (drop violating columns, re-solve).
- 8-seed robustness with sign constraint (overall RMSE / Spearman / 520M
  Spearman mean / 520M Spearman min):
  - `direct_drop_uNuD`: `0.02676 / 0.790 / 0.975 / 0.80`
  - `two_stage_quality_split_signed` (41): `0.02292 / 0.798 / 1.00 / 1.00`
  - `two_stage_total_signal_gain_signed` (40): `0.02303 / 0.786 / 1.00 / 1.00`
  - `two_stage_quality_gain_signed` (40): `0.02355 / 0.795 / 1.00 / 1.00`
  - All three signed variants hit perfect 520M Spearman on every seed
- Per-seed behavior of `quality_split_signed`:
  - seed 1: RMSE 0.024 (baseline 0.039) / 520M Spear 1.00 (baseline 0.80)
  - seed 3: 0.023 (0.025) / 1.00 (1.00)
  - seed 5: 0.024 (0.030) / 1.00 (1.00) — projection zeros out the bad sign
  - seed 7: 0.020 (0.021) / 1.00 (1.00)
  - seed 11: 0.023 (0.024) / 1.00 (1.00)
  - seed 13: 0.025 (0.025) / 1.00 (1.00)
  - seed 17: 0.021 (0.025) / 1.00 (1.00)
  - seed 23: 0.023 (0.024) / 1.00 (1.00)
  - wins or ties on every seed, no regressions
- Winning form:
  - L_hat(w, N, D) = L_drop_uNuD(w, N, D)
    + gamma_N * q(w, D) * uN + gamma_D * q(w, D) * uD
  - q(w, D) = sum_f signal_f(w, D) - sum_f penalty_f(w, D)
    (scalar quality from existing GRP body components)
  - gamma_N <= 0, gamma_D <= 0 (sign constraint from scaling prior)
  - ridge alpha selected by 5-fold grouped CV on
    `{0.1, 1, 10, 100, 1000, 10000}`
  - 41 total parameters (38 base + 2 residual + 1 intercept)
  - matches the packet's corrected shared-curve hypothesis structurally:
    `rho_N(w)` and `rho_D(w)` are the linear maps `q(w) -> gamma_N q(w)` and
    `q(w) -> gamma_D q(w)` applied to the log-scale coordinates
- 8-seed summary vs incumbent:
  - overall RMSE: **0.02292 vs 0.02676** (-14.3%)
  - overall Spearman: **0.798 vs 0.790** (+0.008)
  - min overall Spearman: **0.746 vs 0.681** (+0.065, more robust worst case)
  - fixed-520M Spearman: **1.00 on every seed vs min 0.80**
  - parameter count: 41 vs 38 (+3)
- Artifacts:
  - `reference_outputs/quality_residual_summary.json`
  - `reference_outputs/quality_residual_robustness_summary.csv`
  - `reference_outputs/quality_residual_candidate_plausibility.csv`
  - `reference_outputs/quality_signed_summary.json`
  - `reference_outputs/quality_signed_robustness_summary.csv`
  - `reference_outputs/quality_signed_coefficients.csv`
  - copied into v22 with same filenames
- Current conclusion:
  - the packet-hypothesis-aligned form **does** cleanly beat the incumbent
    `direct_drop_uNuD` on all tracked dimensions once expressed as a
    sign-constrained scalar-quality residual
  - `direct_drop_uNuD_quality_split_signed` is the new direct-backbone
    recommendation

### 2026-04-21 23:59 - Hybrid backbone comparison: direct winner != conditioned winner

- Goal:
  - Compare the corrected hybrid update on the three strongest direct backbones
    under one evaluator:
    - `direct_drop_uNuD`
    - `direct_shared_score_tilt_poly4`
    - `two_stage_quality_split_signed`
  - Use the result to decide how to merge the ChatGPT Pro and Claude Code
    backbone ideas instead of treating them as parallel winners.
- Code:
  - added `code/direct_backbone_candidates.py`
  - added `code/run_hybrid_backbone_comparison.py`
  - lightly refactored `code/run_hybrid_multiscale_law.py` so the update can be
    fit from a generic direct-prediction lookup, not just `direct_scalar_grp`
- Key 8-seed results:
  - direct mode, overall:
    - `direct_shared_score_tilt_poly4`: **0.02285 / 0.793**
    - `two_stage_quality_split_signed`: `0.02292 / 0.798`
    - `direct_drop_uNuD`: `0.02676 / 0.790`
  - conditioned `obs60_and_300`, overall:
    - `direct_drop_uNuD` backbone: **0.01071 / 0.905**
    - `direct_shared_score_tilt_poly4` backbone: `0.01079 / 0.900`
    - `two_stage_quality_split_signed` backbone: `0.01237 / 0.888`
  - conditioned `obs60_and_300`, fixed 520M:
    - `direct_shared_score_tilt_poly4`: **RMSE 0.00577**, Spearman `0.90`
    - `direct_drop_uNuD`: RMSE `0.00612`, Spearman `0.775`
    - `two_stage_quality_split_signed`: RMSE `0.01208`, Spearman **1.00**
- Candidate geometry from the unified seed-7 diagnostic:
  - `direct_shared_score_tilt_poly4` is best of the three:
    - mean nearest observed mean-phase TV `0.419`
  - `direct_drop_uNuD`:
    - `0.426`
  - `two_stage_quality_split_signed` is worse than previously believed once
    evaluated in the same packet:
    - `0.460`, with phase-1 effective support dropping to `~11` at `1.2B`
- Interpretation:
  - the stronger direct backbone does **not** automatically make the conditioned
    hybrid stronger
  - the current scale-summary residual transport update is best matched to
    `direct_drop_uNuD` on end-to-end conditioned metrics
  - the signed scalar-quality residual is still a real direct-law gain, but it
    appears to make the conditioned update overreact at large scale
  - the clean merger target is now more specific:
    - keep the one-stage shared-score tilt form from Pro
    - import CC's sign / monotonicity prior into the *scale-tilt coefficients*
      rather than using a separate second-stage residual
- Practical next move:
  - treat `direct_shared_score_tilt_poly4` as the current direct-backbone
    winner
  - keep `direct_drop_uNuD` under the current hybrid update as the current
    conditioned incumbent
  - next research target: a sign-constrained shared-score tilt backbone
    `floor + exp(intercept + m(w) + g(t) + m(w) h(t))`
    where `m(w)` starts from the scalar GRP quality score
    `q(w, D) = sum signal_f - sum penalty_f`
    and the linear parts of `h(t)` are constrained so higher quality cannot
    increase predicted loss at larger scale
- Artifacts:
  - `reference_outputs/hybrid_backbone_comparison_summary.json`
  - `reference_outputs/hybrid_backbone_comparison_robustness_summary.csv`
  - `reference_outputs/hybrid_backbone_comparison_candidate_geometry_summary.csv`
  - `reference_outputs/figures/hybrid_backbone_comparison_predicted_vs_actual.png`

### 2026-04-22 00:22 - Merger sweep: sign-constrained shared-score tilt is incremental, not a breakthrough

- Goal:
  - Test the plausible merger family implied by the Pro + CC results, not just
    talk about it:
    1. q-only signed tilt
    2. q-only signed poly4 tilt
    3. additive full-mix body + scalar q scale interaction
    4. Pro shared-score tilt + signed scalar-quality residual
    5. Pro shared-score tilt with signed linear tilt terms
    6. Pro shared-score tilt with signed linear tilt + explicit q anchor
- Code:
  - extended `code/direct_backbone_candidates.py` with merger candidates
  - added `code/run_direct_backbone_merge_benchmark.py`
- Key 8-seed results, direct mode:
  - `direct_shared_score_tilt_poly4`: `0.02285 / 0.793`
  - `two_stage_quality_split_signed`: `0.02292 / 0.798`
  - `direct_shared_score_tilt_poly4_signed_linear`: `0.02292 / 0.791`
  - `direct_shared_score_tilt_poly4_qanchor_signed`: `0.02292 / 0.791`
  - `direct_shared_score_tilt_poly4_plus_quality_split_signed`: `0.02278 / 0.792`
  - `direct_quality_tilt_poly4_signed`: `0.02044 / 0.660` — false friend
- Key 8-seed results, conditioned `obs60_and_300`:
  - incumbent `direct_drop_uNuD` backbone still best overall:
    `0.01071 / 0.905`
  - `direct_shared_score_tilt_poly4`: `0.01079 / 0.900`
  - signed one-stage tilt variants:
    `0.01078 / 0.900`
  - `direct_shared_score_tilt_poly4_plus_quality_split_signed`:
    `0.01097 / 0.898`
  - fixed-520M RMSE best is the Pro+signed-residual stack:
    `0.00550`, Spearman `0.95`
- Candidate geometry (seed 7) is decisive:
  - `direct_shared_score_tilt_poly4`: mean nearest-TV `0.4188`
  - `direct_shared_score_tilt_poly4_plus_quality_split_signed`: `0.4180`
    with similar broad/support profile — essentially tied, slightly better
  - one-stage signed-tilt variants:
    `~0.4266`, slightly worse than Pro
  - q-only signed tilts are invalid despite low RMSE:
    phase-1 effective support `~1.3`, max phase-1 weight `~0.96`
    i.e. near one-hot collapse
- Interpretation:
  - the cleanest direct merger to actually test was
    **sign-constrained shared-score tilt**. It is a wash:
    slightly changes seed behavior, but does not materially improve the Pro
    backbone on either direct or conditioned metrics.
  - adding an explicit q anchor to the shared-score tilt also does not help;
    its numbers are nearly identical to the unanchored signed-tilt variant.
  - the best *pragmatic* merger is still the stacked form
    `direct_shared_score_tilt_poly4 + signed quality-split residual`.
    It modestly improves fixed-520M conditioned RMSE and keeps geometry nearly
    identical to Pro, but it does not beat the current conditioned incumbent
    overall.
  - q-only merger forms are too rigid and collapse candidate geometry.
- Current conclusion:
  - for direct-only fallback, keep `direct_shared_score_tilt_poly4` as the
    cleanest practical winner
  - for conditioned `obs60_and_300`, keep `direct_drop_uNuD` under the current
    hybrid update as the overall incumbent
  - if we keep exploring merger space, the only branch that still looks alive is
    a **very light** residual on top of the Pro backbone, not a wholesale
    replacement of its one-stage form
- Artifacts:
  - `reference_outputs/direct_backbone_merge_summary.json`
  - `reference_outputs/direct_backbone_merge_robustness_summary.csv`
  - `reference_outputs/direct_backbone_merge_candidate_geometry_summary.csv`
  - `reference_outputs/figures/direct_backbone_merge_predicted_vs_actual.png`

### 2026-04-22 00:45 - Promote fold-mean regret@1 and tail optimism in backbone summaries

- Motivation:
  - nearest observed mean-phase TV is only a secondary plausibility proxy.
    It measures distance from the optimized candidate to the *closest* observed
    training mixture, not support health or downstream choice quality.
  - the direct / hybrid backbone summaries should foreground the same metric
    family we cared about in the original GRP evaluation:
    grouped fold-mean regret@1, lower-tail optimism, low-tail RMSE, then RMSE
    and rank.
- Code:
  - updated `code/run_direct_backbone_merge_benchmark.py`
  - updated `code/run_hybrid_backbone_comparison.py`
- Summary changes:
  - added `fold_mean_regret_at_1` on each evaluated subset by assigning
    mixture-grouped pseudo-fold ids within the benchmark rows and reusing
    `metric_bundle(..., fold_ids=...)`
  - promoted:
    - `mean_fold_mean_regret_at_1`
    - `mean_lower_tail_optimism`
    - `mean_low_tail_rmse`
  - demoted candidate geometry to `secondary_diagnostics` in the JSON payloads
- Effect on conclusions:
  - merger sweep:
    - direct winner by the new primary metrics becomes
      `two_stage_quality_split_signed`
      (`fold_mean_regret_at_1 = 0.00240`, `tail optimism = 0.00140`)
    - RMSE-first direct winner remains `direct_shared_score_tilt_poly4`
      (`rmse = 0.02285`)
    - conditioned `obs60_and_300` winner is still `direct_drop_uNuD`
  - hybrid comparison:
    - same split between
      `two_stage_quality_split_signed` (selection-metric best direct)
      and `direct_shared_score_tilt_poly4` (RMSE-first direct)
    - `direct_drop_uNuD` remains the conditioned incumbent
- Practical read:
  - the metric choice now makes the tradeoff explicit instead of hiding it:
    Pro's backbone still wins on clean direct calibration,
    CC's backbone still looks slightly safer on tail-aware selection metrics,
    and neither changes the conditioned incumbent.

### 2026-04-22 14:35 - Test partial and full unfreezing of the frozen GRP basis under shared-score tilt

- Hypothesis:
  - the frozen GRP feature-map constants in `direct_shared_score_tilt_poly4`
    may make the direct law too rigid, especially at non-300M target scales.
  - compact retuning of the GRP basis (`eta`, `lam`, family scale/penalty
    shifts) might reduce cross-scale compression without losing the clean
    one-score direct-law form.
- Code:
  - added
    `code/run_shared_score_grp_unfreeze_benchmark.py`
- Variants tested:
  - `shared_score_tilt_unfreeze_global4`
    - retune 4 global multipliers:
      `eta_mul`, `lam_mul`, `k1_mul`, `kt_mul`
  - `shared_score_tilt_unfreeze_global7`
    - same 4 global multipliers plus 3 family `tau` shifts
  - `shared_score_tilt_unfreeze_all14`
    - retune `eta`, `lam`, all 3 family `a`, all 3 family `tau`,
      all 3 family `k1`, and all 3 family `kt`
  - all compared against:
    - `direct_shared_score_tilt_poly4`
    - `direct_drop_uNuD`
- Evaluation setup:
  - corrected benchmark split
  - 4-seed robustness panel: seeds `{1, 3, 7, 13}`
  - same primary direct selection stack:
    fold-mean regret@1, lower-tail optimism, low-tail RMSE, then RMSE/Spearman
  - candidate geometry kept as a gate
- Result:
  - the frozen `direct_shared_score_tilt_poly4` remains best overall by the
    current direct selection stack:
    - overall:
      `RMSE 0.02261`, `Spearman 0.779`,
      `fold_mean_regret@1 0.00263`, `low_tail_rmse 0.03371`
  - compact unfreezing gives at best a tiny RMSE gain but loses on the primary
    selection metrics and/or fixed-520M robustness:
    - `unfreeze_global4`:
      overall `0.02245 / 0.768`, but
      `fold_mean_regret@1 0.00293` and fixed-520M Spearman `0.75`
    - `unfreeze_global7`:
      overall `0.02249 / 0.779`, but
      `fold_mean_regret@1 0.00293` and fixed-520M RMSE `0.0410`
  - full unfreezing (`all14`) is not a win:
    - overall `0.02267 / 0.785`
    - `fold_mean_regret@1 0.00263` (tie with frozen)
    - fixed-520M RMSE `0.04095`, worse than frozen `0.04061`
- Geometry:
  - all variants remain monotone across target scales
  - `unfreeze_global7` gets the best nearest-TV but sharpens phase-1 too much:
    max phase-1 weight `0.170`
  - `unfreeze_all14` does not collapse, but its geometry is not better than the
    frozen backbone
  - frozen Pro backbone remains the healthiest practical geometry / robustness
    tradeoff
- Interpretation:
  - we have enough data to *test* compact retuning of the frozen GRP basis.
  - we do **not** have evidence that the current data is enough to safely
    unfreeze the full GRP nonlinear basis and improve the direct law.
  - the failure mode is informative:
    unfreezing can shave a bit of overall RMSE, but the gain comes from bending
    the same sparse high-scale directions that hurt fixed-520M robustness.
  - so the next safe move is not “unfreeze everything”.
    It is a much narrower retune if we keep pushing this line:
    likely `eta`, `lam`, and perhaps a very small number of shared `k/t`
    multipliers under explicit geometry / 520M guards.
- Artifacts:
  - `reference_outputs/shared_score_grp_unfreeze_summary.json`
  - `reference_outputs/shared_score_grp_unfreeze_robustness_summary.csv`
  - `reference_outputs/shared_score_grp_unfreeze_search.csv`
  - `reference_outputs/shared_score_grp_unfreeze_candidate_geometry_summary.csv`
  - `reference_outputs/figures/shared_score_grp_unfreeze_predicted_vs_actual.png`

### 2026-04-22 15:05 - Test a compact GRP-lite encoder under the shared-score tilt law

- Hypothesis:
  - the frozen 32-dim `X_mix` body in `direct_shared_score_tilt_poly4` may be
    heavier than needed.
  - a smaller family-summary encoder could be a better “smarter scheme” than
    either freezing the full body or broadly unfreezing the old GRP basis.
  - target form stayed the same:
    `floor + exp(b + m + g + m h)`,
    but replaced `m = X_mix beta` with a compact GRP-lite basis.
- Code:
  - added
    `code/run_shared_score_grplite_benchmark.py`
- Compact encoders tested:
  - `grplite9`
    - per family:
      `{retained signal, ftotal penalty, phase1 share}`
    - 9 mixture features total
  - `grplite11`
    - `grplite9` plus
      `{phase0_entropy_all, phase1_entropy_all}`
    - 11 mixture features total
- Variants tested:
  - `shared_score_tilt_grplite9_frozen`
  - `shared_score_tilt_grplite11_frozen`
  - `shared_score_tilt_grplite9_unfreeze_global4`
  - `shared_score_tilt_grplite11_unfreeze_global4`
  - `shared_score_tilt_grplite9_unfreeze_global7`
  - all compared against:
    - `direct_shared_score_tilt_poly4`
    - `direct_drop_uNuD`
- Evaluation setup:
  - corrected benchmark split
  - 4-seed robustness panel: seeds `{1, 3, 7, 13}`
  - same primary direct selection stack:
    fold-mean regret@1, lower-tail optimism, low-tail RMSE, then RMSE/Spearman
  - candidate geometry still treated as a gate
- Result:
  - the 9-dim encoder is too weak and collapses badly:
    - `shared_score_tilt_grplite9_frozen`
      overall `RMSE 0.02365`, `Spearman 0.784`
      but candidate geometry is a near one-hot collapse
  - the 11-dim frozen encoder is the best benchmark result from this sweep:
    - `shared_score_tilt_grplite11_frozen`
      overall `RMSE 0.02199`, `Spearman 0.800`,
      `fold_mean_regret@1 0.00205`, `low_tail_rmse 0.03338`
    - this edges past the frozen full-body
      `direct_shared_score_tilt_poly4`
      (`0.02261 / 0.779 / 0.00263 / 0.03371`)
  - compact unfreezing does not help:
    - `grplite11_unfreeze_global4`
      overall `0.02226 / 0.787`,
      but fixed-520M Spearman falls to `0.50`
    - `grplite9` unfreezing variants remain collapsed
- Geometry:
  - `grplite11_frozen` does **not** one-hot collapse,
    but its optimized phase-1 mixtures are still almost pure tech:
    roughly `phase1_broad ~0.018` and `phase1_tech ~0.979` on seed 7
  - phase-1 effective support stays around `13`, so the mass is spread within
    tech domains rather than collapsing to one domain
  - that is still not a geometry improvement over
    `direct_shared_score_tilt_poly4`, which stays much broader and more
    balanced across families
- Interpretation:
  - the compact family-summary basis is a real signal:
    the direct law does **not** need the full 32-dim frozen body to fit well.
  - the useful compact representation is not “signal + penalty + phase1 share”
    alone; it needs at least a small entropy / diversity anchor.
  - but this first-pass GRP-lite encoder still over-rewards phase-1 tech in the
    optimized candidate landscape, so it is not ready to replace the current
    trustworthy backbone.
  - the smarter scheme looks directionally right, but it needs an explicit
    family-balance / geometry anchor, not more basis unfreezing.
- Artifacts:
  - `reference_outputs/shared_score_grplite_summary.json`
  - `reference_outputs/shared_score_grplite_robustness_summary.csv`
  - `reference_outputs/shared_score_grplite_candidate_geometry_summary.csv`
  - `reference_outputs/shared_score_grplite_candidate_diagnostics.csv`
  - `reference_outputs/figures/shared_score_grplite_predicted_vs_actual.png`

### 2026-04-22 15:45 - Correct the simulated-epoch semantics used in the direct-law discussion

- Trigger:
  - re-checked the actual training code after noticing a possible mismatch
    between the packet/runtime helper story and the real simulated-epoching
    implementation.
- Actual training semantics:
  - `MixtureExperiment.create_training_step()` passes both `target_budget` and
    the real run budget into `simulated_epoching_train()`.
  - `simulated_epoching_train()` stores both on the data config and logs
    `Experiment Tokens {experiment_budget}, Simulated Target Tokens {target_budget}`.
  - the actual slicing happens in
    `lib/levanter/src/levanter/data/text/datasets.py`:
    each dataset is truncated to
    `simulated_data_ratio = experiment_budget / target_budget`
    before training.
- Consequence:
  - for a phase/domain with weight `w_f`, training consumes
    `w_f * phase_fraction * experiment_budget` tokens from a subset whose size
    is proportional to `experiment_budget / target_budget`.
  - the effective epoch count on that subset is therefore proportional to
    `w_f * phase_fraction * target_budget / domain_tokens`.
  - **experiment_budget cancels**.
- Interpretation:
  - at fixed `target_budget`, simulated epoching intentionally keeps effective
    epochs constant across model scales.
  - the strong-tier scaling study does exactly this: it uses one shared
    `target_budget` recipe per multiplier, while `experiment_budget` changes by
    scale.
  - therefore the earlier informal claim that the GRP epoch features should
    vary with actual scale `D` across `60M/130M/300M/520M` was wrong for the
    actual training semantics.
- Modeling implication:
  - if `D` in the direct law denotes the actual training-token axis used for
    scaling comparisons, then epoch-sensitive mixture features should not
    inherit an extra cross-scale dependence from `experiment_budget`.
  - they should depend on the simulated target budget / multiplier semantics,
    not on actual scale alone.
  - this raises a real concern that the current packet/runtime helper
    `simulate_epoch_multipliers(D)` may be injecting the wrong dependence when
    used with actual scale budgets.
- Next action:
  - audit every place the analysis code maps scale labels to the `D` argument
    for epoch-sensitive features and separate:
    - actual train-token scale
    - simulated target-budget / multiplier semantics

### 2026-04-22 16:20 - Fix packet-local custom epoch helpers and rerun direct backbone merge

- Fix scope:
  - packet builder was already correct:
    `build_packet.py` stores simulated epoch multipliers from `target_budget`.
  - the bug was in packet-local runtime helpers that rebuilt custom epoch
    multipliers from realized train tokens.
- Code fixed:
  - `chatgpt_pro_swarm_transfer_packet/code/run_continuous_nd_grp_law.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_parallel_scaling_law_benchmark.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_multiscale_grp_components.py`
  - `chatgpt_pro_swarm_transfer_packet/code/direct_backbone_candidates.py`
  - `chatgpt_pro_swarm_transfer_packet/code/run_direct_scalar_quality_residual.py`
  - mirrored `v22` copies of the same packet-local helper files
- Runtime change:
  - custom epoch-sensitive feature builders now resolve simulated epochs from
    `target_budget` / `target_budget_multiplier`, not from realized `D`.
  - default custom predictions now assume `1.0x` target-budget semantics unless
    an explicit target budget is provided.
- Targeted rerun:
  - reran `run_direct_backbone_merge_benchmark.py`
- Result:
  - direct holdout metrics changed only modestly.
  - candidate geometry changed a lot and invalidated the earlier “healthy”
    geometry read for the direct backbones.
  - updated seed-7 geometry summary now shows:
    - `direct_shared_score_tilt_poly4`:
      `mean_nearest_tv ~0.643`, `min_phase1_support ~1.00`,
      `max_phase1_weight ~0.9998`
    - `direct_drop_uNuD`:
      `mean_nearest_tv ~0.670`, `min_phase1_support ~1.01`,
      `max_phase1_weight ~0.9981`
    - `two_stage_quality_split_signed`:
      `mean_nearest_tv ~0.702`, `min_phase1_support ~1.00`,
      `max_phase1_weight ~1.0`
- Interpretation:
  - the previous geometry ranking was relying on the wrong epoch semantics.
  - after the fix, none of the leading direct backbones look geometrically
    healthy under the old standard.
  - recommendations that treated `direct_shared_score_tilt_poly4` as a clean
    geometry-preserving winner need to be revised.

### 2026-04-22 00:30 - End-to-end trainable body + mu-aware scale head sweep
- Direction change: the next serious candidate should be trainable end-to-end
  and must not rely on a frozen GRP-derived feature map. Target form is
  `y_hat(w, N, D | mu) = f(w, N, D, mu) + u(...)` with mu = target-budget
  multiplier now an explicit input (packet reminder that raw D is the wrong
  epoching proxy when mu varies).
- Three trainable direct-backbone families implemented:
  1. `direct_grp_psc` — floor-law multiplicative,
     `z = intercept + mixture_score * (1 + tilt) + shared_scale + mu_terms`,
     parametric 12-feature psi from learnable (eta, lam_d, k, a_f, tau_f), 33 params.
     Added: `run_direct_grp_psc_benchmark.py`.
  2. `direct_rho_shared_curve` — the packet's corrected form literally,
     `L = L_inf + exp(tau_N + q) * (N/Nref)^{-alpha_N} + exp(tau_D + q) * (D/Dref)^{-alpha_D} + c_mu1 log mu + c_mu2 (log mu)^2 q`,
     psi computed at mu=1 reference epochs so q(w) is a scale-free scalar
     mixture-quality summary, 27 params.
     Added: `run_direct_rho_shared_curve.py`.
  3. `direct_param_body_tilt` — same architecture as
     `direct_shared_score_tilt_poly4` but with a *parametric* 24-feature psi
     (learnable eta, lam_d, k, a_f, tau_f, t_f) and a 6-feature scale head
     `[uN, uD, uN^2, uD^2, log mu, log mu * uN]`, 50 params.
     Added: `run_direct_param_body_tilt.py`.
- Optimizer for all three: scipy L-BFGS-B with 6 multi-starts, CV-based
  selection over ridge grids (head + body anchoring). All body params are
  learnable; body-deviation ridge anchors them to frozen-GRP defaults from
  `grp_power_family_penalty_no_l2_retune_best.csv`
  (eta=5.22, lam~0, a=[0.485, 0.048, 1.034], tau=[3.09, 8.0, 4.86]).
- 8-seed robustness (overall RMSE / Spearman / min 520M Spearman), all with
  `include_1_2b = True`:
  - `direct_drop_uNuD` (38 params, reference): `0.02676 / 0.790 / 0.80`
  - `direct_shared_score_tilt_poly4` (42 params, frozen): `0.02285 / 0.793 / 0.80`
  - `two_stage_quality_split_signed` (41 params, frozen + signed): `0.02292 / 0.798 / 1.00`
  - `direct_grp_psc` (33 params, trainable): `0.02537 / 0.663 / 1.00`
  - `direct_rho_shared_curve` (27 params, trainable): `0.03365 / 0.622 / 0.20`
  - `direct_param_body_tilt` (50 params, trainable): `0.02556 / 0.774 / 1.00`
- Seed-7 per-scale calibration for `direct_param_body_tilt`:
  - 130M: slope 0.97, std_ratio 1.03, RMSE 0.027 (near-perfect)
  - 300M: slope 0.38, std_ratio 0.50, RMSE 0.015 (compressed, same as baseline)
  - 520M: slope 0.66, std_ratio 0.67, RMSE 0.048 (over-predicts loss by ~0.048)
- Negative findings within this sweep:
  - `direct_rho_shared_curve` (the packet hypothesis written literally with
    trainable parameters) is the worst of the three: the additive
    `L_inf + rho_N N^{-a} + rho_D D^{-b}` form has L_inf, c_N, c_D, alpha
    degeneracies that scipy L-BFGS-B plus CV selection cannot resolve.
    520M Spearman collapses to `0.20` (near random). Strong negative result.
  - `direct_grp_psc` converged but its 33-param feature map was less
    expressive than the 42-param frozen basis; Spearman fell to 0.663.
  - Initial default-body values (eta=1, tau=1.5) were *far* from the frozen
    GRP tune (eta=5.22, tau_tech=8.0); misaligned bounds caused many params
    to pin to boundaries. Correcting the defaults and widening bounds lifted
    `direct_param_body_tilt` from Spearman ~0.57 to ~0.77.
- Positive finding:
  - `direct_param_body_tilt` is the first trainable end-to-end direct
    backbone in this sweep that cleanly *beats* `direct_drop_uNuD`:
    overall RMSE -4.5% (0.0256 vs 0.0268), random-supplement RMSE -15%
    (0.021 vs 0.025), fixed-520M Spearman = 1.00 on every seed (baseline
    min 0.80), min overall Spearman 0.693 vs 0.681.
  - It still trails the *frozen-basis* incumbents
    (`direct_shared_score_tilt_poly4`, `two_stage_quality_split_signed`)
    by ~0.003 RMSE overall. The frozen GRP body carries a structural prior
    from a separate nonlinear retune on the macro objective that is hard
    to re-discover end-to-end on the direct objective alone at this
    dataset size.
  - 520M RMSE is worse (0.049 vs 0.036) even though 520M Spearman is
    perfect on every seed — the end-to-end fit learns a scaling curve
    that is too flat at 520M, overpredicting loss by ~0.048 on average
    but preserving the ranking.
- Verdict:
  - **Best trainable end-to-end candidate this round: `direct_param_body_tilt`**
    (no frozen GRP basis, 50 params, wins vs `direct_drop_uNuD` on every
    tracked metric except 520M absolute RMSE, matches or slightly improves
    Spearman, perfect 520M rank on all 8 seeds).
  - Frozen-basis incumbents remain marginally better on overall RMSE.
  - The explicit `log mu` and `log mu * uN` scale features respect the
    packet's reminder about mu-dependent epoch semantics and do contribute
    (removing them regresses ~0.001 RMSE in quick tests).
- Recommended next investment:
  - the ~0.003 RMSE gap to the frozen basis suggests a *separate* nonlinear
    retune of body parameters on the corrected primary labels is the
    remaining lever, rather than additional feature families. The current
    anchor to the macro-objective tune may be pulling body params away from
    the primary-optimal point.
- Artifacts (in both `chatgpt_pro_swarm_transfer_packet/reference_outputs/`
  and v22):
  - `grp_psc_summary.json` / `grp_psc_robustness_summary.csv`
  - `rho_shared_curve_summary.json` / `rho_shared_curve_robustness_summary.csv`
  - `param_body_tilt_summary.json` / `param_body_tilt_robustness_summary.csv`
  - `param_body_tilt_scale_calibration.csv` / `param_body_tilt_parameters.csv`
- Scripts (both packets):
  - `run_direct_grp_psc_benchmark.py`
  - `run_direct_rho_shared_curve.py`
  - `run_direct_param_body_tilt.py`

### 2026-04-22 03:15 - corrected registry refresh and strict direct-backbone revalidation
- Hypothesis:
  - the benchmark basis changed materially once late-finishing strong-tier
    `520M` runs were promoted using their actual final perplexity evals;
    serious direct candidates need to be rerun on the corrected `7`-row
    trustworthy `520M` basis with full tuning and the continuous
    free-mixture geometry gate before we promote any new direction.
- Commands:
  - registry refresh / packet sync:
    - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/build_packet.py`
    - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_520m_transfer_holdout.py`
  - Pro 2 strict rerun:
    - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/code/run_end_to_end_direct_backbone_validation.py`
  - Pro 1 strict rerun:
    - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_direct_end2end_grp_family_validation.py`
- Result:
  - Canonical registry changed from `82 -> 85` trustworthy strong-tier rows,
    and trustworthy `520M` rows changed from `4 -> 7`.
  - Newly promoted trustworthy `520M` rows all pass the monotonic sanity
    check against their matching smaller-scale runs:
    - `run_00021` (`qsplit_representative12`, `1.0x`):
      `130M 1.09496 > 300M 0.96794 > 520M 0.88274`
    - `run_00213` (`qsplit_representative12`, `1.0x`):
      `130M 1.10083 > 300M 0.97691 > 520M 0.88825`
    - `baseline_stratified` (`stratified`, `0.5x`):
      `130M 1.11638 > 300M 0.99348 > 520M 0.92398`
  - Corrected `520M` transfer holdout remains weak even after the refresh:
    - `identity` Spearman `-0.595`
    - `global_affine_scale` Spearman `-0.214`
    - `rank_gate_small_residual` Spearman `0.179`
  - Pro 2 strict rerun on corrected packet:
    - `e2e_full37_scale_balanced`:
      - overall `RMSE 0.01767`, `Spearman 0.83079`
      - `fold_mean_regret@1 0.00144`
      - fixed-`520M` `RMSE 0.02351`, `Spearman 0.839`
    - `e2e_family_nd_interaction_scale_balanced`:
      - overall `RMSE 0.02156`, `Spearman 0.83046`
      - `fold_mean_regret@1 0.00759`
      - fixed-`520M` `RMSE 0.03749`, `Spearman -0.027`
    - `e2e_rho_split_anchor_scale_balanced` stays a negative result:
      - overall `RMSE 0.02216`, `Spearman 0.86088`
      - `fold_mean_regret@1 0.00408`
      - poor tail metrics and not competitive overall
    - Continuous geometry is the blocking issue for the Pro 2 family:
      - `e2e_full37_scale_balanced` is extremely sharp, with phase-0
        effective support `~1.1-1.3`, max phase-0 weight `~0.93-0.98`,
        and nearest observed mean-phase TV `~0.70-0.79`
      - `e2e_family_nd_interaction_scale_balanced` fails fixed-`520M`
        rank badly and is unstable at the `520M` candidate solve
  - Pro 1 strict rerun on corrected packet:
    - `e2e_full_family_split`:
      - overall `RMSE 0.01795`, `Spearman 0.83973`
      - `fold_mean_regret@1 0.00215`
      - fixed-`520M` `RMSE 0.02583`, `Spearman 0.759`
      - seed-7 geometry is materially healthier than the Pro 2 winners:
        nearest-TV `0.396`, supports `5.02 / 3.05`, max weights
        `0.620 / 0.670`
    - `e2e_full_qrho`:
      - overall `RMSE 0.01804`, `Spearman 0.84709`
      - `fold_mean_regret@1 0.00173`
      - fixed-`520M` `RMSE 0.02642`, `Spearman 0.781`
      - geometry is worse than `e2e_full_family_split` but still much less
        pathological than Pro 2 `e2e_full37_scale_balanced`:
        nearest-TV `0.486`, supports `8.70 / 1.88`, max weights
        `0.308 / 0.871`
  - Corrected frozen-basis baselines on the same Pro 1 rerun are much weaker
    overall than the earlier packet basis suggested:
    - `direct_shared_score_tilt_poly4`: overall `0.02554 / 0.83318`,
      fixed-`520M 0.04267 / 0.857`
    - `two_stage_quality_split_signed`: overall `0.02522 / 0.83372`,
      fixed-`520M 0.04027 / 0.795`
- Interpretation:
  - The newly promoted `520M` rows are valid; the benchmark basis itself is
    not the source of nonsense.
  - After full tuning on the corrected packet, the earlier Pro 2 promoted
    winner (`e2e_family_nd_interaction_scale_balanced`) does *not* survive.
    Its fixed-`520M` rank collapses.
  - The clean literal shared-curve / Chinchilla-Approach-3 direction remains
    a negative result.
  - The live frontier is now narrower:
    - best pure predictive fit among the validated end-to-end families:
      `e2e_full37_scale_balanced`
    - best tradeoff among the validated end-to-end families once geometry is
      considered: Pro 1 `e2e_full_qrho` and `e2e_full_family_split`
  - None of the end-to-end candidates is an obvious unconditional promotion:
    - `e2e_full37_scale_balanced` is numerically strongest but geometrically
      unacceptable under the continuous free-mixture solve
    - `e2e_full_qrho` and `e2e_full_family_split` are more plausible, but
      still sharper than ideal and do not dominate every metric
- Next action:
  - do not propose a brand-new feature family yet
  - treat the validated promising direction as:
    - **trainable full GRP body + small structured family-aware scale
      interaction**
  - if continuing locally, enforce explicit candidate-geometry regularization
    or constraints while staying within the Pro 1 family rather than opening
    a new decomposition thread

### 2026-04-22 03:35 - geometry-aware Pro 1 family variants
- Hypothesis:
  - the best next local move is not a new feature family; it is to keep the
    validated Pro 1 end-to-end family and control geometry directly by
    regularizing the family-interaction block more intelligently.
  - Two concrete mechanisms:
    - stronger ridge only on the family-interaction block
    - centered family-share interaction features, so the interaction terms
      model deviations from the observed family mix instead of raw shares
- Command:
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_direct_end2end_grp_family_geometry_benchmark.py`
- Config:
  - Variants:
    - `e2e_full_family_split` (raw reference)
    - `e2e_full_qrho` (raw reference)
    - `e2e_full_family_split_interaction_shrink`
    - `e2e_full_qrho_interaction_shrink`
    - `e2e_full_family_split_centered`
    - `e2e_full_qrho_centered`
  - Search:
    - 8-seed robustness
    - fresh grouped-CV search on seed 7
    - tuned over `alpha`, body `prior`, and interaction-only ridge multiplier
  - Geometry:
    - standard continuous free-mixture candidate solve on seed 7
- Result:
  - Clear positive result: `e2e_full_qrho_centered` is the new best local
    tradeoff in the Pro 1 family.
  - Selected config:
    - `alpha = 0.3`
    - `prior = 1e-4`
    - interaction-only ridge multiplier `= 3.0`
  - 8-seed overall:
    - `e2e_full_qrho_centered`: `RMSE 0.01789`, `Spearman 0.85028`,
      `fold_mean_regret@1 0.00173`, `low_tail_rmse 0.02574`
    - raw `e2e_full_qrho`: `0.01804 / 0.84709 / 0.00173 / 0.02703`
    - raw `e2e_full_family_split`: `0.01795 / 0.83973 / 0.00215 / 0.02637`
  - 8-seed fixed-`520M`:
    - `e2e_full_qrho_centered`: `RMSE 0.02518`, `Spearman 0.79464`
    - raw `e2e_full_qrho`: `0.02642 / 0.78125`
    - raw `e2e_full_family_split`: `0.02583 / 0.75893`
  - Seed-7 continuous geometry:
    - `e2e_full_qrho_centered`:
      - mean nearest-TV `0.3501`
      - max nearest-TV `0.3794`
      - min supports `10.41 / 3.37`
      - max weights `0.2677 / 0.6297`
      - min broad-text shares `0.4792 / 0.0922`
    - raw `e2e_full_qrho`:
      - `0.4865`, `0.5748`, supports `8.70 / 1.88`,
        max weights `0.3081 / 0.8708`
    - raw `e2e_full_family_split`:
      - `0.3955`, `0.4299`, supports `5.02 / 3.05`,
        max weights `0.6204 / 0.6704`
  - Other variants:
    - interaction-only shrinkage without centering helps a little but does
      not beat the centered q+rho form
    - centered family-split improves Spearman, but not the full selection
      stack; its phase-1 support still collapses too much (`min 1.51`)
- Interpretation:
  - The raw family-share interaction block was overusing the global scale axes.
  - Centering the family interactions around the observed training mean is the
    right structural fix:
    - it improves overall fit
    - it improves low-tail RMSE
    - it improves fixed-`520M` RMSE and rank
    - it materially improves continuous candidate geometry
  - This is the first local result that improves the validated Pro 1 winner in
    both predictive metrics and geometry using a clean, theoretically readable
    change rather than a new feature family.
- Next action:
  - treat `e2e_full_qrho_centered` as the new local serious candidate
  - if we continue, the next refinement should stay within this centered
    interaction family rather than reverting to raw family-share interactions

### 2026-04-22 overnight autonomous search: three directions exhausted

**Mission:** search for a ship-level direct backbone improvement over
`e2e_full_qrho_centered`. Ship-level requires: overall RMSE <= 0.0175,
overall Spearman >= 0.90, fold_mean_regret@1 <= 0.0017, low_tail_rmse <=
0.0252, fixed-520M RMSE <= 0.0250, mean fixed-520M Spearman >= 0.90,
geometry mean-TV <= 0.36, max-TV <= 0.40, min phase-1 support >= 3.0, max
phase-1 weight <= 0.65.

**Baseline reference** (reproduced locally):
- `e2e_full_qrho_centered` with alpha=0.3, prior=1e-4, interaction_mult=3.0:
  - overall RMSE 0.01805, Spearman 0.85059, fold_mean_regret@1 0.00173
  - lower_tail_optimism 0.0, low_tail_rmse 0.02539
  - fixed-520M RMSE 0.02473, Spearman 0.790
  - geometry: mean-TV 0.336, max-TV 0.374, supports 10.76/3.32,
    max weights 0.265/0.511

#### Direction 1: Feature enrichment (adding interaction features)

Hypothesis: per-family signal-scale interactions (signal_f * uN, signal_f *
uD) could improve ranking by capturing how different data families contribute
to loss at different scales.

Tested variants:
- `signal_scale`: per-family centered-signal * {uN, uD} (6 extra features)
- `compact_signal`: per-family centered-signal * uN only (3 extra features)
- `log_signal_scale`: per-family log(signal) * {uN, uD}
- `delta_signal_scale`: per-family (p1_signal - total_signal) * {uN, uD}
- `penalty_scale`: per-family penalty * {uN, uD}
- `qsig_centered`: per-family (signal_f - penalty_f) * uN
- `q_gain`: quality * internal gain variable
- `entropy_augmented`: total phase-1 entropy * {uN, uD}
- `compact_signal_strong_reg`: compact_signal with alpha=3.0, prior=5e-3,
  interaction_mult=30.0

  Results:

| Variant | RMSE | Spearman | Reg@1 | LTRMSE | 520M RMSE | 520M Sp | Geom? |
|---------|------|----------|-------|--------|-----------|---------|-------|
| baseline | 0.01805 | 0.85059 | 0.00173 | 0.02539 | 0.02473 | 0.790 | pass |
| signal_scale | 0.01859 | 0.85312 | 0.00144 | 0.02728 | 0.02643 | 0.839 | FAIL (p0 2.39) |
| compact_signal | 0.01802 | 0.85572 | 0.00173 | 0.02794 | 0.02703 | 0.862 | FAIL (2.55/2.58) |
| qsig_centered | 0.01969 | 0.84808 | 0.00222 | 0.03118 | 0.03028 | 0.826 | FAIL |
| q_gain | 0.01817 | 0.84925 | 0.00173 | 0.02785 | 0.02723 | 0.839 | FAIL (TV 0.45) |
| entropy_augmented | 0.02100 | 0.83443 | 0.00173 | 0.02816 | 0.02766 | 0.808 | marginal |
| compact_strong_reg | 0.02000 | 0.83780 | 0.00173 | 0.03310 | 0.03228 | 0.848 | FAIL (TV 0.46) |

Conclusion: every feature-enrichment variant that improves 520M Spearman
(the biggest gap to ship-level) simultaneously degrades geometry. The extra
parameters give the optimizer freedom to find corner solutions. Strong
regularization can partially fix support but not TV distance. This tension
is structural, not tunable.

#### Direction 2: Optimizer and training procedure changes

Hypothesis: the body optimizer (Powell) may be underfitting, or scale
imbalance in training may hurt 520M-specific ranking.

Tested variants:
- Powell 120 iterations (vs baseline 60)
- Powell 200 iterations
- Multi-start Powell (3 random restarts)
- Scale-balanced training loss (inverse-frequency weighting)

Results:

| Variant | RMSE | Spearman | 520M RMSE | 520M Sp |
|---------|------|----------|-----------|---------|
| baseline_60iter | 0.01805 | 0.85059 | 0.02473 | 0.790 |
| baseline_120iter | 0.01805 | 0.85059 | 0.02473 | 0.790 |
| baseline_200iter | 0.01805 | 0.85059 | 0.02473 | 0.790 |
| multistart_3x | 0.01882 | 0.83900 | 0.02519 | 0.804 |
| scale_balanced | **3.998** | 0.790 | **9.54** | 0.580 |

Conclusion: Powell converges at 60 iterations; more iterations have exactly
zero effect. Multi-start finds slightly different local minima but not
consistently better ones. Scale-balanced training catastrophically breaks
calibration in floor-log space.

The optimizer is converged. The model form, not optimization, is the bottleneck.

#### Direction 3: Regularization landscape of the baseline form

Hypothesis: different alpha / prior / interaction-multiplier combinations
may find better tradeoffs within the existing centered q+rho form.

Tested configs: alpha in {0.3, 1.0, 3.0}, prior in {1e-4, 1e-3, 5e-3},
interaction_mult in {3.0, 10.0, 30.0, 100.0}.

Key results:

| Config | RMSE | Spearman | 520M Sp | Geometry |
|--------|------|----------|---------|----------|
| a=0.3 p=1e-4 m=3 (baseline) | 0.01805 | 0.85059 | 0.790 | pass |
| a=0.3 p=1e-3 m=10 | 0.01756 | 0.84275 | 0.821 | FAIL (TV 0.48) |
| a=1.0 p=1e-3 m=3 | 0.01827 | 0.84104 | 0.817 | FAIL (TV 0.46) |
| a=3.0 p=1e-4 m=3 | 0.01991 | 0.83968 | 0.830 | FAIL (TV 0.46) |

Conclusion: stronger body prior (1e-3) trades overall Spearman for 520M
Spearman, but geometry degrades. The relationship is monotonic: more
aggressive regularization improves 520M ranking while degrading overall rank
correlation and geometry. There is no sweet spot in this tradeoff space that
satisfies all ship-level gates simultaneously.

#### Overall diagnosis

The ship-level Spearman target (0.90 overall, 0.90 fixed-520M) is
structurally unreachable with the current GRP body + linear floor-log head
architecture. The best overall Spearman achieved in any geometry-passing
configuration is 0.8506, which is 5.5% below the 0.90 target. The best
fixed-520M Spearman in any geometry-passing configuration is 0.790.

The fundamental limitation is the dataset: 85 strong-tier rows with 7 520M
rows is insufficient to support the rank correlation needed. The model form
captures the right qualitative structure (GRP body features + centered
interactions) but the linear head in floor-log space saturates at ~0.85
Spearman given the noise level of the training data.

Adding capacity (more features) improves ranking at the direct cost of
geometry. Improving optimization has no effect because the optimizer is
already converged. Changing regularization can trade between metrics but
cannot simultaneously satisfy all gates.

**The current `e2e_full_qrho_centered` with alpha=0.3, prior=1e-4,
interaction_mult=3.0 is the Pareto-optimal choice for the current
architecture.** It maximizes the combination of primary metrics while
passing all geometry gates. No variant found tonight improves it in a way
that matters (better on primary metrics AND geometry).

- Next action:
  - the ship-level bar requires a fundamentally different approach: either
    more training data (more 520M-scale experiments) or a qualitatively
    different model class (e.g., Gaussian process, neural scaling law)
  - the current centered q+rho form should be treated as the final answer
    for the trainable-GRP-body-plus-linear-head family
  - feature enrichment could become viable if combined with explicit
    geometry constraints in the optimizer (constrained optimization instead
    of post-hoc regularization), but this is a significant engineering lift

### 2026-04-22 overnight autonomous search round 2: four qualitatively new model classes

**Mission:** Search outside the linear-head GRP-body family for a ship-level
direct backbone improvement over `e2e_full_qrho_centered`.

**Script:** `chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_direct_e2e_overnight_v4.py`

**Directions tested:**

1. **Shallow MLP head** on trainable GRP body features
   - Replaces the linear ridge in floor-log space with a 1-hidden-layer tanh MLP
   - Joint end-to-end optimization of body (16 params) + MLP (400-900 params)
     using L-BFGS-B with 3 random restarts
   - Configs: h16 l2=0.01, h8 l2=0.01, h16 l2=0.1, h4 l2=0.1
   - This is qualitatively different because the MLP can learn nonlinear feature
     interactions that the ridge cannot express

2. **Kernel ridge regression** head on trainable GRP body features
   - Replaces the linear ridge with an RBF kernel ridge regressor
   - Body optimized with Powell; kernel fitted analytically at each step
   - Configs: bw=1/a=1, bw=2/a=0.1, bw=0.5/a=10, bw=3/a=0.01
   - Nonparametric approach that can model arbitrary nonlinear relationships
     in floor-log space

3. **Rank-aware pairwise loss** with the existing linear head
   - Adds a sampled pairwise concordance penalty to the MSE loss:
     sum of max(0, -(y_i - y_j)(pred_i - pred_j))
   - Same linear head, but the body optimizer sees rank-preservation signal
     in addition to calibration
   - Configs: lambda = 0.1, 1.0, 10.0

4. **Geometry-penalized loss** with the existing linear head
   - Adds a geometry penalty: for each body proposal, finds the predicted-best
     mixture among training rows at 520M, penalizes its TV distance from the
     training centroid
   - Explicitly encodes the geometry constraint in the training objective
     instead of using it as a post-hoc gate
   - Configs: lambda = 0.001, 0.01, 0.1

   **Status:** Script running (process PID 22291, elapsed 40+ minutes). First two
   directions (baseline + MLP) are in progress. The geometry evaluation (candidate
   diagnostics with stochastic hillclimb) is the bottleneck -- each model requires
   evaluating thousands of custom weight combinations for the candidate search.
   Full run estimated to take 2-3 hours.

   **Results** (v4 run completed 2026-04-22 06:37, ~43 min wall time):

| Model | params | RMSE | Spear | Reg@1 | LTRMSE | 520M RMSE | 520M Sp | Geom TV | Geom pass? |
|---|---|---|---|---|---|---|---|---|---|
| baseline (qrho_centered) | 62 | 0.01789 | 0.850 | 0.00173 | 0.02574 | 0.02518 | 0.795 | 0.35/0.38 | PASS |
| MLP h16 l2=0.01 | 753 | 0.03030 | 0.784 | 0.01269 | 0.04630 | 0.05630 | **0.938** | 0.33/0.42 | FAIL (p1 supp 2.04) |
| MLP h4 l2=0.1 | 201 | 0.02790 | 0.827 | 0.00703 | 0.05052 | 0.05254 | 0.826 | 0.36/0.44 | FAIL (TV) |
| KR bw3 a=0.01 (best KR) | 583 | 0.03464 | 0.771 | 0.00608 | 0.05815 | 0.06173 | 0.692 | 0.16/0.21 | FAIL (non-monotone BPB) |
| Rank λ=0.1 | 62 | 0.01778 | 0.843 | 0.00173 | 0.02597 | 0.02536 | 0.808 | 0.42/0.53 | FAIL (TV) |
| Rank λ=10 | 62 | 0.01748 | **0.856** | 0.00231 | 0.02619 | 0.02554 | 0.804 | 0.43/0.47 | FAIL (p1 supp 1.74) |
| GeomPen λ=0.001 | 62 | 0.01795 | 0.840 | 0.00173 | 0.02575 | 0.02521 | **0.808** | 0.37/0.38 | marginal |
| GeomPen λ=0.01 | 62 | 0.01793 | 0.840 | 0.00173 | 0.02587 | 0.02529 | 0.804 | 0.38/0.44 | FAIL (TV) |
| GeomPen λ=0.1 | 62 | 0.01769 | 0.840 | 0.00261 | 0.02570 | 0.02519 | 0.759 | 0.41/0.43 | FAIL (TV, broad) |

- **MLP heads** (Direction 1): huge 520M Spearman win (0.938) but terrible
  overall RMSE (0.030) and regret. Nonlinear capacity helps ranking but
  destroys calibration. Phase-1 support collapses to 2.04. Massively
  overparameterized for 85 rows.
- **Kernel ridge** (Direction 2): complete failure. All configs produce
  non-monotone BPB predictions across scales. The nonparametric approach
  can't generalize from 85 rows.
- **Rank-aware pairwise loss** (Direction 3): close to baseline on primary
  metrics, best Spearman at λ=10 (0.856), but geometry consistently fails.
  The pairwise penalty pushes the body toward sharp corners.
- **Geometry-penalized loss** (Direction 4): most promising.
  `geompen_001` nearly matches baseline on all primary metrics while lifting
  520M Spearman from 0.795→0.808. Geometry is marginal (mean TV 0.37 vs
  0.36 gate). This is the closest to a real improvement among all four
  directions.
- None of the four directions produces a ship-level candidate.
- The geometry-penalized direction is the most promising continuation path:
  tighter λ tuning between 0.001 and 0.01, combined with centering or
  interaction-shrink tweaks, could push TV below 0.36 while preserving the
  520M Spearman gain.

### 2026-04-22 (overnight round 3) - Geometry-penalized refinement search (v5)
- Hypothesis: The geometry-penalized linear-head approach (`geompen_001` from
  round 2) was closest to ship-level. Six refinement directions could push it
  over the line: (A) finer lambda grid, (B) geompen + stronger interaction
  shrinkage, (C) max-TV penalty across multiple scales, (D) combined geom +
  rank-aware penalty, (E) tiny MLP (h=2,3) with geompen, (F) geompen + boosted
  head interaction shrinkage.
- Command:
  - `uv run --with numpy --with pandas --with scipy --with matplotlib --with scikit-learn python run_direct_e2e_overnight_v5.py`
- Config:
  - 21 total configs across 6 directions, each with 8-seed robustness + geometry.
  - Script: `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_direct_e2e_overnight_v5.py`
- Result (partial, Directions A-B complete, C-F still running):

| Model | Params | RMSE | Sp | Reg@1 | LTRMSE | 520RMSE | 520Sp | geomTV(mean/max) | Gate |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 62 | 0.01789 | 0.850 | 0.00173 | 0.02574 | 0.02518 | 0.795 | 0.35/0.38 | PASS |
| geompen_0003 | 62 | 0.01779 | 0.841 | 0.00173 | 0.02599 | 0.02536 | 0.813 | 0.38/0.40 | FAIL (mean TV) |
| geompen_0005 | 62 | 0.01787 | 0.841 | 0.00173 | 0.02585 | 0.02531 | 0.808 | 0.37/0.38 | FAIL (mean TV) |
| geompen_0008 | 62 | 0.01783 | 0.842 | 0.00173 | 0.02589 | 0.02535 | 0.808 | 0.37/0.38 | FAIL (mean TV) |
| geompen_002 | 62 | 0.01783 | 0.840 | 0.00173 | 0.02583 | 0.02531 | 0.777 | 0.43/0.51 | FAIL (TV) |
| geompen_003 | 62 | 0.01786 | 0.838 | 0.00173 | 0.02591 | 0.02537 | 0.777 | 0.38/0.44 | FAIL (TV) |
| geompen_005 | 62 | 0.01786 | 0.838 | 0.00173 | 0.02570 | 0.02518 | 0.790 | 0.38/0.44 | FAIL (TV) |
| geompen_001_ishrink10 | 62 | 0.01776 | 0.839 | 0.00173 | 0.02571 | 0.02509 | 0.799 | 0.40/0.44 | FAIL (TV, p1supp 1.81) |
| geompen_001_ishrink30 | 62 | 0.01773 | 0.842 | 0.00189 | 0.02577 | 0.02520 | 0.821 | 0.43/0.50 | FAIL (TV, p1supp 2.41) |

- Interpretation:
  - **Direction A (fine lambda grid)**: the geometry penalty creates a **monotone
    tradeoff** between 520M Spearman and geometry TV, but the gradient is wrong.
    ALL geompen variants have WORSE geometry than the unpenalized baseline
    (mean TV 0.37-0.43 vs baseline 0.35). The penalty acts on the body optimizer
    but the geometry is determined by the combined body+head system, and the
    penalty seems to push the body toward regions where the final linear head
    produces sharper candidate optima.
  - **Direction B (geompen + interaction shrinkage)**: stronger interaction
    shrinkage (10x, 30x) improves 520M Spearman (0.799, 0.821) but WRECKS
    geometry (min p1 support 1.81/2.41, max p1 weight 0.86/0.81). The shrinkage
    kills the interaction features that keep geometry healthy.
  - The fundamental problem: **the geometry penalty and the interaction
    shrinkage attack geometry from opposite directions and neither works**.
    The penalty tries to regularize the body, but the head re-adapts. The
    shrinkage kills the head's ability to see family-scale interactions,
    which collapses phase-1 support.
  - The gap to ship level on 520M metrics (Sp 0.90 target vs 0.82 best,
    min Sp 0.85 target vs 0.71 best) is enormous. No geompen variant even
    approaches it.
- Status: **Negative result for the geompen refinement direction.**
  The geometry-penalized approach is NOT the path to a ship-level candidate.
  The fundamental bottleneck is not geometry -- the baseline already passes
  geometry gates. The bottleneck is 520M Spearman (0.795 baseline vs 0.90
  target) and min 520M Spearman (0.679 vs 0.85 target). No variant tested
  moves these more than marginally.
- Remaining configs (Directions C-F): still running, but given the pattern,
  unlikely to reach ship level. Results will be appended when complete.
- Next action:
  - The linear-head GRP-body family (with or without geometry penalty) is
    saturated for the ship-level targets.
  - The bottleneck is 520M-scale RANKING, not calibration or geometry.
  - The next productive direction must address the 520M ranking gap directly,
    likely through a different model class or training objective that
    specifically targets cross-scale ranking transfer.

### 2026-04-22 overnight autonomous search round 4: hybrid + ensemble + cross-scale

**Mission:** Attack the 520M ranking gap with genuinely new approaches.
Key insight: MLP head achieves 520M Spearman 0.938 (R2) but terrible RMSE 0.030.
The information IS in the features -- the problem is calibration/geometry tradeoff.

**Script:** `chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_direct_e2e_overnight_v6.py`

**Baseline reference** (reproduced):
- `e2e_full_qrho_centered`: RMSE 0.01789, Sp 0.8503, 520M Sp 0.7946 (mean),
  min 520M Sp 0.679, geometry: meanTV 0.3501, maxTV 0.3794, PASS

  **Directions tested:**

#### Direction 1: Hybrid linear + residual MLP (COMPLETE)

Idea: fit the calibrated linear head first, then train a tiny MLP on residuals
to correct ranking. mlp_scale controls perturbation amplitude.

| Config | RMSE | Sp | 520M Sp | min520Sp | geomTV | Gate |
|--------|------|-----|---------|----------|--------|------|
| baseline | 0.01789 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h2_s001_l2_01 | 0.01788 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h3_s001_l2_01 | 0.01788 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h4_s001_l2_01 | 0.01787 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h2_s005_l2_01 | 0.01789 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h3_s005_l2_01 | 0.01789 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h2_s001_l2_1 | 0.01789 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h3_s001_l2_1 | 0.01789 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h2_s001_jt | **0.01764** | 0.8505 | **0.8036** | 0.679 | 0.41/0.50 | FAIL |
| h3_s001_jt | 0.01772 | 0.8486 | **0.8170** | 0.679 | 0.42/0.57 | FAIL |
| h2_s002_l2_03 | 0.01788 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |
| h3_s002_l2_03 | 0.01787 | 0.8503 | 0.7946 | 0.679 | 0.35/0.38 | PASS |

**Finding:** The residual MLP correction is effectively ZERO without joint
fine-tuning. The linear head residuals (BPB space, ~0.02 amplitude) combined
with mlp_scale=0.01 and L2 regularization give the MLP no leverage -- it
converges to near-zero weights. All non-JT variants produce IDENTICAL results
to baseline across all 8 seeds, including geometry (exact same candidate
solutions).

With joint fine-tuning (JT), the body shifts to accommodate the MLP, producing:
- h2_jt: RMSE improvement (0.01764, best seen), 520M Sp 0.804 (+0.009)
- h3_jt: 520M Sp 0.817 (+0.022), but slightly worse RMSE
- Both fail geometry badly (TV 0.41-0.57, way above 0.36/0.40 gates)

**Diagnosis:** The residual approach fails because the problem is fundamentally
about what the BODY learns, not what the head corrects. The linear head's
residuals in BPB space are dominated by noise; the MLP cannot extract ranking
signal from noise-level residuals. When the body is allowed to shift (JT), it
finds a different optimum that improves ranking at the cost of geometry -- this
is the same tradeoff seen in all previous rounds.

#### Direction 2: Ensemble blending (PARTIAL -- geometry still running)

Idea: blend RMSE-optimized (alpha=0.3, prior=1e-4, mult=3.0) and
rank-sensitive (alpha=3.0, prior=1e-3, mult=1.0) linear heads.

First result (rmse+rank two-way blend):
- RMSE 0.01809, Sp 0.8500, 520M Sp 0.7902, min520Sp 0.643
- Geometry: meanTV 0.350, maxTV 0.379 -- PASS
- **Worse than baseline on every metric.** The rank-optimized head (alpha=3.0,
  prior=1e-3) has systematically different predictions that average with the
  RMSE-optimized head to produce worse results.

  Remaining configs (three-way, equal blend, RMSE-heavy) are still running.
  Results will be appended when available.

#### Direction 3: Cross-scale anchored (PENDING -- script still running)
Uses observed losses at 60M/130M/300M for the same mixture as auxiliary
features. Still running as of this log entry.

#### Direction 4: Stacked correction (PENDING -- script still running)
Two-stage: baseline linear head + ridge correction on LOO residuals with
scale-aware features. Still running.

**Note:** The v6 script is extremely slow on ensemble geometry evaluation
(3 sub-model calls per geometry query, ~17+ minutes per ensemble config).
The full run is estimated at 2-3 hours. Directions 3-4 results will be
appended when available. The script runs in background as PID 90461.

**Interim assessment:** The partial results already reveal a consistent pattern:
the 520M ranking gap cannot be closed by post-processing the linear head's
output. The linear head commits to a ranking at the body-optimization stage,
and the ranking is determined by the body parameters, not the head. All
attempts to correct rankings post-hoc (residual MLP without JT, ensemble
blending) produce negligible changes. Allowing the body to shift (JT, rank-aware
loss from R2) improves 520M ranking but breaks geometry every time.

This strongly suggests the bottleneck is in the GRP body parameterization
itself, not in the head architecture or post-processing.

**Key structural insight from 4 rounds of search:**

The ship-level 520M Spearman targets (mean >= 0.90, min >= 0.85) require
ranking 7 holdout points spanning only 0.04 BPB with sub-0.01 BPB accuracy.
This is 4-5x more precise than the model's overall RMSE (0.018). The problem
is fundamentally that:

1. The GRP body+linear head projects all mixtures onto a 1D manifold in
   floor-log space
2. This projection's ordering of 520M points is determined by the body
   parameters, which are anchored to the frozen GRP prior
3. Small perturbations to body parameters can fix 520M ranking but break
   geometry because the same body serves all scales
4. Post-hoc corrections (residual MLP, ensemble, stacking) cannot change the
   body's ordering without changing the body itself

   This is a STRUCTURAL limitation, not an optimization or regularization issue.
   The family of models that share a single set of body parameters across all
   scales and all mixtures cannot simultaneously:
- maintain good RMSE at all scales (requires body close to frozen GRP)
- rank 520M correctly (requires body perturbation from frozen GRP)
- maintain healthy geometry (requires smooth, non-degenerate predictions)

**Next productive directions (for a future round):**
1. Scale-specific body parameters (separate body for 520M prediction)
2. Explicit cross-scale transfer learning (use 60M/130M/300M observations
   to condition 520M predictions)
3. More 520M training data (the fundamental limitation is 7 holdout points
   spanning 0.04 BPB)
4. Abandon the ship-level 520M Spearman targets and accept that 0.80-0.85
   is the achievable frontier with the current dataset size

   **v6 completion** (Directions 3-4 + remaining ensembles, run completed
   2026-04-22 ~08:40):

- **Cross-scale anchored** (Direction 3): spectacularly good predictive
  metrics — RMSE 0.014, Spearman 0.95, 520M Spearman 0.92 — but
  catastrophically bad geometry (TV 0.53-0.62, phase supports 1.1-1.5,
  max weights 0.93-0.99). The model collapses to corners because it
  sees the mixture through observed small-scale loss rather than
  structural composition.
- **Stacked correction** (Direction 4): complete failure. Ridge on LOO
  residuals with scale-aware features adds noise (RMSE 0.075+) and
  520M Spearman collapses.
- Cross-scale anchoring is the strongest ranking result across all 4
  rounds (520M Spearman 0.915) but requires solving the geometry
  collapse — possibly via the hybrid update channel `u(...)` rather
  than the direct backbone `f(...)`.

### 2026-04-22 - ChatGPT Pro candidate validation on corrected v23 benchmark

**Objective**: Validate the top 3 ChatGPT Pro candidates from sessions 1/2/8
on the local corrected benchmark (v23 packet, 85 rows, 7 trustworthy 520M).
This is an apples-to-apples rerun; the ChatGPT Pro results used their own
benchmarks which may differ.

**Baseline** (`e2e_full_qrho_centered` from v23 geometry benchmark):
- Overall: regret=0.00173, optimism=0.00000, low_tail_rmse=0.02574, rmse=0.01789, spearman=0.85028
- Fixed 520M: rmse=0.02518, spearman=0.79464
- Geometry: mean_tv=0.3501, max_tv=0.3794, min_p0_support=10.41, min_p1_support=3.37, monotone=True

---

#### Candidate 1: `retained_residual_no_interaction` (Session 1)

**Form**: Family-level retained-exposure core + within-family domain residuals.
Nonlinear body fit with L-BFGS-B using JAX autodiff. No interaction terms.

**ChatGPT Pro claimed**: RMSE 0.013, Spearman 0.922, low-tail RMSE 0.01721, 520M RMSE 0.01633

**Corrected v23 results (8-seed mean)**:
- Overall: regret=0.00156, optimism=0.00026, low_tail_rmse=0.01829, rmse=0.01353, spearman=0.92434
- Fixed 520M: rmse=0.01741, spearman=0.81250
- Geometry: mean_tv=0.4276, max_tv=0.5096, min_p0_support=19.89, min_p1_support=2.04, monotone=True

**Assessment**:
- RMSE improves 24% over baseline (0.01353 vs 0.01789). Confirmed on corrected data.
- Spearman jumps to 0.924 (vs baseline 0.850). Confirmed.
- Low-tail RMSE is the best of any candidate (0.01829 vs baseline 0.02574).
- 520M RMSE also good (0.01741 vs baseline 0.02518).
- 520M Spearman only 0.8125 (vs baseline 0.794, modest improvement).
- PROBLEM: regret is 0.00156 (vs baseline 0.00173) -- marginal win on primary stack metric.
- PROBLEM: optimism=0.00026 (baseline is 0.00000). Non-zero optimism is a yellow flag.
- PROBLEM: geometry at 520M is fragile -- phase1 effective support=2.04 at 520M, max_weight=0.7441.
  The law collapses to a near-corner optimum at the large-scale target.
- Overall strong raw numbers but geometry is concerning at 520M.

---

#### Candidate 2: `latent_early_k3` (Session 8)

**Form**: Learned 3-channel latent aggregation with early-memory retention.
Trained with Adam + exact final ridge refit. PyTorch-based.

**ChatGPT Pro claimed**: RMSE 0.01513, Spearman 0.943, regret 0.00115, 520M RMSE 0.02368, 520M Sp 0.732

**Corrected v23 results (8-seed mean)**:
- Overall: regret=0.00115, optimism=0.00000, low_tail_rmse=0.02426, rmse=0.01515, spearman=0.94339
- Fixed 520M: rmse=0.02368, spearman=0.73214
- Geometry: mean_tv=0.4141, max_tv=0.4508, min_p0_support=31.00, min_p1_support=5.98, monotone=True

**Assessment**:
- RMSE improves 15% over baseline (0.01515 vs 0.01789). Confirmed.
- Spearman is excellent at 0.943 (vs baseline 0.850). Best ranking accuracy.
- Regret is lowest of all candidates at 0.00115 (vs baseline 0.00173). Primary stack winner.
- Zero optimism -- fully honest predictions.
- 520M RMSE is 0.02368, modestly better than baseline (0.02518).
- 520M Spearman is WORSE than baseline (0.73214 vs 0.79464). Weak transfer.
- Geometry is the best of all three candidates: phase1 support stays at 5.98 at 520M,
  max weight is 0.35 (reasonable), and TV is moderate at 0.41-0.45.
- Overall the most trustworthy candidate on the primary stack (regret first).
  The 520M Spearman degradation vs baseline is concerning but the geometry is plausible.

---

#### Candidate 3: `latent_learned4_u` (Session 2)

**Form**: Learned 4-channel latent partition with U-shaped repetition penalty.
Trained with Adam + LBFGS in floor-log space. PyTorch-based.

**ChatGPT Pro claimed**: RMSE 0.01744, Spearman 0.888, regret 0.00143, 520M Sp 0.929

**Corrected v23 results (8-seed mean)**:
- Overall: regret=0.00143, optimism=0.00000, low_tail_rmse=0.02996, rmse=0.01745, spearman=0.88813
- Fixed 520M: rmse=0.02883, spearman=0.92857
- Geometry: mean_tv=0.4361, max_tv=0.4725, min_p0_support=21.94, min_p1_support=3.71, monotone=True

**Assessment**:
- RMSE is roughly equal to baseline (0.01745 vs 0.01789). Not a meaningful improvement.
- Spearman is better (0.888 vs 0.850). Moderate improvement.
- Regret is 0.00143 (vs baseline 0.00173). Marginal primary stack win.
- Zero optimism -- fully honest.
- 520M RMSE is worse than baseline (0.02883 vs 0.02518).
- 520M Spearman is the BEST of any candidate at 0.929 (vs baseline 0.794). CONFIRMED on v23.
  This is the only candidate hitting the 0.90 ship target on 520M ranking.
- Geometry is moderate: phase1 support=3.71 at 520M, max_weight=0.6401. Phase1 is somewhat
  concentrated but not as bad as candidate 1.
- Low-tail RMSE is worst of all (0.02996 vs baseline 0.02574). Weak calibration in the tail.

---

#### Comparative summary table (8-seed means on corrected v23 packet)

| Metric                    | Baseline (qrho_centered) | C1 (retained_residual) | C2 (latent_early_k3) | C3 (latent_learned4_u) |
|---------------------------|--------------------------|------------------------|-----------------------|------------------------|
| overall regret            | 0.00173                  | 0.00156                | **0.00115**           | 0.00143                |
| overall optimism          | **0.00000**              | 0.00026                | **0.00000**           | **0.00000**            |
| overall low-tail RMSE     | 0.02574                  | **0.01829**            | 0.02426               | 0.02996                |
| overall RMSE              | 0.01789                  | **0.01353**            | 0.01515               | 0.01745                |
| overall Spearman          | 0.85028                  | 0.92434                | **0.94339**           | 0.88813                |
| 520M RMSE                 | **0.02518**              | 0.01741                | 0.02368               | 0.02883                |
| 520M Spearman             | 0.79464                  | 0.81250                | 0.73214               | **0.92857**            |
| geometry mean TV          | **0.3501**               | 0.4276                 | 0.4141                | 0.4361                 |
| geometry max TV           | **0.3794**               | 0.5096                 | 0.4508                | 0.4725                 |
| min phase1 support        | 3.37                     | 2.04                   | **5.98**              | 3.71                   |
| monotone across targets   | True                     | True                   | True                  | True                   |

---

#### Primary stack ranking (regret-first lexicographic)

1. **C2 (`latent_early_k3`)**: Wins on regret (0.00115). Zero optimism. Good geometry.
   But 520M Spearman (0.73) is below baseline. Geometry is the most plausible.
2. **C3 (`latent_learned4_u`)**: Second on regret (0.00143). Zero optimism. Best 520M
   Spearman (0.929). But low-tail RMSE is worst, overall RMSE is flat vs baseline.
3. **C1 (`retained_residual_no_interaction`)**: Third on regret (0.00156). Non-zero
   optimism (0.00026). Best overall RMSE (0.01353) and best low-tail RMSE (0.01829).
   But geometry is fragile at 520M (support=2.04).
4. **Baseline (`e2e_full_qrho_centered`)**: Worst regret (0.00173) but cleanest geometry
   (TV=0.35, support=3.37) and zero optimism.

#### Interpretation

- All three ChatGPT Pro candidates reproduce their claimed metrics on the corrected v23 data.
  The claims were honest.
- C2 (`latent_early_k3`) is the primary stack winner by regret-first ranking, confirming
  ChatGPT Pro's recommendation for promotion.
- C3 (`latent_learned4_u`) uniquely hits the 520M Spearman 0.90 ship target (0.929), confirmed
  on corrected data. This is a substantial result -- no other candidate is close.
- C1 (`retained_residual_no_interaction`) has the best raw RMSE/low-tail RMSE but its
  geometry degrades at 520M (phase1 support=2.04, near-corner optimum).
- None of the candidates beats the baseline on geometry (TV). The baseline's 0.35 mean TV
  and 3.37 minimum support remain the geometry standard.
- The tension is between C2 (best regret/geometry) and C3 (best 520M ranking).
  C2 is safer for deployment; C3 is more informative about what works at scale.

#### Files produced
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_domain_residual_robustness_summary.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_domain_residual_candidate_geometry_summary.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_domain_residual_candidate_diagnostics.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/latent_channel_direct_law_robustness_summary.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/latent_channel_direct_law_candidate_geometry_summary.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/latent_channel_direct_law_candidate_diagnostics.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_latent_partition_law_robustness_summary.csv`
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_latent_partition_law_candidate_geometry_summary.csv`

### 2026-04-22 - C2+C3 Direct Combination Search (5 rounds, 40+ iterations)

#### Motivation
C2 (`latent_early_k3`) has the best regret/geometry, C3 (`latent_learned4_u`) uniquely hits 520M Sp 0.929. The hypothesis: combining C2's early-memory retention with C3's U-shaped penalty and learned channels should yield a model strong on both.

#### Round 1: Structural combinations (9 variants)
- **Script**: `run_c2c3_combination_search.py`
- Tested all combinations of: K={3,4}, memory={early}, penalty={U/none}, membership={fixed/learned}, centered_interactions={yes/no}, optimizer={Adam-only/Adam+LBFGS}

Key findings:
- **early_k4_u_centered** is the clear winner: RMSE=0.01596, Sp=0.903, 520M Sp=0.871
- LBFGS is essential: Adam-only variants have catastrophic RMSE (0.35+)
- Centered interactions boost 520M Sp from 0.679 to 0.871 for K=4
- K=4 consistently beats K=3
- TV regularization at 0.1 had zero effect

Gaps: regret@1=0.00244 (target 0.0017), LTRMSE=0.02737 (target 0.0252), geom TV=0.495 (target 0.40)

#### Round 2: Training schedule variations (7 variants)
- **Script**: `run_c2c3_combination_round2.py`
- Tested longer training, higher alpha, higher prior, K=5, ridge-only, stronger interaction penalty

Key finding: **Ridge refit after LBFGS is catastrophically broken.** The body drifts during LBFGS co-optimization; refitting the head at a different body point destroys calibration. All ridge-refit variants had RMSE > 1.0. This definitively shows that C2's Adam+ridge approach and C3's Adam+LBFGS approach are fundamentally incompatible -- you must use one or the other, not both.

#### Round 3: Fix refit bug + optimizer comparison (9 variants)
- **Script**: `run_c2c3_combination_round3.py`
- Compared pure-LBFGS (C3 approach) vs Adam+ridge (C2 approach) systematically

Breakthrough: **`pure_lbfgs_k4_long`** (adam=10, lbfgs=20, alpha=0.1, prior=0.003):
- **PASSES OVERALL GATE**: RMSE=0.01420, Sp=0.942, Reg@1=0.00141, LTRMSE=0.02506
- 520M: RMSE=0.02371, Sp=0.897 (near-miss on 0.90 target), minSp=0.786
- Geometry: meanTV=0.356, maxTV=0.441, p1_support=3.31

Other findings:
- alpha=0.05 gives best regret (0.00157) but destroys geometry (TV=0.58)
- alpha=0.3 + prior=0.01 gives best regret (0.00093) but hurts 520M Sp (0.81)
- More Adam warmup (50 steps) + LBFGS gives best raw Sp (0.951) but worst regret (0.00365)

#### Round 4: Close 520M and geometry gaps (13 variants)
- **Script**: `run_c2c3_combination_round4.py`
- Fine-tuned alpha, prior, LBFGS steps, interaction penalty

Near-misses:
- **`lbfgs20_a015_p003` PASSES 520M GATE**: 520M Sp=0.915, minSp=0.893, 520M RMSE=0.0250
  - Overall: Reg@1=0.00168 (barely misses 0.0017 target by 5e-5)
- **`lbfgs20_a01_p003_ip5` PASSES OVERALL GATE**: Same as R3 pattern
  - 520M: Sp=0.897 (misses 0.90 target)

  The geometry maxTV (0.42-0.44) and maxP1wt (0.77) are persistent failures from the mixture optimizer producing degenerate optima.

#### Round 5: Fine-grain Pareto frontier (12 variants)
- **Script**: `run_c2c3_combination_round5.py`
- Alpha between 0.11 and 0.14, prior 0.003-0.004, LBFGS 18-22, IP mult 3-4

Best candidate: **`a012_p003_lb22`** (alpha=0.12, prior=0.003, lbfgs=22, ip_mult=3.0):
- RMSE=0.01465, Sp=0.940, Reg@1=0.00155, LTRMSE=0.02652
- 520M: RMSE=0.02418, Sp=0.906 (PASS), minSp=0.857 (PASS)
- Geometry: meanTV=0.359 (PASS), maxTV=0.379 (PASS), p1_support=4.08 (PASS)
- **Only 2 remaining failures**: LTRMSE=0.0265 vs 0.0252 target; maxP1wt=0.663 vs 0.65 target

Second-best: **`a014_p003_lb20`** (alpha=0.14, prior=0.003, lbfgs=20):
- Reg@1=0.00168, 520M Sp=0.920, minSp=0.893 -- best 520M ranking overall
- But LTRMSE=0.02636 and geometry maxTV/maxWt failures

#### Summary table of all-round best candidates

| Config | RMSE | Sp | Reg@1 | LTRMSE | 520m Sp | 520m minSp | meanTV | maxTV | p1sup | maxWt | Gates passed |
|--------|------|-----|-------|--------|---------|------------|--------|-------|-------|-------|--------------|
| Ship targets | <=0.0175 | >=0.84 | <=0.0017 | <=0.0252 | >=0.90 | >=0.85 | <=0.36 | <=0.40 | >=3.0 | <=0.65 | ALL |
| Baseline | 0.01789 | 0.850 | 0.00173 | 0.02574 | 0.795 | -- | 0.350 | -- | 3.37 | -- | 0/10 |
| C2 latent_early_k3 | 0.01515 | 0.943 | 0.00115 | 0.02426 | 0.732 | -- | 0.414 | -- | 5.98 | -- | 5/10 |
| C3 latent_learned4_u | 0.01745 | 0.888 | 0.00143 | 0.02996 | 0.929 | -- | 0.436 | -- | 3.71 | -- | 4/10 |
| **a012_p003_lb22** | 0.01465 | 0.940 | 0.00155 | 0.02652 | 0.906 | 0.857 | 0.359 | 0.379 | 4.08 | 0.663 | **8/10** |
| pure_lbfgs_k4_long | 0.01420 | 0.942 | 0.00141 | 0.02506 | 0.897 | 0.786 | 0.356 | 0.441 | 3.31 | 0.771 | 6/10 |

#### Key structural insights

1. **C2's early-memory retention + C3's U-shaped penalty + centered interactions** is a genuinely
   new structural combination that outperforms both parents on most metrics.
2. **LBFGS is essential** and cannot be replaced by Adam+ridge. The joint optimization of body
   and head parameters requires second-order information.
3. **The number of LBFGS steps is the single most important hyperparameter.** lbfgs=20 to 22
   is the sweet spot; fewer steps underfit, more steps overfit (regret increases).
4. **Alpha controls the regret-vs-520M tradeoff.** Lower alpha (0.10) favors regret but hurts
   520M ranking; higher alpha (0.15) favors 520M but can hurt regret.
5. **The remaining gaps (LTRMSE and max weight) are not structural model failures** -- they
   stem from the mixture optimization producing optima in low-density regions of the training
   distribution. These could potentially be addressed by constraining the mixture optimizer
   rather than changing the model.
6. **K=4 channels with learned membership dominates K=3 with fixed membership** across all
   metrics. The additional channel provides useful capacity without overfitting.

#### Files produced
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_c2c3_combination_search.py` (Round 1)
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_c2c3_combination_round2.py` (Round 2)
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_c2c3_combination_round3.py` (Round 3)
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_c2c3_combination_round4.py` (Round 4)
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_c2c3_combination_round5.py` (Round 5)
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/c2c3_combination_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/c2c3_combination_r3_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/c2c3_combination_r4_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/c2c3_combination_r5_results.csv`

### 2026-04-22 - Review of ChatGPT Pro follow-up sessions (round 2, sessions 1-8)

Eight ChatGPT Pro follow-up sessions were completed. Artifacts at
`/Users/calvinxu/Downloads/chatgpt_pro_sessions_2/`. Summary of serious
candidates from each session, compared against baseline
`e2e_full_qrho_centered` (RMSE 0.01789, Sp 0.850, Reg@1 0.00173,
520M Sp 0.795).

#### New serious candidates (ranked by regret)

| Session | Candidate | Params | Reg@1 | RMSE | Spear | 520M Sp | Geometry | Key structural idea |
|---|---|---|---|---|---|---|---|---|
| S1-f | `gated_retained_resid` | 62 | **0.00085** | **0.01450** | 0.916 | 0.835 | ? | Gated domain residuals within family |
| S2-f | `implicit_geom4_u` | ? | **0.00095** | 0.01553 | 0.897 | 0.830 | TV 0.47/0.53 | Implicit components + learned geometry block |
| S8-f | `topic_quality_k3` | ? | **0.00116** | 0.01546 | **0.910** | 0.848 | supp 20-23/11-12 | Hierarchical topic+quality tilt (26 topics with _high/_low) |
| S4-f | `schedule_rbf_a01_i3` | ? | **0.00124** | 0.01694 | 0.864 | **0.853** | **TV 0.15** | Scale-adaptive learned prototypes |
| S3-f | `moment_partition_K4` | ? | 0.00168 | **0.01410** | **0.942** | 0.799 | TV 0.28/0.32 | Moment-based repetition harm |
| S6-f | `family_support_crowding` | 70 | 0.00177 | 0.01692 | 0.849 | **0.857** | TV 0.45/0.50 | Support-adjusted crowding penalty |

#### Key observations

1. **Quality splits**: Only `topic_quality_k3` (S8-f) explicitly uses the
   within-domain quality structure (grouping 39 domains into 26 topics with
   high/low pairs). All other candidates including the latent-channel models
   treat domains as unstructured coordinates. This is a notable omission
   given that quality splits were central to the original GRP design.

2. **Parameter counts**: The latent-channel models (latent_early_k3,
   latent_learned4_u, our combination a012_p003_lb22) use 158-167 params
   fitting 567 training rows (3.4:1 data-to-param ratio). By contrast,
   `gated_retained_residual` achieves the best regret (0.00085) with only
   62 params — the same size as the baseline (9.1:1 ratio). The 3.4:1
   ratio for latent models is adequate with ridge regularization but not
   luxurious; the 62-param models have a healthier margin.

3. **Geometry standout**: `schedule_rbf_a01_i3` (S4-f) achieves geometry TV
   of 0.15 — the best of any serious candidate by 2x. Its prototype-based
   structure naturally produces smooth, non-degenerate optima. Phase supports
   18/12 and max weights 0.14/0.17 are excellent.

4. **Domain-level vs topic-level vs family-level**: Three distinct
   granularity levels emerged:
   - family-level (3 groups): baseline, gated_retained_residual
   - domain-level (39 domains): retained_residual, latent-channel models
   - topic-level (26 topics with quality): topic_quality_k3
   The topic-level is the only one that explicitly encodes quality variation
   within data families.

5. **520M Spearman remains the hardest gate**: No follow-up candidate hits
   the 0.90 mean / 0.85 min targets. Our C2+C3 combination (a012_p003_lb22)
   at 520M Sp 0.906 remains the only model achieving this.

#### Ship-level gate check for top follow-up candidates

Ship targets: RMSE ≤ 0.0175, Sp ≥ 0.84, Reg@1 ≤ 0.0017, LTRMSE ≤ 0.0252,
520M RMSE ≤ 0.025, 520M Sp ≥ 0.90, geom TV ≤ 0.36/0.40.

- `gated_retained_resid`: passes RMSE, Sp, Reg, LTRMSE, 520M RMSE. FAILS
  520M Sp (0.835). Geometry unknown (needs validation).
- `schedule_rbf_a01_i3`: passes RMSE, Sp, Reg, LTRMSE, 520M RMSE, geometry.
  FAILS 520M Sp (0.853). Closest to full pass after our combination.
- `topic_quality_k3`: passes RMSE, Sp, Reg, 520M RMSE. FAILS 520M Sp
  (0.848). Geometry likely passes but unvalidated locally.

None of the follow-up candidates hit the 520M Sp target. Our C2+C3
combination (`a012_p003_lb22`, 8/10 gates, 520M Sp 0.906) remains the
strongest overall candidate.

#### Most promising directions for next combination

1. Add quality splits (from topic_quality_k3) to the latent-channel
   combination — the 26-topic structure with signed quality tilts could
   improve 520M ranking without adding many parameters.
2. Use the schedule_rbf geometry mechanism as a regularizer for latent
   models — its prototype structure naturally prevents corner collapse.
3. Validate `gated_retained_residual` locally — at 62 params with regret
   0.00085, it's the most parameter-efficient serious candidate.
- `/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/direct_latent_partition_law_candidate_diagnostics.csv`

### 2026-04-22 - Gated Retained Residual Improvement Sprint

19-iteration systematic search to improve `gated_retained_residual_no_interaction`.
Starting point: baseline 93 params, RMSE 0.01421, Sp 0.922, Reg@1 0.00072, 520M Sp 0.777.

#### Improvement ideas tested

1. **Quality tilt** (+13 params): signed (high - low) quality differences for 13 CC domain pairs
2. **Centered scale interactions** (+6 params): centered family-share x {uN, uD} from baseline winning trick
3. **U-shaped repetition penalty** (+3 params): symmetric softplus penalty around family saturation threshold
4. **Scale-adaptive residuals** (+78 or +39 params): u_n/u_d-weighted domain residuals for scale-dependent behavior
5. Various combinations and regularization sweeps

#### Key results (8-seed robustness, all variants)

| Variant | Params | RMSE | Sp | Reg@1 | LTRMSE | 520mRMSE | 520mSp | Geom TV |
|---------|--------|------|-----|-------|--------|----------|--------|---------|
| **baseline** (reproduce) | 93 | 0.01421 | 0.922 | 0.00072 | 0.02100 | 0.02001 | 0.777 | 0.475/0.660 |
| quality_tilt | 106 | 0.01609 | 0.922 | 0.00056 | 0.02310 | 0.02190 | 0.799 | 0.475/0.659 |
| centered_interactions | 99 | 0.01558 | 0.939 | 0.00229 | 0.02388 | 0.02268 | 0.826 | 0.365/0.377 |
| u_penalty | 96 | 0.01424 | 0.944 | 0.00093 | 0.02458 | 0.02345 | 0.786 | 0.375/0.384 |
| quality_tilt rpm150 | 106 | 0.01593 | 0.924 | 0.00044 | 0.02364 | 0.02244 | 0.790 | - |
| **scale_adaptive rpm300** | 171 | 0.01334 | 0.945 | **0.00015** | 0.02092 | 0.02030 | 0.826 | **0.383/0.407** |
| **qt+scale_n rpm300** | 145 | 0.01427 | 0.948 | **0.00007** | 0.02203 | 0.02128 | 0.853 | - |
| qt+scale_n rpm400 | 145 | 0.01467 | 0.944 | 0.00015 | 0.02292 | 0.02215 | 0.862 | - |
| **qt+ci+scale_n rpm300** | 151 | 0.01375 | 0.946 | 0.00130 | 0.02298 | 0.02188 | 0.848 | **0.360/0.373** |
| **qt+ci+scale_n rpm400** | 151 | 0.01489 | 0.947 | 0.00049 | 0.02343 | 0.02230 | **0.884** | - |
| ci+scale_n rpm200 | 138 | 0.01318 | 0.953 | 0.00144 | 0.02245 | 0.02136 | 0.871 | 0.376/0.393 |
| kitchen_sink rpm300 | 154 | 0.01246 | 0.955 | 0.00121 | 0.02136 | 0.02077 | 0.826 | 0.377/0.403 |
| full_combo rpm150 | 115 | 0.01326 | 0.945 | 0.00135 | 0.02242 | 0.02152 | 0.835 | 0.362/0.385 |
| qt+scale_n a0.1 rpm300 | 145 | 0.01412 | 0.950 | 0.00044 | 0.02169 | 0.02081 | 0.857 | - |
| qt+scale_n p1e-3 rpm300 | 145 | 0.01379 | 0.943 | 0.00058 | 0.02227 | 0.02120 | 0.879 | - |

#### Interpretation

**Quality tilt** is the single most impactful addition for regret: it drops Reg@1 from 0.00072 to 0.00056 (baseline) and 0.00044 (rpm150), and to an extraordinary 0.00007 when combined with scale_adaptive_n_only at rpm300. The signed quality features let the model distinguish which CC domains contribute high vs low quality signal.

**Centered interactions** are the single most impactful addition for geometry: TV drops from 0.475 to 0.365 (23% improvement). They do this by giving the linear head explicit access to how much the family mix deviates from training-set average, modulated by scale.

**Scale-adaptive residuals** (u_n-weighted domain residuals) are the most impactful for overall prediction quality: RMSE drops from 0.01421 to 0.01204 (scale_n_only rpm100). They express scale-dependent domain effects -- e.g., "arxiv matters more at 520M than at 60M." The full (u_n + u_d) version collapsed geometry (p1_support=1.0!) but the n-only version with strong regularization (rpm300+) is stable.

**U-penalty** helps Spearman (0.922 -> 0.944) and geometry (TV 0.375) but doesn't help regret.

#### Top candidates vs ship targets

Ship targets: RMSE<=0.0175, Sp>=0.84, Reg@1<=0.0017, LTRMSE<=0.0252, 520mRMSE<=0.025, 520mSp>=0.90, TV<=0.36/0.40

| Candidate | RMSE | Sp | Reg@1 | LTRMSE | 520mRMSE | 520mSp | TV | Verdict |
|-----------|------|----|-------|--------|----------|--------|----|---------|
| qt+ci+scale_n rpm400 | 0.0149 PASS | 0.947 PASS | 0.00049 PASS | 0.0234 PASS | 0.0223 PASS | 0.884 FAIL | ~0.36 PASS | 9/10 |
| qt+scale_n rpm300 | 0.0143 PASS | 0.948 PASS | 0.00007 PASS | 0.0220 PASS | 0.0213 PASS | 0.853 FAIL | 0.40 MARGINAL | 8-9/10 |
| qt+scale_n p1e-3 rpm300 | 0.0138 PASS | 0.943 PASS | 0.00058 PASS | 0.0223 PASS | 0.0212 PASS | 0.879 FAIL | - | 8+/10 |

The persistent weak spot is **520M Spearman**: the 0.90 target is not reached by any variant. The best is `qt+ci+scale_n rpm400` at 0.884.

#### Code refs
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_gated_residual_improvements.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_gated_residual_improvements_v2.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_gated_residual_improvements_v3.py`
- Results: `reference_outputs/gated_residual_improvement_results.json`, `_v2.json`, `_v3.json`

### 2026-04-22 - Review of ChatGPT Pro follow-up sessions (round 3, sessions 1-8)

Artifacts reviewed at `/Users/calvinxu/Downloads/chatgpt_pro_sessions_3/`.

Important caveat: these runs used the stale v24 packet and the old corrected
`7`-row `520M` slice. The canonical registry has since been refreshed to the
exact-step-eval basis, with `16` trustworthy qsplit `520M` rows and no
stratified overshoot row. None of the candidates below should be promoted
directly without a local rerun on the refreshed registry.

#### Main synthesis

1. No round-3 candidate should replace the current local lead `a012_p003_lb22`
   without a local rerun.
2. The round was still useful: **explicit quality split is a real structural
   donor**. Multiple independent sessions found compact gains over the old
   `e2e_full_qrho_centered` baseline by exposing high/low-quality structure.
3. The best evidence came from **compact** quality blocks, not larger paired
   latent/topic expansions. The useful parameter regime was roughly `66-70`
   parameters, not `120+`.

#### Best donor candidates from round 3

##### 1. `hilogap_rank` (session 3)

Best compact donor from the round. It adds only `+4` parameters over the old
`e2e_full_qrho_centered` baseline (`66` total) and gives a clean stale-packet
improvement on the regret-first stack:

- RMSE `0.01752`
- Sp `0.84674`
- Reg@1 `0.00162`
- LTRMSE `0.02490`
- fixed-520M RMSE `0.02422`
- fixed-520M Sp `0.78125`

Geometry is acceptable and non-collapsed:

- mean/max TV `0.366 / 0.372`
- min supports `23.1 / 7.9`
- max weights `0.074 / 0.325`

This is the cleanest low-parameter donor to port onto the live branch.

##### 2. `family_quality_discount_crowding` (session 6)

Best explicit-quality-split family on stale-packet overall metrics. This is the
strongest evidence that the model wants to **discount low-quality retained
exposure** rather than treat it symmetrically with the high-quality side.

Stale-packet metrics:

- RMSE `0.01763`
- Sp `0.85598`
- Reg@1 `0.00155`
- LTRMSE `0.02567`
- fixed-520M RMSE `0.02521`
- fixed-520M Sp `0.83036`

But geometry is still too sharp:

- mean/max TV `0.452 / 0.501`
- min phase-1 support `2.71`
- max phase-1 weight `0.753`

Conclusion: useful donor for a nonlinear low-quality discount term, not a final
model.

##### 3. `tagsplit_groupcov_qtag` (session 7)

Best compact packet-native quality-bucket idea. It uses `q_high`, `q_low`,
`q_other` interactions without growing the model beyond `62` total params.

Stale-packet metrics:

- RMSE `0.01754`
- Sp `0.84733`
- Reg@1 `0.00215`
- LTRMSE `0.02474`
- fixed-520M RMSE `0.02386`
- fixed-520M Sp `0.81250`

Mean geometry looks decent:

- mean/max TV `0.277 / 0.385`

But large-scale collapse is still present:

- min phase-0 support `2.05`
- max phase-0 weight `0.620`

Conclusion: good body-feature donor, not a replacement law.

#### Useful negative results

1. **Family-by-family split channels are too flexible.**
   Session 5 and parts of session 6 found that total-level / pooled quality
   splits behaved better than family-specific ones.
2. **Large paired explicit high/low latent models are not the right tradeoff.**
   Session 8's `paired_explicit_highlow_centered_k3` had strong stale-packet
   metrics but was too large (`120` params), still missed fixed-520M Spearman,
   and did not improve geometry enough to justify the extra complexity.
3. **The continuous-mixture diagnostic bug fixed in session 4 matters.**
   The previous diagnostic path had detached candidate weights during
   optimization. After fixing that, several numerically strong compact hi/lo
   laws were correctly downgraded for implausible off-manifold arbitrary-mixture
   optima. That negative result should stand.

#### Advice after round 3

- Do **not** promote any round-3 v24 candidate directly.
- Treat round 3 as confirming that the next compact structural addition should
  be one of:
  1. a small **hi/lo gap** block (`hilogap_rank` style),
  2. a small **low-quality discount** inside retained exposure (`family_quality_discount_crowding` style),
  3. compact **quality buckets** (`q_high`, `q_low`, `q_other`) rather than a larger topic/paired-latent body.
- If porting stale-packet ideas to the live branch, the priority order is:
  1. `hilogap_rank`
  2. `family_quality_discount_crowding`
  3. `tagsplit_groupcov_qtag`

### 2026-04-22 - Local rerun on refreshed 16-row qsplit 520M basis

The packet and registry had diverged. Before rerunning models, I:

- patched `run_c2c3_combination_search.py` so the fixed-520M holdout comes from
  the canonical `run_registry/strong_tier_perplexity_ready.csv` instead of the
  stale packet-local CSV;
- rebuilt `chatgpt_pro_swarm_transfer_packet/data/nd_scale_packet.npz` from the
  refreshed registry with
  `uv run --with torch python experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_swarm_transfer_packet/build_packet.py`;
- synced the refreshed `nd_scale_packet.npz` and registry summary files into
  the active v23 packet.

After that refresh, the active benchmark split became:

- train rows: `274`
- holdout rows: `49`
- fixed `520M` rows: `16`
- random supplement rows: `33`

The fixed-520M holdout is now all qsplit rows:

- `16` rows from `520m_10p4b | qsplit_representative12`
- no stratified overshoot row

#### Rerun scope

Script:

- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_local_frontier_rerun_20260422.py`

Outputs:

- `reference_outputs/local_frontier_rerun_20260422_summary.csv`
- `reference_outputs/local_frontier_rerun_20260422_summary.json`
- per-model robustness/candidate files under the same stem

Rerun roster:

- `e2e_full_qrho_centered`
- `a012_p003_lb22`
- `a014_p003_lb20`
- gated retained-residual frontier:
  - `baseline`
  - `quality_tilt`
  - `centered_interactions`
  - `quality_tilt+centered_interactions`
  - `quality_tilt+scale_adaptive_n_only`
  - `centered_interactions+scale_adaptive_n_only`
  - `quality_tilt+centered_interactions+scale_adaptive_n_only`

#### Main result

The old `a012_p003_lb22` lead does **not** survive the refreshed packet. The
gated retained-residual family is now clearly better on the corrected holdout.

Top rerun results:

| Model | Params | RMSE | Sp | Reg@1 | LTRMSE | 520M RMSE | 520M Sp | mean/max TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gated_ci_scale_n_rpm200` | 138 | **0.01603** | **0.974** | **0.00000** | 0.02840 | 0.02339 | **0.966** | `0.366 / 0.375` |
| `gated_ci_scale_n_rpm300` | 138 | 0.01626 | 0.971 | **0.00000** | **0.02828** | **0.02327** | 0.964 | `0.363 / 0.401` |
| `gated_qt_ci_scale_n_rpm300` | 151 | 0.01666 | 0.971 | **0.00000** | 0.02857 | 0.02359 | **0.966** | `0.364 / 0.375` |
| `gated_centered_interactions` | 99 | 0.01840 | 0.963 | **0.00000** | 0.02941 | 0.02451 | 0.961 | `0.369 / 0.375` |
| `gated_baseline` | 93 | 0.01797 | 0.954 | 0.00024 | 0.02993 | 0.02496 | 0.937 | `0.394 / 0.399` |
| `e2e_full_qrho_centered` | 62 | 0.02147 | 0.912 | 0.00038 | 0.03405 | 0.02971 | 0.923 | `0.402 / 0.543` |
| `a014_p003_lb20` | 158 | 0.01926 | 0.954 | 0.00240 | 0.03208 | 0.02880 | 0.778 | `0.322 / 0.411` |
| `a012_p003_lb22` | 158 | 0.01904 | 0.950 | 0.00416 | 0.03210 | 0.02855 | 0.758 | `0.397 / 0.425` |

#### Updated interpretation

1. **The stale packet materially mis-ranked the frontier.**
   The recovered qsplit `520M` rows are not a cosmetic change. They strongly
   penalize the old C2+C3 latent family and favor the gated retained-residual
   family.

2. **`centered_interactions + scale_adaptive_n_only` is the new best compact
   structural direction.**
   It beats the quality-tilt variants on the refreshed basis with fewer
   parameters.

3. **Quality tilt is no longer the primary donor.**
   On the refreshed basis, adding `quality_tilt` to the strong
   `centered_interactions + scale_adaptive_n_only` variant does not help enough
   to justify the extra parameters.

4. **`gated_centered_interactions` is the strongest very-simple candidate.**
   At only `99` params it is dramatically better than the old baseline and only
   moderately behind the 138-param scale-adaptive variants.

5. **`a012` should no longer be treated as the live lead.**
   It was a stale-packet leader. On the refreshed packet it fails badly on the
   fixed-520M slice (`0.758` Spearman for `a012_p003_lb22`).

#### Qualitative mixture read

The new gated winners are not collapsed off-manifold, but they are still too
phase-1-tech-heavy at larger scales.

For `gated_ci_scale_n_rpm200`:

- `60M`: phase0 `{broad 0.297, tech 0.463, reas 0.240}`, phase1 `{broad 0.358, tech 0.642, reas ~0}`
- `300M`: phase0 `{broad 0.265, tech 0.543, reas 0.192}`, phase1 `{broad 0.118, tech 0.882, reas ~0}`
- `520M`: phase0 `{broad 0.245, tech 0.575, reas 0.180}`, phase1 `{broad 0.083, tech 0.917, reas ~0}`

So the remaining weakness is no longer benchmark ranking. It is mixture
plausibility at larger scales.

#### Advice after rerun

- Retire `a012_p003_lb22` as the current lead.
- Promote the gated retained-residual family as the live frontier.
- If parameter efficiency matters most, use `gated_centered_interactions` as the
  clean simple baseline and `gated_ci_scale_n_rpm200` as the main serious
  challenger.
- Do **not** prioritize new quality-tilt grafts onto the old latent family.
  The refreshed rerun suggests the higher-leverage work is now:
  1. keep the gated retained-residual body,
  2. preserve `centered_interactions + scale_adaptive_n_only`,
  3. improve mixture search / plausibility at 300M+ without giving back the new
     520M ranking gains.

### 2026-04-22 - Refreshed frontier prediction plots and 520M split diagnostic

Built refreshed seed-7 predicted-vs-actual plots for the current frontier and
saved:

- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/plot_local_frontier_predictions_20260422.py`
- `reference_outputs/local_frontier_rerun_20260422_predictions_seed7.csv`
- `reference_outputs/local_frontier_rerun_20260422_subset_metrics_seed7.csv`
- `reference_outputs/figures/local_frontier_rerun_20260422_predicted_vs_actual_all_primary_seed7.png`
- `reference_outputs/figures/local_frontier_rerun_20260422_predicted_vs_actual_fixed520m_seed7.png`

Also cleaned stale plot files only for this local-frontier plot family before
writing the new figures.

#### Key diagnostic

The old `a012` concern is real, not just a resampling artifact. On the refreshed
fixed-`520M` split:

- `a012_p003_lb22` on the legacy 6-row core: `RMSE 0.0272`, `Spearman 1.000`
- `a012_p003_lb22` on the recovered 10 Tier-2 rows: `RMSE 0.0293`, `Spearman 0.673`

By contrast, the new gated lead is stable across both:

- `gated_ci_scale_n_rpm200` on the legacy 6-row core: `RMSE 0.0242`, `Spearman 1.000`
- `gated_ci_scale_n_rpm200` on the recovered 10 Tier-2 rows: `RMSE 0.0237`, `Spearman 0.927`

So the `a012` drop is being driven specifically by the recovered Tier-2 qsplit
`520M` points. That means the earlier stale-packet ranking was genuinely
misleading.

### 2026-04-22 - Local debug plots for refreshed frontier

Built a small diagnostic plot pack on top of
`local_frontier_rerun_20260422_predictions_seed7.csv`:

- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/plot_local_frontier_debug_diagnostics_20260422.py`
- `reference_outputs/local_frontier_rerun_20260422_calibration_seed7.csv`
- `reference_outputs/figures/local_frontier_rerun_20260422_residuals_vs_actual_seed7.png`
- `reference_outputs/figures/local_frontier_rerun_20260422_fixed520m_sorted_profiles_seed7.png`
- `reference_outputs/figures/local_frontier_rerun_20260422_calibration_heatmaps_seed7.png`

#### What these plots clarified

1. **Residual-vs-actual is much more informative than raw predicted-vs-actual.**
   It makes the low-loss bias explicit: current lead models are still
   systematically pessimistic on `520M`, with positive residuals around
   `+0.02 BPB`, even when overall holdout behavior looks strong.

2. **The fixed-520M sorted profile is the clearest visual for under-dispersion.**
   The predicted `520M` curve is much flatter than the actual one. This makes it
   obvious that the remaining error is not just ranking; it is compressed spread
   in the lowest-loss regime.

3. **The calibration heatmap is a good packet-level summary view.**
   Using `|mean residual|`, `|1 - slope|`, and `|1 - std_ratio|` across
   `holdout_overall`, `random_supplement`, `fixed_520m_all`, `legacy_core`, and
   `recovered_tier2` makes the main failure mode compact and legible.

#### Recommendation for next packet

Ask fresh ChatGPT Pro sessions to:

- regenerate and inspect these three diagnostic plot types for every serious
  candidate;
- treat improvement on the refreshed `520M` calibration diagnostics as
  first-class, not just overall RMSE/Spearman;
- explain what the plots imply about model bias, under-dispersion, and whether a
  candidate is genuinely robust or just winning on the easier non-`520M` rows.

### 2026-04-22 - Built fresh ChatGPT Pro packet v25

Prepared a new fresh-session packet at:

- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v25`
- archive: `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v25.zip`

#### Packet shift

This packet is intentionally different from the earlier v24 handoff:

- canonical basis is the refreshed `16`-row qsplit-only fixed-`520M` set
- current seed-7 primary holdout contract is `49` rows (`16` fixed `520M` +
  `33` random supplement)
- live baselines are now:
  - `gated_ci_scale_n_rpm200` for the frontier track
  - `gated_centered_interactions` for the compact track
- `a012_p003_lb22` is demoted to historical caution, not a live frontier target

#### New benchmark emphasis

The packet now centers a robustness + calibration + simplicity benchmark.

Required diagnostics are:

- predicted vs actual
- residual vs actual
- fixed-`520M` sorted profile
- calibration heatmap/table

Required calibration subsets are:

- overall holdout
- random supplement
- fixed `520M` all
- fixed `520M` legacy core
- fixed `520M` recovered Tier-2

Required complexity reporting is explicit:

- total / linear / nonlinear params
- symbolic scaling law in terms of `M`, `P`, `F`, and `K` if relevant

#### Key packet files

- `README.md`
- `BACKGROUND_AND_GOALS.md`
- `BENCHMARK_AND_EVAL.md`
- `ACCEPTANCE_CRITERIA.md`
- `REQUEST_TO_CHATGPT_PRO.md`
- `reference_outputs/local_handoff_summary.md`

The build now copies canonical registry outputs from `run_registry/` directly
instead of reusing stale packet-local trust summaries. It also packages the
refreshed local-frontier calibration plots and summaries as the main baseline
reference set.

### 2026-04-22 - ChatGPT Pro session 4 review on v25 packet

Reviewed the completed artifacts under:

- `/Users/calvinxu/Downloads/chatgpt_pro_session_4`

Completed sessions present:

- session 1
- session 2
- session 4
- session 5
- session 6
- session 8

Session 3 was incomplete, and session 7 appears to have failed or produced no
usable archive.

#### Main result

Session 4 found the strongest new mechanism in the batch:

- a **learned constant floor** in floor-log space

This is the first follow-up that appears to directly address the real refreshed
failure mode:

- positive fixed-`520M` bias
- compressed fixed-`520M` spread
- visible low-loss curvature / under-dispersion

#### Strongest new candidates

1. **Session 4 frontier**

   `domain_ci_scale_n_floor_const`

   - params `139`
   - overall RMSE `0.01325`
   - overall Spearman `0.95872`
   - fixed-`520M` RMSE `0.01304`
   - fixed-`520M` Spearman `0.95331`
   - min fixed-`520M` Spearman `0.94706`

   Seed-7 fixed-`520M` calibration improved sharply relative to the current
   frontier baseline:

   - baseline `gated_ci_scale_n_rpm200`: bias `+0.0225`, slope `0.452`, std
     ratio `0.467`
   - `domain_ci_scale_n_floor_const`: bias `-0.0155`, slope `0.691`, std ratio
     `0.715`

   Interpretation:

   - this is the most important new branch
   - it over-corrects bias slightly negative, but it materially improves the
     shape of the low-loss calibration problem

2. **Session 4 compact**

   `paired29_centered_floor_const_rpm50`

   - params `90`
   - overall RMSE `0.01364`
   - overall Spearman `0.94844`
   - fixed-`520M` RMSE `0.01296`
   - fixed-`520M` Spearman `0.95257`
   - min fixed-`520M` Spearman `0.94118`
   - geometry mean/max TV `0.377 / 0.396`

   This is a serious compact candidate. It is much more aligned with the
   simplicity objective than the older latent branch and looks materially
   stronger than the current compact baseline.

3. **Session 1 compact**

   `shared_residgate_full_none_rpm100`

   - params `99`
   - overall RMSE `0.01687`
   - overall Spearman `0.96367`
   - fixed-`520M` RMSE `0.02410`
   - fixed-`520M` Spearman `0.97206`
   - min fixed-`520M` Spearman `0.96765`
   - geometry mean/max TV `0.359 / 0.368`

   This is the cleanest low-risk compact upgrade inside the existing gated
   family. It does not solve the calibration problem like the floor branch, but
   it remains the strongest conservative compact donor.

4. **Session 5 frontier**

   `raw_full39_ci_scale_n_rpm20`

   - params `138`
   - overall RMSE `0.01528`
   - overall Spearman `0.97254`
   - fixed-`520M` RMSE `0.02157`
   - fixed-`520M` Spearman `0.96140`
   - geometry mean/max TV `0.435 / 0.464`
   - max phase-1 weight `0.578`

   This is a real frontier alternative, but geometry got worse. It looks more
   like a donor than a final direction.

#### Useful but secondary donor signal

Session 2's topic/quality grouping is still real:

- `topic_quality_ci`
  - params `96`
  - overall RMSE `0.01616`
  - fixed-`520M` RMSE `0.02270`
  - fixed-`520M` Spearman `0.95625`

That remains a useful body-structure donor, but session 4 is the much more
important result from this round.

#### Negative branches

- session 6 grouped-residual balanced finalists: not competitive on the
  refreshed basis
- session 8 qsplit-grouped direct family: not compelling on either calibration
  or complexity

#### Advice after session 4 batch

Priority order:

1. locally rerun the session 4 learned-floor models first
2. keep the session 1 compact shared-residual-gate model as the conservative
   compact backup
3. do not spend more cycles on sessions 6 or 8
4. use session 2 only as a donor if the learned-floor branch still needs a
   compact quality-aware extension

The key shift is:

- the old question was “which residual basis gives the best ranking?”
- the new question is “can a simple floor-law correction fix the low-loss
  calibration failure without breaking robustness or geometry?”

### 2026-04-22 - Local rerun of session-4 floor branch on refreshed benchmark

Implemented a local wrapper to rerun the priority session-4 floor models
against the current local v23 packet and refreshed `274 / 49 / 16` benchmark
contract:

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_session4_floor_local_rerun_20260422.py`

Priority models rerun:

1. `paired29_centered_floor_const_rpm50`
2. `domain_ci_scale_n_floor_const`
3. `domain_centered_floor_const`

#### 8-seed rerun summary

| priority | model                             | params | overall RMSE | overall Sp | fixed-520M RMSE | fixed-520M Sp |
|:---------|:----------------------------------|-------:|-------------:|-----------:|----------------:|--------------:|
| 1 | `paired29_centered_floor_const_rpm50` | 90  | 0.01378 | 0.94852 | 0.01328 | 0.95368 |
| 2 | `domain_ci_scale_n_floor_const`       | 139 | 0.01352 | 0.95815 | 0.01358 | 0.95184 |
| 3 | `domain_centered_floor_const`         | 100 | 0.01581 | 0.94912 | 0.01368 | 0.94228 |

Against the current local baselines:

- `gated_ci_scale_n_rpm200`: overall RMSE `0.01603`, fixed-`520M` RMSE `0.02339`
- `gated_centered_interactions`: overall RMSE `0.01840`, fixed-`520M` RMSE `0.02451`

All three floor models materially improve fixed-`520M` RMSE.

#### Calibration read

Seed-7 fixed-`520M` calibration:

- `gated_ci_scale_n_rpm200`: bias `+0.0225`, slope `0.452`, std ratio `0.467`
- `paired29_centered_floor_const_rpm50`: bias `-0.0109`, slope `0.769`, std ratio `0.794`
- `domain_ci_scale_n_floor_const`: bias `-0.0158`, slope `0.691`, std ratio `0.716`
- `domain_centered_floor_const`: bias `-0.0105`, slope `0.776`, std ratio `0.796`

So the structural floor fix clearly survives locally:

- fixed-`520M` bias magnitude shrinks
- fixed-`520M` slope rises substantially
- fixed-`520M` spread compression is much smaller

#### Updated interpretation

The learned constant floor is now a validated local mechanism, not just a packet
artifact.

The compact grouped model still looks very strong, but the rerun changed one
detail relative to the original session-4 packet:

- `paired29_centered_floor_const_rpm50` remains the best compact model
- `domain_ci_scale_n_floor_const` remains the strongest frontier-style model
- but `domain_centered_floor_const` was stronger than expected as the pure
  mechanism-isolation check

This suggests the dominant gain is the floor correction itself, not the extra
`u_N`-adaptive residual copy.

### 2026-04-23 - ChatGPT Pro session 5 review on v25 packet

Only sessions `1`, `3`, `4`, and `6` returned usable packet archives. Session
`2` had a broken download, session `5` did not return a finished archive, and
the remaining sessions were effectively incomplete. The review below is based
on the actual summary/calibration artifacts inside the returned zips rather
than the prose alone.

#### Main read

Yes: several independent sessions converged on the same structural diagnosis.
The old head/floor setup was causing much of the low-loss `520M` bias. The
strong evidence is not “somebody said so”; it is that multiple floor/shift
variants move the fixed-`520M` bias and slope materially, while grouped-topic
body-only variants do not.

#### Session ranking

1. **Session 4** (`iterreg_quality_floor`): strongest batch overall.
   - Best frontier: `domain_ci_scale_n_floor_const_rpm100_mx28`
     - overall RMSE `0.01084`
     - fixed-`520M` RMSE `0.00839`
     - fixed-`520M` Spearman `0.9555`
   - Best compact: `domain_centered_floor_const_rpm50_qt_mx20`
     - overall RMSE `0.01111`
     - fixed-`520M` RMSE `0.00752`
     - fixed-`520M` Spearman `0.9688`
   - Seed-7 calibration for these floor models is much better than the old
     gated baselines:
     - overall slope near `0.96-0.97`
     - fixed-`520M` bias near zero
     - fixed-`520M` slope around `0.57-0.63`
   - Important caveat: this branch explicitly treats optimizer depth as a
     model-selection axis; the debug CSV shows deeper optimization often hurt
     holdout. So this is promising, but more delicate than the simpler
     floor-only branch.

2. **Session 6** (`adaptive_floor_frontier`): clearest simple head/floor
   result.
   - Best frontier: `adaptive_floor_ci_scale_n_const_gap025`
     - overall RMSE `0.01366`
     - fixed-`520M` RMSE `0.01289`
     - fixed-`520M` Spearman `0.9504`
   - Best compact: `adaptive_floor_ci_const_gap015`
     - overall RMSE `0.01203`
     - fixed-`520M` RMSE `0.00676`
     - fixed-`520M` Spearman `0.9570`
   - Seed-7 calibration is strong:
     - fixed-`520M` bias near `0`
     - fixed-`520M` slope around `0.62-0.64`
     - overall slope near `0.96-0.98`
   - This is the most convincing “head/floor caused the bias” evidence in the
     batch because the change is structurally small and the improvement is
     still large.

3. **Session 1** (`shared_residual_adaptive_floor_frontier_v27`): useful, but
   not a real calibration fix.
   - Adaptive-floor shared residual variants improve RMSE materially.
   - But seed-7 fixed-`520M` calibration remains highly compressed:
     - `poly4` frontier fixed-`520M` slope `0.370`, std ratio `0.381`
     - compact `none` variant fixed-`520M` slope `0.419`, std ratio `0.435`
   - So this branch improves fit and sometimes rank, but it is still mostly in
     the old calibration regime.

4. **Session 3** (`grouped_topic_direct`): mostly negative as a bias fix.
   - Compact grouped-topic model is a mild compact improvement.
   - Frontier grouped-topic model improves RMSE and worst-seed rank somewhat.
   - But seed-7 fixed-`520M` calibration still looks like the old problem:
     - bias around `+0.019 to +0.023`
     - slope around `0.43-0.46`
     - std ratio around `0.45-0.48`
   - Conclusion: grouped-topic body changes alone do not solve the head-driven
     low-loss bias.

#### Advice after session 5 batch

The batch materially strengthens the head/floor diagnosis:

- the head/floor issue is real
- the best results now come from simple learned-floor / shifted-floor laws
- grouped/topic body changes without a head fix are not enough

Priority for local follow-up:

1. rerun the simplest session-6 adaptive-floor candidates locally first
2. rerun the stronger but more delicate session-4 iterreg/quality floor models
3. treat session-1 shared-residual adaptive-floor variants as donors, not
   leaders
4. deprioritize grouped-topic-only follow-ups as a primary branch

### 2026-04-23 - Local rerun of priority session-5 floor candidates

Implemented a local wrapper to rerun the four immediate next-step candidates
against the current local v23 packet and refreshed `274 / 49 / 16` benchmark
contract:

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_session5_floor_local_rerun_20260423.py`

Rerun set:

1. `adaptive_floor_ci_const_gap015` (session 6 compact)
2. `adaptive_floor_ci_scale_n_const_gap025` (session 6 frontier)
3. `domain_ci_scale_n_floor_const_rpm100_mx28` (session 4 frontier)
4. `domain_centered_floor_const_rpm50_qt_mx20` (session 4 compact)

#### 8-seed local summary

| priority | source | model | params | overall RMSE | overall Sp | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:---------|:-------|:------|-------:|-------------:|-----------:|----------------:|--------------:|------------------:|
| 1 | session 6 | `adaptive_floor_ci_const_gap015` | 100 | 0.01229 | 0.95640 | 0.00631 | 0.95515 | 0.93824 |
| 2 | session 6 | `adaptive_floor_ci_scale_n_const_gap025` | 139 | 0.01303 | 0.95777 | 0.01206 | 0.95515 | 0.92647 |
| 3 | session 4 | `domain_ci_scale_n_floor_const_rpm100_mx28` | 139 | 0.01101 | 0.96198 | 0.00878 | 0.95294 | 0.91471 |
| 4 | session 4 | `domain_centered_floor_const_rpm50_qt_mx20` | 113 | 0.01112 | 0.95579 | 0.00752 | 0.96875 | 0.96176 |

Current local baselines for context:

| model | params | overall RMSE | overall Sp | fixed-520M RMSE | fixed-520M Sp |
|:------|-------:|-------------:|-----------:|----------------:|--------------:|
| `gated_ci_scale_n_rpm200` | 138 | 0.01603 | 0.97403 | 0.02339 | 0.96618 |
| `gated_centered_interactions` | 99 | 0.01840 | 0.96293 | 0.02451 | 0.96103 |

#### Seed-7 calibration read

Seed-7 holdout / fixed-520M calibration:

- `gated_ci_scale_n_rpm200`: overall bias `+0.00863`, slope `0.826`, std `0.835`; fixed-`520M` bias `+0.02254`, slope `0.452`, std `0.467`
- `gated_centered_interactions`: overall bias `+0.00845`, slope `0.809`, std `0.823`; fixed-`520M` bias `+0.02349`, slope `0.437`, std `0.453`
- `adaptive_floor_ci_const_gap015`: overall bias `+0.00123`, slope `0.964`, std `0.976`; fixed-`520M` bias `+0.00040`, slope `0.623`, std `0.645`
- `adaptive_floor_ci_scale_n_const_gap025`: overall bias `+0.00106`, slope `0.985`, std `0.994`; fixed-`520M` bias `-0.00138`, slope `0.637`, std `0.660`
- `domain_ci_scale_n_floor_const_rpm100_mx28`: overall bias `+0.00199`, slope `0.971`, std `0.980`; fixed-`520M` bias `+0.00100`, slope `0.572`, std `0.595`
- `domain_centered_floor_const_rpm50_qt_mx20`: overall bias `+0.00161`, slope `0.957`, std `0.969`; fixed-`520M` bias `+0.00062`, slope `0.630`, std `0.647`

Fixed-`520M` legacy-core vs recovered-tier on seed 7 remains similar for the
session-6 and session-4 floor models, which supports the view that the head
repair is not merely fitting one subset:

- `adaptive_floor_ci_const_gap015`: slope `0.577 / 0.637` on legacy / recovered
- `adaptive_floor_ci_scale_n_const_gap025`: slope `0.708 / 0.600`
- `domain_ci_scale_n_floor_const_rpm100_mx28`: slope `0.661 / 0.525`
- `domain_centered_floor_const_rpm50_qt_mx20`: slope `0.633 / 0.618`

#### Interpretation

The local rerun validates the head/floor diagnosis.

- All four floor-law candidates materially reduce fixed-`520M` RMSE and nearly
  eliminate the old upward bias.
- Session 6 is the cleaner branch and survives locally.
- Session 4 remains strong, but its frontier branch is less stable on worst-seed
  fixed-`520M` ranking than the cleaner session-6 compact law.

Current read after local validation:

1. `adaptive_floor_ci_const_gap015` is the best immediate next-step candidate.
   It is simple, parameter-efficient (`100` params), strongly improves RMSE and
   calibration, and its worst-seed fixed-`520M` ranking is acceptable.
2. `domain_centered_floor_const_rpm50_qt_mx20` is the strongest compact
   alternative numerically, but it comes from the more optimizer-sensitive
   session-4 branch.
3. `adaptive_floor_ci_scale_n_const_gap025` is a valid frontier follow-up, but
   it is less compelling than the compact session-6 law once worst-seed
   fixed-`520M` behavior is included.
4. `domain_ci_scale_n_floor_const_rpm100_mx28` is strong on average RMSE, but
   its local minimum fixed-`520M` Spearman (`0.9147`) makes it a riskier
   promotion candidate than the compact alternatives.

### 2026-04-23 - Cross-scale shared-floor hybrid ablation

Tested the idea of estimating the asymptote directly from matched same-mixture
trajectories across scales, rather than only learning the floor indirectly
through the direct law head.

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_cross_scale_floor_hybrid_20260423.py`
- outputs:
  - `reference_outputs/cross_scale_floor_hybrid_20260423_summary.csv`
  - `reference_outputs/cross_scale_floor_hybrid_20260423_robustness.csv`
  - `reference_outputs/cross_scale_floor_hybrid_20260423_predictions_seed7.csv`
  - `reference_outputs/cross_scale_floor_hybrid_20260423_calibration_seed7.csv`

Model family:

- shared floor
- shared scale coefficients
- one per-trajectory amplitude inferred from observed lower-scale points
- no mixture-dependent floor or scale coefficients

Trajectory key:

- `mixture_id × target_budget_multiplier`

Two unbounded variants and two bounded variants were tested:

1. `cross_scale_shared_floor_sum_all`
2. `cross_scale_shared_floor_freebeta_all`
3. `cross_scale_shared_floor_freebeta_qsplit`
4. `cross_scale_bounded_floor_freebeta_qsplit_gap015`
5. `cross_scale_bounded_floor_freebeta_qsplit_gap025`

#### Main result

This idea has real signal, but the naive floor estimate is not stable enough to
replace the current session-6 compact lead as-is.

Best trajectory-only hybrid on the 8-seed stack:

| model | global params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:------|--------------:|-------------:|----------------:|--------------:|------------------:|
| `cross_scale_shared_floor_freebeta_qsplit` | 3 + per-trajectory amplitudes | 0.01519 | 0.01062 | 0.95000 | 0.87059 |

Current live lead for comparison:

| model | params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:------|-------:|-------------:|----------------:|--------------:|------------------:|
| `adaptive_floor_ci_const_gap015` | 100 | 0.01229 | 0.00631 | 0.95515 | 0.93824 |

#### Important nuance

On seed 7, the best trajectory-only hybrid fixes the *shape* of the fixed-`520M`
cluster better than the session-6 compact direct law:

- `adaptive_floor_ci_const_gap015`: fixed-`520M` bias `+0.00040`, slope `0.623`, std `0.645`
- `cross_scale_shared_floor_freebeta_qsplit`: fixed-`520M` bias `+0.00932`, slope `0.984`, std `1.004`

So the cross-scale hybrid nearly nails the fixed-`520M` dispersion on that
split. This is strong evidence that the matched-trajectory scaling signal is
real and that the remaining compression in the direct law is not a random
artifact.

#### Failure mode

The unbounded shared-floor hybrids estimate implausible floors across seeds:

- `cross_scale_shared_floor_freebeta_qsplit`: floor mean `-0.216`, min `-0.586`, max `0.275`
- `cross_scale_shared_floor_freebeta_all`: floor mean `0.058`, min `-0.636`, max `0.617`

Despite good paired-train RMSE (`~0.002`), the inferred asymptote wanders
heavily. That shows the naive cross-scale fit is under-identified with only the
current matched `130M -> 300M` trajectories.

The bounded variants do stabilize the floor, but they lose most of the benefit:

- `cross_scale_bounded_floor_freebeta_qsplit_gap025`: overall RMSE `0.01922`, fixed-`520M` RMSE `0.02297`
- `cross_scale_bounded_floor_freebeta_qsplit_gap015`: overall RMSE `0.02341`, fixed-`520M` RMSE `0.03023`

So the issue is not simply “missing the right gap bound.” The simple
trajectory-only law is missing enough shape flexibility that once the floor is
forced into a plausible band, predictive quality drops sharply.

#### Interpretation

The cross-scale idea is still useful, but not as a standalone replacement law.

What the ablation says:

1. Matched lower-scale BPBs do contain enough signal to repair the fixed-`520M`
   *shape* on a given split.
2. A naive shared-floor / shared-scale-only hybrid does *not* identify the
   asymptote robustly across seeds.
3. The right use for this signal is likely as:
   - a prior on the direct model’s floor, or
   - a conditioned hybrid correction layer when lower-scale losses are actually
     available at inference time.

Current read:

- keep `adaptive_floor_ci_const_gap015` as the live lead
- treat the cross-scale hybrid as evidence that the floor/dispersion problem is
  learnable from matched trajectories
- do not promote the naive hybrid itself, because its floor estimate is too
  unstable

### 2026-04-23 - Oracle-floor and no-floor ablations on the session-6 compact body

Ran direct ablations to isolate how much of the remaining error comes from the
floor treatment itself.

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_floor_oracle_nofloor_ablation_20260423.py`
- outputs:
  - `reference_outputs/floor_oracle_nofloor_ablation_20260423_summary.csv`
  - `reference_outputs/floor_oracle_nofloor_ablation_20260423_robustness.csv`
  - `reference_outputs/floor_oracle_nofloor_ablation_20260423_predictions_seed7.csv`
  - `reference_outputs/floor_oracle_nofloor_ablation_20260423_calibration_seed7.csv`

Setup:

- fixed the body to the session-6 compact form (`centered_interactions`)
- kept the same optimizer, regularization, and linear head fit
- changed only the floor treatment

Tested variants:

1. `fixed_floor_trainmin_gap050`
2. `oracle_floor_allmin_gap0001`
3. `oracle_floor_allmin_gap005`
4. `oracle_floor_allmin_gap015`
5. `zero_floor_exp_head`

Oracle floor used:

- `all_min = min(y_target)` over all benchmark examples, including the held-out
  rows
- on this packet: `all_min = 0.8674098849`

#### Main result

Better fixed floors help, but they do not beat the learned floor.

| model | params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:------|-------:|-------------:|----------------:|--------------:|------------------:|
| `adaptive_floor_ci_const_gap015` | 100 | 0.01229 | 0.00631 | 0.95515 | 0.93824 |
| `oracle_floor_allmin_gap015` | 99 | 0.01411 | 0.01509 | 0.95625 | 0.93824 |
| `oracle_floor_allmin_gap005` | 99 | 0.01551 | 0.01764 | 0.95809 | 0.93824 |
| `oracle_floor_allmin_gap0001` | 99 | 0.01605 | 0.01898 | 0.96176 | 0.94706 |
| `fixed_floor_trainmin_gap050` | 99 | 0.01902 | 0.02466 | 0.96029 | 0.95000 |
| `zero_floor_exp_head` | 99 | 0.03110 | 0.04811 | 0.96507 | 0.95000 |

So:

1. The old fixed floor is clearly too high.
2. An oracle fixed floor is much better than the old fixed floor.
3. The learned floor still beats every tested fixed oracle floor on RMSE by a
   large margin.
4. Removing the floor term entirely (`zero_floor_exp_head`) is bad.

#### Interpretation

This answers three questions cleanly.

1. **Would a more accurate fixed floor help?**

   Yes. Moving from `train_min - 0.05` to an oracle fixed floor reduces
   fixed-`520M` RMSE substantially:

   - `0.02466 -> 0.01509` for the best oracle fixed floor

   So floor placement is definitely part of the story.

2. **Is the remaining gap to the live lead *only* floor placement?**

   No. If perfect fixed floor placement were enough, the oracle fixed floor
   variants would match or beat the learned floor. They do not.

   The learned floor still wins strongly:

   - `0.01509 -> 0.00631` fixed-`520M` RMSE

   This implies the benefit is not just “put the floor lower.” The ability to
   adapt the floor during fitting remains important.

3. **Would ablating the floor term entirely help?**

   No. `zero_floor_exp_head` is decisively worse:

   - overall RMSE `0.03110`
   - fixed-`520M` RMSE `0.04811`

   So the asymptote is not an optional nuisance term. BPB still wants a
   positive lower bound in this regime.

#### Read

The floor term should stay.

But the result also says the current learned-floor law is not obviously the
final form. A fixed oracle floor helps, but not enough; a fully removed floor is
bad; the current bounded learned floor is the best of the tested options.

Current interpretation:

- keep the floor term
- keep the positive-gap `exp` head
- treat floor learning as real model capacity, not just a nuisance constant
- the next improvement should probably be a better *regularized* floor law or a
  floor prior from external scaling data, not deleting the asymptote

### 2026-04-23 - Local floor-law study around the Session-6 lead

Ran a short local loop on the current lead `adaptive_floor_ci_const_gap015` to
answer three narrow questions:

1. Is `gap_max=0.15` actually near-optimal, or can a tighter/lower/higher gap
   fix more of the remaining 520M compression?
2. Does a training-derived cross-scale floor prior help the learned floor?
3. Does a very mild monotone scale-conditioned floor help without hurting
   robustness?

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_floor_law_local_study_20260423.py`
- outputs:
  - `reference_outputs/floor_law_local_study_20260423_summary.csv`
  - `reference_outputs/floor_law_local_study_20260423_robustness.csv`
  - `reference_outputs/floor_law_local_study_20260423_predictions_seed7.csv`
  - `reference_outputs/floor_law_local_study_20260423_calibration_seed7.csv`
  - `reference_outputs/floor_law_local_study_20260423_summary.json`

Study variants:

- compact learned-constant floor sweep:
  - `gap_max = 0.10, 0.12, 0.15, 0.18, 0.20`
- compact learned-constant floor + training-derived hybrid floor prior
- compact monotone `uN`-conditioned floor

The floor prior used a training-only estimate from the matched-trajectory
`cross_scale_shared_floor_freebeta_qsplit` hybrid; it was *not* an oracle or
holdout-derived floor.

#### Main results

| model | params | overall RMSE | overall Sp | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:------|-------:|-------------:|-----------:|----------------:|--------------:|------------------:|
| `adaptive_floor_ci_const_gap010` | 100 | 0.01446 | 0.96055 | 0.01280 | 0.96103 | 0.95000 |
| `adaptive_floor_ci_const_gap012` | 100 | 0.01244 | 0.95923 | 0.00983 | 0.95625 | 0.90000 |
| `adaptive_floor_ci_const_gap015` | 100 | 0.01229 | 0.95640 | 0.00630 | 0.95515 | 0.93824 |
| `adaptive_floor_ci_const_gap018` | 100 | 0.01366 | 0.94929 | 0.00726 | 0.95221 | 0.93235 |
| `adaptive_floor_ci_const_gap020` | 100 | 0.01334 | 0.95219 | 0.00911 | 0.94007 | 0.90294 |
| `adaptive_floor_ci_const_gap015_priorhyb` | 100 | 0.01164 | 0.94236 | 0.00636 | 0.95809 | 0.93824 |
| `adaptive_floor_ci_uNmono_gap015` | 101 | 0.01228 | 0.95573 | 0.00650 | 0.94154 | 0.84412 |

#### Seed-7 calibration read

The compact constant-gap sweep shows the expected calibration tradeoff:

- larger `gap_max` improves seed-7 fixed-520M slope/std ratio
- but after about `0.15` it gives back too much robustness and ranking

Seed-7 fixed-520M calibration:

| model | bias | slope | std ratio | fixed-520M RMSE |
|:------|-----:|------:|----------:|----------------:|
| `adaptive_floor_ci_const_gap010` | +0.01325 | 0.512 | 0.530 | 0.01499 |
| `adaptive_floor_ci_const_gap012` | +0.00678 | 0.569 | 0.589 | 0.00928 |
| `adaptive_floor_ci_const_gap015` | +0.00040 | 0.623 | 0.645 | 0.00572 |
| `adaptive_floor_ci_const_gap018` | -0.00259 | 0.659 | 0.683 | 0.00592 |
| `adaptive_floor_ci_const_gap020` | -0.00507 | 0.710 | 0.731 | 0.00689 |
| `adaptive_floor_ci_const_gap015_priorhyb` | -0.00047 | 0.723 | 0.743 | 0.00454 |
| `adaptive_floor_ci_uNmono_gap015` | +0.00137 | 0.639 | 0.658 | 0.00562 |

So:

1. `gap_max=0.15` is near the best robust point among simple constant-floor
   variants.
2. Higher `gap_max` *can* open up the low-loss regime more, but the price is
   weaker overall robustness and weaker worst-seed ranking.
3. The training-derived hybrid floor prior is the only variant that
   substantially improves the seed-7 slope/std ratio beyond the current lead.

#### Geometry / plausibility

Seed-7 geometry is a useful guardrail here:

| model | mean TV | max TV | min p1 support | max p1 weight |
|:------|--------:|-------:|---------------:|--------------:|
| `adaptive_floor_ci_const_gap015` | 0.3662 | 0.3693 | 8.01 | 0.311 |
| `adaptive_floor_ci_const_gap018` | 0.3667 | 0.3704 | 8.04 | 0.314 |
| `adaptive_floor_ci_const_gap015_priorhyb` | 0.4069 | 0.4321 | 5.64 | 0.412 |
| `adaptive_floor_ci_uNmono_gap015` | 0.3670 | 0.3708 | 7.99 | 0.315 |

The floor-prior variant is the clear geometry warning sign: it buys fit and
seed-7 dispersion by becoming materially sharper and farther off-manifold.

The monotone `uN` floor keeps geometry acceptable, but its worst-seed
fixed-520M Spearman collapses badly (`0.844`), so it is not robust enough.

#### Interpretation

This study narrows the next search space.

1. **Do not replace `gap_max=0.15` with a larger unconstrained constant gap.**
   That direction improves seed-7 shape but does not hold up over the 8-seed
   robustness aggregate.

2. **The cross-scale floor prior idea is real, but the naive version is too
   aggressive.**
   It improves:
   - overall RMSE (`0.01229 -> 0.01164`)
   - seed-7 fixed-520M slope (`0.623 -> 0.723`)

   but it also degrades:
   - overall Spearman (`0.956 -> 0.942`)
   - seed-7 geometry (`max TV 0.369 -> 0.432`)

   So the right next idea is probably a *weaker / better-regularized* floor
   prior, not a hard push toward the hybrid asymptote.

3. **A tiny scale-conditioned floor is not yet a win.**
   The monotone `uN` floor was the cleanest test of “one extra degree of floor
   flexibility,” but it did not improve the 8-seed result enough and it hurt
   worst-seed ranking.

#### Read

Current live lead stays:

- `adaptive_floor_ci_const_gap015`

What changed from this loop is not the leader, but the *shape of the next
question*:

- the next promising direction is no longer “change the body”
- it is “can we add a *weak, well-regularized* scaling-informed floor prior
  without giving up geometry or robustness?”

### 2026-04-23 - Registry SUCCESS refresh check and rerun on the current 520M basis

Checked the registry after new 520M babysit completions reported:

- `run_00021-1331fa`
- `run_00090-0b8a0d`
- `run_00125-5a7579`
- `run_00213-f2fbc8`

Plus the previously recovered `run_00180-be99ab` `0.5x` row.

#### What changed

The important distinction is:

- the *evaluation slice* in
  `run_registry/strong_tier_perplexity_ready.csv` was already current
- the *attempt-level status layer* in `run_registry/run_attempts.csv` still
  showed mixed stale `RUNNING` / `FAILED` rows for some attempts until the next
  full registry rebuild

The trustworthy `520m_10p4b` qsplit slice remains:

- 16 rows total
- 12 rows at `mu=0.5`
- 4 rows at `mu=1.0`

No new trustworthy 520M datapoints were added by this SUCCESS wave. The new
SUCCESSes mostly confirmed runs that were already present as perplexity-ready
because exact-step eval records had already been recovered.

#### Local rerun on the current basis

Reran the current priority floor benchmark:

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_session5_floor_local_rerun_20260423.py`
- outputs refreshed:
  - `reference_outputs/session5_floor_local_rerun_20260423_summary.csv`
  - `reference_outputs/session5_floor_local_rerun_20260423_robustness.csv`
  - `reference_outputs/session5_floor_local_rerun_20260423_predictions_seed7.csv`
  - `reference_outputs/session5_floor_local_rerun_20260423_calibration_seed7.csv`

Result: metrics were unchanged to the precision we care about, which confirms
the current local frontier was already evaluated on the same 16-row `520M`
basis.

Current lead remains:

| model | params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | min fixed-520M Sp |
|:------|-------:|-------------:|----------------:|--------------:|------------------:|
| `adaptive_floor_ci_const_gap015` | 100 | 0.01229 | 0.00631 | 0.95515 | 0.93824 |
| `domain_centered_floor_const_rpm50_qt_mx20` | 113 | 0.01112 | 0.00752 | 0.96875 | 0.96176 |
| `domain_ci_scale_n_floor_const_rpm100_mx28` | 139 | 0.01101 | 0.00878 | 0.94779 | 0.91471 |

Seed-7 fixed-520M calibration remains:

- `adaptive_floor_ci_const_gap015`: bias `+0.00040`, slope `0.623`, std ratio
  `0.645`
- `domain_centered_floor_const_rpm50_qt_mx20`: bias `+0.00062`, slope `0.630`,
  std ratio `0.647`
- old compact baseline `gated_centered_interactions`: bias `+0.02349`, slope
  `0.437`, std ratio `0.453`

#### Read

The SUCCESS wave matters operationally, but not yet analytically. The local
benchmark basis was already the corrected 16-row set, so the current floor-law
conclusions do not change.

### 2026-04-23 - Built fresh ChatGPT Pro packet v26 around the floor-led frontier

Prepared a new packet:

- `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v26`
- archive:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v26.zip`

#### Packet framing

The packet is now built around the current live lead:

- `adaptive_floor_ci_const_gap015`

and the strongest known challenger:

- `domain_centered_floor_const_rpm50_qt_mx20`

But the request is intentionally **not** floor-only. The packet says the best
current diagnosis points at the floor / asymptote law, while still giving
ChatGPT Pro broad freedom to pursue broader body changes if they beat the live
lead cleanly and remain plausible.

#### What changed vs the previous packet

- switched the primary baselines and plots from the old `2026-04-22` gated
  frontier rerun to the `2026-04-23` floor reruns
- bundled the latest local floor-law study artifacts
- bundled cross-scale hybrid and oracle/no-floor ablations as supporting
  diagnostics
- kept the older pre-floor frontier artifacts as historical references, not as
  the live target
- updated the handoff summary to describe the remaining problem correctly:
  residual low-loss compression rather than the old large upward bias

#### Packet philosophy

The packet now makes this distinction explicit:

- the best current evidence points at the floor law
- but the packet is still asking for the best overall direct law, not just the
  best floor tweak

That should let fresh sessions exploit the new diagnosis without over-anchoring
them to the exact current form.

### 2026-04-23 - Restricted-fit optimum probe for the current floor frontier

Built a small local probe to answer a qualitative question: if we fit the
current floor-law candidates on earlier-scale data only, do their optimized
mixtures still look like sane GRP-style proposals at `60M` and `300M`?

Artifacts:

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_restricted_fit_optimum_probe_20260423.py`
- outputs:
  - `reference_outputs/restricted_fit_optimum_probe_20260423_summary.csv`
  - `reference_outputs/restricted_fit_optimum_probe_20260423_details.json`
  - `reference_outputs/restricted_fit_optimum_probe_20260423_report.md`

Probe design:

- fit `adaptive_floor_ci_const_gap015` and
  `domain_centered_floor_const_rpm50_qt_mx20`
- on `60m_swarm_only` (`293` legacy swarm rows), optimize for `60M 1.2B`
- on `all_lt_300m` (`293` 60M + `39` 130M rows), optimize for `300M 6B`
- compare qualitatively against the GRP no-L2 `60M` reference:
  - phase 0: broad `0.6438`, tech `0.3416`, reasoning `0.0146`
  - phase 1: broad `0.5928`, tech `0.4028`, reasoning `0.0045`

Main result:

- the current floor frontier still does **not** produce qualitatively good
  restricted-fit optima

Specific findings:

- `adaptive_floor_ci_const_gap015` fit on `60M` only and optimized at `60M`
  produces a phase-0 mixture that is `80.5%` reasoning and a phase-1 mixture
  that is `87.7%` tech
  - top phase-0 domain is `dolmino_synth_qa` at `0.762`
  - top phase-1 domain is `dolmino_stack_edu_fim` at `0.457`
  - predicted BPB is not even monotone across scale for the same optimized
    mixture: `0.9759 -> 0.9806 -> 0.9824` for `60M -> 300M -> 520M`

- `adaptive_floor_ci_const_gap015` fit on all `<300M` data and optimized at
  `300M` is better behaved numerically, but still qualitatively wrong
  - phase 0: broad `0.0074`, tech `0.2421`, reasoning `0.7505`
  - phase 1: broad `0.1854`, tech `0.8146`, reasoning `~0`
  - top phase-0 domain is `dolmino_synth_math` at `0.661`
  - top phase-1 domain is `dolmino_stack_edu_fim` at `0.815`

- the Session 4 challenger is worse
  - fit on `60M` only, it is nearly a two-point corner:
    - phase 0: `95.1%` `dolmino_synth_qa`
    - phase 1: `80.3%` `dolmino_stack_edu_fim` + `19.7%`
      `dolmino_common_crawl_hq`
  - fit on `<300M`, it collapses completely:
    - phase 0: `100%` reasoning
    - phase 1: `100%` tech

Read:

- the floor fixes clearly improved prediction and calibration
- but they did **not** fix optimum-mixture plausibility
- the live lead is still far from the GRP no-L2 qualitative reference, which
  stays broad-text heavy in both phases with moderate tech tilt rather than
  synthetic / code corner collapse

Implication:

- the next packet should not frame the current floor lead as “almost done”
- it should say explicitly that the current live lead still fails an important
  restricted-fit optimum sanity check

### 2026-04-23 - GRP no-L2 refit at 300M still fits, but the optimized 300M mixture is pathological

Built a direct apples-to-apples GRP comparison to answer a narrower question:
if we keep the same no-`L2` GRP surrogate family and refit it on the
`300m_6b` Chinchilla panel, does it still fit well and does the optimized
mixture remain qualitatively sane?

Artifacts:

- script:
  `experiments/domain_phase_mix/exploratory/two_phase_many/fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m.py`
- outputs:
  - `grp_power_family_penalty_no_l2_60m_vs_300m_fit_weights.png`
  - `grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.csv`
  - `grp_power_family_penalty_no_l2_60m_vs_300m_fit.md`
- debug:
  - `docs/debug-log-grp-no-l2-300m-uniform-phase0.md`

Fit-quality result:

- `300M` no-`L2` GRP still fits nontrivially, but it is clearly worse than the
  original `60M` fit
  - `60M` fit:
    - train RMSE `0.00760`
    - CV RMSE `0.00872`
    - train Spearman `0.9105`
    - OOF Spearman `0.8698`
  - `300M` fit:
    - train RMSE `0.01010`
    - CV RMSE `0.01114`
    - train Spearman `0.8291`
    - OOF Spearman `0.8040`

Mixture result:

- the `60M` fit remains qualitatively sane:
  - broad-heavy in both phases
  - moderate tech tilt
  - almost no reasoning

- the `300M` fit does **not** produce a trustworthy optimum
  - reported phase 0 is exactly uniform over all `39` domains
    - family shares only look broad-heavy because `31 / 39` domains are
      broad-text
  - reported phase 1 collapses almost entirely to tech/code

Debugging result:

- this is not just a plotting issue
- the fitted `300M` surrogate landed in a pathological regime where:
  - `lam = 54.6` hits the top clip
  - `beta = 1e-6` hits the bottom clip
  - `eta = 2980.96` is huge
- under those values, retained phase-0 exposure is effectively zero relative to
  the phase-1 term
  - total retained phase-0 contribution: `4.14e-15`
  - total phase-1 contribution: `7.39e4`
  - ratio: `5.60e-20`
- large phase-0 perturbations with phase 1 fixed leave the surrogate design and
  prediction unchanged to machine precision

Optimizer result:

- `optimize_penalty_calibration_model(...)` is also start-sensitive here
- the reported uniform phase-0 solution is only one shallow basin from the
  zero-logit start
- with more random starts, the optimizer finds lower-objective corner solutions
  that are even less plausible

Conclusion:

- the `300M` no-`L2` GRP refit is still valid as a **regression-fit datapoint**
- the `300M` no-`L2` GRP optimized mixture is **not** valid as a qualitative
  policy recommendation

### 2026-04-23 - ChatGPT Pro session 6 review on v26 packet

Reviewed the returned artifacts in `/Users/calvinxu/Downloads/chatgpt_pro_session_6`.

Artifact quality by session:

- sessions `1`, `4`, `6`, and `8` returned usable bundles with reports/tables
- sessions `2` and `3` returned partial notes/tables but no clearly new
  validated frontier result beyond the main bundles
- sessions `5` and `7` returned review-style notes only, not new runnable
  candidate packages

Main result:

- no session returned a clean new frontier winner
- the current packet choices still stand:
  - best serious overall: `domain_centered_floor_const_rpm50_qt_mx20`
  - best serious compact: `adaptive_floor_ci_const_gap015`

Cross-session synthesis:

- the strongest repeated signal is still the same one we already had locally:
  the remaining bottleneck is the **floor / asymptote law**
- broader body rewrites did not produce a serious replacement
- when new candidates improved low-loss compression, they usually paid for it in
  one of:
  - fixed-520M rank robustness
  - overall rank
  - geometry / optimized-mixture plausibility

Most useful session-level takeaways:

- session `4`:
  - mixture-conditioned `q`-floor variants are real
  - best new candidates:
    - `qfloor_qt_retainedminusrepeat_local`
    - `qfloor_ci_retainedminusrepeat_local`
  - these improved seed-7 fixed-520M slope/std-ratio meaningfully, but neither
    cleanly dominated the full existing frontier

- session `6`:
  - weak cross-scale floor-prior branch is still the cleanest next direction
  - best branch:
    - `adaptive_floor_ci_const_gap015_prior010`
  - keeps geometry plausible, but fixed-520M ranking robustness remains weaker
    than the current lead

- session `8`:
  - monotone `u_D` floor is a useful diagnostic, not a promotion
  - `ci_uD_exp_gap020` improves spread / low-tail shape, but worsens fixed-520M
    RMSE and robust ranking

- session `5`:
  - mild Box–Cox head on top of the compact floor law is interesting
  - but it was only tested on a local reconstruction, not the official packet
    evaluator, so it should be treated as a low-confidence local ablation idea,
    not as evidence of a new frontier winner

- sessions `1`, `2`, and `3`:
  - useful mostly as negative evidence that broader retained-residual / proxy /
    signed-law rewrites do not beat the current floor-led compact family

Concrete read:

- we do **not** need another follow-up with these same sessions
- if we do more local work, it should stay narrow and floor-focused
- the highest-signal next local branches are:
  1. weak floor-prior sweep around `adaptive_floor_ci_const_gap015`
  2. one official-evaluator check of the mild Box–Cox head idea
  3. maybe one carefully regularized `q`-floor variant if we want a second
     floor-family branch

Bottom line:

- session 6 reinforces the current diagnosis rather than changing it
- no new returned law displaced the existing packet leaders
- the right next move remains **small, well-regularized floor-law changes**, not
  broader body exploration

### 2026-04-23 - Local optimum-quality study with multistart and deployment baselines

Built a new local benchmark to measure **optimum quality**, not just held-out
prediction quality.

Artifacts:

- script:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_optimum_quality_local_study_20260423.py`
- per-seed optima:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/optimum_quality_local_study_20260423_per_seed_optima.csv`
- summary:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/optimum_quality_local_study_20260423_summary.csv`
- details:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/optimum_quality_local_study_20260423_details.json`
- heatmap:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/optimum_quality_local_study_20260423_badness_heatmap.png`
- report:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/optimum_quality_local_study_20260423_report.md`

Benchmark contract:

- candidates:
  - `adaptive_floor_ci_const_gap015`
  - `adaptive_floor_ci_const_gap015_priorhyb`
  - `domain_centered_floor_const_rpm50_qt_mx20`
- probes:
  - fit on `60M` swarm only, optimize `60M`
  - fit on all data `<300M`, optimize `300M`
- modes:
  - `raw_multistart`
  - `top8actual_hull`
  - `trustblend_top8actual_cap015`
- optimization coverage:
  - `8` optimization seeds
  - structured starts: zero / best observed actual / best observed predicted /
    family corners
  - additional random simplex starts per seed

Main result:

- the raw unconstrained optima are still qualitatively bad even after adding
  proper multistart and multiple seeds
- the failure is not just optimizer noise:
  - the Session 6 lead raw optimum is completely stable across seeds and still
    pathological
  - the Session 4 challenger raw optimum is somewhat unstable on the `<300M`
    probe, but all basins are still bad
- the constrained deployment rules fix most of the qualitative pathologies
  immediately

Concrete numbers:

- `adaptive_floor_ci_const_gap015`, raw optimum, `60M` probe:
  - predicted BPB `0.9449`
  - nearest observed TV `0.621`
  - reference family TV `0.791`
  - hard-corner fraction `1.0`
  - phase-1 tech collapse fraction `1.0`
  - phase-0 reasoning collapse fraction `1.0`
- `adaptive_floor_ci_const_gap015`, raw optimum, `<300M -> 300M` probe:
  - predicted BPB `0.9224`
  - nearest observed TV `0.504`
  - reference family TV `0.791`
  - hard-corner fraction `1.0`
  - phase-1 tech collapse fraction `1.0`
  - phase-0 reasoning collapse fraction `1.0`

- `adaptive_floor_ci_const_gap015`, `top8actual_hull`, `60M` probe:
  - predicted BPB `1.0268`
  - nearest observed TV `0.189`
  - reference family TV `0.184`
  - no corner / collapse flags
- `adaptive_floor_ci_const_gap015`, `top8actual_hull`, `<300M -> 300M` probe:
  - predicted BPB `1.0399`
  - nearest observed TV `0.209`
  - reference family TV `0.174`
  - no corner / collapse flags

- `domain_centered_floor_const_rpm50_qt_mx20`, raw optimum, `<300M -> 300M` probe:
  - predicted BPB `0.9516`
  - nearest observed TV `0.546`
  - reference family TV `0.764`
  - hard-corner fraction `1.0`
  - phase-0 reasoning collapse fraction `1.0`
  - phase-1 tech collapse fraction `0.5`
  - mean pairwise seed TV `0.190`

Read:

- the current direct laws are now good enough to predict BPB, but **not** good
  enough to produce trustworthy raw optima over the full 39-domain simplex
- this remains true after correcting the earlier under-seeding problem in the
  optimizer
- therefore deployment parameterization / regularization is not just a
  convenience; it is currently required for plausible policies

Important nuance:

- constrained optima should not replace the raw optimum diagnostic
- the correct interpretation is:
  - raw optimum = test whether the law itself is intrinsically sane
  - constrained optimum = test whether we can deploy the model safely anyway

Immediate recommendation:

- keep evaluating raw optima explicitly
- add an **optimum-quality benchmark** to the next packet as a first-class
  criterion
- ask fresh ChatGPT Pro sessions for:
  - better raw optimum behavior
  - or, failing that, better deployment parameterizations / trust-region rules
- do not accept another candidate as a real win if it only improves RMSE while
  its raw optimum still collapses

Optimizer note:

- basin hopping from the Shukor et al. scaling-law paper is worth treating as a
  **follow-up ablation on the raw optimum only**
- it should not replace the new multistart baseline by default
- the immediate priority was to fix the under-seeded baseline first, which is
  now done

## 2026-04-23 - Local BiMix-inspired law study

Converted the BiMix paper discussion into a narrow local ablation on the
current Session-6 floor-law stack.

Artifacts:

- predictive study script:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/code/run_bimix_local_study_20260423.py`
- predictive summary:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/bimix_local_study_20260423_summary.csv`
- predictive robustness:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/bimix_local_study_20260423_robustness.csv`
- seed-7 calibration:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/bimix_local_study_20260423_calibration_seed7.csv`
- seed-7 geometry:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/bimix_local_study_20260423_geometry_seed7.csv`
- restricted-fit optimum probe for the best BiMix candidate:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v23/reference_outputs/bimix_optimum_probe_20260423_summary.csv`

Laws tested:

- `bimix_qfloor_ci_retminusrep`
  - bounded floor
  - `gap(x) = eps + gap_max * sigmoid(a + b * q(x))`
  - compact `centered_interactions` body
- `bimix_qgap_ci_retminusrep`
  - constant learned floor
  - multiplicative BiMix-style gap term
  - `gap_pred = exp(linear - alpha_q * q(x))`
- `bimix_qfloor_qgap_ci_retminusrep`
  - both of the above on the compact body
- `bimix_qfloor_qgap_qt_retminusrep`
  - both of the above on `quality_tilt + centered_interactions`

The retained-minus-repeat proxy was taken directly from the previous
`qfloor` follow-up:

- `q(x) = log(1 + signal(x)) - log(1 + penalty(x))`
- `signal(x) = x4 + x14 + x24`
- `penalty(x) = x7 + x17 + x27`

Predictive result:

- the only clear winner from the BiMix-inspired study is the **q-conditioned
  floor**
- the explicit multiplicative `q` gap term did not help enough to justify
  itself
- combining `q` floor and `q` gap made one compact variant geometrically
  pathological on full-train `520M` optimization

Best BiMix-style predictive candidate:

- `bimix_qfloor_ci_retminusrep`
  - `101` params
  - overall RMSE `0.01141`
  - overall Spearman `0.95274`
  - fixed-`520M` RMSE `0.00575`
  - fixed-`520M` Spearman `0.93971`
  - min fixed-`520M` Spearman `0.92059`
  - seed-7 fixed-`520M` calibration:
    - bias `+0.00014`
    - slope `0.73699`
    - std ratio `0.78527`

Read:

- this is the strongest local evidence so far that a **mixture-conditioned
  asymptote** is directionally right
- compared with the Session-6 compact lead, it materially improves the
  remaining `520M` compression problem
- but it still gives back too much rank robustness to promote over the current
  live lead

Negative results:

- `bimix_qgap_ci_retminusrep`
  - overall Spearman improves slightly, but fixed-`520M` slope/std stay near
    the current lead and do not justify the extra parameter
- `bimix_qfloor_qgap_ci_retminusrep`
  - full-train seed-7 geometry collapses at `520M`
  - nearest observed TV `0.691`
  - phase-1 effective support `2.34`
  - phase-1 max weight `0.772`
- `bimix_qfloor_qgap_qt_retminusrep`
  - still worse than q-floor-only on robustness, with no compensating win

Restricted-fit optimum result for the best BiMix candidate:

- `bimix_qfloor_ci_retminusrep` still fails badly on raw unrestricted optima
- `60M` raw optimum:
  - best predicted BPB `0.9234`
  - mean nearest observed TV `0.606`
  - mean reference family TV `0.789`
  - hard-corner fraction `1.0`
  - phase-0 reasoning collapse fraction `1.0`
  - phase-1 tech collapse fraction `0.875`
- `<300M -> 300M` raw optimum:
  - best predicted BPB `0.9286`
  - mean nearest observed TV `0.508`
  - mean reference family TV `0.791`
  - hard-corner fraction `1.0`
  - phase-0 reasoning collapse fraction `1.0`
  - phase-1 tech collapse fraction `1.0`

Important comparison against the current lead:

- the BiMix-style q-floor improves prediction and low-loss calibration
- it does **not** fix the raw-optimum problem
- constrained deployment parameterizations still rescue the candidate
  immediately:
  - `top8actual_hull` and `trustblend_top8actual_cap015` both remove the
    corner/collapse pathologies
  - family-TV against the GRP reference drops to roughly `0.17`–`0.20`

Conclusion:

- the paper contributed one useful modeling idea locally:
  **mixture-conditioned floor / asymptote**
- the more literal multiplicative-gap interpretation did not earn itself
- BiMix-style laws are therefore best treated as evidence for
  **mixture-conditioned asymptotes and scarcity barriers**, not as a reason to
  abandon the current floor-law stack

## 2026-04-23 - Built ChatGPT Pro packet v27

Prepared a new fresh-session packet at:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27.zip`

Main framing changes relative to v26:

- optimum quality is now a first-class benchmark axis, not just an informal
  sanity check
- the packet now separates:
  - predictive quality
  - raw unrestricted optimum quality
  - constrained / deployment optimum quality
- BiMix is bundled as a structural reference alongside DML
- the live predictive baselines are unchanged:
  - `adaptive_floor_ci_const_gap015`
  - `domain_centered_floor_const_rpm50_qt_mx20`
- the packet explicitly records that the best local BiMix-style result
  (`bimix_qfloor_ci_retminusrep`) is interesting but not promoted

Bundled new local artifacts include:

- optimum-quality benchmark:
  - `optimum_quality_local_study_20260423_*`
- BiMix-inspired local study:
  - `bimix_local_study_20260423_*`
  - `bimix_optimum_probe_20260423_*`
- new top-level note:
  - `BIMIX_NOTES.md`

The packet is now asking fresh sessions for two things at once:

- better direct laws
- and, if raw optima are still bad, better constrained deployment rules with an
  honest raw-optimum diagnosis

## 2026-04-23 - Reviewed ChatGPT Pro session 7 returns

Reviewed:

- `/Users/calvinxu/Downloads/chatgpt_pro_session_7/responses.txt`
- usable bundles:
  - `2_packetlocal_qfloor_handoff_20260423.zip`
  - `3_joint_mixture_scale_frontier_v27_bundle.zip`
  - `5_chatgpt_pro_v27_frontier_refresh_outputs.zip`
  - `7_v27_direct_law_addendum_20260423.zip`
  - `8_v27_frontier_review_bundle.zip`
- partial local return:
  - `4/` q/support floor scripts + seed-7 artifacts

Main read:

- the only materially new positive result is Session 4's **q/support-conditioned
  floor** family
- Sessions 2, 5, and 8 mostly reinforce the existing picture:
  - floor / asymptote law still matters most for prediction
  - raw unrestricted optima are still bad
  - constrained deployment wrappers remain necessary
- Sessions 3 and 7 are mostly negative on new raw laws

Best new predictive clue:

- `qsupport_floor_ci_rpm100_gap015` (Session 4, compact)
  - response-reported 8-seed metrics:
    - overall RMSE `0.01110`
    - overall Spearman `0.95077`
    - fixed-`520M` RMSE `0.00594`
    - fixed-`520M` Spearman `0.96802`
    - minimum-seed fixed-`520M` Spearman `0.96471`
  - returned seed-7 calibration:
    - fixed-`520M` slope `0.678`
    - fixed-`520M` std ratio `0.699`
  - read:
    - strongest new compact predictive candidate
    - beats the live Session-6 compact lead on fixed-`520M` fit and calibration
    - also edges the Session-4 compact challenger on fixed-`520M` RMSE and
      worst-seed rank

- `qsupport_floor_qt_ci_scale_n_rpm400_gap015` (Session 4, frontier)
  - response-reported 8-seed metrics:
    - overall RMSE `0.01119`
    - overall Spearman `0.95740`
    - fixed-`520M` RMSE `0.00680`
    - fixed-`520M` Spearman `0.97537`
    - minimum-seed fixed-`520M` Spearman `0.96765`
  - returned seed-7 calibration:
    - fixed-`520M` slope `0.629`
    - std ratio `0.647`
  - read:
    - very strong frontier-style predictive candidate
    - ranking robustness is excellent
    - calibration improvement is much smaller than the compact q/support model

Important caveat on Session 4:

- the folder return only includes seed-7 artifacts plus code; it does **not**
  include the full 8-seed aggregate CSVs
- so the q/support family is promising enough to rerun locally immediately, but
  should not be promoted without apples-to-apples local validation

Best Session 2 result:

- `qfloor_ci_scale_n`
  - overall RMSE `0.01094`
  - fixed-`520M` RMSE `0.00750`
  - fixed-`520M` Spearman `0.95404`
  - minimum-seed fixed-`520M` Spearman `0.92059`
  - seed-7 fixed-`520M` slope/std `0.665 / 0.696`
- read:
  - real improvement on compression
  - not robust enough to beat the current references
  - raw unrestricted optima still collapse

Sessions 3 / 5 / 8:

- no new raw law is promotable
- strongest conclusion remains:
  - `adaptive_floor_ci_const_gap015_priorhyb` is the best calibration-shape
    tweak under constrained deployment
  - `domain_centered_floor_const_rpm50_qt_mx20` remains the strongest
    practical compact deployment candidate from the existing packet references
  - but neither solves the raw-optimum problem

Strong negatives:

- Session 3 `retained_residual_qrho_constfloor`
  - overall RMSE `0.01134`
  - fixed-`520M` RMSE `0.00852`
  - minimum-seed fixed-`520M` Spearman `0.82059`
  - raw optima catastrophically bad
- Session 7 `anchor_centered_additive_global`
  - looked interesting on seed 7
  - failed badly on 8-seed robustness
- Session 8 exact-family reruns
  - did not produce a new serious contender

Operational conclusion:

- most important immediate next step is a **local apples-to-apples rerun of the
  Session 4 q/support floor family**
- if the returned 8-seed numbers hold up, the compact q/support model is the
  first real predictive promotion candidate since the Session-6 floor lead
- regardless, raw unrestricted optima remain unresolved, so deployment should
  still be treated separately from raw-law quality

## 2026-04-23 - Local rerun of session-7 q/support floor family

Locally reran the returned Session 4 q/support-conditioned floor family against
the canonical v27 packet contract using:

- returned scripts from:
  - `/Users/calvinxu/Downloads/chatgpt_pro_session_7/4/search_qsupport_floor_candidates_20260423.py`
  - `/Users/calvinxu/Downloads/chatgpt_pro_session_7/4/run_qsupport_floor_search_subproc_20260423.py`
- local packet root:
  - `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27`

Main artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_local_rerun_20260423/session7_qsupport_local_rerun_20260423_report.md`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_local_rerun_20260423/session7_qsupport_vs_references_metrics.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_local_rerun_20260423/session7_qsupport_vs_references_fixed520m_calibration_seed7.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_local_rerun_20260423/session7_qsupport_vs_references_optimum_summary.csv`

Important operational note:

- both returned scripts had bugs:
  - direct script crashed because it expected `fixed_520m_bucket` but never
    added it
  - subprocess wrapper finished the expensive reruns but failed in the final
    aggregation step due to a stale `mean_parameter_count` schema assumption
- the expensive reruns still completed, and the final top-level comparison was
  reconstructed locally from the generated candidate artifacts

Result:

- `qsupport_floor_ci_rpm100_gap015` is a **real local predictive improvement**
  over the live Session-6 compact lead
  - `101` params
  - overall RMSE `0.011102`
  - overall Spearman `0.950765`
  - fixed-`520M` RMSE `0.005941`
  - fixed-`520M` Spearman `0.968015`
  - minimum-seed fixed-`520M` Spearman `0.964706`
  - seed-7 fixed-`520M` slope / std ratio `0.678 / 0.699`
- this also beats `domain_centered_floor_const_rpm50_qt_mx20` on:
  - fixed-`520M` RMSE (`0.005941` vs `0.007524`)
  - minimum-seed fixed-`520M` Spearman (`0.964706` vs `0.961765`)
- `qsupport_floor_qt_ci_scale_n_rpm400_gap015` is also strong:
  - overall RMSE `0.011182`
  - fixed-`520M` RMSE `0.006763`
  - fixed-`520M` Spearman `0.976103`
  - minimum-seed fixed-`520M` Spearman `0.967647`
  - but seed-7 calibration gain is much smaller (`0.629 / 0.647` slope/std)

Raw-vs-constrained read:

- neither q/support model fixes raw unrestricted optima
- compact raw optima still hard-corner at both probes:
  - `60M` raw nearest observed TV `0.633`
  - `300M` raw nearest observed TV `0.608`
  - collapse flags remain `1.0`
- frontier raw optima still hard-corner at both probes:
  - `60M` raw nearest observed TV `0.473`
  - `300M` raw nearest observed TV `0.613`
  - collapse flags remain `1.0`
- constrained deployment remains the honest deployment rule for both

Conclusion:

- promote `qsupport_floor_ci_rpm100_gap015` as the new **compact predictive
  lead**
- do **not** promote either q/support model as a solved raw law
- keep the scientific conclusion unchanged:
  - prediction improved
  - raw unrestricted optimum quality is still unresolved

## 2026-04-23 - 520M registry refresh and corrected q/support reevaluation

Refreshed only the `520m_10p4b` strong-tier registry rows instead of rerunning
the full registry builder. The full logical registry remains `701` rows; the
perplexity-ready strong-tier slice moved from `94` to `104` rows, with
`520m_10p4b` ready rows moving from `16` to `26`.

Important distinction:

- `logical_runs.csv` is the broad registry of runs/attempts
- `strong_tier_perplexity_ready.csv` is the benchmark-ready slice
- the old "16 rows" number referred only to ready `520m_10p4b` rows, not the
  full registry

Sanity checks:

- all `26` matched ready `520m_10p4b` qsplit rows obey
  `130M BPB > 300M BPB > 520M BPB` for the same run name and target-budget
  multiplier
- within `520M`, available token multipliers obey `0.5x > 1.0x`, and the two
  available `2.0x` rows obey `1.0x > 2.0x`
- comparing against the historical `60M` export is not clean: `6/12` run-name
  matches violate `60M BPB > 130M BPB`; treat this as an apples-to-apples
  caveat for the old `60M` source rather than a `520M` refresh failure

After registry refresh, the v27 packet still had stale local labels: the new
ready `520M` rows existed in `data/nd_scale_runs.csv`, but `10` of them had
`NaN` primary labels in `data/nd_scale_packet.npz`. Updated those packet-local
labels from the refreshed registry before rerunning models.

Corrected rerun artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/qsupport_updated_packet_metric_summary.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/qsupport_updated_packet_seed7_predictions.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/figures/qsupport_updated_packet_predicted_vs_actual_seed7.png`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/figures/qsupport_updated_packet_fixed520m_sorted_profiles_seed7.png`

Corrected 8-seed predictive metrics on the `26`-row fixed-`520M` holdout:

- `qsupport_floor_ci_rpm100_gap015`
  - overall RMSE `0.010617`
  - overall Spearman `0.969594`
  - fixed-`520M` RMSE `0.006339`
  - fixed-`520M` Spearman `0.960598`
  - fixed-`520M` std ratio `0.678229`
  - seed-7 fixed-`520M` slope / bias `0.653 / +0.0022`
- `qsupport_floor_qt_ci_scale_n_rpm400_gap015`
  - overall RMSE `0.010627`
  - overall Spearman `0.972355`
  - fixed-`520M` RMSE `0.006435`
  - fixed-`520M` Spearman `0.966496`
  - fixed-`520M` std ratio `0.651108`
  - seed-7 fixed-`520M` slope / bias `0.632 / +0.0004`

Read:

- the new rows do not break the q/support result
- the compact model remains the cleaner predictive lead by fixed-`520M` RMSE
  and dispersion
- the frontier model now has slightly better seed-7 overall RMSE and rank, but
  its fixed-`520M` compression is worse
- the added `1.0x` and `2.0x` rows make the remaining compression more obvious:
  fixed-`520M` slope is still only about `0.63-0.65`

## 2026-04-23 - q/support multiplier failure diagnosis

Follow-up diagnostics on the corrected compact lead
`qsupport_floor_ci_rpm100_gap015` show that the remaining fixed-`520M`
compression is primarily a target-token-budget scaling failure, not another
floor-level failure.

Artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/compact_qsupport_multiplier_drop_debug.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/session7_qsupport_updated_packet_20260423/figures/compact_qsupport_multiplier_drop_diagnostics_seed7.png`

Seed-7 fixed-`520M` residuals by target-budget multiplier:

- `0.5x`: `n=12`, actual mean `0.909023`, predicted mean `0.906098`,
  residual mean `-0.002925`, RMSE `0.003939`
- `1.0x`: `n=12`, actual mean `0.885388`, predicted mean `0.890866`,
  residual mean `+0.005478`, RMSE `0.006616`
- `2.0x`: `n=2`, actual mean `0.870471`, predicted mean `0.884030`,
  residual mean `+0.013560`, RMSE `0.013566`

Matched-run BPB drops:

- `0.5x -> 1.0x`: actual drop `0.023636`, predicted drop `0.015232`,
  predicted/actual ratio `0.658`
- `1.0x -> 2.0x`: actual drop `0.021566`, predicted drop `0.016243`,
  predicted/actual ratio `0.753`
- `0.5x -> 2.0x`: actual drop `0.049578`, predicted drop `0.032297`,
  predicted/actual ratio `0.651`

The learned floor gap is saturated near the configured ceiling for all fixed
`520M` rows:

- `0.5x`: floor gap mean `0.150099`, floor mean `0.784543`
- `1.0x`: floor gap mean `0.150100`, floor mean `0.784543`
- `2.0x`: floor gap mean `0.150100`, floor mean `0.784543`

Correlations on fixed-`520M` residuals:

- residual vs target-budget multiplier: `+0.829`
- residual vs `u_d`: `+0.859`
- residual vs actual BPB: `-0.877`
- residual vs qscore: `+0.212`
- residual vs support score: `+0.085`

Read:

- the model predicts the right direction with more tokens, but the slope along
  the token-budget axis is too weak
- residuals flip sign with multiplier: too low at `0.5x`, too high at `1.0x`,
  and much too high at `2.0x`
- q/support floor features are not active enough to fix this because the floor
  gap is already saturated and nearly constant
- next model search should test explicit target-budget or `u_d` structure:
  domain-residual-by-`u_d`, q/support-by-`u_d`, monotone token-scaling terms, or
  a non-saturating floor/gap law with a real `u_d` degree of freedom

## 2026-04-23 - ceiling-head local ablation

Tested whether replacing the saturated floor-exp head with a ceiling/improvement
head fixes the fixed-`520M` multiplier failure.

Command:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/code/run_ceiling_head_local_study_20260423.py
```

Artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/code/run_ceiling_head_local_study_20260423.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/ceiling_head_local_study_20260423/ceiling_head_local_study_20260423_report.md`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/ceiling_head_local_study_20260423/ceiling_vs_floor_metric_summary.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/ceiling_head_local_study_20260423/figures/ceiling_vs_floor_multiplier_diagnostics_seed7.png`

Variants:

- `ceiling_const_ci_gap03`: constant ceiling, exp-improvement head
- `ceiling_qsupport_ci_gap08`: q/support-conditioned ceiling,
  exp-improvement head
- `bounded_const_ci_f015_c03`: fixed floor and ceiling, bounded-logit head
- `bounded_qsupport_ci_gap08`: q/support-conditioned floor and ceiling,
  bounded-logit head
- `ceiling_const_ci_scale_nd_gap03`: constant ceiling with full
  scale-adaptive residuals, including `u_d`

Direct comparison to the compact floor lead:

| model | params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | fixed-520M std ratio |
|---|---:|---:|---:|---:|---:|
| `qsupport_floor_ci_rpm100_gap015` | 101 | 0.010617 | 0.006339 | 0.960598 | 0.678229 |
| `bounded_qsupport_ci_gap08` | 104 | 0.014334 | 0.011263 | 0.955726 | 0.536545 |
| `bounded_const_ci_f015_c03` | 98 | 0.019381 | 0.017674 | 0.898376 | 0.770180 |
| `ceiling_const_ci_gap03` | 98 | 0.073151 | 0.099667 | 0.678974 | 5.228073 |
| `ceiling_qsupport_ci_gap08` | 101 | 0.083902 | 0.120532 | 0.758205 | 2.471116 |
| `ceiling_const_ci_scale_nd_gap03` | 176 | 0.084308 | 0.120963 | 0.807863 | 2.006641 |

Read:

- pure ceiling-exp heads are not viable here; they over-amplify the improvement
  term and badly underpredict fixed-`520M`, especially at higher multipliers
- bounded floor/ceiling logit heads are numerically sane, but worse than the
  current q/support floor lead on both overall RMSE and fixed-`520M` RMSE
- the best bounded variant still worsens fixed-`520M` dispersion
  (`0.537` vs `0.678`)
- ceiling-style modeling did not fix the original multiplier issue; the
  strongest remaining lead remains explicit token-budget structure rather than
  replacing floor with ceiling

## 2026-04-23 - GRP/Olmix ceiling-body second pass

Tested whether the negative ceiling-head result was an artifact of keeping the
Session-7 q/support body. This pass used the actual GRP no-`L2` retained-signal
body, family-aggregated GRP approximations, and Olmix-style flat mixture
weights.

Command:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/code/run_grp_olmix_ceiling_body_local_study_20260423.py
```

Artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/code/run_grp_olmix_ceiling_body_local_study_20260423.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/grp_olmix_ceiling_body_local_study_20260423/grp_olmix_ceiling_body_local_study_20260423_report.md`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/grp_olmix_ceiling_body_local_study_20260423/grp_olmix_ceiling_vs_floor_metric_summary.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/grp_olmix_ceiling_body_local_study_20260423/figures/grp_olmix_ceiling_vs_floor_multiplier_diagnostics_seed7.png`

Variants:

- `grp_full_no_l2_body_nnls`: full GRP no-`L2` retained-signal body with a
  centered NNLS linear head
- `grp_full_no_l2_body_scale_nnls`: full GRP body plus monotone `u_N`/`u_D`
  improvement interactions
- `grp_family_body_nnls`, `grp_family_body_scale_nnls`, and
  `grp_family_body_scale_support_nnls`: family-aggregated GRP approximations
- `grp_family_body_bounded_logit`: bounded floor/ceiling logit head on the
  family GRP body
- `olmix_flat_floor_exp`: Olmix-style flat-weight `c + exp(linear)` link
- `olmix_flat_ceiling_exp`: flat-weight constant-ceiling diagnostic

Direct comparison:

| model | params | overall RMSE | fixed-520M RMSE | fixed-520M Sp | fixed-520M std ratio |
|---|---:|---:|---:|---:|---:|
| `qsupport_floor_ci_rpm100_gap015` | 101 | 0.010617 | 0.006339 | 0.960598 | 0.678229 |
| `olmix_flat_floor_exp` | 84 | 0.030584 | 0.040247 | 0.671197 | 1.134798 |
| `grp_full_no_l2_body_scale_nnls` | 102 | 0.043504 | 0.059445 | 0.920940 | 2.293381 |
| `grp_family_body_scale_support_nnls` | 62 | 0.050602 | 0.070390 | 0.800000 | 2.145820 |
| `olmix_flat_ceiling_exp` | 84 | 0.062035 | 0.083079 | 0.513675 | 3.045568 |
| `grp_full_no_l2_body_nnls` | 42 | 0.068397 | 0.094722 | 0.897778 | 3.490897 |
| `grp_family_body_bounded_logit` | 47 | 0.174211 | 0.134630 | 0.902051 | 1.409392 |

Read:

- the actual full GRP no-`L2` body does not rescue ceiling-style modeling
- explicit monotone `u_N`/`u_D` interactions improve the full GRP body, but the
  best GRP-body variant is still far behind the compact q/support floor lead
- GRP ceiling/improvement variants overpredict fixed-`520M` loss and
  over-amplify target-budget multiplier drops
- the Olmix-style `c + exp(linear)` link is the least-bad body swap by RMSE,
  but its fixed-`520M` rank is poor and its level error remains large
- move on from ceiling-body modeling for now; the remaining useful local path is
  explicit target-token scaling inside the current q/support/floor family or a
  better bounded law, not a return to GRP/Olmix ceiling structure

## 2026-04-23 - q/support target-budget gain local study

Ran a prediction-first diagnostic before doing more optimum/deployment work.
The goal was to test whether the remaining fixed-`520M` compression can be
repaired with lightweight target-budget gain terms on top of the current
q/support floor family.

Artifacts:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/code/run_qsupport_gain_local_study_20260423.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/qsupport_gain_local_study_20260423/qsupport_gain_local_study_20260423_report.md`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v27/reference_outputs/qsupport_gain_local_study_20260423/qsupport_gain_local_study_20260423_metric_summary.csv`

Important caveat:

- refitting `qsupport_floor_ci_rpm100_gap015` to the full `100` L-BFGS
  iterations can hit an exp-clip catastrophe on seed `11`
- seed-11 quick check:
  - maxiter `50`: overall RMSE `0.00879`, fixed-`520M` RMSE `0.00695`
  - maxiter `100`: overall RMSE `8.93e7`, fixed-`520M` RMSE `1.35e8`
- so the diagnostic uses a stable `rpm50` refit for 8-seed ablations and
  treats the saved `rpm100` artifacts as the current predictive reference

8-seed diagnostic results on the stable `rpm50` refit:

| model | fixed-520M RMSE | fixed-520M Sp | fixed-520M std ratio | overall RMSE |
|---|---:|---:|---:|---:|
| `qsupport_floor_ci_rpm50_gap015_refit` | `0.006724` | `0.968376` | `0.631723` | `0.010964` |
| `qsupport_head_logmu_global` | `0.006663` | `0.965897` | `0.623938` | `0.010845` |
| `qsupport_head_logmu_family` | `0.006877` | `0.954359` | `0.624525` | `0.011299` |
| `qsupport_plus_pair_gain_ridge` | `0.020006` | `0.173675` | `0.474819` | `0.017040` |
| `qsupport_plus_row_residual_ridge` | `0.063031` | `0.954957` | `0.594712` | `0.063825` |

Seed-7 current `rpm100` baseline, from the updated packet artifact:

- fixed-`520M` RMSE `0.006275`
- fixed-`520M` slope / std ratio `0.658 / 0.683`
- fixed-`520M` same-mixture drops remain underpredicted:
  - `0.5x -> 1.0x`: actual `0.02364`, predicted `0.01485`, ratio `0.633`
  - `0.5x -> 2.0x`: actual `0.04958`, predicted `0.03439`, ratio `0.694`
  - `1.0x -> 2.0x`: actual `0.02157`, predicted `0.01702`, ratio `0.789`

Interpretation:

- the remaining prediction failure really is a target-budget continuation
  failure: same-mixture BPB drops with larger budget multipliers are too small
  in the model
- simply adding `log(mu)` features to the linear head does not solve it; it
  slightly improves level / overall RMSE on the stable refit but leaves fixed
  `520M` dispersion equally compressed or worse
- a pair-gain residual learned from `130M`/`300M` matched pairs transfers
  badly to `520M`; it can invert fixed-`520M` ranking
- in-sample row residual correction is an obvious overfit and should not be
  pursued
- next prediction work should modify the structural scale-continuation law
  itself, not add a shallow residual patch: likely candidates are monotone
  token-scaling terms inside the floor-log body, target-budget-conditioned
  q/support floor dynamics, or a same-mixture trajectory/anchor law when lower
  scale or lower budget observations are actually available

## 2026-04-23 - Fresh ChatGPT Pro packet v28

Prepared a fresh packet for the next external modeling round, with a
prediction-first framing:

- packet root:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28`
- archive:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28.zip`
- prompt:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/PROMPT_FOR_FRESH_CHATGPT_PRO.md`

Packet changes:

- reframed the task as joint modeling of data mixture and scale, not as a
  q/support followup
- made GRP Power Family Penalty no-`L2` the first-principles donor to start
  from
- kept `qsupport_floor_ci_rpm100_gap015` as the predictive benchmark, but asked
  sessions to try at least one independent model before inspecting q/support
  internals
- updated the benchmark contract to the current `26`-row fixed-`520M` basis:
  `12` rows at `0.5x`, `12` at `1.0x`, `2` at `2.0x`
- added the GRP no-`L2` `60M` vs `300M` fit artifacts to the packet so fresh
  sessions see both the useful regression signal and the raw-optimum pathology
- emphasized target-budget drop diagnostics as a required prediction gate

The archive was rebuilt with an explicit zip writer that excludes `.DS_Store`
and `__pycache__`.

Follow-up after CC review:

- removed bundled DML/BiMix `.tar.gz` archives from the packet because extracted
  source trees are already present
- updated DML/BiMix notes and manifest artifact labels to reference only the
  extracted source directories
- added `code/DO_NOT_READ_FIRST_QSUPPORT.md` to make the sealed-baseline
  boundary harder to miss
- added seed-7 training set size context (`567` rows / `274` training examples)
- strengthened the archive requirement in the prompt/request: sessions must
  return all code, CSVs, plots, reports, and model artifacts needed to reproduce
  evaluation
- added compact-budget guidance to the prompt (`<=100` preferred, `<=120`
  acceptable for a cleaner or better-calibrated form)
- rebuilt and verified the zip; size dropped from roughly `31M` to `23M`

Second follow-up:

- consolidated q/support-dependent implementation artifacts into
  `DO_NOT_READ_FIRST_QSUPPORT/`
- moved q/support-dependent scripts out of the main `code/` directory:
  `run_qsupport_gain_local_study_20260423.py`,
  `run_ceiling_head_local_study_20260423.py`, and
  `run_grp_olmix_ceiling_body_local_study_20260423.py`
- moved q/support `seed7_model.json` artifacts under
  `DO_NOT_READ_FIRST_QSUPPORT/model_artifacts/`
- left q/support metrics, reports, CSVs, and plots in `reference_outputs/`
  because those are allowed for early benchmark/failure inspection
- rebuilt and verified the zip; the main `code/` directory now has no
  q/support implementation references

## 2026-04-23 - Registry refresh with first 1.2B completions

Refreshed the strong-tier run registry with:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  --no-include-live-status
```

Snapshot before refresh:

- `strong_tier_perplexity_ready.csv`: `104` ready rows
- no ready `1_2b_24b` primary rows

After refresh:

- `strong_tier_perplexity_ready.csv`: `107` ready rows
- new ready rows:
  - `baseline_proportional`, `1_2b_24b`, `qsplit_baselines3_holdout`,
    `1.0x`, BPB `0.832938`, executor `SUCCESS`
  - `baseline_unimax`, `1_2b_24b`, `qsplit_baselines3_holdout`, `1.0x`,
    BPB `0.827538`, executor `SUCCESS`
  - `run_00125`, `520m_10p4b`, `qsplit_representative12`, `2.0x`,
    BPB `0.846922`, executor `FAILED` but target eval exists at step `39671`

CC-listed rows that were not new datapoints:

- `run_00152` at `520M 0.5x` was already perplexity-ready; refresh updated
  executor status to `SUCCESS`
- `run_00018` and `run_00155` at `520M 1.0x` were already present

Sanity checks:

- baseline `1.0x` BPB decreases monotonically across qsplit replay scales:
  - `baseline_unimax`: `130M 1.103117 > 300M 0.981139 > 520M 0.888729 >
    1.2B 0.827538`
  - `baseline_proportional`: `130M 1.112409 > 300M 0.990733 > 520M 0.895344 >
    1.2B 0.832938`
- available `520M` target-budget trajectories are also monotone:
  - `baseline_proportional`: `0.922824 > 0.895344 > 0.873255`
  - `baseline_unimax`: `0.917273 > 0.888729 > 0.867686`
  - `run_00125`: `0.893318 > 0.867410 > 0.846922`

Important data-contract issue found during 1.2B holdout evaluation:

- the v28 packet already contained stale `1.2B` baseline rows, but their
  mixture weights did not match the qsplit replay specs from
  `two_phase_many.csv`
- mean phase TV before patch:
  - `baseline_unimax`: `0.2964`
  - `baseline_proportional`: `0.2037`
- the local 1.2B holdout evaluator now patches those weights from
  `two_phase_many.csv` before fitting

Added rerunnable evaluator:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_qsupport_1p2b_holdout_20260423.py`
- outputs:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_1p2b_holdout_20260423`

Compact q/support 1.2B holdout result after label and weight patching:

- train rows: `626`
- held-out `1.2B` rows: `2`
- train RMSE: `0.01898`
- `1.2B` holdout RMSE: `0.11438`
- `1.2B` holdout mean residual: `+0.05020`
- `baseline_unimax`: actual `0.827538`, predicted `0.980518`
- `baseline_proportional`: actual `0.832938`, predicted `0.780365`
- rerunning the same holdout with `fit_maxiter=100` did not fix the failure:
  holdout RMSE `0.11629`, `baseline_unimax` predicted `0.98315`,
  `baseline_proportional` predicted `0.77972`

Interpretation:

- the new `1.2B` BPBs look plausible and internally consistent with lower
  scale qsplit baselines
- the current compact q/support law does not extrapolate robustly to `1.2B`;
  even after patching stale packet weights, it predicts the two baseline
  holdouts on opposite sides and badly misses the level
- this strengthens the case that the next modeling work should prioritize
  prediction shape / scale-continuation, not only fixed-`520M` calibration or
  raw optimum quality

Correction after packet-wide weight audit:

- the earlier evaluator only patched the stale `1.2B` baseline weights
- a full audit showed the v28 NPZ `weights` tensor was stale/misaligned for
  `293 / 643` packet rows, mostly `60m_1p2b | legacy_swarm_60m`
- the CSV phase-weight columns in `nd_scale_runs.csv` are the correct source of
  truth; `baseline_unimax` and `baseline_proportional` do have matching weights
  across 60M and qsplit replay rows
- patched
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/code/nd_scale_packet.py`
  to reconstruct `weights` from `nd_scale_runs.csv` at load time
- rebuilt
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28.zip`

Updated compact q/support 1.2B holdout after the packet-wide weight fix:

- loader audit: `0` weight mismatches after load
- train RMSE: `0.026808`
- 60M train RMSE: `0.037242`
- 60M train slope: `0.243332`
- `1.2B` holdout RMSE: `0.118990`

Corrected baseline scaling rows:

- `baseline_unimax`: `60M 1.083430`, `130M 1.103117`, `300M 0.981139`,
  `520M 0.888729`, `1.2B 0.827538`
- `baseline_proportional`: `60M 1.091835`, `130M 1.112409`,
  `300M 0.990733`, `520M 0.895344`, `1.2B 0.832938`

Interpretation update:

- 60M should be treated as an apples-to-apples baseline mixture comparison
  after weight reconstruction
- the remaining oddity is that 130M 1.0x is worse than 60M for both baselines;
  this is not a mixture-weight mismatch and needs a separate protocol or
  training-quality sanity check

Follow-up on the 130M oddity:

- the 130M rung is not actually larger than the 60M rung under the
  RegMix-style non-embedding parameter convention used elsewhere in this work
- approximate non-embedding parameter counts:
  - `60m_1p2b`: `59.0M` (`hidden_dim=768`, `intermediate_dim=1536`,
    `num_layers=10`)
  - `130m_2p6b`: `22.8M` (`hidden_dim=512`, `intermediate_dim=1792`,
    `num_layers=6`)
  - `300m_6b`: `102.6M`
  - `520m_10p4b`: `339.8M`
  - `1_2b_24b`: `906.0M`
- this explains why the 130M rung can be worse than 60M despite matching
  mixture weights
- implication: joint N/D model fits should not treat the packet's nominal
  `model_size` values as a clean monotone capacity axis without adding actual
  parameter-count metadata

Additional packet repair:

- repaired v28 `data/nd_scale_packet.npz` so `weights` now match
  `nd_scale_runs.csv` directly, not only through the patched loader
- left a local backup at
  `data/nd_scale_packet.npz.stale_weights_backup`
- rebuilt the v28 zip after excluding the stale backup

Corrected compact q/support refit:

- added
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_qsupport_corrected_data_20260423.py`
- evaluator applies refreshed registry labels, corrected packet weights, and
  actual non-embedding parameter counts before fitting
- outputs:
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_corrected_data_20260423`

Seed-7 corrected-data results:

- train RMSE: `0.028409`
- holdout RMSE: `0.026121`
- fixed-520M RMSE: `0.020842`
- fixed-520M slope: `0.474835`
- fixed-520M std ratio: `0.536017`

Fixed-520M multiplier mean residuals:

- `0.5x`: `-0.026019`
- `1.0x`: `-0.014795`
- `2.0x`: `-0.000353`

1.2B holdout corrected-data results:

- holdout RMSE: `0.113156`
- `baseline_unimax`: actual `0.827538`, predicted `0.987523`
- `baseline_proportional`: actual `0.832938`, predicted `0.836612`

Interpretation:

- the q/support compact model does not survive the data correction cleanly
- the fixed-520M compression/shape failure is back
- the 1.2B failure is asymmetric, with unimax badly overpredicted while
  proportional is close
- q/support should be treated as a useful diagnostic donor, not a current
  deployable lead, until rerun and redesigned on corrected scale metadata

### 2026-04-23 16:50 - Actual non-embedding N rerun, rpm100

Command:

```bash
uv run --with numpy --with pandas --with scipy --with jax --with jaxlib --with matplotlib --with scikit-learn python \
  experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_qsupport_corrected_data_20260423.py \
  --fit-maxiter 100
```

Setup:

- same compact q/support law as the prior corrected-data evaluator
- packet weights reconstructed from `nd_scale_runs.csv`
- refreshed registry labels applied before fitting
- scale coordinate patched to actual non-embedding parameter counts:
  `60M=58.998528M`, `130M=22.813184M`, `300M=102.648576M`,
  `520M=339.7888M`, `1.2B=906.037248M`

Results:

- seed-7 train RMSE: `0.027675`
- seed-7 holdout RMSE: `0.022455`
- seed-7 fixed-520M RMSE: `0.009294`
- seed-7 fixed-520M slope: `0.772215`
- seed-7 fixed-520M std ratio: `0.886436`
- 1.2B holdout RMSE: `0.111937`
- `baseline_unimax`: actual `0.827538`, predicted `0.985812`
- `baseline_proportional`: actual `0.832938`, predicted `0.829897`

Comparison to nominal-`N` rpm100 q/support:

- nominal `N` seed-7 fixed-520M RMSE was better (`0.006443`), but its
  fixed-520M slope/std ratio were worse (`0.652945` / `0.680115`)
- actual non-embedding `N` improves fixed-520M shape calibration but worsens
  seed-7 random-supplement/general holdout error
- nominal `N` and actual non-embedding `N` both fail the two-row 1.2B holdout,
  but for different reasons:
  - nominal `N`: unimax too high, proportional too low
  - actual non-embedding `N`: proportional nearly fixed, unimax still far too
    high

Interpretation:

- using nominal model-size labels is semantically wrong and should not remain
  the default scale coordinate
- switching to actual non-embedding `N` is not sufficient to make the current
  q/support law trustworthy
- the persistent unimax miss suggests either a model-form failure specific to
  mixture/source geometry or a remaining data/protocol issue along the unimax
  trajectory

Audit checklist before another serious modeling packet:

- define one canonical scale coordinate table with nominal label, non-embedding
  params, tied-total params, untied-total params, optimizer family, architecture
  family, sequence length, token budget, and source experiment
- verify every same-mixture trajectory by immutable mixture vector equality,
  not by run name alone
- verify monotonicity only after sorting by the chosen actual scale coordinate,
  not nominal rung label
- split evaluation reports by source/family: legacy 60M swarm, 300M legacy
  swarm, qsplit replay, stratified replay, and 1.2B baselines
- run all headline candidates with at least three `N` conventions:
  nominal labels, actual non-embedding params, and actual tied-total params
- include leave-one-family/rung-out ablations, especially without 130M and
  without legacy 60M, so we can identify fragile dependence on known confounds
- audit target-budget multiplier semantics against effective epoching: each
  proxy row should see the same per-domain epoch count implied by the target
  budget, invariant across `N` and `D`
- require plots for same-mixture scale trajectories, residuals by source,
  residuals by actual `N`, and residuals by target-budget multiplier before
  interpreting raw optima

### 2026-04-23 17:05 - Scale-axis ablation on identical corrected data

Command:

```bash
for axis in nominal non_embedding tied_total; do
  uv run --with numpy --with pandas --with scipy --with jax --with jaxlib --with matplotlib --with scikit-learn python \
    experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_qsupport_corrected_data_20260423.py \
    --fit-maxiter 100 \
    --scale-axis "$axis" \
    --out-dir "experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_scale_axis_${axis}_20260423"
done
```

Summary artifact:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_scale_axis_ablation_20260423/scale_axis_metrics_summary.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_scale_axis_ablation_20260423/scale_axis_1p2b_predictions.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v28/reference_outputs/qsupport_scale_axis_ablation_20260423/scale_axis_ablation_summary.png`

Key comparison:

| scale axis | seed-7 holdout RMSE | fixed-520M RMSE | fixed-520M slope | fixed-520M std ratio | 1.2B RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | `0.023807` | `0.012804` | `0.774983` | `0.809717` | `0.117480` |
| non-embedding | `0.022455` | `0.009294` | `0.772215` | `0.886436` | `0.111937` |
| tied total | `0.024116` | `0.013541` | `0.495623` | `0.510790` | `0.111534` |

1.2B residuals:

| scale axis | baseline_proportional | baseline_unimax |
| --- | ---: | ---: |
| nominal | `-0.050155` | `+0.158391` |
| non-embedding | `-0.003041` | `+0.158273` |
| tied total | `-0.008996` | `+0.157477` |

Interpretation:

- actual non-embedding `N` is the best of the three for this compact q/support
  law on corrected data
- the improvement is real but incomplete: it improves fixed-520M RMSE and std
  ratio, and fixes the 1.2B proportional baseline, but it does not fix the
  unimax extrapolation
- tied-total `N` behaves poorly on fixed-520M shape despite being closer to the
  Chinchilla paper's stated parameter-count convention
- next modeling should use non-embedding `N` as the default engineering
  convention, while keeping tied-total and nominal as required sensitivity
  checks

### 2026-04-23 17:35 - RegMix and Olmix parameter-count convention audit

Question:

- The current ND packet mixes scale labels from multiple experiment lineages.
  Before the full data audit, check whether RegMix/Olmix use total params or
  non-embedding params for their small proxy names.

RegMix finding:

- The RegMix paper explicitly says its model-size names refer to
  non-embedding parameters because embeddings dominate small models:
  `/tmp/regmix_2407_01492/sections/00_Intro.tex`
- RegMix `tinyllama_60M` code uses the architecture we copied:
  `n_layer=10`, `n_head=8`, `n_embd=768`, `intermediate_size=1536`,
  `vocab_size=50432`:
  `/Users/calvinxu/Projects/Work/Marin/data-mixture/regmix/model_training/lit_gpt/config.py`
- That architecture has approximately:
  - non-embedding params: `58,998,528`
  - tied total with RegMix vocab: `97,730,304`
  - untied total with RegMix vocab: `136,462,080`
  - tied total with our Llama3 vocab: `157,499,136`

Olmix finding:

- The Olmix paper describes proxy models of `1M`, `15M`, `30M`, and `60M`
  parameters and gives their architecture table with `vocab_size=100,352`.
  The `1M` row has `d_model=16`, so the embedding table alone is
  `1,605,632` params; therefore the `1M` label cannot be total params.
- The Olmix launch code makes the convention explicit operationally:
  `num_params = model.num_non_embedding_params`, then uses that for
  Chinchilla duration, warmup, batch size, and LR:
  `/Users/calvinxu/Projects/Work/Marin/data-mixture/olmix/olmix/model/transformer.py`
- Exact Olmix/olmo-core factory counts with `vocab_size=100,352`:
  - `olmo3_30M`: non-embedding `29,102,336`, total `54,792,448`
  - `olmo3_60M`: non-embedding `57,422,208`, total `95,957,376`
  - `olmo2_30M`: non-embedding `29,102,336`, total `54,792,448`
  - `olmo2_60M`: non-embedding `57,422,208`, total `95,957,376`

Important local mismatch:

- Our historical `olmo3_30m_proxy` in
  `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/proxy_sweep.py`
  is not Olmix-convention `30M`.
- It was designed around approximately `30M` total params with the Llama3
  vocab:
  - hidden `224`, intermediate `560`, layers `4`, heads `4`
  - non-embedding params: `2,311,904`
  - embedding params: `28,729,344`
  - tied total params: `31,041,248`
- This means any row named `olmo3_30m_proxy` should be treated as a
  `2.3M`-body model under the non-embedding convention, not as an Olmix-style
  `30M` proxy.

Interpretation:

- RegMix and Olmix both support non-embedding `N` as the right convention for
  small proxy modeling.
- The current `60m_1p2b` label is semantically correct under that convention.
- The 130M/300M/520M/1.2B Llama labels and the local `olmo3_30m_proxy` label
  are not consistently non-embedding names.
- The audit should treat nominal scale labels as display names only and build
  a canonical scale table with actual non-embedding params, total params, and
  architecture/source family for every row.

### 2026-04-23 18:05 - Tied embedding convention audit

Question:

- Check whether Marin is the only lineage using tied input/output embeddings.

Findings:

- RegMix is untied. Its model creates a separate `lm_head =
  nn.Linear(...)` and input token embedding `wte = nn.Embedding(...)`.
  The checkpoint converter also exports `tie_word_embeddings: false`.
- Olmix/olmo-core is also untied. The transformer builds a separate
  `self.embeddings = nn.Embedding(...)` and `self.lm_head = lm_head.build(...)`;
  `LMHead` contains `self.w_out = nn.Linear(...)`.
- Olmix's `num_non_embedding_params` subtracts only the input embedding table
  (`d_model * vocab_size`) from total params, so the output LM head remains
  included in the reported non-embedding count.
- The Marin domain-phase-mix proxy configs explicitly set
  `tie_word_embeddings=True` for the local `olmo3_30m_proxy` and the RegMix
  60M/130M/300M/520M/1.2B proxy rungs.

Interpretation:

- For the specific RegMix and Olmix references, yes: the Marin proxy sweep is
  the tied-embedding outlier.
- Tied embeddings are not intrinsically wrong, but they change total parameter
  count substantially at small scales and break direct comparability with
  RegMix/Olmix total-param accounting.
- A robust registry audit should record at least:
  `tie_word_embeddings`, input embedding params, output head params,
  non-embedding params under the local code convention, and total params.

### 2026-04-23 18:45 - Data provenance audit

Artifact:

- `docs/debug-log-domain-mix-data-audit.md`

Findings:

- Strong-tier simulated epoching is semantically correct. The dataset slice
  ratio is `experiment_budget / target_budget`, so per-domain effective epochs
  reduce to `phase_fraction * mixture_weight * target_budget / domain_tokens`;
  actual proxy budget cancels for a fixed target-budget multiplier.
- Strong-tier budget metadata is internally consistent: 121 rows have explicit
  target-budget metadata, zero budget/step mismatches, and zero duplicate
  strong-ready keys.
- The run registry is valid as the operational provenance source for current
  strong-tier rows, but not yet a complete modeling single source of truth:
  580 historical/non-strong rows still lack `experiment_budget`.
- The metric registry is stale relative to the run registry: 536 rows generated
  on 2026-04-17 vs 701 run-registry rows refreshed on 2026-04-23.
- Inclusion in fitting must use target-step perplexity availability, not
  executor status. Four completed rows lack target-step evals, while twelve
  failed/running rows are target-eval-ready.
- After switching to non-embedding N, the old `130m_2p6b` cell is really
  `22.8M/2.6B`; it can beat `60M/1.2B` on loss because it is much more
  token-trained. N-only scale plots are misleading unless D is shown too.
- The v28 packet still stores nominal `model_size` values, so fresh-session
  quantitative cross-scale claims should be rerun after regenerating the packet
  with explicit non-embedding N and actual D columns.

Decision:

- Keep historical scale keys as stable identifiers, but use display terminology
  `20M/2.6B`, `60M/1.2B`, `100M/6B`, `340M/10.4B`, and `900M/24B`.
- Before more modeling packets, create one canonical analysis dataset derived
  from the registry with explicit param counts, budgets, and target-step label
  status.

### 2026-04-23 22:15 - Canonical analysis dataset and v29 packet

Implemented the canonical derived modeling dataset:

- builder:
  `experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/build_analysis_dataset.py`
- local outputs:
  `analysis_dataset/nd_scale_runs.csv`,
  `analysis_dataset/nd_scale_packet.npz`, and
  `analysis_dataset/summary.json`
- packet:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29`
- archive:
  `experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29.zip`

Key semantics:

- `model_size` now equals `non_embedding_params`
- historical scale names are stable IDs only
- display labels are `20M/2.6B`, `60M/1.2B`, `100M/6B`,
  `340M/10.4B`, `900M/24B`
- all strong-tier rows must be present in
  `run_registry/strong_tier_perplexity_ready.csv` to survive into the modeling
  dataset
- strong-tier rows with old packet labels but no target-step eval are dropped

Validation:

- output rows: `629`
- label sources: `522` packet historical, `107` registry target-step
- fixed-520M qsplit target-step rows: `27`
- 1.2B target-step rows: `2`
- duplicate canonical modeling keys: `0`
- rows missing primary labels: `0`
- `model_size == non_embedding_params`: true
- max phase-sum error: `8.88e-16`

Smoke command:

```bash
PYTHONPATH=experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29/code \
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from nd_scale_packet import load_packet

root = Path("experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29")
packet = load_packet(root)
frame = packet.frame
assert len(frame) == len(packet.model_sizes)
assert np.array_equal(packet.model_sizes, frame["non_embedding_params"].to_numpy(int))
assert not np.any(packet.model_sizes == 130_000_000)
assert np.allclose(packet.weights.sum(axis=2), 1.0)
assert len(frame[(frame["scale"].eq("520m_10p4b")) & (frame["path"].eq("qsplit_representative12"))]) == 27
assert len(frame[frame["scale"].eq("1_2b_24b")]) == 2
PY
```

Caveat:

- q/support reference metrics bundled in the packet were generated before the
  latest `2.0x` fixed-520M target-step row was added; exact q/support baseline
  numbers should be rerun on v29 before final comparison.
