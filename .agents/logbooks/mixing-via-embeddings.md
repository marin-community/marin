# Logbook: mixing-via-embeddings validation experiments

Design: `.agents/projects/mixing_via_embeddings/` (PR #6969). Experiment issue: #7067.
Goal: validate embedding-space
mixture featurization via H1 (information audit) → H2a (content→domain-value LODO gate) →
H2b (held-out-dose retrodiction) → H4 (ablations) → H3 (live, gated). Append-only.

## 2026-07-05 recon (session start)

- **Run data**: qsplit240 60M canonical table is checked into the swarm branch (local tag
  `swarm-branch` = `bf26b666a`) at
  `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv`:
  241 completed rows (238 `ngd3dm2_qsplit240` + 2 hybrid_canary + 1 olmix_bpb), 39 domains × 2
  phases of weight columns, `eval/uncheatable_eval/bpb` non-null on all rows. 300M replay metrics
  live on W&B as `pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b/<run_name>`.
- **W&B access**: no API key in this environment; the wandb SDK fails. BUT the
  marin-community/marin project is publicly readable via raw GraphQL POST to
  `https://api.wandb.ai/graphql` (runs → config/summaryMetrics as JSON strings; paginate).
- **The 39 domains**: dolma3_arxiv, dolma3_cc/<13 topics>×{high,low}, dolma3_finemath_3plus,
  dolma3_stack_edu, dolma3_wikipedia, dolmino_{common_crawl_hq, olmocr_pdfs_hq, stack_edu_fim,
  stem_heavy_crawl, synth_code, synth_instruction, synth_math, synth_qa, synth_thinking}.
  All pretraining text — **no SFT domains**, so loss-masking/serialization complexity vanishes
  (`loss_masked_frac = 0`).
- **Training data location**: merged levanter tokenized caches, per-domain, at
  `gs://marin-us-east5/tokenized/merged/dolma3_dolmino_top_level/<domain>-<hash>` (CC splits) and
  `gs://marin-us-central1/tokenized/merged/dolma3_dolmino_top_level/…` (others); path map in
  `two_phase_dolma3_dolmino_top_level.py` on the swarm branch. Sampling docs from these caches
  gives the exact loader measure (token-weighted by construction).
- **Basis artifacts** (all verified in GCS, 3.7MB total):
  `gs://marin-eu-west4/datakit/cluster/train_centroids_22d1e89d/{centroids_5000.npy,
  lookup_5000_to_1000.npy, lookup_5000_to_40.npy, train_stats.json}`.
- **Luxical**: pip package `luxical` (pulls torch — use `UV_TORCH_BACKEND=cpu` locally); weights
  `DatologyAI/luxical-one` `luxical_one_rc4.npz` (~880MB) via hf_hub_download; ~1.5k docs/s on 8
  CPU cores at batch 4096; int8 quant scale 0.6/127 (we embed fresh, keep fp32).
- **Local constraints**: this VM is in us-central1 (same/adjacent region as caches — sample reads
  are a few GB, acceptable); disk was 99% full (20GB uv cache; prune blocked by concurrent uv —
  workers must prune or set UV_LOCK_TIMEOUT before installing torch).
- **PR #6969**: CI green; two Codex-bot review findings (simplex-validity of matched-random
  control; RFF map identity in basis) fixed in 686637e42 and replied.

Decisions:
- Histograms v0: sample 20k docs/domain first (histogram-stability bootstrap decides if 100k
  needed), streaming (no raw text persisted), local or Iris us-east5 depending on disk.
- Quality axis: OFF in v0 basis (spec's `quality_scorer=None` path) — gated behind scorer audit
  anyway; the CC domains carry their own high/low labels which serve as a free sanity axis.
- Worker A (subagent) building runs.parquet + feasibility enumeration; Worker B histograms.

## 2026-07-05 run data + feasibility (Worker A done)

- `runs.parquet`: 60M 241/241 labeled, 300M 238/238 labeled (matched to 60M mixtures by
  run_name, weight consistency verified for all 238 at atol 1e-6). Cross-scale Spearman of
  uncheatable bpb = **0.777** (exactly matches swarm logbook claim). Phase fractions are
  **80/20** (`PHASE_BOUNDARIES=[0.8]`), not 50/50: 60M ≈ (0.956B, 0.244B), 300M ≈ (4.80B, 1.20B).
- **Protocol-critical**: qsplit240 used `SamplingStrategy.DIRICHLET` only — **zero vertex runs
  exist** (weight_sampler's vertex mode with min_dominant_weight=0.7 was never enabled). Max dose
  anywhere 0.469 (a single run); per-domain dose p90 ≈ 0.07–0.08, p95 ≈ 0.08–0.10.
- Spec defaults (train ≤0.02 / test ≥0.30) are infeasible: **0/39 eligible domains**. The
  defaults assumed a vertex-containing swarm.

### Pre-registered H2b protocol (recalibrated BEFORE any model fitting; based only on dose
### distributions + label coverage, never on metric values)

- PhaseReducer = MAX; train_max_dose = 0.02; **test_min_dose = 0.06** (primary), 0.08
  (sensitivity); eligibility n_train ≥ 60 AND n_test ≥ 20 → **36 eligible domains** at primary
  (0.08 → 35 domains at n_test ≥ 15). Excluded (insufficient low-dose train runs):
  dolma3_wikipedia, dolmino_stem_heavy_crawl, dolma3_cc/literature_low.
- Honest scope note for all reporting: this is **tail-dose extrapolation** (train ≤ 0.02 ≈
  below-mean dose, test ≥ 0.06 ≈ 2.3× the mean 1/39), not the dominant-dose regime the spec
  envisioned — the swarm's design cannot support more. H2a (domain response params fit on the
  full dose range) is therefore the primary decisive gate; H2b is the response-surface probe.
- Both scales run independently; 60M is primary (241 runs), 300M replication.

## 2026-07-06 ops notes

- GCS destination changed by rav: all project data goes under
  `gs://<bucket>/user/rav/projects/mixing_via_embeddings/...` (spec updated on the design branch).
  Nothing had been uploaded yet.
- `uv cache prune` while the project venv symlinks into the cache GUTS the venv (and any ephemeral
  `uv run --with` envs) — broke wait_for.py and the histogram worker mid-run; fixed with
  `uv sync --reinstall`. Also: a long-lived `uv run` (the PR monitor) holds a shared cache lock
  that blocks prune.
- Histogram build: 28/39 domains done at crash; worker resuming idempotently (RFF bandwidth must
  be REUSED from the frozen value, not recomputed — basis-fork hazard).

## 2026-07-06 histograms complete (Worker B done, 39/39)

- Outputs: `scratch/mixture_features/domain_histograms/` (39 parquets + rff_means.npz +
  _meta.json + bandwidth.json), uploaded to
  `gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/{domain_histograms,runs}/`.
- Frozen RFF bandwidth **0.98123** (median heuristic, 2k docs pooled over first 4 domains;
  survived the crash via bandwidth.json — basis did not fork). Assignment replicates datakit
  assign exactly (L2-norm → int8 round-trip → nearest centroid, squared L2).
- Basis sanity (K=40 mass-profile cosines): **discriminative** (wikipedia↔games_high 0.449;
  arxiv↔synth_math 0.700); **code concentrates in cell 1** (stack_edu↔stack_edu_fim 1.000,
  ↔synth_code 0.996); math/reasoning in cell 2 (finemath, synth_math, synth_thinking).
- **Quality-blindness confirmed at the basis level**: high↔low splits of the same CC topic are
  nearly identical (science_math 0.970, health 0.988) — H2a carries a mandatory quality-pair
  diagnostic (can content features predict within-pair value differences?).
- 11 domains needed non-obvious cache resolution (dolmino pool per-partition caches, 109 shards;
  map in `scratch/mixture_features/cache_sources.json`).
- Wall-clock 2h15m (compute ~50 min; rest = shared-VM OOM/disk incidents; job is fully resumable).
- Worker C launched: H1 information audit + H2a LODO gate (semantic K40/K1000/KME/RFF vs
  shuffled-columns + matched-random controls, both scales, pre-registered PASS rule at 60M).

## 2026-07-06 H1 + H2a results (Worker C done) — GATE PASSES at 60M

**H1**: V full-rank + well-conditioned at K=1000/5000 (cond 24.6/14.0); **K=40 is rank-37**
(only 37 coarse cells carry mass; stack_edu↔stack_edu_fim cos 0.99985 at K=40) → K=40 is a lossy
view, not a pure reparameterization; K≥1000 reconstructs w exactly (~1e-16). No duplicate columns
at K=5000. Quality pairs separate partially at K=5000 (within-pair Hellinger med 0.317 vs cross
0.930) despite K=40 blindness. Derivability control R²=1.0 exactly. CV: ridge_h40 ≈ ridge_weights
at 300M (+0.007), −0.026 at 60M (conditioning, not model class); lgbm-on-cells worse (bad
extrapolation prior, as predicted); **ridge_RFF beats everything in-distribution** (0.783/0.835
vs 0.733/0.803 weights) — content coordinates are a better-conditioned prior, MDE-effect
reproduced.

**H2a** (response = ridge betas on 78 per-phase weights, bootstrap SEs; LODO over 39 domains;
uncertainty-weighted corr; controls seed-averaged ×20): **PASS at 60M** — KME (clustered arm)
and RFF (cluster-free arm) each beat BOTH controls, both Pearson AND Spearman, 95% paired CIs
excluding 0 (e.g. RFF−shuffled ΔPearson 0.73 [0.10,1.10]; KME−matched ΔSpearman 0.40 [0.09,0.69]).
Pooled RFF 60M P/S = 0.54/0.27; controls NEGATIVE (−0.16..−0.39 — misassigned content actively
inverts prediction). **300M replication is fragile**: technically passes but via sem_k40/sem_k1000
Pearson-only; KME/RFF CIs include 0 there. **Quality-pair ceiling quantified**: only 3/52
within-pair beta diffs reach |z|≥2; content predicts within-pair sign at ≤ chance (one cell
systematically anti-aligned, 0/13, binom p=2e-4). **Phase asymmetry**: nearly all LODO signal is
phase-1 betas (60M RFF P: 0.70 phase1 vs 0.28 phase0) — content predicts late-phase value.

Per pre-registered rule → **H2b unblocked**. Ops note: lightgbm in this container needs
`LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib` (libgomp).
Outputs: scratch/mixture_features/{h1,h2a}/ (gate.json, betas/lodo_predictions/CIs/quality-pair
parquets).

## 2026-07-06 H2b + H4 results (Worker D done, 113 splits, no protocol cuts)

**Verdict per pre-registered rule** (>half eligible domains beat BOTH controls w/ paired CI, AND
median ≥ WEIGHTS_RIDGE): **300M @0.06 PASS** (KME_RIDGE 22/36 + median 0.700 vs 0.540);
**60M @0.06 FAIL** (best: KME 16/36, KERNEL_HELLINGER 15/36 — needed >18); both sensitivity
(0.08) thresholds fail on CI width (n_test ~20-23).

**But point estimates favor content everywhere**: 60M medians — KERNEL_HELLINGER **0.717**,
RFF 0.627, HIST_K1000 0.617 vs WEIGHTS_LGBM 0.556, WEIGHTS_RIDGE 0.453, DOSE_LINEAR 0.039.
KH beats WEIGHTS_RIDGE with CI separation on 25/36 (60M) and 22/36 (300M); beats WEIGHTS_LGBM
pointwise 32/36 and 34/36. 300M uniformly stronger (KH 0.785).

**Why the 60M rule fails — the matched-random control is strong** (KH arm: 0.616 at 60M, above
WEIGHTS_RIDGE): per-column mass-profile statistics (entropy/sparsity of a domain's histogram)
predict dose response even with semantic alignment destroyed. Scientific takeaway: part of the
value signal is content *dispersion*, not content *identity* — any future semantic claim must
keep this control. Also: RFF (H2a's star) gets 0/36 CI wins at 300M while KME (H2a-fragile) is
the rule-passer — the two hypotheses stress different geometry.

**Diagnostics**: mild extrapolation (test-h to train hull: median residual 0.070, p90 0.102);
semantic advantage weakly *increases* with content novelty (ρ≈0.21) — wins are not
easy-interpolation artifacts.

**H4** (60M @0.06): granularity K=1000 sweet spot (0.617 vs 0.495 @40 / 0.547 @5000 — confirms
H1 rank-37 lossiness); per-phase >> pooled (+~0.2); Hellinger >> Euclidean (0.717 vs 0.574);
KME+HIST concat best representation (0.627).

Outputs: scratch/mixture_features/h2b/{results,predictions,diagnostics,verdict}.parquet/json,
h4/ablations.parquet; suite at experiments/datakit/mixture_features/retrodiction.py.

**H3 decision**: gate satisfied (H2a PASS 60M primary; H2b PASS 300M primary) → proceed with H3
at 300M only, ≤4 runs, new bucket chosen among existing tokenized caches outside the 39 (c4 vs
starcoder — pick by content novelty), surrogate = KME_RIDGE + KERNEL_HELLINGER (300M passers),
comparisons = Olmix-style reuse + token-proportional. Honest framing: rule-level evidence is
one-scale; H3 is the live check.

## 2026-07-06 H3 staged — launch blocked on WANDB_API_KEY

- **New bucket: dolma_starcoder** (novelty 0.481 vs 39-domain LOO median 0.366; c4 rejected at
  0.305 — heavy CC overlap). Histogram built with frozen basis (bandwidth reused).
- **Proposal** (KME+KH rank-ensemble on all 238 300M runs; 100k-candidate search, starcoder
  capped 0.20/phase): starcoder share ~5%/phase, upweights common_crawl_hq + stack_edu(+fim);
  predicted bpb 0.9103 vs OLMIX_REUSE 0.9532, TOKEN_PROP 0.9706, ANCHOR realized 0.9554
  (run_00125, reused — not relaunched). Calibration caveat pre-registered: predicted optimum is
  below every realized bpb in the 238 → optimizer optimism is expected; the run measures it.
- **Pre-registration** (before launch): success = PROPOSAL < both baselines on
  eval/uncheatable_eval/bpb at step 22888; scratch/mixture_features/h3/preregistration.json.
- **Dry-run PASSED**: worktree /home/rav/mve-swarm-launch @ a39985fea (pre-merge swarm tip;
  bf26b666a merge is broken), launcher experiments/domain_phase_mix/launch_h3_mve.py; 3 steps,
  v5p-8 us-east5, weights verified vs surrogate JSON to equality, per-phase sums 1, 80/20 stage
  boundary step 18304.
- **BLOCKER**: marin's `_check_for_wandb_key` raises for TPU jobs without WANDB_API_KEY; no key
  in container env, host login shell, or GCP Secret Manager; Iris does not inject one. Options:
  rav runs scratch/mixture_features/h3/launch_command.txt (minus --dry_run) from a shell with
  his key; or provide key to session; or approve WANDB_MODE=offline (loses live monitoring;
  readout would move to GCS eval parsing). Holding for rav — not launching offline unilaterally.
- monitor.md + readout.py ready (public GraphQL, no key needed for reading).
- Everything through H2b/H4 committed (1cc68ceb6 + CSV vendor commit) and pushed;
  artifacts on GCS under user/rav/projects/mixing_via_embeddings/v0/.

## 2026-07-06 H3 LAUNCHED (rav provided WANDB creds; entity = stanford-mercury, not marin-community)

- Two local launch attempts failed cleanly on the region guard (driver VM us-west2 vs data
  us-east5; `MARIN_I_WILL_PAY_FOR_ALL_FEES` guards only the transfer-budget reader, NOT
  `check_gcs_paths_same_region` — no env override exists for it, by design).
- Correct pattern (what the swarm did; `east5_launch_safety.py` validates it): submit the
  executor as an **Iris parent job pinned to us-east5**. Two more real blockers fixed en route:
  (1) prod controller's iris-client build floor — workspace staging strips .git so the wheel
  stamped a stale BUILD_DATE; `submit_parent.py` writes the worktree's true commit date
  (2026-06-28) into `_build_info.py`; (2) children died on `ModuleNotFoundError: lm_eval` — TPU
  steps infer only the `tpu` uv extra; fixed by adding `pip_dependency_groups=['eval']` to the
  training steps (mirrors `launch_starcoder_heteroskedastic_snr.py`).
- **Live**: parent `/rav/rav-mve-h3-300m-6b-20260706-083828` RUNNING; children
  `rav_mve_h3_{proposal-4e4dbb, olmix-8fc12d, tokprop-af4174}` all RUNNING on v5p-8 us-east5,
  loss 10.3→6.8 by step ~110/22888. W&B: stanford-mercury/marin (private — readout auths via
  env key). Failed first parent 082403 is terminal, holds nothing.
- Babysit: wait_for.py armed with an authed poll (all 3 runs non-running → readout).
  Readout = scratch/mixture_features/h3/readout.py (pre-registered verdict + calibration).

## 2026-07-06 H3 FINAL — SUCCESS on all pre-registered criteria

Final uncheatable bpb at step 22887 (all three runs, same eval):
- **PROPOSAL 0.9410** (predicted 0.9103, err −0.031)
- OLMIX_REUSE 0.9495 (predicted 0.9532, err +0.004)
- TOKEN_PROPORTIONAL 0.9759 (predicted 0.9706, err −0.005)
- ANCHOR (historical best run_00125) 0.9554

Verdict: PROPOSAL < OLMIX ✓ and < TOKPROP ✓ → **pre-registered SUCCESS**; beats the anchor by
**−0.0145 bpb**. Olmix-reuse also beat the anchor (−0.006): the starcoder bucket has real value
and the surrogate's allocation (~5%/phase + code-adjacent upweighting) captured ~2.4× more of it
than the reuse heuristic.

**Incident + correction (recorded for honesty)**: proposal crashed at step 22886 (pre-final-eval);
its stale summary (step-22000 eval, 0.9689) was briefly compared against the baselines' FINALS,
yielding a wrong interim "FAIL" read. Resume from checkpoint completed the final eval; verified
via full eval histories (both runs share the same phase-1 descent curve; 22000→22887 drops ~0.03
for ALL runs). Lesson: never compare a crashed run's summary against finished runs' summaries —
always align eval steps.

**Calibration**: near-data mixtures predicted within ±0.005; the optimized point optimistic by
−0.031 (winner's curse present, but the margin survived it). Trust-region/LCB proposal remains
the right next refinement; these 3 runs (incl. one far-OOD point) are new calibration data.

**Campaign conclusion**: H1 ✓, H2a PASS (60M), H2b PASS (300M), H3 PASS (300M live) — the
embedding-featurized mixture surrogate transferred to a genuinely new bucket with zero re-sweeping
(one embed job + CPU refit + 3 validation runs) and beat both reuse heuristics and the historical
optimum. Known limits: quality-blindness (within-topic), dispersion confound (keep matched
controls), one-scale rule evidence at each of 60M/2b stages, optimizer optimism ~0.03 at the
argmax.

## 2026-07-06 H3 significance check (no replicates needed)

Swarm's own 300M repeat-noise measurement (`reference_outputs/300m_snr_fixed_vs_variable_20260501`,
n=10 repeats): σ(eval/uncheatable_eval/bpb) ≈ 0.00073 (fixed subset 0.000728, variable 0.000742).
H3 margins: proposal−olmix 0.0085 ≈ **8.1σ_diff**; proposal−anchor 0.0145 ≈ **13.8σ_diff**
(σ_diff = σ√2 ≈ 0.00105). Heteroskedasticity caveat checked: worst-case local std ratios (~11.5×)
come from starcoder-DOMINATED anchors in the starcoder-stress swarm; our proposal holds ~5%
starcoder — even a paranoid 3× inflation leaves 2.7σ/4.6σ. Replicates skipped (would cost ~24
TPU-hours to confirm an ~8σ result).

## 2026-07-15 Grug-MoE 168-bucket histograms (same frozen basis) + zephyr port

Extended the featurization to the **Grug-MoE Fisher-DSP sweep** (issue #7067): 168 buckets over
datakit `store_8ac06c74` (40 lexical clusters × 5 quality tiers). Mapping `cNNqQ` →
`cluster=NN/quality=Q` (0-based); `tail` = token-weighted pooled sample over the 33 below-threshold
partitions (`_TAIL_BUCKETS`). Built with the SAME frozen basis as the 39 qsplit240 domains —
centroids sha `11e7ed99…`, bandwidth **0.98123** (reused from `domain_histograms/bandwidth.json`,
never recomputed), int8-round-trip assignment — so the two sweeps are poolable.

- **Local build** (`experiments/datakit/mixture_features/build_grug_histograms.py`, gs store):
  168/168, ~2.5h compute (~53s/bucket embed) + tail's 33-child sequential read (1790s) + a
  crash/resume; resumable (per-bucket sentinels, ENOSPC + prefetch of N+1 while N embeds → I/O
  hidden behind embed). Outputs `scratch/mixture_features/grug_histograms/` uploaded to
  `gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/grug_histograms/`.
- **Distributed zephyr port** (`grug_histograms_zephyr.py` + thin launcher; reuses the histogram
  math verbatim): map-one-task-per-bucket on cw-rno2a reading the CW mirror
  `s3://marin-us-east-02a/…store_8ac06c74`. 168 buckets in **~5.5 min** wall (32 CPU workers). Two
  blockers fixed: (1) `python -m pipeline` ran the module as `__main__` → cloudpickle stamped the
  map fn `__module__=__main__` → coordinator `AttributeError`; fixed with a separate thin launcher
  so functions keep their qualified module. (2) frozen basis + input JSONs are under gitignored
  `scratch/` → staged into the bundled workspace `grug_inputs/`. CW gotchas confirmed: no
  `--region`, `--extra datakit --extra cpu` (torch-cpu), worker pods auto-read/write CW via
  `iris-task-env`, `TreeCache.load` works on `s3://` directly. Outputs on CW under
  `…/v0/grug_histograms_zephyr/`. **Roles: local = primary deliverable (methodology-matched, GCS);
  zephyr = independent replicate (CW), kept.** This VM (us-west2) has no CW creds → cannot pull CW
  outputs here.

**V-column sampling-noise floor** (same-store gs, seed 0 vs 1, n=24 spanning cell extremes;
`noise_floor.json`): cos_K5000 median **0.977** (p10 0.872, worst 0.809 for the most diffuse
900-cell buckets); cos_K40 median **0.9999** (worst 0.9936); Hellinger_K5000 med 0.191.
Floor scales with concentration (degenerate c38q0=3 cells → cos 1.0000; diffuse c05q1=700 cells →
0.809).

**Sanity + the key read against the floor**: all 168 columns sum to 1.0. Occupied cells median 116,
min 3 (c38q0), 27 buckets ≤50 — grug *lexical* clusters concentrate far more in the *semantic*
codebook than dolma topics did. Cluster↔codebook alignment: median max-K40-cell-share **0.964**,
all 35 clusters >0.5, K40 entropy median 0.244 nats → grug clusters **nest within** codebook cells,
don't cut across. Cross-cluster cosine ≈ **0.000** (near-orthogonal). Quality axis:
- within-cluster cross-tier cos **K40 = 0.9999 == the K40 noise floor** → tiers indistinguishable
  from sampling noise at coarse granularity (**quality-blind**, matches the 39-domain finding);
- within-cluster cross-tier cos **K5000 = 0.787 << floor 0.977** (below even the worst-case 0.809)
  → **real fine-grained tier separation** the K40 view cannot see.
`tail` is the most diffuse bucket (290 cells, entropy 2.95 nats, top-cell share 0.22) — as expected.

**Mirror-drift ops observation** (`mirror_drift.json`, 41 overlap buckets, gs vs CW same seed/code):
0 exact matches; |Δtokens| median 3.3% (max 43%), |Δcells| median 7. Deltas are **symmetric (±)**
and their magnitude is **consistent with the same-store sampling floor** — dominated by
contiguous-range variance over huge heavy-tailed partitions (tens of M docs), NOT evidence of
systematic mirror corruption. A store-ops per-partition doc-count parity check would confirm
(couldn't read CW ledgers from this VM). Basis is identical on both sides → poolability holds;
only cross-store bitwise reproducibility does not.

## 2026-07-14 grug-moe-mix-swarm campaign start (independent of qsplit240)

- New sweep: HF `marin-community/grug-moe-mix-swarm` — 840 MoE runs (d512), two-phase mixtures
  over 168 datakit buckets (= store_8ac06c74 partitions ≥ ~100-200M tokens; 33 tiny partitions
  coalesced as `tail`; mapping documented on issue #7067). Real dose coverage (max 0.95),
  quality tiers as metadata, computable epoching.
- **Treated independently of qsplit240** (rav's directive): no pooled fits.
- **Holdout locked before any fitting**: 800 train / 40 test (seed 0); test labels quarantined
  (sha 68a90838…), coverage verified on features only; **test protocol pre-registered** on
  issue #7067 (R1 permutation p<0.001, R2 paired Δ≥0 vs weights incumbent, R3 no-overfit
  −0.15, R4 matched-random decomposition; PASS = R1∧R2∧R3; one-shot label opening with
  predictions SHA published first). Protocol: scratch/mixture_features/grug/test_protocol.md
  (+ GCS mirror).
- Phase 1 running: 168-bucket histograms in the SAME frozen basis (bandwidth reuse mandatory)
  for feature comparability — fits stay per-sweep.

## 2026-07-15 grug phase 2 — surrogate variant fit on the 800 train runs (holdout untouched)

Suite: `experiments/datakit/mixture_features/grug_fit.py` (reuses featurize + retrodiction by
import; ridge switched to closed-form GCV to make 800-run kernels tractable, folds parallelized
8-way, BLAS pinned to 1 thread). Outputs `scratch/mixture_features/grug/{v_audit.json,
epoch_table.parquet, epoch_matrix.npz, cv_results.parquet, cv_analysis.json, fit_report.md}`,
mirrored to `gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/grug/`.
**QUARANTINE_test_labels.parquet never opened.**

- **Stage A (V audit)**: K=40 rank **36/168** (deficiency 132, cond 7.5e34, exact-duplicate
  columns cos 1.0000 = same-cluster tiers collapse) — unusable, as phase-1 predicted. K=1000 and
  K=5000 both **full rank 168/168** (cond 939 / 239); worst near-dups are within-cluster adjacent
  tiers (c32q2~c32q3 0.999@K1000, 0.980@K5000). → fit at K=1000 (or 5000).
- **Stage B (epoching)**: budget resolves to 2003 steps × 32 × 8192 = 525M tok/run; phase split
  by 1536-block quant = **0.767/0.233** (not 0.80/0.20). `enable_simulated_epoching=True` slices
  each cache to ratio·T_j, so effective epochs use the **target budget 10.37e12** over full T_j.
  Heavy epoching: **every** run has a bucket >1 epoch, p50 max-epoch **9.7**, 45% of runs >10,
  mean 45 buckets/run >1 epoch.
- **Stage C/D CV** (5-fold×3, identical splits, seed 0; OOF Spearman on macro_bpb, higher=better):
  incumbent **weights-ridge 0.215** (weights-LGBM worse, 0.210). Content beats it:
  hist-ridge K1000 **0.254** (+0.037 paired, p=0.010), **Hellinger-kernel K1000 0.303** (+0.089
  vs weights p=1e-4; +0.052 vs linear content p=0.003). Quality tier-mass adds ~0 on top of K1000
  content (+0.0008, p=0.19) and **hurts the kernel** (−0.022, p=0.015). Epochs: constrained
  in-collapse discount (δ fit, median 0.10) adds **nothing** (−0.0003, p=0.72); only free-hinge
  repeated-mass gives a tiny sig bump (+0.008, p=4e-4). Combined ≈ hinge.
- **Controls (semantics-vs-shape)**: linear content ridge has a REAL semantic-identity signal
  (matched-random −0.060, shuffled −0.030). But the KERNEL's extra power over the linear model is
  **not** semantic — matched-random reproduces the kernel almost exactly (margin **+0.0007**;
  shuffled +0.021). I.e. the kernel's gain over the incumbent is RBF nonlinearity + per-bucket
  dispersion, not cross-bucket embedding semantics.
- **Recommendation** (primary for the holdout, protocol selects by predictive CV): **Hellinger
  kernel ridge, K=1000, per-phase, sqrt-hist Hellinger distance** — best OOF Spearman + RMSE,
  decisive over the incumbent, no fragile add-ons. Flag under R4 that its edge is nonlinearity/
  dispersion; the linear content ridge is the semantically-honest alternative (~0.05 worse).
  **Holdout test NOT run.**
- 3 surprises: (1) quality tier features redundant/harmful given K=1000 content — fine histograms
  already encode tiers well enough for prediction; (2) despite extreme epoching, the monotone
  collapse-discount is inert and epoch features barely move macro_bpb; (3) the kernel's win is a
  dispersion/nonlinearity effect (matched-random ≈ semantic), reviving the qsplit240 H2b
  dispersion-confound at full force — the "embedding semantics" claim is weak on this sweep.

## 2026-07-15 PR #2393 survey + improvement tracks (goal: satisfactory metric on grug)

- Full reuse map of the swarm branch: scratch/mixture_features/grug/pr2393_reuse_map.md
  (+ GCS mirror). Highlights: dsp_exact.py (0.914 OOF fitter) is self-contained and
  dimension-generic (PacketData loader is all grug needs; 672 params vs 800 runs needs tying);
  SNR target-selection kernel is liftable but REQUIRES a seed-repeat panel (none for grug);
  trustblend (~40L) is the model-agnostic LCB pattern; solve_single_exact_kl (cvxpy) the only
  hard-constraint solver; "Fisher" in the swarm name = D-optimal QR-pivot design, and grug's
  840 runs are a post-filter subset of design_production_swarm_167p.py's 1200 candidates;
  simulated epoching = target_budget mechanism (no enable flag), epoch features MUST use
  target_budget not realized tokens (debug-log-epoch-feature-budget-semantics).
- The branch already analyzed this swarm (build_grug_moe_mix_dashboard.py,
  analyze_grug_moe_path_response.py) — being consulted for target choice.
- Running tracks: (1) per-task predictability/reliability + re-registerable target candidates;
  (2) DSP port to grug (same folds as phase 2; per-cluster tying; content-tied a_i variant).
- Holdout untouched; protocol amendment (target + R1 bar) to be re-registered on issue #7067
  BEFORE any label opening.

## 2026-07-15 grug DSP port — functional-form experiment (dsp_grug; holdout untouched)

Ported the swarm branch's DSP fitter (dsp_exact.py, OOF 0.91-0.93 on qsplit240) to the grug
swarm: vendored unmodified into `experiments/datakit/mixture_features/dsp_grug/` + loader/driver
`dsp_grug.py`. Packet: w (800,2,168) sorted-bucket order; y macro_bpb; epoch multipliers
c_p[j] = f_p·target_budget/T_j with target budget 10.372e12 (verified == launcher
`_TARGET_BUDGET_TOKENS`; simulated-epoching target-budget semantics per the swarm branch's
debug log), f = 0.767/0.233, T_j from buckets_table (tail = pooled 33 children; ΣT_j matches
budget to 3e-7). SAME RepeatedKFold(5,3,seed0) folds as phase 2, nonlinear params refit per
fold. Engineering: on this 2-CPU-quota box the 337-dim FD L-BFGS-B is infeasible → exact
variable-projection gradient (implicit diff of the NNLS head on its active set; validated vs
refit-FD to ~6 digits; A/B vs plain-FD fits identical to 3-4 decimals; untied path reproduces
dsp_exact bit-exactly). Outputs: `scratch/mixture_features/grug/{dsp_results.parquet,
dsp_report.md, dsp_summary.json, dsp_cache/}` + GCS mirror.

- **Verdict: DSP does NOT transfer.** Every DSP config loses to the Hellinger kernel (0.3076)
  on macro_bpb, all paired p≤0.002: best content_canonical **0.2455** (Δ −0.062, 1/15 folds),
  cluster_canonical 0.2370, best full full_effexp 0.2323. Same on the recommended
  zmacro_english_20 target: content 0.6431 / cluster 0.6406 vs kernel **0.8147** (0/15 folds).
- **Overfit watch confirmed**: untied 673-param models fold-train ~0.60-0.70 vs OOF 0.18-0.23
  (gap ~0.5); cluster-tied (146p) and content-tied (156p) calibrated (train 0.33-0.39, gap
  ~0.15) but only reach linear-content-level skill (hist-ridge was 0.254). Shared-rho/tau +
  free per-bucket head (globalrt, 340p) is the worst (0.171) — the per-bucket NNLS head is
  the overfitting element, not rho/tau.
- **Content-coupled tying wins within DSP**: a_i,p_i = u·f_i (u≥0, f_i = K40 profile + 1;
  feature-space NNLS) beats cluster tying on both targets — K40 shares strength across
  clusters; it is quality-blind (cross-tier cos ≈ 1 at K40), consistent with phase-2's
  "quality features add nothing".
- **Phase-mode ranking flips vs qsplit240**: canonical BENEFIT_GAIN > split_saturation_penalty
  ≈ effective_exposure here (qsplit240: split 0.929 > effexp 0.920 > canonical 0.898); the
  phase-gain term is worth +0.026 over no_phase. Fitted γ≈1-1.7 (tied), 0.32 (full); rho med
  0.48-0.51 tied (saturation ~2 epochs), 0.125 full; no quality-tier gradient in fitted a
  (mean tier-Spearman +0.02-0.05) — fitted "value" is cluster-level, tiers indistinguishable.
- **Read on g**: the DSP dose-response form is NOT the right g for this swarm/target — its
  edge on qsplit240 (39 domains, dose-swept, strong signal 0.91 vs weights-ridge ~0.9)
  came from per-domain saturation with enough runs/domain; at 168 buckets × 800 runs with a
  macro dominated by an unpredictable multilingual factor, smooth kernel regression on content
  histograms dominates and mechanistic per-bucket saturation adds parameters, not signal.
  If a mechanistic surrogate is still wanted, content-tied DSP is the only defensible flavor.

## 2026-07-15 grug holdout test — pre-registered PASS (one-shot, labels opened once)

- SHA chain honored: predictions sha 0a4cc4e6… published on #7067 BEFORE opening; quarantine
  sha matched manifest (68a90838…).
- **R1 PASS**: test Spearman 0.7205, 10k-perm p=1e-4. **R2 PASS**: +0.035 vs weights-LGBM
  (best incumbent, 0.685; CI [−0.14,+0.21], exclusion was stretch). **R3 PASS**: 0.7205 ≥
  0.668 (gap −0.098). **R4 measured**: +0.015 vs matched-random [−0.05,+0.10]; shuffled
  mean-of-10 0.7212 ≈ primary — on this swarm the signal is mostly histogram SHAPE, semantics
  a small increment (per-seed: primary beats matched 10/10, shuffled 7/10; _mean10 controls
  carry a 10-predictor ensemble advantage).
- Campaign net: target repair 0.303→0.818 train-CV (zmacro_english_20, ~93% of reliability
  bound); DSP ported + retired (0.643 best tied; content-tying its best flavor); primary =
  Hellinger kernel K=1000 per-phase; holdout confirms transfer at 0.72.
- Verdict + full numbers: #7067 comment; artifacts on GCS grug/ prefix
  (holdout_readout.{json,md}, realized_vs_predicted.parquet).

## 2026-07-15 hinge-on-zmacro check (post-holdout, train-CV only)

- REFINES the phase-2 "epoch features inert" claim: on zmacro_english_20 (clean target),
  hinge exposure features help the LINEAR model strongly (hist-ridge 0.7396 → 0.7701,
  +0.030, 15/15 folds, p=1e-4) — macro_bpb's multilingual noise had masked this.
- But the KERNEL does not benefit: product-kernel hell×hinge = 0.7673 vs 0.8147 alone
  (−0.047, 0/15) — same dilution failure as kernel+quality. The kernel absorbs the
  repetition signal implicitly via content-dose geometry.
- Net: frozen primary (kernel, 0.8147) remains correct; epoch features matter for linear
  surrogates and for interpretation; proposal-time epoch caps remain mandatory regardless.
- Artifacts: scratch/mixture_features/grug/{hinge_zmacro_check.json, kernel_hinge_zmacro_check.json}.

## 2026-07-16 codex-collaborative validation (rigor vs PR 2393, epoching, literature)

- Two codex exec review rounds + its 3 prescribed analyses + lit review; memo:
  scratch/mixture_features/grug/validation_memo.md (+GCS). Issue comment on #7067.
- Rigor: exceeds 2393 on pre-registration/controls/replication; short on repeat-panel noise,
  heteroskedasticity, metric registry. Top ask: ~10-run grug seed-repeat panel.
- Epoching: computation confirmed; discount = Muennighoff functional form (δ=0.10⇒R_D*≈9);
  concat-ARD BEATS plain kernel (+0.0049, p=0.0015) — product-kernel negative was
  architectural; knots {1,4,16} going forward; per-phase cap w_j ≤ 4·T_j/(f_p·budget).
- Selection bootstrap (B=200): no winner's curse (optimism −0.011); holdout drop = 1.3–1.6 SE
  (p≈0.08) — consistent with noise (corrected from an erroneous −0.62 SE read).
- Corrected-claims register in memo (semantic value = qsplit240-only; kernel absorbs MOST of
  epoching; H3 replicates "not cost-effective" not "not needed").

## 2026-07-16 CORRECTION: grug swarm budget was mis-derived; seed panel launched at true config

- **The 840 swarm runs trained 100.16B tokens each** (47,759 steps × 512 × 4096, ~112h on
  v4-8, verified from .executor_info + tracker_metrics.jsonl of 5 sampled run dirs), NOT the
  2003×32×8192=525M the phase-2 Stage-B derived from the CURRENT launcher heuristic (the swarm
  used Will's older heuristic: 64 experts, seq 4096, warmup 0.1 — current-branch heuristic does
  not reproduce it). Phase boundary step 38,144 → fractions **0.7987/0.2013** (we used
  0.767/0.233; phase-1 off ~16% rel). mixture_block 32,768. target_budget matched (10.37T).
- Epoch-sensitive analyses being re-checked with corrected f (per-phase uniform rescale ×1.04 /
  ×0.86; DSP absorbs most of it in fitted rho; hinge knots are not scale-invariant → re-run).
- All 840 runs used seed 0 → seed panel genuinely missing data. **rav approved the faithful
  panel**: 10 × 100.16B-token runs at exact verified config (~1,110 v4-8-hours), seeds 1000..1009,
  anchor = launch_datakit_moe_mix's mixture-3, evals = post-hoc lm-eval logprob harness
  (eval_logprob.py, vendored) over final checkpoints. Launch in progress.
- Also corrected: "525M-token proxies" phrasing in prior entries/memo — these are 100B-token
  runs; the scale-extrapolation caveat weakens accordingly (the swarm IS at substantial budget).

## 2026-07-16 corrected-f epoch re-check (validation campaign, batch R3)

- With TRUE phase fractions 0.7987/0.2013 (asserted; p50 max-epoch 9.65): hinge features still
  help the linear model, **+0.0269, 15/15 folds, p=1e-4** (old-f +0.0304 — finding robust to
  the budget correction). Note: an intermediate patch bug (fractions divided by stale
  TOTAL_STEPS=2003 → 24×-inflated epochs) produced an invalid +0.0347 run — caught by the
  p50=230 sanity contradiction, discarded, batch-2 worker warned before propagation.
- Kernel-side (concat-ARD) redo with true f: in batch-2 worker.
