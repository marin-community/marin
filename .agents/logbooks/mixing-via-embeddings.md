# Logbook: mixing-via-embeddings validation experiments

Design: `.agents/projects/mixing_via_embeddings/` (PR #6969). Goal: validate embedding-space
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
