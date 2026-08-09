---
topic: grug-cross-layer-expert-merging
description: Tied-from-scratch and post-hoc cross-layer routed-expert sharing in Grug MoE
author: dlwh
---

# Grug Cross-Layer Expert Merging: Task Logbook

## Scope

- Goal: Test whether Grug can train with explicit shared routed-expert banks, then whether an untied checkpoint can be converted into that architecture by functional expert matching and recovery training.
- Primary metrics: `eval/paloma/macro_loss`, throughput, peak accelerator memory, routing health, expert gradient/update norms, immediate conversion loss, affected-layer MoE NRMSE, and recovery tokens.
- Constraints: Tie only `MoEExpertMlp`; keep attention, routers and QB biases, norms and GatedNorms, shared dense MLPs, and residual blocks unique per layer. Use contemporaneous controls. Do not launch surgery before the tied-from-scratch architecture gate passes.
- Coordinating issue: https://github.com/marin-community/marin/issues/8032
- Experiment series: `GRUG-XEM`.

## Current TL;DR

- `GRUG-XEM-001` passed the d512 architecture gate on matched 1.44B-token runs in `us-central1`. Pairwise unscaled and `1/sqrt(g)` tying finished +0.02050 and +0.02203 Paloma macro loss above the 3.58622 untied control; middle-four variants finished +0.03817 and +0.04541, all inside their screening thresholds. Routing remained healthy and middle-four tying removed 50% of unique routed-expert parameters.
- `GRUG-XEM-002` completed six layers 2-3 conversion/recovery arms in `us-central1`. Spectral matching missed its initialization gate. Native no-prefit finished +0.02813 validation/+0.02814 Paloma after 250M online tokens; spectral plus per-expert prefit finished +0.02611/+0.02638. Native aggregate prefit finished within 0.00010/0.00029 of native no-prefit. Direct joint recovery needed about 50M more tokens to reach comparable quality. No arm met the required +0.02 validation target.
- `GRUG-XEM-003` ports explicit expert banks and legacy-checkpoint migration into the array-stacked June 67B-A2B implementation. The selected no-copy teacher, data caches, eval caches, output bucket, and TPU resources are all in `us-central2`; no checkpoint payload has been read outside that region.
- `GRUG-XEM-004` passed the d768 architecture screen. Middle-four unscaled tying was +0.02855 Paloma above the matched untied control, down from +0.03817 at d512, with zero overflow and 37.5% fewer unique routed-expert parameters. Effective speed was 0.849x, so the architecture is a conversion target rather than a compute-efficiency result.
- `GRUG-XEM-005` found that CE+KL supervision materially improves frozen-router Stage A, but its sole Stage-B continuation still failed the strict surgery gate: +0.03835 validation/+0.03535 Paloma at 100.14M total tokens and +0.02769/+0.02583 at 250.09M. Routing stayed healthy, so no larger surgery launched.
- `GRUG-XEM-006` completed the causal recovery-unlock matrix. Bank-only recovery matched the router-unlocked control, MLP-input-norm recovery regressed to +0.13190 validation/+0.13161 Paloma, and the untied capacity oracle improved bank-only by +0.00490/+0.00461. The capacity result missed the preregistered 0.005 signal and dominant-capacity thresholds. No shared arm passed the strict 100.14M-total-token promotion gate.
- `GRUG-XEM-007` found that a layer-3 rank-8 routed-function adapter is not a useful unlock. It improved the matched bank-only control by only 0.000028 validation and 0.000026 Paloma at 50.07M continuation tokens, capturing 0.57% of the untied-oracle advantage while missing both utility and promotion gates.
- `GRUG-XEM-008` found material direct shared-bank gradient conflict only at the 25.03M-token midpoint, not at the Stage-A start or 50.07M endpoint. The preregistered persistent-conflict gate required two of three checkpoints and returned inconclusive. No optimizer intervention or larger surgery is justified by this diagnostic.
- `GRUG-XEM-009` was canceled before TPU allocation after a 3-hour-48-minute central2 capacity wait. Neither arm produced a W&B run or checkpoint. Central1 is not a valid substitute because it lacks the exact approximately 25 TB training cache and does not offer `v4-2048`; the large architecture result remains unmeasured.
- `GRUG-XEM-011` passed the d1024 architecture and compression-normalized gates. Tying removed 45.45% of expert parameters and finished +0.03076 Paloma above the matched control, with zero overflow, all experts active, balanced shared-bank updates, and 1.66% higher throughput. The raw penalty grew by 0.00222 from d768 and effective speed was 0.812x, so tying remains an architecture target rather than a compute-speed recipe.
- The pushed research branch is `research/grug-matcher-jit` in `/tmp/marin-grug-xem-jit`. No PR exists.

## Baseline

- Date: 2026-08-05
- Code ref: `c26285a61654a9e6a9029cfdb3d018badc35d71c`
- Current d512 size reference: 6 layers, batch 32, sequence length 4096, 10,980 steps, about 1.44B tokens. The recorded Paloma value was measured under older attention and loss defaults, so it is not the matched control for this experiment.

## Hypothesis Queue

### Active

- `GRUG-XEM-H4`: One adjacent middle-layer pair can recover to the tied architecture's quality target after checkpoint surgery. Current best shared result: +0.02769 validation/+0.02583 Paloma after 250.09M online tokens, above the required +0.02 validation gate. Resume only with a new preregistered d512 shared-bank hypothesis.
- `GRUG-XEM-H15`: With two anchors at each end, one singleton core layer, and the remaining core layers tied in groups of four, the tied-from-scratch Paloma penalty remains at most +0.04 at d1280. This is the final central1 architecture-scale test in the registered progression, not an effective-speed promotion gate.

### Blocked

- `GRUG-XEM-H6`: Post-hoc 67B checkpoint surgery is blocked because d512 surgery has not passed, tied-target legacy migration is undefined, and teacher-plus-student HBM feasibility is unknown. Fresh tied-from-initialization architecture training is tracked separately under H12.
- `GRUG-XEM-H12`: The diminishing d512-to-d768 tied-architecture penalty may continue at 67B-A2B compute scale, but the registered central2 smoke was canceled before allocation. Resume only with an exact data-local capacity plan; central1 lacks the training cache and registered hardware.

### Falsified / Dead End

- `GRUG-XEM-H5`: Spectral matching missed its gate. Relative to native-only matching it improved the common assignment objective by 0.5%, Stage-A MoE loss by 1.1%, and the final combined spectral-plus-prefit recovery gap by 6-7%; none reaches the required 15%/20% margin. Keep spectral probes as diagnostics, not the production initializer.
- `GRUG-XEM-H9`: The `S/R/N/U` matrix did not identify a promotable one-factor unlock. Frozen routing was neutral, norm unlocking was harmful, and independent bank capacity improved validation/Paloma by 0.00490/0.00461, just below the fixed 0.005 signal. No tested factor explains the remaining shared-model gap by itself.
- `GRUG-XEM-H10`: The rank-8 layer-3 routed-function adapter captured only 0.57% of the untied oracle's 50M validation and Paloma advantage. It passed local-fit, routing, and throughput checks but failed its 25M and 50M utility bounds and the original promotion gate.
- `GRUG-XEM-H11`: Persistent material direct shared-bank gradient conflict was not supported. Only the 25.03M midpoint passed all five conflict criteria; the Stage-A start and 50.07M endpoint failed the aggregate-cosine and norm-balance criteria. The preregistered outcome is inconclusive, so do not launch PCGrad or an optimizer counterfactual from this result.
- `GRUG-XEM-H13`: Correspondence-free cached hard-top-4 refactorization beat the fixed-route aggregate comparator by only 3.20%, missed its held-out loss and all-expert activity gates, and overfit the cached trace. No online screen launched.

### Promoted

- `GRUG-XEM-H1`: Pairwise d512 tying is stable and within the +0.03 Paloma macro screening gate on a matched full run.
- `GRUG-XEM-H2`: Middle-four d512 tying is stable and within the +0.06 Paloma macro screening gate on a matched full run.
- `GRUG-XEM-H3`: The LR ablation did not support `1/sqrt(g)` as best for this d512 MuonH recipe; unscaled tying was slightly better at full schedule for both topologies. Keep LR scaling configurable rather than treating Jaggi's setting as a Grug default.
- `GRUG-XEM-H7`: The d768 middle-four penalty diminished to +0.02855 Paloma from +0.03817 at d512 with unscaled MuonH. The d768 tied architecture passed the +0.06 screening gate but had 0.849x effective speed.
- `GRUG-XEM-H8`: CE+KL bank-only Stage A improved the MoE-only control by 0.01837 validation and 0.01967 Paloma at 50M tokens without material local-fit regression. The later shared recovery still missed H4's strict validation gate.
- `GRUG-XEM-H14`: The d1024 tied core passed at +0.03076 Paloma with 45.45% fewer expert parameters. Its compression-normalized penalty was below +0.03460, routing and updates were healthy, and throughput was 1.66% higher; raw penalty grew slightly from d768 and effective speed was 0.812x.

## Background Research Brief

- Effort: medium.
- Stop rule: stop when primary-source checks no longer change the initial architecture or optimizer matrix.
- Date: 2026-08-05.

### Question

Which parts of the proposal are directly supported by prior tied-expert work, and which are new Grug-specific or post-hoc conversion hypotheses?

### Current Marin Context

- `experiments/grug/moe/model.py` currently stores one `MoEExpertMlp` inside every block's `MoEMLP`.
- `experiments/grug/moe/README.md` identifies d512 and d768 as the first comparison scales and warns that published d512/d768 baselines used older attention and loss defaults.
- Grug uses QB bias updates instead of an auxiliary load-balancing loss, so router/QB state must remain per-layer and should be monitored independently from tied-bank quality.
- Grug's current optimizer already classifies expert tensors separately in the AdamH path, while the current MuonH path will need a distinct tied-bank parameter group to express per-bank LR divisors.

### External Prior Art

- Jaggi's expert-tie condition shares only routed expert gate/up/down tensors while retaining per-layer attention, routers, and normalization.
- The controlled depth-32 study leaves a 2+2 prelude/coda untied and finds this anchor structure is the largest single architectural improvement.
- Production-style experiments use group sizes `g=2` and `g=4`; `g=4` reduces unique expert parameters in a group by 75% with a modest loss cost at reduced scale and near-zero loss cost in the reported 7B OLMoE run.
- Tied expert tensors use LR divided by `sqrt(g)`. A `g=4` ablation finds both `1/sqrt(g)` and `1/g` better than no scaling. Weight decay is intentionally left uncompensated.
- Cross-loop top-1 routing agreement is diagnostic rather than a success target; independent routers can reduce agreement substantially without a comparable loss change.
- Jaggi trains tied models from initialization. Functional matching, spectral probes, bijective router permutation, shared-bank prefit, and teacher-on-student-state recovery are new hypotheses in this project.

### Negative / Failed Leads

- No existing Marin tied-expert or post-hoc expert-merging implementation was found in `docs/`, `.agents/`, or `experiments/grug/`.
- The historical d512 Paloma value is not a valid sole baseline because its attention and loss defaults differ from the current recipe.
- Python object aliasing, used in the reference Hugging Face implementation, is unsuitable for this Equinox checkpoint/pytree contract; explicit single ownership is required here.

### Evidence Map

#### Claim: The first architecture should tie only routed experts

- Support: Jaggi 2026, Table 1 and Sections 1/3 define expert-tie as shared routed FFN experts with per-layer attention, routers, and norms.
- Contradictions: The paper reports modest quality cost at reduced scale; it does not establish zero-cost tying for shallow six-layer Grug.
- Directness to Marin: High architectural similarity, different framework, optimizer implementation, depth, routing/QB mechanism, data, and hardware.
- Confidence: exploratory for Grug.
- Action: implement explicit banks and matched controls.

#### Claim: Tied-expert LR scaling is necessary to interpret the architecture result

- Support: Jaggi 2026 Section 3.5 and Appendix B report unscaled `g=4` as worse than both `1/sqrt(g)` and `1/g`, and leave weight decay uncompensated.
- Contradictions: Grug uses MuonH rather than the exact optimizer stack in the paper, so the best divisor may differ.
- Directness to Marin: Medium-high; the paper explicitly motivates scaling by repeated forward use even when Muon normalizes backward updates.
- Confidence: exploratory for Grug, directly supported as an ablation requirement.
- Action: expose all three divisors in the smoke matrix.

#### Claim: Spectral probes improve post-hoc matching

- Support: no direct source found; this is a proposed functional-coverage heuristic.
- Contradictions: ordinary routed native states may already cover the relevant support, making eigen/JVP machinery unnecessary.
- Directness to Marin: unvalidated.
- Confidence: speculative.
- Action: require identity and native-only ablations and drop spectral probes unless they meet the stated gate.

### Recommended Next Experiments

#### 1. Explicit-bank untied parity

- Minimum experiment: initialize equivalent tiny untied models before and after refactor from the same key and compare forward values and gradients.
- Baseline/control: current one-bank-per-block topology.
- Expected signal: numerical parity at the existing dtype/tolerance contract and exactly one unique bank per layer.
- Falsifier: any value, gradient, checkpoint, or state-dict drift in the untied mapping.
- Cost/risk: local CPU-only tests; checkpoint and inference export are the main compatibility risks.

#### 2. d512 pairwise architecture smoke

- Minimum experiment: 500 steps for matched untied and `(0,1,1,2,2,3)` with expert LR divisors `1`, `sqrt(2)`, and `2`.
- Baseline/control: same commit, data, seed, batch 32, sequence length 4096, and current attention/loss defaults.
- Expected signal: stable loss/routing, lower unique parameter count and memory, no pathological tied-bank update concentration.
- Falsifier: divergence, routing collapse, or clear degradation that persists under scaled LR.
- Cost/risk: accelerator time; must be babysat after launch.

#### 3. d512 middle-four smoke

- Minimum experiment: 500 steps for `(0,1,1,1,1,2)` with LR divisors `sqrt(4)` and `4`, after pairwise health is established.
- Baseline/control: matched untied smoke.
- Expected signal: stable optimization and no routing/update pathology.
- Falsifier: instability or a quality gap inconsistent with the architecture gate.
- Cost/risk: accelerator time; postpone if pairwise fails.

### Hypothesis Queue Update

- Add: `GRUG-XEM-H1` through `GRUG-XEM-H5` above.
- Revise: none.
- Falsify / stop: none.
- Promote: none.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Jaggi, *Tying the Loop* v1, 2026-06-15 | paper | https://arxiv.org/abs/2606.16825 | expert-tie condition, anchors, group sizes, LR scaling, routing diagnostic | high | Primary source; no post-hoc conversion result. |
| Grug MoE README | Marin code docs | `experiments/grug/moe/README.md` | current architecture, d512/d768 gates, historical-baseline caveat | high | Current checkout. |
| Grug MoE model/optimizer/train | Marin code | `experiments/grug/moe/` | implementation seams and QB/MuonH differences | high | Current checkout at baseline hash above. |

### Handoff

- Suggested issue `Prior work` block: use the concise Jaggi relationship in the user's proposal, but state explicitly that post-hoc matching and recovery are new and that the Grug anchor counts are scale-driven approximations.
- Open questions: safest checkpoint/schema migration; exact parameter-group representation for different bank group sizes; whether cross-loop metrics should be computed online or during eval only.
- Stop reason: the primary source and local implementation scan now determine the initial architecture and optimizer matrix; further literature search would not change `GRUG-XEM-001`.

## Entry Log

### 2026-08-05 00:00 - GRUG-XEM-001 prologue and forage

- Hypothesis: Explicit single-owner expert banks plus tied-LR ablations can isolate the architecture claim without conflating it with pytree aliasing or optimizer scale.
- Commit Hash: `c26285a61654a9e6a9029cfdb3d018badc35d71c`
- Command: local `rg`/source inspection plus primary-source check of arXiv:2606.16825v1.
- Config: architecture phase only; d512 mappings `(0,1,2,3,4,5)`, `(0,1,1,2,2,3)`, `(0,1,1,1,1,2)`; LR divisors `1`, `sqrt(g)`, `g`.
- Result: Jaggi's architecture/LR claims were verified. No existing Marin tied-expert implementation was found. Historical d512 metrics are not a matched baseline under current defaults. The checkout is dirty on an unrelated branch, so branch/issue creation is deferred.
- Interpretation: proceed with a local explicit-bank refactor and behavioral tests; do not launch or publish until the diff is isolated and reviewed.
- Next action: map checkpoint/export compatibility, then implement untied parity before tied launch wiring.

### 2026-08-05 18:40 - GRUG-XEM-001 architecture implementation gate

- Hypothesis: Explicit single-owner expert banks preserve untied Grug numerics while enabling tied banks to receive gradients from several layer use sites with bank-specific learning-rate scaling.
- Commit Hash: `c26285a61654a9e6a9029cfdb3d018badc35d71c` plus the uncommitted `GRUG-XEM-001` implementation diff.
- Commands:
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run pyrefly check experiments/grug/moe/optimizer.py experiments/grug/moe/train.py experiments/grug/moe/launch.py experiments/grug/moe/launch_tied_experts.py experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py`
  - `uv run pytest -q experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py tests/test_snowball_grug_parity.py tests/test_grug_checkpointing.py`
  - `GRUG_TIED_PHASE=smoke uv run python -c 'from experiments.grug.moe.launch_tied_experts import tied_expert_runs; print([step.name for step in tied_expert_runs(version="dev")])'`
- Config: d512, sequence length 4096, batch size 32, 500 smoke steps on the full-schedule LR horizon; baseline plus pairwise and middle-four topologies, each with unscaled, `1/sqrt(g)`, and `1/g` tied-bank LR variants.
- Result: changed-file lint passed; targeted Pyrefly reported zero errors; 22 focused tests passed in 45.87 seconds. The no-launch graph produced seven distinct artifacts. The pinned Snowball parity suite confirms the explicit-bank untied topology preserves the pre-refactor forward computation.
- Interpretation: the local architecture gate is ready for an isolated snapshot and accelerator smoke. The implementation keeps routed experts single-owner in the pytree, expands them only for the established HF state-dict schema, leaves routers/QB/shared dense MLPs per layer, and records bank update plus cross-loop routing diagnostics.
- Next action: isolate the diff from the unrelated dirty checkout, snapshot it on a research branch, then launch and babysit the seven-run d512 smoke matrix.

### 2026-08-06 00:30 - GRUG-XEM-001 review hardening

- Hypothesis: The tied-gradient and routing diagnostics can be validated through observable full-model behavior without pinning incidental implementation details.
- Commit Hash: `840c40e5769841194e0c76cddd6318fa3756cd77` before the review-fix commit.
- Commands:
  - `./infra/pre-commit.py --review --agent-command='codex exec'`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py tests/test_snowball_grug_parity.py tests/test_grug_checkpointing.py`
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check experiments/grug/moe/launch.py experiments/grug/moe/launch_tied_experts.py experiments/grug/moe/optimizer.py experiments/grug/moe/train.py`
- Config: rebased `research/grug-cross-layer-expert-merging` branch on `origin/main`; same seven-run d512 smoke matrix.
- Result: 23 focused tests passed in 56.37 seconds; targeted production-file type checking reported zero errors; the no-launch graph produced the baseline and six topology/LR ablation artifacts. Advisory findings led to shared dataset construction, typed launch variants, centralized RNG/group-name constants, exact cross-loop metric assertions, and a full-Transformer tied-gradient equivalence test. The existing monolithic training entry point was not refactored because splitting unrelated trainer setup is outside this architecture experiment.
- Interpretation: the experiment snapshot is ready to push and launch. The stronger gradient test verifies that one tied bank's gradient equals the sum of the corresponding two independent bank gradients for otherwise identical full Transformer computations.
- Next action: commit the review fixes, push the research branch, create the experiment issue, and launch the smoke matrix.

### 2026-08-06 00:45 - GRUG-XEM-001 compatibility audit

- Hypothesis: Removing per-block expert ownership must not strand the repository's manual legacy-checkpoint conversion path.
- Commit Hash: `2af135eaa534af5e66aba2b9bab0699a5311b7c4` before the compatibility-fix commit.
- Commands:
  - `./infra/pre-commit.py --review --agent-command='codex exec'`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py tests/test_snowball_grug_parity.py tests/test_grug_checkpointing.py`
  - `/Users/dlwh/src/marin/.venv/bin/python -m py_compile tests/vllm/grugmoe_real_checkpoint_backend.py`
- Config: explicit `TiedExpertPhase.SMOKE` graph construction; no accelerator launch.
- Result: 23 focused tests passed in 41.48 seconds; changed-file lint and targeted production type checking passed. The legacy real-checkpoint backend now loads the historical split-expert tree without assuming current blocks own experts, then constructs an explicit-bank Transformer for execution. End-to-end Snowball logit parity replaces a test helper that duplicated production layer dispatch.
- Interpretation: all second-pass findings were fixed or narrowed. The undefined cross-loop baseline metric is explicitly documented as NaN, and reused-bank optimizer groups are documented separately from the three base groups.
- Next action: commit and push the compatibility fixes, file the tracking issue, then launch the seven-run smoke matrix.

### 2026-08-06 02:05 - GRUG-XEM-002 one-pair conversion prototype

- Hypothesis: A bijective expert assignment plus local bank distillation can initialize topology `(0,1,2,2,3,4)` without changing the source layer's router function, and constrained recovery can isolate bank error from whole-model adaptation.
- Commit Hash: `20dfc92e4fc6b88694bf6151e03eaa240bf3a13d`.
- Commands:
  - `./infra/pre-commit.py --changed-files --fix`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_expert_merge.py experiments/grug/moe/test_expert_prefit.py experiments/grug/moe/test_merge_checkpoint.py experiments/grug/moe/test_merge_recovery.py experiments/grug/moe/test_merge_storage.py tests/test_snowball_grug_parity.py tests/test_grug_checkpointing.py`
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check experiments/grug/moe/expert_merge.py experiments/grug/moe/expert_prefit.py experiments/grug/moe/merge_checkpoint.py experiments/grug/moe/merge_recovery.py experiments/grug/moe/merge_storage.py`
  - `GRUG_TIED_PHASE=smoke /Users/dlwh/src/marin/.venv/bin/python -c 'from experiments.grug.moe.launch_tied_experts import tied_expert_runs; print([step.name for step in tied_expert_runs(version="dev")])'`
- Config: one-pair layers 2-3 merge; source-to-shared permutations are explicit bijections; covariance rank 32, 16 centers, eight sensitive directions, four directions per center, radii 0.15/0.35, and native-plus-tangent cost weight 0.5. Prefit defaults to AdamW at `1e-4` for 2,000 steps with held-out early stopping. Stage A trains only the merged bank; Stage B adds the affected routers and affected-only QB updates.
- Result: 52 focused tests passed in 109.05 seconds. The Part II production files reported zero Pyrefly errors. The native checkpoint round trip reconstructed the five-bank topology and manifest, reset optimizer state, and preserved the pending QB permutation. The no-launch architecture graph still contains seven d512 smoke artifacts. No accelerator job, GCS copy, issue, push, or PR was created.
- Interpretation: the local prototype now tests assignment direction, routed-function preservation, spectral-probe bounds, weighted reservoirs, balanced prefit, checkpoint topology, teacher-on-student-state targets, and Stage A/B freezing. A positive logit-KL weight fails unless the caller supplies a fused or streaming implementation; this avoids materializing two full-vocabulary logit tensors. Merge artifact paths are checked without source-path skips, and the d512 TPU matrix is pinned to `us-central1`.
- Next action: publish the isolated branch when authorized, launch and babysit the d512 smoke in `us-central1`, then use its untied checkpoint for calibration only if the architecture gates pass.

### 2026-08-06 04:30 - GRUG-XEM-002 lint-catalog review

- Hypothesis: The one-pair prototype should expose one routed-MoE dispatch path and one QB application path so calibration and recovery cannot silently diverge from ordinary training.
- Commit Hash: `c6217e6bf25a47fef0c71262ee4dbf74a931bb8f`.
- Commands:
  - `./infra/pre-commit.py --review --no-lint-compose --lint-lane interfaces --lint-lane robustness --lint-lane cruft --lint-lane prose --lint-lane meta --agent-command='/opt/homebrew/bin/codex exec'`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_expert_merge.py experiments/grug/moe/test_expert_prefit.py experiments/grug/moe/test_merge_checkpoint.py experiments/grug/moe/test_merge_storage.py`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_merge_recovery.py`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest -q experiments/grug/moe/test_expert_tying.py experiments/grug/moe/test_optimizer.py tests/test_snowball_grug_parity.py tests/test_grug_checkpointing.py`
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check experiments/grug/moe/train.py experiments/grug/moe/expert_merge.py experiments/grug/moe/expert_prefit.py experiments/grug/moe/merge_checkpoint.py experiments/grug/moe/merge_recovery.py experiments/grug/moe/merge_storage.py experiments/grug/moe/launch_tied_experts.py`
- Result: the review reported 30 findings. The follow-up removed duplicate expert dispatch and QB loops, positional cost tuples, a conversion wrapper, a boolean split flag, repeated numerical defaults, stale parity prose, and low-value storage assertions. It also made run-ID inputs explicit and centralized the d512 label. The three focused pytest groups passed 25, 3, and 23 tests; targeted Pyrefly reported zero errors. Review log: `/tmp/marin-linter/research-grug-cross-layer-expert-merging/20260806T091647-o9zh5gru`.
- Interpretation: calibration and recovery still have explicit model traversals because they retain different intermediate state and teacher evaluations; both now consume the same `BlockCallOptions`, block trace, and routed dispatch implementation. The explicit trainer/eval block in the experiment launcher remains because it is part of the matched experiment contract. Small duplicate test configs remain local to their checkpoint schemas, and the legacy vLLM adapter retains `Any` at its dynamic historical-schema boundary.
- Next action: push the clean branch when authorized, create the experiment issue, and launch the seven-run d512 smoke in `us-central1`.

### 2026-08-06 06:45 - GRUG-XEM-002 launchable one-pair pipeline

- Hypothesis: Persisted calibration and matching artifacts plus phase-specific recovery checkpoints are sufficient to run the identity, native, spectral, and spectral-plus-prefit ablations without hidden checkpoint or cross-region dependencies.
- Commit Hash: `97213b7eb34ee9a9d1bb402616d36e95ad6a8f44` plus the uncommitted launch/runtime diff.
- Commands:
  - `./infra/pre-commit.py --changed-files --fix`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest --session-timeout=3600 -q` over the ten `experiments/grug/moe/test_*.py` files in three focused groups.
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check experiments/grug/moe/merge_artifacts.py experiments/grug/moe/merge_jobs.py experiments/grug/moe/merge_job_runtime.py experiments/grug/moe/merge_recovery.py experiments/grug/moe/merge_recovery_runtime.py experiments/grug/moe/merge_storage.py experiments/grug/moe/launch_merge_recovery.py`
- Config: calibration collects at most 2 million tokens into bf16 weighted train/held-out reservoirs for layers 2 and 3. The graph shares one calibration, matching, and spectral-prefit artifact across four conversion branches. Stage A trains for 50 million tokens. Stage B trains for 200 million tokens with exact chunked teacher-to-student KL at weight 0.1. All accelerator resources are pinned to `us-central1`, and every worker checks all material GCS inputs and outputs for one region before loading them.
- Result: 52 focused tests passed across the three groups; targeted Pyrefly reported zero errors; changed-file lint passed. A tiny real CPU integration completed conversion, one-step balanced prefit, Stage A, Stage B, checkpoint resume, teacher/student evaluation, and exact chunked KL. Matching summaries, labeled per-cluster held-out NRMSE, affected-layer MoE/block NRMSE, permutation-aware router agreement, and teacher-relative recovery tokens are persisted. No accelerator job, GCS copy, push, issue, or PR was created.
- Interpretation: the one-pair pipeline is locally launchable once a fully untied d512 teacher artifact is supplied in `us-central1`. Runtime path validation rejects mixed local/GCS inputs, non-GCS accelerator paths, and cross-region GCS dependencies. The d768 follow-up and multi-pair/four-layer surgeries remain gated on the d512 architecture and one-pair results.
- Next action: commit and review this runtime snapshot. After authorization, push the branch, create the experiment issue, launch and babysit the tied-from-scratch d512 smoke, then point `GRUG_MERGE_TEACHER` at the successful same-region untied checkpoint before dispatching the one-pair pipeline.

### 2026-08-06 07:30 - GRUG-XEM-002 runtime review fixes

- Commit Hash: `b8470c4dd9ea4224363c1a822d01d24249317813` before the follow-up fixes.
- Commands:
  - `./infra/pre-commit.py --changed-files --fix`
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest --session-timeout=3600 -q experiments/grug/moe/test_merge_artifacts.py experiments/grug/moe/test_merge_storage.py experiments/grug/moe/test_launch_merge_recovery.py experiments/grug/moe/test_merge_checkpoint.py experiments/grug/moe/test_merge_recovery.py experiments/grug/moe/test_merge_recovery_runtime.py`
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check` over the seven merge runtime and launch files.
- Result: two read-only reviews found cross-region data validation, mutable-output provenance, routed-output capture, and recovery gate-metric gaps. The follow-up validates the calibration mixture before JAX initialization, records an 8,192-token aligned trace per affected layer, rejects stale prefit/conversion checkpoints, records source commit provenance when supplied, and persists throughput, routing, capacity-overflow, Paloma, and recovery metrics. Converted checkpoints no longer store an unused full source optimizer state. Resume-at-milestone evaluation retains teacher deltas. The six focused files passed 24 tests; targeted Pyrefly reported zero errors; changed-file lint passed.
- Next action: commit the review fixes. External publication and TPU dispatch remain pending authorization and a same-region teacher artifact.

### 2026-08-06 08:00 - GRUG-XEM-001 regional launch audit

- Commit Hash: `8d40777e7f004f84c94cdf76446314722ab68636` plus the cost-control launcher diff.
- Commands:
  - `gcloud storage ls` for representative Nemotron and Paloma cache paths in `gs://marin-us-central1`.
  - `MARIN_PREFIX=gs://marin-us-central1 .venv/bin/python` no-run construction of the two smoke waves.
  - `/Users/dlwh/src/marin/.venv/bin/python -m pytest --session-timeout=3600 -q experiments/grug/moe/test_launch_tied_experts.py`
- Result: the required training and evaluation cache families resolve under the `us-central1` bucket, while all seven fixed output paths for version `2026.08.06` are unused. The launcher now accepts `GRUG_TIED_VARIANTS`, allowing a four-run baseline/pairwise wave followed by a three-run middle-four wave without changing artifact identities. Two no-launch tests and targeted Pyrefly pass. The isolated worktree's `.venv/bin/python` must be used; the main checkout interpreter imports stale Grug modules. The controller must set `MARIN_PREFIX=gs://marin-us-central1` because a local controller otherwise defaults artifact records to `/tmp/marin`.
- Next action: after dispatch authorization, run wave 1 from the isolated worktree with the regional prefix and babysit it before starting wave 2.

### 2026-08-06 10:20 - GRUG-XEM-001 d512 architecture gate

- Hypothesis: Pairwise and middle-four expert tying can complete the matched d512 schedule inside the +0.03 and +0.06 Paloma macro screening thresholds without pathological routing or expert updates.
- Commit Hash: `884b213ff4`.
- Commands: the exact four Iris resubmit commands are recorded in `scratch/20260806-0903_monitoring_state.json`, `scratch/20260806-0933_monitoring_state.json`, `scratch/20260806-0948_monitoring_state.json`, and `scratch/20260806-1000_monitoring_state.json`. Every controller and child was pinned to `us-central1` with `MARIN_PREFIX=gs://marin-us-central1`.
- Config: d512, six layers, batch 32, sequence length 4096. Smoke used 500 steps. Full runs used 10,993 steps and 1,440,874,496 tokens. The full matrix retained untied, pairwise unscaled/`1/sqrt(g)`, and middle-four unscaled/`1/sqrt(g)` variants after smoke established that `1/g` was not competitive.
- Result:

  | Variant | Paloma macro | Delta | Tokens/s |
  |---|---:|---:|---:|
  | Untied | 3.586223 | — | 360,969 |
  | Pairwise unscaled | 3.606721 | +0.020498 | 370,333 |
  | Pairwise `1/sqrt(g)` | 3.608257 | +0.022033 | 369,366 |
  | Middle-four unscaled | 3.624390 | +0.038167 | 373,910 |
  | Middle-four `1/sqrt(g)` | 3.631630 | +0.045407 | 373,994 |

  All 256 experts remained active, routing entropy stayed near 5.53, and capacity overflow remained zero. Pairwise peak HBM was about 18.2%; the middle-four Iris metric was unavailable, but both runs completed without memory failure. Middle-four uses 301,989,888 unique routed-expert parameters, 50% fewer than untied.
- Interpretation: the architecture claim passes its initial d512 gate. Low cross-loop routing agreement is expected because routers remain layer-specific. Contrary to the prior-work default, unscaled tied-expert LR was slightly better than `1/sqrt(g)` for this shallow MuonH setup, so scale must remain an empirical knob.
- Next action: use the untied full checkpoint at `gs://marin-us-central1/grug/tied_experts/d512/full/baseline/2026.08.06/checkpoints/step-10993` as the same-region surgery teacher.

### 2026-08-06 12:45 - GRUG-XEM-002 calibration and matching sharding failures

- Hypothesis: The local one-pair pipeline will carry over to a four-device TPU worker if merge-only expert dimensions remain local and dynamic expert gathers specify their output sharding.
- Commit Hash: initial launch `884b213ff4`; compact merge-mesh fix `5d1513ae1e11eac87eddd98872f8aca0be8f4dce`; explicit spectral-gather fix `1533bac743d8e67adf8ccc98b16b59cde6c97897`.
- Commands: exact sequential pipeline resubmits are recorded in `scratch/20260806-1123_monitoring_state.json`, `scratch/20260806-1211_monitoring_state.json`, and `scratch/20260806-1305_monitoring_state.json`.
- Config: layers 2-3, spectral assignment with offline prefit, v5p-8 in `us-central1`, central1 teacher/data/eval/output only.
- Result: the first calibration attempt failed because the default compact mesh assigned four devices to the `data` axis, leaving a closed-over expert down projection with local output width 128 against a 512-wide residual. The merge mesh now places non-expert devices on the replica axis and keeps merge tensors local; a four-virtual-device CPU regression covers this path. Calibration then completed and committed `gs://marin-us-central1/grug/expert_merge/d512/calibration-layers-2-3/2026.08.06`. Matching subsequently failed because dynamic per-expert gathers lacked explicit output sharding; the gather layouts are now specified and covered through spectral probe construction and expert evaluation. The second retry is running matching after one automatically recovered TPU preemption.
- Interpretation: both failures were deterministic SPMD boundary bugs rather than evidence against expert matching. No downstream artifacts launched from either failed attempt, and all material paths remained in `us-central1`.
- Next action: wait for the shared matching manifest to commit, then run identity, native, spectral, and spectral-plus-prefit branches against exactly the same calibration and cost artifacts.

### 2026-08-06 13:30 - GRUG-XEM-003 June 67B regional and topology audit

- Hypothesis: A completed recent June 67B-A2B checkpoint can serve as the first large-model surgery teacher without copying checkpoint or dataset payloads across regions.
- Commit Hash: `dd7c64baeb552ee8c8f4b6f33f2ae75e338692fd`.
- Command: local source inspection plus metadata-only GCS/Iris audit; no checkpoint tensor payload was read.
- Config: June MoE has 26 layers, hidden size 2560, expert intermediate size 1280, 256 routed experts with top-4 routing, one always-on shared dense expert, and array-stacked execution. The recommended one-pair target shares layers 12-13. The tied-from-scratch follow-up uses two input and two output anchors with mostly four-layer middle groups.
- Result: the recommended completed teacher is `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step102k-3dac46/checkpoints/step-105149/` with recorded Paloma macro 2.224. Its training cache, Paloma and uncheatable caches, output bucket, and v4-2048 resources are all available in `us-central2`. The main 10T run is still active and is not selected as the first teacher. Explicit expert-bank ownership is now implemented in the vendored June model for both unstacked and array-stacked execution; eight focused tests, the variant-contract suite, lint, and Pyrefly pass.
- Interpretation: the no-copy large-scale route is central2 compute against the completed step-105149 central2 checkpoint. The refactor changes the native checkpoint paths from per-block experts to `params.expert_banks`, so a tested legacy schema adapter is required before any large worker is launched.
- Next action: finish the local legacy-checkpoint adapter and tied-bank update scaling, then port the smallest one-pair conversion smoke. Do not dispatch it until the d512 one-pair gate passes.

### 2026-08-06 17:45 - GRUG-XEM-003 June 67B one-pair calibration and matching gate

- Hypothesis: The completed step-105149 checkpoint can be calibrated without changing its EP1 routed function, and its two middle expert banks can be matched without loading attention or duplicating dense host eigendecompositions.
- Commit Hash: `c71a00ca46` plus the uncommitted June calibration/matching diff.
- Commands:
  - `/Users/dlwh/src/marin/.venv/bin/pytest --session-timeout=100000 -q` over all five June test files, canonical expert-merge/artifact tests, and `tests/test_grug_variant_contracts.py`.
  - Four-virtual-device CPU runs of the June trace parity test on a data-axis mesh and the exploded expert evaluator on a model-axis mesh.
  - `/Users/dlwh/src/marin/.venv/bin/pyrefly check` over the canonical expert matching and June checkpoint/runtime/launch files.
  - `MARIN_PREFIX=gs://marin-us-central2 ...python -m experiments.june_tpu_67b_a2b.moe.launch_merge_recovery --stage calibration --version dev` without `--run`.
- Config: source layers 12-13, exact source `qk_mult=1.57`, sequence length 8192, batch 128, a 2 million-token target rounded to two complete batches (2,097,152 tokens), 131,072 sampled states per batch, 2,048 states per expert, and zero tolerated capacity overflow. Calibration uses v4-256 with `(replica=1,data=128,expert=1,model=1)`, preserving the teacher's EP1 backend. Expert-only matching uses v4-64 with `(replica=2,data=1,expert=1,model=16)`.
- Result: 56 tests passed and one unrelated variant-contract case skipped. Focused lint and Pyrefly passed. The trace path is one ordinary full-layer scan and matches the normal stacked forward path on a data-sharded mesh. The exploded evaluator uses direct chunked SwiGLU matmuls and supports model-axis sharding. Weighted covariance estimation uses deterministic randomized rank-32 SVD without forming a 2560-by-2560 covariance. CPU probe preparation is divided across matching hosts, gathered once, then finalized through synchronized TPU JVPs. The matching worker loads only the legacy routed-expert subtree, not attention or optimizer state. Every checkpoint, dataset cache, artifact, and TPU resource remains in `us-central2`.
- Interpretation: the invalid model-axis attention geometry, EP256 capacity clipping, EP-sharded 128-sample evaluator, duplicated dense eigendecomposition, and source-config mismatch found in review are removed. Calibration still all-gathers the sampled trace across 32 workers, but only process zero retains the roughly 10.7 GB float32 reservoirs; the v4-256 topology reduces replicated host traffic eightfold from the rejected v4-2048 design.
- Next action: snapshot this implementation, but keep the `us-central2` calibration unlaunched until the d512 one-pair surgery passes its recovery gate. Then babysit compile, HBM, strict checkpoint coverage, zero overflow, and artifact commit before launching expert-only matching.

### 2026-08-06 18:15 - GRUG-XEM-004 d768 two-anchor launch gate

- Hypothesis: Four-layer middle-bank tying behind two input and two output anchors has a smaller tied-minus-untied loss gap at d768 than the matched d512 middle-four gaps of +0.03817 unscaled and +0.04541 with `1/sqrt(g)`.
- Commit Hash: `61e4d3b66f` plus the uncommitted d768 launch diff.
- Coordinating issue: https://github.com/marin-community/marin/issues/8032, registered as a sub-issue of #4281.
- Config: budget `2.81e18` FLOPs, d768, eight layers, sequence length 4096, batch 128, 500 smoke steps, and an 8,453-step full-schedule horizon totaling 4,431,806,464 tokens. The matrix contains untied `(0,1,2,3,4,5,6,7)` and two `(0,1,2,2,2,2,3,4)` arms with unscaled and `1/sqrt(4)` shared-bank learning rates.
- Result: the no-run graph emitted exactly three configs. Twenty-one focused launch/compatibility tests passed and one contract case skipped; lint, Pyrefly, and diff checks passed. All nine training caches, 16 Paloma caches, seven Uncheatable caches, v5p-8 resources, output roots, permanent checkpoints, and temporary checkpoints resolve in `us-central1`.
- Interpretation: both tied-LR treatments are required because unscaled MuonH was slightly better at d512. The contemporaneous untied run avoids comparison against the published d768 result measured under earlier attention and loss defaults.
- Next action: snapshot and push the launch code, submit the three-run d768 smoke in `us-central1`, and assign one babysitter. Launch the three full runs only after all smoke arms show finite loss, healthy routing, and zero capacity overflow.

### 2026-08-06 18:25 - GRUG-XEM-004 d768 smoke launch

- Commit Hash: `5df0bc3ef4c9347696f8841aecf93ce5079dc697`.
- Controller: `/dlwh/grug-xem-d768-smoke-20260806`, https://iris.oa.dev/#/job/%2Fdlwh%2Fgrug-xem-d768-smoke-20260806.
- Result: the first controller submit was rejected before job creation because a `--reserve v5p-8` availability constraint combined with the controller's non-preemptible tag had no schedulable central1 group. The resubmit used the established one-CPU Marin controller shape; all three child steps retain their central1 v5p-8 `ResourceConfig`. Iris accepted the controller, and one subagent owns continuous monitoring via `scratch/20260806-1753_monitoring_state.json`.
- Next action: require all three 500-step arms to finish with finite loss, active routing, and zero overflow. If they pass, launch the matched 8,453-step matrix from the same snapshot and region.

### 2026-08-06 18:35 - GRUG-XEM-004 d768 smoke gate and full launch

- Hypothesis: The two-anchor middle-four topology remains stable at d768 and its full-schedule Paloma gap can be measured without a smoke-stage routing or memory failure.
- Commit Hash: `85b3942f920e9c8f083891c5561ddd16d4b7ce59`.
- Commands: the corrected smoke resubmit and exact full-matrix command are recorded in `scratch/20260806-1753_monitoring_state.json` and `scratch/20260806-1830-d768-full-monitoring_state.json`. Both use a one-CPU Iris controller, direct `WANDB_API_KEY` expansion, child v5p-8 resources, and `MARIN_PREFIX=gs://marin-us-central1`.
- Config: d768, eight layers, sequence length 4096, batch 128. Smoke used 500 steps and 262,144,000 tokens per arm. Full runs use 8,453 steps and 4,431,806,464 tokens per arm for untied, middle-four unscaled, and middle-four `1/sqrt(4)` variants.
- Result:

  | Smoke variant | Paloma macro | Delta | Tokens/s | Routing entropy | Overflow |
  |---|---:|---:|---:|---:|---:|
  | Untied | 4.329813 | — | 288,542 | 5.531397 | 0 |
  | Middle-four unscaled | 4.345620 | +0.015807 | 292,919 | 5.529754 | 0 |
  | Middle-four `1/sqrt(4)` | 4.353538 | +0.023725 | 292,240 | 5.532593 | 0 |

  All smoke arms finished exactly 262,144,000 tokens with finite loss. Mid-run checks found all 256 experts active in every layer, no routing collapse, and no OOM or runtime error. Permanent step-500 checkpoint metadata and artifact markers exist for all three variants. Iris accepted full controller [`/dlwh/grug-xem-d768-full-20260806`](https://iris.oa.dev/#/job/%2Fdlwh%2Fgrug-xem-d768-full-20260806); all three child TPU jobs and W&B runs reached running state with zero initial failures or preemptions.
- Interpretation: the d768 smoke passes the stability gate. At 500 steps, unscaled tying is +0.01581 Paloma and `1/sqrt(4)` is +0.02373 relative to the matched untied arm. These are startup signals; the 8,453-step result determines the scale comparison.
- Next action: babysit all three full runs through permanent checkpoint and artifact commit, then compare tied-minus-untied Paloma gaps against d512.

### 2026-08-06 23:20 - GRUG-XEM-004 d768 two-anchor full result

- Hypothesis: Four middle layers sharing one routed-expert bank behind two input and two output anchors have a smaller tied-minus-untied Paloma gap at d768 than the matched d512 middle-four result.
- Commit Hash: `85b3942f920e9c8f083891c5561ddd16d4b7ce59`.
- Commands: `/dlwh/grug-xem-d768-full-20260806` ran the three fixed variants in `us-central1`; W&B API histories supplied aligned evaluation, routing, MFU, and last-100-step throughput; Iris summaries supplied terminal state and preemption counts; `gcloud storage ls` verified each permanent checkpoint and artifact marker.
- Config: 8,453 steps and 4,431,806,464 tokens per arm. The tied arms use `(0,1,2,2,2,2,3,4)`, reducing unique routed-expert parameters by 37.5% across the model, with either unscaled or `1/sqrt(4)` shared-bank learning rate.
- Result:

  | Variant | Paloma macro | Delta | Last-100 tokens/s | Mean MFU | Routing entropy |
  |---|---:|---:|---:|---:|---:|
  | Untied | 3.255989 | — | 286,462 | 19.128% | 5.536646 |
  | Middle-four unscaled | 3.284537 | +0.028548 | 291,657 (+1.81%) | 19.411% | 5.536528 |
  | Middle-four `1/sqrt(4)` | 3.293547 | +0.037558 | 291,097 (+1.62%) | 19.373% | 5.537422 |

  Every run reached its exact token target, finished in W&B, and committed `.artifact.json`, `manifest.ocdbt`, and `checkpoints/step-8453/metadata.json`. All 256 experts were active in every layer and capacity overflow was zero. The baseline child automatically recovered from one TPU preemption; both tied children had none. All children and the controller exited successfully. Iris did not expose peak accelerator memory, but no HBM or OOM signature appeared.
- Interpretation: the d768 two-anchor architecture gate passes. Relative to d512, the tied-minus-untied gap fell from +0.038167 to +0.028548 for unscaled LR and from +0.045407 to +0.037558 for `1/sqrt(4)`. Unscaled MuonH remained better by 0.009010 Paloma, so the Jaggi scaling is not the default for the Grug follow-up. The tied arms were also about 1.6-1.8% faster over their final 100 training steps.
- Next action: retain the unscaled two-anchor d768 topology as the tied-from-scratch reference. Keep the post-hoc surgery sequence at d512 until the one-pair conversion reaches its recovery gate; then repeat the successful conversion at d768 rather than opening a new tied-architecture sweep.

### 2026-08-07 13:15 - GRUG-XEM-002 no-prefit ablation launch

- Hypothesis: Native-output or spectral expert assignment reduces the immediate one-pair conversion error and recovery cost relative to identity IDs, without requiring offline shared-bank prefit.
- Commit Hash: `75db260d33b1878c101c536dc4d25ebf8d2e8841`.
- Commands: three fixed one-CPU Iris controllers run `experiments.grug.moe.launch_merge_recovery` with `--branch identity`, `--branch native`, or `--branch spectral`, version `2026.08.06`, and `--max-concurrent 1`. Exact resubmit commands are in `scratch/20260807-1312-{identity,native,spectral}-monitoring-state.json`.
- Config: all three branches adopt the matched d512 teacher and reuse the same completed layers 2-3 calibration and matching artifacts in `gs://marin-us-central1`. Each conversion is followed by 50 million Stage-A tokens and 200 million Stage-B tokens. No branch uses prefit.
- Result: the teacher checkpoint, calibration marker, and matching marker exist. All nine branch output roots were empty before submission. Iris accepted `/dlwh/grug-xem-merge-identity-20260807`, `/dlwh/grug-xem-merge-native-20260807`, and `/dlwh/grug-xem-merge-spectral-20260807`. Initial logs explicitly pruned calibration and matching as already succeeded and dispatched only the three conversion children. No initial HBM, OOM, runtime, or dependency-region error appeared.
- Interpretation: the ablation starts from one fixed teacher/calibration/matching snapshot. Any difference among identity, native, and spectral is downstream of assignment choice; shared dependency reruns cannot confound the comparison.
- Next action: verify each conversion manifest and immediate validation/MoE error, then babysit Stage A and Stage B through permanent checkpoints and final metrics. The separately owned spectral-plus-prefit controller remains outside this monitor.

### 2026-08-07 13:45 - GRUG-XEM-002 no-prefit Stage-A comparison

- Hypothesis: Native-output or spectral assignment yields a materially better 50-million-token local-recovery trajectory than identity IDs.
- Commit Hash: launch `75db260d33b1878c101c536dc4d25ebf8d2e8841`; recovery sharding fix `31018a607f`.
- Result: all three conversions committed step-0 checkpoints and reused the same teacher, calibration, and matching artifacts. Their first Stage-A attempts failed at a jitted source-to-shared assignment gather whose replicated permutation lacked an explicit sharded result. A four-CPU-device regression reproduced the fault. The fix reshards the permutation as replicated and gives the indexed result the batch/expert layout; focused tests and required lint pass. After stop and resubmit, all three Stage-A runs committed 50-million-token artifacts.

  | Assignment | Token-0 validation delta | 50M MoE loss | 50M MoE NRMSE L2 | 50M MoE NRMSE L3 |
  |---|---:|---:|---:|---:|
  | Identity | +0.208200 | 0.230568 | 0.248910 | 0.631806 |
  | Native | +0.186542 | 0.244239 | 0.268925 | 0.645102 |
  | Spectral | +0.192147 | 0.241543 | 0.258572 | 0.645156 |

  Every aligned checkpoint retained exact teacher router top-1 and top-k agreement and zero capacity overflow. At 50 million tokens spectral improved native MoE loss by 1.1% and layer-2 MoE NRMSE by 3.9%, while layer-3 NRMSE was effectively tied and its token-0 validation spike was 3.0% worse. This does not meet the 15% spectral initialization gate.
- Interpretation: spectral matching does not clearly overtake native-only matching in the no-prefit arm. The initial native advantage is modest, and identity ultimately has the lowest local-distillation error, showing that Stage-A adaptation can reorder the initializations. Native remains the default initializer because the decision criterion prioritizes immediate conversion quality and does not justify the spectral machinery.
- Next action: keep the committed Stage-A artifacts, stop identity and spectral before Stage B, and babysit native through the 200-million-token preservation stage. The separately owned spectral-plus-prefit controller remains outside this monitor.

### 2026-08-07 15:30 - GRUG-XEM-002 native no-prefit recovery result

- Hypothesis: End-to-end preservation recovery can close the native-assignment conversion gap while routers adapt without collapse.
- Commit Hash: `31018a607f`.
- Config: native-output assignment without prefit; layers 2-3 share one bank. Stage A trained only the shared bank for 50 million tokens. Stage B trained the shared bank and affected routers for 200 million tokens with cross-entropy, logit KL weight 0.1, and MoE preservation weight 1.0.
- Result:

  | Stage-B tokens | Validation delta | Paloma macro delta | MoE NRMSE L2 | MoE NRMSE L3 | Top-1 agreement L2/L3 |
  |---:|---:|---:|---:|---:|---:|
  | 0 | +0.067233 | +0.066467 | — | — | — |
  | 25M | +0.043983 | +0.044288 | 0.305849 | 0.631412 | 0.9466 / 0.9455 |
  | 100M | +0.034079 | +0.033993 | 0.320656 | 0.626005 | 0.9390 / 0.9255 |
  | 200M | +0.028132 | +0.028137 | 0.332374 | 0.604745 | 0.9358 / 0.9034 |

  Layer-2/layer-3 block NRMSE finished at 0.086370/0.221102. Routing entropy finished at 5.5314/5.5319, top-k teacher agreement at 0.9285/0.9099, and capacity overflow at zero. The controller succeeded without application failure or preemption. The output committed `.artifact.json` and a permanent step-1526 checkpoint with `manifest.ocdbt`, `merge_manifest.json`, and `metadata.json`.
- Interpretation: recovery passes the final Paloma `+0.03` screening threshold at 200 million tokens but misses the stricter `+0.01` to `+0.02` validation-loss target and did not do so by the preferred 100-million-token horizon. Layer-3 MoE error continues downward, while layer-2 MoE error rises as its router adapts. The conversion is promising but does not pass the full single-pair surgery gate.
- Next action: compare against the separately owned spectral-plus-prefit arm if it completes. Do not advance to two middle pairs until a recovery variant closes the remaining validation gap or the gate is explicitly revised.

### 2026-08-07 16:50 - GRUG-XEM-002 spectral-plus-prefit recovery result

- Hypothesis: Spectral assignment plus offline shared-bank prefit materially reduces one-pair conversion error and reaches the untied teacher within the 100-million-token recovery gate.
- Commit Hash: `f54e315ff92084fd3f11b7d1935c7c982aa82ca3`; source teacher commit `884b213ff4`.
- Commands:
  - `iris --controller-url=http://localhost:10000 job summary /dlwh/grug-xem-merge-spectral-prefit-r4-20260806`
  - `iris --controller-url=http://localhost:10000 job summary /dlwh/grug-xem-merge-spectral-prefit-r4-20260806/grug-train-grug-xem-spectral_prefit-stage-b-d512-l2-l3`
  - `gcloud storage ls` and `gcloud storage cat` over the matching, prefit, conversion, Stage-A, and Stage-B roots under `gs://marin-us-central1/grug/expert_merge/d512/`.
  - W&B API queries by exact display name and `config.run_id` for `grug-xem-spectral_prefit-stage-b-d512-l2-l3`.
- Config: layers 2-3 share one 256-expert bank. Spectral matching used the common native-plus-tangent objective and Hungarian assignment. Offline prefit ran 2,000 AdamW steps at `1e-4`; Stage A trained only the bank for 50M tokens; Stage B trained the bank and affected routers for 200M tokens with CE, `lambda_moe=1.0`, and `lambda_KL=0.1`.
- Result:

  | Recovery point | Validation delta | Paloma macro delta | Paloma macro |
  |---|---:|---:|---:|
  | Converted, 0 tokens | +0.078243 | +0.079437 | 3.585186 |
  | Stage A, 25M | +0.056781 | +0.057416 | 3.563165 |
  | Stage A, 50M / Stage B initialization | +0.049134 | +0.049702 | 3.555451 |
  | Stage B, 25M | +0.039620 | +0.040201 | 3.545950 |
  | Stage B, 100M | +0.031165 | +0.031286 | 3.537036 |
  | Stage B, 200M | +0.026114 | +0.026380 | 3.532129 |

  The complete recovery trajectory closes 66.6% of the initial validation gap and 66.8% of the initial Paloma macro gap. At 200M Stage-B tokens, native-only matching is +0.028132 validation and +0.028137 Paloma macro; the combined spectral-plus-prefit branch leaves 7.2% and 6.2% smaller gaps. At the no-prefit Stage-A checkpoint, spectral improves native MoE loss by only 1.1%, and the common matching objective differs by 0.5%. Spectral matching therefore misses the 15% initialization gate.

  Prefit has a clear initialization effect but a modest surviving end-to-end benefit. It lowers the spectral branch's token-0 validation spike from +0.192147 without prefit to +0.078243 and its 25M Stage-A validation gap from +0.072868 to +0.056781. The final 200M comparison cannot isolate prefit from spectral assignment, and its 6-7% advantage over native is below the gate.

  Final aggregate routed-MoE NRMSE is 0.47533, with layer 2/3 at 0.33882/0.58059 and block-output NRMSE at 0.08804/0.21336. Router entropy is 5.531/5.532, capacity overflow is zero, top-1 teacher agreement is 0.9371/0.9119, and throughput is 285,452 tokens/s. Stage B trades some layer-2 fidelity for layer-3 improvement; aggregate MoE NRMSE plateaus even as validation quality continues to recover.

  Iris reports the controller and Stage-B child succeeded with exit 0, zero failures, and zero preemptions. The final checkpoint contains `metadata.json`, `manifest.ocdbt`, and `merge_manifest.json` at `gs://marin-us-central1/grug/expert_merge/d512/spectral_prefit/stage-b/2026.08.06/checkpoints/step-1526/`. Artifact markers exist for matching, prefit, conversion, Stage A, and Stage B. The custom recovery worker did not create a W&B run; exact-name and config-ID queries returned no match, so the versioned GCS evaluation and training JSON files are the canonical metric record for this run.
- Interpretation: the aggregate routed-MoE surgery is recoverable but does not pass the strict single-pair gate. It reaches the Paloma +0.03 bound only after 200M Stage-B tokens, twice the proposed budget, and never reaches the +0.01-0.02 validation target. Healthy routing rules out collapse as the limiting failure. Spectral probes do not justify their complexity; retain them only as diagnostics. Offline prefit helps the starting point and early recovery, but its final advantage is too small and confounded to support a production requirement.
- Next action: do not expand to two middle pairs. Compare the final student with the pairwise tied-from-scratch reference to separate architectural from surgery loss, then test a simpler native initializer with aggregate layer-function distillation before revisiting assignment machinery.

### 2026-08-07 18:25 - GRUG-XEM-002 native aggregate-prefit recovery result

- Hypothesis: Prefitting the native-assigned shared bank against each layer's complete routed MoE output gives a simpler and better recovery initializer than per-expert probe distillation.
- Commit Hash: `5f1461ebb1530dc00cab2ef0753d66259726efb8`; source teacher commit `884b213ff4`.
- Commands: `/dlwh/grug-xem-merge-native-aggregate-prefit-20260807` ran `experiments.grug.moe.launch_merge_recovery` with branch `native_aggregate_prefit`, version `2026.08.06`, `MARIN_PREFIX=gs://marin-us-central1`, and `--max-concurrent 1`. The fixed resubmit command is in `scratch/20260807-1646-native-aggregate-prefit-monitoring-state.json`. The launch relied on Iris secret injection; the submitted command contained no explicit credential argument.
- Config: layers 2-3 share one 256-expert bank under the native Hungarian assignment. Aggregate prefit trained the shared bank against both source layers' complete routed MoE outputs. Conversion and recovery reused the completed central1 teacher, calibration, and matching artifacts. Stage A trained only the bank for 50M tokens. Stage B trained the bank and affected routers for 200M tokens with CE, `lambda_moe=1.0`, and `lambda_KL=0.1`.
- Result: aggregate prefit early-stopped at step 700 after its best held-out loss of `0.388719` at step 200. Best held-out aggregate routed-MoE NRMSE was `0.303322` for layer 2 and `0.827910` for layer 3. Prefit, conversion, Stage A, Stage B, and the controller all succeeded with zero failures and zero preemptions.

  | Recovery point | Aggregate validation delta | Aggregate Paloma delta | Native no-prefit validation/Paloma | Spectral-prefit validation/Paloma |
  |---|---:|---:|---:|---:|
  | Converted, 0 tokens | +0.109906 | +0.111230 | +0.186542 / +0.187500 | +0.078243 / +0.079437 |
  | Stage A, 25M | +0.083993 | +0.083432 | — | +0.056781 / +0.057416 |
  | Stage A, 50M / Stage B initialization | +0.286815 | +0.272560 | +0.067233 / +0.066467 | +0.049134 / +0.049702 |
  | Stage B, 25M | +0.044332 | +0.044741 | +0.043983 / +0.044288 | +0.039620 / +0.040201 |
  | Stage B, 100M | +0.034361 | +0.034409 | +0.034079 / +0.033993 | +0.031165 / +0.031286 |
  | Stage B, 200M | +0.028228 | +0.028431 | +0.028132 / +0.028137 | +0.026114 / +0.026380 |

  Aggregate prefit reduced the converted validation spike by 41.1% relative to native no-prefit, but remained 40.5% above spectral per-expert prefit. Its Stage-A last-batch local metrics looked competitive:

  | Initializer | 50M MoE loss | MoE NRMSE L2/L3 | Block NRMSE L2/L3 |
  |---|---:|---:|---:|
  | Native aggregate prefit | 0.227605 | 0.272206 / 0.617344 | 0.070189 / 0.190765 |
  | Native no prefit | 0.244239 | 0.268925 / 0.645102 | 0.069343 / 0.222910 |
  | Spectral per-expert prefit | 0.222851 | 0.287350 / 0.602604 | 0.074094 / 0.216943 |

  The aligned Stage-B token-0 evaluation shows that the final 25M Stage-A tokens caused a rollout/generalization failure that the local training batch metrics did not detect. The audit found no checkpoint or evaluator mismatch. Stage B's `init_checkpoint_dir` is the aggregate Stage-A checkpoint root, and its startup log resolved that root to the permanent step-382 checkpoint. Recovery initialization loads every parameter and pending QB leaf with partial loading disabled, resets only optimizer and step state, and evaluates token 0 before its first update. Stage A and Stage B have identical teacher, data config, batch size, seed, assignment, prefit flag, and affected layers. The Stage-A 25M evaluation was written at `00:01:02Z`, the step-382 metadata at `00:03:00Z`, and the Stage-B token-0 evaluation at `00:06:23Z`.

  Stage B recovered from the regression within 25M tokens. At 200M, aggregate prefit was effectively tied with native no-prefit: its validation gap was larger by `0.000096` and its Paloma gap by `0.000295`. It remained behind spectral per-expert prefit by `0.002114` validation and `0.002052` Paloma. Final aggregate MoE NRMSE was `0.332497/0.606397`, block NRMSE was `0.086402/0.221530`, top-1 teacher agreement was `0.9364/0.9026`, top-k agreement was `0.9287/0.9100`, routing entropy was `5.5315/5.5319`, and capacity overflow was zero.

  Permanent `.artifact.json` markers and final checkpoint metadata exist through Stage-B step 1526. The Stage-A and Stage-B artifact provenance records contain a redaction sentinel and no sensitive environment key or value. No W&B run was created by the custom recovery worker; the versioned GCS evaluation and training JSON files are the metric record.
- Interpretation: aggregate layer-function prefit improves the native conversion starting point, but the benefit does not survive recovery. The frozen-router Stage-A objective can improve sampled local MoE and block NRMSE while making held-out language-model loss much worse. The final result gives no reason to replace spectral per-expert prefit or native no-prefit with this aggregate-prefit schedule.
- Next action: do not expand to two middle pairs. Add aligned validation at the final Stage-A checkpoint and early-stop Stage A on held-out model loss before testing another distillation objective. A shorter aggregate Stage A or a loss that constrains block rollout may prevent the 25M-to-50M regression, but neither is supported by this run yet.

### 2026-08-07 19:10 - GRUG-XEM-002 native direct-joint recovery result

- Hypothesis: Starting preservation recovery directly from the native-assigned converted checkpoint can match or beat the staged native schedule without the 50-million-token frozen-router Stage A.
- Commit Hash: `65e699cd313585e1b928bcab995d3b2c9461d6c7`; source teacher commit `884b213ff4`.
- Commands: `/dlwh/grug-xem-merge-native-joint-20260807` ran `experiments.grug.moe.launch_merge_recovery` with branch `native_joint`, version `2026.08.06`, `MARIN_PREFIX=gs://marin-us-central1`, and `--max-concurrent 1`. The fixed resubmit command and monitoring record are in `scratch/20260807-1836-native-joint-monitoring-state.json`. Iris injected credentials; the submitted command and inherited provenance command line contained no sensitive key.
- Config: layers 2-3 share one 256-expert bank under the native Hungarian assignment. Conversion reused the completed central1 teacher, calibration, and matching artifacts. Stage B initialized directly from the converted step-0 checkpoint and trained the shared bank plus affected routers for 200M tokens with CE, `lambda_moe=1.0`, and `lambda_KL=0.1`. The resolved dependency graph contained no Stage-A step.
- Result:

  | Recovery tokens | Validation delta | Paloma macro delta | Paloma macro | MoE NRMSE L2/L3 | Top-1 agreement L2/L3 |
  |---:|---:|---:|---:|---:|---:|
  | 0 | +0.186542 | +0.187500 | 3.693250 | — | — |
  | 25M | +0.064923 | +0.065779 | 3.571528 | 0.231887 / 0.813056 | 0.9437 / 0.9371 |
  | 100M | +0.043559 | +0.043602 | 3.549351 | 0.274619 / 0.727573 | 0.9349 / 0.9068 |
  | 200M | +0.033419 | +0.033358 | 3.539107 | 0.303122 / 0.664928 | 0.9324 / 0.8862 |

  The token-0 evaluation exactly matches the native no-prefit conversion, confirming the intended initializer. Direct joint recovery reached the staged native schedule's post-Stage-A quality in 25M tokens: `+0.064923/+0.065779` versus `+0.067233/+0.066467` after 50M frozen-router tokens. The staged schedule subsequently remained more token-efficient. Its 150M-total point, after 50M Stage A and 100M Stage B, was `+0.034079/+0.033993`; direct joint required 200M tokens to reach `+0.033419/+0.033358`. At their terminal horizons, direct joint trailed native staged by `0.005287` validation and `0.005221` Paloma, spectral per-expert prefit by `0.007305` and `0.006978`, and native aggregate prefit by `0.005191` and `0.004926`. The staged arms used 250M online recovery tokens, compared with 200M for direct joint.

  Final block NRMSE was `0.078769/0.242148`, top-k teacher agreement was `0.9266/0.8876`, routing entropy was `5.5315/5.5322`, and capacity overflow was zero. Layer-3 MoE error continued downward throughout recovery, while layer-2 error rose as its router adapted. The controller, conversion, and Stage-B jobs succeeded with zero failures and zero preemptions. Stage B committed `.artifact.json`, all scheduled evaluation and training JSON files, and the permanent step-1526 checkpoint with `metadata.json`, `manifest.ocdbt`, and `merge_manifest.json`. The final manifest records `converted_step_zero`, source topology `(0,1,2,3,4,5)`, and target topology `(0,1,2,2,3,4)`. Artifact provenance contains no sensitive key or value. The custom recovery worker created no W&B run, so the versioned GCS JSON files are canonical.
- Interpretation: direct preservation recovery is stable and avoids the aggregate-prefit Stage-A rollout failure, but it does not pass the Paloma `+0.03` screening bound or the stricter validation gate at 200M tokens. Frozen-router native Stage A is not merely overhead: it produces a modest token-efficiency advantage that survives Stage B. Healthy routing and decreasing layer-3 error again identify shared-bank function mismatch, rather than router collapse, as the remaining limitation.
- Next action: retain the staged native or spectral-prefit schedule as the stronger one-pair reference. Do not expand to two middle pairs. If Stage A is revised, early-stop it on aligned held-out model loss and compare at equal total recovery tokens; direct joint is the control for whether the revised local objective adds value.

### 2026-08-07 22:20 - GRUG-XEM-005 validation-aligned Stage-A launch gate

- Hypothesis: A small language-model preservation term during frozen-router Stage A improves held-out rollout quality without preventing the shared bank from fitting the two affected routed-MoE functions.
- Commit Hash: `2f440eee114d35771b0b529d342140dae254164f`; source teacher commit `884b213ff4`.
- Config: all four branches reuse the native Hungarian, no-prefit converted d512 checkpoint with layers 2-3 sharing one 256-expert bank. Only that bank trains. The control uses routed-MoE weight 1.0; treatments add CE weight 0.05, teacher-logit KL weight 0.1, or both. Each branch requests 50M tokens with held-out evaluations at 12.5M-token intervals. The selected checkpoint minimizes the Paloma-tag macro loss. Requested and exact batch-aligned processed-token counts are recorded separately.
- Gate: a treatment must improve both selected Paloma macro delta and micro-average validation delta by at least 0.01 absolute relative to the selected MoE-only control. Aggregate routed-MoE loss may be at most 10% worse, each affected layer's routed-MoE NRMSE may increase by at most 0.03 absolute, throughput must be at least 80% of control, capacity overflow must be zero, and teacher router top-1/top-4 agreement must remain 1.0. Only the lowest-Paloma treatment that passes continues to Stage B. The Stage-B milestone superset includes the complement needed to observe a requested 100M total from any selected Stage-A quarter checkpoint.
- Validation: 21 focused tests pass across the recovery objective, launch graph, and runtime checkpoint path. The runtime test selects the best of two Paloma checkpoints, distinguishes a requested 15-token horizon from 16 processed tokens, and verifies Stage B restores the selected path. Changed-file lint and Pyrefly report no errors. An independent review found no remaining blocker.
- Reproducibility: issue #8032 defines Stage A and Stage B, the four arms, exact gate, launch template, regional dependencies, and output roots. A reader given no session context returned PASS after two repair rounds. All checkpoints, caches, outputs, and v5p-8 children remain in `us-central1`.
- Next action: lower all four graphs from the pushed commit, verify empty Stage-A roots and reused central1 dependencies, then launch and babysit the four controllers. Do not launch Stage B until the complete matched result selects a treatment under the fixed gate.

### 2026-08-07 22:36 - GRUG-XEM-005 throughput-gate correction

- Correction: the launch entry incorrectly made 80% of control throughput a promotion veto. That conflicts with the experiment constraint that TPU time is available while cross-region data and checkpoint movement are the cost to avoid. Throughput remains a reported diagnostic but no longer blocks promotion. The held-out quality, MoE-fit, per-layer NRMSE, exact frozen-routing, and zero-overflow gates are unchanged. This correction was recorded before the KL and CE+KL arms completed; all experiment payloads remain in `us-central1`.

### 2026-08-07 22:42 - GRUG-XEM-005 validation-aligned Stage-A result

- Hypothesis: A small CE or teacher-logit KL term during frozen-router bank-only recovery improves held-out rollout quality relative to routed-MoE distillation alone.
- Commit Hash: launch `cd7a220d2b`; implementation `2f440eee114d35771b0b529d342140dae254164f`; source teacher `884b213ff4`.
- Commands: four central1 controllers ran `experiments.grug.moe.launch_merge_recovery` with branches `native_local_selected`, `native_local_ce`, `native_local_kl`, and `native_local_ce_kl`, stage `local`, version `2026.08.06`, and maximum concurrency 1. Monitoring state is `scratch/20260807-2220-stagea-matrix-monitoring-state.json`.
- Result: every selector chose the final step 382, which corresponds to 50,069,504 processed tokens for the 50M request.

  | Stage-A objective | Micro validation delta | Paloma macro delta | MoE loss | MoE NRMSE L2/L3 | Tokens/s |
  |---|---:|---:|---:|---:|---:|
  | MoE-only | +0.067233 | +0.064075 | 0.244239 | 0.268925 / 0.645102 | 907,177 |
  | +0.05 CE | +0.052840 | +0.048003 | 0.246119 | 0.265491 / 0.649424 | 430,179 |
  | +0.1 KL | +0.050560 | +0.045887 | 0.244659 | 0.268216 / 0.646048 | 376,735 |
  | +0.05 CE + 0.1 KL | +0.048862 | +0.044407 | 0.247440 | 0.265816 / 0.651324 | 286,700 |

  CE+KL improves the control's Paloma gap by 0.019668 and its micro-validation gap by 0.018371, exceeding the 0.01 thresholds. Its MoE loss is 1.31% higher, layer-2 NRMSE is 0.00311 lower, and layer-3 NRMSE is 0.00622 higher, all inside the fixed fit gates. Top-1 and top-4 teacher route agreement are exactly 1.0, capacity overflow is zero, and all losses are finite. All four held-out trajectories improved monotonically across the 12.58M, 25.03M, 37.62M, and 50.07M processed-token evaluations.
- Interpretation: model-preservation supervision during frozen-router Stage A materially improves rollout quality without sacrificing local shared-bank fit. CE+KL is the quality winner and passes the promotion gate. KL alone retains most of the gain at higher throughput, but TPU time is not a veto for this experiment. The result supports shared-bank distillation as the operative conversion step; it does not revive expert-level spectral matching.
- Operations: all four controllers and children succeeded. The KL child recovered automatically from one ordinary TPU preemption. Every final checkpoint, selector, evaluation/training JSON, and `.artifact.json` is present. No HBM/OOM signature appeared, no W&B run was configured, and GCS JSON is the canonical metric source. All inputs, outputs, compute, and provenance paths remain in `us-central1`; no explicit credential appears in provenance.
- Next action: launch only the CE+KL Stage-B preservation branch from its selected step-382 checkpoint. Evaluate requested 50M Stage-B tokens for the 100M-total gate, then 100M and 200M Stage-B tokens only if recovery remains stable. Do not launch any other Stage-B arm.

### 2026-08-07 23:31 - GRUG-XEM-005 CE+KL Stage-B result

- Hypothesis: The CE+KL-selected Stage-A checkpoint reaches the strict one-pair surgery target by 100M total online recovery tokens and improves the eventual recovery plateau.
- Commit Hash: launch `adadc1fefe`; implementation `2f440eee114d35771b0b529d342140dae254164f`; source teacher `884b213ff4`.
- Commands: `/dlwh/grug-xem-stageb-ce-kl-20260807` ran branch `native_local_ce_kl`, stage `preservation`, version `2026.08.06`, with a central1 CPU controller and central1 v5p-8 child. Monitoring state is `scratch/20260807-2243-stageb-ce-kl-monitoring-state.json`.
- Restore audit: Stage B loaded `gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl/stage-a/2026.08.06/checkpoints/step-382`, the exact selected Stage-A path. Its token-0 micro-validation and Paloma deltas, +0.048862 and +0.044407, exactly match the selector evaluation. The final step-1526 merge manifest records that initializer, native assignment, and target topology `(0,1,2,2,3,4)`.
- Result:

  | Stage-B tokens | Exact total recovery tokens | Micro validation delta | Paloma delta | Paloma macro |
  |---:|---:|---:|---:|---:|
  | 0 | 50,069,504 | +0.048862 | +0.044407 | 3.639071 |
  | 25,034,752 | 75,104,256 | +0.042008 | +0.039278 | 3.633942 |
  | 50,069,504 | 100,139,008 | +0.038352 | +0.035354 | 3.630018 |
  | 100,007,936 | 150,077,440 | +0.033351 | +0.030642 | 3.625306 |
  | 200,015,872 | 250,085,376 | +0.027693 | +0.025834 | 3.620498 |

  The requested 100M-total point misses both fixed limits. At the terminal horizon, Paloma is inside +0.03 but validation remains above +0.02 and `merge/recovery_tokens_to_threshold` is `-1`. The complete staged run recovers 43.3% of its selected Stage-A validation gap and 41.8% of its Paloma gap.
- Routing and fit: final layer-2/layer-3 MoE NRMSE is 0.332001/0.606683, block NRMSE is 0.086273/0.221950, top-1 teacher agreement is 0.935524/0.902855, top-4 agreement is 0.928247/0.909456, entropy is 5.5314/5.5319, and overflow is zero. Routers adapted and the affected-layer QB update path remained enabled. Final throughput is 284,250 tokens/s. Losses are finite and no HBM/OOM signature appeared.
- Interpretation: adding CE+KL to frozen-router Stage A improves the initialization and the final staged result, but does not make the one-pair surgery pass its validation gate. The result strengthens the view that shared-bank distillation is the useful conversion operation and spectral correspondence is unnecessary; it does not establish that an untied checkpoint can reach the tied manifold within modest recovery. The still-decreasing terminal trajectory suggests more tokens could close part of the gap, but the fixed budget already failed and no extension is authorized.
- Operations: the controller and child succeeded with zero failures and preemptions. `.artifact.json`, every requested evaluation, periodic training metrics, and the permanent step-1526 checkpoint are present. GCS JSON is the canonical metric source. Every dependency, output, resource, and provenance path is in `us-central1`; no explicit credential appears in provenance.
- Next action: mark the strict d512 one-pair surgery gate failed. Do not launch two middle pairs, d768 surgery, or the central2 67B-A2B experiment. A future experiment would need an explicitly revised recovery hypothesis and gate rather than additional scale under the current procedure.

### 2026-08-08 00:12 - GRUG-XEM-006 causal recovery-unlock matrix

- Hypothesis: A matched one-factor matrix can identify whether the remaining one-pair rollout gap comes from router freedom, frozen MLP-input conditioning, or the shared-bank capacity/gradient-interference constraint.
- Commit Hash: `cc0f33717f5f06c2233f11289b83bc7eeb296737`; source teacher `884b213ff4`; common selected initializer `gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl/stage-a/2026.08.06/checkpoints/step-382`.
- Config: every arm starts from the same function at the selected CE+KL Stage-A checkpoint, uses CE `1.0` + teacher-logit KL `0.1` + routed-MoE loss `1.0`, AdamW `1e-4`, zero weight decay, batch 32, sequence length 4096, seed 0, and 50M requested tokens. Evaluations are requested at 12.5M, 25M, 37.5M, and 50M. The completed `R` arm supplies its existing exact 50.07M Stage-B result.

  | Arm | Trainable state | Diagnostic comparison |
  |---|---|---|
  | `S` bank-only | Shared bank; routers and QB frozen | Common control |
  | `R` router unlock | Shared bank plus layer-2/3 routers; QB live | `R-S` measures routing freedom |
  | `N` norm unlock | `R` plus layer-2/3 `rms_mlp` and `mlp_gated_norm` | `N-R` measures frozen input conditioning |
  | `U` capacity oracle | Two independent copies of the recovered bank; routers and QB frozen | `U-S` measures sharing/interference |

  `U` is diagnostic only. Its split converts `(0,1,2,2,3,4)` to `(0,1,2,3,4,5)` by copying the recovered bank into a distinct pytree subtree for layer 3. It does not restore the teacher's old layer-3 bank. Token-0 logits are exactly identical, and the already-permuted router, bias, and pending QB state are preserved.
- Gates: existing `R@50M` is +0.03835249 validation, +0.03535414 Paloma, MoE loss 0.255401, and layer-2/3 NRMSE 0.310988/0.643498. A norm signal requires `N <= +0.03335249` validation and `<= +0.03035414` Paloma. A router signal requires `S >= +0.04335249` and `>= +0.04035414`. A capacity signal requires `U` to beat `S` by at least 0.005 on both; call capacity dominant only if `U <= +0.03335249/+0.03035414`, also beating `R` by 0.005. Require the improvement sign at both 25M and 50M, MoE loss no more than 10% above `R`, layer NRMSE no higher than 0.340988/0.673498, finite loss, zero overflow, and all experts active. `S` and `U` must retain exact teacher route agreement.
- Promotion: only a shared arm can reopen scale-up. At the exact 100.14M total horizon it must satisfy the original `<= +0.020` validation and `<= +0.030` Paloma gate. The untied `U` arm cannot promote even if it wins.
- Validation: 30 focused tests passed across checkpoint conversion, recovery behavior, launcher topology, and a real tiny conversion/selection/split/recovery checkpoint round trip. Changed-file lint passed; targeted Pyrefly reported zero errors. Independent review returned PASS. Checkpoints persist trainable scope and objective weights; resume fails closed on drift. The capacity splitter verifies the selected checkpoint path and saved step against both selector and manifest before duplicating the bank.
- Regional audit: all d512 inputs, data, workers, and outputs remain in `us-central1`. The eventual larger target remains the completed 502 GiB step-105149 67B-A2B checkpoint in `us-central2`, with all 69 dependencies local there. It remains blocked on this shared-arm gate, stacked sharded checkpoint migration, teacher-plus-student HBM proof, and central2 v4-2048 availability. No payload was copied across regions.
- Next action: verify the three new output roots are empty, lower each graph from the immutable commit, launch `S`, `N`, and `U` in `us-central1`, and assign one monitor. Compare them with the already completed `R` 25M/50M milestones before authorizing any longer shared recovery.

### 2026-08-08 00:22 - GRUG-XEM-006 causal recovery-unlock launch

- Commands: three CPU controllers launched `experiments.grug.moe.launch_merge_recovery` at branch head `98334f2c76561f2690f2adf1488b85c38d0cc120`, stage `preservation`, version `2026.08.06`, and maximum concurrency 1. The branches are `native_local_ce_kl_bank_only` (`S`), `native_local_ce_kl_mlp_norms` (`N`), and `native_local_ce_kl_capacity_oracle` (`U`). Iris assigned controller IDs `/dlwh/dlwh-grug-xem-unlock-{s,n,u}-20260808`.
- Preflight: lowering resolved `S` and `N` to the common Stage-A selector with `best_validation`; both restore step 382. `U` resolves the same selector through its explicit capacity split and restores the split's step-0 checkpoint. The trainable scopes are `shared_bank`, `shared_bank_routers_and_mlp_norms`, and `affected_expert_banks`, respectively. Each arm requests 50M tokens and the four requested milestones.
- Storage: the `S` and `N` Stage-B roots and both `U` split and Stage-B roots matched no objects before submission. The teacher, common initializer, matching artifact, all tokenized caches, controller and child compute, evaluations, outputs, and checkpoints resolve to `us-central1`. No payload or checkpoint was copied.
- Gate clarification: compare exact arm-token checkpoints 25,034,752 and 50,069,504. At 25M, `N` signals a norm limitation only at validation/Paloma `<= +0.037008/+0.034278`, while `S` signals useful router freedom only at `>= +0.047008/+0.044278`. At 50M, the corresponding thresholds are `N <= +0.03335249/+0.03035414` and `S >= +0.04335249/+0.04035414`. At both checkpoints, capacity requires `U <= S - 0.005` on both metrics, and dominant capacity additionally requires `U <= R - 0.005` on both. The run-validity gates must also hold at both checkpoints.
- Promotion: `S`, `R`, and `N` are eligible independently of the attribution labels; `U` never is. An eligible arm reopens scale-up only if its exact 100,139,008-total-token result is validation `<= +0.020`, Paloma `<= +0.030`, and run-valid. The current 50M arms alone cannot establish that terminal promotion result.
- Operations: issue #8032 passed a fresh zero-context readability review after the exact gates and regional invariant were added. The accepted launch links and regional audit are recorded in issue comment `5225105921`. A dedicated monitor owns all three controllers through terminal artifacts and gate application.

### 2026-08-08 01:29 - GRUG-XEM-006 causal recovery-unlock result

- Hypothesis: Router freedom, frozen MLP-input conditioning, or shared-bank capacity/gradient interference accounts for at least 0.005 of the remaining one-pair recovery gap.
- Commit Hash: launch `98334f2c76561f2690f2adf1488b85c38d0cc120`; implementation `cc0f33717f5f06c2233f11289b83bc7eeb296737`; source teacher `884b213ff4`.
- Commands: controllers `/dlwh/dlwh-grug-xem-unlock-{s,n,u}-20260808` ran `experiments.grug.moe.launch_merge_recovery` with stage `preservation`, version `2026.08.06`, and branches `native_local_ce_kl_bank_only`, `native_local_ce_kl_mlp_norms`, and `native_local_ce_kl_capacity_oracle`. Read-only Iris jobs `/dlwh/dlwh-grug-xem-unlock-validate{,2,3}-20260808` inspected the canonical GCS JSON and manifests from `us-central1`.
- Restore audit: `S` and `N` loaded the exact selected CE+KL Stage-A checkpoint `step-382`. `U` duplicated that checkpoint's recovered bank into an independent layer-3 bank at split `step-0`; its manifest records source recovery step 382, identical-start oracle kind, and untied topology `(0,1,2,3,4,5)`. Every recovery manifest records CE `1.0`, KL `0.1`, MoE `1.0`, the intended trainable scope, and recovery step 382.
- Result:

  | Arm | 25M validation | 25M Paloma | 50M validation | 50M Paloma | 50M MoE loss | 50M NRMSE L2/L3 |
  |---|---:|---:|---:|---:|---:|---:|
  | `S` bank-only | +0.042001 | +0.039157 | +0.038294 | +0.035120 | 0.250479 | 0.295958 / 0.642935 |
  | `R` router unlock | +0.042008 | +0.039278 | +0.038352 | +0.035354 | 0.255401 | 0.310988 / 0.643498 |
  | `N` norm unlock | +0.142956 | +0.143200 | +0.131897 | +0.131611 | 0.079827 | 0.389206 / 0.090405 |
  | `U` untied capacity oracle | +0.038885 | +0.036324 | +0.033396 | +0.030512 | 0.218323 | 0.233883 / 0.618017 |

  The 25M and 50M entries are exact Stage-B processed-token counts 25,034,752 and 50,069,504. The 50M checkpoint is 100,139,008 total recovery tokens after adding Stage A.
- Gate application: router freedom fails its signal gate because `S` is marginally better than `R`, rather than at least 0.005 worse. Norm unlocking misses its validation/Paloma thresholds by 0.098545/0.101257. `U` improves `S` by 0.004898 validation and 0.004608 Paloma at 50M, short of 0.005 on both; it also misses the dominant-capacity bounds by 0.000043 validation and 0.000158 Paloma. The same improvement sign holds at 25M, but the effect is only 0.003117/0.002834 there. No attribution gate passes.
- Routing and validity: `S` and `U` retain exact teacher top-1/top-4 agreement. All four arms have finite metrics, zero capacity overflow, entropy near 5.53, and all 256 experts active in both affected layers. `S`, `R`, and `U` satisfy the MoE and per-layer NRMSE health bounds. `N` fails the layer-2 NRMSE bound and its layer-3 teacher route agreement falls to 0.0128 top-1 and 0.0193 top-4 despite full expert activity and high entropy.
- Promotion: no shared arm meets validation `<= +0.020` and Paloma `<= +0.030` at the exact 100,139,008-total-token checkpoint. `U` is untied and cannot promote by construction. Do not launch d768 surgery, two-pair surgery, or the central2 67B experiment from this matrix.
- Operations: the three controllers, three recovery children, capacity split child, and three read-only validation jobs succeeded without resubmission. Each recovery root has five scheduled evaluation JSON files, final training JSON, `.artifact.json`, and checkpoints through step 382 with `metadata.json`, `manifest.ocdbt`, and `merge_manifest.json`. The split root has a complete step-0 checkpoint. All inputs, compute, outputs, and validation reads remained in `us-central1`; no payload was copied across regions.
- Interpretation: fixed routers are not the obstruction. Unfreezing MLP-input norms under the current `1e-4` objective creates severe rollout loss and layer-3 route drift. Independent bank capacity produces a consistent but sub-threshold benefit, so sharing interference is measurable but does not explain the full remaining gap. The current surgery remains below promotion quality.
- Next action: keep scale-up blocked. A new d512 hypothesis should change shared-bank expressivity or the preservation objective and preregister a fresh gate; longer runs or larger models under the current recipe are not supported by this result.

### 2026-08-08 02:15 - GRUG-XEM-007 layer-conditioned adapter design gate

- Hypothesis: the untied oracle's growing advantage comes from layer-conditioned specialization or cross-layer update interference, and a layer-3-only rank-8 input/output residual adapter can recover most of that advantage without restoring a second expert bank.
- Internal evidence: `U-S` held-out validation/Paloma improvement grows monotonically from `0.001838/0.001728` at 12.58M Stage-B tokens to `0.004898/0.004608` at 50.07M. Both arms start from identical logits, use the same seed, data order, objective, and frozen routing. Under `S`, layer-2 MoE NRMSE worsens from `0.2658` to `0.2960` while layer 3 improves only slightly; under `U`, both layers improve to `0.2339/0.6180`. This is more consistent with a shared-function compromise than ordinary run variance. It does not distinguish layer specialization from destructive gradient or optimizer-state interference.
- Prior work: Jaggi establishes strict routed-expert tying but does not convert untied checkpoints or add depth-specific deltas ([arXiv:2606.16825](https://arxiv.org/abs/2606.16825)). Bae et al. convert pretrained dense Transformers into tied recursive models and restore depth-specific flexibility with layer-wise LoRA; they initialize deltas by truncated SVD of original-minus-shared weights ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672)). ResidualTransformer likewise represents each shared matrix as a full-rank centroid plus a layer-specific low-rank residual ([arXiv:2310.02489](https://arxiv.org/abs/2310.02489)). Those weight-space initializations assume comparable internal factorizations. Grug's native/spectral experiments found expert correspondence weak, so this arm applies low-rank deltas at the routed function boundary instead of SVD-aligning SwiGLU weights.
- Architecture: keep topology `(0,1,2,2,3,4)`. Layer 2 uses the shared bank directly. Layer 3 computes routes on the original MLP input `x`, applies `x' = x + (x A_in) B_in` only before expert dispatch, and applies `y' = y + (y A_out) B_out` after sparse combine. The always-on dense MLP still consumes the original `x`. `A_in/A_out` are deterministic nonzero matrices and `B_in/B_out` are zero, making the augmented step-0 checkpoint exactly function-identical with live first-step gradients into both `B` factors.
- Parameter budget: at d512 and rank 8 the adapter adds `4 * 512 * 8 = 16,384` parameters. A routed bank has `100,663,296` parameters, so the arm retains `100,646,912`, or `99.9837%`, of the one-bank saving. It adds no expert routing, dispatch, or collective. Report throughput; a regression greater than 5% relative to `S'` triggers profiling but does not veto the scientific result because TPU time is available and cross-region traffic is the cost constraint.
- Matrix: rerun an adapter-disabled `S'` control and one `A8` arm from the exact CE+KL Stage-A step-382 selection. Both use CE `1.0`, teacher-logit KL `0.1`, routed-MoE loss `1.0`, AdamW `1e-4`, zero weight decay, batch 32, sequence length 4096, seed 0, frozen routers/QB/norms/attention/dense MLPs, and 50M requested Stage-B tokens. `A8` trains only the one shared bank and layer-3 adapter. Evaluate at the existing 12.5M, 25M, 37.5M, and 50M requested milestones.
- Diagnostic gate: token-0 logits, routed outputs, selected experts, combine weights, QB values, and pending QB state must match `S'` exactly. `A8` must not be worse than `S'` on validation or Paloma at any milestone. At 25.03M it must recover at least 50% of the existing `U-S` improvement: validation `<= 0.04044294`, Paloma `<= 0.03774047`. At 50.07M it must recover at least 75%: validation `<= 0.03462035`, Paloma `<= 0.03166438`. At 50M, MoE loss must be `<= 0.250479`, layer-2/3 MoE NRMSE `<= 0.295958/0.642935`, teacher-route agreement on the same student state exactly 1.0, overflow zero, all 256 experts active, and metrics finite.
- Promotion: the diagnostic gate only establishes that a tiny layer-conditioned delta captures the untied-oracle benefit. Scale-up remains blocked unless a shared arm also reaches validation `<= +0.020` and Paloma `<= +0.030` at exactly 100,139,008 total recovery tokens. Do not reinterpret a utility pass as a surgery promotion pass.
- Regional constraint: the initializer, teacher, matching artifact, tokenized caches, controllers, v5p-8 children, evaluations, outputs, and checkpoints must remain in `us-central1`. No checkpoint or payload copy is allowed.
- Falsifier and next action: if `A8` misses the diagnostic gate, measure per-layer shared-bank gradient conflict before increasing rank. A width-320 shared-bank arm is the next raw-capacity test only after that diagnostic; increasing the expert count is last because it changes router/QB shape and lacks a clean exact initialization.

### 2026-08-08 02:31 - GRUG-XEM-007 implementation and launch-readiness gate

- Implementation: added an explicit layer-3 routed-expert adapter leaf, function-preserving Stage-A checkpoint augmentation with format-v3 provenance, a shared-bank-plus-adapter recovery scope, and matched `S'`/`A8` central1 graph branches. The control restores the selected Stage-A checkpoint directly. `A8` restores a permanent step-0 augmented checkpoint whose manifest binds the exact selected source path, selected step, rank, layer, and input topology.
- Numerical contract: routing and QB statistics use the original MLP input; only the shared expert evaluation sees the adapted input, and the always-on dense MLP remains unadapted. Both low-rank output factors initialize to zero. Residual corrections are cast back to the activation dtype before addition, so fp32 adapter parameters preserve bf16 inputs exactly at token 0. Export includes the adapter config and all four adapter tensors.
- Resume and schema safety: adapter recovery requires the explicit augmented initialization, preservation stage, shared-bank-plus-adapter trainable scope, exact CE+KL-selected source checkpoint and step, and expected rank 8. Resumes reject initializer, objective, scope, source, step, rank, layer, or topology drift. Legacy format-v2 manifests remain readable only when they contain no adapter provenance.
- Validation: 48 focused model, checkpoint, recovery, runtime, and launcher tests pass, including a real tiny conversion/selection/augmentation/recovery/resume round trip and four-axis abstract ring-MoE recovery lowering. The corrected fp32-adapter/bf16-activation parity test also passes separately. Changed-file pre-commit and `git diff --check` pass. Targeted Pyrefly reports zero errors outside the two unchanged `model.py` baseline errors. Independent review returned PASS after the mixed-dtype and provenance fixes.
- Lint-catalog review: centralized the adapter rank, horizon, and milestone constants and replaced two experiment-history docstrings with caller-visible contracts. Findings in unchanged matcher, prefit, storage, June 67B, and Snowball code predate this snapshot. The broad recovery runtime and graph builder also predate `A8`; their explicit branch-local configs are retained here to avoid refactoring every completed artifact fingerprint immediately before launch. A separate cleanup can split those modules after the matched run without changing this experiment's pinned graph.
- Operations: neither arm has launched. The planned roots are `gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl_adapter_control/stage-b/2026.08.06` and `gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl_adapter_r8/{augment,stage-b}/2026.08.06`. Before submission, snapshot and push the implementation, lower both branches, verify the selected source path and all runtime regions, and confirm every planned output prefix is empty.

### 2026-08-08 02:43 - GRUG-XEM-007 prelaunch snapshot

- Snapshot: implementation commit `154621fd28`; lint-catalog follow-up `4dd7d0abc3`; pushed branch head `4dd7d0abc3e5066035449942d9e57e6c30320c4c`.
- Lowering: both `native_local_ce_kl_adapter_control` and `native_local_ce_kl_adapter_r8` preservation graphs lower successfully with version `2026.08.06`, `MARIN_PREFIX=gs://marin-us-central1`, the d512 teacher root in `us-central1`, teacher commit `884b213ff4`, and maximum concurrency 1. The A8 graph explicitly depends on the CE+KL Stage-A selector in addition to its augmentation checkpoint.
- Resolved contract: control initialization is `best_validation` from the selected Stage-A root. A8 initialization is `latest` from its step-0 augmentation root and separately binds the selected Stage-A root, rank 8, topology `(0,1,2,2,3,4)`, CE `1.0`, KL `0.1`, MoE `1.0`, and exact milestones 12,582,912, 25,034,752, 37,617,664, and 50,069,504 tokens. Child resources resolve to v5p-8 in `us-central1`.
- Storage preflight: recursive metadata listings for the control Stage-B root, A8 augmentation root, and A8 Stage-B root each matched no objects. No checkpoint or payload was copied. Launch two central1 CPU controllers from the pushed head, then assign one babysitter through terminal artifact and gate verification.

### 2026-08-08 02:45 - GRUG-XEM-007 matched launch, control result, and A8 launch incident

- Commit Hash: initial launch `70a2f6a183bf50119bc5944a9053619aee69269a`; source teacher `884b213ff4`.
- Commands: `/dlwh/grug-xem-adapter-control-20260808` and `/dlwh/grug-xem-adapter-r8-20260808` ran `experiments.grug.moe.launch_merge_recovery` with branches `native_local_ce_kl_adapter_control` and `native_local_ce_kl_adapter_r8`, stage `preservation`, version `2026.08.06`, `MARIN_PREFIX=gs://marin-us-central1`, `GRUG_MERGE_TEACHER=gs://marin-us-central1/grug/tied_experts/d512/full/baseline/2026.08.06`, teacher commit `884b213ff4`, and maximum concurrency 1. Exact fixed resubmit commands are in `scratch/20260808-0245-adapter-{control,r8}-monitoring-state.json`; Iris injected credentials and no explicit credential was passed.
- Control restore audit: `S'` loaded the exact selected CE+KL Stage-A checkpoint `gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl/stage-a/2026.08.06/checkpoints/step-382`. Its token-0 validation and Paloma deltas, `+0.0488624573/+0.0444073677`, exactly match the Stage-A selector. All four subsequent milestones reproduce the earlier `S` bank-only arm:

  | Stage-B tokens | Validation delta | Paloma delta |
  |---:|---:|---:|
  | 12,582,912 | +0.0434539318 | +0.0396811962 |
  | 25,034,752 | +0.0420012474 | +0.0391573906 |
  | 37,617,664 | +0.0397439003 | +0.0364453793 |
  | 50,069,504 | +0.0382940769 | +0.0351204872 |

- Control final metrics: routed-MoE loss `0.250478536`, layer-2/3 MoE NRMSE `0.295958132/0.642935336`, teacher top-1/top-4 route agreement `1.0/1.0` at both layers, zero overflow, all 256 experts active at both layers, and throughput `286,373.8` tokens/s. The controller and central1 v5p-8 child succeeded with zero failures and preemptions. All five evaluation JSON files, final training JSON, `.artifact.json`, and the permanent step-382 checkpoint with metadata and merge manifest are present.
- A8 incident: the initial augmentation child failed before writing a checkpoint. The historical selected Stage-A checkpoint has a format-v2 merge manifest from before recovery recipe fields were added, so its `recovery_stage`, `recovery_trainable_scope`, and CE/MoE/KL fields are null. The new validator rejected those absent fields even though the Stage-A artifact config records `stage=local`, native assignment, no prefit, CE `0.05`, MoE `1.0`, KL `0.1`, best-validation selection, the expected output root, and fingerprint `62564720`. The selector independently binds step 382, `50,069,504` tokens, and Paloma `3.6390709877`. The failure is limited to provenance-schema compatibility; it occurred before model execution and had no infrastructure error.
- Next action: validate historical recovery recipe fields against the immutable Stage-A `.artifact.json` when the old manifest lacks them, retain fail-closed source/selector/topology checks, then resubmit only A8. Keep the completed control fixed.

### 2026-08-08 03:05 - GRUG-XEM-007 A8 provenance fix and resubmit

- Commit Hash: `e35250729aa5e2e89535634c280f014a2d72ef21`.
- Fix: adapter augmentation now accepts a historical format-v2 recovery manifest only when the adjacent immutable Stage-A artifact config proves the exact expected local CE+KL bank-only recipe. New manifests continue through the direct manifest-field path. Focused tests and an independent review passed before resubmission.
- Command: `/dlwh/grug-xem-adapter-r8-fix1-20260808` reran `experiments.grug.moe.launch_merge_recovery` with branch `native_local_ce_kl_adapter_r8`, stage `preservation`, version `2026.08.06`, `MARIN_PREFIX=gs://marin-us-central1`, the central1 d512 teacher root and commit `884b213ff4`, and maximum concurrency 1. The exact fixed command is in `scratch/20260808-0245-adapter-r8-monitoring-state.json`.
- Augmentation result: the central1 CPU child succeeded and wrote permanent step 0. Its format-v3 manifest binds source checkpoint `.../native_local_ce_kl/stage-a/2026.08.06/checkpoints/step-382`, source recovery step 382, output step 0, layer 3, rank 8, input and output topology `(0,1,2,2,3,4)`, zero-initialized input/output adapter kind, native assignment, no prefit, teacher step 10993, and source commit `884b213ff4`. It contains no capacity oracle.
- Current status: the A8 central1 v5p-8 preservation child has started. Accept training only after its token-0 evaluation exactly matches `S'`; then apply the preregistered milestone utility and promotion gates through 50,069,504 Stage-B tokens.

### 2026-08-08 03:27 - GRUG-XEM-007 layer-conditioned rank-8 adapter result

- Hypothesis: a layer-3-only rank-8 routed-function adapter can recover most of the untied oracle's advantage while preserving one shared expert bank.
- Commit Hash: implementation and launch `70a2f6a183bf50119bc5944a9053619aee69269a`; provenance fix `e35250729aa5e2e89535634c280f014a2d72ef21`; launch/control snapshot `38d636e6dd`.
- Command: controller `/dlwh/grug-xem-adapter-r8-fix1-20260808` ran branch `native_local_ce_kl_adapter_r8`, stage `preservation`, version `2026.08.06`, with a central1 CPU controller and central1 v5p-8 child. The exact resubmit command and restart count are in `scratch/20260808-0245-adapter-r8-monitoring-state.json`.
- Initialization audit: the permanent augmentation checkpoint is step 0 with a format-v3 manifest binding the exact CE+KL Stage-A step-382 source, source recovery step 382, layer 3, rank 8, topology `(0,1,2,2,3,4)`, and zero-initialized input/output adapter kind. The live token-0 held-out scalar reductions were within `2.7e-6` of `S'` but were not bit-identical: validation `+0.0488629341` versus `+0.0488624573`, and Paloma `+0.0444047451` versus `+0.0444073677`. The first training-step loss, cross-entropy, and routed-MoE loss matched `S'` bit-for-bit; logit KL differed by `3.7e-9`. Teacher route agreement was exact. The runtime artifacts do not contain a direct tensor-level comparison of logits, routed outputs, QB values, or pending QB state, so the strict live token-0 tensor gate is not independently demonstrated beyond the conversion tests and source-bound manifest.
- Result:

  | Stage-B tokens | A8 validation | S' validation | A8 Paloma | S' Paloma |
  |---:|---:|---:|---:|---:|
  | 12,582,912 | +0.0434265137 | +0.0434539318 | +0.0396578312 | +0.0396811962 |
  | 25,034,752 | +0.0419437885 | +0.0420012474 | +0.0391027927 | +0.0391573906 |
  | 37,617,664 | +0.0396561623 | +0.0397439003 | +0.0363512039 | +0.0364453793 |
  | 50,069,504 | +0.0382659435 | +0.0382940769 | +0.0350942612 | +0.0351204872 |

- Gate application: A8 is slightly better than `S'` at every post-update milestone, so the no-worse condition passes. At 25.03M it misses the 50%-oracle-capture validation/Paloma bounds `+0.04044294/+0.03774047` by `0.00150085/0.00136232`; its improvement over `S'` captures only `1.84%/1.93%` of the prior `U-S` gap. At 50.07M it misses the 75%-capture bounds `+0.03462035/+0.03166438` by `0.00364559/0.00342988` and captures only `0.57%/0.57%` of the prior gap. The utility gate fails.
- Fit and validity: final routed-MoE loss is `0.248744696`, layer-2/3 MoE NRMSE is `0.292287886/0.641916811`, and block NRMSE is `0.075367078/0.228661582`. These all pass the fixed fit bounds. Teacher top-1/top-4 route agreement remains exactly `1.0`, capacity overflow is zero, all 256 experts are active in both layers, entropy is `5.52882/5.53291`, and all logged metrics are finite. Throughput is `284,458.8` tokens/s, or `99.33%` of `S'`, above the 95% diagnostic threshold. Adapter parameter norms were not logged; reading the checkpoint from the local workstation would violate the regional data constraint, so no post-hoc local checkpoint load was performed.
- Promotion: at exactly `100,139,008` total recovery tokens, A8 is validation `+0.0382659435` and Paloma `+0.0350942612`. It misses the original `+0.020/+0.030` surgery limits. The central1 controller, augmentation child, and preservation child succeeded with zero failures and preemptions. The augmentation and Stage-B roots contain `.artifact.json`, all scheduled JSON metrics, and permanent checkpoint metadata/manifests. No W&B run was configured.
- Interpretation: the rank-8 layer-conditioned boundary adapter improves local MoE fit and produces a consistent but negligible held-out gain. It does not recover the untied oracle's advantage. Rank increase is not supported yet; the preregistered next diagnostic is per-layer shared-bank gradient conflict. Scale-up to d768 surgery, two-pair surgery, or 67B remains blocked.

### 2026-08-08 03:35 - GRUG-XEM-008 direct shared-bank gradient-conflict gate

- Hypothesis: layers 2 and 3 produce persistently opposing direct gradients for the one shared routed-expert bank, causing material cancellation during frozen-router bank recovery.
- Scope: this is a read-only d512 checkpoint diagnostic, not training and not a promotion experiment. Probe the selected CE+KL Stage-A step-382 checkpoint and the matched `S'` Stage-B step-191 and step-382 checkpoints. These are the shared-recovery start, 25,034,752-token midpoint, and 50,069,504-token endpoint.
- Data: use the same 16 deterministic continuation batches at every checkpoint: loader indices 382 through 397 with batch 32, sequence length 4096, seed 0, and the recovery loader's existing `fold_in(..., 2)` key. This covers 2,097,152 positions per checkpoint and does not reuse the 382 batches consumed by `S'`.
- Gradient definition: roll out each student checkpoint once per batch, stop-gradient each affected layer's current MLP input, and evaluate the untied teacher MoE on that student state. For layer `l`, differentiate the existing normalized routed-output MSE `E_l` only with respect to the shared bank while holding the router, combine weights, adapter, teacher target, and state fixed. Thus `sqrt(E_l)` remains the reported MoE NRMSE. This measures direct local functional compatibility. It deliberately excludes CE, KL, indirect layer-2-to-layer-3 state effects, Adam moments, and long-horizon validation behavior.
- Metrics: for every batch and for the mean gradient over 16 batches, record layer-2/layer-3 bank-gradient norms, dot product, cosine, norm ratio, and cancellation `1 - ||g2 + g3|| / (||g2|| + ||g3||)`. Record the same reductions for gate/up/down projections. Per-expert cosine, negative-dot count, negative-dot energy share, and routed traffic are descriptive; experts are not statistical replicates. Accumulate gradients and reductions in fp32 and persist no gradient or model tensor.
- Statistical summary: the batch is the replicate. Report mean cosine with a 10,000-resample percentile-bootstrap 95% interval using seed 8032, median/IQR, and negative sign count. The primary aggregate cosine is computed from the two mean gradient trees, not by averaging per-batch cosines.
- Conflict gate: support persistent material local conflict only if at least two of three checkpoints have aggregate cosine `<= -0.10`, mean per-batch cosine with bootstrap upper bound below zero, at least 12 of 16 negative batch cosines, cancellation `>= 0.10`, and `min(||g2||,||g3||) / max(||g2||,||g3||) >= 0.25`. Call dominant direct conflict unsupported if all three aggregate cosines are greater than `-0.05`, no checkpoint has more than 4 of 16 negative batches, and all cancellation ratios are below `0.05`. Other results are inconclusive.
- Controls: require the gradient of `E2 + E3` to equal `g2 + g3` within relative error `1e-5` in the numerical test harness; routes must remain teacher-exact, overflow must remain zero, all metrics finite, and all 256 experts active across the aggregate. Use sharding-safe leafwise fp32 products and reductions; do not ravel the 3-D expert tensors.
- Decision: a gate pass justifies a second diagnostic that decomposes the exact CE `1.0` + KL `0.1` + MoE `1.0` Stage-B objective by use site and tests saved-Adam update counterfactuals. A gate failure stops the dominant direct-conflict hypothesis and moves the queue toward shared functional capacity or conditioning. Do not launch PCGrad, increase adapter rank/bank width, or scale surgery from cosine alone.
- Prior work: PCGrad defines local conflict by negative gradient dot product but argues that magnitude and curvature also matter ([Yu et al., 2020](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)). Recon motivates measuring conflict by shared component and over training rather than once ([Shi et al., 2023](https://arxiv.org/abs/2302.11289)). ForkMerge and the scalarization study caution that gradient conflict is weak causal evidence for held-out transfer or optimizer benefit ([Jiang et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf); [Kurin et al., 2022](https://arxiv.org/abs/2201.04122)). No primary study found in the low-effort search directly validates this diagnostic for one sparse expert bank reused at multiple Transformer depths.
- Regional constraint: controller, v5p-8 worker, teacher/student checkpoint reads, data, and small scalar outputs remain in `us-central1`. Write no checkpoint, raw gradient, model tensor, W&B artifact, or cross-region payload. The planned root is `gs://marin-us-central1/grug/expert_merge/d512/shared_bank_gradient_conflict/diagnostic/2026.08.08`.

### 2026-08-08 03:48 - GRUG-XEM-008 implementation and prelaunch snapshot

- Commit Hash: `38f1dcf79f16b10b339b09c0c13a7b6f157cb1a1`.
- Implementation: added one sequential read-only diagnostic worker over the selected Stage-A step-382 checkpoint and the matched `S'` step-191/step-382 checkpoints. Each batch performs one student rollout, detaches the two affected MLP inputs, computes the two direct shared-bank gradients, accumulates mean gradients in fp32 on device, and persists only scalar, projection, per-expert, routing-control, bootstrap, and decision JSON.
- Provenance: the nonlocal job fail-closes on teacher artifact `38d1fe9b`, Stage-A artifact `62564720`, `S'` artifact `59969d51`, teacher commit `884b213ff4`, exact checkpoint paths/steps/tokens/roles, source and target topology, assignment, Stage-A selector and CE+KL recipe, `S'` preservation recipe and initializer, and canonical equality with the `S'` data config. Selected expert IDs must match the mapped teacher IDs in order, not merely as a set.
- Validation: nine focused mathematical, runtime, launch, provenance, and four-axis lowering tests pass. The only warning is unused donation in the tiny CPU runtime test. Changed-file pre-commit, Pyrefly, and `git diff --check` pass. Independent scientific and operational reviews both returned PASS after adding exact ordered-ID checks, NRMSE output, artifact/data/role binding, and negative regressions.
- Lowering: the production graph lowers at fingerprint `73ebb9b1` with version `2026.08.08`, one v5p-8 worker in `us-central1`, the exact three frozen checkpoint dependencies, loader indices 382 through 397, bootstrap seed 8032, and scalar-only output root `gs://marin-us-central1/grug/expert_merge/d512/shared_bank_gradient_conflict/diagnostic/2026.08.08`. The root matched no objects immediately before submission.
- Next action: push this prelaunch snapshot, launch one central1 CPU controller with maximum concurrency 1, and assign a dedicated babysitter through terminal artifact and gate verification. Do not launch any training or larger surgery.

### 2026-08-08 04:55 - GRUG-XEM-008 direct shared-bank gradient-conflict result

- Hypothesis: layers 2 and 3 request persistently opposing direct routed-MoE updates from the shared bank during frozen-router recovery.
- Commit Hash: implementation `38f1dcf79f16b10b339b09c0c13a7b6f157cb1a1`; launch provenance `cc029e85546bf8523f89a7811ea8fe4fd26ccb18`.
- Command: controller `/dlwh/grug-xem-gradient-conflict-20260808` ran `python -m experiments.grug.moe.launch_gradient_conflict --version 2026.08.08 --run --max-concurrent 1` with a central1 CPU controller and one central1 v5p-8 child. `MARIN_PREFIX` and the teacher root both resolved under `gs://marin-us-central1`; teacher commit was `884b213ff4`. The exact fixed resubmit command and restart count are in `scratch/20260808-0442_monitoring_state.json`.
- Inputs: artifact fingerprint `73ebb9b1` binds teacher `38d1fe9b`, selected Stage A `62564720`, and matched `S'` `59969d51`. The checkpoint roles and paths are the selected Stage-A step 382, `S'` midpoint step 191, and `S'` endpoint step 382. Every checkpoint used loader indices 382 through 397, batch 32, sequence length 4096, seed 0, `fold_in(..., 2)`, 10,000 bootstrap resamples, and seed 8032.
- Result:

  | Checkpoint | Aggregate cosine | Cancellation | Norm ratio | Batch mean cosine (95% bootstrap interval) | Negative batches | Material-conflict gate |
  |---|---:|---:|---:|---:|---:|---|
  | Stage-A start, 0 continuation tokens | -0.045448 | 0.146092 | 0.180523 | -0.151560 [-0.185357, -0.112829] | 14/16 | Fail |
  | `S'` midpoint, 25,034,752 tokens | -0.104190 | 0.329587 | 0.899402 | -0.064347 [-0.076626, -0.050098] | 16/16 | Pass |
  | `S'` endpoint, 50,069,504 tokens | -0.017830 | 0.161887 | 0.216225 | -0.066230 [-0.079956, -0.050099] | 14/16 | Fail |

- Projection and expert detail: Stage-A down/gate/up cosines are `-0.074415/-0.040544/-0.038550`, with norm ratios `0.321547/0.148713/0.149368`; 254 of 256 expert dots are negative and their negative-dot energy share is `0.894903`. The midpoint values are `-0.147733/-0.087410/-0.079980`, ratios `0.738705/0.993027/0.979890`, 234 negative expert dots, and `0.922178` negative energy. The endpoint values are `-0.043601/-0.013726/-0.009698`, ratios `0.409318/0.172733/0.182272`, 235 negative expert dots, and `0.810439` negative energy. Experts are descriptive components, not statistical replicates.
- Local fit and controls: mean layer-2/layer-3 MoE NRMSE is `0.250003/0.648712` at the start, `0.270446/0.646441` at the midpoint, and `0.278323/0.639970` at the endpoint. Selected expert IDs and top-1 routes match the mapped teacher exactly at both affected layers, combine-weight maximum absolute difference is zero, overflow is zero, all 256 experts are active in both layers, and all reported scalars are finite.
- Operations: the controller and child both succeeded with exit 0, zero failures, zero preemptions, and zero restarts. The output root is `gs://marin-us-central1/grug/expert_merge/d512/shared_bank_gradient_conflict/diagnostic/2026.08.08`. It contains the standard artifact/executor metadata and only one scientific payload, `gradient_conflict.json`; no checkpoint, gradient tensor, model tensor, or W&B artifact was written.
- Interpretation: only the midpoint passes all five preregistered material-conflict criteria. The start and endpoint have negative per-batch means but fail the primary mean-gradient-tree cosine and gradient-norm-balance criteria. One of three checkpoints is insufficient for the persistent-conflict gate, while the strict no-conflict gate also does not pass. The registered outcome is inconclusive, and persistent dominant direct conflict is not supported.
- Next action: stop this line before any PCGrad, full-objective gradient decomposition, optimizer counterfactual, rank increase, bank-width increase, d768 surgery, two-pair surgery, or 67B launch. Resume H4 only with a separately preregistered shared-functional-capacity or conditioning hypothesis.

### 2026-08-08 05:35 - GRUG-XEM-009 June 67B tied-from-initialization preregistration

- Hypothesis: the quality penalty of routed-expert tying continues to diminish at the June 67B-A2B compute topology, while routing and shared-bank updates remain healthy.
- Scientific scope: this is an architecture test from random initialization, not conversion, offline prefit, Stage A, or Stage B. It reads no Snowball or 67B teacher checkpoint. The failed d512 surgery gate remains binding for all post-hoc conversion work.
- Matrix: one fresh untied control with topology `(0,1,...,25)` and one fresh tied treatment with topology `(0,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,8,9)`. The treatment keeps two anchors at each end, uses five four-layer core groups and one two-layer group, and shares only routed expert gate/up/down tensors. Attention, routers and QB state, norms/GatedNorms, always-on dense MLPs, and blocks remain layer-specific.
- Parameter accounting: the untied model has about 67.08B unique parameters and 65.43B unique routed-expert parameters. The tied treatment has about 26.81B and 25.17B, reductions of 60.0% and 61.5%, while active FLOPs per token are unchanged. Call the treatment “67B-A2B compute topology, 26.8B unique,” not a 67B-parameter checkpoint.
- Shared config: d2560, 26 layers, 256 experts, top-4, sequence length 8,192, batch 8,192, seed 0, EP1, replica axis 8, array-stacked blocks, v4-2048, MuonH, unscaled tied-bank learning rate, and the original 150,000-step/10.066T-token LR horizon. Both arms use the phase-0 June datakit mixture, which is exact through the planned 3,000-step milestone. “Matched” means the same data order, optimizer, schedule, and seed. All layer-specific tensors and the treatment's shared bank for a group use the same initialization key as the corresponding control tensor at that group's first layer; only the control's non-representative expert banks lack treatment counterparts.
- Smoke: 100 optimizer updates or 6,710,886,400 tokens per arm. Require both W&B runs to reach state `finished`, terminal zero-indexed `global_step == 99`, and `throughput/total_tokens == 6,710,886,400`. Require finite `train/cross_entropy_loss` and all `tying/expert_{gradient,update}_norm_by_bank/*` values at every logged step; `train/router/layer_i/capacity_overflow_rate == 0` at every step and layer; nonzero counts in all 256 buckets of each `train/router/layer_i/routing_hist` over the union of steps 90-99; and the arithmetic mean of `train/router/layer_i/routing_entropy` over all 26 layers and steps 90-99 at least 5.4. For the six reused treatment banks 2-7, take each bank's median whole-tree gradient norm and update norm over steps 20-99; for each metric require `max(six medians) / median(six medians) <= 4`. Finally, require the treatment's median `train/cross_entropy_loss` over steps 90-99 minus the control's corresponding median to be at most 0.15. Throughput over steps 20-99 and peak HBM are reported but do not veto because TPU time is available.
- Smoke decision: launch the 3,000-step milestone only if every health check passes. A failed compile/HBM check triggers one topology-preserving implementation diagnosis. A routing/update/loss failure stops H12 rather than changing the gate.
- Milestone: fresh matched 3,000-update runs, 201,326,592,000 tokens each, with Paloma after updates 500, 1,000, 1,500, 2,000, 2,500, and 3,000, logged at zero-indexed W&B steps 499, 999, 1,499, 1,999, 2,499, and 2,999. These correspond to 33.55B, 67.11B, 100.66B, 134.22B, 167.77B, and 201.33B tokens. This is an early-quality milestone, not a full 10T result. At the terminal evaluation require `tied eval/paloma/macro_loss - control eval/paloma/macro_loss <= 0.04`. Across the last three evaluations, each adjacent tied-minus-control Paloma macro gap must be no greater than the preceding gap. Apply the finite-loss, zero-overflow, 256-expert-traffic, entropy `>=5.4`, and six-bank norm-ratio `<=4` rules above to each 100-step window ending at the six evaluation steps. Throughput and unique-parameter reduction remain descriptive.
- Instrumentation gate: before submission, verify explicit bank topology under array-stacked scan; tied checkpoint round-trip; summed shared-bank gradients; per-layer router/QB and shared dense ownership; per-bank gradient/update norms; activation norms; cross-loop top-1/top-k agreement; and four-axis lowering.
- Regional gate: all training/eval reads, controller and TPU compute, evaluations, checkpoints, temporary checkpoints, profiles, and outputs remain in `us-central2`. Smoke roots are `gs://marin-us-central2/grug/tied_experts/june67b/smoke/{baseline,middle_groups_unscaled}/2026.08.08`; both must be empty before first submission. Metadata audit found all 200 datakit prefixes and all 23 eval artifact markers in `gs://marin-us-central2`. The completed step-105149 teacher is also central2 but is deliberately unused. Snowball exports are in `s3://marin-us-east-02a` and must not be read or copied. The only ready reserved central2 v4-2048 is occupied by the active June 10T run, so the new jobs may queue; do not move them to another region.
- Operational sequence: snapshot and push the tested launcher, submit only the 100-step two-arm smoke with `MARIN_PREFIX=gs://marin-us-central2` and a central2 controller, then babysit both children through terminal W&B and artifact checks. Do not submit the 3,000-step milestone until the smoke gate is applied.

### 2026-08-08 05:50 - GRUG-XEM-009 implementation and prelaunch snapshot

- Commit Hash: `7be2cf679a2a6337245792c7b0621cfb00a00cf4`.
- Implementation: added the central2-only matched June 67B launcher, phase-selectable 100-step smoke and 3,000-step milestone, explicit 26-layer core grouping, unscaled stacked-bank MuonH treatment, region-prefixed datakit inputs, and full-schedule LR timing. Ported cross-loop top-1/top-k agreement, per-layer activation norms, topology/unique-parameter summaries, and per-bank whole-tree gradient/update norms into the June array-stacked path.
- Behavioral coverage: tied stacked checkpoint round-trip; stacked forward parity; exact shared-bank gradient accumulation against duplicated untied use sites; independent per-layer router/QB updates; independent per-layer always-on dense MLPs; known-value cross-loop statistics; bank-wise norm reduction; and a four-CPU-device data-plus-expert-axis JIT regression.
- Validation: 52 project-relevant tests passed and one unrelated variant-contract case skipped. The broader command's sole failure was an unchanged `experiments/grug/base` CPU explicit-sharding test, outside the June variant and failing before this diff's code paths. Changed-file pre-commit, Pyrefly, diff checks, and no-run lowering passed. Independent focused runs over the June suite passed 34/34.
- Immutable smoke identities: baseline fingerprint `87630764` at `gs://marin-us-central2/grug/tied_experts/june67b/smoke/baseline/2026.08.08`; tied fingerprint `7f976538` at `gs://marin-us-central2/grug/tied_experts/june67b/smoke/middle_groups_unscaled/2026.08.08`.
- Operational audit: `marin-us-central2` reports location `US-CENTRAL2`; all 200 datakit prefixes and 23 eval markers exist there. The active June 10T run occupies the sole ready reserved central2 v4-2048, so the two smoke children are expected to queue. Snowball remains in east-region S3 and is not an input.
- Next action: require a zero-context issue reread to pass, verify both smoke roots are empty, push this logbook snapshot, launch one central2 controller for only the two 100-step arms, and assign a dedicated babysitter. Do not launch the 3,000-step milestone yet.

### 2026-08-08 05:32 - GRUG-XEM-009 smoke launch

- Commit Hash: implementation `7be2cf679a2a6337245792c7b0621cfb00a00cf4`; preregistration `d093d3bf81`; zero-indexed counter clarification and launch snapshot `e7260c050a`.
- Command: controller `/dlwh/grug-xem-june67b-smoke-20260808` runs `python -m experiments.june_tpu_67b_a2b.moe.launch_tied_experts --version 2026.08.08 --run --max-concurrent 1` with `MARIN_PREFIX=gs://marin-us-central2`, a central2 CPU controller, production priority, and non-preemptible scheduling. The controller serializes the two v4-2048 children.
- Immutable identities: baseline fingerprint `87630764`; tied fingerprint `7f976538`. Expected output roots are `gs://marin-us-central2/grug/tied_experts/june67b/smoke/baseline/2026.08.08` and `gs://marin-us-central2/grug/tied_experts/june67b/smoke/middle_groups_unscaled/2026.08.08`.
- Preflight: both roots were empty immediately before submission. The full gate, zero-indexed W&B step convention, exact token counts, initialization coupling, and regional contract are in issue #8032. A zero-context issue reader returned PASS after the definitions were made mechanical.
- Operations: job submission succeeded at 2026-08-08 05:31 PDT. Capacity waiting is expected while the active June 10T run occupies the ready central2 v4-2048. A dedicated babysitter owns the controller through terminal state. Recovery may stop and resubmit only this controller after a diagnosed topology-preserving failure; no cluster mutation, region change, Snowball read, teacher-checkpoint read, or 3,000-step milestone launch is authorized.
- Issue update: https://github.com/marin-community/marin/issues/8032#issuecomment-5226116533
- Next action: verify both W&B arms and central2 artifacts through terminal state, apply the preregistered smoke gate, and launch the 3,000-update milestone only on a full pass.

### 2026-08-08 06:53 - GRUG-XEM-009 capacity and central1 locality audit

- Status: the central2 controller is healthy. Its serialized baseline child is pending on the 256-worker `v4-2048` coscheduling group; no W&B run or checkpoint exists before allocation. The tied child has not been submitted because maximum concurrency is one.
- Capacity estimate: the active June 10T run was at zero-indexed step 118,052 of 150,000. Over its latest 200 steps it averaged 26.593 seconds/step, implying about 236 hours to finish if no additional central2 slice becomes available.
- Central1 data audit: bucket metadata confirms `marin-us-central1` is `US-CENTRAL1`, but `gs://marin-us-central1/datakit/store_8ac06c74/` has no objects. All 200 configured phase-0 prefixes `cluster={0..39}/quality={0..4}` are therefore missing, and the launcher has `auto_build_caches=False`. All 23 configured Paloma and Uncheatable eval artifacts exist in central1, but they do not repair the training-cache absence. No source checkpoint is configured; the smoke starts from seed-0 initialization.
- Central1 capacity audit: no central1 zone exposes the registered `v4-2048` accelerator type, and Iris has no central1 `v4-2048` scaling group or worker. Central1 `v5p-2048` would be a different hardware experiment.
- Decision: leave the controller queued in central2. Do not copy caches across regions, regenerate an unvalidated substitute cache, or change hardware. Continue monitoring for an additional central2 slice or release of the active one.
- Issue update: https://github.com/marin-community/marin/issues/8032#issuecomment-5226403715

### 2026-08-08 07:17 - GRUG-XEM-009 exact-cache regeneration audit

- Question: can the exact `datakit/store_8ac06c74` training cache be regenerated entirely within central1, avoiding both the long central2 queue and cross-region transfer?
- Provenance: the 200 cache leaves are produced by a 116-source Datakit DAG with 465 direct logical dependencies: per-source tokenize, decontaminate, cluster-assignment, and quality outputs plus one global dedup artifact. The final store contains 10,372,343,704,053 tokens. Central2 object metadata implies about 25 TB compressed; the logical int32 token payload is 41.49 TB.
- Missing central1 inputs: the exact May centroid model, combined eval bloom, global dedup artifact, Sonnet-4.6 quality model, and production intermediate trees are absent. Central1 has nearby normalized/raw sources for much of the mixture, but not the immutable intermediate bytes needed for exact replay.
- Fingerprint limitation: StepSpec hash `8ac06c74` covers dependency names and settings, not implementation code or output bytes. Recomputing embeddings, K-means assignments, quality inference, fuzzy deduplication, and the store under current code could reuse the suffix without proving the registered data content or order.
- Cost estimate: the historical final store stage alone used 6,301 tasks, scaled to 3,072 workers at 2 CPU/32 GiB each, and took about 16.5 hours. Rebuilding the missing upstream graph would be substantially larger.
- Decision: exact central1-only regeneration is NO-GO. Continue waiting for central2. Do not copy or rebuild the cache and do not substitute a central1 v5p experiment without a separate preregistration and an identical local training input.
- Issue update: https://github.com/marin-community/marin/issues/8032#issuecomment-5226495632

### 2026-08-08 09:19 - GRUG-XEM-009 canceled before TPU allocation

- User decision: stop the central2 smoke and revisit the large-model capacity plan separately.
- Action: stopped only `/dlwh/grug-xem-june67b-smoke-20260808`. Iris reports the controller and its pending baseline child as `killed`. The incumbent June run and Iris cluster were untouched.
- Artifact audit: the baseline never acquired the 256-worker `v4-2048` group, the tied child was never created, and neither arm produced W&B data or a checkpoint. Each planned root contains only `.executor_info` and a six-byte `.executor_status` containing `FAILED`; no training payload exists.
- Interpretation: GRUG-XEM-009 is operationally inconclusive. Cancellation does not change the d512/d768 architecture evidence or any d512 conversion result. The 3,000-update milestone remains unlaunched.
- Issue update: https://github.com/marin-community/marin/issues/8032#issuecomment-5226971210
- Resume condition: preregister a new data-local capacity plan. Do not copy the approximately 25 TB cache across regions or treat a rebuilt cache/hardware substitution as the same experiment.

### 2026-08-08 09:33 - GRUG-XEM-010 correspondence-free joint refactorization design

- Hypothesis: the native Hungarian initialization traps hard top-4 recovery in a poor expert-slot factorization. Training the representative bank and both affected routers jointly against complete cached teacher MoE outputs, without consuming teacher expert IDs, combine weights, assignments, or a matching artifact, can produce a better sparse factorization.
- Nonredundancy audit: current Stage A and Stage B already minimize complete teacher-on-student-state routed-output error. `native_joint` already trains the bank and routers directly from native conversion. Native aggregate prefit already matches complete cached routed outputs, but it replays Hungarian-mapped teacher IDs and teacher combine weights while keeping routers frozen. GRUG-XEM-010 is new only because its cached batches contain layer ID, MLP input, and complete routed output, while the student generates its own production routes and both routers train from the beginning.
- Initialization: load the untied d512 teacher at step 10993, retain layer 2's expert bank as the shared bank, remove layer 3's bank, and keep both layers' router matrices, router biases, and pending QB values in their original unpermuted column order. This is a representative-bank/no-permutation starting point, not identity expert supervision. Router biases and pending QB values remain frozen during offline refactorization.
- Offline objective: use the existing 8,192-state calibration trace per layer with a deterministic 80/20 split. Sample 256 train states per layer per step and retain a fixed 512-state held-out batch per layer. For each layer, run its student router and the common bank under the production QB top-4 path, then minimize the layer-balanced normalized complete routed-output MSE. Train only the one bank and two router matrices with AdamW `1e-4`, zero weight decay, 2,000-step cap, evaluation every 100 steps, and five-evaluation held-out patience. Do not use a dense all-256 or relaxed router in this first test.
- Offline gate: require finite losses, zero held-out capacity overflow, nonzero held-out traffic to all 256 experts in both layers, and routing entropy at least 5.3. The best held-out mean normalized MSE must be at most `0.349847`, 10% below the completed fixed-route native aggregate-prefit best `0.388719`. Otherwise stop without online recovery.
- Online screen: if the offline gate passes, convert the selected refactor state into topology `(0,1,2,2,3,4)` with an explicit correspondence-free manifest, reset optimizer/step state, and run 25,034,752 online tokens of direct preservation. Train the shared bank and layer-2/3 router matrices, resume ordinary QB updates, and use CE `1.0` + teacher-logit KL `0.1` + routed-MoE loss `1.0`. The teacher MoE remains evaluated on the current student state. Attention, norms, dense MLPs, embeddings/head, and unmerged banks remain frozen.
- Online signal gate: the completed native direct-joint control at 25,034,752 tokens is validation `+0.064923` and Paloma `+0.065779`. GRUG-XEM-010 must reach validation `<= +0.059923` and Paloma `<= +0.060779`, improving both by at least 0.005. Require finite metrics, zero overflow, all 256 experts active in each affected layer, and routing entropy at least 5.3. Expert-ID agreement is undefined for layer 3 and must be recorded as such rather than compared under an implicit identity map.
- Decision: a signal pass permits a separately launched continuation to 100M online tokens, where the original surgery promotion bounds remain validation `<= +0.020` and Paloma `<= +0.030`. Failure of either 25M quality bound stops correspondence-free hard-top-4 refactorization and the post-hoc surgery line. It does not falsify dense or relaxed routing, which would be a different architecture/compute experiment.
- Regional contract: teacher, calibration trace, train/eval caches, v5p-8 worker, outputs, and checkpoints remain in `us-central1`. The refactor graph has no matching dependency and reads no Snowball or central2 artifact.
- Prior-work update: [ConMoE](https://arxiv.org/abs/2605.29350) supports local cross-layer prototype reuse but shows distance-only selection and parameter fusion are weak; it freezes routers and does no recovery. [Router KD](https://arxiv.org/abs/2603.02217) shows that compression-induced router mismatch can be recoverable but router recalibration is not sufficient on every task. [REAP](https://openreview.net/pdf?id=ukGxWd2aDG) finds that fixed expert merging can destroy input-dependent functional breadth. [MC-SMoE](https://arxiv.org/abs/2310.01334) aligns hidden neurons for within-layer parameter averaging and does not support cross-layer expert-ID matching. [HC-SMoE](https://proceedings.mlr.press/v267/chen25aq.html) instead clusters by functional outputs. [Jaggi](https://arxiv.org/abs/2606.16825) remains tied-from-initialization architecture evidence only. No primary source was found that performs this exact post-hoc cross-layer shared-bank-plus-router refactorization.
- Next action: implement the correspondence-free numerical core, structural conversion and format-4 provenance, a dedicated refactor worker/launcher, and behavior-focused tests. Before launch, require central1 path validation, empty output roots, four-device lowering, pushed snapshot, issue-body update, and a zero-context issue reread.

### 2026-08-08 10:24 - GRUG-XEM-010 implementation snapshot

- Snapshot: commit `826185bd63` on `research/grug-matcher-jit` implements the correspondence-free numerical core, exact format-4 conversion provenance, resumable offline worker, no-matching online recovery initialization, central1-only launcher, and focused tests.
- Lowered graph: offline fingerprint `bd1de543`; 25.03M-token screen fingerprint `77d28282`. Dependencies are the adopted central1 teacher, the adopted central1 calibration trace, and the existing `2026.06.28` Grug train/evaluation caches. The graph contains no matching, Snowball, or central2 dependency.
- Verification: `pre-commit.py` passed on all changed files; `pyrefly` reported zero errors; 7 expert-refactor tests, 19 checkpoint/launcher tests, 12 recovery-objective tests, and the full local offline-to-online worker test passed. Coverage includes a four-device data-sharded JIT update, route-field erasure, permutation equivariance, router-gradient sensitivity, strict manifest reconstruction, idempotent offline output, and undefined expert-ID agreement for the correspondence-free layer.
- Launch remains blocked on the preregistration communication gate: replace issue #8032's ambiguous historical body, define Stage A and Stage B before use, and obtain a pass from a fresh context-free reader. Then confirm both versioned output roots are empty before submission.

### 2026-08-08 12:14 - GRUG-XEM-010 launched

- Communication gate: issue #8032 was rewritten as a self-contained current record. A first context-free reader found eight remaining reproducibility gaps; after exact data fingerprints, RNG draw order, optimizer/precision, router/QB math, loss reductions, entropy, delta keys, and proof paths were added, a second context-free reader returned PASS.
- Preflight: pushed head `293fc3d94a`; offline fingerprint `bd1de543`; screen fingerprint `77d28282`. Both central1 versioned roots matched no objects immediately before submission. The lowered graph contains no matching, Snowball, central2, or cross-region dependency.
- Controller: `/dlwh/grug-xem-010-joint-refactor-screen-20260808`, https://iris.oa.dev/#/job/%2Fdlwh%2Fgrug-xem-010-joint-refactor-screen-20260808. It runs the offline gate first and launches the exact 25,034,752-token screen only on a pass. No 100M continuation is present.
- Issue launch record: https://github.com/marin-community/marin/issues/8032#issuecomment-5227687564

### 2026-08-08 12:34 - GRUG-XEM-010 failed the offline gate

- Terminal state: the controller and offline child failed on the intended scientific gate after 18m55s controller wall time. The worker early-stopped at update 800 after five stale evaluations; automatic Iris retries re-read the deterministic failed state and terminated. No manual restart, resubmission, or mutation occurred.
- Selected result: best update 300, held-out mean normalized MSE `0.3762871325` versus required `<=0.349847`. This is only a 3.20% reduction from the frozen fixed-route native aggregate comparator `0.3887194693`, versus the required 10%.
- Per-layer diagnostics: held-out NRMSE `0.2266446352/0.8373807073`; entropy `5.3723006/5.3716583` passed; overflow `0/0` passed; active experts `251/253` failed the required `256/256`. Training loss reached `0.012866` at update 800 while held-out selection stayed at update 300, showing cached-trace overfit.
- Proof: `gs://marin-us-central1/grug/expert_merge/d512/joint-refactor-layers-2-3/2026.08.08/refactor_checkpoints/step-800/joint_refactor_training_manifest.json`. There is no converted `checkpoints/step-0`; the Iris graph contains no online child; the screen root still matches no objects; no W&B run or quality evaluation exists.
- Decision: stop correspondence-free cached hard-top-4 refactorization and the current post-hoc surgery line. Do not retry, run the online screen, implement the 100M continuation, or advance to two-pair, d768, or post-hoc 67B surgery. This does not change the positive tied-from-scratch d512/d768 architecture result.
- Issue result: https://github.com/marin-community/marin/issues/8032#issuecomment-5227768427

### 2026-08-08 13:05 - GRUG-XEM-011 d1024 architecture-scale preregistration

- Hypothesis: the tied-from-scratch quality penalty remains at most +0.04 Paloma at d1024 when the 11-layer model keeps two untied input anchors, two untied output anchors, and ties the seven-layer core in groups of four and three.
- Protocol deviation: tied experts had effective speed `0.846x` at d512 and `0.849x` at d768, so they failed the formal `agent.md` Gate-1 requirement for Gate 2. GRUG-XEM-011 is authorized by the user as an architecture-scaling test with free TPU time and is not a production or effective-speed promotion experiment.
- Matched arms: control topology `(0,1,2,3,4,5,6,7,8,9,10)`; treatment `(0,1,2,2,2,2,3,3,3,4,5)`. The treatment has two anchors at each end, reused-bank group sizes four and three, six unique banks instead of eleven, and removes 45.45% of unique routed banks. Both use unscaled MuonH, seed 0, sequence length 4096, global batch 128, identical current Grug defaults, and the same central1 `2026.06.28` mixture/data order.
- Smoke: 500 updates and exactly 262,144,000 tokens per arm on one central1 v5p-8 each. Both must finish at zero-indexed W&B step 499 with exact total tokens, permanent step-500 checkpoint/artifact, finite CE and expert-bank gradient/update metrics, and no compile/HBM/checkpoint failure. Across steps 450-499, require zero capacity overflow at every layer/step, traffic to all 256 experts in every layer over the union, and mean hard-route entropy at least 5.3 separately per layer. The treatment's reused banks 2 and 3 must have finite median gradient and update norms over steps 20-499; for each metric, the larger median divided by the smaller must be at most 4. The treatment-minus-control median CE over steps 450-499 must be at most +0.15.
- Full schedule: launch fresh matched arms only after both smoke arms pass. Current heuristic gives 16,149 updates and exactly 8,466,726,912 tokens per arm. The architecture passes if final tied-minus-control Paloma macro is `<=+0.040` and the final 100-step window passes the smoke routing/finite checks. Report last-100 throughput, unique parameter count, bank gradient/update norms, activation norms, and effective speed without using effective speed as a veto.
- Scale diagnosis: raw penalty `<=+0.028548` is diminishing relative to d768. The d768 penalty per removed-bank fraction is `0.028548/(3/8)=0.076128`; multiplying by d1024's `5/11` removed fraction gives `+0.034604`. A d1024 gap `<=+0.034604` is compression-normalized non-worsening; `(0.034604,0.040]` remains architecture-viable but indicates the normalized penalty is growing; `>+0.040` stops d1280.
- Regional contract: both arms initialize from seed 0 and read no checkpoint. Controller, nine training caches, 23 validation artifacts, v5p-8 workers, profiles, checkpoints, evaluations, and outputs remain in `us-central1`. Snowball and the central2 June cache are not inputs. Any cross-region dependency fails preflight rather than being copied.
- Progression: a full d1024 pass permits a separately preregistered d1280 matched comparison; it does not unblock post-hoc d1024/d1280/67B surgery. A smoke failure stops before full training. A full failure stops larger tied-from-scratch scaling under this topology.

### 2026-08-08 13:10 - GRUG-XEM-011 d1024 smoke launch

- Communication gate: issue #8032 now defines Stage A and Stage B before use and contains the active d1024 model, optimizer groups, effective zero weight decay, QB equations, data ordering, exact Iris commands, W&B sample cardinalities, median rules, proof sources, authority, and stop conditions. A third reader with no conversation context returned `PASS` for the active XEM-011 section.
- Immutable code: commit `7d6fa47220e8e24a4884802748cfb6cf915dfe3d`, pushed on `research/grug-matcher-jit`. Smoke fingerprints are baseline `e951ba9c` and treatment `2ab01bdb`.
- Preflight: `gs://marin-us-central1` reports `US-CENTRAL1`. Both versioned smoke roots matched no objects immediately before submission. No-run lowering from the immutable commit resolved the nine registered training caches, 16 Paloma components, and seven Uncheatable components; its serialized graph contains no `central2`, Snowball, S3, or east-region reference. The worktree and upstream were both at the immutable commit.
- Command: `.venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait --region us-central1 --cpu 1 --memory 2GB --extra cpu --job-name grug-xem-011-d1024-smoke-20260808 -e MARIN_PREFIX gs://marin-us-central1 -e GRUG_TIED_MODEL d1024 -e GRUG_TIED_PHASE smoke -- python -m experiments.grug.moe.launch_tied_experts --version 2026.08.08 --run --max-concurrent 2`.
- Controller: `/dlwh/grug-xem-011-d1024-smoke-20260808`, https://iris.oa.dev/#/job/%2Fdlwh%2Fgrug-xem-011-d1024-smoke-20260808. Submission completed at 13:09:46 PDT. The controller may run the two central1 v5p-8 children concurrently.
- Monitoring: one dedicated babysitter owns the controller and both W&B arms through terminal state, exact history-cardinality checks, terminal evaluation, step-500 checkpoint/artifact checks, and mechanical smoke-gate application. It may not mutate a cluster, change region/hardware/data/version, or launch full training.
- Next action: wait for both arms to finish. Launch the fresh 16,149-update comparison only if every registered smoke condition passes.

### 2026-08-08 14:02 - GRUG-XEM-011 smoke pass and full launch

- Smoke terminal state: controller and both children succeeded without retry. W&B state is `finished` for both registered run IDs at zero-indexed step 499 and exactly 262,144,000 tokens. Each arm has 500 unique finite CE steps; 500 finite gradient/update samples for every expected bank; and 500 routing, histogram, overflow, and activation samples for every layer. Both registered roots contain `.artifact.json` and `checkpoints/step-500/metadata.json`; all 23 terminal evaluation components are present.
- Smoke quality and health: terminal Paloma is control `4.313209534` and treatment `4.335586548`, a descriptive `+0.022377014`. Final-50 CE medians are `3.936008096` and `3.966751814`, delta `+0.030743718` versus the `<=+0.15` gate. Final-50 overflow is zero at every layer/step, all 256 experts are active in every layer, and minimum per-layer mean entropy is `5.523508` control and `5.519211` treatment versus `>=5.3`.
- Shared-bank update health: treatment bank-2/bank-3 gradient medians over steps 20-499 are `0.02470750455/0.01758269500`, ratio `1.405217`; update medians are `2.655682087/2.654492140`, ratio `1.000448`. Both are below the ratio-4 limit.
- Decision: every smoke condition passes. This authorizes the separately initialized full comparison under the issue's standing user-authority rule. The smoke metrics are not substituted for the full architecture gate.
- Full preflight: both full roots matched no objects immediately before submission. The branch was clean and pushed at `eff3b300f158908733539fa15616ef6e2621e72c`; experiment code remains the immutable `7d6fa47220` snapshot, with only the append-only launch log added afterward. Full fingerprints are baseline `87745e4c` and treatment `5ccacf53`.
- Full command: `.venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait --region us-central1 --cpu 1 --memory 2GB --extra cpu --job-name grug-xem-011-d1024-full-20260808 -e MARIN_PREFIX gs://marin-us-central1 -e GRUG_TIED_MODEL d1024 -e GRUG_TIED_PHASE full -- python -m experiments.grug.moe.launch_tied_experts --version 2026.08.08 --run --max-concurrent 2`.
- Full controller: `/dlwh/grug-xem-011-d1024-full-20260808`, https://iris.oa.dev/#/job/%2Fdlwh%2Fgrug-xem-011-d1024-full-20260808. Submission completed at 14:02:23 PDT. A dedicated babysitter owns both arms through terminal W&B, step-16149 artifacts, exact final-100 history checks, and gate application.
- Next action: wait for both 8,466,726,912-token arms. A full Paloma delta `<=+0.040` with all health checks passes the architecture gate; no post-hoc larger surgery is authorized.

### 2026-08-09 05:41 - GRUG-XEM-011 d1024 full result

- Terminal state: the controller and both children succeeded without retry. Both W&B runs are `finished` at zero-indexed step 16148 and exactly 8,466,726,912 tokens. Both registered roots contain `.artifact.json` and `checkpoints/step-16149/metadata.json`; all 23 terminal component losses are finite.
- Quality: terminal Paloma is control `3.039182186` and treatment `3.069946766`, delta `+0.030764580`. This passes the `<=+0.040` architecture gate and the `<=+0.034604` compression-normalized non-worsening threshold. The raw penalty is `+0.002216580` larger than d768's `+0.028548`, so the raw gap stopped diminishing.
- Exact final-100 audit: both arms contain one complete finite sample for every required metric at each step 16049-16148. Capacity overflow is zero in every layer/step; every layer uses all 256 experts over the window; per-layer mean entropy ranges `5.5307-5.5381` control and `5.5272-5.5386` treatment.
- Shared-bank health: treatment bank-2/bank-3 gradient medians are `0.0376510/0.0354184`, ratio `1.06304`; update medians are `0.1438597/0.1438694`, ratio `1.00007`. Cross-loop top-1 agreement median is `0.0315427`; top-4 set-overlap median is `0.0747550`.
- Throughput and parameters: final-100 median throughput is control `166,013.747` and treatment `168,767.628` tokens/s, a 1.659% treatment increase. Effective speed is `0.811946x` with `C_needed=9.26489e18`, so the experiment does not pass the formal compute-speed protocol. Unique counts are control `4,764,584,704` total/`4,429,185,024` expert and treatment `2,751,318,784` total/`2,415,919,104` expert. The treatment removes `2,013,265,920` expert parameters: 45.455% of expert and 42.255% of whole-model unique parameters.
- Activation medians, layers 0-10: control `[0.055693,0.063832,0.074180,0.089233,0.110958,0.146620,0.187044,0.235773,0.308889,0.397997,0.581765]`; treatment `[0.054812,0.068244,0.078559,0.093141,0.111923,0.138898,0.172576,0.216190,0.276104,0.377443,0.592719]`.
- Interpretation: the architecture hypothesis remains alive at d1024. The raw loss penalty did not continue shrinking, but it remained acceptable after accounting for the larger removed-bank fraction. Low cross-loop agreement confirms independent routers use the common dictionary differently. Tying is not a compute-speed win under this schedule.
- Decision: promote H14 and prepare the separately preregistered d1280 architecture comparison. Do not launch post-hoc d1024/d1280/67B surgery from this result.

### 2026-08-09 05:50 - GRUG-XEM-012 d1280 architecture preregistration

- Hypothesis: a d1280 model with two anchors at each end, two tied four-layer core groups, and one singleton core layer remains within +0.040 Paloma of a contemporaneous untied control. The raw d1024 penalty stopped diminishing, so this is the final registered central1 architecture-scale test, not a compute-speed promotion experiment.
- Matched arms: control topology `(0,1,2,3,4,5,6,7,8,9,10,11,12)`; treatment `(0,1,2,2,2,2,3,3,3,3,4,5,6)`. Layers 0-1 and 11-12 are anchors, layers 2-5 share bank 2, layers 6-9 share bank 3, and core layer 10 is singleton bank 4. The treatment uses seven banks instead of thirteen and removes 3,774,873,600 expert parameters, 46.154% of the control expert count.
- Resolved config: hidden 1,280; 13 layers; routed intermediate 640; shared dense intermediate 1,280; 256 top-4 experts; sequence 4,096; global batch 256; seed 0; unscaled MuonH. Matrix LR is `0.009572908762445573`, Adam LR `0.002209132791333594`, beta1 `0.9062`, beta2 `0.992027944069944`, epsilon `1.1576799437233098e-15`, no clipping, effective zero explicit weight decay, 143 warmup steps, and a full 14,315-step schedule horizon.
- Immutable identities: implementation `7d6fa47220`; smoke fingerprints control `235ee64c` and treatment `d7dfe006`; full fingerprints control `a1a2b02c` and treatment `27a96496`; output version `2026.08.09`.
- Smoke gate: 500 updates and exactly 524,288,000 tokens per arm. Require W&B finished at zero-indexed step 499, exact 500-sample finite histories for all required layer/bank metrics, all 23 terminal evaluation components, permanent step-500 artifacts, zero final-50 overflow, all 256 experts active per layer, mean entropy at least 5.3, tied bank-2/bank-3 gradient/update ratios at most four, and final-50 CE-median delta at most +0.15. Missing or duplicate samples fail.
- Full gate: on a complete smoke pass, launch fresh 14,315-update arms with exactly 15,010,365,440 tokens each. Require terminal Paloma delta `<=+0.040` and the exact final-100 health/proof audit over steps 14215-14314. Raw non-growth is `<=+0.030764580`; compression-normalized non-worsening is `<=+0.031237881`, from `(0.030764580/(5/11))*(6/13)`.
- Regional contract: both arms start from seed 0 and read no checkpoint. All controller work, nine training caches, 23 evaluation artifacts, v5p-8 workers, W&B, profiles, checkpoints, and outputs remain in `us-central1`. Reject central2, Snowball, S3, east-region, or other cross-region material references. Both phase roots must be empty immediately before first submission.
- Communication gate: issue #8032 now contains the exact d1280 model, optimizer, topology, arithmetic, commands, sample cardinalities, gates, proof paths, authority, and stop rules. A context-isolated reader using only that body returned `PASS`.
- Progression: a smoke pass authorizes full automatically. XEM-012 completion authorizes no further width or post-hoc surgery; a full failure stops autonomous scaling under this topology.
