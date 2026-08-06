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
- Coordinating issue/PR: none yet.
- Experiment series: `GRUG-XEM`.

## Current TL;DR

- `GRUG-XEM-001` has local snapshots for explicit expert-bank ownership, tied-bank LR ablations, and the seven-run d512 architecture smoke matrix. No accelerator run has launched.
- The one-pair layers 2-3 conversion prototype now covers calibration reservoirs, native/spectral assignment, balanced bank prefit, topology-native checkpoint conversion, and constrained recovery. It has local numerical tests but remains gated on the architecture smoke and a contemporaneous untied checkpoint.
- The isolated branch is `research/grug-cross-layer-expert-merging` in `/tmp/marin-grug-xem`. It has not been pushed, and no issue or PR exists.

## Baseline

- Date: 2026-08-05
- Code ref: `c26285a61654a9e6a9029cfdb3d018badc35d71c`
- Current d512 size reference: 6 layers, batch 32, sequence length 4096, 10,980 steps, about 1.44B tokens. The recorded Paloma value was measured under older attention and loss defaults, so it is not the matched control for this experiment.

## Hypothesis Queue

### Active

- `GRUG-XEM-H1`: Pairwise middle-layer routed-expert tying `(0,1,1,2,2,3)` is stable and finishes within 0.03 Paloma macro loss of a matched untied control. Next test: 500-step d512 smoke after topology and optimizer tests pass.
- `GRUG-XEM-H2`: Four middle layers sharing one routed-expert bank `(0,1,1,1,1,2)` remain within 0.06 Paloma macro loss of the matched control without routing or update concentration. Next test: add after pairwise smoke is healthy.
- `GRUG-XEM-H3`: Scaling a bank's expert LR by `1/sqrt(g)` avoids update-scale artifacts and outperforms an unscaled LR; `1/g` is the conservative control. Next test: matched d512 smoke ablation.
- `GRUG-XEM-H4`: After architecture validation, functional matching plus shared-bank prefit can merge one adjacent middle-layer pair more efficiently than identity-ID conversion. Next test: collect layers 2-3 calibration states from the matched untied checkpoint after the architecture gate.
- `GRUG-XEM-H5`: Covariance-aware finite-difference spectral matching materially improves immediate MoE error, validation-loss spike, or recovery tokens over native-state-only matching. Next test: compare saved identity, native-only, and spectral cost matrices on the same calibration artifact; drop spectral probes if they miss the stated 15%/20% gates.

### Blocked

- `GRUG-XEM-H4`: Blocker: architecture gate and source checkpoint. Resume when: d512 baseline and tied-from-scratch runs finish.
- `GRUG-XEM-H5`: Blocker: no trained untied checkpoint or calibration artifact. Resume when: the d512 architecture gate finishes and calibration runs in the checkpoint region.

### Falsified / Dead End

- None.

### Promoted

- None.

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
