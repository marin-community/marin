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
- `GRUG-XEM-002` completed the layers 2-3 spectral-plus-prefit conversion in `us-central1`. After 50M local-recovery and 200M preservation tokens, the student is +0.02611 validation loss and +0.02638 Paloma macro loss above the untied teacher. This recovers about two-thirds of the initial gap, but misses the +0.01-0.02 validation target and reaches the Paloma +0.03 bound only after the proposed 100M-token budget.
- `GRUG-XEM-003` ports explicit expert banks and legacy-checkpoint migration into the array-stacked June 67B-A2B implementation. The selected no-copy teacher, data caches, eval caches, output bucket, and TPU resources are all in `us-central2`; no checkpoint payload has been read outside that region.
- `GRUG-XEM-004` is the d768 scale comparison tracked in issue #8032. Its matched untied, two-anchor unscaled, and two-anchor `1/sqrt(g)` smoke configurations are ready on central1-only data and outputs.
- The isolated branch is `research/grug-cross-layer-expert-merging` in `/tmp/marin-grug-xem`. It has not been pushed, and no PR exists.

## Baseline

- Date: 2026-08-05
- Code ref: `c26285a61654a9e6a9029cfdb3d018badc35d71c`
- Current d512 size reference: 6 layers, batch 32, sequence length 4096, 10,980 steps, about 1.44B tokens. The recorded Paloma value was measured under older attention and loss defaults, so it is not the matched control for this experiment.

## Hypothesis Queue

### Active

- `GRUG-XEM-H4`: Functional matching plus shared-bank prefit can merge one adjacent middle-layer pair with partial aggregate recovery. Current result: spectral-plus-prefit recovered about two-thirds of the teacher gap after 250M total recovery tokens, but did not pass the strict single-pair gate. Next test: compare against the tied-from-scratch reference before changing the recovery objective or expanding to two pairs.
- `GRUG-XEM-H6`: A recent June 67B-A2B checkpoint admits a one-pair middle-layer merge without cross-region checkpoint or data movement. Next test: validate legacy schema migration locally, then dispatch a central2-only one-pair smoke from step 105149 after the d512 surgery gate passes.
- `GRUG-XEM-H7`: The tied-minus-untied loss gap diminishes at d768 when four middle layers share one bank behind two-layer input and output anchors. Next test: matched d768 untied, unscaled-tied, and `1/sqrt(g)`-tied runs at `2.81e18` FLOPs.

### Blocked

- `GRUG-XEM-H6`: Blocker: d512 surgery has not passed and the June checkpoint adapter is not yet validated. Resume when: both gates pass.

### Falsified / Dead End

- `GRUG-XEM-H5`: Spectral matching missed its gate. Relative to native-only matching it improved the common assignment objective by 0.5%, Stage-A MoE loss by 1.1%, and the final combined spectral-plus-prefit recovery gap by 6-7%; none reaches the required 15%/20% margin. Keep spectral probes as diagnostics, not the production initializer.

### Promoted

- `GRUG-XEM-H1`: Pairwise d512 tying is stable and within the +0.03 Paloma macro screening gate on a matched full run.
- `GRUG-XEM-H2`: Middle-four d512 tying is stable and within the +0.06 Paloma macro screening gate on a matched full run.
- `GRUG-XEM-H3`: The LR ablation did not support `1/sqrt(g)` as best for this d512 MuonH recipe; unscaled tying was slightly better at full schedule for both topologies. Keep LR scaling configurable rather than treating Jaggi's setting as a Grug default.

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
