# Grug MoE d2560 MFU: Research Logbook

## Scope
- Goal: make the issue #6044 d=2560 May Recipe Grug MoE shape fast on CoreWeave H100s.
- Primary metrics: training MFU >= 20%, tokens/sec, step time, profile-attributed time in MoE, attention, FSDP collectives, and optimizer/update.
- Constraints: use the Grug MoE experiment path, keep R2 configuration in durable user config, avoid cross-region storage movement, and keep each optimization claim tied to a repeatable profile.
- Tracking issue: https://github.com/marin-community/marin/issues/6367
- Source recipe: https://github.com/marin-community/marin/issues/6044#issuecomment-4607416665
- Branch: `codex/research-grug-moe-d2560-mfu`
- Experiment prefix: `GM2560-MFU`

## Baseline
- Date: 2026-06-13
- Code refs:
  - `experiments/grug/moe/launch_cw_may_d2560.py`
  - `experiments/grug/moe/model.py`
  - `experiments/grug/moe/optimizer.py`
  - `experiments/grug/moe/train.py`
- Baseline numbers: not measured yet on H100. The first required result is a profile-bearing run of the issue #6044 d=2560 shape.
- Fixed baseline case: `MAY_GPU_REPLICAS=32`, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_REMAT=save_moe`, `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`, `MAY_CHECKPOINTS=local`, `MAY_DATA=slimpajama` unless the Nemotron mix is available on CoreWeave-readable storage.

## Initial Hypotheses
- GM2560-MFU-H1: sharded optimizer state is already present because optimizer state is initialized from explicitly sharded params; the next memory/perf question is whether persistent bf16 live params plus an fp32 master tree is worth implementing.
- GM2560-MFU-H2: with the d=2560, 256-expert shape, MoE dispatch/combination and expert matmuls are likely the main bottleneck at small per-device batch; ring EP with `MAY_EXPERT_AXIS=8` should be the first H100 baseline.
- GM2560-MFU-H3: attention may dominate once MoE remat is reduced via `save_moe`, especially with 4096 context and PKO/half-RoPE.
- GM2560-MFU-H4: FSDP all-gather and optimizer update overhead may become visible with fp32 params; `params=bfloat16` is a diagnostic knob, but not the requested fp32-master mode.

## First Experiment Matrix
- GM2560-MFU-001: compile/sanity dry run of the dedicated launcher config; no cluster submission.
- GM2560-MFU-002: 32-node H100 short profile with the fixed baseline case and profiler enabled for a short post-warmup window.
- GM2560-MFU-003: if GM2560-MFU-002 is below 20% MFU, classify profile time into MoE, attention, FSDP collectives, optimizer/update, and data/input.
- GM2560-MFU-004: one-axis follow-ups only: EP axis, remat mode, precision policy, optimizer grouping, and attention backend.

## Stop Criteria
- Stop or seal the milestone when a repeatable H100 profile shows >= 20% MFU for the issue #6044 shape and the research issue/logbook identify remaining bottlenecks.
- Escalate before long training if a short profile shows persistent < 20% MFU with no single dominant optimization target.

## Experiment Log

### 2026-06-13 22:27 PDT - GM2560-MFU-001 kickoff
- Hypothesis: a dedicated launcher plus explicit profile loop is needed before making more performance changes.
- Command:
  - `gh api repos/marin-community/marin/issues/comments/4607416665 --jq '{url: .html_url, body: .body, created_at: .created_at, updated_at: .updated_at}'`
  - `git switch -c codex/research-grug-moe-d2560-mfu`
- Config: issue #6044 d=2560 May Recipe shape, current uncommitted launcher defaults.
- Result: source recipe verified from the GitHub comment; no duplicate open issue found for `d2560 Grug MoE MFU profile`.
- Interpretation: kickoff artifacts should track the speed work separately from the original architecture summary.
- Next action: launch or prepare GM2560-MFU-002 with profiler settings.

### 2026-06-13 22:36 PDT - GM2560-MFU-002 H100 profile launch
- Hypothesis: the 32-node H100 baseline with `expert_axis=8`, `save_moe`, sharded fp32 params, and bf16 compute will reveal the dominant gap to 20% MFU.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --submit --run-id GM2560-MFU-002-cw-20260613-2248`
- Config:
  - Job id: `/dlwh/iris-run-job-20260614-053509`
  - Commit: `5d5bc369a4b858551f8f36c35ca9f52122cd87f2`
  - Cluster: `cw-us-east-02a`
  - Nodes: 32 H100 nodes, 8 GPUs per node
  - Mesh: `MAY_REPLICA_AXIS=1`, `MAY_EXPERT_AXIS=8`
  - Batch/sequence: `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`
  - Steps/profile: `MAY_STEPS=30`, `MAY_PROFILER_START=12`, `MAY_PROFILER_STEPS=8`, `MAY_PROFILER_ENABLE_HLO_PROTO=true`
  - Precision/remat: `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`, `MAY_REMAT=save_moe`
  - Tracker/checkpoints/data: `MAY_TRACKER=wandb`, `MAY_CHECKPOINTS=local`, `MAY_DATA=slimpajama`
- Result: dispatcher job submitted; profile babysitting assigned to subagent Hooke (`019ec4a1-9e0d-7633-920e-edb54757dec1`).
- Interpretation: this should produce the first W&B-backed `jax_profile` artifact if the run reaches step 20.
- Next action: monitor job state, capture W&B/profile artifact, ingest with `lib/marin/tools/profile_summary.py`, and classify MoE/attention/FSDP/optimizer overhead.

### 2026-06-13 22:40 PDT - train-step profiling labels for follow-up runs
- Hypothesis: the first profile may distinguish MoE and attention internals but leave optimizer/update overhead hard to read if the whole JIT step appears as one region.
- Command:
  - `uv run pytest experiments/grug/moe/test_optimizer.py tests/test_grug_variant_contracts.py::test_grug_moe_may_recipe_attention_flags_lower -q`
  - `uv run python - <<'PY' ... wandb.Api() ...`
- Config: local code only; the already-running GM2560-MFU-002 job is still on commit `5d5bc369a4b858551f8f36c35ca9f52122cd87f2`.
- Result: added `jax.named_scope` regions inside `experiments/grug/moe/train.py` for `apply_qb_betas`, `forward_backward`, `optimizer_update`, `apply_updates`, `ema_update`, and `watch_stats`. Focused tests passed. W&B run `GM2560-MFU-002-cw-20260613-2248` exists and is running, but has no metrics yet.
- Interpretation: if GM2560-MFU-002 is too coarse for optimizer/FSDP attribution, GM2560-MFU-003 should relaunch from the follow-up instrumentation commit.
- Next action: keep monitoring GM2560-MFU-002 until profiler artifact or terminal failure.

### 2026-06-13 22:45 PDT - select FA4 CuTe as next H100 attention baseline
- Hypothesis: dense reference attention will dominate or materially depress MFU on H100; the d=2560 packed-LM data path carries segment IDs, so `gpu_fa4_cute` should be the first attention backend to test.
- Command:
  - `rg -n "GrugAttentionImplementation|attention_implementation|gpu_fa4" lib/levanter/src/levanter/grug experiments/grug -g '*.py'`
  - `sed -n '340,530p' lib/levanter/src/levanter/data/text/datasets.py`
- Config: local code only; GM2560-MFU-002 is still running on the earlier default, which used `MAY_ATTENTION_IMPLEMENTATION` unset and therefore GPU reference attention.
- Result: changed the dedicated May d2560 launcher and wrapper default to `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`; exposed `run_cw_may_d2560.sh --attention`.
- Interpretation: GM2560-MFU-003 should use the FA4 default if GM2560-MFU-002 confirms attention is hot or overall MFU is below target.
- Next action: wait for GM2560-MFU-002 profile result before deciding whether to relaunch immediately with FA4 or investigate another bottleneck first.

### 2026-06-13 22:45 PDT - GM2560-MFU-002 stopped, GM2560-MFU-003 FA4 profile launched
- Hypothesis: GM2560-MFU-002 was spending 32 H100 nodes on a known-slow attention baseline before any metrics or profile artifacts existed; stopping it before the profiler window and relaunching with FA4 is a better use of the same profiling budget.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-053509`
  - `experiments/grug/moe/run_cw_may_d2560.sh --submit --run-id GM2560-MFU-003-cw-20260613-2245 --attention gpu_fa4_cute`
- Config:
  - Stopped reference-attention parent: `/dlwh/iris-run-job-20260614-053509`
  - Stopped reference-attention child: `/dlwh/iris-run-job-20260614-053509/grug-train-GM2560-MFU-002-cw-20260613-2248`
  - New FA4 parent: `/dlwh/iris-run-job-20260614-054507`
  - Commit: `c3af352ce`
  - Key change from GM2560-MFU-002: `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`
- Result: GM2560-MFU-002 was terminated before any W&B scalar metrics or profile artifacts. GM2560-MFU-003 dispatcher submitted successfully.
- Interpretation: GM2560-MFU-003 is now the first serious H100 fast-path profile candidate for the >=20% MFU target.
- Next action: monitor GM2560-MFU-003 for startup failures from FA4 dependencies, then ingest the W&B `jax_profile` artifact if it reaches the profile window.

### 2026-06-13 22:58 PDT - fp32-master/bf16-live A/B knob added while GM2560-MFU-003 compiles
- Hypothesis: if the GM2560-MFU-003 profile shows FSDP parameter movement or fp32 parameter gathers are hot, a persistent bf16 live parameter tree with a sharded fp32 master tree should reduce forward/backward parameter traffic without giving up fp32 optimizer updates.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_keep_fp32_master tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers tests/test_grug_variant_contracts.py::test_grug_moe_may_recipe_attention_flags_lower experiments/grug/moe/test_optimizer.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uvx pyrefly@1.0.0 check --baseline .pyrefly-baseline.json`
  - `experiments/grug/moe/run_cw_may_d2560.sh --live-param-mode compute_with_master --run-id dry-run-live-param-mode-test`
- Config:
  - New opt-in mode: `GrugTrainerConfig.live_param_mode="compute_with_master"`
  - Wrapper flag: `experiments/grug/moe/run_cw_may_d2560.sh --live-param-mode compute_with_master`
  - Default remains `live_param_mode=param`, so GM2560-MFU-003 is unchanged.
- Result: focused tests, changed-file pre-commit, Pyrefly, and the wrapper dry run passed. GM2560-MFU-003 remains running with 32/32 worker tasks and no W&B history rows or profile artifact yet.
- Interpretation: sharded optimizer state was already present; this adds the next precision-storage A/B without relaunching the active baseline. Current GM2560-MFU-003 logs show XLA SPMD partitioning warnings about involuntary full rematerialization from replicated/maximal or batch-sharded tensors into the 256-way batch/expert layout, so sharding transitions should be checked in the first profile.
- Next action: keep waiting for GM2560-MFU-003 to finish first compile and emit step metrics/profile, then decide whether the first follow-up is `compute_with_master`, a sharding/remat fix, or an MoE/attention change.

### 2026-06-13 23:11 PDT - GM2560-MFU-003 rank-0 W&B artifact bounce
- Hypothesis: a non-terminal Iris retry can still preserve the FA4 profile attempt if the second attempt reaches the profiler window.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job summary --json /dlwh/iris-run-job-20260614-054507/grug-train-GM2560-MFU-003-cw-20260613-2245`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 900 /dlwh/iris-run-job-20260614-054507 | rg -n -C 30 "FileNotFoundError|config\\.yaml|Traceback|Exception|Error"`
- Config: GM2560-MFU-003, FA4 attention (`MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`), 32 H100 nodes, source head `c3af352ce`.
- Result: child job is still running with 32/32 tasks, but `failure_count=1`. Rank 0 hit a background W&B artifact upload `FileNotFoundError` for `/tmp/tmpogy38bhr/config.yaml`; Iris bounced the coscheduled group and relaunched all ranks. W&B remains running with `last_history_step=-1`, code/requirements artifacts from both attempts, and no profile artifact yet.
- Interpretation: no manual relaunch is justified yet. This is a retryable launch/tracker artifact failure unless it repeats; the active run should continue until it either reaches metrics/profile or fails terminally.
- Next action: keep babysitting GM2560-MFU-003; if the same artifact-path failure repeats, report before any recovery.

### 2026-06-13 23:16 PDT - next-run attention layout fixes
- Hypothesis: GM2560-MFU-003 is compiling the FA4 path with two avoidable SPMD layout hazards: a runtime q/kv segment-id equality guard that lowers through `conditional`, and three separate q/k/v projections that each force the same attention-input layout movement.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_may_recipe_attention_flags_lower tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers -q`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py::test_fa4_frontend_uses_query_segment_ids_without_dynamic_equality_check -q`
  - `uv run python -m py_compile experiments/grug/moe/model.py lib/levanter/src/levanter/grug/attention/_fa4_cute.py`
- Config: local code only; active GM2560-MFU-003 still runs source head `c3af352ce`, so it does not include these changes.
- Result: removed the FA4/CuTe dynamic q/kv segment-id equality check for packed self-attention and fused the MoE variant q/k/v projection into one `einsum` plus splits. Focused tests and py_compile passed. GM2560-MFU-003 still has no W&B history rows and is emitting repeated old-code `[SPMD] ... get-tuple-element(%conditional)` warnings.
- Interpretation: the old-code run is unlikely to be the right baseline if it never reaches a step soon; the next profile should relaunch from this patched commit with FA4 and default `live_param_mode=param`, then use `compute_with_master` only if the profile shows parameter movement or optimizer/update overhead.
- Next action: run changed-file checks, commit/push, then stop and relaunch the FA4 profile if GM2560-MFU-003 has not reached step metrics.

### 2026-06-13 23:18 PDT - GM2560-MFU-004 patched FA4 profile launch
- Hypothesis: the patched FA4 metadata path and fused q/k/v projection should reduce compile-time SPMD layout churn enough to get a cleaner first H100 profile.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-054507`
  - `experiments/grug/moe/run_cw_may_d2560.sh --submit --run-id GM2560-MFU-004-cw-20260613-2318 --attention gpu_fa4_cute`
- Config:
  - Stopped stale parent: `/dlwh/iris-run-job-20260614-054507`
  - New parent: `/dlwh/iris-run-job-20260614-061825`
  - Commit: `88dad4287e9e78458d469e02684563307640afe7`
  - Same profile shape: 32 H100 nodes, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_STEPS=30`, `MAY_PROFILER_START=12`, `MAY_PROFILER_STEPS=8`
  - Attention/precision: `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`, `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`, `MAY_LIVE_PARAM_MODE=param`
- Result: GM2560-MFU-003 was stopped before any W&B history or profile artifact. GM2560-MFU-004 dispatcher submitted successfully.
- Interpretation: GM2560-MFU-004 is now the active first-profile candidate; `compute_with_master` remains a follow-up A/B only after this profile shows parameter or optimizer overhead.
- Next action: babysit GM2560-MFU-004 through startup, first metrics, and profile artifact upload.

### 2026-06-13 23:33 PDT - FA4 packed metadata precompute follow-up
- Hypothesis: the FA4/CuTe path still rebuilds packed causal `lower_bounds` and validity metadata inside every attention layer even though the short and long packed masks are shape-stable for a step; caching those arrays on the mask should remove repeated per-layer shape/index work without changing attention semantics.
- Command:
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py::test_fa4_frontend_uses_query_segment_ids_without_dynamic_equality_check lib/levanter/tests/grug/test_fa4_cute_attention.py::test_fa4_cute_metadata_is_precomputed_per_sliding_window -q`
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_may_recipe_attention_flags_lower tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers -q`
  - `uv run python -m py_compile experiments/grug/moe/model.py lib/levanter/src/levanter/grug/attention/_core.py lib/levanter/src/levanter/grug/attention/_fa4_cute.py lib/levanter/src/levanter/grug/attention/__init__.py`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uvx pyrefly@1.0.0 check --baseline .pyrefly-baseline.json`
- Config: local code only; active GM2560-MFU-004 is still running source head `88dad4287` and does not include this metadata-cache follow-up.
- Result: added `Fa4CuteMetadata`, cached the short/long packed FA4 metadata once per transformer call, and taught `gpu_fa4_cute_attention` to reuse it. Focused tests, py_compile, changed-file pre-commit, and Pyrefly passed. GM2560-MFU-004 remains healthy with 32/32 tasks running, zero failures/preemptions, current W&B heartbeat, and no scalar metrics or profile artifact yet.
- Interpretation: this is the next low-risk relaunch candidate if GM2560-MFU-004 stalls or profiles compile/shape overhead in FA4 metadata construction. GM2560-MFU-004 logs still show `[SPMD] Involuntary full rematerialization` for `%fake_parameter.2 = bf16[1,4096,2560]` from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}`, so the remaining compiler-visible issue is attention-input/projection sharding rather than the removed q/kv guard.
- Next action: keep GM2560-MFU-004 running until it reaches metrics/profile or terminal failure; in parallel inspect the activation/projection sharding mismatch before launching any GM2560-MFU-005.

### 2026-06-13 23:43 PDT - GM2560-MFU-004 killed after startup OOM
- Hypothesis: the patched FA4 run still exceeded per-rank memory before the first completed train step, so a blind relaunch with the same shape is unlikely to produce an MFU profile.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix /dlwh/iris-run-job-20260614-061825`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job summary --json /dlwh/iris-run-job-20260614-061825/grug-train-GM2560-MFU-004-cw-20260613-2318`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 900 /dlwh/iris-run-job-20260614-061825 | rg -i -e "RESOURCE_EXHAUSTED|Out of memory|Very slow compile|Progress on:train|wandb|profile|profiler"`
- Config: GM2560-MFU-004, 32 H100 nodes, source head `88dad4287`, `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_LIVE_PARAM_MODE=param`.
- Result: parent and child ended `JOB_STATE_KILLED` with `error="Terminated by user"`. The child had `failure_count=1`, `task_state_counts={"killed": 32}`, and no preemptions. Logs show the first attempt reached `Progress on:train -/30` after about 16:47 elapsed, then multiple ranks hit `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 60.91GiB` from the JAX profiler wrapper. Iris then reported coordination `CANCELLED`/connection-refused messages and began a fresh `syncing deps` attempt, but the job was killed before recovery. W&B is failed with `last_history_step=-1`, code/requirements artifacts only, and no profile artifact.
- Interpretation: GM2560-MFU-004 did not produce usable scalar metrics or a profile. The immediate blocker is memory at the d2560/batch256/expert_axis8 shape, not merely missing instrumentation.
- Next action: do not relaunch this exact config. Reduce the HBM requirement first, likely by fixing the remaining attention/projection sharding rematerialization and/or switching the next run to the lower-memory live-parameter/optimizer layout before considering GM2560-MFU-005.

### 2026-06-13 23:49 PDT - force streaming XLA CE for next H100 profile
- Hypothesis: GM2560-MFU-004 selected the GPU pallas CE path and then died on a 60.91 GiB allocation before step 0; forcing the streaming XLA CE backend should avoid a full-logits-shaped allocation while preserving the d2560/batch256 profile target. Pairing this with `compute_with_master` tests the sharded bf16 live-parameter layout with a sharded fp32 master tree.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_may_recipe_attention_flags_lower tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers -q`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py::test_fa4_cute_metadata_is_precomputed_per_sliding_window -q`
  - `uv run python -m py_compile experiments/grug/moe/model.py experiments/grug/moe/launch_cw_may_d2560.py`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id dry-run-ce-xla --ce-implementation xla`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uvx pyrefly@1.0.0 check --baseline .pyrefly-baseline.json`
- Config: local code only so far; added `MAY_CE_IMPLEMENTATION` / `--ce-implementation` and set the May lowering contract to exercise `cross_entropy_implementation="xla"`.
- Result: root tests, Levanter FA4 metadata test, py_compile, dry-run wrapper, changed-file pre-commit, and Pyrefly passed. A mixed root+lib pytest invocation hit the repo's conftest import mismatch, so the same tests were rerun as separate invocations.
- Interpretation: GM2560-MFU-005 should launch from this commit with `--ce-implementation xla --live-param-mode compute_with_master` before reducing batch or expert axis. If it still OOMs, the remaining likely levers are `--remat recompute_all` and/or smaller batch.
- Next action: commit/push the CE override, launch GM2560-MFU-005, and babysit until first metrics/profile or terminal failure.

### 2026-06-13 23:48 PDT - GM2560-MFU-005 launched with XLA CE and bf16 live params
- Hypothesis: streaming XLA CE removes the GM2560-MFU-004 full-logits-shaped HBM allocation, while `compute_with_master` avoids repeated fp32-to-bf16 parameter casts in the forward/backward path.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --submit --run-id GM2560-MFU-005-cw-20260613-2350 --ce-implementation xla --live-param-mode compute_with_master`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260614-064720`
- Config:
  - Parent: `/dlwh/iris-run-job-20260614-064720`
  - Child: `/dlwh/iris-run-job-20260614-064720/grug-train-GM2560-MFU-005-cw-20260613-2350`
  - Commit: `bbba806e9`
  - Same 32-node shape: `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_STEPS=30`, `MAY_PROFILER_START=12`, `MAY_PROFILER_STEPS=8`
  - Attention/precision: `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`, `MAY_CE_IMPLEMENTATION=xla`, `MAY_LIVE_PARAM_MODE=compute_with_master`, `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`
- Result: dispatcher submitted successfully. First poll showed the parent running and the child running with 32 tasks in `building`; W&B was not visible yet.
- Interpretation: GM2560-MFU-005 is the active first-profile candidate after the CE OOM fix.
- Next action: Hooke is babysitting on a 10-minute heartbeat; watch for W&B visibility, first scalar metrics, profile artifact, or another pre-step OOM.

### 2026-06-14 00:10 PDT - GM2560-MFU-005 killed after XLA CE startup OOM
- Hypothesis: forcing XLA CE removed the exact pallas CE path but did not reduce the d2560/batch256 peak enough; `compute_with_master` is not a guaranteed memory reduction for this shape.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix /dlwh/iris-run-job-20260614-064720`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job summary --json /dlwh/iris-run-job-20260614-064720/grug-train-GM2560-MFU-005-cw-20260613-2350`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 1200 /dlwh/iris-run-job-20260614-064720/grug-train-GM2560-MFU-005-cw-20260613-2350 | rg -i -e "Progress on:train|RESOURCE_EXHAUSTED|Out of memory|trying to allocate 60|Fatal error|train.py|step = int|WatchTasksAsync failed|wandb|profile|profiler"`
- Config: GM2560-MFU-005, 32 H100 nodes, source head `bbba806e9`, `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_CE_IMPLEMENTATION=xla`, `MAY_LIVE_PARAM_MODE=compute_with_master`, `MAY_REMAT=save_moe`.
- Result: parent and child ended `JOB_STATE_KILLED` with `error="Terminated by user"`. The child had `task_state_counts={"killed": 32}`, `failure_count=0`, and no preemptions. Logs show train progress still at `-/30`, with elapsed around 15:22-15:50, then multiple ranks hit BFC allocator OOMs trying to allocate `60.23GiB`; the fatal stack raises `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED` while converting `state.step` in `experiments/grug/moe/train.py:556`. W&B is failed with `last_history_step=-1`, only source/requirements artifacts, and no scalar metrics or profile artifact.
- Interpretation: GM2560-MFU-005 did not solve the startup HBM peak. The failure is still pre-step and therefore no MFU/profile data exists.
- Next action: do not blind relaunch. The next sensible axes are default `live_param_mode=param` with XLA CE to isolate the `compute_with_master` memory tradeoff, then `--remat recompute_all` and/or smaller batch if the default live-param layout still OOMs.

### 2026-06-14 00:33 PDT - explicit H100 XLA memory fraction for GM2560-MFU-006
- Hypothesis: the `60.23GiB` allocator request is the compiled executable's temp arena colliding with JAX's default 75% H100 memory pool, not a single logits tensor. `60.23GiB / 0.75` is about 80.3 GiB, matching H100 HBM scale.
- Command:
  - `uv run python - <<'PY' ... abstract-state sharding summary ... PY`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id dry-run-memory-fraction --ce-implementation xla --live-param-mode param --xla-memory-fraction 0.95`
- Config: local validation only. The abstract mesh was `(replica_dcn=1, data=32, expert=8, model=1)` with the same d=2560, 256-expert May model and MuonH optimizer as GM005.
- Result: abstract-state sizing shows fp32 params are about 249.9 GiB global / 1.34 GiB local, optimizer state about 252.4 GiB global / 1.48 GiB local, and `compute_with_master` adds about 0.67 GiB local bf16 live params. The wrapper dry run now forwards `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to Iris.
- Interpretation: params, fp32 master params, and optimizer state are already sharded enough that they cannot explain a 60 GiB per-GPU allocation. The next full-shape attempt should keep `live_param_mode=param`, force XLA CE, and expand the XLA GPU pool before switching remat or reducing batch.
- Next action: validate the launcher change, commit/push it, then launch GM2560-MFU-006 with `--ce-implementation xla --live-param-mode param --xla-memory-fraction 0.95` and babysit it to first metrics/profile or terminal failure.
