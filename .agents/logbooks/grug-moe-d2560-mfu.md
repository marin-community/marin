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

### 2026-06-14 00:35 PDT - GM2560-MFU-006 launched with 0.95 XLA memory fraction
- Hypothesis: increasing the JAX/XLA GPU memory pool from the default 75% to 95% should allow the full-shape compiled train-step temp arena to coexist with the already-sharded params and optimizer state.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --submit --run-id GM2560-MFU-006-cw-20260614-0035 --ce-implementation xla --live-param-mode param --xla-memory-fraction 0.95`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260614-073514`
- Config:
  - Parent: `/dlwh/iris-run-job-20260614-073514`
  - Child: `/dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`
  - Commit: `77c87b0fe`
  - Same 32-node shape: `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_STEPS=30`, `MAY_PROFILER_START=12`, `MAY_PROFILER_STEPS=8`
  - Attention/precision: `MAY_ATTENTION_IMPLEMENTATION=gpu_fa4_cute`, `MAY_CE_IMPLEMENTATION=xla`, `MAY_LIVE_PARAM_MODE=param`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`
- Result: dispatcher submitted successfully. First child poll showed the child running with 32/32 tasks, zero failures, and W&B not visible yet.
- Interpretation: this is the active full-shape first-profile candidate after the allocator-pool fix.
- Next action: Hooke is babysitting on a 10-minute heartbeat; watch for W&B visibility, first scalar metrics, profile artifact, or another pre-step OOM.

### 2026-06-14 01:00 PDT - GM2560-MFU-006 first attempt hit collective-permute rendezvous abort
- Hypothesis: GM2560-MFU-006's first retry point will distinguish the old 60 GiB allocator failure from a sharding/collective failure in the full d2560 train-step executable.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix /dlwh/iris-run-job-20260614-073514`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job bug-report --tail 80 /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 3600 --max-lines 5000 /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035 | rg "task=.*/1 \\|" | sed -n '1,260p'`
  - `uv run pytest tests/test_grug_variant_contracts.py -q`
  - `uv run python -m py_compile experiments/grug/moe/model.py lib/levanter/src/levanter/grug/sharding.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config:
  - Live run: GM2560-MFU-006, commit `77c87b0fe`, 32 H100 nodes, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_CE_IMPLEMENTATION=xla`, `MAY_LIVE_PARAM_MODE=param`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.
  - Local patch candidate: keep persistent params sharded, but fully replicate the non-MoE attention/shared-MLP weights at matmul call sites using `unshard`; leave routed MoE expert weights on the existing expert-parallel path.
- Result: GM2560-MFU-006 is still running after Iris retried the coscheduled group, but `failure_count=1`. The failed first attempt did not show the previous 60 GiB BFC allocator OOM. Task 1 exited 139 after XLA rendezvous reported `first call after collective operation: kind=kCollectivePermute`; only 4 of 8 local threads arrived by the 40s termination timeout. The same attempt emitted the known SPMD warning for `%fake_parameter.2 = bf16[1,4096,2560]`, moving from batch sharding `{devices=[256,1,1]}` to hidden/data sharding `{devices=[1,1,32,8] last_tile_dim_replicate}`. The active retry has no W&B history rows or profile artifact yet.
- Interpretation: the 0.95 memory pool likely moved past the old allocator reservation, but the full-shape executable is now exposing a sharding-induced collective-permute/rendezvous failure before step 0. The suspicious path is non-MoE dense matmuls whose weights are sharded on the input hidden dimension over `data`, forcing XLA to move batch-sharded activations to hidden-dimension sharding. The local patch makes those dense weights temporary-all-gathered at call sites instead, preserving sharded persistent state while avoiding that activation reshard for attention/shared MLP. Focused compile, full Grug contract tests, rank-2 `unshard` sanity check, and changed-file pre-commit passed.
- Next action: keep babysitting GM2560-MFU-006's active retry until it reaches metrics/profile or repeats the collective failure. If it repeats or stalls terminally, launch the next full-shape candidate from the dense-weight all-gather patch with the same config.

### 2026-06-14 01:05 PDT - GM2560-MFU-006 retry also stuck before metrics
- Hypothesis: if the retry is healthy, it should either emit W&B scalar history or reach the profiler window; if it is the same failure mode, logs should show XLA clique/rendezvous symptoms before step 0.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix /dlwh/iris-run-job-20260614-073514`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job summary --json /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 7200 --max-lines 8000 /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035 | rg -i -e "Progress on:train|loss|global_step|step|profile|profiler|wandb|RESOURCE_EXHAUSTED|Out of memory|trying to allocate|collective|rendezvous|Termination timeout|Fatal error|Traceback|Exception|exit 139|segmentation|failed"`
  - `uv run python - <<'PY' ... wandb.Api().run("marin-community/marin_moe/GM2560-MFU-006-cw-20260614-0035") ... PY`
- Config: same live GM2560-MFU-006 retry as above, still from commit `77c87b0fe`.
- Result: parent and child remain `JOB_STATE_RUNNING`; child has 32/32 tasks running, `failure_count=1`, and `preemption_count=0`. W&B state is still `running`, but it has no history rows, no `_step`, no scalar summary, and no `jax_profile` artifact. Logs from the retry show many XLA `Initialize clique` rendezvous warnings where all local threads joined but the leader did not complete the rendezvous; there is still no training progress beyond startup.
- Interpretation: the retry has not disproved the sharding/collective hypothesis. It is no longer presenting as the 60 GiB allocator-pool failure; the current signal is distributed clique/rendezvous startup fragility before metrics/profile.
- Next action: commit and push the dense-weight all-gather patch so the next candidate can run from a stable snapshot after GM2560-MFU-006 terminates or the user explicitly authorizes replacing it.

### 2026-06-14 01:12 PDT - GM2560-MFU-006 remains live but pre-metrics; warning matches dense activation reshard
- Hypothesis: the SPMD rematerialization warning can distinguish whether the bad path is an accidental hidden-dimension activation reshard or a different source such as logits/CE or MoE expert dispatch.
- Command:
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix /dlwh/iris-run-job-20260614-073514`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job summary --json /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`
  - `uv run --package marin-iris --extra controller iris --config lib/iris/config/cw-us-east-02a.yaml job logs --since-seconds 7200 --max-lines 12000 /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035 | rg -C 4 "fake_parameter\\.2|full rematerialization|SPMD|bf16\\[1,4096,2560\\]|devices=\\[256,1,1\\]|devices=\\[1,1,32,8\\]"`
  - `uv run python - <<'PY' ... wandb.Api().run("marin-community/marin_moe/GM2560-MFU-006-cw-20260614-0035") ... PY`
- Config: live GM2560-MFU-006 is still the pre-patch commit `77c87b0fe`. Local branch head is the next candidate commit `ce06fd232`.
- Result: as of 2026-06-14 01:10 PDT, GM2560-MFU-006 is still `JOB_STATE_RUNNING`; child has 32/32 tasks running, `failure_count=1`, `preemption_count=0`, and no pending reason. W&B remains `running` with zero history rows, no scalar summaries, and no profile artifact. Hooke's latest handoff agrees and reports no new OOM/allocator/fatal signature; the last material child logs are `Initialize clique` rendezvous warnings around 08:02:37 UTC. The SPMD warning source/target is exactly `%fake_parameter.2 = bf16[1,4096,2560]`, from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}`.
- Interpretation: source `{devices=[256,1,1]}` is the local `[batch, seq, hidden]` activation sharded over the full batch axis; target `{devices=[1,1,32,8] last_tile_dim_replicate}` is hidden-dimension sharding over `data` replicated over `expert`. That matches the dense qkv/output/shared-MLP weight sharding fixed by `ce06fd232`, rather than params/optimizer state, CE logits, or the intended MoE expert-parallel path. The output CE wrapper already reshards `lm_head` to `P(None, None)` inside its shard map, so no additional CE patch is justified before the `ce06fd232` run.
- Next action: do not stop GM2560-MFU-006 without user approval while it is still live. If it terminates, launch the same full-shape candidate from `ce06fd232` with `--ce-implementation xla --live-param-mode param --xla-memory-fraction 0.95`; if it remains live-but-stalled, ask for approval before replacing it.

### 2026-06-14 15:17 PDT - GM2560-MFU-006 killed after long clique stall
- Hypothesis: after GM2560-MFU-006 spent many hours with no scalar metrics, no profile artifact, and repeated XLA clique/rendezvous warnings, replacing it with a minimizer would yield more information than continuing to burn the same 32-node slot.
- Command:
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260614-073514`
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`
  - `uv run iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-073514`
  - `uv run iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260614-073514`
- Config: GM2560-MFU-006, source commit `77c87b0fe`, 32 H100 nodes, `MAY_EXPERT_AXIS=8`, `MAY_REPLICA_AXIS=1`, `MAY_BATCH=256`, `MAY_SEQ_LEN=4096`, `MAY_CE_IMPLEMENTATION=xla`, `MAY_LIVE_PARAM_MODE=param`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.
- Result: with user approval, stopped parent `/dlwh/iris-run-job-20260614-073514` and child `/dlwh/iris-run-job-20260614-073514/grug-train-GM2560-MFU-006-cw-20260614-0035`. Final child state was `JOB_STATE_KILLED`, 32 killed tasks, `failure_count=1`, `preemption_count=0`; parent was also `JOB_STATE_KILLED`. W&B had no scalar history rows and no `jax_profile` artifact.
- Interpretation: GM2560-MFU-006 is a negative result for the full d2560 candidate: raising the memory pool avoided the earlier explicit 60 GiB allocator failure, but the run then wedged before metrics in XLA collective/clique initialization. This still leaves the dense activation-reshard patch at `ce06fd232` unvalidated on a full-shape run.
- Next action: launch a topology-preserving minimizer to determine whether the clique failure is tied to the 32-node communicator shape itself or to the full d2560 executable.

### 2026-06-14 15:18 PDT - GM2560-MIN-001 32-node d128 minimizer launched
- Hypothesis: a tiny model on the same 32-node H100 topology can separate distributed clique/setup problems from full d2560 memory/sharding problems. If the minimizer wedges, the bad path is likely topology/collective setup; if it reaches steps, the next full-shape attempt should use the `ce06fd232` dense-weight all-gather patch.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu -e ... -- python -m experiments.grug.moe.launch_cw_scale`
  - `uv run iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260614-221814`
  - `uv run iris --cluster=cw-us-east-02a job logs --tail --max-lines 500 /dlwh/iris-run-job-20260614-221814`
- Config:
  - Run id: `GM2560-MIN-001-cw-20260614-1518`
  - Parent: `/dlwh/iris-run-job-20260614-221814`
  - Child: `/dlwh/iris-run-job-20260614-221814/grug-train-GM2560-MIN-001-cw-20260614-1518`
  - Topology: `SCALE_GPU_REPLICAS=32`, `SCALE_EXPERT_AXIS=8`, `SCALE_REPLICA_AXIS=1`
  - Tiny model: `SCALE_HIDDEN_DIM=128`, `SCALE_NUM_LAYERS=1`, `SCALE_NUM_EXPERTS=8`, `SCALE_TOP_K=1`
  - Tiny workload: `SCALE_BATCH=256`, `SCALE_SEQ_LEN=16`, `SCALE_STEPS=2`
  - Runtime: `SCALE_CHECKPOINTS=local`, `SCALE_TRACKER=json_logger`, `SCALE_MP=params=float32,compute=bfloat16,output=bfloat16`
- Result: dispatcher submitted successfully. Parent is `JOB_STATE_RUNNING`; child is `JOB_STATE_RUNNING` with 32 tasks in `building`, `failure_count=0`, and `preemption_count=0` as of 2026-06-14 15:24 PDT. Parent logs show the tokenized SlimPajama input step was skipped as already succeeded, the Grug training step lock was acquired, and the child Fray job was dispatched. The child had not emitted training logs yet at the latest poll.
- Interpretation: this is not an MFU/profile candidate; it is a distributed-path minimizer. Its outcome will decide whether to immediately relaunch full d2560 from `ce06fd232` or keep minimizing the collective/clique path.
- Next action: Hooke is babysitting the minimizer on a 10-minute heartbeat. If it reaches steps, launch the next full-shape candidate from `ce06fd232` with `--ce-implementation xla --live-param-mode param --xla-memory-fraction 0.95`; if it wedges with the same clique symptoms, minimize further before spending another full d2560 compile.

### 2026-06-14 15:30 PDT - minimizer blocked by Kueue CPU admission, not JAX
- Hypothesis: the minimizer's `building` state might be slow dependency setup, but Kubernetes can distinguish unadmitted pods from running init containers.
- Command:
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl get pods -n iris -o wide | rg -i 'NAME|gm2560-min|221814|grug-train'`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl describe pod -n iris iris-dlwh-iris-run-job-20260614-221814-grug-train-gm-030c618c-0`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl describe workload -n iris iris-pg-9e48e2f0ec8df97c-0`
  - `uv run python -m py_compile experiments/grug/moe/launch_cw_scale.py experiments/grug/moe/launch_cw_may_d2560.py`
  - `experiments/grug/moe/run_cw_scale.sh --run-id dry-run-worker-cpu --worker-cpu 8 --smoke`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id dry-run-worker-cpu --worker-cpu 8 --tracker json_logger --profiler-steps 0`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: active minimizer still requests 32 CPU per H100 pod because `launch_cw_scale.py` hard-coded `cpu=32`. Local follow-up adds `SCALE_CPU_PER_REPLICA` and `MAY_CPU_PER_REPLICA`, exposed as `--worker-cpu` in both CoreWeave wrappers.
- Result: all 32 minimizer child pods are `SchedulingGated`, not running init containers. The Kueue workload reports `couldn't assign flavors ... topology "infiniband" ... excluded: resource "cpu"` and at one point could fit only 14 of 32 pods. There is also unrelated Iris controller probe churn in the event tail, but the direct blocker for the child pod group is CPU admission. The CPU knob py_compile, wrapper dry-runs, and diff-scoped pre-commit passed.
- Interpretation: GM2560-MIN-001 has not tested the clique path yet. The current job may admit if other pods release CPU, but a faster diagnostic is to relaunch the minimizer with a lower per-worker CPU request after explicit approval. The same knob will also help the next full d2560 candidate avoid admission stalls if CPU contention persists.
- Next action: ask before stopping/relaunching GM2560-MIN-001. Recommended replacement is the same minimizer with `--worker-cpu 8` after committing/pushing the CPU-request knob.

### 2026-06-14 16:06 PDT - synthetic topology probe support and GM2560-MIN-006
- Hypothesis: the 32-node clique/rendezvous symptoms should be separated from real-data loading and tokenized-cache behavior before another full d2560 compile.
- Command:
  - `experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --run-id dry-synthetic`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu -e ... -- python -m experiments.grug.moe.launch_cw_scale`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-230656`
- Config:
  - Run id: `GM2560-MIN-006-cw-20260614-1606`
  - Parent: `/dlwh/iris-run-job-20260614-230656`
  - Child: `/dlwh/iris-run-job-20260614-230656/grug-train-GM2560-MIN-006-cw-20260614-1606`
  - Topology: `SCALE_GPU_REPLICAS=32`, `SCALE_EXPERT_AXIS=8`, `SCALE_REPLICA_AXIS=1`
  - Tiny model/data: `SCALE_HIDDEN_DIM=128`, `SCALE_NUM_LAYERS=1`, `SCALE_NUM_EXPERTS=8`, `SCALE_TOP_K=1`, `SCALE_BATCH=256`, `SCALE_SEQ_LEN=16`, `SCALE_DATA=synthetic`
  - Runtime: `SCALE_CPU_PER_REPLICA=2`, `SCALE_CHECKPOINTS=local`, `SCALE_TRACKER=json_logger`
- Result: the synthetic run admitted 32/32 tasks and emitted hparams/static summaries, proving the CPU admission and real-data stall were separate from the basic 32-node launch path. It still reached no train-step metrics before being stopped. Logs showed the same SPMD full-rematerialization warning family as the full run, now on tiny activations: moving batch-sharded tensors from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}` around RMSNorm/forward-backward. No clique, OOM, or long data-loader stall appeared before stop.
- Interpretation: the remaining critical path is activation layout/sharding, not R2/data loading. Synthetic data is now available for cheap topology-preserving probes.
- Next action: try a model-axis probe to reduce the data-axis hidden-sharding pressure, and disable watch stats for throughput probes so step-0 compilation is less polluted.

### 2026-06-14 16:21 PDT - GM2560-MIN-007 invalid model-axis probe
- Hypothesis: adding a model/tensor-parallel axis may reduce the bad activation reshards by separating hidden/vocab work from the batch/expert axes.
- Command:
  - `SCALE_MODEL_AXIS=8 SCALE_HIDDEN_DIM=128 ... python -m experiments.grug.moe.launch_cw_scale`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-231809`
- Config:
  - Run id: `GM2560-MIN-007-cw-20260614-1618`
  - Parent: `/dlwh/iris-run-job-20260614-231809`
  - Child: `/dlwh/iris-run-job-20260614-231809/grug-train-GM2560-MIN-007-cw-20260614-1618`
  - New knobs: `SCALE_MODEL_AXIS=8`, `SCALE_WATCH_INTERVAL=0`; otherwise the same tiny synthetic topology as MIN-006.
- Result: the run failed before a train step with `ValueError: Sharding spec ('model',) implies that array axis 2 is partitioned 8 times, but does not evenly divide the dimension size 1`, on shape `(256, 16, 1, 128)` and spec `P(('replica_dcn', 'data', 'expert'), None, 'model', None)`. The failure is deterministic: this Grug attention code shards the attention head axis over `model`, and the tiny d128 probe has one head. May d2560 has 20 heads, so `model_axis=8` would also be invalid.
- Interpretation: model-axis is still worth testing, but the valid May-aligned choice is `model_axis=4`, not 8. Added launch/train validation so bad model-axis choices fail locally or in the dispatcher before allocating the 32-worker child group.
- Next action: launch `GM2560-MIN-008` with `SCALE_MODEL_AXIS=4` and `SCALE_HIDDEN_DIM=512` so the proxy has four attention heads.

### 2026-06-14 16:31 PDT - GM2560-MIN-008 stopped during pallas CE autotune
- Hypothesis: a valid `model_axis=4` proxy with four attention heads should reveal whether tensor/model parallelism removes the tiny-shape SPMD rematerialization path seen in MIN-006.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 4 --watch-interval 0 --run-id GM2560-MIN-008-cw-20260614-1623 --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260614-232401`
- Config:
  - Run id: `GM2560-MIN-008-cw-20260614-1623`
  - Parent: `/dlwh/iris-run-job-20260614-232401`
  - Child: `/dlwh/iris-run-job-20260614-232401/grug-train-GM2560-MIN-008-cw-20260614-1623`
  - Topology: 32 H100 nodes, `SCALE_EXPERT_AXIS=8`, `SCALE_REPLICA_AXIS=1`, `SCALE_MODEL_AXIS=4`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data
- Result: the child admitted and ran 32/32 tasks with zero failures/preemptions. It got past the MIN-007 model-axis divisibility error and emitted hparams/static summaries. It then spent the useful observation window in pallas GPU fused-CE autotuning (`Fused CE autotune miss for pallas_gpu. Sweeping 7 block-size candidates`) and produced no train-step metrics before being stopped.
- Interpretation: `model_axis=4` is valid for the proxy, but pallas CE autotune is unnecessary noise for sharding minimizers. This run did not answer the SPMD-remat question.
- Next action: relaunch the same proxy as `GM2560-MIN-009` with `SCALE_CE_IMPLEMENTATION=xla`, and keep the pallas CE path as a separate throughput/profile question.

### 2026-06-14 16:32 PDT - GM2560-MIN-009 XLA-CE model-axis probe launched
- Hypothesis: replacing pallas CE with XLA CE in the tiny proxy should get to the train-step compile/run path faster, making model-axis sharding effects easier to observe.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 4 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-009-cw-20260614-1632 --submit`
- Config:
  - Run id: `GM2560-MIN-009-cw-20260614-1632`
  - Parent: `/dlwh/iris-run-job-20260614-233208`
  - Topology/shape: same as MIN-008, with `SCALE_CE_IMPLEMENTATION=xla`
- Result: parent submitted successfully. The child admitted after Curie's one-node EP sidecar was stopped to free a topology slot, then ran 32/32 tasks. The remote hparams confirmed `cross_entropy_implementation="xla"`, `model_axis_size=4`, `expert_axis_size=8`, and `watch.interval=0`. Before any train-step metric, the compiler still emitted SPMD full-rematerialization warnings between `{devices=[64,1,1,4]}` and `{devices=[1,1,8,32]}` for `[4,16,512]` activations. At 23:42:05 UTC, the first attempt aborted with local XLA rendezvous timeouts after `kCollectivePermute` (`Expected 8 threads ... only 4-7 arrived` on multiple tasks). The parent/child were then stopped while Iris was assigning a retry.
- Interpretation: `SCALE_CE_IMPLEMENTATION=xla` removes pallas CE autotune noise, but `model_axis=4` does not remove the bad activation layout transition. More importantly, with the current `(replica_dcn, data, expert, model)` device layout, `expert_axis=8` and `model_axis=4` cannot both be intra-node on 8-GPU workers; the effective expert ring spans nodes and reproduces the collective-permute/rendezvous failure.
- Next action: launch `GM2560-MIN-010` with the same tiny proxy and `SCALE_MODEL_AXIS=4`, but set `SCALE_EXPERT_AXIS=1` to isolate whether the failure is the cross-node EP path.

### 2026-06-14 16:49 PDT - GM2560-MIN-010 isolates model-axis from EP
- Hypothesis: if the MIN-009 failure is caused by `expert_axis=8` spanning nodes when combined with `model_axis=4`, then the same d512 proxy with `expert_axis=1` should reach train steps.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_EXPERT_AXIS=1 SCALE_REPLICA_AXIS=1 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 4 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-010-cw-20260614-1643 --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260614-234410/grug-train-GM2560-MIN-010-cw-20260614-1643`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs --since-seconds 7200 --max-lines 3000 /dlwh/iris-run-job-20260614-234410/grug-train-GM2560-MIN-010-cw-20260614-1643 | rg -i 'Grug compact mesh|train/loss|throughput|Progress on:train|SPMD|kCollectivePermute|Traceback|RESOURCE_EXHAUSTED|OOM'`
- Config:
  - Run id: `GM2560-MIN-010-cw-20260614-1643`
  - Parent: `/dlwh/iris-run-job-20260614-234410`
  - Child: `/dlwh/iris-run-job-20260614-234410/grug-train-GM2560-MIN-010-cw-20260614-1643`
  - Topology: 32 H100 nodes, `SCALE_REPLICA_AXIS=1`, `SCALE_EXPERT_AXIS=1`, `SCALE_MODEL_AXIS=4`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data, XLA CE, local checkpoints, watch disabled.
- Result: parent and child succeeded; child completed 32/32 tasks with `failure_count=0` and `preemption_count=0`. The logged compact mesh was `{'replica_dcn': 1, 'data': 64, 'expert': 1, 'model': 4}`, `batch_shards=64`, with `parameter_count=136060936`. It reached `2/2` train progress with `train/loss ~= 11.8`; representative final per-task throughput was roughly 290-294 examples/s, 4.65-4.70k tokens/s, and MFU around `7.4e-4` for this tiny diagnostic.
- Interpretation: `model_axis=4` is not by itself enough to trigger the local `kCollectivePermute` failure. The MIN-009 failure is much more likely the cross-node EP path created by trying to fit `expert_axis=8` and `model_axis=4` onto 8-GPU nodes.
- Next action: test `SCALE_EXPERT_AXIS=2` with `SCALE_MODEL_AXIS=4`. The product `expert_axis * model_axis = 8` should remain local to each H100 worker, while still exercising nontrivial EP.

### 2026-06-14 16:52 PDT - GM2560-MIN-011 local EP2/model4 probe launched
- Hypothesis: `expert_axis=2, model_axis=4` keeps the model and expert axes within each 8-GPU H100 worker, so it should avoid the MIN-009 local collective-permute timeout while retaining some expert parallelism.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=2 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 4 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-011-cw-20260614-1652 --submit`
- Config:
  - Run id: `GM2560-MIN-011-cw-20260614-1652`
  - Parent: `/dlwh/iris-run-job-20260614-235241`
  - Intended topology: 32 H100 nodes, `SCALE_REPLICA_AXIS=1`, `SCALE_EXPERT_AXIS=2`, `SCALE_MODEL_AXIS=4`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data, XLA CE, local checkpoints, watch disabled.
- Result: submitted successfully at 16:52 PDT; initial parent state was `running/building` with no failures.
- Result: the child reached the train loop and logged compact mesh `{'replica_dcn': 1, 'data': 32, 'expert': 2, 'model': 4}`, `batch_shards=64`. It then failed before any train-step metric. Logs showed NCCL communicator creation failures (`ncclCommInitRankConfig`, CUDA error 999), `Initialize clique` waits over 64-device groups, `Acquire clique` waits over 4-device local groups, then fatal `kCollectivePermute` rendezvous termination: expected 8 threads but only 1 arrived. The parent and child were stopped after the decisive failure.
- Interpretation: keeping `expert_axis * model_axis = 8` is not sufficient for the model4 path. With `model_axis=4`, even EP2 introduces enough collective structure to reproduce the bad `kCollectivePermute` failure; only `expert_axis=1` has succeeded so far.
- Next action: test `SCALE_EXPERT_AXIS=4`, `SCALE_MODEL_AXIS=2`, which is a more plausible full-model compromise if it can step.

### 2026-06-14 16:59 PDT - GM2560-MIN-012 EP4/model2 probe launched
- Hypothesis: `expert_axis=4, model_axis=2` keeps the product local to each 8-GPU worker while reducing model-axis collective pressure relative to `expert_axis=2, model_axis=4`.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=4 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 2 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-012-cw-20260614-1659 --submit`
- Config:
  - Run id: `GM2560-MIN-012-cw-20260614-1659`
  - Parent: `/dlwh/iris-run-job-20260614-235834`
  - Intended topology: 32 H100 nodes, `SCALE_REPLICA_AXIS=1`, `SCALE_EXPERT_AXIS=4`, `SCALE_MODEL_AXIS=2`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data, XLA CE, local checkpoints, watch disabled.
- Result: the child reached 32/32 running tasks and logged compact mesh `{'replica_dcn': 1, 'data': 32, 'expert': 4, 'model': 2}`, `batch_shards=128`. It reached `Progress on:train -/2`, then produced repeated 64-device `Initialize clique` waits and 64-device `Acquire clique` waits with `local_participants=2`; no train-step metric appeared. The parent and child were stopped after the decisive pre-metrics collective stall.
- Interpretation: default ring MoE is not compatible with any simultaneous `expert_axis > 1` and `model_axis > 1` layout observed so far. The only model-axis run that steps is `expert_axis=1, model_axis=4`; the ring EP path should keep `model_axis=1`.
- Next action: do not spend more 32-node probes on mixed ring-EP/model-axis layouts unless the MoE implementation changes. Continue with either EP-only (`expert_axis=8, model_axis=1`) throughput probes or model-axis-only (`expert_axis=1, model_axis=4`) attention/CE diagnostics.

### 2026-06-14 16:58 PDT - local expert/model axis guard added
- Hypothesis: MIN-009 failed because `SCALE_EXPERT_AXIS=8` and `SCALE_MODEL_AXIS=4` made `expert_axis * model_axis = 32`, so the expert/model axes could not fit inside one 8-GPU H100 worker. MIN-010 succeeded because `expert_axis=1`, `model_axis=4` kept that product local.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_coreweave_axes_keep_expert_and_model_groups_local tests/test_grug_variant_contracts.py::test_grug_coreweave_axes_reject_cross_node_expert_model_product -q`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q`
  - `uv run python -m py_compile experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_may_d2560.py experiments/grug/moe/launch_cw_scale.py lib/levanter/src/levanter/grug/attention/_fa4_cute.py experiments/grug/moe/train.py tests/test_grug_variant_contracts.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: local code only. Added a shared launcher validation that requires `expert_axis * model_axis` to divide the 8 GPUs on each CoreWeave worker. The guard rejects `8 * 4 = 32` before remote allocation and accepts the intended `2 * 4 = 8` probe.
- Result: focused topology tests passed, FA4 CuTe tests passed locally (`3 passed, 2 skipped` GPU-only), py_compile passed, changed-file pre-commit passed. Existing GM2560-MIN-011 was still running with 32/32 tasks and zero failures at the status check; logs had reached `Grug compact mesh shape: {'replica_dcn': 1, 'data': 32, 'expert': 2, 'model': 4}` and still showed SPMD activation rematerialization warnings, but no `kCollectivePermute`, OOM, or traceback in the first observation window.
- Interpretation: the launcher now encodes the topology lesson from MIN-009/MIN-010. `expert_axis=2, model_axis=4` remains the right next topology probe, but the SPMD activation layout warning is still present and should be treated as a separate attention/projection sharding bottleneck.
- Next action: let GM2560-MIN-011 reach terminal state or a decisive failure, then only consider a full d2560 probe with `--expert-axis 2 --model-axis 4 --ce-implementation xla --watch-interval 0 --live-param-mode param --xla-memory-fraction 0.95` after checking HBM impact from lower EP.

### 2026-06-14 17:00 PDT - GM2560-MIN-011 also hit ring EP collective timeout
- Hypothesis: if `expert_axis=2, model_axis=4` keeps EP/model groups local, then the prior MIN-009 failure was only cross-node EP. If it still fails, default ring EP likely cannot be combined with model parallelism yet.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260614-235241/grug-train-GM2560-MIN-011-cw-20260614-1652`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs --since-seconds 3600 --max-lines 5000 /dlwh/iris-run-job-20260614-235241/grug-train-GM2560-MIN-011-cw-20260614-1652 | rg -i 'Progress on:train|train/loss|throughput|SPMD|kCollectivePermute|rendezvous|RESOURCE_EXHAUSTED|OOM|Traceback|Terminated|killed|Fatal|segmentation|exit 139'`
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_coreweave_ring_ep_rejects_simultaneous_expert_and_model_axes tests/test_grug_variant_contracts.py::test_grug_coreweave_non_ring_backend_can_try_simultaneous_expert_and_model_axes -q`
- Config: existing GM2560-MIN-011, `SCALE_EXPERT_AXIS=2`, `SCALE_MODEL_AXIS=4`, d512 tiny synthetic proxy, XLA CE, watch disabled. No new job was launched by this check.
- Result: child ended `JOB_STATE_KILLED` with `failure_count=1`, `preemption_count=0`, and all 32 tasks killed. Logs reached train setup and throughput summaries, then emitted the same SPMD activation rematerialization warnings. It then hit local XLA rendezvous timeouts after `kCollectivePermute`; task 10 and task 21 exceeded the 40s termination timeout with only 1 of 8 expected threads arrived and aborted. I did not stop this run. Added launcher validation to reject simultaneous `expert_axis > 1` and `model_axis > 1` for the default ring MoE backend while allowing explicit alternate backend experiments.
- Interpretation: the earlier hypothesis was too weak. Keeping `expert_axis * model_axis == 8` local is not sufficient for the current default ring EP backend. The robust rule for now is: use `model_axis > 1` only with `expert_axis=1` for attention/model-axis diagnostics, or use `expert_axis > 1` only with `model_axis=1` for ring-EP diagnostics. Treat simultaneous ring EP and model-axis sharding as a separate backend bug, not a throughput tuning knob.
- Next action: do not spend a full d2560 run on `expert_axis=2, model_axis=4`. For attention throughput, use `expert_axis=1, model_axis=4` if memory permits or a smaller synthetic proxy; for production MoE EP, stay with `expert_axis=8, model_axis=1` until ring EP + model-axis collectives are fixed or an alternate MoE backend is validated.

### 2026-06-14 17:14 PDT - one-node EP8 ring and ragged sidecars completed
- Hypothesis: EP8 is viable when the entire expert axis stays on one 8xH100 worker with `model_axis=1`; ring should be the safer default, while `ragged_all_to_all` may reduce communication but has more runtime/compiler risk.
- Commands:
  - `env SCALE_GPU_REPLICAS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=4 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=4 SCALE_BATCH=64 SCALE_SEQ_LEN=256 SCALE_STEPS=6 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_CHECKPOINTS=local SCALE_REMAT=save_moe experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 1 --watch-interval 0 --ce-implementation xla --moe-implementation ring --run-id GM-EP8-RING-001-cw-20260614-1658 --submit`
  - `env SCALE_GPU_REPLICAS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=4 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=4 SCALE_BATCH=64 SCALE_SEQ_LEN=256 SCALE_STEPS=6 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_CHECKPOINTS=local SCALE_REMAT=save_moe experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 1 --watch-interval 0 --ce-implementation xla --moe-implementation ragged_all_to_all --run-id GM-EP8-RAGGED-001-cw-20260614-1703 --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260614-235731/grug-train-GM-EP8-RING-001-cw-20260614-1658`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-000259/grug-train-GM-EP8-RAGGED-001-cw-20260614-1703`
- Config:
  - Ring child: `/dlwh/iris-run-job-20260614-235731/grug-train-GM-EP8-RING-001-cw-20260614-1658`
  - Ragged child: `/dlwh/iris-run-job-20260615-000259/grug-train-GM-EP8-RAGGED-001-cw-20260614-1703`
  - Shared topology/shape: one 8xH100 worker, `SCALE_EXPERT_AXIS=8`, `SCALE_MODEL_AXIS=1`, `SCALE_REPLICA_AXIS=1`, d512, 4 layers, 8 experts, top-4, batch 64, sequence 256, 6 train steps, synthetic data, XLA CE, `save_moe`, local checkpoints, watch disabled.
- Result: both sidecars succeeded with `failure_count=0` and `preemption_count=0`. Both logged compact mesh `{'replica_dcn': 1, 'data': 1, 'expert': 8, 'model': 1}`, `batch_shards=8`, and `parameter_count=149451808`. Ring completed in about 146s and ended with final logged throughput `568052 tokens/s`, `throughput/mfu=3.32496`, `train/loss=11.7375`. Ragged completed in about 226s and ended with final logged throughput `326453 tokens/s`, `throughput/mfu=1.91081`, `train/loss=11.7364`. Unique post-warmup step samples were roughly 769k, 642k, 436k, 568k tokens/s for ring and 577k, 599k, 343k, 326k tokens/s for ragged. The absolute MFU is not meaningful for this tiny synthetic proxy because it reports values over 1, but relative tokens/s and job duration consistently favored ring.
- Implementation note: default `ring` all-gathers activations and routing data over the expert axis, computes local ragged expert GEMMs, then `psum_scatter`s outputs. `ragged_all_to_all` sorts local assignments, exchanges ragged token buffers, computes local ragged expert GEMMs, exchanges outputs back, then unpermutes. Ragged should reduce payload when top-k is much smaller than EP, but this short EP8 H100 sidecar did not realize that advantage.
- Interpretation: EP8-local itself is healthy on H100 when `model_axis=1`. The failures seen in MIN-009/MIN-011/MIN-012 are not caused by EP8 alone; they are caused by combining ring EP with `model_axis > 1` or otherwise forcing expert/model collectives into the bad topology path. For now, the recommended EP8 throughput setting is `expert_axis=8`, `model_axis=1`, XLA CE for probes, synthetic data, local checkpoints, and `ring`.
- Next action: use the new MoE implementation launch knob only for explicit backend experiments. Continue main throughput work with ring EP8-local unless a larger replicated ragged test overturns this result.

### 2026-06-14 17:12 PDT - GM2560-MIN-013 failed during workdir staging
- Hypothesis: after the mixed ring-EP/model-axis failures, a 32-node EP-only probe with `expert_axis=8` and `model_axis=1` should tell whether the ring EP path is healthy across the full CoreWeave topology.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 1 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-013-cw-20260614-1706 --submit`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl get pod -n iris iris-dlwh-iris-run-job-20260615-000555-0-4b7f1d5e-0 -o json | jq '{phase:.status.phase, conditions:.status.conditions, initContainerStatuses:.status.initContainerStatuses, containerStatuses:.status.containerStatuses}'`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl logs -n iris iris-dlwh-iris-run-job-20260615-000555-0-4b7f1d5e-0 -c stage-workdir --tail=200`
- Config:
  - Run id: `GM2560-MIN-013-cw-20260614-1706`
  - Parent: `/dlwh/iris-run-job-20260615-000555`
  - Intended topology: 32 H100 nodes, `SCALE_REPLICA_AXIS=1`, `SCALE_EXPERT_AXIS=8`, `SCALE_MODEL_AXIS=1`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data, XLA CE, local checkpoints, watch disabled.
- Result: the parent failed before the user task container started and never dispatched a child job. Kubernetes showed the `stage-workdir` init container terminated with exit code 1 while the main task container remained in `PodInitializing`. The init log showed two bundle fetch retries followed by `urllib.error.URLError: <urlopen error [Errno 113] No route to host>`.
- Interpretation: MIN-013 is not evidence about Grug, XLA, EP, or model sharding. It is an Iris/Kubernetes bundle-staging network failure before Python user code ran.
- Next action: relaunch the same EP-only 32-node probe with a new run id.

### 2026-06-14 17:14 PDT - GM2560-MIN-014 32-node EP8/model1 probe succeeded
- Hypothesis: ring EP8 should work across 32 nodes when `model_axis=1`; if this reaches train steps, it confirms the recent collective failures are specific to mixed ring EP plus model-axis sharding.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=512 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=8 SCALE_TOP_K=1 SCALE_BATCH=256 SCALE_SEQ_LEN=16 SCALE_STEPS=2 SCALE_CHECKPOINTS=local experiments/grug/moe/run_cw_scale.sh --full --worker-cpu 2 --data synthetic --model-axis 1 --watch-interval 0 --ce-implementation xla --run-id GM2560-MIN-014-cw-20260614-1713 --submit`
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-001248`
  - `uv run iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-001248 --tail --max-lines 160`
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-001248/grug-train-GM2560-MIN-014-cw-20260614-1713`
- Config:
  - Run id: `GM2560-MIN-014-cw-20260614-1713`
  - Parent: `/dlwh/iris-run-job-20260615-001248`
  - Child: `/dlwh/iris-run-job-20260615-001248/grug-train-GM2560-MIN-014-cw-20260614-1713`
  - Topology: 32 H100 nodes, `SCALE_REPLICA_AXIS=1`, `SCALE_EXPERT_AXIS=8`, `SCALE_MODEL_AXIS=1`
  - Proxy shape: d512, 1 layer, 8 experts, top-1, batch 256, seq 16, synthetic data, XLA CE, local checkpoints, watch disabled.
- Result: parent submitted and dispatched the child. The child succeeded with 32/32 tasks succeeded, `failure_count=0`, and `preemption_count=0`. It logged compact mesh `{'replica_dcn': 1, 'data': 32, 'expert': 8, 'model': 1}`, `batch_shards=256`, and `parameter_count=136060936`. Rank 0 reached `2/2` train progress with final `train/loss=11.7991`. The one post-warmup tiny-proxy throughput sample was `1720.5 examples/s`, `27528 tokens/s`, and `throughput/mfu=0.004377`. The PJRT `WatchTasksAsync` connection-refused messages appeared after completion during coordination-service shutdown, not as a job failure.
- Interpretation: the 32-node EP8 ring topology is healthy when `model_axis=1`. The recent clique/rendezvous failures are not caused by 32-node EP8 alone; they are specific to mixed ring EP plus model-axis sharding or full-shape executable pressure.
- Next action: use `expert_axis=8`, `model_axis=1`, XLA CE, synthetic or controlled data, local checkpoints, and watch disabled for the next full d2560 memory/throughput attempt. Continue treating mixed ring EP/model-axis as invalid unless a non-ring backend probe succeeds.

### 2026-06-14 17:20 PDT - GM2560-MFU-007 full d2560 synthetic probe launched
- Hypothesis: after MIN-014 proved the 32-node EP8/model1 topology, the next failure boundary is the full d2560 executable/memory footprint rather than distributed setup. Use a short synthetic no-checkpoint/no-profile run to test whether the full issue #6044 shape can step before paying for a profiling window.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --data synthetic --tracker json_logger --profiler-steps 0 --checkpoints none --watch-interval 0 --worker-cpu 32 --ce-implementation xla --model-axis 1 --expert-axis 8 --steps 4 --run-id GM2560-MFU-007-cw-20260614-1720 --submit`
- Config:
  - Run id: `GM2560-MFU-007-cw-20260614-1720`
  - Parent: `/dlwh/iris-run-job-20260615-001940`
  - Topology: 32 H100 nodes, `MAY_REPLICA_AXIS=1`, `MAY_EXPERT_AXIS=8`, `MAY_MODEL_AXIS=1`
  - Model/workload: May d2560 issue #6044 shape, 256 experts, top-4, batch 256, seq 4096, 4 train steps, synthetic data, XLA CE.
  - Runtime: json logger, profiler disabled, checkpointing disabled, watch disabled, `live_param_mode=param`, `MAY_MP=params=float32,compute=bfloat16,output=bfloat16`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.
- Result: parent submitted successfully. Child not yet discovered at launch time.
- Interpretation: this run is the current full-shape memory/executable gate. If it steps, relaunch a W&B/profile version with the same topology; if it OOMs or wedges, inspect whether the full d2560 issue is activation layout, optimizer state/update, or CE/attention memory.

### 2026-06-14 17:34 PDT - MFU-007 active; issue and sidecar lanes refreshed
- Hypothesis: if GM2560-MFU-007 is past the old allocator and topology failures, it should next either complete a train step, produce a new full-shape failure signature, or spend long enough pre-step to justify minimizing the full executable path.
- Command:
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 | jq '{job_id:(.job_id // .id), state:(.state // .status), task_count:(.tasks|length? // .task_count), states:([.tasks[]? | (.state // .status)] | group_by(.) | map({state:.[0], count:length})), failure_count, preemption_count, error:(.error // .status_message // null)}'`
  - `uv run iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 --since-seconds 900 --max-lines 8000 | rg -i 'Progress on:train|train/loss|throughput/total_tokens|throughput/tokens_per_second|throughput/mfu|parameter_count|flops_per_token|RESOURCE_EXHAUSTED|OOM|out of memory|Traceback|Fatal|kCollectivePermute|rendezvous|clique|failed|Error|compil|memory|allocation|buffer|rematerialization|Data loader stalled'`
  - `gh issue view 6367 --repo marin-community/marin --json number,title,state,labels,body,url,comments`
- Config:
  - Active run: `GM2560-MFU-007-cw-20260614-1720`
  - Parent: `/dlwh/iris-run-job-20260615-001940`
  - Child: `/dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720`
  - Full May d2560 shape, 32 H100 nodes, `expert_axis=8`, `model_axis=1`, synthetic data, XLA CE, ring MoE, profiler/checkpoints/watch disabled.
- Result: child is `running` with 32/32 tasks running, `failure_count=0`, and `preemption_count=0`. Logs reached full static train summaries on ranks, including `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, and `Progress on:train -/4`, but no completed train-step metric appeared in the sampled logs. No fresh OOM, traceback, clique/rendezvous, or `kCollectivePermute` error appeared in the latest samples. Issue #6367 body was updated to reflect MFU-007 as the active gate instead of the obsolete MIN-001/Kueue CPU state.
- Sidecar status:
  - Fermat completed read-only bottleneck scouting. Added likely lanes: data/host stalls even with synthetic, per-step metric/logging overhead, CE/lm-head sharding, router/QB bookkeeping, and startup HLO/JAXPR logging/cache overhead.
  - Franklin completed an attention patch: FA4 metadata reuse for matching shape/window, stale window refresh, and stale metadata rejection at the kernel boundary. Validation reported `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q` -> `6 passed, 2 skipped`, py_compile passed, and focused pre-commit passed.
  - New sidecars: Meitner owns data/host synthetic-loader overhead; Bernoulli owns CE/lm-head sharding.
- Interpretation: the public tracking issue is current again, and the active run remains a live full-shape gate rather than a terminal result. The next decision should wait for either a completed step metric or a decisive failure/stall window. If it steps, launch a W&B/profile repeat with the same topology and profile scopes for attention, MoE, CE, router/QB, optimizer/update, FSDP/reshards, and host/data. If it fails on HBM, try `--remat recompute_all`; if it remains pre-step without errors, minimize full-shape executable components before another 32-node full profile.
- Next action: keep babysitting MFU-007; poll active sidecars; run changed-file validation after sidecar patches settle.

### 2026-06-14 17:38 PDT - MFU-007 still compiling; activation reshard warning persists
- Hypothesis: if the full d2560 executable is still following the bad activation layout path, logs should reproduce the SPMD full-rematerialization warning before any step metric.
- Command:
  - `uv run iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 | jq '{job_id:(.job_id // .id), state:(.state // .status), task_count:(.tasks|length? // .task_count), states:([.tasks[]? | (.state // .status)] | group_by(.) | map({state:.[0], count:length})), failure_count, preemption_count, error:(.error // .status_message // null)}'`
  - `uv run iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 --since-seconds 600 --max-lines 6000 | rg -i 'Progress on:train|train/loss|throughput/total_tokens|throughput/tokens_per_second|throughput/mfu|parameter_count|flops_per_token|RESOURCE_EXHAUSTED|OOM|out of memory|Traceback|Fatal|kCollectivePermute|rendezvous|clique|failed|Error|compil|memory|allocation|buffer|rematerialization|Data loader stalled'`
- Config: same GM2560-MFU-007 full-shape synthetic gate.
- Result: child remains `running` with 32/32 tasks running, `failure_count=0`, and `preemption_count=0`. No completed train-step metric appeared. Logs now show repeated XLA SPMD warnings across ranks for `%fake_parameter.2 = bf16[1,4096,2560]`, moving from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}` via involuntary full rematerialization. No fresh OOM, traceback, clique/rendezvous, or `kCollectivePermute` error appeared in this sample.
- Local code progress: added launcher controls for `MAY_LOG_EVERY`, `MAY_LOG_JAXPRS`, `MAY_LOG_XLA_HLO`, and the matching `SCALE_*` controls. Defaults for the CoreWeave launchers now disable JAXPR/HLO dumps for throughput probes unless explicitly re-enabled. Validation so far: py_compile passed for touched Python files, `bash -n` passed for both wrappers, and both May/scale dry-runs forwarded `log_every=10`, `log_jaxprs=false`, and `log_xla_hlo=false`.
- Interpretation: MFU-007 has not failed, but the full d2560 compile is still exercising the known dense activation reshard path. If it reaches steps, this warning becomes a profile target; if it remains pre-step much longer, the next minimizer should isolate the non-MoE dense projection/shared-MLP activation sharding at full seq/batch before another full d2560 profile.
- Next action: continue babysitting until a step metric, terminal failure, or a long enough pre-step stall to justify replacing the job.

### 2026-06-14 17:43 PDT - XLA CE model-axis sharded vocab path
- Hypothesis: the current XLA CE path reshards `lm_head` to fully replicated inside the CE `shard_map`; that is safe for `model_axis=1` but becomes a bottleneck if a future non-ring or model-axis-only profile uses `model_axis>1`.
- Command:
  - `uv run pytest lib/levanter/tests/grug/test_loss.py -q`
  - `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest lib/levanter/tests/grug/test_loss.py -q`
  - `uv run python -m py_compile lib/levanter/src/levanter/grug/loss.py lib/levanter/tests/grug/test_loss.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: local code only; active MFU-007 uses `model_axis=1`, so this patch does not affect the live job.
- Result: added an XLA-only CE path that keeps `lm_head` as `P(None, "model")` when `implementation="xla"` and mesh `model` axis size is >1. Each model shard computes local XLA CE and logsumexp over its vocab slice, then combines the global logsumexp and selected-label logit across the `model` axis. Added value and gradient parity tests for model-sharded vocab CE. Validation passed: normal test run `3 passed, 1 skipped`; forced two-device host mesh `4 passed`; py_compile passed; changed-file pre-commit passed. Pytest emitted JAX GC cleanup warnings after test completion, but exited 0.
- Interpretation: this is a safe future-path patch for model-axis CE profiles. It does not solve MFU-007's current CE/lm-head replication because `model_axis=1` leaves no vocab/model axis to shard over. If a profile shows CE HBM or `lm_head` reshard/all-gather hot, the next experiment should use a topology where a non-ring backend or model-axis-only diagnostic can exercise this sharded CE path.
- Next action: keep current full-shape gate on EP8/model1; only use this CE path after a viable model-axis topology exists or for isolated CE diagnostics.

### 2026-06-14 17:50 PDT - MFU-007 failed in 256-device clique init and was stopped
- Hypothesis: if MFU-007 is following the same bad path as GM006, it should either remain pre-step in clique/rendezvous init or hit a distributed coordination fatal before producing train-step metrics.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 --since-seconds 5400 --max-lines 50000 > /tmp/mfu007-full.log`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job bug-report /dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720 > /tmp/mfu007-bug-report.txt`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-001940`
- Config:
  - Run id: `GM2560-MFU-007-cw-20260614-1720`
  - Parent: `/dlwh/iris-run-job-20260615-001940`
  - Child: `/dlwh/iris-run-job-20260615-001940/grug-train-GM2560-MFU-007-cw-20260614-1720`
  - Full May d2560 shape, 32 H100 nodes, `expert_axis=8`, `model_axis=1`, synthetic data, XLA CE, ring MoE.
- Result: no train step completed. Logs reached `Progress on:train -/4`, selected XLA CE, and emitted the SPMD involuntary full-rematerialization warning for `%fake_parameter.2 = bf16[1,4096,2560]` from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}`. At ~00:44 UTC, ranks emitted 256-device `Initialize clique` warnings; at ~00:45 UTC the JAX coordination service declared tasks 17, 21, and 9 unhealthy, then the remaining tasks terminated with `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats)`. Iris showed the child as `running` but all 32 tasks back in `building`, `failure_count=1`, task 0 `Error`, and the other tasks bounced for atomic re-scheduling. The parent and child were stopped to avoid rebuilding the same failed full-shape attempt.
- Sidecar result: Kierkegaard's one-node EP8 ring/ragged probes (`GM2560-EP8-RING-D2560ISH-001` and `GM2560-EP8-RAGGED-D2560ISH-002`) never got a worker/container and were stopped from `BUILDING`; they add no runtime evidence, only a capacity/placement note.
- Interpretation: MFU-007 is a failed full-shape gate, not a slow compile. The signal points at the large `[1,4096,2560]` activation reshard / 256-device clique initialization path. It is still not an allocator OOM; it is a distributed executable/runtime failure after SPMD partitioning.
- Next action: launch a 32-node one-layer d2560 minimizer that preserves seq4096, batch256, e256/top4, EP8/model1, ring, XLA CE, and disables JAXPR/HLO dumps. If one layer fails the same way, fix activation sharding before more full-depth runs; if it steps, increase layers/depth to find the threshold.

### 2026-06-14 17:51 PDT - MIN-015 one-layer full-activation minimizer launched
- Hypothesis: preserving the full `[batch, seq, hidden] = [256,4096,2560]` activation shape with only one layer will tell us whether the bad sharding/clique path is independent of full 26-layer depth.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=2560 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_BATCH=256 SCALE_SEQ_LEN=4096 SCALE_STEPS=2 SCALE_REMAT=save_moe SCALE_PROFILER_STEPS=0 experiments/grug/moe/run_cw_scale.sh --full --data synthetic --checkpoints none --worker-cpu 32 --model-axis 1 --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --run-id GM2560-MIN-015-cw-20260614-1752 --submit`
- Config:
  - Run id: `GM2560-MIN-015-cw-20260614-1752`
  - Parent: `/dlwh/iris-run-job-20260615-005101`
  - Shape/topology: d2560, 1 layer, 256 experts, top-4, batch 256, seq 4096, 32 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, XLA CE.
  - Runtime: synthetic data, no checkpoints, watch disabled, profiler disabled, JAXPR/HLO dumps explicitly disabled.
- Result: parent submitted successfully. Child not yet discovered at launch time.
- Interpretation: this is the active minimizer for the MFU-007 failure boundary.
- Next action: babysit MIN-015 for child discovery, task placement, first-step metrics, or the same SPMD/clique/fatal signature.

### 2026-06-14 17:56 PDT - MIN-015 launcher bug fixed; MIN-016 replacement launched
- Hypothesis: MIN-015's failure was a launcher/config-versioning bug, not evidence about the d2560 one-layer executable.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-005101 --tail --max-lines 260`
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_is_dataclass_replaceable tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_vectorizes_token_generation tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_preserves_eos_segments -q`
  - `uv run python -m py_compile experiments/grug/moe/launch.py tests/test_grug_variant_contracts.py`
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=2560 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_BATCH=256 SCALE_SEQ_LEN=4096 SCALE_STEPS=2 SCALE_REMAT=save_moe SCALE_PROFILER_STEPS=0 experiments/grug/moe/run_cw_scale.sh --full --data synthetic --checkpoints none --worker-cpu 32 --model-axis 1 --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --run-id GM2560-MIN-016-cw-20260614-1758 --submit`
- Config:
  - Failed run id: `GM2560-MIN-015-cw-20260614-1752`
  - Failed parent: `/dlwh/iris-run-job-20260615-005101`
  - Replacement run id: `GM2560-MIN-016-cw-20260614-1758`
  - Replacement parent: `/dlwh/iris-run-job-20260615-005537`
  - Same one-layer full-activation shape/topology as MIN-015.
- Result: MIN-015 failed in the parent dispatcher before child launch with `ValueError: field _positions is declared with init=False, it cannot be specified with replace()`. The cause was `SyntheticGrugDataset` caching runtime arrays as dataclass `init=False` fields while Marin executor recursively versions configs through `dataclasses.replace`. Fixed the caches to be runtime attributes outside dataclass fields and added a regression test that the synthetic dataset is `dataclasses.replace`-safe. Focused tests passed (`3 passed`) and py_compile passed. MIN-016 submitted successfully; first poll showed the parent `pending` with no failures and no logs yet.
- Sidecar results:
  - Halley completed the attention lane: dynamic segment-id values do not retrace the jitted FA4 frontend; added `fa4_cute_metadata`, `fa4_cute_prepare_metadata`, and `fa4_cute_kernel` profiler scopes and a regression test. Validation: FA4 tests `7 passed, 2 skipped`, py_compile and focused pre-commit passed.
  - Harvey completed the host-overhead lane: gated explicit loop tracker writes by `log_every` and replaced repeated per-step `int(state.step)` reads with a Python loop counter. Validation: focused log-every test passed, full `tests/test_grug_variant_contracts.py` passed, and focused pre-commit passed.
- Interpretation: MIN-015 is not model evidence. MIN-016 is now the active one-layer full-activation minimizer, using the fixed synthetic dataset path and the same no-JAXPR/no-HLO controls.
- Next action: babysit MIN-016 for dispatcher launch, child placement, and either first-step metrics or the MFU-007 SPMD/clique/fatal signature.

### 2026-06-14 18:05 PDT - qkv output sharding patch; MIN-017 replacement launched
- Hypothesis: the `%fake_parameter.2 = bf16[1,4096,2560]` rematerialization warning may come from the first attention qkv projection: the qkv weight was replicated, but the `einsum` had no explicit output sharding, leaving XLA free to choose a hidden-dimension-sharded activation layout from the original weight layout.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers -q`
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_keep_fp32_master -q`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-005537`
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=2560 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_BATCH=256 SCALE_SEQ_LEN=4096 SCALE_STEPS=2 SCALE_REMAT=save_moe SCALE_PROFILER_STEPS=0 experiments/grug/moe/run_cw_scale.sh --full --data synthetic --checkpoints none --worker-cpu 32 --model-axis 1 --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --run-id GM2560-MIN-017-cw-20260614-1806 --submit`
- Config:
  - Stopped stale parent: `/dlwh/iris-run-job-20260615-005537`
  - Stopped stale child: `/dlwh/iris-run-job-20260615-005537/grug-train-GM2560-MIN-016-cw-20260614-1758`
  - New run id: `GM2560-MIN-017-cw-20260614-1806`
  - New parent: `/dlwh/iris-run-job-20260615-010535`
  - Same one-layer full-activation shape/topology as MIN-016.
- Result: MIN-016 got past dispatcher launch and created a child, but all 32 child tasks were still `building` with no worker assignment and zero failures/preemptions. Because it had not taken GPUs yet, it was stopped before placement. Added `out_sharding=_batch_spec()` to the attention qkv projection. Focused compile/lowering tests passed. MIN-017 submitted successfully from the patched bundle.
- Interpretation: MIN-017 supersedes MIN-016. It tests the same one-layer full-activation minimizer while also forcing qkv projection output back to canonical batch sharding, which should reduce the chance of the known hidden-dimension activation reshard.
- Next action: babysit MIN-017 for child discovery, task placement, and either first-step metrics or the MFU-007 SPMD/clique/fatal signature.

### 2026-06-14 18:18 PDT - MIN-017 invalidated by pre-init JAX array serialization
- Hypothesis: MIN-017's early `jax.distributed.initialize()` failure was caused by a JAX backend touch before Levanter distributed initialization, not by the d2560 executable.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-010535/grug-train-GM2560-MIN-017-cw-20260614-1806`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs --since-seconds 1800 --max-lines 8000 /dlwh/iris-run-job-20260615-010535/grug-train-GM2560-MIN-017-cw-20260614-1806 | rg -n -C 40 "jax\\.distributed\\.initialize\\(\\) must be called|Traceback|RuntimeError|XLA backend|libtpu|Running main|distributed"`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-010535`
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_vectorizes_token_generation tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_preserves_eos_segments tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_is_dataclass_replaceable tests/test_grug_variant_contracts.py::test_grug_moe_synthetic_dataset_serializes_without_jax_arrays tests/test_grug_variant_contracts.py::test_grug_moe_compute_live_params_one_step_lowers -q`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q`
- Config:
  - Invalidated run id: `GM2560-MIN-017-cw-20260614-1806`
  - Parent: `/dlwh/iris-run-job-20260615-010535`
  - Child: `/dlwh/iris-run-job-20260615-010535/grug-train-GM2560-MIN-017-cw-20260614-1806`
  - Same one-layer full-activation shape/topology as MIN-016.
- Result: MIN-017 reached worker startup, then task 0 failed twice before training with `RuntimeError: jax.distributed.initialize() must be called before any JAX calls that might initialise the XLA backend`. The child was still non-terminal and flapping (`failure_count=2`) when stopped. The likely cause was `SyntheticGrugDataset` carrying a cached `_loss_weight` as a `jax.Array` inside the Iris-submitted config; unpickling that config on the worker can initialize XLA before `trainer.initialize()`. The fix keeps synthetic dataset caches NumPy-only until `get_batch()` materializes `GrugLmExample`s after distributed init. Regression coverage verifies dataclass replaceability, pickle round-trip serialization with no `jax.Array` runtime attrs, and batch materialization. Focused synthetic/train tests passed (`5 passed`); FA4 tests passed (`8 passed, 2 skipped`); py_compile passed.
- Interpretation: MIN-017 is not model evidence. It never reached compile or the SPMD/clique path.
- Next action: relaunch the same one-layer d2560 minimizer as MIN-018 with the serialization fix and the current shared-dense token-sharding patch.

### 2026-06-14 18:29 PDT - MIN-018 waiting for placement; next sharding patch queued
- Hypothesis: after the synthetic serialization fix, MIN-018 should get past the pre-init JAX failure; remaining one-layer failure evidence should be a true model compile/runtime signal.
- Command:
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=2560 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_BATCH=256 SCALE_SEQ_LEN=4096 SCALE_STEPS=2 SCALE_REMAT=save_moe SCALE_PROFILER_STEPS=0 experiments/grug/moe/run_cw_scale.sh --full --data synthetic --checkpoints none --worker-cpu 32 --model-axis 1 --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --run-id GM2560-MIN-018-cw-20260614-1820 --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-012036`
  - `uv run pytest tests/test_grug_variant_contracts.py -q`
  - `uv run pytest lib/levanter/tests/grug/test_loss.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
- Config:
  - Run id: `GM2560-MIN-018-cw-20260614-1820`
  - Parent: `/dlwh/iris-run-job-20260615-012036`
  - Child: `/dlwh/iris-run-job-20260615-012036/grug-train-GM2560-MIN-018-cw-20260614-1820`
  - Same one-layer full-activation shape/topology as MIN-017, from checkpoint `7d7c78aba`.
- Result: MIN-018 submitted successfully and created the child job. As of the 18:29 PDT poll, parent and child are both running; the child is still `32/32 building`, with `failure_count=0`, `preemption_count=0`, no worker ids, and no child logs. This is placement wait, not model evidence yet. While waiting, added a local follow-up candidate that pins `GatedNorm` dense gate outputs and the attention head gate to the canonical batch/token sharding to remove two remaining unannotated `[B,S,*]` dense paths. Validation passed: Grug variant contracts `28 passed`, focused GatedNorm/attention/live-param tests `3 passed`, loss tests `3 passed, 1 skipped`, and changed-file pre-commit passed.
- Interpretation: keep MIN-018 running until it reaches worker startup/compile or fails. The new GatedNorm/attention-gate sharding patch is not in MIN-018 and should only be used for a replacement run if MIN-018 reproduces an activation-layout failure or must be relaunched for another reason.
- Next action: babysit MIN-018 for first worker logs. If it still fails with the `[1,4096,2560]` full-rematerialization/clique path, relaunch from the GatedNorm/attention-gate sharding checkpoint.

### 2026-06-14 18:43 PDT - MIN-019 one-layer d2560 gate succeeded after CPU admission fix
- Hypothesis: MIN-018 was not blocked on model compile or sharding; it was a Kueue topology admission issue from requesting too much CPU per 8xH100 worker. Relaunching the same one-layer d2560 gate from the latest sharding checkpoint with lower worker CPU should reach the real model signal.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-012036/grug-train-GM2560-MIN-018-cw-20260614-1820`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl get workload.kueue.x-k8s.io -n iris iris-pg-af8b3f892795fcb5-0 -o json | jq '{metadata:{name:.metadata.name,creationTimestamp:.metadata.creationTimestamp}, spec:{queueName:.spec.queueName, podSets:.spec.podSets}, status:.status}'`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-012036/grug-train-GM2560-MIN-018-cw-20260614-1820`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-012036`
  - `env SCALE_GPU_REPLICAS=32 SCALE_EXPERT_AXIS=8 SCALE_REPLICA_AXIS=1 SCALE_HIDDEN_DIM=2560 SCALE_NUM_LAYERS=1 SCALE_NUM_EXPERTS=256 SCALE_TOP_K=4 SCALE_BATCH=256 SCALE_SEQ_LEN=4096 SCALE_STEPS=2 SCALE_REMAT=save_moe SCALE_PROFILER_STEPS=0 experiments/grug/moe/run_cw_scale.sh --full --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --run-id GM2560-MIN-019-cw-20260614-1839 --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-013929/grug-train-GM2560-MIN-019-cw-20260614-1839`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-013929/grug-train-GM2560-MIN-019-cw-20260614-1839 --since-seconds 900 --max-lines 50000 > /tmp/min019-full.log`
- Config:
  - Stopped blocked run id: `GM2560-MIN-018-cw-20260614-1820`
  - Stopped parent: `/dlwh/iris-run-job-20260615-012036`
  - Stopped child: `/dlwh/iris-run-job-20260615-012036/grug-train-GM2560-MIN-018-cw-20260614-1820`
  - New run id: `GM2560-MIN-019-cw-20260614-1839`
  - New parent: `/dlwh/iris-run-job-20260615-013929`
  - New child: `/dlwh/iris-run-job-20260615-013929/grug-train-GM2560-MIN-019-cw-20260614-1839`
  - Same one-layer d2560 shape/topology as MIN-018, but `--worker-cpu 8` and source checkpoint `776b77e3c`.
- Result: Kueue reported MIN-018 could fit only `31` of `32` pods under topology `infiniband`, with `excluded: resource "cpu": 1`, for total requests of `cpu=1024`, `memory=8Ti`, `nvidia.com/gpu=256`, and `rdma/ib=256`. MIN-018 was stopped and replaced. MIN-019 admitted immediately, reached all 32 worker commands, and succeeded with `failure_count=0`, `preemption_count=0`. It completed `2/2` train progress. Across the 32 rank-local finish summaries: `mean_mfu=4.087078`, `p50_mfu=3.994016`, `p10_mfu=3.143514`, `p90_mfu=4.881084`, `max_mfu=5.617014`, `mean_tokens_per_s=4,151,640.98`, and `mean_examples_per_s=1,013.58`. Representative per-rank final loss was `train/loss=11.9346`, `cross_entropy_loss=11.9024`. The compiler still emitted the known SPMD full-rematerialization warning for `%fake_parameter.2 = bf16[1,4096,2560]` from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}`, but this one-layer run stepped and exited cleanly. Shutdown `WatchTasksAsync failed ... connection refused` warnings appeared after the coordinator task exited and did not affect Iris success.
- Sidecar results:
  - Copernicus completed the CE/logits lane. Streaming XLA CE does not materialize full `[tokens, vocab]` logits; at the issue shape a global fp32 logits tensor would be about `501 GiB`, and the earlier `4096 * 128256 * 32 * 4 = 62.625 GiB` number is a logits-family size rather than a tensor observed in the forced XLA CE path. Added a CE-side safety patch so untuned XLA CE uses the default block-size table as the batch-tile cap. This leaves the tuned H100 path at about an `8 MiB` fp32 logits tile for the current EP8/model1 shape, while preventing untuned/model-axis diagnostics from inferring much larger tiles.
  - Ohm completed the attention lane. May d2560 uses `gpu_fa4_cute` by default; dynamic segment ids flow through FA4/CuTe metadata as `[B,S]` lower bounds and validity, not `[B,S,S]` masks. Added PKO document-start reuse once per transformer call and removed the per-row `valid` load from the FA4/CuTe score predicate because invalid queries are encoded as `lower_bounds == seq_len`.
  - Parfit completed the activation-layout lane. Shared DenseMLP, GatedNorm, attention gate, routed MoE output, block residual stream, and PKO segment starts now have focused jaxpr coverage for canonical batch/token sharding. The remaining warning is not yet tied to an obvious unpinned high-level Grug op.
- Validation:
  - `uv run pytest tests/test_grug_variant_contracts.py -q` -> `30 passed, 2 warnings`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q` -> `9 passed, 2 skipped`
  - `uv run pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_fused_cross_entropy_xla_infer_uses_tuned_batch_block_size_when_available lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_fused_cross_entropy_xla_infer_uses_default_batch_block_size_without_tuned_match lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_fused_cross_entropy_xla_infer_falls_back_when_tuned_batch_block_size_is_unsafe -q` -> `3 passed`
  - `uv run pytest lib/levanter/tests/grug/test_loss.py -q` -> `3 passed, 1 skipped`
  - `./infra/pre-commit.py --changed-files --fix` -> OK
- Interpretation: the one-layer full-activation gate is no longer blocked by the prior startup bugs, Kueue CPU admission, or immediate clique failure. Absolute one-layer MFU is still far below the final `>=20` target and is based on one post-warmup sample, but it proves the d2560/seq4096/batch256/EP8/model1 executable can step. The persistent SPMD warning remains a profile target, not a hard one-layer blocker.
- Next action: commit/push the integrated CE/attention/activation patches and launch the next full-shape or depth-escalation synthetic run with `--worker-cpu 8`, `--ce-implementation xla`, `--moe-implementation ring`, `--model-axis 1`, `save_moe`, no checkpoints, and a short profiler window once startup is stable.

### 2026-06-14 19:09 PDT - MAY-020 reproduced the full-shape SPMD/clique failure
- Hypothesis: after the one-layer gate succeeded, the next useful signal is the exact issue #6044 May d=2560 architecture at full depth, synthetic data, and no profiler. If full depth reaches first train progress, we can add a short profile window; if it fails, the failure should identify the remaining bad sharding/collective path.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-020-cw-20260615-0151 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-015120/grug-train-GM2560-MAY-020-cw-20260615-0151 --since-seconds 900 --max-lines 120000 | rg -i "F0615|E0615|traceback|fatal|terminating process|unhealthy|stopped sending heartbeats|tasks have crashed|RESOURCE_EXHAUSTED|out of memory|oom|killed|signal|rendezvous|clique|spmd|rematerial|failed|preempt|evict|absl::Status" > /tmp/may020-failure.log`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-015120/grug-train-GM2560-MAY-020-cw-20260615-0151`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-015120`
- Config:
  - Run id: `GM2560-MAY-020-cw-20260615-0151`
  - Parent: `/dlwh/iris-run-job-20260615-015120`
  - Child: `/dlwh/iris-run-job-20260615-015120/grug-train-GM2560-MAY-020-cw-20260615-0151`
  - Source checkpoint: `334a99214`
  - Shape: May d=2560, `num_layers=26`, `num_heads=20`, `num_kv_heads=5`, `seq_len=4096`, `batch=256`, `num_experts=256`, top-4 routing, shared expert width 2560, `gpu_fa4_cute`, PKO enabled, ring MoE, EP=8/model=1/replica=1, XLA CE, synthetic data, no checkpoints, no profiler.
- Result: The run admitted and started all 32 workers. Hparams and parameter summaries were emitted; rank-local `parameter_count=67078882816`. It did not reach train progress. During first compile/initialization every worker emitted the same SPMD warning for `%fake_parameter.2 = bf16[1,4096,2560]` moving from `{devices=[256,1,1]}` to `{devices=[1,1,32,8] last_tile_dim_replicate}`. Several workers then logged 256-device `Initialize clique` waits; many of those reported as false-positive unsticks after 14-38 seconds, but the run later hit JAX coordination fatal errors: a rank terminated because tasks stopped sending heartbeats. Iris reported the child as still `running` but with `failure_count=1` and all tasks back in `assigned`, so the old run was retrying. I stopped both child and parent to avoid churning the same known-bad checkpoint.
- Sidecar results queued for replacement:
  - Ohm removed the separate FA4 `valid[B,S]` metadata from the Python custom VJP, FFI signatures, and CuTe launchers, relying only on `lower_bounds == seq_len` for invalid queries, and pinned FA4 lower bounds to token sharding. Validation: FA4 tests `9 passed, 2 skipped`, Grug variant contracts `34 passed`, changed-file pre-commit OK.
  - Parfit added `remat_mode="none"` as a diagnostic launcher/model option while keeping May default `save_moe`. Validation: Grug variant contracts `34 passed`, changed-file pre-commit OK. Recommendation remains `save_moe` for real d2560/26-layer runs; use `none` only on narrow probes.
  - Copernicus changed Pallas GPU CE backward to write vocab-block weight gradients into the final accumulator instead of stacking all fp32 vocab blocks. Validation: focused CE tests `3 passed`; still recommend `--ce-implementation xla` for the next May run.
- Validation:
  - `uv run pytest tests/test_grug_variant_contracts.py -q` -> `34 passed, 2 warnings`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q` -> `9 passed, 2 skipped`
  - `uv run pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_pallas_gpu_backward_streaming_from_lse_matches_reference_gradients lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_fused_cross_entropy_xla_infer_uses_default_batch_block_size_without_tuned_match lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_fused_cross_entropy_xla_infer_uses_tuned_batch_block_size_when_available -q` -> `3 passed`
  - `./infra/pre-commit.py --changed-files --fix` -> OK
- Interpretation: full depth, not one-layer, still triggers the SPMD/clique/heartbeat failure. The 60 GiB logits hypothesis remains unlikely because XLA CE was selected before failure and the observed warning is a `[1,4096,2560]` activation-layout tensor, not a vocab logits tile. The next attempt should first remove the extra FA4 validity metadata and pin lower-bound metadata sharding, because that is low-risk and directly reduces per-layer dynamic-segment metadata traffic. If that still fails, use a one-layer or few-layer `MAY_REMAT=none` diagnostic to test whether the checkpoint boundary is generating the `%fake_parameter.2` source sharding.
- Next action: commit/push the queued attention/remat/CE patches, then relaunch a short full-May synthetic gate from that new checkpoint with `save_moe`, XLA CE, EP=8, CPU=8, and no profiler.

### 2026-06-14 19:20 PDT - MAY-021 blocked on 32-node topology; launched N16 diagnostic
- Hypothesis: the replacement checkpoint `7af0c01b8` should be tested against the same full-May shape as MAY-020. If 32 H100 nodes are not immediately admissible, a 16-node full-depth diagnostic with EP=8 should still test whether the FA4 lower-bound metadata patch changes the compile/clique path while keeping capacity productive.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-021-cw-20260615-0210 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --submit`
  - `KUBECONFIG=$HOME/.kube/coreweave-iris-gpu kubectl get workload.kueue.x-k8s.io -n iris iris-pg-20b235ffcfca4a51-0 -o json | jq '{metadata:{name:.metadata.name,creationTimestamp:.metadata.creationTimestamp}, spec:{queueName:.spec.queueName, podSets:.spec.podSets}, status:.status}'`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-021028/grug-train-GM2560-MAY-021-cw-20260615-0210`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-021028`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-022N16-cw-20260615-0220 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --submit`
- Config:
  - Stopped 32-node parent: `/dlwh/iris-run-job-20260615-021028`
  - Stopped 32-node child: `/dlwh/iris-run-job-20260615-021028/grug-train-GM2560-MAY-021-cw-20260615-0210`
  - New 16-node parent: `/dlwh/iris-run-job-20260615-021953`
  - Run id: `GM2560-MAY-022N16-cw-20260615-0220`
  - Source checkpoint: `7af0c01b8`
  - Same full-May d=2560/L26/seq4096/batch256/EP8/model1/save_moe/XLA-CE/ring/synthetic/no-profiler config as MAY-021 except `MAY_GPU_REPLICAS=16`.
- Result: MAY-021 created child workload `iris-pg-20b235ffcfca4a51-0`, but Kueue reported topology `infiniband` could fit only `26` of `32` pods. Resource requests were `cpu=256`, `memory=8Ti`, `nvidia.com/gpu=256`, `rdma/ib=256`, and no worker pods existed. I stopped the 32-node parent/child before any model logs. MAY-022N16 submitted successfully; child/admission still pending at the time of this entry.
- Interpretation: MAY-021 did not test the replacement code. The N16 run is a capacity workaround and not the final target, but it preserves the critical code variable from MAY-020 to isolate whether the FA4 metadata sharding patch changes compile behavior at full depth.
- Next action: babysit MAY-022N16 for admission, hparams, SPMD warning presence/absence, clique/rendezvous behavior, and train progress. If N16 succeeds, retry 32 nodes when topology can fit them. If N16 still fails with the same `[1,4096,2560]` warning, switch to `MAY_REMAT=none` or lower depth to isolate the checkpoint boundary.

### 2026-06-14 19:35 PDT - MAY-022N16 exposed FA4/CuTe backward compiler failure
- Hypothesis: if the FA4 metadata sharding patch changes the full-depth failure mode, the 16-node full-May diagnostic should either reach first train progress or fail before the old 256-device clique path.
- Command:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-021953/grug-train-GM2560-MAY-022N16-cw-20260615-0220 --since-seconds 360 --max-lines 300 | tail -n 120`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-021953/grug-train-GM2560-MAY-022N16-cw-20260615-0220`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-021953`
  - Dry-run for next diagnostic: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-023L1REF-cw-20260615-0238 --nodes 16 --layers 1 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --attention reference --ce-implementation xla --moe-implementation ring`
- Config:
  - Run id: `GM2560-MAY-022N16-cw-20260615-0220`
  - Parent: `/dlwh/iris-run-job-20260615-021953`
  - Child: `/dlwh/iris-run-job-20260615-021953/grug-train-GM2560-MAY-022N16-cw-20260615-0220`
  - Source checkpoint: `7af0c01b8`
  - Shape: full May d=2560/L26/seq4096/batch256/EP8/model1/save_moe/XLA-CE/ring/synthetic/no-profiler on 16 H100 nodes.
- Result: The run admitted and started 16/16 workers with `failure_count=0` until compile. It emitted the exact target hparams and `parameter_count=67078882816`; XLA CE selected on ranks. It then failed in the FA4/CuTe segmented backward compiler, not in the earlier clique path: `_fa4_cute_segmented_bwd.py:1285` calls `cute.arch.atomic_add(utils.elem_pointer(tdQgdQaccum_atomic, i), acc_dQ_atomic[i])`, and CUTLASS DSL raised `TypeError: atomicrmw() missing 1 required positional argument: 'a'` while compiling the dQ atomic-add loop. I stopped the child and parent to avoid retry churn.
- Sidecar/code changes queued:
  - Attention: Ptolemy pinned the forward FA4 lower-bound metadata FFI spec to `TensorSpec(mode=(0, 1), static=True)`, matching backward metadata layout. Validation: `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q` -> `10 passed, 2 skipped`; changed-file pre-commit OK.
  - CE/output: Turing added H100/A100/GB10 tuned Pallas GPU CE blocks for `H=2560,V=128256` local `B in {4096,8192}` so the Pallas comparison is launch-viable. Mainline remains XLA CE because it avoids full logits and should use larger vocab tiles. Validation: focused H100 test `2 passed`; changed-file pre-commit OK.
  - Diagnostics: the May launcher now accepts `MAY_NUM_LAYERS` / `--layers` plus `MAY_USE_PKO` and `MAY_PKO_ON_LAST_LAYER` toggles, so narrow full-width probes can isolate attention, PKO, remat, and MoE without changing the default issue shape.
- Validation:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_may_launcher_diagnostic_overrides -q` -> `1 passed`
  - `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -q` -> `10 passed, 2 skipped`
  - `uv run pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py::test_h100_d2560_pallas_gpu_block_sizes_fit_nvidia_tile_budget -q` -> `2 passed`
  - `./infra/pre-commit.py --changed-files --fix` -> OK
- Interpretation: the old full-depth SPMD/clique symptom is no longer the only blocker; the current `gpu_fa4_cute` backward is itself not compiling on the deployed CUTLASS DSL for this shape. `gpu_fa4_thd` is not a full-model fallback yet because the model only builds FA4/CuTe metadata and THD rejects the short sliding-window layers. The next diagnostic should bypass FA4/CuTe backward with `--attention reference` and `--layers 1` to test whether MoE/optimizer/CE can still step from the new launcher checkpoint. If that works, escalate layer count/reference only enough to preserve cluster productivity while the FA4 backward atomic path is fixed.
- Next action: commit/push the queued attention, CE, and diagnostic launcher changes, launch `GM2560-MAY-023L1REF-cw-20260615-0238`, and babysit it for train progress or a non-attention failure.

### 2026-06-14 20:15 PDT - LAYOUT-004 disproved output-head-only SPMD hypothesis
- Hypothesis: the `%fake_parameter.2 = bf16[2,4096,2560]` full-rematerialization warning might be caused by the output projection/lm-head sharding path; replicating the output projection should remove the warning if that is the root cause.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-LAYOUT-004OUTREP-ref-rematnone-n16-20260615-0301 --nodes 16 --layers 1 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 16 --batch 256 --seq-len 4096 --steps 2 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --attention reference --ce-implementation xla --moe-implementation ring --remat none --use-pko false --pko-on-last-layer false --output-proj-sharding replicated --submit`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-030119/grug-train-GM2560-LAYOUT-004OUTREP-ref-rematnone-n16-20260615-0301`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-030119/grug-train-GM2560-LAYOUT-004OUTREP-ref-rematnone-n16-20260615-0301 --since-seconds 7200 --max-lines 200000 | rg -i -m 20 'spmd|fake_parameter|involuntary full rematerialization|This may be solved by using maximal sharding'`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-030119`
  - Child: `/dlwh/iris-run-job-20260615-030119/grug-train-GM2560-LAYOUT-004OUTREP-ref-rematnone-n16-20260615-0301`
  - Source checkpoint: `2b2070d03`
  - Shape: d2560, 1 layer, reference attention, no PKO, `remat=none`, XLA CE, ring MoE, synthetic data, no checkpoints, no profiler.
  - Topology: 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=16`, so `data=1` and `batch_shards=128`.
  - Diagnostic knob: `output_proj_sharding=replicated`.
- Result: The run succeeded with `16/16` tasks, zero failures, and zero preemptions. It completed `2/2` train steps. The output-projection replication did not remove the warning: every task still logged `[SPMD] Involuntary full rematerialization` for `%fake_parameter.2 = bf16[2,4096,2560]`, moving from `{devices=[128,1,1]<=[128]}` to `{devices=[1,1,16,8]<=[128] last_tile_dim_replicate}`. Rank-local finish summaries reported duplicated one-sample values with `min_mfu=4.07199`, `p50_mfu=4.34803`, `mean_mfu=4.54394`, and `max_mfu=7.22924`; the max is a clear one-sample outlier.
- Interpretation: output head sharding is not sufficient to explain the warning. The target layout shards the hidden dimension over a cross-slice axis while replicating over expert, so the next likely suspects are embedding/hidden-axis parameter layouts or a transpose-like backward path, not logits materialization. Because this probe used reference attention and no remat, the warning is also not specific to FA4/CuTe or checkpoint boundaries.
- Next action: run a one-axis diagnostic that changes tensor/model layout without reintroducing FA4 or remat. The next candidate is L1/reference/remat-none with `model_axis=2` on 16 H100 nodes to see whether moving hidden/channel sharding off `replica_dcn` changes or removes the `%fake_parameter.2` warning.

### 2026-06-14 20:19 PDT - LAYOUT-005M2 invalid under local EP/model group rule
- Hypothesis: `model_axis=2` with `expert_axis=8` would test whether moving hidden/channel sharding off the cross-slice `replica_dcn` axis changes the `%fake_parameter.2` warning while otherwise matching LAYOUT-004.
- Command:
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-LAYOUT-005M2-ref-rematnone-n16-20260615-0322 --nodes 16 --layers 1 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 2 --expert-axis 8 --replica-axis 1 --batch 256 --seq-len 4096 --steps 2 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --attention reference --ce-implementation xla --moe-implementation ring --remat none --use-pko false --pko-on-last-layer false --output-proj-sharding replicated --submit`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-031750`
  - Intended topology: 16 H100 nodes, `expert_axis=8`, `model_axis=2`, `replica_axis=1`.
- Result: The dispatcher failed before child submission with `ValueError: MAY_EXPERT_AXIS * MAY_MODEL_AXIS must divide the 8 GPUs on each worker so expert/model groups stay local; got 8 * 2 = 16.`
- Interpretation: this run is launcher-constraint evidence only, not model evidence. The current local-group rule prevents testing model parallelism while keeping EP8 on 8-GPU H100 nodes.
- Next action: use either a valid approximation (`expert_axis=4`, `model_axis=2`) to test hidden/model-axis layout effects, or add a separate embedding-sharding diagnostic that keeps EP8 fixed.

### 2026-06-14 20:22 PDT - LAYOUT-006 failed the ring MoE model-axis guard before child submission
- Hypothesis: reducing expert parallelism to `expert_axis=4` while using `model_axis=2` would keep expert/model groups local on 8-GPU H100 workers and allow a one-layer/reference diagnostic to test whether hidden/model-axis layout changes the `%fake_parameter.2` SPMD warning.
- Command:
  - Parent summary: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-032039 | jq '{state, task_count, completed_count, failure_count, preemption_count, task_state_counts, has_children: (.children != null and (.children|length > 0)), children}'`
  - Parent list: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-032039 | jq '[.[] | {name, state, task_count, completed_count, failure_count, preemption_count, task_state_counts}]'`
  - Parent logs: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-032039 --since-seconds 7200 --max-lines 200000 | rg -i "GM2560-LAYOUT-006EP4M2|grug-train|Job submitted|iris-run-job|child|submitted|state|failed|traceback|exception|error|warning|fake_parameter|spmd|mfu|train/progress"`
- Config:
  - Run id: `GM2560-LAYOUT-006EP4M2-ref-rematnone-n16-20260615-0321`
  - Parent: `/dlwh/iris-run-job-20260615-032039`
  - Child: none created.
  - Intended shape: d2560, 1 layer, reference attention, no remat, XLA CE, ring MoE, `model_axis=2`, `expert_axis=4`, `replica_axis=1`, `output_proj_sharding=replicated`, 16 H100 nodes.
- Result: The parent failed after one task and did not create a child job (`has_children=false`). Iris summary reported `state=failed`, `failure_count=1`, `preemption_count=0`, `task_count=1`, `completed_count=0`, and task state counts `{"failed": 1}`. Logs show `ValueError: MAY_MOE_IMPLEMENTATION=ring currently requires either MAY_EXPERT_AXIS=1 or MAY_MODEL_AXIS=1 on CoreWeave H100s; got MAY_EXPERT_AXIS=4, MAY_MODEL_AXIS=2. Use model_axis>1 only for attention/model-axis diagnostics with expert_axis=1, or keep model_axis=1 for ring expert-parallel runs.`
- Interpretation: LAYOUT-006 did not admit/start and produced no SPMD warning or MFU metrics. The remaining valid model-axis diagnostic must use `expert_axis=1, model_axis=2`; any ring MoE EP diagnostic must keep `model_axis=1`.
- Next action: run the model-axis diagnostic with `expert_axis=1, model_axis=2` if the immediate question is hidden/model-axis layout, or keep `expert_axis=8, model_axis=1` and inspect HLO/sharding dumps to localize the `%fake_parameter.2` source without changing ring EP.

### 2026-06-14 20:25 PDT - LAYOUT-007 EP8 input-embedding replication launched
- Hypothesis: the persistent LAYOUT-004 warning comes from input embedding hidden-batch sharding rather than output head sharding. Replicating both input embedding and output projection should remove or change `%fake_parameter.2` if embedding gradient/layout is responsible, while preserving the EP8/model1 topology.
- Command:
  - `uv run pytest tests/test_grug_variant_contracts.py::test_grug_moe_may_launcher_diagnostic_overrides -q`
  - `uv run python -m py_compile experiments/grug/moe/model.py experiments/grug/moe/launch_cw_may_d2560.py`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id DRYRUN-EMBED-REPL --nodes 16 --layers 1 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 16 --batch 256 --seq-len 4096 --steps 2 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --attention reference --ce-implementation xla --moe-implementation ring --remat none --use-pko false --pko-on-last-layer false --input-embed-sharding replicated --output-proj-sharding replicated`
  - `./infra/pre-commit.py --changed-files --fix`
  - `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-LAYOUT-007EMBREP-ref-rematnone-n16-20260615-0333 --nodes 16 --layers 1 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 16 --batch 256 --seq-len 4096 --steps 2 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --attention reference --ce-implementation xla --moe-implementation ring --remat none --use-pko false --pko-on-last-layer false --input-embed-sharding replicated --output-proj-sharding replicated --submit`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-032423`
  - Source: dirty worktree with new `input_embed_sharding` diagnostic knob, validated by focused test, py_compile, dry run, and changed-file pre-commit.
  - Shape/topology: same as LAYOUT-004 except `input_embed_sharding=replicated` and `output_proj_sharding=replicated`.
- Result: parent submitted successfully and passed the immediate validation window; at first poll parent was `JOB_STATE_RUNNING`, `failure_count=0`, with no child visible yet.
- Interpretation: this is the first EP8-preserving diagnostic that directly tests the embedding-hidden-sharding hypothesis.
- Next action: babysit LAYOUT-007 for child discovery, SPMD warning presence/source-target sharding, and MFU metrics if it succeeds.

### 2026-06-14 20:28 PDT - LAYOUT-007 succeeded and removed the fake_parameter.2 warning
- Hypothesis: the persistent LAYOUT-004 warning came from input embedding hidden-batch sharding rather than output head sharding. Replicating both input embedding and output projection should remove or change `%fake_parameter.2` if embedding gradient/layout was responsible, while preserving the EP8/model1 topology.
- Command:
  - Child discovery: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-032423`
  - Child summary: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-032423/grug-train-GM2560-LAYOUT-007EMBREP-ref-rematnone-n16-20260615-0333`
  - Warning scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-032423/grug-train-GM2560-LAYOUT-007EMBREP-ref-rematnone-n16-20260615-0333 --since-seconds 3600 --max-lines 400000 | rg -n -C 4 "fake_parameter\\.2|Involuntary full rematerialization|source sharding|target sharding|spmd_partitioner|compiler_base\\.cc:2587"`
  - MFU extraction: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-032423/grug-train-GM2560-LAYOUT-007EMBREP-ref-rematnone-n16-20260615-0333 --since-seconds 3600 --max-lines 400000 | rg -o '"throughput/mfu": [0-9.]+' | sed 's/.*: //' | sort -n -u`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-032423`
  - Child: `/dlwh/iris-run-job-20260615-032423/grug-train-GM2560-LAYOUT-007EMBREP-ref-rematnone-n16-20260615-0333`
  - Shape/topology: d2560, 1 layer, reference attention, no PKO, `remat=none`, XLA CE, ring MoE, synthetic data, no checkpoints, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=16`, `data=1`, `batch_shards=128`.
  - Diagnostic knobs: `input_embed_sharding=replicated`, `output_proj_sharding=replicated`.
- Result: parent and child both reached `JOB_STATE_SUCCEEDED`. The child had `task_count=16`, `completed_count=16`, `failure_count=0`, `preemption_count=0`, and `task_state_counts={"succeeded": 16}`. The warning scan returned no matches for `%fake_parameter.2`, involuntary full rematerialization, source/target sharding, `spmd_partitioner`, or `compiler_base.cc:2587`. Final per-rank one-sample MFU values were `3.500033, 3.618005, 3.677199, 3.832916, 3.900821, 3.935097, 3.980240, 4.115334, 4.122488, 4.124135, 4.186020, 4.190615, 4.443462, 4.638330, 5.774185, 7.422358`; deduped rank summary `count=16`, `min=3.500033`, median bracket `4.115334..4.122488`, `mean=4.341327`, `max=7.422358`.
- Interpretation: replicated input embedding is the first EP8-preserving change that removes the `%fake_parameter.2` SPMD rematerialization warning. This strongly implicates the input embedding hidden/batch sharding or its gradient path, not the output projection alone. It does not improve MFU by itself; this short L1/reference diagnostic is still around 4.34 mean MFU with one outlier rank.
- Next action: keep `input_embed_sharding=replicated` for subsequent EP8 diagnostics, then re-enable FA4/CuTe after the segmented-backward atomic fix and compare against the LAYOUT-004 reference baseline.

### 2026-06-14 20:35 PDT - MAY-024 full-depth N16 embedding-layout gate launched
- Hypothesis: if LAYOUT-007 correctly localized the `%fake_parameter.2` warning to the input embedding hidden/batch sharding path, then the full May d2560 shape should at least compile farther with `input_embed_sharding=replicated` and `output_proj_sharding=replicated` while keeping EP8/model1 and the FA4/CuTe attention path enabled.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-024EMBREP-N16-cw-20260615-0335 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-024EMBREP-N16-cw-20260615-0335 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --submit`
  - Initial poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-033410 | jq '[.[] | {name, state, task_count, completed_count, failure_count, preemption_count, task_state_counts}]'`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-033410`
  - Child discovered: `/dlwh/iris-run-job-20260615-033410/grug-train-gm2560-may-024embrep-n16-cw-20260615-0335`
  - Source checkpoint: `6b33c814a5`
  - Shape: full May d2560 default layer count, seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler.
  - Topology/backend: 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE.
  - Diagnostic knobs: `input_embed_sharding=replicated`, `output_proj_sharding=replicated`.
- Result: Dry run matched the intended full-depth N16 gate. Submit created parent `/dlwh/iris-run-job-20260615-033410`; first read-only poll found both parent and child in `JOB_STATE_RUNNING`, with child `task_count=16`, `task_state_counts={"running": 16}`, `failure_count=0`, and `preemption_count=0`. Second poll found hparams on the child confirming full d2560/L26, FA4/CuTe attention, XLA CE, `input_embed_sharding=replicated`, `output_proj_sharding=replicated`, `remat_mode=save_moe`, `expert_axis_size=8`, `model_axis_size=1`, `replica_axis_size=1`; mesh summaries confirmed `data=16`, `expert=8`, `model=1`, `batch_shards=128`; parameter summaries reported `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, and `throughput/theoretical_flops=1.26656e+17`. Compile then failed in the segmented FA4/CuTe backward kernel before any MFU was emitted: `_fa4_cute_segmented_bwd.py:1285` called `utils.atomic_add_fp32`, which entered `flash_attn/cute/utils.py:486` and raised `TypeError: atomicrmw() got an unexpected keyword argument 'res'`. I stopped the child and parent after the compile failure. No `%fake_parameter.2`, involuntary-rematerialization, clique, `compiler_base.cc:2587`, OOM, or HBM signal appeared before the FA4 failure.
- Interpretation: the replicated input/output embedding layout kept the old SPMD/clique warning out of this full-depth N16 attempt, but FA4/CuTe backward still blocks the fast-attention path. The new failure is narrower than MAY-022: the previous positional atomic signature patch got to the flash-attn helper, but that helper is stale against `nvidia-cutlass-dsl==4.5.2` because generated `nvvm.atomicrmw` now infers result type and no longer accepts `res=`.
- Next action: replace the three segmented-backward atomic accumulations with CUTLASS DSL's public version-aware `cute.arch.atomic_add(ptr, value)` wrapper, validate statically/local tests, commit, and relaunch the full-depth N16 gate as MAY-025.

### 2026-06-14 21:05 PDT - MAY-025 passed FA4 atomic compile and hit first-step HBM OOM
- Hypothesis: replacing the segmented FA4/CuTe backward atomic accumulations with CUTLASS DSL's public `cute.arch.atomic_add(ptr, value)` wrapper should clear the MAY-024 compile failure and let the full May d2560 N16 gate reach train progress.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-025ATOM-N16-cw-20260615-0345 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-025ATOM-N16-cw-20260615-0345 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --submit`
  - State poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-034953`
  - Error scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-034953/grug-train-GM2560-MAY-025ATOM-N16-cw-20260615-0345 --since-seconds 1800 --max-lines 100000 | rg -n -C 12 'Out of memory while trying to allocate 74\\.75GiB|RESOURCE_EXHAUSTED|Fatal error in grug training loop|Traceback|Largest program allocations|Program hbm requirement|fake_parameter\\.2|Involuntary full rematerialization|atomicrmw|TypeError'`
  - Stop after unrecoverable OOM: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-034953/grug-train-GM2560-MAY-025ATOM-N16-cw-20260615-0345`
  - Stop parent: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-034953`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-034953`
  - Child: `/dlwh/iris-run-job-20260615-034953/grug-train-GM2560-MAY-025ATOM-N16-cw-20260615-0345`
  - Source checkpoint: `303788a88`
  - Shape: full May d2560/L26, seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler.
  - Topology/backend: 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE.
  - Memory/layout knobs: `input_embed_sharding=replicated`, `output_proj_sharding=replicated`, `live_param_mode=param`, `mp=params=float32,compute=bfloat16,output=bfloat16`.
- Result: The child admitted and matched the intended hparams. It emitted full d2560/L26 summaries with `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, `throughput/flops_per_example_analytic=70124065259520.0`, and `throughput/theoretical_flops=1.26656e+17`. It did not reproduce the MAY-024 FA4 atomic compile failure (`atomicrmw`, `atomic_add`, and `TypeError` absent) and did not reproduce the earlier `%fake_parameter.2` / involuntary rematerialization warning. It reached the train loop (`Progress on:train -/4`) and then failed before first completed step: multiple ranks raised `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 74.75GiB`, with BFC rounding the request to `80265772288` bytes on individual GPUs. The stack top was `experiments/grug/moe/train.py:580`, `jax.block_until_ready(metrics["train/loss"])`. I stopped the child and parent; final state for both is `JOB_STATE_KILLED`, child `task_count=16`, `completed_count=16`, `failure_count=1`, `preemption_count=0`, `task_state_counts={"killed": 16}`.
- Interpretation: the CUTLASS atomic wrapper fixed the immediate FA4/CuTe backward compile blocker. The next blocker is first-step HBM, not the old clique/SPMD warning. Because MAY-025 used `live_param_mode=param`, the first memory-isolation probe should keep the exact same shape/backend and switch only to `compute_with_master`, which keeps bf16 live parameters for forward/backward plus fp32 master params for optimizer updates. If the same 74.75 GiB allocation remains, the offending buffer is likely independent of live parameter dtype and we should localize with HLO or a batch/layer/remat axis.
- Next action: launch MAY-026 with the same N16 full-May gate plus `--live-param-mode compute_with_master`.

### 2026-06-14 21:10 PDT - MAY-026 live-bf16/master-fp32 memory gate launched
- Hypothesis: if MAY-025's first-step 74.75 GiB allocation is driven by fp32 live parameters or fp32 parameter all-gathers in the forward/backward path, switching only `live_param_mode` from `param` to `compute_with_master` should reduce live parameter dtype pressure while preserving a sharded fp32 master tree for optimizer state and updates.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-026LIVEBF16-N16-cw-20260615-0408 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-026LIVEBF16-N16-cw-20260615-0408 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --submit`
  - Initial poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-040414`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-040414`
  - Expected child: `/dlwh/iris-run-job-20260615-040414/grug-train-GM2560-MAY-026LIVEBF16-N16-cw-20260615-0408`
  - Source checkpoint: `303788a88`
  - Same as MAY-025 except `live_param_mode=compute_with_master`.
- Result: submit created parent `/dlwh/iris-run-job-20260615-040414`. First poll found parent `JOB_STATE_RUNNING`, child discovered as `/dlwh/iris-run-job-20260615-040414/grug-train-gm2560-may-026livebf16-n16-cw-20260615-0408`, child `JOB_STATE_PENDING`, `task_count=16`, `failure_count=0`, `preemption_count=0`, and `task_state_counts={"pending": 16}`.
- Interpretation: admission started cleanly. This is a one-axis memory isolation run, not a new performance claim.
- Next action: babysit for hparams confirmation, first train step, MFU, or recurrence of the 74.75 GiB OOM.

### 2026-06-14 21:25 PDT - MAY-026 reproduced HBM OOM under live bf16 params
- Hypothesis: if MAY-025's first-step 74.75 GiB allocation was caused by fp32 live parameters or fp32 parameter all-gathers in the forward/backward path, `live_param_mode=compute_with_master` should substantially reduce the first-step allocation while preserving sharded fp32 master parameters for optimizer updates.
- Command:
  - Status poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-040414`
  - Error scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-040414 --since-seconds 1800 --max-lines 220000 | rg -i -m 320 -C 6 'RESOURCE_EXHAUSTED|Out of memory|74\\.75GiB|80265772288|Fatal error in grug training loop|Traceback|jax.errors.JaxRuntimeError|Largest program allocations|Program hbm requirement|train/loss|throughput/mfu|Progress on:train|failed|Error|Exception|OwnerDied|CANCELLED|atomicrmw|TypeError|fake_parameter\\.2|Involuntary full rematerialization|rendezvous|clique'`
  - Stop after unrecoverable OOM: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-040414/grug-train-GM2560-MAY-026LIVEBF16-N16-cw-20260615-0408`
  - Stop parent: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-040414`
  - Final state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-040414`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-040414`
  - Child: `/dlwh/iris-run-job-20260615-040414/grug-train-GM2560-MAY-026LIVEBF16-N16-cw-20260615-0408`
  - Source checkpoint: `303788a88`
  - Same as MAY-025 except `live_param_mode=compute_with_master`; full May d2560/L26, seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings.
- Result: The child admitted and confirmed the intended hparams: `live_param_mode=compute_with_master`, `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, `throughput/flops_per_example_analytic=70124065259520.0`, and `throughput/theoretical_flops=1.26656e+17`. It reached train progress `-/4` and then failed before the first completed step with the same class of first-step HBM exhaustion: repeated BFC allocator messages on rank 0 GPUs requested `74.42GiB`, rounded to `79905595648` bytes, followed by `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 74.42GiB` at `experiments/grug/moe/train.py:580`, `jax.block_until_ready(metrics["train/loss"])`. I stopped the child and parent after the automatic retry appeared; final states were `JOB_STATE_KILLED`, child `task_count=16`, `completed_count=16`, `failure_count=1`, `preemption_count=0`, `task_state_counts={"killed": 16}`.
- Interpretation: the 74 GiB allocation is not primarily caused by fp32 live-parameter storage in the forward/backward path. The recurrence under bf16 live params keeps the likely causes in CE/program layout, remat/FSDP buffering, optimizer/update layout, or a full-program compiler allocation. Local source inspection found that with `model_axis=1`, XLA CE uses a replicated `lm_head_spec=P(None, None)` and streams local `B=8192` with small logits tiles; the explicit CE tile and local weight-gradient accumulator are far smaller than 74 GiB, so this is likely repeated/full-program buffering rather than a single Python-visible logits tensor.
- Next action: run a one-axis CE probe that keeps the same full May N16/EP8/FA4/replicated-embedding shape and switches only from XLA CE to `pallas_gpu`, using `compute_with_master`, to determine whether the XLA CE path is the allocator trigger.

### 2026-06-14 21:30 PDT - MAY-027 Pallas CE memory-localization probe launched
- Hypothesis: if the recurring 74 GiB first-step allocation is triggered by the XLA CE streaming/lm-head layout path rather than the rest of the model or optimizer, switching only the CE implementation to `pallas_gpu` should change the failure mode or allow the first step to complete.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-027PALLASCE-N16-cw-20260615-0427 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation pallas_gpu --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-027PALLASCE-N16-cw-20260615-0427 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation pallas_gpu --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --submit`
  - Initial poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-042201`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-042201`
  - Expected child: `/dlwh/iris-run-job-20260615-042201/grug-train-GM2560-MAY-027PALLASCE-N16-cw-20260615-0427`
  - Source checkpoint: `303788a88`
  - Same full May d2560/L26 N16 gate as MAY-026: seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, FA4/CuTe attention, ring MoE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
  - One-axis change from MAY-026: `cross_entropy_implementation=pallas_gpu` instead of `xla`.
- Result: Dry run matched the intended CE-only localization probe. Submit created parent `/dlwh/iris-run-job-20260615-042201`. First poll found only the parent, `JOB_STATE_RUNNING`, `task_count=1`, `failure_count=0`, `preemption_count=0`, and no child logs yet. The next poll found the child `/dlwh/iris-run-job-20260615-042201/grug-train-gm2560-may-027pallasce-n16-cw-20260615-0427` running `16/16`, with `failure_count=0` and `preemption_count=0`. Logs then confirmed the full hparams with `cross_entropy_implementation=pallas_gpu`, `live_param_mode=compute_with_master`, `attention_implementation=gpu_fa4_cute`, `input_embed_sharding=replicated`, and `output_proj_sharding=replicated`; summaries matched the prior full-May gates with `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, `throughput/flops_per_example_analytic=70124065259520.0`, and `throughput/theoretical_flops=1.26656e+17`. All ranks selected `Fused cross-entropy selected implementation: pallas_gpu` and reached `Progress on:train -/4`.
- Interpretation: the run is now a valid CE-only comparison against MAY-026. It has not yet proven whether `pallas_gpu` avoids the 74 GiB first-step allocation.
- Next action: babysit for first train step, MFU, CE autotune/compile behavior, or any allocator/atomic/SPMD/clique failure.

### 2026-06-14 21:36 PDT - MAY-027 Pallas CE reproduced first-step HBM OOM
- Hypothesis: if the recurring 74 GiB first-step allocation is triggered by the XLA CE streaming/lm-head layout path rather than the rest of the model or optimizer, switching only the CE implementation to `pallas_gpu` should change the failure mode or allow the first step to complete.
- Command:
  - Status poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-042201`
  - Failure scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-042201/grug-train-GM2560-MAY-027PALLASCE-N16-cw-20260615-0427 --since-seconds 2400 --max-lines 600000 | rg -v '"event": "hparams"' | rg -i -m 80 -C 4 '"event": "train"|train/loss|throughput/mfu|examples_per_second|tokens_per_second|step_time|Progress on:train'`
  - Error scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-042201/grug-train-GM2560-MAY-027PALLASCE-N16-cw-20260615-0427 --since-seconds 2400 --max-lines 600000 | rg -v '"event": "hparams"|xla_bridge Unable to initialize backend|pip install jax\\[k8s' | rg -i -m 240 -C 8 'Fatal error in grug training loop|Traceback|jax\\.errors|RESOURCE_EXHAUSTED|Out of memory|Program hbm requirement|Largest program allocations|CUDA_ERROR|segmentation fault|RuntimeError|ValueError|TypeError|Exception|failed|Error received from peer|WatchTasksAsync failed|CANCELLED|UNAVAILABLE|atomicrmw|Involuntary full rematerialization|fake_parameter\\.2|clique|Pallas|pallas|triton|LLVM|nvptx'`
  - Stop child after recurrence: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-042201/grug-train-GM2560-MAY-027PALLASCE-N16-cw-20260615-0427`
  - Stop parent: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260615-042201`
  - Final state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-042201`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-042201`
  - Child: `/dlwh/iris-run-job-20260615-042201/grug-train-GM2560-MAY-027PALLASCE-N16-cw-20260615-0427`
  - Source checkpoint: `303788a88`
  - Same full May d2560/L26 N16 gate as MAY-026 with `live_param_mode=compute_with_master`, `input_embed_sharding=replicated`, and `output_proj_sharding=replicated`.
  - One-axis change from MAY-026: `cross_entropy_implementation=pallas_gpu` instead of `xla`.
- Result: MAY-027 selected `pallas_gpu` on all ranks and reached train progress `-/4`, but the first attempt failed before any completed train event, MFU, W&B run, or profile artifact. Multiple tasks reported the same allocator shape as MAY-025/MAY-026: BFC tried to allocate `74.41GiB`, rounded to `79894201856` bytes, then raised `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 74.41GiB` at `experiments/grug/moe/train.py:580`, `jax.block_until_ready(metrics["train/loss"])`. Iris bounced the coscheduled child for a retry after `failure_count=1`; I stopped the retry and parent. Final child state is `JOB_STATE_KILLED`, `task_count=16`, `completed_count=16`, `failure_count=1`, `preemption_count=0`, `task_state_counts={"killed": 16}`.
- Interpretation: CE implementation is not the primary trigger for the 74 GiB first-step allocation. With `model_axis=1`, the current `lm_head` output-projection spec shards only over a size-1 `model` axis, so the explicit replicated output-projection diagnostic is not meaningfully different for lm-head memory at this topology. The next useful axis is allocation scaling: reduce global batch from 256 to 128 on the same N16/EP8 topology. If the request drops substantially or the run fits, the buffer is batch/activation-like; if it remains near 74 GiB, suspect optimizer/update or a parameter/program-layout buffer.
- Next action: launch MAY-028 at batch 128 with the same N16/EP8/full-depth settings, XLA CE, `compute_with_master`, and replicated input/output embeddings.

### 2026-06-14 21:38 PDT - MAY-028 batch-128 allocation-scaling probe launched
- Hypothesis: if the recurring 74 GiB first-step allocation is batch/activation-like, halving global batch from 256 to 128 on the same N16/EP8/full-depth topology should substantially reduce the allocation and may allow the first train step to complete. If the request remains near 74 GiB, suspect optimizer/update, parameter layout, or a compiler program buffer that does not scale primarily with batch.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-028B128-N16-cw-20260615-0440 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --batch 128 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-028B128-N16-cw-20260615-0440 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --batch 128 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --submit`
  - Initial poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-043736`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-043736`
  - Expected child: `/dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440`
  - Source checkpoint: `303788a88`
  - Same as MAY-026 except `MAY_BATCH=128`: full May d2560/L26, seq_len 4096, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: submit created parent `/dlwh/iris-run-job-20260615-043736`. Initial poll found only the parent, `JOB_STATE_RUNNING`, `task_count=1`, `failure_count=0`, `preemption_count=0`; no child logs yet.
- Interpretation: the launch is a clean one-axis batch-scaling probe, not a throughput claim yet.
- Next action: babysit for child creation, hparams confirmation (`train_batch_size=128`, `batch_shards=128`), first train step/MFU, or the same allocator failure with its requested size.

### 2026-06-14 21:55 PDT - MAY-028 passed prior first-step OOM window
- Hypothesis: if the recurring 74 GiB first-step allocation is batch/activation-like, halving global batch from 256 to 128 on the same N16/EP8/full-depth topology should either reduce the allocation or allow the first train step to complete.
- Command:
  - Status poll: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-043736`
  - Child summary: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440`
  - Log scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440 --since-seconds 1200 --max-lines 800000 | rg -v '"event": "hparams"|xla_bridge Unable to initialize backend|pip install jax\\[k8s' | rg -i -m 800 -C 2 '"event": "log"|"event": "train"|train/loss|throughput/mfu|throughput/|step_time|examples_per_second|tokens_per_second|Progress on:train|RESOURCE_EXHAUSTED|Out of memory|Fatal error|Traceback|JaxRuntimeError|Compilation|Finished XLA compilation|slow_operation|Program hbm|Largest program'`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-043736`
  - Child: `/dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440`
  - Source checkpoint: `303788a88`
  - Full May d2560/L26, seq_len 4096, batch 128, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: the child is running with `task_count=16`, `task_state_counts={"running": 16}`, `failure_count=0`, and `preemption_count=0` after about 17 minutes of task runtime. Logs confirmed `batch_shards=128`, `parameter_count=67078882816`, `throughput/flops_per_token_analytic=5706711040.0`, `throughput/flops_per_example_analytic=70124065259520.0`, `throughput/theoretical_flops=1.26656e+17`, and `Fused cross-entropy selected implementation: xla`. Unlike the B256 gates, no `RESOURCE_EXHAUSTED`, BFC allocation request, `Program hbm requirement`, `fake_parameter.2`, involuntary rematerialization, atomic, or clique signature appeared. Around 04:50:52 UTC, all ranks emitted step-0 `throughput/hook_time` and `throughput/loading_time` log rows, but no `train/loss`, MFU, full step-time metric, W&B run, or profile artifact has appeared yet.
- Interpretation: this is the first full-depth N16 May probe to pass the prior B256 first-step OOM window, which strongly supports a batch-scaling activation/workspace cause rather than optimizer state or static parameter storage. It is not a throughput result until a full train row lands. If it completes steps, the next axis should return toward batch 256 using vocab/model-axis sharding or lower local-token pressure; if it stalls after hook/loading rows, inspect the train loop around metric emission and consider a smaller/shorter run to separate step execution from logging/progress.
- Next action: keep babysitting MAY-028 for a completed `train/loss`/MFU row, terminal success, or a late OOM/error; do not stop it while it remains healthy.

### 2026-06-14 22:01 PDT - MAY-029 one-layer B256 depth-localization probe launched
- Hypothesis: if B256's 74 GiB first-step allocation is dominated by the output/loss local-token workspace, reducing the model from full L26 to one layer while keeping B256 and the same N16/EP8/model1 topology may still fail near the same allocation. If one layer fits, the full-depth OOM is more likely cumulative activation/remat/optimizer-update liveness.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-029L1B256-N16-cw-20260615-0503 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --layers 1 --batch 256 --steps 1 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-029L1B256-N16-cw-20260615-0503 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --layers 1 --batch 256 --steps 1 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --submit`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-050105`
  - Expected child: `/dlwh/iris-run-job-20260615-050105/grug-train-GM2560-MAY-029L1B256-N16-cw-20260615-0503`
  - Source checkpoint: `303788a88`
  - Same as MAY-026 except `MAY_NUM_LAYERS=1` and `MAY_STEPS=1`: seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: dry run matched the intended one-axis depth-localization probe. Submit created parent `/dlwh/iris-run-job-20260615-050105`.
- Interpretation: this should distinguish a depth-dependent full-program memory issue from a B256 local-token/output workspace issue without changing the original failing batch size.
- Next action: poll for child creation, hparams confirmation (`num_layers=1`, `train_batch_size=256`), first step, terminal success, or an allocator failure and requested size.

### 2026-06-14 22:03 PDT - MAY-028 completed at B128 without OOM but produced no MFU sample
- Hypothesis: if the recurring B256 74 GiB first-step allocation is batch/activation-like, B128 should fit on the same full-depth N16/EP8 topology; if it remains near-static, B128 should fail similarly.
- Command:
  - Final state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-043736`
  - Child summary: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job summary --json /dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440`
  - Metric extraction: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440 --since-seconds 3600 --max-lines 2000000 | rg 'grug_moe_cw_may\\.metrics \\{"tracker": "json_logger", "event": "log"' | tail -n 200`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-043736`
  - Child: `/dlwh/iris-run-job-20260615-043736/grug-train-GM2560-MAY-028B128-N16-cw-20260615-0440`
  - Source checkpoint: `303788a88`
  - Full May d2560/L26, seq_len 4096, batch 128, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: the child and parent both reached `JOB_STATE_SUCCEEDED`. The child summary reported `task_count=16`, `completed_count=16`, `failure_count=0`, `preemption_count=0`, `task_state_counts={"succeeded": 16}`, `exit_code=0` on all tasks, and duration about 1,394,808 ms. No OOM, allocator request, SPMD/clique, FA4 atomic, or fatal trace appeared. Logs confirmed a full step completed before the manual `throughput/hook_time` / `throughput/loading_time` rows at step 0 on all ranks, and at least one step-1 hook/loading row. The standard Levanter callback metrics (`train/loss`, `throughput/mfu`, `throughput/duration`, `throughput/tokens_per_second`, `throughput/examples_per_second`) did not appear in the JSON logs despite the child succeeding; this run therefore has no usable MFU sample. The observed wall-clock between first step-0 and step-1 manual hook rows was about 602 seconds, so this configuration is far below the 20 MFU target even though it fits memory.
- Interpretation: B128 fitting while B256 full-depth fails strongly supports a batch/local-token/activation workspace blocker for the B256 OOM, and rules down static optimizer-state or parameter-storage explanations. The missing standard callback metrics are a short-run logging gap: Levanter's hook runner skips non-forced hooks for `step <= 1`, so tiny JSON probes can finish without an MFU row. The next memory-localization question is whether B256 still fails at one layer; if one layer succeeds, full-depth activation/remat liveness is the likely memory source. If one layer OOMs near 74 GiB, the B256 output/local-token path is sufficient to explain the failure.
- Next action: babysit MAY-029 one-layer B256. In parallel, fix or work around the missing JSON callback metrics before the next throughput/profile run; W&B may avoid this, but JSON needs to be trustworthy for fast cluster probes.

### 2026-06-14 22:06 PDT - MAY-029 one-layer B256 fits
- Hypothesis: if B256's 74 GiB first-step allocation is dominated by the output/loss local-token workspace, reducing the model from full L26 to one layer while keeping B256 and the same N16/EP8/model1 topology may still fail near the same allocation. If one layer fits, the full-depth OOM is more likely cumulative activation/remat/optimizer-update liveness.
- Command:
  - Final state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-050105`
  - Log scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-050105 --since-seconds 900 --max-lines 500000 | rg -v '"event": "hparams"|xla_bridge Unable to initialize backend|pip install jax\\[k8s' | rg -i -m 600 -C 4 'Dispatching grug|Grug compact mesh|num_layers|train_batch_size|batch_shards|parameter_count|Fused cross-entropy|Progress on:train|"event": "log"|train/loss|throughput/mfu|RESOURCE_EXHAUSTED|Out of memory|trying to allocate|rounded to|Fatal error|Traceback|JaxRuntimeError|failed|Exception|succeeded'`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-050105`
  - Child: `/dlwh/iris-run-job-20260615-050105/grug-train-GM2560-MAY-029L1B256-N16-cw-20260615-0503`
  - Source checkpoint: `303788a88`
  - Same as MAY-026 except `MAY_NUM_LAYERS=1` and `MAY_STEPS=1`: seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: parent and child both reached `JOB_STATE_SUCCEEDED`: child `task_count=16`, `completed_count=16`, `failure_count=0`, `preemption_count=0`, `task_state_counts={"succeeded": 16}`. Logs confirmed the intended mesh (`data=16`, `expert=8`, `model=1`, `batch_shards=128`) and one-layer shape via summaries: `parameter_count=3212636416`, `throughput/flops_per_token_analytic=850903040.0`, `throughput/flops_per_example_analytic=10455896555520.0`, `throughput/theoretical_flops=1.26656e+17`. The executor reported success in `0:02:06.031074`. Shutdown produced expected `WatchTasksAsync failed ... Connection refused` noise after task completion, but Iris state and executor status were successful. There was no OOM or 74 GiB allocator request.
- Interpretation: the B256 failure is not explained by the loss/output local-token workspace alone. Full-depth L26 liveness is required to reproduce it, making activation/remat policy the next primary axis. In particular, `save_moe` preserves several MoE tensors across layers; switching to `recompute_all` on the original full-depth B256 shape should test whether saved activations are the difference between fitting and OOM.
- Next action: launch MAY-030 with full L26/B256/N16/EP8, same settings as MAY-026, changing only `--remat recompute_all`.

### 2026-06-14 22:06 PDT - MAY-030 full-depth B256 recompute-all remat probe launched
- Hypothesis: if the full-depth B256 OOM is caused by saved activation/remat liveness under `save_moe`, switching only `MAY_REMAT` to `recompute_all` should reduce HBM enough to move the original B256 shape past the first step, at the cost of more compute.
- Command:
  - Dry run: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-030RECOMP-B256-N16-cw-20260615-0508 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --batch 256 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --remat recompute_all`
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-030RECOMP-B256-N16-cw-20260615-0508 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --batch 256 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --remat recompute_all --submit`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-050534`
  - Expected child: `/dlwh/iris-run-job-20260615-050534/grug-train-GM2560-MAY-030RECOMP-B256-N16-cw-20260615-0508`
  - Source checkpoint: `303788a88`
  - Same as MAY-026 except `MAY_REMAT=recompute_all`: full May d2560/L26, seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: dry run matched the intended one-axis remat probe. Submit created parent `/dlwh/iris-run-job-20260615-050534`.
- Interpretation: this is the direct remat/liveness test for the original B256 full-depth OOM.
- Next action: babysit for child creation, hparams confirmation (`remat_mode=recompute_all`), first step, MFU if any, terminal success, or allocator failure and requested size.

### 2026-06-14 22:20 PDT - MAY-030 recompute-all still OOMs
- Hypothesis: if the full-depth B256 OOM is caused mostly by saved activation/remat liveness under `save_moe`, switching only `MAY_REMAT` to `recompute_all` should reduce HBM enough to move the original B256 shape past the first step.
- Command:
  - Final state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-050534`
  - Log scan: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job logs /dlwh/iris-run-job-20260615-050534 --since-seconds 7200 --max-lines 800000 | rg -v '"event": "hparams"|xla_bridge Unable to initialize backend|pip install jax\\[k8s' | rg -i -m 80 -C 3 'RESOURCE_EXHAUSTED|Out of memory|trying to allocate|rounded to|Fatal error|Traceback|JaxRuntimeError|failed|Exception|remat_mode|seq_len|train_batch_size'`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-050534`
  - Child: `/dlwh/iris-run-job-20260615-050534/grug-train-gm2560-may-030recomp-b256-n16-cw-20260615-0508`
  - Source checkpoint: `5353b2eea767bb05a41603a213c9c457b1e07ecb`
  - Same as MAY-026 except `MAY_REMAT=recompute_all`: full May d2560/L26, seq_len 4096, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`.
- Result: the child and parent are terminal `JOB_STATE_KILLED` after manual stop of the deterministic retry. The first attempt reproduced the first-step allocator failure at `experiments/grug/moe/train.py:580` on `jax.block_until_ready(metrics["train/loss"])`: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 72.03GiB`, with BFC lines `rounded to 77344390144` across local GPUs. Child summary via `job list` showed `task_count=16`, `completed_count=16`, `failure_count=1`, `preemption_count=0`, `task_state_counts={"killed": 16}`, and `pending_reason="Terminated by user"`.
- Interpretation: `recompute_all` helped only modestly versus the previous full-depth B256 failures (~74.4 GiB -> 72.03 GiB) and is not enough to make the target full-sequence B256 shape fit. Since B128/full-depth fits and B256/one-layer fits, the next one-axis test should scale sequence length while keeping B256/full-depth and `recompute_all` to localize whether the remaining request is token/activation sized.
- Next action: launch MAY-031 with the same full-depth B256/N16/EP8/model1 settings as MAY-030, changing only `seq_len=2048`.

### 2026-06-14 22:21 PDT - MAY-031 seq-len 2048 diagnostic launched
- Hypothesis: if the remaining 72.03 GiB first-step allocation is primarily token/activation sized, halving sequence length from 4096 to 2048 while keeping full depth, B256, N16/EP8/model1, `recompute_all`, and the same loss/embedding sharding should fit or reduce the allocator request substantially. If it OOMs near the same size, the bad path is less sequence-scaled and the next axis should be model/vocab or parameter/update liveness.
- Command:
  - Submit: `experiments/grug/moe/run_cw_may_d2560.sh --run-id GM2560-MAY-031S2048-B256-N16-cw-20260615-0520 --nodes 16 --data synthetic --checkpoints none --worker-cpu 8 --model-axis 1 --expert-axis 8 --replica-axis 1 --batch 256 --seq-len 2048 --steps 4 --profiler-steps 0 --tracker json_logger --watch-interval 0 --log-every 1 --log-jaxprs false --log-xla-hlo false --ce-implementation xla --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --live-param-mode compute_with_master --remat recompute_all --submit`
  - Initial state: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job list --json --prefix /dlwh/iris-run-job-20260615-052048`
- Config:
  - Parent: `/dlwh/iris-run-job-20260615-052048`
  - Child: `/dlwh/iris-run-job-20260615-052048/grug-train-gm2560-may-031s2048-b256-n16-cw-20260615-0520`
  - Source checkpoint: `5353b2eea767bb05a41603a213c9c457b1e07ecb`
  - Same as MAY-030 except `MAY_SEQ_LEN=2048`: full May d2560/L26, batch 256, synthetic data, checkpoints disabled, no profiler, 16 H100 nodes, `expert_axis=8`, `model_axis=1`, `replica_axis=1`, ring MoE, FA4/CuTe attention, XLA CE, replicated input/output embeddings, `live_param_mode=compute_with_master`, `remat_mode=recompute_all`.
- Result: submit created parent `/dlwh/iris-run-job-20260615-052048`. Initial Iris poll found parent and child both `JOB_STATE_RUNNING`; the child had `task_count=16`, `completed_count=0`, `failure_count=0`, `preemption_count=0`, and `task_state_counts={"running": 16}`.
- Interpretation: the one-axis sequence-length diagnostic is live and not yet at a decision point.
- Next action: monitor for hparams confirmation (`seq_len=2048`, `train_batch_size=256`, `num_layers=26`, `remat_mode=recompute_all`), first train step/MFU, terminal success, or an allocator failure and requested size. Subagent `Mill` (`019ec9ba-17b9-7bf3-8cda-68fbd574152a`) owns read-only babysitting, and heartbeat `poll-grug-d2560-gm006-babysitter` now points at MAY-031.
