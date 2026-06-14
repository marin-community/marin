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
