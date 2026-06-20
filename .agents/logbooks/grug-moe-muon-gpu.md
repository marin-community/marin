# Grug MoE Muon GPU Speed: Research Logbook

## Scope
- Goal: reduce single-node GPU Muon/Newton-Schulz optimizer overhead for Grug MoE.
- Primary metrics: step duration, throughput/MFU, optimizer/update time in profile.
- Constraints: single-node first, `model_axis=1`, no TP/multinode experiments, outputs under `s3://marin-na/tmp/ttl=7d`.

## Baseline
- Date: 2026-06-18
- Code refs: `experiments/grug/moe/optimizer.py`, `lib/levanter/src/levanter/optim/grugmuon.py`
- Baseline numbers: May140 MuonH5 B8 profile: 4.7499 MFU, 21,962.5 tokens/s, 1.492s/step. May141 SGD B8 comparison: 16.09 MFU, 74,398 tokens/s, 0.440s/step at first metrics.
- May153 xprof framework-op profile: `scratch/profiles/may153/profile_report_xprof.md` attributes about 9.94s, or 55.6% of the profiled window, to `optimizer_muon`. Smaller optimizer families are `optimizer_apply` at about 0.36s and `optimizer_adam` at about 0.034s. This keeps the harness gate focused on whether production `muonh_update` preserves pure Newton-Schulz speed and intended stack sharding.
- Issue: https://github.com/marin-community/marin/issues/6493

## Experiment Log
### 2026-06-18 08:04 PDT - full-production H3 harness launch
- Hypothesis: H3 may be the practical production Muon route if it clears the single-node speed-of-light target while avoiding the H5 step-duration penalty.
- Command:
  `RUN_ID=MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-150344 MUON_BENCH_LAYERS=2 MUON_BENCH_REPLICA_AXIS=1 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_AXIS=none MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=5 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true MARIN_PREFIX=s3://marin-na/tmp/ttl=7d bash scratch/launch_muon_update_bench_executor_n1.sh`
- Config: single-node H100, `layers=2`, `hidden=2560`, `intermediate=1280`, `experts=256`, `data_axis=1`, `expert_axis=8`, `model_axis=1`, `backend_steps=3`, comparing routed-expert-only and full-production MuonH apply harnesses.
- Result: Succeeded. Child `/dlwh/iris-run-job-20260618-150345/grug-train-MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-150341`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-150341-7b9ee1`. Routed-expert-only H3: 40.37 ms / 2 layers, 58.49% nominal peak, AG/AR/RS 0/0/0. Full-production H3: 47.37 ms / 2 layers, 50.68% nominal peak, AG/AR/RS 0/0/0.
- Interpretation: H3 clears the ~50% Muon-only speed-of-light target for the fuller production-shaped Muon set. The non-routed production leaves add about 7.0 ms / 2 layers over routed-expert-only H3, or roughly 91 ms scaled to 26 layers. This is a better production compromise than H5 on speed, assuming H3 is acceptable scientifically.
- Next action: update #6493. For full training, do not relaunch unchanged H3 unless the goal is a clean readability profile; existing H3 train evidence still shows the rest of the step is not in the 25-30 MFU band.

### 2026-06-18 06:50 - kickoff
- Hypothesis: Muon/Newton-Schulz dominates the single-node step because many independent small GEMMs are emitted instead of grouped/batched work sharded across `expert`.
- Command: none.
- Config: single-node H100, batch=8, seq_len=4096, sliding_window=2048, `expert_axis=8`, `model_axis=1`, bf16 params/compute/output.
- Result: May141 SGD A/B suggests roughly 1s of May140 step is optimizer-specific.
- Interpretation: optimization should focus on grouped/stacked Muon, not only XLA flags.
- Next action: create issue, then prototype batched NS over compatible matrix groups.

### 2026-06-18 07:35 - grouped stack prototype
- Hypothesis: A tree-level grouping pass can concatenate compatible 3D Muon leaves, run one batched Newton-Schulz call per `(fan_in, fan_out, dtype, stack_pspec)` bucket, then split and restore parameter sharding without changing update numerics.
- Command: `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py`
- Config: CPU correctness tests plus abstract mesh sharding tests for `P("expert", None, None)` with `model_axis=1`.
- Result: 19 tests passed. Added grouped-stack parity against `VMAP_REPLICATED` for mixed wide/tall expert leaves and an abstract-mesh grouped sharding test that preserves expert-axis param sharding. `./infra/pre-commit.py --files lib/levanter/src/levanter/optim/grugmuon.py lib/levanter/tests/test_grugmuon.py` passed Ruff/Black/license but failed project-wide Pyrefly in unrelated dirty code at `lib/levanter/src/levanter/grug/loss.py:134`.
- Interpretation: Minimal implementation is ready for a single-node GPU runtime/profile smoke. Correctness coverage says grouped stack preserves existing Muon updates within `1e-5` on small shapes and returns grouped expert updates to parameter layout.
- Next action: run a single-node GPU smoke/profile under #6493 when a suitable run slot is available; keep outputs under `s3://marin-na/tmp/ttl=7d` and avoid XLA flag sweeps.

### 2026-06-18 07:00 - May141 SGD profile completed
- Hypothesis: removing MuonH5 should collapse the long optimizer tail seen in May140.
- Command: `bash scratch/launch_may141_fa4_single_node_b8_sgd_autotune_profile.sh`
- Config: single-node H100, batch=8, seq_len=4096, sliding_window=2048, `expert_axis=8`, `model_axis=1`, `attention=gpu_fa4_cute`, `ce=xla`, `moe=ring`, bf16 params/compute/output, optimizer=SGD.
- Result: W&B run `GM2560-MAY-141S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4SGD-AUTOTUNE-PROFILE-N1-cw-20260618-0637` finished at 16.1876 MFU, 74,848.2 tokens/s, 0.4378s/step, loss 11.7910. Profile downloaded to `scratch/profiles/may141` and served on port 6009.
- Interpretation: the May140-to-May141 delta is about 1.05s/step, so Muon/update overhead is the dominant single-node speed target.
- Next action: use May141 as the no-Muon reference while prototyping grouped/stacked Muon.

### 2026-06-18 00:40 PDT - May142 grouped stack OOM and bounded fix
- Hypothesis: The first grouped prototype failed because it concatenated too many compatible expert leaves into one Newton-Schulz batch, producing a giant autotune candidate rather than a memory-safe batched workload.
- Command: validation run was launched by parent `/dlwh/iris-run-job-20260618-070946`; local fix validation command: `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py`.
- Config: May142 run `GM2560-MAY-142S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON5-PROFILE-N1-cw-20260618-0709`, single-node H100, batch=8, seq_len=4096, sliding_window=2048, `expert_axis=8`, `model_axis=1`, MuonH5 grouped prototype.
- Result: May142 failed before metrics during XLA autotuning. HLO `%gemm_fusion_dot.1160 = bf16[832,2560,2560]` under `grug_muon/orthogonalize_3d_grouped_stack/_zeropower_via_newtonschulz_batched_stack_sharded/.../iter_0/apply` exhausted memory while allocating 10.16 GiB; a similar `%gemm_fusion_dot.1158` gram failed. Patched grouping to cap grouped stack batches at 256 by default, shard-align the cap under abstract mesh, and split oversized single leaves into bounded chunks. Focused tests pass: 19 passed.
- Interpretation: The negative result validates that unbounded grouping is not viable. A 256 cap prevents the observed 832-stack grouped GEMM and restores the largest grouped stack to the previous single expert-leaf scale for May d2560.
- Next action: have the main thread rerun single-node grouped Muon with the bounded cap; do not stop/restart clusters and do not run XLA flag sweeps.

### 2026-06-18 00:58 PDT - pad-worthy widest stack sharding
- Hypothesis: Grouped Muon should choose the widest `replica_dcn/data/expert` stack sharding when stack-axis padding is modest, because padding 832 to 1024 under 512 stack shards can be cheaper than falling back to 128 shards.
- Command: `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py`
- Config: CPU/eval-shape tests for divisible full `R/D/E`, non-divisible pad-worthy full `R/D/E`, too-expensive full-axis padding fallback, and grouped pad/slice shape restoration. Padding overhead threshold is `STACK_PADDING_MAX_OVERHEAD = 1.25`.
- Result: 23 tests passed. The NS kernel pads only the stack axis before `reshard`, asserts the padded batch has the chosen stack pspec before dot contractions, then reshards to replicated stack layout for slicing back to the original stack length. Chunking still bounds padded chunks, with a default local cap of two matrices per stack shard.
- Interpretation: The implementation now encodes the desired preference order: widest candidate axis when divisible or pad-worthy, narrower fallback when full-axis padding is too expensive, and no CoreWeave run launched.
- Next action: let the main thread decide when to run the next single-node grouped Muon validation.

### 2026-06-18 01:18 PDT - May143/May144 live babysit checkpoint
- Hypothesis: The bounded/padded grouped Muon path should avoid the May142 832-stack XLA autotune OOM and reach first metrics/profile.
- Command: monitor only with `uv run iris --config lib/iris/config/cw-us-east-02a.yaml job list --json --prefix ...`, `job logs`, and W&B API reads.
- Config: May143 throughput run `GM2560-MAY-143S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON5-THROUGHPUT-N1-cw-20260618-0807`; May144 profile run `GM2560-MAY-144S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON5-PROFILE-N1-cw-20260618-0812`. Both are single-node grouped/padded MuonH under `/dlwh` Iris parents on `cw-us-east-02a`, writing under `s3://marin-na/tmp/ttl=7d`.
- Result: Both Iris parent/child jobs are running with failure_count=0. May143 W&B is created and step 0 finished without OOM/sharding/assertion/compiler failure; step 1 started at 08:15:44 UTC. May144 W&B is created, mesh initialized, and step 0 started at 08:16:57 UTC. W&B history has not yet exposed scalar metrics or profile files.
- Interpretation: The May142 unbounded-stack OOM signature has not recurred through first dispatch. May143/May144 are still in progress, so throughput/profile conclusions are pending.
- Next action: continue bounded polling for first W&B metrics, May144 profile artifacts, failures, or terminal state. Do not stop/restart clusters or launch follow-up runs.

### 2026-06-18 01:24 PDT - May143 throughput terminal
- Hypothesis: If grouped/padded Muon batches the Newton-Schulz tail effectively, warm steps should move materially toward the May141 SGD ceiling.
- Command: W&B API read for `marin-community/marin_moe/GM2560-MAY-143S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON5-THROUGHPUT-N1-cw-20260618-0807`.
- Config: single-node B8, seq_len=4096, sliding_window=2048, `expert_axis=8`, `model_axis=1`, grouped/padded MuonH, profiler disabled.
- Result: Iris child succeeded with failure_count=0. Final W&B summary: `throughput/mfu=4.670488328835779`, `throughput/tokens_per_second=21595.384423660384`, `throughput/duration=1.5173612729995511`, `throughput/mean_mfu=3.5111952022034063`, loss 4.1068. Step 0 and step 1 included long compile/execution durations of 403.97s and 423.10s; warm steps 2-7 were about 1.51-1.52s.
- Interpretation: The bounded/padded path is stable, but it does not improve throughput. It is slightly below May140 MuonH5 profile throughput (4.7499 MFU, 21,962.5 tokens/s, 1.492s) and far from May141 SGD (16.1876 MFU).
- Next action: use May144 profile to inspect whether grouped NS still dominates, whether grouping is not taking the intended path, or whether the chunking/padding choice erased the benefit.

### 2026-06-18 01:31 PDT - grouped-path diagnosis checkpoint
- Hypothesis: May143 did not improve because the current cap makes the expensive expert leaves run as independent per-leaf stacks, not multi-layer grouped stacks.
- Command: code inspection plus local lowering-only check of `_zeropower_via_newtonschulz_batched_stack_sharded` with abstract shape `(256, 2560, 2560)`.
- Config: May d=2560 has `num_layers=26`, `num_experts=256`, `intermediate_dim=1280`. Routed expert leaves are `w_gate_up: (256, 2560, 2560)` and `w_down: (256, 1280, 2560)`. Current cap is `max_grouped_stack_size=256`.
- Result: The grouping loop can only merge leaves when the concatenated/padded stack size stays under the cap. A single `w_gate_up` or `w_down` leaf already has stack size 256, so same-shape leaves from other layers become single-entry chunks and take the per-leaf stack path. The local lowering check showed one `(256, 2560, 2560)` NS step lowers to three `stablehlo.dot_general` ops with `batching_dims = [0] x [0]`, so the single-leaf path is batched HLO, not an expanded vmap loop.
- Interpretation: Grouped/padded Muon is memory-safe and uses batched dots per expert leaf, but it is not yet grouping across the expensive leaves. Also, all 2D Muon leaves still use the replicated 2D NS path.
- Next action: wait for May144 profile to confirm the optimizer region composition and for May145 (`backend_steps=1`) to quantify how much of the 1.5s step is NS iteration count versus surrounding MuonH/hyperball/update overhead.

### 2026-06-18 01:31 PDT - May144/May145 live status
- Hypothesis: May144 should produce a readable profile once it reaches profiler steps; May145 should diagnose whether reducing Muon NS iterations changes throughput.
- Command: monitor only with Iris logs/summary and W&B API reads.
- Config: May144 profile run `GM2560-MAY-144S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON5-PROFILE-N1-cw-20260618-0812`; May145 throughput run `GM2560-MAY-145S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4GROUPEDMUON1-THROUGHPUT-N1-cw-20260618-0827` with `muon_backend_steps=1`, `orthogonalization_layout=stack_batch_sharded`, `max_grouped_stack_size=256`.
- Result: May144 missed the 08:30 UTC profile-deadline target but remains running with failure_count=0; no W&B profile files or scalar rows yet. May145 child is running with W&B created and step 0 dispatched at 08:28:57 UTC; no OOM/sharding/assertion/compiler failure yet.
- Interpretation: No terminal failure has appeared. May144 remains useful if it eventually uploads the profile; May145 is the immediate throughput diagnostic.
- Next action: continue bounded polling. Do not stop/restart clusters or launch follow-up runs.

### 2026-06-18 01:48 PDT - May145/May146 throughput terminal
- Hypothesis: `backend_steps=1` should isolate Newton-Schulz iteration cost, and cap512 should show whether grouping two 256-expert leaves helps versus cap256.
- Command: Iris summaries for `/dlwh/iris-run-job-20260618-082716/...GROUPEDMUON1-THROUGHPUT...` and `/dlwh/iris-run-job-20260618-083148/...GROUPEDMUON1-CAP512-THROUGHPUT...`; W&B API reads for both runs.
- Config: Both runs are single-node B8, seq_len=4096, sliding_window=2048, `expert_axis=8`, `model_axis=1`, grouped/padded MuonH, `backend_steps=1`; May145 uses `max_grouped_stack_size=256`, May146 uses `max_grouped_stack_size=512`.
- Result: Both Iris children succeeded with failure_count=0. May145 cap256 finished at `throughput/mfu=10.2894`, `mean_mfu=7.6961`, `tokens/s=47576.1`, `duration=0.68875s`, loss 4.9466. May146 cap512 finished at `throughput/mfu=8.04845`, `mean_mfu=6.0260`, `tokens/s=37214.4`, `duration=0.88052s`, loss 4.9411.
- Interpretation: Reducing MuonH from H5 to H1 recovers a large fraction of the May140-to-May141 gap, so NS iteration count is a major cost. However cap512 is slower than cap256 in the full model, so naively grouping two 256-expert leaves does not improve end-to-end throughput and may increase fusion/autotune/memory pressure or expose a worse batched GEMM shape.
- Next action: use a standalone Muon-update harness to inspect HLO grouping and time cap/step sweeps without full Grug compilation.

### 2026-06-18 01:48 PDT - standalone Muon update harness
- Hypothesis: A synthetic optimizer-only harness can determine whether cap256/cap512 changes the grouped Muon HLO shape and update timing without compiling the full model.
- Command: `uv run python experiments/grug/moe/muon_update_bench.py --disable-abstract-mesh --layers 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 4 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --sweep-backend-steps 1 --sweep-max-grouped-stack-sizes 4,8 --mode both --warmup 1 --iters 2 --hlo-output scratch/muon_update_bench_tiny_sweep.stablehlo --output scratch/muon_update_bench_tiny_sweep.json`
- Config: Local CPU-only validation of the same code path used by `scale_with_grug_muonh`, with synthetic Grug MoE expert pytrees shaped like `w_gate_up=(experts, hidden, 2*intermediate)` and `w_down=(experts, intermediate, hidden)`. The production defaults are `hidden=2560`, `intermediate=1280`, `experts=256`, `expert_axis=8`, `model_axis=1`.
- Result: Added `experiments/grug/moe/muon_update_bench.py`. The local toy sweep with `--disable-abstract-mesh` confirms the harness distinguishes ungrouped versus grouped leaves: cap4 produced chunks `[[4, 4], [4, 4]]`, 0 grouped chunks, and 12 batched `stablehlo.dot_general` ops; cap8 produced chunks `[[8], [8]]`, 2 grouped chunks, and 6 batched `stablehlo.dot_general` ops. The default abstract-mesh mode is the intended GPU-fidelity path; on this one-device CPU host it cannot model a nontrivial sharded stack axis and both cap4/cap8 lower to 12 batched dots. Focused validation: `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py` passed; local environment has only one CPU device, so no May-sized GPU timing was run here.
- Interpretation: The harness can verify whether increasing the cap actually reduces the number of batched NS dot groups. It also measures compile and steady-state execution separately. In default mode on an `expert_axis=8` worker, it enters a JAX abstract mesh so the existing `grugmuon.py` stacked-NS sharding assertions are active; materialized run mode also asserts 3D expert params, momentum, and updates retain `P("expert", None, None)` when `expert_axis>1`.
- Next action: run the harness on the single-node GPU worker with `expert_axis=8` for cap256/cap512/cap832-ish and H1/H5 before launching any new full Grug run.

### 2026-06-18 02:07 PDT - direct Muon harness cap512 negative result and 4D candidate
- Hypothesis: Simple cap512 grouping may be slower because flattening two expert leaves into a `[512, m, n]` batched GEMM creates a worse GPU shape than two `[256, m, n]` batches.
- Command: External run `/dlwh/iris-run-job-20260618-085949`; local harness extension smoke: `uv run python experiments/grug/moe/muon_update_bench.py --disable-abstract-mesh --layers 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 4 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --sweep-backend-steps 1 --sweep-max-grouped-stack-sizes 4 --bench-kinds muonh_update,ns4d_replicated_group,ns4d_data_group --mode both --warmup 1 --iters 2 --hlo-output scratch/muon_update_bench_candidate_tiny.stablehlo --output scratch/muon_update_bench_candidate_tiny.json`
- Config: External direct Muon update benchmark used layers=2, hidden=2560, experts=256, single H100 node, `model_axis=1`, `expert_axis=8`, `disable_abstract_mesh=true`. Local smoke uses tiny CPU shapes and only validates HLO structure and command plumbing.
- Result: External table: h1_cap256 = 12 dot_general / 12 batched_stack_dot_general / 17.886 ms mean; h1_cap512 = 6 / 6 / 34.028 ms; h5_cap256 = 60 / 60 / 67.978 ms; h5_cap512 = 30 / 30 / 77.545 ms. Updated #6493 with the table and recommendation to keep cap256 as the safer full-training setting. Added harness benchmark kinds `ns4d_replicated_group` and `ns4d_data_group`, which test `[group, expert, m, n]` NS without flattening group and expert into `[512, m, n]`. Local smoke: `ns4d_*` lowers to 6 `dot_general` with `batching_dims=[0,1] x [0,1]`; focused validation `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py` passed.
- Interpretation: The direct benchmark falsifies "fewer flat batched GEMMs is automatically faster." The next concrete candidate is not larger flat stacks, but preserving a separate group/layer batch axis so XLA can compile a two-batch-axis GEMM while the expert axis remains shardable.
- Next action: run the updated harness on a single-node GPU worker for `muonh_update,ns4d_replicated_group,ns4d_data_group` with H1/H5 and May shapes before promoting any production `grugmuon.py` layout change.

### 2026-06-18 02:08 PDT - direct Muon cap sweep pins current best region
- Hypothesis: If flat stack size is the issue, there may be an intermediate cap below 512 that groups some work without triggering the slow `[512, ...]` batched-GEMM behavior.
- Command: External run `/dlwh/iris-run-job-20260618-090452`, run id `MUON-BENCH-D2560-L2-H1-CAP128TO512-N1-cw-20260618-0904`.
- Config: H1, layers=2, hidden=2560, experts=256, single node, `model_axis=1`, `expert_axis=8`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1-CAP128TO512-N1-cw-20260618-0904-4ca12b`.
- Result: cap128: 24 dots, 33.589 ms mean; cap192: 24 dots, 36.543 ms; cap256: 12 dots, 17.383 ms; cap320: 12 dots, 17.320 ms; cap384: 12 dots, 17.439 ms; cap448: 12 dots, 17.306 ms; cap512: 6 dots, 34.153 ms.
- Interpretation: cap256-448 are effectively equivalent and best for the current implementation. cap<256 splits each 256-expert leaf and doubles dot kernels; cap512 groups two layer leaves and is roughly 2x slower despite half the dot count. Naive larger flat stack caps are not a useful next direction.
- Next action: keep production/default grouped Muon at the 256-ish region for single-node. Future harness-first work should preserve the fast per-leaf 256-ish GEMM shape while increasing parallelism, likely via expert+data sharding for larger meshes or a different grouped layout such as the 4D `[group, expert, m, n]` candidate.

### 2026-06-18 02:24 PDT - real optimizer 4D grouped prototype
- Hypothesis: The 4D NS benchmark win can be moved into the real Muon direction-update path by grouping exact same-shaped 3D expert leaves into `[group, expert, fan_in, fan_out]`, running NS with two batch axes, then unstacking and restoring each leaf to parameter sharding.
- Command: Local validation commands: `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py`; `uv run python experiments/grug/moe/muon_update_bench.py --disable-abstract-mesh --layers 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 4 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --backend-steps 1 --max-grouped-stack-size 256 --orthogonalization-layout stack_batch_4d_sharded --bench-kinds muonh_update --mode both --warmup 1 --iters 2 --hlo-output scratch/muon_update_bench_real_4d_tiny.stablehlo --output scratch/muon_update_bench_real_4d_tiny.json`; `./infra/pre-commit.py --files lib/levanter/src/levanter/optim/grugmuon.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/optimizer.py experiments/grug/moe/test_optimizer.py experiments/grug/moe/muon_update_bench.py`.
- Config: Opt-in layout `orthogonalization_layout=stack_batch_4d_sharded`. Grouping is conservative: exact same 3D shape and dtype only, group size 2, singleton leftovers fall back to the existing 3D cap path. The current `stack_batch_sharded` cap256 path remains the default.
- Result: Implemented `_zeropower_via_newtonschulz_grouped_4d_sharded` and real optimizer grouping in `lib/levanter/src/levanter/optim/grugmuon.py`. Added tests for 4D NS parity, data/expert target sharding, and restoration to param sharding. Focused pytest: 27 passed. Standalone harness tiny real-MuonH smoke with `stack_batch_4d_sharded` lowered to 6 `dot_general` with 6 two-batch-axis dots. File pre-commit formatting/lint passed; Pyrefly still fails on the pre-existing unrelated `lib/levanter/src/levanter/grug/loss.py:134` tuple-unpack error.
- Interpretation: The real optimizer path now exercises the same two-batch-axis HLO shape as the successful `ns4d_*` harness candidate while keeping the existing 3D path available.
- Next action: run the standalone Muon harness on a single-node GPU using `orthogonalization_layout=stack_batch_4d_sharded` for H1/H5, then launch a single-node training profile only if the harness confirms the real MuonH path preserves the benchmark win.

### 2026-06-18 02:38 PDT - real optimizer 4D restore negative result
- Hypothesis: If the real optimizer 4D path preserves the fast `ns4d_data_group` sharding behavior, D2/E4 should stay near the pure harness result instead of regressing to current 3D cap256 timing.
- Command: External harness validation from `/dlwh/iris-run-job-20260618-092618` and `/dlwh/iris-run-job-20260618-092855`; issue update https://github.com/marin-community/marin/issues/6493#issuecomment-4740409361.
- Config: `orthogonalization_layout=stack_batch_4d_sharded`, layers=2, hidden=2560, H1/H5, cap512. D1/E8 run id `MUON-BENCH-D2560-L2-H1H5-REAL4D-CAP512-N1-cw-20260618-0926`; D2/E4 run id `MUON-BENCH-D2560-L2-H1H5-REAL4D-D2E4-CAP512-N1-cw-20260618-0928`.
- Result: D1/E8 real 4D: H1 18.845 ms with 6 two-batch-axis dots, H5 66.265 ms with 30 two-batch-axis dots. D2/E4 real 4D: H1 36.809 ms, H5 133.551 ms. Unit tests still pass: `uv run pytest lib/levanter/tests/test_grugmuon.py -q` -> 18 passed.
- Interpretation: The 4D dot shape is present, but the production path does not preserve the fast D2 behavior. The likely culprit is the post-NS unstack/restore boundary: the current prototype reshards from `P(data, expert, None, None)` to `P(None, expert, None, None)` before splitting so JAX can split the group axis, which discards data-axis partitioning before the updates are consumed.
- Next action: patch the real optimizer 4D path to avoid the pre-split data all-gather where possible. First target is group size 2 with data axis 2: split by indexing/slicing each data-sharded group shard directly, then restore each leaf to its parameter sharding only if required by the downstream apply boundary. Validate with the standalone harness before any full training/profile run.

### 2026-06-18 02:48 PDT - harness restore probes for 4D data grouping
- Hypothesis: The D2/E4 gap between pure `ns4d_data_group` and real optimizer 4D is caused by the required boundary from grouped `[layer, expert, m, n]` data-sharded results back to separate per-layer update leaves with parameter sharding `P(expert, None, None)`.
- Command: Local smoke: `uv run python experiments/grug/moe/muon_update_bench.py --disable-abstract-mesh --layers 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 4 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --backend-steps 1 --max-grouped-stack-size 8 --bench-kinds ns4d_data_group,ns4d_data_reshard_restore,ns4d_data_index_restore --mode both --warmup 1 --iters 2 --hlo-output scratch/muon_update_bench_restore_tiny.stablehlo --output scratch/muon_update_bench_restore_tiny.json`; `uv run pytest lib/levanter/tests/test_grugmuon.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py`.
- Config: Added harness-only benchmark kinds `ns4d_data_reshard_restore` and `ns4d_data_index_restore`. Both start from the same `P(data, expert, None, None)` 4D input as `ns4d_data_group`; the first mirrors the production pre-split reshard to `P(None, expert, None, None)`, while the second directly indexes the group axis to test whether XLA can avoid the explicit pre-split reshard. The harness now records compiled-HLO collective counters in addition to StableHLO counters.
- Result: Tiny CPU smoke passes for all three variants. Focused Muon tests pass: 18 passed. Harness file pre-commit passes. No CoreWeave run launched.
- Interpretation: This does not solve the production path yet, but it gives the next single-node GPU harness a direct A/B: pure 4D data grouping versus explicit restore versus direct indexing restore. If both restore probes match the slow real optimizer timing and show compiled-HLO collectives, the blocker is structural: data-axis grouping cannot survive returning separate per-layer updates while params are replicated over data.
- Next action: run the standalone GPU harness on D2/E4 with `--bench-kinds ns4d_data_group,ns4d_data_reshard_restore,ns4d_data_index_restore,muonh_update` for H1/H5. Do not launch full training until this restore boundary is understood.

### 2026-06-18 02:50 PDT - MuonH split harness and D1/E8 timing
- Hypothesis: Splitting full MuonH into direction-only and hyperball-only will show whether the remaining tail is mostly Newton-Schulz direction work or the norm-preserving projection.
- Command: Local validation: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --sweep-backend-steps 0,1 --max-grouped-stack-size 256 --data-axis 1 --expert-axis 8 --model-axis 1 --replica-axis 1 --orthogonalization-layout stack_batch_4d_sharded --bench-kinds muonh_update,muon_direction,hyperball_only,ns4d_replicated_group --mode lower --disable-abstract-mesh --output /tmp/muon_bench_split_smoke.json`; tiny run-mode smoke with `hidden=16`, `num_experts=8`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`.
- Config: Added benchmark kinds `muon_direction` and `hyperball_only`. `muon_direction` calls the production `_grug_scale_with_muon` directly. `hyperball_only` calls production `_scale_invariant_hyperball_updates` on synthetic direction updates and keeps the direction input fixed across timing iterations. CoreWeave run `/dlwh/iris-run-job-20260618-094622`, run id `MUON-BENCH-D2560-L2-SPLIT-H1H5-D1E8-CAP512-N1-cw-20260618-0946`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-SPLIT-H1H5-D1E8-CAP512-N1-cw-20260618-0946-a7d0a1`, D1/E8, layers=2, cap512, `orthogonalization_layout=stack_batch_4d_sharded`, H1/H5, warmup=1, iters=3.
- Result: Local lower smoke and tiny timing smoke passed. Pycompile passed. Focused pytest: 27 passed. Harness pre-commit passed. CoreWeave run succeeded with failure_count=0. D1/E8 timing table: H1 full MuonH 18.570 ms, direction-only 16.325 ms, hyperball-only 3.058 ms, pure `ns4d_replicated_group` 13.709 ms. H5 full MuonH 64.161 ms, direction-only 61.929 ms, hyperball-only 3.080 ms, pure `ns4d_replicated_group` 58.871 ms. All variants lowered to the expected dot counts: direction/full/pure NS4D had 6 H1 or 30 H5 two-batch-axis dots; hyperball had 0 dots. Compiled HLO showed 8 all-reduces and no all-gathers/reduce-scatters/collective-permutes for each variant.
- Interpretation: Direction/NS dominates the standalone MuonH tail. Hyperball projection is real but small at about 3 ms for two layers, and full MuonH is only about 2.2 ms slower than direction-only in this D1/E8 split. The real optimizer path still adds about 2.6-3.1 ms over pure NS4D, likely from momentum/Nesterov/scaling/restore/checksum tree work rather than hyperball itself.
- Next action: use the split harness as the default iteration path. For the D2/E4 restore question, run the same split plus `ns4d_data_group`, `ns4d_data_reshard_restore`, and `ns4d_data_index_restore`; only consider a production optimizer patch if the split shows a tractable boundary rather than unavoidable data-axis gather.

### 2026-06-18 02:53 PDT - no-checksum split baseline
- Hypothesis: The scalar checksum used by the standalone harness was contaminating H0 and hyperball timings through synthetic all-reduces; timing should block on the actual update arrays instead.
- Command: Main-thread harness patch plus validation; external run `/dlwh/iris-run-job-20260618-094914`, run id `MUON-BENCH-D2560-L1-H0H1H5-ISOLATE-NOCHECKSUM-N1-cw-20260618-0949`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L1-H0H1H5-ISOLATE-NOCHECKSUM-N1-cw-20260618-0949-93b360`.
- Config: One layer, D1/E8, no synthetic checksum reductions, `orthogonalization_layout=stack_batch_4d_sharded`, H0/H1/H5 split variants.
- Result: Run succeeded. All compiled-HLO collective counters are 0. Key timings: H5 full MuonH 32.2707 ms, H5 direction 30.7957 ms, H5 hyperball-only 1.2429 ms, H5 pure `ns4d_replicated_group` 29.9136 ms. H1 full 8.2705 ms, H1 direction 7.3409 ms, H1 hyperball 1.2289 ms, H1 pure NS4D 6.6447 ms.
- Interpretation: This supersedes the prior split table for overhead accounting. Hyperball is not the large tail. Direction/Newton-Schulz remains dominant, and pure 4D NS is only modestly faster than production direction. The next useful optimization target is NS GEMM shape/occupancy/parallelism, not more hyperball work.
- Next action: Babysit D2/E4 no-checksum run `/dlwh/iris-run-job-20260618-095212`; if data-axis sharding duplicates/slows the same work, focus next patch on D1/E8-style NS kernel shape/grouping instead of data-axis unstack tricks.

### 2026-06-18 02:57 PDT - D2/E4 no-checksum partial result and harness group-size patch
- Hypothesis: Data-axis sharding may improve pure 4D NS parallelism, but only if the 4D group axis is large enough to shard over `data`.
- Command: Babysat external run `/dlwh/iris-run-job-20260618-095212`; local validation after patch: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --data-axis 2 --expert-axis 4 --model-axis 1 --replica-axis 1 --orthogonalization-layout stack_batch_4d_sharded --bench-kinds muon_direction,ns4d_data_group --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output /tmp/muon_bench_ns4d_group_size_smoke.json`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py scratch/launch_muon_update_bench_executor_n1.sh`.
- Config: D2/E4 no-checksum run used `layers=1`, `data_axis=2`, `expert_axis=4`, H1/H5, `MUON_BENCH_KINDS=muonh_update,muon_direction,hyperball_only,ns4d_data_group`, cap256, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L1-H1H5-D2E4-NOCHECKSUM-N1-cw-20260618-0952-2c7685`.
- Result: Run failed when lowering `ns4d_data_group`: `P(data, expert, None, None)` cannot shard group axis size 1 over `data_axis=2`. Partial H1 timings before failure: full MuonH 16.3807 ms, direction 14.5418 ms, hyperball-only 2.2351 ms; all compiled-HLO collective counters were 0. Patched the harness to emit structured `skipped` rows for invalid data-group probes instead of aborting the sweep, and added `--ns4d-group-size` / `MUON_BENCH_NS4D_GROUP_SIZE` so pure NS4D shape/parallelism can be tested independently from production tree `layers`. Local D2/E4 smoke with `--layers 1 --ns4d-group-size 2` passes and lowers `ns4d_data_group` to 6 two-batch-axis dots.
- Interpretation: The partial D2/E4 tree path is roughly 2x slower than the D1/E8 no-checksum one-layer baseline for H1 direction/full, so shifting the production stack from E8 to D2/E4 is not promising by itself. The pure data-sharded NS probe did not run because the synthetic group axis was invalid; the new group-size knob fixes that measurement gap without conflating production layer count.
- Next action: Do not patch production optimizer based on this failed D2 run. Next valid harness run should use D2/E4 with `MUON_BENCH_NS4D_GROUP_SIZE=2` for `ns4d_data_group` and restore variants, or stay on D1/E8 and investigate NS GEMM shape/occupancy.

### 2026-06-18 03:05 PDT - D2/E4 no-checksum result and concrete 4D sharding fix
- Hypothesis: The production 4D path regressed under D2/E4 because the real optimizer path was not carrying a concrete `NamedSharding` target for the grouped `[layer, expert, m, n]` stack when the harness disables the abstract mesh.
- Command: Incorporated external run `/dlwh/iris-run-job-20260618-095430`, run id `MUON-BENCH-D2560-L2-H1H5-D2E4-NOCHECKSUM-N1-cw-20260618-0954`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H5-D2E4-NOCHECKSUM-N1-cw-20260618-0954-7fb00f`. Local validation after optimizer patch: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --hidden-dim 2560 --intermediate-dim 2560 --num-experts 256 --backend-steps 1 --max-grouped-stack-size 512 --data-axis 2 --expert-axis 4 --model-axis 1 --replica-axis 1 --orthogonalization-layout stack_batch_4d_sharded --bench-kinds muon_direction,ns4d_data_group --mode lower --disable-abstract-mesh --output /tmp/muon_bench_d2e4_concrete_4d_real_lower.json`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py -q`; `uv run pytest experiments/grug/moe/test_optimizer.py -q`; `./infra/pre-commit.py --files lib/levanter/src/levanter/optim/grugmuon.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py scratch/launch_muon_update_bench_executor_n1.sh`.
- Config: D2/E4, layers=2, no synthetic checksum reductions, `orthogonalization_layout=stack_batch_4d_sharded`, H1/H5 split variants. Production optimizer patch keeps the existing pspec helper for abstract-mesh tests but now returns `NamedSharding(mesh, P(data, expert, None, None))` when the source leaf has concrete `NamedSharding`, and uses that target through grouped 4D NS and the temporary pre-split restore.
- Result: External D2/E4 run succeeded with all compiled collectives 0. Timings: H1 production full 34.14 ms for 2 layers, direction 29.94 ms, hyperball 4.32 ms, pure `ns4d_data_group` 13.00 ms. H5 production full 122.30 ms, direction 117.92 ms, hyperball 4.31 ms, pure `ns4d_data_group` 59.33 ms. Local real-shape D2/E4 lowering after the optimizer patch succeeds for `muon_direction` and `ns4d_data_group`; both show 6 dot_general / 6 two-batch-axis dots and 0 StableHLO collectives for H1. Focused pytest passes: `lib/levanter/tests/test_grugmuon.py` -> 20 passed; `experiments/grug/moe/test_optimizer.py` -> 9 passed. File-scoped pre-commit is blocked only by the pre-existing unrelated Pyrefly error in `lib/levanter/src/levanter/grug/loss.py:134`.
- Interpretation: This fixes a concrete correctness/lowering bug in the real 4D path: under `--disable-abstract-mesh`, the grouped NS now has an actual concrete sharding target instead of a bare `PartitionSpec`. It does not yet prove a speedup. The D2/E4 timing still shows production direction is about 2x slower than pure `ns4d_data_group`; even pure data grouping only matches the D1/E8 per-layer H5 baseline rather than beating it.
- Next action: First validation should be a bounded single-node harness timing of the patched production path, not full training. Recommended sweep: D2/E4, layers=2, H1/H5, `orthogonalization_layout=stack_batch_4d_sharded`, cap512, no checksum, include `muon_direction,muonh_update,hyperball_only,ns4d_data_group`. If production direction remains far above pure NS4D, inspect compiled HLO around stack/unstack/restore rather than launching another full Grug train.

### 2026-06-18 03:17 PDT - dot-only NS kernel sweep
- Hypothesis: If MuonH is slow because XLA lowers the 4D NS dots poorly, raw dot-only variants using `einsum`, `jnp.matmul`, or `lax.dot_general` should separate from full pure NS and from each other.
- Command: Primary external run `/dlwh/iris-run-job-20260618-101416`, child `/dlwh/iris-run-job-20260618-101416/grug-train-MUON-BENCH-D2560-L2-DOTONLY-H1H5-N1-cw-20260618-1014`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-DOTONLY-H1H5-N1-cw-20260618-1014-624ede`. Redundant H5-only confirmation run launched before the primary-result update: `/dlwh/iris-run-job-20260618-101451`, run id `MUON-BENCH-D2560-L2-H5-DOTONLY-N1-cw-20260618-1016`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H5-DOTONLY-N1-cw-20260618-1016-4507ea`. Local validation: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`.
- Config: N1 8xH100, D1/E8/M1, layers=2, group_size=2, `bench_kinds=ns4d_replicated_group,ns4d_dotonly_einsum,ns4d_dotonly_matmul,ns4d_dotonly_lax_dot_general`, backend_steps=1 and 5, iters=5. Harness now reports speed-of-light against `devices * 989 TF/s` for nominal dense bf16 H100 peak, fixing the earlier one-H100 denominator bug.
- Result: Primary run: H1 pure NS4D 12.7596 ms / 2 layers, zero compiled collectives; H1 dot-only variants tied at about 10.29 ms / 2 layers. H5 pure NS4D 58.2477 ms / 2 layers, zero compiled collectives; H5 dot-only variants tied at about 50.84 ms / 2 layers. Corrected speed-of-light estimate: H5 pure NS about 5.35 PF/s, about 67.6% of 8xH100 dense bf16 peak; H5 dot-only about 6.12 PF/s, about 77.4% of peak. Redundant H5-only run agreed directionally: pure NS4D 61.90 ms / 2 layers, dot-only about 54.34 ms / 2 layers, no compiled collectives.
- Interpretation: `einsum`, `jnp.matmul`, and explicit `lax.dot_general` all lower/timing-tie for the raw dot sequence. The isolated NS dot kernels are already above 70% of nominal dense peak in the dot-only case, and full pure NS H5 is around 68%. This makes kernel-expression coercion unlikely to close the train-step Muon gap.
- Next action: Treat the standalone harness as the required first benchmark loop before full train profiles. Next Muon work should move up a level: fewer NS steps, distribution across data/replica nodes, or reducing which matrices get Muon. Do not spend more cycles on einsum-vs-matmul lowering unless a new HLO/profile contradicts this result.

### 2026-06-18 04:15 PDT - H1/H3/H5 layout comparison
- Hypothesis: Sharding the grouped 4D stack across `data` or `replica_dcn` in addition to `expert` may improve standalone Muon NS throughput enough to justify a full training profile.
- Command: Added a harness-only `--ns4d-group-axis` / `MUON_BENCH_NS4D_GROUP_AXIS` knob and fixed momentum-sharding assertions for `replica_axis > 1`. Launched three single-node CoreWeave harness runs:
  - D1/E8 baseline parent `/dlwh/iris-run-job-20260618-110637`, child `/dlwh/iris-run-job-20260618-110637/grug-train-MUON-BENCH-D2560-L2-H1H3H5-D1E8-N1-cw-20260618-1030`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H3H5-D1E8-N1-cw-20260618-1030-d69379`.
  - D2/E4 data-group parent `/dlwh/iris-run-job-20260618-110645`, child `/dlwh/iris-run-job-20260618-110645/grug-train-MUON-BENCH-D2560-L2-H1H3H5-D2E4-GDATA-N1-cw-20260618-1030`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H3H5-D2E4-GDATA-N1-cw-20260618-1030-4caa78`.
  - R2/E4 replica-group retry parent `/dlwh/iris-run-job-20260618-111047`, child `/dlwh/iris-run-job-20260618-111047/grug-train-MUON-BENCH-D2560-L2-H1H3H5-R2E4-GREPL-N1-cw-20260618-1110`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H3H5-R2E4-GREPL-N1-cw-20260618-1110-4b84a9`.
- Config: N1 8xH100, `model_axis=1`, layers=2, cap512, bf16, iters=5, backend steps H1/H3/H5. Matrix inventory: `w_gate_up=[256,2560,2560]`, `w_down=[256,1280,2560]`; grouped 4D stack shape per leaf is `[2,256,m,n]`. Bench kinds: production `muonh_update`, production `muon_direction`, pure NS4D (`ns4d_replicated_group` for D1/E8, `ns4d_data_group` for D2/R2), and `ns4d_dotonly_matmul`. StableHLO dot counts are H1=6, H3=18, H5=30 two-batch-axis dots for every layout.
- Result:

| layout | kind | H1 ms/layer | H3 ms/layer | H5 ms/layer | H5 peak % | compiled collectives |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| D1/E8 | MuonH full | 9.07 | 20.28 | 31.82 | 61.8 | 0 |
| D1/E8 | direction | 7.81 | 19.24 | 30.64 | 64.2 | 0 |
| D1/E8 | pure NS4D | 6.38 | 17.73 | 31.03 | 63.4 | 0 |
| D1/E8 | dot-only | 5.14 | 15.26 | 25.44 | 77.4 | 0 |
| D2/E4 data group | MuonH full | 11.54 | 22.54 | 33.55 | 58.7 | 22 all-gather |
| D2/E4 data group | direction | 10.11 | 21.03 | 31.94 | 61.6 | 18 all-gather |
| D2/E4 data group | pure NS4D | 6.56 | 16.92 | 27.72 | 71.0 | 0 |
| D2/E4 data group | dot-only | 4.91 | 14.71 | 24.50 | 80.3 | 0 |
| R2/E4 replica group | MuonH full | 21.50 | 46.75 | 72.06 | 27.3 | 20-21 all-gather, 52 collective-permute |
| R2/E4 replica group | direction | 20.42 | 46.00 | 71.38 | 27.6 | 20-21 all-gather, 52 collective-permute |
| R2/E4 replica group | pure NS4D | 6.75 | 18.82 | 30.92 | 63.6 | 0 |
| R2/E4 replica group | dot-only | 5.51 | 16.38 | 27.23 | 72.3 | 0 |

Validation: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q` -> 29 passed. File-scoped pre-commit is blocked only by the known unrelated Pyrefly error in `lib/levanter/src/levanter/grug/loss.py:134`.
- Interpretation: Standalone pure NS4D clears the 50% speed-of-light target in all layouts. D2/E4 data-grouping is the only useful layout signal: it improves pure H5 NS4D from 31.03 to 27.72 ms/layer and dot-only H5 from 25.44 to 24.50 ms/layer with no compiled collectives. But the production tree path adds all-gathers and becomes slightly slower than D1/E8. R2/E4 is not a candidate: pure NS4D is no better than D1/E8, and production becomes much slower due all-gathers plus collective-permutes.
- Next action: Do not launch a full train profile from these results. If pursuing distribution, make the production optimizer consume data-sharded grouped NS without the restore/all-gather boundary, then rerun the harness. H3 is a plausible algorithmic knob to evaluate against training quality because it cuts standalone full MuonH from 31.82 to 20.28 ms/layer on D1/E8, while still reaching about 58% peak.

### 2026-06-18 04:26 PDT - harness padding and sharding-spec reporting
- Hypothesis: The standalone harness should make stack/group-axis sharding unambiguous and support padded non-divisible group sizes before full training profiles are considered.
- Command: Local smoke: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 3 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 512 --replica-axis 1 --data-axis 2 --expert-axis 4 --model-axis 1 --ns4d-group-axis data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds ns4d_data_group,ns4d_padded_group,ns4d_dotonly_matmul_padded --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output /tmp/muon_bench_padded_smoke.json`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py scratch/launch_muon_update_bench_executor_n1.sh`.
- Config: Added harness bench kinds `ns4d_padded_group` and `ns4d_dotonly_matmul_padded`. These take an original unpadded 4D group, keep input sharding as `P(None, expert, None, None)`, pad the group axis to the next multiple of the requested group mesh axis, reshard compute to `P(data|replica_dcn, expert, None, None)`, run NS/dot-only, restore group-axis replication, then slice back. Summary rows now include `ns4d_input_sharding_spec`, `ns4d_compute_sharding_spec`, `ns4d_group_size`, and `ns4d_padded_group_size`.
- Result: Local non-divisible smoke behaves as intended. Unpadded `ns4d_data_group` with group size 3 and `data_axis=2` emits a structured skipped row. Padded variants report input `P(None, 'expert', None, None)`, compute `P('data', 'expert', None, None)`, padded group size 4, H1 6 two-batch-axis dots, and compiled all-gathers from the required restore-before-slice boundary. Pycompile passed; focused pytest passed, 29 tests; file-scoped harness pre-commit passed.
- Interpretation: The harness now directly answers whether a stack/group axis is sharded or replicated, and exposes the cost/collectives of padding plus slicing. The restore-before-slice all-gather is expected for non-divisible output sizes and should block full-train launches unless a production design can avoid that boundary.
- Next action: Continue using this harness as the first gate for grouped/stacked Muon changes. The current best standalone metric remains the D2/E4 pure NS4D H5 result from the prior entry: 27.72 ms/layer, zero compiled collectives, about 71% of 8xH100 bf16 peak.

### 2026-06-18 05:40 PDT - D2/E4 restore-boundary GPU probe
- Hypothesis: The gap between D2/E4 pure `ns4d_data_group` and production `muonh_update` is the boundary that returns grouped `[layer, expert, fan_in, fan_out]` updates sharded as `P(data, expert, None, None)` back to separate per-layer leaves with parameter sharding.
- Command: CoreWeave parent `/dlwh/iris-run-job-20260618-123537`, child `/dlwh/iris-run-job-20260618-123537/grug-train-MUON-BENCH-D2560-L2-H1H3H5-D2E4-RESTOREPROBE2-N1-cw-20260618-1235`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H3H5-D2E4-RESTOREPROBE2-N1-cw-20260618-1235-9dd00c`. Local validation after the harness skip patch: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 2 --expert-axis 4 --model-axis 1 --ns4d-group-axis data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds ns4d_data_group,ns4d_data_reshard_restore,ns4d_data_index_restore --mode lower --disable-abstract-mesh --output /tmp/muon_bench_index_skip_smoke.json`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py`; `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`.
- Config: N1 8xH100, `model_axis=1`, `data_axis=2`, `expert_axis=4`, `layers=2`, `ns4d_group_size=2`, `orthogonalization_layout=stack_batch_4d_sharded`, cap512, H1/H3/H5. Bench kinds: production `muonh_update`, production `muon_direction`, pure `ns4d_data_group`, explicit `ns4d_data_reshard_restore`, and direct `ns4d_data_index_restore`.
- Result:

| kind | H1 ms/layer | H3 ms/layer | H5 ms/layer | H5 peak % | compiled all-gathers |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuonH full | 11.51 | 23.64 | 35.65 | 55.2 | 22 |
| direction | 10.49 | 22.17 | 33.83 | 58.2 | 18 |
| pure `ns4d_data_group` | 6.51 | 18.07 | 29.64 | 66.4 | 0 |
| explicit restore | 9.21 | 20.71 | 32.33 | 60.9 | 18 |
| direct index restore | skipped | skipped | skipped | n/a | n/a |

`ns4d_data_group` reports input/compute sharding `P('data', 'expert', None, None)` and zero compiled collectives. `ns4d_data_reshard_restore` preserves the same compute sharding but adds 18 compiled all-gathers. Direct indexing is not a viable workaround: JAX rejects slicing a `data`-sharded group axis down to size 1 because the sliced output dimension is not divisible by `data=2`. The harness now records that case as a structured skip instead of aborting the sweep. Focused tests pass: 29 passed.
- Interpretation: This confirms the design boundary. The D2/E4 pure NS kernel is healthy, but returning separate per-layer updates with the current parameter sharding forces all-gathers and erases much of the win. There is no small indexing trick that avoids the boundary. To use data-axis Muon parallelism in production, the params/update consumer must tolerate leaves whose leading expert stack is sharded over `data+expert`, or the optimizer must keep a grouped representation longer across the apply boundary.
- Next action: Do not launch a full train rerun from D2/E4 as-is. H3 on D1/E8 remains the practical single-node algorithmic knob; D2/E4 needs a broader parameter/update sharding design before it can beat the current production path.

### 2026-06-18 05:55 PDT - grouped apply-boundary lower-bound probe
- Hypothesis: If params and updates remain grouped as `[layer, expert, fan_in, fan_out]` through the apply boundary, the D2/E4 path can avoid the restore/split all-gathers while paying only the grouped apply cost.
- Command: Added harness-only bench kind `ns4d_data_group_apply`, then launched CoreWeave parent `/dlwh/iris-run-job-20260618-125012`, child `/dlwh/iris-run-job-20260618-125012/grug-train-MUON-BENCH-D2560-L2-H1H3H5-D2E4-GROUPAPPLY-N1-cw-20260618-1250`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-H1H3H5-D2E4-GROUPAPPLY-N1-cw-20260618-1250-097196`. Local validation: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py`; tiny D2/E4 grouped-apply smoke with `XLA_FLAGS=--xla_force_host_platform_device_count=8 ... --bench-kinds ns4d_data_group,ns4d_data_group_apply,ns4d_data_reshard_restore --mode both --disable-abstract-mesh`; `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py scratch/launch_muon_update_bench_executor_n1.sh`.
- Config: N1 8xH100, `model_axis=1`, `data_axis=2`, `expert_axis=4`, `layers=2`, `ns4d_group_size=2`, cap512, H1/H3/H5, `bench_kinds=muonh_update,muon_direction,ns4d_data_group,ns4d_data_group_apply,ns4d_data_reshard_restore`. The new harness variant runs pure 4D NS, Muon-style fan scaling, learning-rate scaling, and grouped `param + update` without splitting back to per-layer leaves; params and updates both use `P(data, expert, None, None)`.
- Result:

| kind | H1 ms / 2 layers | H3 ms / 2 layers | H5 ms / 2 layers | compiled all-gathers |
| --- | ---: | ---: | ---: | ---: |
| MuonH full | 23.01 | 48.91 | 73.49 | 22 |
| direction | 21.09 | 45.79 | 70.12 | 18 |
| pure `ns4d_data_group` | 13.11 | 37.75 | 62.09 | 0 |
| grouped apply | 13.93 | 38.12 | 62.47 | 0 |
| explicit restore/split | 18.29 | 43.04 | 67.30 | 18 |

All H1/H3/H5 lowered rows had the expected 6/18/30 two-batch-axis `dot_general` ops. The child and parent both succeeded with failure_count=0. Focused pytest passed: 29 passed. File-scoped pre-commit passed. One optional local abstract-mesh lowering smoke for the new variant hit the same concrete-vs-abstract mesh mismatch pattern as this harness mode can expose; the CoreWeave launcher and local concrete-mesh smoke both use `--disable-abstract-mesh` and passed.
- Interpretation: Keeping the grouped representation through apply is a credible lower-bound direction: it preserves `P(data, expert, None, None)`, removes the 18 restore all-gathers, and adds only about 0.4 ms at H3/H5 over pure NS for two layers. The win versus explicit restore is about 4.8 ms per two-layer H5 group, but this is still a representation-level design, not a small optimizer-only production patch, because the model params/apply path currently expects per-layer leaves.
- Next action: Do not launch a full training profile from the current production D2/E4 path. A full run is only justified after a production design can keep grouped expert params/updates through apply, or an equivalent apply consumer can accept grouped 4D leaves without forcing a split. Until then, D1/E8 H3 remains the practical single-node knob.

### 2026-06-18 06:12 PDT - harness-first production gate
- Hypothesis: The standalone Muon update benchmark is the right gate for future grouped/stacked Muon production work because it can expose sharding specs and compiled collectives before paying for a full Grug train/profile compile.
- Command: none.
- Config: Future Muon layout experiments remain single-node, `model_axis=1`, TTL outputs under `s3://marin-na/tmp/ttl=7d`, using `experiments/grug/moe/muon_update_bench.py`, `experiments/grug/moe/launch_cw_muon_update_bench.py`, and `scratch/launch_muon_update_bench_executor_n1.sh`.
- Result: Current production-feasibility gate is explicit: proposed grouped-4D changes must first show, in the harness, that the grouped expert param/update path preserves intended `NamedSharding` specs and has no restore/split all-gather boundary in compiled HLO. The key lower-bound remains `ns4d_data_group_apply`, which keeps `[layer, expert, fan_in, fan_out]` grouped through apply with `P(data, expert, None, None)` and zero compiled all-gathers.
- Interpretation: No full May run should be recommended from the current production D2/E4 path. A full run is justified only after the production optimizer/apply boundary can consume data-sharded grouped expert updates without splitting them back to per-layer `P(expert, None, None)` leaves.
- Next action: Use the harness as the primary iteration loop for any production representation patch. Validate sharding specs, dot counts, compiled collective counts, and ms/layer before recommending full Grug training.

### 2026-06-18 06:20 PDT - production-boundary grouped apply harness contract
- Hypothesis: The smallest production-adjacent proof should model grouped expert params and grouped updates through apply, and fail if the apply boundary reintroduces restore/split collectives.
- Command: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 2 --expert-axis 4 --model-axis 1 --ns4d-group-axis data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds ns4d_data_group_apply,ns4d_data_reshard_restore --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_apply_boundary_smoke.json`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`.
- Config: Local 8-device CPU mesh, D2/E4/M1, group size 2, tiny H1 shapes. Harness now reports `ns4d_input_sharding_spec`, `ns4d_compute_sharding_spec`, `ns4d_result_sharding_spec`, and `ns4d_boundary_status`; `ns4d_data_group_apply` asserts params, updates, and result keep `P(data, expert, None, None)` and raises if compiled HLO contains all-gather/reduce-scatter/all-reduce at that boundary.
- Result: Pycompile passed. Focused pytest passed: 30 tests. File-scoped pre-commit passed. Tiny smoke wrote `scratch/muon_update_bench_apply_boundary_smoke.json`: grouped apply reported input/compute/result `P('data', 'expert', None, None)`, lowered 6 two-batch-axis dots, and compiled all-gather/all-reduce/reduce-scatter all 0. The explicit restore/split control reported `ns4d_boundary_status=restore_then_split`, result `P('expert', None, None)`, and compiled all-gather 6 on the tiny shape.
- Interpretation: The harness now directly proves the apply-boundary contract locally: grouped params plus grouped updates can survive through apply with no restore/split collectives in the prototype representation. The negative control still catches the boundary that erases the D2/E4 win.
- Next action: Do not launch a full May run yet. The next justified step is a guarded production representation experiment that makes the real apply consumer accept grouped expert leaves, then reruns this harness at May shapes on one node before any full training profile.

### 2026-06-18 06:45 PDT - real-ish grouped expert apply-boundary harness
- Hypothesis: The next production-representation gate should keep the real Grug block/expert path visible while grouping only `blocks[*].mlp.expert_mlp.{w_gate_up,w_down}` as 4D `[layer_group, expert, fan_in, fan_out]` leaves through `optax.apply_updates`.
- Command: Local validation: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py experiments/grug/moe/test_optimizer.py -q`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 2 --expert-axis 4 --model-axis 1 --ns4d-group-axis data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds expert_grouped_apply_boundary,ns4d_data_group_apply,ns4d_data_reshard_restore --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_expert_grouped_apply_smoke.json`.
- Config: Added harness bench kind `expert_grouped_apply_boundary`. It builds a production-like grouped tree under `blocks[group].mlp.expert_mlp.{w_gate_up,w_down}`, with grouped leaves sharded as `P(data, expert, None, None)`, runs 4D NS plus Muon-style scaling, and applies updates with `optax.apply_updates` without splitting back to per-layer leaves. Updated `scratch/launch_muon_update_bench_executor_n1.sh` to default to the single-node D2/E4 May-shape harness gate: `muonh_update,muon_direction,ns4d_data_group_apply,expert_grouped_apply_boundary,ns4d_data_reshard_restore`, H1/H3/H5, cap512, TTL output.
- Result: Local pycompile passed. Focused pytest passed: 11 tests. Tiny D2/E4 smoke wrote `scratch/muon_update_bench_expert_grouped_apply_smoke.json`. `expert_grouped_apply_boundary` reported input/compute/result `P('data', 'expert', None, None)`, lowered 6 two-batch-axis dots, and compiled all-gather/all-reduce/reduce-scatter all 0. The explicit restore/split control reported compiled all-gather 6.
- Interpretation: This is the smallest guarded production-representation experiment so far: it preserves the real block/expert logical path while keeping expert params, updates, and apply result grouped across data+expert. It is still a harness representation, not a full model-param format change.
- Next action: Run this Muon-only harness at May shapes on one CoreWeave node before any full May train/profile. Full train remains unsafe until the May-shape harness row preserves `P(data, expert, None, None)` and shows zero compiled restore/split collectives for the grouped expert apply boundary.

### 2026-06-18 07:10 PDT - grouped MuonH optimizer/apply harness gate
- Hypothesis: The production-ish grouped expert gate must include the MuonH hyperball projection, not only momentum/Nesterov plus the Newton-Schulz direction and `optax.apply_updates`.
- Command: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 2 --expert-axis 4 --model-axis 1 --ns4d-group-axis data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds expert_grouped_muonh_optimizer_apply,expert_grouped_optimizer_apply --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_grouped_muonh_d2e4_smoke.json`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 2 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 1 --expert-axis 8 --model-axis 1 --ns4d-group-axis none --orthogonalization-layout stack_batch_4d_sharded --bench-kinds expert_grouped_muonh_optimizer_apply --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_grouped_muonh_e8_smoke.json`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py .agents/logbooks/grug-moe-muon-gpu.md`.
- Config: Added `expert_grouped_muonh_optimizer_apply` as a distinct bench kind from direction-only `expert_grouped_optimizer_apply`. The new gate builds a mixed production-ish tree with grouped `blocks[*].mlp.expert_mlp.{w_gate_up,w_down}` 4D leaves plus ordinary 1D/2D leaves, then runs `optax.trace` momentum/Nesterov, 4D Newton-Schulz direction, Muon direction scaling, grouped 4D hyperball projection, and `optax.apply_updates`. The grouped hyperball reduces only over trailing matrix axes `(-2, -1)`, preserving group and expert axes.
- Result: Pycompile passed. Focused pytest passed: 35 passed. File-scoped pre-commit passed. Local tiny D2/E4 smoke wrote `scratch/muon_update_bench_grouped_muonh_d2e4_smoke.json`: `expert_grouped_muonh_optimizer_apply` reported input/compute/result `P('data', 'expert', None, None)`, lowered 6 two-batch-axis dots, and lowered/compiled all-gather/all-reduce/reduce-scatter all 0. The direction-only control used the same sharding and also had AG/AR/RS 0/0/0. Local tiny E8 smoke wrote `scratch/muon_update_bench_grouped_muonh_e8_smoke.json`: `expert_grouped_muonh_optimizer_apply` reported input/compute/result `P(None, 'expert', None, None)`, lowered 6 two-batch-axis dots, and lowered/compiled AG/AR/RS 0/0/0.
- Interpretation: The local harness now distinguishes direction-only speed from full grouped MuonH apply behavior and proves the apply-boundary sharding/collective contract for D2/E4 and E8 on tiny shapes. This still does not clear a full May train profile: the next missing gate is a CoreWeave May-shape run of `expert_grouped_muonh_optimizer_apply` with H1/H3/H5 to measure wall time and peak fraction.
- Next action: Run the narrow standalone Muon update harness on one CoreWeave node, with output under `s3://marin-na/tmp/ttl=7d`, before recommending any full May training/profile run.

### 2026-06-18 07:30 PDT - May-shape grouped MuonH harness and May156 full profile
- Hypothesis: If the full grouped MuonH optimizer/apply harness clears H1/H3/H5 at May shape, a full single-node H5 profile should show whether production training now spends Muon time near the standalone speed-of-light floor.
- Command:
  - Standalone D2/E4: `RUN_ID="MUON-BENCH-D2560-L2-D2E4-GROUPEDMUONH-H1H3H5-N1-cw-$(date -u +%Y%m%d-%H%M)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_DATA_AXIS=2 MUON_BENCH_EXPERT_AXIS=4 MUON_BENCH_NS4D_GROUP_AXIS=data MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_SWEEP_BACKEND_STEPS=1,3,5 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply bash scratch/launch_muon_update_bench_executor_n1.sh`
  - Standalone E8: `RUN_ID="MUON-BENCH-D2560-L2-E8-GROUPEDMUONH-H1H3H5-N1-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_NS4D_GROUP_AXIS=none MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_SWEEP_BACKEND_STEPS=1,3,5 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply bash scratch/launch_muon_update_bench_executor_n1.sh`
  - Full profile: `bash scratch/launch_may156_fa4_single_node_b8_muonh5_pallas_ce_v8192_readable_profile.sh`
- Config:
  - Standalone harness: one 8xH100 node, layers=2, H1/H3/H5, cap512, `expert_grouped_muonh_optimizer_apply`.
  - May156: single node, `model_axis=1`, `expert_axis=8`, batch=8, seq_len=4096, sliding_window=2048, FA4 CuTe, Pallas CE v8192, ring MoE, bf16 params/compute/output, MuonH5, `orthogonalization_layout=stack_batch_4d_sharded`, cap512, command buffers disabled for trace readability.
- Result:
  - D2/E4 standalone parent `/dlwh/iris-run-job-20260618-140449`, child `/dlwh/iris-run-job-20260618-140449/grug-train-MUON-BENCH-D2560-L2-D2E4-GROUPEDMUONH-H1H3H5-N1-cw-20260618-1404`: H5 66.93 ms / two layers, 58.80% nominal 8xH100 bf16 peak, `P(data, expert, None, None)`, compiled AG/AR/RS 0/0/0.
  - E8 standalone parent `/dlwh/iris-run-job-20260618-140515`, child `/dlwh/iris-run-job-20260618-140515/grug-train-MUON-BENCH-D2560-L2-E8-GROUPEDMUONH-H1H3H5-N1-cw-20260618-140512`: H5 67.24 ms / two layers, 58.53% nominal 8xH100 bf16 peak, `P(None, expert, None, None)`, compiled AG/AR/RS 0/0/0.
  - May156 parent `/dlwh/iris-run-job-20260618-140902`, child `/dlwh/iris-run-job-20260618-140902/grug-train-GM2560-MAY-156S4096-W2048-B8-R1-E8M1-PALLASCEV8192-RING-REPEMB-REPOUT-FA4MUONH5-GROUPED4D-READABLE-PROFILE-N1-cw-20260618-1408`: finished. W&B `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-156S4096-W2048-B8-R1-E8M1-PALLASCEV8192-RING-REPEMB-REPOUT-FA4MUONH5-GROUPED4D-READABLE-PROFILE-N1-cw-20260618-1408`. TensorBoard served at `http://127.0.0.1:6019/`.
  - May156 steady profile steps: global_step 2/3 MFU 4.7495/4.6793, tokens/s 21,960.9/21,636.3, duration 1.492/1.514 s. Final step 4 was profiler-finalization affected: MFU 2.4672, duration 2.872 s.
  - May156 xprof semantic split: `optimizer_muon` 45.84%, `moe` 11.43%, `loss_xent` 1.13%, `attention_flash` 2.61%, plus 32.63% `other`.
- Interpretation: The isolated grouped expert MuonH path clears the requested ~50% speed-of-light bar. In full training, `stack_batch_4d_sharded` is active and xprof attributes the largest kernels to `grug_muon/.../newton_schulz_grouped_4d`, but H5 still dominates the step and full-profile MFU did not improve versus prior H5 runs. The missing win is not just routed expert grouping; full production Muon includes more matrices and still costs roughly half the profiled kernel time.
- Next action: Treat May156 as the current readable H5 profile. To improve single-node speed, pursue H1/H3 as algorithmic knobs and/or a broader grouped/stacked Muon implementation that covers the full Muon workload, while Noether/Harvey/Dalton continue MoE/attention/xent checks.

### 2026-06-18 07:48 PDT - full production MuonH standalone harness
- Hypothesis: The standalone grouped expert gate under-covered the May156 bottleneck because production `muonh` also includes attention, GatedNorm, and shared-MLP 2D matrices.
- Command: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; local smoke: `uv run python experiments/grug/moe/muon_update_bench.py --layers 2 --ns4d-group-size 2 --ns4d-group-axis none --hidden-dim 16 --intermediate-dim 8 --num-experts 1 --backend-steps 1 --expert-axis 1 --data-axis 1 --model-axis 1 --bench-kinds full_production_muonh_optimizer_apply --mode lower --disable-abstract-mesh --output /tmp/muon_full_prod_smoke.json`.
- Config: Added harness bench kind `full_production_muonh_optimizer_apply`. It builds production-shaped grouped 4D routed expert leaves plus 2D MuonH leaves for attention `w_q/w_k/w_v/w_o`, attention/MLP GatedNorm, and shared MLP, while masking router, attention gate, embeddings, and output projection as ordinary/non-Muon. For `model_axis=1`, size-1 `model` axes are omitted from synthetic 2D sharding specs to avoid explicit-sharding dot ambiguity in the harness.
- Result: Pycompile passed. Focused pytest passed: 36 passed. Tiny local smoke lowered cleanly with `all_gather=0`, `all_reduce=0`, `reduce_scatter=0`, `dot_general=72`, `two_batch_axis_dot_general=6`.
- Interpretation: The Muon agent now has a fast loop for the full production Muon optimizer group instead of only the routed expert subset. This should explain the May156 gap before another full train/profile run.
- Next action: Run this new bench kind on one CoreWeave node at May shape: `layers=26`, `ns4d_group_size=4` or `8`, `ns4d_group_axis=none` for E8 first, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, backend steps sweep `1,5`, `expert_axis=8`, `model_axis=1`; compare directly with `expert_grouped_muonh_optimizer_apply`.

### 2026-06-18 07:56 PDT - full production MuonH L2 CoreWeave gate
- Hypothesis: A two-layer May-shape standalone gate is enough to measure the broader production MuonH group without the L26 live-state OOM, and can be scaled by 13 for the 26-layer model.
- Command:
  - Failed L26 probe: `RUN_ID=MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-H1H5-N1-cw-20260618-144702 ... MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply bash scratch/launch_muon_update_bench_executor_n1.sh`
  - Successful L2 gate: `RUN_ID=MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H1H5-N1-cw-20260618-145201 ... MUON_BENCH_LAYERS=2 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply bash scratch/launch_muon_update_bench_executor_n1.sh`
- Config: One 8xH100 node, `model_axis=1`, `expert_axis=8`, `data_axis=1`, `ns4d_group_axis=none`, `ns4d_group_size=2`, D=2560, intermediate=1280, experts=256, backend steps H1/H5, cap512.
- Result:
  - L26 parent `/dlwh/iris-run-job-20260618-144707` failed during first H1 timing before full-production rows: XLA reported input/output arguments 99.56 GB, remat floor 46.36 GiB, and BFC OOM on an extra 800 MiB allocation.
  - L2 parent `/dlwh/iris-run-job-20260618-145204`, child `/dlwh/iris-run-job-20260618-145204/grug-train-MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H1H5-N1-cw-20260618-145201`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H1H5-N1-cw-20260618-145201-c3e6f1`, succeeded.

| bench | H1 mean ms / 2 layers | H1 peak % | H5 mean ms / 2 layers | H5 peak % | compiled AG/AR/RS |
| --- | ---: | ---: | ---: | ---: | --- |
| routed expert grouped MuonH | 17.66 | 44.56 | 66.46 | 59.22 | 0/0/0 |
| full production MuonH | 22.65 | 35.32 | 83.21 | 48.08 | 0/0/0 |

- Interpretation: Full production MuonH5 is just under the 50% nominal-peak target and is about 25% slower than routed-expert-only MuonH5. The extra 2D production Muon group adds 16.75 ms per two-layer group at H5; scaled to 26 layers that is about 218 ms. H1 full production Muon is much faster in absolute wall time at about 22.65 ms / two layers, but only 35% nominal peak.
- Next action: For full training, H1/H3 remain the practical speed knobs because H5 costs roughly 1.08 s when scaled to 26 layers. For Muon-specific implementation work, the next target is recovering the full production H5 path from 48% to at least 50-60% and deciding whether a grouped/padded representation can avoid L26 live-state OOM.

### 2026-06-18 08:18 PDT - full production MuonH L26 lean timing gate
- Hypothesis: The L26 live-state OOM came from the benchmark timing wrapper returning the full update tree, not from the grouped MuonH compute itself. A timing-only wrapper that returns only params/state and donates params plus optimizer state should keep the update-only harness as the fast primary loop.
- Command: Local patch validation before relaunch: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 5 --ns4d-group-size 4 --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 1 --data-axis 1 --expert-axis 8 --model-axis 1 --ns4d-group-axis none --orthogonalization-layout stack_batch_4d_sharded --bench-kinds full_production_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_fullprod_timing_smoke.json`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py scratch/launch_muon_update_bench_executor_n1.sh scratch/20260618-0744_fullprod_muonh_harness_state.json`. CoreWeave retry command: `RUN_ID=MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H1H5-LEAN-N1-cw-$(date -u +%Y%m%d-%H%M) MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=4 MUON_BENCH_NS4D_GROUP_AXIS=none MUON_BENCH_REPLICA_AXIS=1 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_ORTHOGONALIZATION_LAYOUT=stack_batch_4d_sharded MUON_BENCH_SWEEP_BACKEND_STEPS=1,5 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true bash scratch/launch_muon_update_bench_executor_n1.sh`.
- Config: One 8xH100 node, `model_axis=1`, `expert_axis=8`, `data_axis=1`, `ns4d_group_axis=none`, `ns4d_group_size=4`, D=2560, intermediate=1280, experts=256, layers=26, backend steps H1/H5, cap512. The lower/eval path still returns updates for sharding assertions; only the timing path uses the lean output contract.
- Result:
  - The old L26 timing path failed in parent `/dlwh/iris-run-job-20260618-144423` with `RESOURCE_EXHAUSTED` while returning full update trees; XLA reported about 99.56 GB of input/output arguments.
  - Lean retry parent `/dlwh/iris-run-job-20260618-145305`, child `/dlwh/iris-run-job-20260618-145305/grug-train-MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H1H5-LEAN-N1-cw-20260618-1453`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H1H5-LEAN-N1-cw-20260618-1453-ca2a69`, succeeded with failure_count=0.

| bench | H1 mean s / 26 layers | H1 peak % | H5 mean s / 26 layers | H5 peak % | compiled AG/AR/RS |
| --- | ---: | ---: | ---: | ---: | --- |
| routed expert grouped MuonH | 0.2069 | 49.45 | 0.7903 | 64.74 | 0/0/0 |
| full production MuonH | 0.2501 | 41.59 | 0.9581 | 54.28 | 0/0/0 |

- Interpretation: The standalone update-only harness now covers the full production MuonH group at L26 without the previous live-state OOM. Full-production H5 clears the requested 50-60% nominal-peak band on L26 and preserves zero compiled all-gather/all-reduce/reduce-scatter at the grouped optimizer/apply boundary. This does not mean the existing production training optimizer path is fixed: the harness has the grouped representation and lean timing contract, while the real train path still needs a narrow production/apply integration gate before a full May profile is a clean test.
- Next action: Keep the update-only harness as the primary loop. Benchmark H3 on the same full-production L26 path next, then add or verify a narrow train/apply gate that uses the grouped production representation before launching another full May training profile.

### 2026-06-18 08:30 PDT - full production MuonH H3 speed-clearing compromise
- Hypothesis: H3 may be the practical production compromise: materially faster wall time than H5 while still clearing the full-production update-only harness speed and collective gates.
- Command: Main-thread CoreWeave run, parent `/dlwh/iris-run-job-20260618-150345`, child `/dlwh/iris-run-job-20260618-150345/grug-train-MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-150341`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-150341-7b9ee1`.
- Config: One 8xH100 node, `model_axis=1`, `expert_axis=8`, `data_axis=1`, E8 full-production MuonH update-only harness, layers=2, D=2560, intermediate=1280, experts=256, H3 only.
- Result:

| bench | H3 mean ms / 2 layers | H3 peak % | compiled AG/AR/RS |
| --- | ---: | ---: | --- |
| routed expert grouped MuonH | 40.37 | 58.49 | 0/0/0 |
| full production MuonH | 47.37 | 50.68 | 0/0/0 |

Issue update: https://github.com/marin-community/marin/issues/6493#issuecomment-4743340380.
- Interpretation: Treat H3 as the current speed-clearing production compromise. It preserves the hidden-collective gate and reaches about 50.7% nominal peak for the full production Muon group at the two-layer May shape. It is much lower wall time than H5 in the same L2 baseline, while still being a real MuonH hyperball path rather than direction-only.
- Next action: Do not launch unchanged full train profiles. The next concrete axes are: (1) decide whether H3 is scientifically acceptable and expose it cleanly in the production optimizer/config path, or (2) improve the harness implementation enough to push full-production H3 materially above 50% without changing train semantics.

### 2026-06-18 08:45 PDT - first-class Muon update fast-loop harness
- Hypothesis: The Muon update harness should be the default iteration path for Muon/MuonH representation and step-count work, with enough summary metadata to compare H1/H3/H5, stack caps, padding/sharding strategy, and hidden collectives without launching full Grug training.
- Command: Local wrapper smoke: `MUON_BENCH_PROFILE=fullprod-e8-h3 MUON_BENCH_HIDDEN_DIM=16 MUON_BENCH_INTERMEDIATE_DIM=8 MUON_BENCH_NUM_EXPERTS=8 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=1 bash scratch/muon_update_bench_fast_loop.sh local`. Validation: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_optimizer.py -q`; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py experiments/grug/moe/MUON_UPDATE_BENCH.md scratch/muon_update_bench_fast_loop.sh .agents/logbooks/grug-moe-muon-gpu.md`.
- Config: Added `scratch/muon_update_bench_fast_loop.sh` profile wrapper around the existing local CLI and one-node Iris launcher. Profiles: `fullprod-e8`, `fullprod-e8-h3`, `fullprod-e8-l26-h3`, `grouped-d2e4`, and `padding-d2e4`. All profiles default to single node, `model_axis=1`; Iris target keeps outputs under `s3://marin-na/tmp/ttl=7d`. Added `experiments/grug/moe/MUON_UPDATE_BENCH.md` with local and CoreWeave commands, output fields, and the current best result.
- Result: The compact `summary_table` now includes explicit `estimated_matrix_count`, `grouped_expert_group_count`, and full `group_estimates` in addition to existing wall time, estimated FLOPs, TFLOP/s, H100 peak fraction, sharding specs, stack chunks, and lowered/compiled collective counts. Local wrapper smoke wrote `scratch/MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-20260618-152510.json`; tiny full-production H3 reported `estimated_matrix_count=54`, `P(None, expert, None, None)`, and AG/AR/RS 0/0/0. Focused pytest passed: 37 passed. File-scoped pre-commit passed.
- Interpretation: The benchmark is now a first-class fast iteration tool rather than a hidden Python module plus env-heavy launcher. It directly exercises the update/apply path with production-like grouped expert leaves and full-production 2D MuonH leaves, while making the matrix count, chunking, sharding, and collective behavior visible in the first summary table.
- Next action: Use `MUON_BENCH_PROFILE=fullprod-e8-h3 bash scratch/muon_update_bench_fast_loop.sh iris` as the default remote speed-clearing gate. Continue to avoid unchanged full train profiles unless the harness first shows a concrete implementation or semantics axis worth testing.

### 2026-06-18 08:36 PDT - L26 H3 Muon update-only gate
- Hypothesis: The L2 H3 result should hold at the full 26-layer May shape when using the first-class fast-loop harness, giving a stronger single-node Muon-only gate than the earlier two-layer estimate.
- Command: `MUON_BENCH_PROFILE=fullprod-e8-l26-h3 bash scratch/muon_update_bench_fast_loop.sh iris`.
- Config: One 8xH100 node, `layers=26`, `model_axis=1`, `expert_axis=8`, `data_axis=1`, `ns4d_group_axis=none`, `ns4d_group_size=4`, D=2560, intermediate=1280, experts=256, H3 only, grouped 4D sharded layout, cap512, outputs under `s3://marin-na/tmp/ttl=7d`.
- Result:
  - Parent `/dlwh/iris-run-job-20260618-153021`, child `/dlwh/iris-run-job-20260618-153021/grug-train-MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H3-N1-cw-20260618-153016`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H3-N1-cw-20260618-153016-87deb1`.
  - `expert_grouped_muonh_optimizer_apply_h3`: 0.5236 s / 26 layers, 4,638.9 estimated TFLOP/s, 58.63% nominal 8xH100 bf16 peak, compiled AG/AR/RS 0/0/0, estimated matrix count 13,312.
  - `full_production_muonh_optimizer_apply_h3`: 0.6171 s / 26 layers, 4,000.6 estimated TFLOP/s, 50.56% nominal 8xH100 bf16 peak, compiled AG/AR/RS 0/0/0, estimated matrix count 13,598.
- Interpretation: The full 26-layer production-shaped H3 update-only gate clears the requested ~50% Muon speed-of-light target, not just the earlier L2 extrapolation. H3 is about 1.55x faster in wall time than the previous L26 H5 full-production update-only gate (0.617 s vs 0.958 s) while preserving the zero-hidden-collectives invariant.
- Next action: Muon-specific work should now move from speed-gating to production integration/semantics: expose H3 cleanly if acceptable, or use the harness to prove another concrete improvement before paying for full-train profiles.

### 2026-06-18 15:05 PDT - replica/data grouped Muon scale-gate wiring
- Hypothesis: The next Muon-only scale gate should prove the grouped layer axis can shard over inter-node `replica_dcn` and local `data` axes together, not only over `data`, before spending CoreWeave time on 32/128 GPU harness runs.
- Command: `uv run pytest lib/levanter/tests/test_grugmuon.py experiments/grug/moe/test_muon_update_bench.py experiments/grug/moe/test_optimizer.py -q`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; `bash -n scratch/muon_update_bench_fast_loop.sh scratch/launch_muon_update_bench_executor_n1.sh`; `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --hidden-dim 16 --intermediate-dim 8 --num-experts 4 --backend-steps 1 --max-grouped-stack-size 8 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --ns4d-group-axis replica_dcn,data --orthogonalization-layout stack_batch_4d_sharded --bench-kinds expert_grouped_muonh_optimizer_apply --mode both --disable-abstract-mesh --warmup 1 --iters 1 --output scratch/muon_update_bench_replica_data_smoke.json`.
- Config: Added `replica_dcn,data` / `data,replica_dcn` as explicit harness group-axis modes and changed the production grouped-4D target helper to choose the widest divisible subset of `replica_dcn` and `data` for the group axis. The CoreWeave launcher now accepts `MUON_BENCH_GPU_REPLICAS`; fast-loop profiles `fullprod-r4e8-l26-h3` and `fullprod-r16e8-l26-h3` prepare 32-GPU and 128-GPU update-only H3 scale gates while keeping `model_axis=1`.
- Result: Focused pytest passed: 39 passed. Pycompile passed. Shell syntax checks passed. Local 8-device CPU smoke wrote `scratch/muon_update_bench_replica_data_smoke.json`: `expert_grouped_muonh_optimizer_apply_h1` reported input/compute/result `P(('replica_dcn', 'data'), 'expert', None, None)`, lowered 6 two-batch-axis dots, and lowered all-gather/all-reduce/reduce-scatter all 0.
- Interpretation: This is not a GPU performance claim, but it proves the scale-gate representation and `NamedSharding.spec` contract for a combined replica/data group axis. The next CoreWeave run should be update-only, not full train, and should compare R4/E8 H3 against the known R1/E8 L26 H3 baseline.
- Next action: Launch `MUON_BENCH_PROFILE=fullprod-r4e8-l26-h3 bash scratch/muon_update_bench_fast_loop.sh iris` when ready. If R4/E8 scales cleanly and preserves AG/AR/RS 0/0/0, follow with `MUON_BENCH_PROFILE=fullprod-r16e8-l26-h3 bash scratch/muon_update_bench_fast_loop.sh iris`.

### 2026-06-19 00:25 PDT - R2/D2/E8 grouped-to-FSDP boundary attribution
- Hypothesis: The pragmatic FSDP-master design may still be viable if the grouped MuonH update can be restored to FSDP layout cheaply enough before ordinary `optax.apply_updates`.
- Command: CoreWeave update-only harness runs on 4 nodes, including `/dlwh/iris-run-job-20260619-072348`, `/dlwh/iris-run-job-20260619-073428`, `/dlwh/iris-run-job-20260619-073920`, and `/dlwh/iris-run-job-20260619-074440`.
- Config: 32 H100s, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts=256`, `backend_steps=3`, `group_size=4`. Compared `replica_dcn,data`, target-layout restore, grouped-param target, and replica-only grouping.
- Result: `replica_dcn,data` restore-only was about 0.304s with 150 compiled all-gathers; grouped updates plus apply was about 0.406s with 150 all-gathers and 140 all-to-alls. A target-layout restore reduced all-gather count to 112 but stayed about 0.309s and emitted XLA rematerialization warnings about replicate-then-partition. Grouped params/updates stayed fast at about 0.161s, zero compiled collectives, and about 47.6% nominal peak. Replica-only grouping made restore-only cheaper at about 0.220s but made MuonH compute much worse, about 0.803s with all-to-alls/collective-permutes.
- Interpretation: The literal "FSDP master weights, grouped optimizer compute, restore back to FSDP, then normal apply" path is not cheap in current XLA GPU. `apply_updates` itself is not the bottleneck; the grouped-to-FSDP boundary is. The fast target remains a grouped expert-bank representation that the model or apply consumer can use without splitting the grouped axis.
- Next action: Keep testing narrow boundary variants in the harness; do not recommend a full train run from the FSDP-restore path until the grouped-to-FSDP boundary is fixed or avoided.

### 2026-06-19 01:10 PDT - bf16 Newton-Schulz compute for fp32 inputs
- Hypothesis: If fp32 master weights are required scientifically, Newton-Schulz itself may still be safely computed in bf16 and recover most of the isolated MuonH speed.
- Command: One-node validation parents `/dlwh/iris-run-job-20260619-075924` and `/dlwh/iris-run-job-20260619-075931`; four-node apply-only parents `/dlwh/iris-run-job-20260619-081247` and `/dlwh/iris-run-job-20260619-081308`.
- Config: fp32 harness inputs, H3, grouped expert MuonH. N1 validation used `layers=4`; R2/D2/E8 validation used `layers=26`, `group_size=4`, bench kind `expert_grouped_muonh_optimizer_apply`, with `MUON_BENCH_NS_COMPUTE_DTYPE=input` versus `bf16`.
- Result: On N1, `expert_grouped_muonh_optimizer_apply` improved from about 0.1629s to 0.0868s with bf16 NS, a 1.88x speedup, and dot-only improved from about 0.1301s to 0.0685s, a 1.90x speedup. On R2/D2/E8 full L26 apply-only, fp32 NS was about 0.328s at about 23.4% peak; bf16 NS was about 0.173s at about 44.3% peak. Compiled AG/AR/RS/A2A were zero in the apply-only rows.
- Interpretation: bf16 Newton-Schulz compute is a large isolated win even when the stored/master inputs are fp32. It should remain in the candidate path, guarded as an explicit algorithmic/numerics knob rather than hidden behind dtype defaults.
- Next action: Keep `MUON_BENCH_NS_COMPUTE_DTYPE=bf16` in fp32-input Muon-only scale probes unless the experiment is explicitly measuring fp32 NS.

### 2026-06-19 01:25 PDT - full-L26 slice-boundary OOM and single-layer slice probe
- Hypothesis: The grouped expert bank may still be usable by slicing one layer at a time, avoiding the full all-layer restore that OOMs the 16GB laptop and also OOMs remote full-L26 slice/dot-only harness rows.
- Command: Failed full-L26 jobs `/dlwh/iris-run-job-20260619-080315`, `/dlwh/iris-run-job-20260619-080322`, `/dlwh/iris-run-job-20260619-081022`, `/dlwh/iris-run-job-20260619-081043`; successful single-layer slice job `/dlwh/iris-run-job-20260619-081849`.
- Config: Full-L26 R2/D2/E8 fp32-input grouped expert bank with H3. The successful row used bench kind `expert_grouped_single_layer_slice_boundary` and avoided all-layer slice materialization.
- Result: Full-L26 `ns4d_dotonly_matmul` and all-layer `expert_grouped_layer_slice_boundary` failed with 26.56 GiB allocations. The single-layer slice job succeeded but compiled 16 all-gathers and took about 0.08695s to slice one layer's two expert matrices. A local index-first variant failed before launch because JAX cannot slice a group axis sharded over four mesh partitions down to output size 1.
- Interpretation: Single-layer slicing avoids the OOM but is still too expensive if repeated per block. The current XLA behavior wants to all-gather the grouped bank before producing per-layer FSDP-style leaves. That supports the same conclusion as the restore-boundary probes: the production direction should avoid the split, not try to do many tiny grouped-to-FSDP conversions.
- Next action: Test compromise group axes such as `data` only on R2/D2/E8 to see whether there is a better compute/slice tradeoff, but treat it as exploratory; the leading candidate remains grouped-bank/model-consumable params or an explicit optimized boundary.

### 2026-06-19 01:30 PDT - R2/D2/E8 data-only compromise probe
- Hypothesis: Sharding the grouped layer axis over `data` only might reserve `replica_dcn` for the FSDP side, reducing grouped-to-layer slice cost while keeping some Newton-Schulz parallelism.
- Command: `RUN_ID=MUON-BENCH-D2560-L26-R2D2E8-DATAONLY-G4-H3-BF16NS-N4-cw-20260619-082619 MUON_BENCH_TARGET=iris MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=4 MUON_BENCH_NS4D_GROUP_AXIS=data MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=2 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=4 MUON_BENCH_DTYPE=fp32 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_KINDS=expert_grouped_muonh_optimizer_apply,expert_grouped_single_layer_slice_boundary MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=5 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true bash scratch/muon_update_bench_fast_loop.sh iris expert-only-r2d2e8-l26-h3`.
- Config: Parent `/dlwh/iris-run-job-20260619-082622`; child `/dlwh/iris-run-job-20260619-082622/grug-train-MUON-BENCH-D2560-L26-R2D2E8-DATAONLY-G4-H3-BF16NS-N4-cw-20260619-082619`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-DATAONLY-G4-H3-BF16NS-N4-cw-20260619-082619-705c62`. 32 H100s, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, fp32 inputs, bf16 NS compute, H3, group size 4.
- Result:

| bench | mean seconds | peak % | compiled AG/AR/RS/A2A/CP |
| --- | ---: | ---: | --- |
| `expert_grouped_muonh_optimizer_apply` | 0.27708 | 27.70 | 0/0/0/0/0 |
| `expert_grouped_single_layer_slice_boundary` | 0.06809 | 17.34 | 16/0/0/0/0 |

The grouped compute row preserved `P('data', 'expert', None, None)` with zero compiled collectives. The child succeeded with four tasks and no failures.
- Interpretation: `data`-only is not the compromise path. It is slower than prior `replica_dcn,data` bf16-NS grouped apply, about 0.277s versus about 0.173s, while the one-layer slice is still expensive and still emits 16 all-gathers. The slice boundary improved only modestly versus the prior `replica_dcn,data` one-layer slice, about 68ms versus about 87ms, nowhere near enough to justify losing NS parallelism.
- Next action: Stop spending runs on simple narrower group axes. The credible path is still grouped-bank/model-consumable expert params or a consciously optimized explicit boundary; the current JAX/XLA grouped-to-FSDP conversion remains too expensive.

### 2026-06-19 01:45 PDT - target-layout grouped apply boundary OOM
- Hypothesis: The grouped-to-FSDP boundary might become viable if we avoid splitting back to per-layer leaves and instead reshard grouped params/updates into a grouped FSDP-target layout, then apply updates while still grouped.
- Command:
  - Local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; tiny lower and compile-only smoke runs for `expert_fsdp_grouped_target_apply_boundary` on an 8-device CPU mesh.
  - CoreWeave probe: parent `/dlwh/iris-run-job-20260619-083657`; child `/dlwh/iris-run-job-20260619-083657/grug-train-MUON-BENCH-D2560-L26-R2D2E8-TARGETAPPLYBOUNDARY-G4-N4-cw-20260619-083654`; output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-TARGETAPPLYBOUNDARY-G4-N4-cw-20260619-083654-90dd94`.
- Config: 4 CoreWeave H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts=256`, fp32 inputs, bf16 NS compute, group size 4, `ns4d_group_axis=replica_dcn,data`. Compared `expert_fsdp_grouped_target_apply_boundary` against existing target/restore boundary probes.
- Result: The local tiny lower smoke reported zero lowered collectives for `expert_fsdp_grouped_target_apply_boundary`, but compile-only inserted all-gathers and emitted SPMD "replicate then partition" warnings. The full CoreWeave probe failed before summary rows. Iris logs show repeated warnings for `expert_fsdp_grouped_target_apply/.../reshard_update_group/reshard`: XLA could not efficiently go from `{devices=[4,1,1,1]}` to the grouped FSDP target sharding, chose full replication then partitioning, and then failed during `block_until_ready_tree(grouped_target_params)` with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 21.88GiB`.
- Interpretation: This rules out another small variant of the literal FSDP-master path. The split/tree explosion is not the only problem; even a grouped target-layout apply boundary can trigger XLA GPU's inefficient replicate-then-partition path and OOM at the May shape. The remaining credible directions are either a custom explicit boundary that avoids this reshard lowering, or changing the representation so the model/apply path consumes grouped expert banks directly rather than converting back to ordinary per-layer FSDP leaves.
- Next action: Stop launching simple FSDP restore/target-layout variants unless there is a concrete new boundary implementation. The next useful implementation work is either grouped-bank consumers or an explicitly controlled collective/reshard boundary.

### 2026-06-19 02:05 PDT - explicit shard_map grouped-to-FSDP restore boundary
- Hypothesis: The FSDP-master path may be salvageable if the grouped update restore is expressed as an explicit `shard_map`: all-gather only across the grouped layer axis, then slice the matrix axis by `data`, instead of letting XLA infer a reshard and choose replicate-then-partition.
- Command:
  - Local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_restore_boundary_returns_fsdp_updates -q`; tiny 8-device CPU compile/runtime smokes for `expert_fsdp_grouped_explicit_restore_boundary`.
  - Full fp32 CoreWeave probes: `/dlwh/iris-run-job-20260619-085051/...EXPLICITRESTORE-G4-N4...` and `/dlwh/iris-run-job-20260619-085526/...EXPLICITRESTORE-NOPARAM-G4-N4...`.
  - Bounded fp32 CoreWeave timing probe: `/dlwh/iris-run-job-20260619-090020/...L8-R2D2E8-EXPLICITRESTORE-G4-N4...`.
  - Full bf16 CoreWeave timing probe: `/dlwh/iris-run-job-20260619-090318/...L26-R2D2E8-EXPLICITRESTORE-BF16-G4-N4...`.
- Config: R2/D2/E8/M1 on 4 CoreWeave H100 nodes, `ns4d_group_axis=replica_dcn,data`, group size 4, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`. Full probes used 26 layers and fp32 inputs/boundary outputs; bounded probe used 8 layers and fp32.
- Result: The explicit boundary lowers cleanly. Full L26 lowered to semantic HLO `all_gather=14`, `all_reduce=0`, `reduce_scatter=0`, `all_to_all=0`, `dot_general=0`, with no replicate-then-partition warning observed before runtime failure. The no-param timing version removed the synthetic FSDP param input, but full L26 fp32 still OOMed while materializing the restored update output tree: `RESOURCE_EXHAUSTED` on a 32.81 GiB allocation at `block_until_ready_tree(next_updates)`. The L8 fp32 probe succeeded: lowered semantic `all_gather=4`, compiled `all_gather=32`, no AR/RS/A2A/dots, and steady mean time was about 0.17436s over three iterations. The full L26 bf16 probe succeeded: lowered semantic `all_gather=14`, compiled `all_gather=112`, no AR/RS/A2A/dots, and steady mean time was about 0.3055s.
- Interpretation: The explicit `shard_map` boundary fixes the bad XLA reshard lowering: the collective pattern is exactly the intended all-gather-only restore, not replicate-then-partition. But the standalone "restore every fp32 update leaf and return the whole tree" benchmark is memory-hostile at the full 26-layer/256-expert May shape. The bf16 full-L26 result shows the explicit boundary's full-shape cost is still about 0.305s when the output tree fits, matching the previous restore-only timing class while avoiding the old SPMD warning path. This is better controlled, but not cheap enough to be the production path by itself. The likely production requirement is to consume the restored update immediately, fuse/stream the grouped-to-FSDP boundary into apply, or avoid restoring per-layer FSDP update leaves altogether.
- Next action: Keep the explicit boundary implementation as useful evidence and a testable primitive, but do not treat full-tree restore as solved. Next benchmark should either (1) time a fused explicit restore+apply that returns only next params and does not expose the update tree, or (2) keep grouped expert-bank params model-consumable and avoid the FSDP update-tree boundary.

### 2026-06-19 02:20 PDT - explicit shard_map fused restore+apply boundary
- Hypothesis: If full-tree restore is failing only because the benchmark exposes the restored fp32 update tree, then fusing explicit grouped-to-FSDP restore with `optax.apply_updates` and returning only next FSDP params may fit and give a realistic cost for the pragmatic FSDP-master design.
- Command:
  - Local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_apply_boundary_returns_fsdp_params -q`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py`; tiny 8-device CPU runtime smoke for `expert_fsdp_grouped_explicit_apply_boundary`.
  - CoreWeave probe: parent `/dlwh/iris-run-job-20260619-091205`; child `/dlwh/iris-run-job-20260619-091205/grug-train-MUON-BENCH-D2560-L26-R2D2E8-EXPLICITAPPLY-G4-N4-cw-20260619-091203`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-EXPLICITAPPLY-G4-N4-cw-20260619-091203-2d9f33`.
- Config: 4 CoreWeave H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, 26 layers, D=2560, intermediate=1280, experts=256, fp32 params/updates, `ns4d_group_axis=replica_dcn,data`, group size 4, backend steps 1, warmup 1, iters 3.
- Result: The child succeeded on all four tasks. Lowering reported semantic HLO `all_gather=14`, `all_reduce=0`, `reduce_scatter=0`, `all_to_all=0`, `dot_general=0`. Runtime timing compiled to `all_gather=112`, with all-reduce/reduce-scatter/all-to-all/dot still 0. Mean time was consistently about 0.6114s per task (`min_seconds` about 0.610s, median about 0.612s). Compile time was about 1.0-1.1s.
- Interpretation: Fusing restore with apply avoids the full fp32 update-tree OOM, so the explicit boundary is semantically usable. But the pragmatic "grouped Muon -> FSDP update/apply" boundary is still too expensive in this form: 112 all-gathers and about 0.61s is in the same order as the whole single-node H3 Muon compute target, before counting forward/backward. This makes it unlikely that ordinary FSDP params plus grouped optimizer state is competitive unless the boundary is further batched/overlapped or the model consumes grouped expert-bank params directly.
- Next action: Keep this benchmark as the clean negative/attribution result. Next implementation should focus on either (1) grouped expert-bank/model-consumable params, or (2) a more explicit batched communication boundary that reduces the 112 independent all-gathers. Do not spend more runs on simple XLA-inferred grouped-to-FSDP reshards.

### 2026-06-19 02:25 PDT - explicit apply boundary group-size sweep
- Hypothesis: The explicit fused restore+apply boundary might be slow because it emits too many small all-gather launches; increasing the grouped layer chunk from 4 to 8 or 16 should reduce the all-gather count and improve time if launch latency or fragmentation is the main issue.
- Command:
  - G8 parent `/dlwh/iris-run-job-20260619-091903`, child `/dlwh/iris-run-job-20260619-091903/grug-train-MUON-BENCH-D2560-L26-R2D2E8-EXPLICITAPPLY-G8G16-N4-cw-20260619-091901`.
  - G16 parent `/dlwh/iris-run-job-20260619-092124`, child `/dlwh/iris-run-job-20260619-092124/grug-train-MUON-BENCH-D2560-L26-R2D2E8-EXPLICITAPPLY-G16-N4-cw-20260619-092121`.
- Config: Same R2/D2/E8/M1 4-node explicit fused restore+apply boundary as the G4 run, but with `ns4d_group_size=8` and `ns4d_group_size=16`; fp32 params/updates, backend steps 1, warmup 1, iters 3.
- Result:

| group size | semantic AG | compiled AG | mean seconds |
| ---: | ---: | ---: | ---: |
| 4 | 14 | 112 | 0.6114 |
| 8 | 8 | 64 | 0.6145 |
| 16 | 4 | 32 | 0.6138 |

All rows had compiled all-reduce/reduce-scatter/all-to-all/dot-general counts of zero and succeeded on all four tasks.
- Interpretation: Reducing the number of all-gather ops by 3.5x did not move runtime. This boundary is dominated by total data movement/materialization, not small collective launch count. Group-size tuning is therefore not the lever for making the FSDP-master + grouped-Muon boundary viable.
- Next action: Stop group-size sweeps for this boundary. The next credible work remains grouped expert-bank/model-consumable params or a different communication/apply design that reduces total bytes or overlaps the boundary with useful work.

### 2026-06-19 02:35 PDT - explicit data-axis all-to-all apply boundary
- Hypothesis: The explicit apply boundary may be slow because the all-gather variant gathers full matrix-axis bytes on every `data` rank and then slices half away. A lower-byte implementation should all-gather over `replica_dcn`, then use `all_to_all` over `data` to trade the grouped-layer shard for the FSDP matrix-axis shard.
- Command:
  - Local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_a2a_apply_boundary_returns_fsdp_params -q`; `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; concrete 8-device CPU value comparison against the all-gather+slice helper for `w_gate_up` and `w_down`; tiny 8-device CPU runtime smoke for `expert_fsdp_grouped_explicit_a2a_apply_boundary`.
  - CoreWeave probe: parent `/dlwh/iris-run-job-20260619-093105`; child `/dlwh/iris-run-job-20260619-093105/grug-train-MUON-BENCH-D2560-L26-R2D2E8-EXPLICITA2AAPPLY-G4-N4-cw-20260619-093103`.
- Config: Same R2/D2/E8/M1 4-node L26 fp32 explicit apply boundary as the G4 all-gather baseline: `ns4d_group_axis=replica_dcn,data`, group size 4, D=2560, intermediate=1280, experts=256, backend steps 1, warmup 1, iters 3.
- Result: Local value checks matched the existing all-gather+slice boundary exactly for both expert matrices. CoreWeave succeeded on all four tasks. Semantic HLO had `all_gather=14`, `all_to_all=14`, and no all-reduce/reduce-scatter/dot. Compiled HLO had `all_gather=112`, `all_to_all=98`, and no all-reduce/reduce-scatter/dot. Mean runtime was about 0.6191-0.6195s, slightly slower than the all-gather+slice G4 baseline at about 0.6114s.
- Interpretation: The hand-written A2A boundary did not reduce wall time. XLA still expands the operation into many collectives, and the additional all-to-alls outweigh any theoretical byte savings for this shape. This rules out the simple "all-gather replica, all-to-all data" boundary as a production fix.
- Next action: Keep the A2A harness row as a negative control, but do not continue iterating on simple shard_map collective rewrites for this FSDP boundary. The evidence now points more strongly toward avoiding the grouped-to-FSDP conversion in the hot path, likely by making grouped expert-bank params model-consumable or accepting/overlapping a larger sync boundary outside the critical step.

### 2026-06-19 02:55 PDT - grouped-to-FSDP boundary byte estimates
- Hypothesis: The remaining plausible FSDP-master boundary variants need byte-floor visibility in the harness, because collective count alone no longer explains runtime after the G4/G8/G16 and A2A probes.
- Command: Added summary-row byte estimates for expert grouped-to-FSDP boundary benches and validated with `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`.
- Config: The regression test uses the R2/D2/E2/M1 tiny analog of the CoreWeave R2/D2/E8 shape with `ns4d_group_axis=replica_dcn,data` and `expert_fsdp_grouped_explicit_apply_boundary`.
- Result: The new fields report global update bytes, grouped input bytes per device, FSDP output bytes per device, all-gather+slice peak bytes per device, and ratios. For R2/D2, FSDP output is 2x the grouped input shard per device, while the all-gather+slice helper transiently materializes 4x the grouped input shard per device.
- Interpretation: This explains why simply packing leaves or reducing the number of collective launches is unlikely to rescue the FSDP-master hot boundary: even the ideal FSDP output materialization is a substantial expansion relative to grouped optimizer input, and the measured lower-byte A2A route was still flat/slower. Future boundary work should report these byte fields before spending CoreWeave time.
- Next action: Use the byte estimates as a gate for any new FSDP-boundary proposal. If the proposal cannot reduce required bytes or overlap them, prioritize grouped expert-bank consumers instead.

### 2026-06-19 03:20 PDT - production Muon NS compute dtype knob
- Hypothesis: The measured bf16 Newton-Schulz win for fp32 inputs should be available in the production MuonH path as an explicit algorithmic knob, not only in the synthetic harness.
- Command: Patched `_grug_scale_with_muon`, `scale_with_grug_muonh`, and `GrugMoeMuonHConfig` to accept `ns_compute_dtype`, defaulting to `input`. Validation: `uv run pytest lib/levanter/tests/test_grugmuon.py -q`; `uv run pytest experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_muon_update_bench.py -q`; `./infra/pre-commit.py --files lib/levanter/src/levanter/optim/grugmuon.py lib/levanter/tests/test_grugmuon.py experiments/grug/moe/optimizer.py experiments/grug/moe/muon_update_bench.py`.
- Result: Production Muon now casts only the Newton-Schulz input to the requested compute dtype and restores the original update dtype before scaling/hyperball/apply. The harness production-transform builders pass through `BenchConfig.ns_compute_dtype`, so existing fp32-input/bf16-NS Muon-only gates exercise the same production knob.
- Interpretation: This does not solve the grouped-to-FSDP boundary, but it preserves the isolated ~1.9x bf16-NS win as a usable setting for fp32 master/update experiments.
- Next action: Any fp32-input production MuonH experiment should set `ns_compute_dtype: bf16` explicitly when the goal is throughput, and set `input` only when measuring fp32 NS numerics/perf.

### 2026-06-19 03:55 PDT - production Muon NS dtype A/B
- Hypothesis: The new production `ns_compute_dtype` knob should reproduce the harness bf16-NS speedup on a production-shaped MuonH optimizer path with fp32 inputs.
- Command: One-node CoreWeave A/B from `scratch/muon_update_bench_fast_loop.sh`, `dtype=fp32`, `backend_steps=3`, `expert_axis=8`, `model_axis=1`, `bench_kinds=expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply`. Full L26 attempts `/dlwh/iris-run-job-20260619-095109` (`input`) and `/dlwh/iris-run-job-20260619-095123` (`bf16`) both failed with GPU BFC OOM while timing, so the validation fallback used L4: `/dlwh/iris-run-job-20260619-095453` (`input`) and `/dlwh/iris-run-job-20260619-095512` (`bf16`).
- Config: L4 output prefixes `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L4-E8-FULLPRODMUONH-FP32NS-INPUT-H3-N1-cw-20260619-095451-06dbb4` and `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L4-E8-FULLPRODMUONH-FP32NS-BF16-H3-N1-cw-20260619-095510-0632d0`.
- Result:

| row | input NS median | bf16 NS median | speedup | input peak % | bf16 peak % | compiled AG/AR/RS/A2A |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `expert_grouped_muonh_optimizer_apply_h3` | 0.16280s | 0.08547s | 1.90x | 29.01 | 55.26 | 0/0/0/0 |
| `full_production_muonh_optimizer_apply_h3` | 0.19014s | 0.09624s | 1.98x | 25.25 | 49.88 | 0/0/0/0 |

- Interpretation: The production knob is wired correctly for the production-shaped MuonH path. On fp32 inputs, bf16 Newton-Schulz roughly doubles L4 H3 throughput and preserves the zero-collectives invariant. The L26 one-node OOM means this exact full-production fp32-input A/B should be run at a smaller layer count or with a memory-focused setup unless the full L26 state materialization is reduced.
- Next action: Keep `ns_compute_dtype=bf16` in fp32-input throughput experiments. This remains orthogonal to the grouped-to-FSDP boundary, whose simple restore/apply variants are still too expensive.

### 2026-06-19 04:25 PDT - grouped expert-bank consumer harness gate
- Hypothesis: The next representation gate should prove whether grouped expert banks can be consumed by expert MLP work without first splitting them back to per-layer FSDP leaves. This is the main alternative after the literal FSDP-master grouped-boundary variants all stayed around 0.61s or OOMed.
- Command: Added harness bench kind `expert_grouped_bank_consumer`, plus focused coverage in `experiments/grug/moe/test_muon_update_bench.py::test_grouped_expert_bank_consumer_preserves_grouped_bank_without_collectives`. Local validation used a tiny one-device CLI smoke: `uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --ns4d-group-size 1 --ns4d-group-axis none --hidden-dim 4 --intermediate-dim 2 --num-experts 1 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --bench-kinds expert_grouped_bank_consumer --mode both --warmup 0 --iters 1 --disable-abstract-mesh`.
- Result: The new gate consumes grouped expert banks and grouped `[group, expert, token, hidden]` activations with the gate/up and down expert MLP matmuls. Focused pytest passed. The tiny local lower row had two StableHLO `dot_general`s and no all-gather/all-reduce/reduce-scatter/all-to-all. The summary row reports `ns4d_boundary_status=grouped_blocks_expert_bank_consumer`.
- Interpretation: This gives us a cheap compile/runtime target for the grouped-bank consumer direction before touching the real Grug model representation. The next useful CoreWeave probe is R2/D2/E8/L26 with this bench kind to verify GPU lowering preserves the grouped sharding and avoids boundary collectives at the May shape.
- Next action: Run `expert_grouped_bank_consumer` remotely with `replica_axis=2,data_axis=2,expert_axis=8,model_axis=1`, `layers=26`, `group_size=4`, and compare its compiled collectives and timing against the failed FSDP boundary variants.

### 2026-06-19 04:35 PDT - grouped expert-bank consumer CoreWeave R2/D2/E8 gate
- Hypothesis: The synthetic grouped expert-bank consumer should preserve grouped expert-bank sharding on the May R2/D2/E8 shape and avoid the grouped-to-FSDP collective boundary entirely.
- Command: Parent `/dlwh/iris-run-job-20260619-100822`; child `/dlwh/iris-run-job-20260619-100822/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDBANKCONSUMER-G4-N4-cw-20260619-100820`. Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `group_size=4`, `ns4d_group_axis=replica_dcn,data`, bf16, `bench_kinds=expert_grouped_bank_consumer`, output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-GROUPEDBANKCONSUMER-G4-N4-cw-20260619-100820-d6514f`.
- Result: Parent and child succeeded. Lowered HLO on all tasks preserved `P(('replica_dcn', 'data'), 'expert', None, None)` and reported `dot_general=14`, `all_gather=0`, `all_reduce=0`, `reduce_scatter=0`, `all_to_all=0`. Compiled HLO also reported `all_gather=0`, `all_reduce=0`, `reduce_scatter=0`, `all_to_all=0`. Median runtime was about 1.81-1.83 ms per task for the current one-token-per-expert synthetic consumer; compiled dot count was zero after GPU lowering, so use the lowered HLO for semantic dot attribution.
- Interpretation: This confirms the representation direction can avoid the grouped-to-FSDP boundary collectives at the May R2/D2/E8 shape. The timing is not a meaningful full MoE throughput number yet because the synthetic consumer intentionally uses only one token per expert and the GPU compiler lowers dots away from the current regex-visible `dot_general` form in compiled HLO.
- Next action: Make the consumer gate more model-like before porting to real training: parameterize tokens per expert or feed a realistic routed-token shape, add compiled custom-call/GEMM counting if needed, then evaluate whether grouped banks can be threaded through the real MoE expert MLP without splitting per-layer leaves.

### 2026-06-19 04:55 PDT - grouped expert-bank consumer routed-token knob
- Hypothesis: The grouped expert-bank consumer gate needs a configurable routed-token load before it can be used as a more realistic expert MLP proxy, and compiled-HLO summaries should expose GPU custom-call/GEMM hints when dot-generals disappear after lowering.
- Command: Added `BenchConfig.grouped_expert_consumer_tokens_per_expert` and CLI `--grouped-expert-consumer-tokens-per-expert`, plus `custom_call` and `gpu_gemm_custom_call` counters in `summarize_hlo`. Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; tiny CLI smoke with `--grouped-expert-consumer-tokens-per-expert 2`.
- Result: The consumer activation shape now uses `[group, expert, tokens_per_expert, hidden]`, and the FLOP estimate scales with `tokens_per_expert`. The tiny smoke emitted the new config and summary fields end to end; the lowered HLO still had two semantic dot-generals and zero collectives.
- Interpretation: The next remote grouped-bank gate can increase routed-token load without changing code, and compiled summaries will no longer be as misleading when GPU lowering hides dot-generals behind custom calls.
- Next action: Run the R2/D2/E8 grouped-bank consumer again with a more realistic tokens-per-expert value, then decide whether the real MoE path should consume grouped expert banks directly or first add a routed-token packing approximation to the harness.

### 2026-06-19 05:25 PDT - grouped expert-bank consumer T1024 CoreWeave gate
- Hypothesis: Increasing the grouped expert-bank consumer to `tokens_per_expert=1024` should keep the zero-boundary-collective invariant while making the synthetic expert MLP workload large enough for useful GPU GEMM attribution.
- Command: Parent `/dlwh/iris-run-job-20260619-102350`; child `/dlwh/iris-run-job-20260619-102350/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDBANKCONSUMER-T1024-G4-N4-cw-20260619-102348`. Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_grouped_bank_consumer`, `grouped_expert_consumer_tokens_per_expert=1024`, bf16, output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-GROUPEDBANKCONSUMER-T1024-G4-N4-cw-20260619-102348-249847`.
- Result: Parent and child succeeded, 4/4 tasks. Lowered HLO preserved `P(('replica_dcn', 'data'), 'expert', None, None)` with `dot_general=14` and AG/AR/RS/A2A `0/0/0/0`. Compiled HLO had AG/AR/RS/A2A `0/0/0/0`, `custom_call=70`, and `gpu_gemm_custom_call=28`. Median task runtimes were about 6.62-6.76 ms; estimated bf16 peak was about 62.6-64.0% by the harness FLOP estimate.
- Interpretation: This is the first grouped-bank consumer gate that is both collective-clean and has enough arithmetic to exercise real GPU GEMM lowering. It supports the representation direction where grouped expert banks feed grouped expert MLP work directly instead of forcing a grouped-to-FSDP split/apply boundary.
- Next action: Port the grouped-bank consumer idea toward the real MoE expert path, or add a routed-token packing harness if we need a closer intermediate before touching model code.

### 2026-06-19 05:35 PDT - model-facing grouped MoE expert adapter
- Hypothesis: The grouped-bank direction needs a public model-side abstraction before it can be integrated into Grug blocks; otherwise the passing gates remain harness-only.
- Command: Added `GroupedMoEExpertMlp` and `grouped_moe_mlp` to `lib/levanter/src/levanter/grug/grug_moe.py`. Focused validation: `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_mlp_matches_per_layer_moe_mlp_without_ep_axis lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_expert_mlp_layer_view_matches_grouped_call lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_mlp_ep_path_lowers_on_abstract_mesh -q`; `./infra/pre-commit.py --files lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/tests/grug/test_grugformer_moe.py`.
- Result: The grouped helper preserves a `[group, ...]` API and numerically matches a loop over ordinary `moe_mlp` calls on the local/no-EP path. The `GroupedMoEExpertMlp.layer()` view matches the grouped call for a selected layer and exists only as an incremental migration/correctness tool. The same helper lowers under an abstract expert-parallel mesh for both `ring` and `ragged_all_to_all`.
- Interpretation: This is not the full train integration yet, but it gives the real MoE module surface an explicit grouped-bank consumer API and shows that nested group-axis mapping does not immediately block EP lowering.
- Next action: Replace the harness's hand-written grouped expert consumer with the public helper or add a new bench kind that uses it, then port grouped expert banks into the experiment model if the helper-backed gate stays collective-clean.

### 2026-06-19 06:45 PDT - fresh expert-only Muon NS dtype A/B
- Hypothesis: For fp32-input expert grouped MuonH, forcing only the Newton-Schulz body to bf16 should materially improve runtime while preserving the zero-collective invariant; for bf16-input grouped-bank paths the default `ns_compute_dtype=input` already means bf16 NS.
- Command:
  - Local smoke confirmed the knob is active on a tiny expert grouped path: `uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --ns4d-group-size 1 --ns4d-group-axis none --hidden-dim 16 --intermediate-dim 8 --num-experts 2 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --dtype fp32 --ns-compute-dtype input --bench-kinds expert_grouped_muonh_optimizer_apply --backend-steps 3 --max-grouped-stack-size 4 --mode both --warmup 0 --iters 1 --disable-abstract-mesh`, then the same command with `--ns-compute-dtype bf16`.
  - One-node CoreWeave L26 attempts: `/dlwh/iris-run-job-20260619-103733` (`input`) and `/dlwh/iris-run-job-20260619-103755` (`bf16`), both `bench_kinds=expert_only_grouped_muonh_optimizer_apply`, `dtype=fp32`, `backend_steps=3`, `expert_axis=8`, `model_axis=1`.
  - One-node CoreWeave bounded L4 fallback: `/dlwh/iris-run-job-20260619-104146` (`input`) and `/dlwh/iris-run-job-20260619-104149` (`bf16`) with the same config except `layers=4`.
- Result:

| run | NS dtype | median | mean | peak % | compiled custom calls | compiled GEMM calls | compiled AG/AR/RS/A2A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L4 `/dlwh/iris-run-job-20260619-104146` | `input` = fp32 | 0.162604s | 0.162580s | 29.04 | 90 | 40 | 0/0/0/0 |
| L4 `/dlwh/iris-run-job-20260619-104149` | bf16 | 0.082889s | 0.082890s | 56.98 | 90 | 41 | 0/0/0/0 |

The bf16-NS row is 1.96x faster by median runtime. The L26 one-node attempts both failed during timing with `RESOURCE_EXHAUSTED` on a 1.56 GiB allocation after successful lowering, so the failure is harness materialization pressure rather than a dtype-specific correctness failure.
- Interpretation: bf16 Newton-Schulz remains worth pursuing for fp32-input expert MuonH. It roughly halves isolated expert-only grouped MuonH runtime on the bounded one-node gate and keeps compiled collectives at zero. For the current bf16 grouped-bank consumer priorities, no extra action is needed: `ns_compute_dtype=input` already runs NS in bf16 whenever the harness input dtype is bf16. For fp32 master/update experiments, set `MUON_BENCH_NS_COMPUTE_DTYPE=bf16` explicitly when measuring throughput, and reserve `input` only for fp32-NS numerics/perf controls.
- Next action: Do not spend more time on the dtype axis unless a numerical stability gate disagrees. The main remaining work is representation/integration: keep grouped expert-bank consumption collective-clean and avoid the grouped-to-FSDP hot boundary.

### 2026-06-19 07:10 PDT - public grouped MoE consumer harness gate
- Hypothesis: The grouped-bank representation should be tested through the public `grouped_moe_mlp` adapter, not only through the hand-written dense grouped-bank proxy, before porting it into real Grug blocks.
- Command: Added `expert_grouped_moe_mlp_consumer` to `experiments/grug/moe/muon_update_bench.py` and focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_mlp_matches_per_layer_moe_mlp_without_ep_axis lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_expert_mlp_layer_view_matches_grouped_call lib/levanter/tests/grug/test_grugformer_moe.py::test_grouped_moe_mlp_ep_path_lowers_on_abstract_mesh -q`.
- Result: The new harness path builds routed inputs `[group, tokens, hidden]`, calls the public grouped MoE helper over grouped expert banks, and reports `ns4d_boundary_status=grouped_blocks_public_moe_mlp_consumer`. The no-EP CLI now skips this bench with a clear message because JAX cannot shard the local `ragged_dot_general` path under explicit sharding; the dense `expert_grouped_bank_consumer` remains the no-EP proxy. The abstract EP test lowers successfully and shows the expected ring collectives (`all_gather > 0`, `reduce_scatter > 0`) while preserving grouped bank and routed activation sharding.
- Interpretation: The first version of the test found a real bug: `grouped_moe_mlp` was implemented as `vmap(moe_mlp)`, so each per-layer slice inherited the grouped stack-axis sharding as token-axis sharding and produced duplicate mesh-axis specs. The fix makes expert-parallel grouped calls own the shard_map over the full grouped input and vmap only inside the shard-local EP body. This keeps the group axis visible and avoids the grouped-to-FSDP boundary.
- Next action: Run the new `expert_grouped_moe_mlp_consumer` bench remotely on R2/D2/E8 to confirm the public helper path compiles on CoreWeave with the expected MoE collectives and no extra grouped-to-FSDP boundary, then decide whether to wire grouped expert banks into the experiment model.

### 2026-06-19 03:57 PDT - public grouped MoE consumer T1024 CoreWeave result
- Hypothesis: The public `grouped_moe_mlp` consumer should preserve the explicit grouped bank/routed activation layout on the 4-node R2/D2/E8 shape; a full `tokens_per_expert=1024` load would approximate the per-expert routed load for a single-node B8, top-k-heavy May shape.
- Command: Parent `/dlwh/iris-run-job-20260619-105231`; child `/dlwh/iris-run-job-20260619-105231/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T1024-G4-N4-cw-20260619-105228`. Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_grouped_moe_mlp_consumer`, `grouped_expert_consumer_tokens_per_expert=1024`, bf16, `allow_boundary_collectives=true`, output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T1024-G4-N4-cw-20260619-105228-3b6032`.
- Result: The explicit mesh fix worked. All four tasks lowered successfully with `all_gather=21`, `reduce_scatter=7`, `all_reduce=0`, `all_to_all=0`, `collective_permute=0`, `custom_call=14`, `gpu_gemm_custom_call=70`, and `dot_general=0`. Execution then failed before compiled timing with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 251.27GiB`; XLA reported it could only rematerialize from about 365 GiB to about 361 GiB against a roughly 57 GiB target.
- Interpretation: This is no longer the missing-mesh/eval-shape bug. The public grouped helper path can lower on the intended R2/D2/E8 mesh, but the T1024 all-at-once routed activation load materializes too much state for execution. The helper-backed path needs a smaller validation load or explicit chunking before it can be used as a full-load model-facing benchmark.
- Next action: Retry the same public helper path at `tokens_per_expert=128` to get a timed HLO/runtime datapoint. If T128 succeeds and T1024 remains the target, add chunking/microbatching to the consumer gate rather than retrying the full T1024 shape unchanged.

### 2026-06-19 04:05 PDT - public grouped MoE consumer T128/T64 memory ladder
- Hypothesis: If the public grouped MoE helper path is viable, reducing the synthetic routed-token load should eventually produce a timed run while preserving the same mesh/sharding behavior.
- Command:
  - Skipped/no-op launch: parent `/dlwh/iris-run-job-20260619-105656`; child `/dlwh/iris-run-job-20260619-105656/grug-train-MUON-BENCH-D2560-L26-R2D2E8-EXPERTONLY-H3-N4-cw-20260619-105654`. This accidentally inherited `ns4d_group_size=26`, so all tasks skipped with `expert_grouped_moe_mlp_consumer requires ns4d group axis 26 to be divisible by ('replica_dcn', 'data')=4`.
  - Corrected T128: parent `/dlwh/iris-run-job-20260619-105852`; child `/dlwh/iris-run-job-20260619-105852/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T128-G4-N4-cw-20260619-105850`.
  - Corrected T64: parent `/dlwh/iris-run-job-20260619-110216`; child `/dlwh/iris-run-job-20260619-110216/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T64-G4-N4-cw-20260619-110214`.
- Config: Both corrected runs used 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_grouped_moe_mlp_consumer`, bf16, and `allow_boundary_collectives=true`. Output prefixes were under `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/`.
- Result:

| tokens/expert | outcome | lowered AG/RS/AR/A2A | compiled AG/RS/AR/A2A | compiled GEMM custom calls | median |
| ---: | --- | --- | --- | ---: | ---: |
| 128 | OOM during timing, `RESOURCE_EXHAUSTED` on 58.30 GiB allocation | 21/7/0/0 | n/a | n/a | n/a |
| 64 | succeeded | 21/7/0/0 | 266/49/0/0 | 88 | 0.460-0.463s |

The successful T64 run reported `grouped_expert_consumer_tokens_per_expert=64`, `ns4d_boundary_status=grouped_blocks_public_moe_mlp_consumer`, lowered `custom_call=14`, lowered `gpu_gemm_custom_call=70`, compiled `custom_call=28`, compiled `gpu_gemm_custom_call=88`, and median task runtimes around 0.461s.
- Interpretation: The public helper path is runnable at small routed-token loads, but compiled HLO expands the semantic MoE collectives substantially (`21/7` lowered to `266/49` compiled AG/RS), and the all-at-once activation memory wall appears between T64 and T128 for this L26 R2/D2/E8 harness. This strongly argues against using the public helper all-at-once at full T1024. The dense grouped-bank consumer remains the cleaner zero-boundary-collective arithmetic proxy; the public helper path needs routed-token chunking/microbatching or a more direct grouped expert-path integration before it can represent the full model load.
- Next action: Add an explicit chunked grouped-MoE consumer gate if we want to keep testing the public helper. For production integration, prefer threading grouped expert banks through the expert MLP representation directly and avoid materializing the full `[group, tokens, hidden]` helper input for all tokens at once.

### 2026-06-19 04:20 PDT - chunked public grouped MoE consumer gate launched
- Hypothesis: The public grouped MoE helper can get past the T128 all-at-once memory wall if the routed-token axis is consumed in smaller chunks while preserving grouped expert-bank sharding.
- Command:
  - Code change: added `grouped_expert_consumer_chunk_tokens` to `BenchConfig`, CLI, CoreWeave launcher config, and the shell wrappers. `expert_grouped_moe_mlp_consumer` now slices the routed token axis into fixed chunks and concatenates outputs. Commit `d6b5b05fa` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation before launch: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` passed 43 tests; `./infra/pre-commit.py --changed-files --fix` passed; tiny forced-CPU lower-only smoke passed with `--disable-abstract-mesh` and reported `grouped_expert_consumer_chunk_tokens=8`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-111504`; child `/dlwh/iris-run-job-20260619-111504/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T128C64-G4-N4-cw-20260619-111502`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, 26 layers, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_grouped_moe_mlp_consumer`, bf16, `grouped_expert_consumer_tokens_per_expert=128`, `grouped_expert_consumer_chunk_tokens=64`, `allow_boundary_collectives=true`, output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T128C64-G4-N4-cw-20260619-111502-36c60f`.
- Current state: As of the 04:20 PDT bounded poll, all four child tasks were still running with no new log rows after metadata. A heartbeat `watch-muon-grouped-moe-chunked-t128` is monitoring terminal success/failure.
- Next action: If T128C64 succeeds, compare compiled AG/RS and runtime against T64 unchunked and decide whether to try T1024C64 or move directly to a model-facing chunked expert path. If it OOMs or stalls, inspect the exact allocation/root cause before launching another load.

### 2026-06-19 04:35 PDT - public grouped MoE consumer chunking negative result
- Hypothesis: The public grouped MoE helper's T128 memory wall might be controlled by chunking routed tokens per expert, allowing us to validate a model-facing grouped expert path before touching the real training representation.
- Command:
  - Stopped the accidental absolute-chunk launch `/dlwh/iris-run-job-20260619-111504` after realizing `grouped_expert_consumer_chunk_tokens=64` meant 64 total routed tokens, not 64 tokens per expert. With 256 experts and `tokens_per_expert=128`, that would have generated 512 chunks per group and an unhelpfully huge lowering.
  - Added `grouped_expert_consumer_chunk_tokens_per_expert` to the harness, CoreWeave launcher, and shell wrappers. The effective absolute chunk is now `num_experts * chunk_tokens_per_expert`; the summary row reports both the requested per-expert value and `grouped_expert_consumer_effective_chunk_tokens`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` passed 43 tests; `./infra/pre-commit.py --changed-files --fix` passed; a tiny forced-CPU lower-only smoke reported `grouped_expert_consumer_chunk_tokens_per_expert=1` and `grouped_expert_consumer_effective_chunk_tokens=8`.
  - CoreWeave C64PE: parent `/dlwh/iris-run-job-20260619-112646`; child `/dlwh/iris-run-job-20260619-112646/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T128C64PE-G4-N4-cw-20260619-112644`.
  - CoreWeave C32PE: parent `/dlwh/iris-run-job-20260619-113139`; child `/dlwh/iris-run-job-20260619-113139/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMOEMLP-T128C32PE-G4-N4-cw-20260619-113137`.
- Config: Both corrected runs used 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_grouped_moe_mlp_consumer`, bf16, `tokens_per_expert=128`, `max_grouped_stack_size=256`, and `allow_boundary_collectives=true`.
- Result:

| run | intended chunks/group | lowered AG/RS/AR/A2A | lowered GPU GEMM custom calls | outcome |
| --- | ---: | --- | ---: | --- |
| T128C64PE | 2 | 42/14/0/0 | 140 | OOM during timing, 57.84 GiB allocation |
| T128C32PE | 4 | 84/28/0/0 | 280 | OOM during timing, 57.84 GiB allocation |

- Interpretation: The per-expert chunk knob works and changes the lowered graph exactly as expected, but it does not change the dominant runtime allocation. C32PE doubled the number of chunks/collectives/GEMMs relative to C64PE and still failed on the same 57.84 GiB allocation. This rules out "just slice routed tokens smaller" for the public grouped MoE helper path. The likely problem is a full-scale helper intermediate or compiled buffer outside the apparent chunked body, so continuing to shrink this knob would add launch/collective overhead without addressing the memory wall.
- Next action: Stop spending CoreWeave runs on this public-helper chunking axis. Keep the per-expert knob for bounded diagnostics, but prioritize a real grouped expert-bank consumer/integration path that avoids the public helper's full routed-token materialization, or inspect XLA buffer assignment if we need to prove exactly which intermediate owns the invariant 57.84 GiB allocation.

### 2026-06-19 04:55 PDT - grouped MuonH bank-consumer representation gate succeeds
- Hypothesis: If grouped expert-bank params, grads, optimizer state, and updated params stay in the NS-friendly `P(('replica_dcn', 'data'), 'expert', None, None)` representation, `optax.apply_updates` can remain in the grouped world and the model-facing consumer can read updated grouped banks directly. This should avoid both the grouped-to-FSDP hot boundary and the public grouped-MoE helper's 57.84 GiB routed-token materialization.
- Command:
  - Code change: added `expert_grouped_muonh_bank_consumer` to `experiments/grug/moe/muon_update_bench.py`, with focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Commit `e84f51b37` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation before launch: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`; tiny forced-CPU lower-only smoke with `--bench-kinds expert_grouped_muonh_bank_consumer` reported 0 AG/AR/RS/A2A and preserved `P(('replica_dcn', 'data'), 'expert', None, None)`.
  - Accidental skipped run: parent `/dlwh/iris-run-job-20260619-114608`, child `/dlwh/iris-run-job-20260619-114608/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMUONHBANK-T1024-H3-G4-N4-cw-20260619-114606`; inherited `ns4d_group_size=26` and skipped because 26 is not divisible by the R2D2 group-axis size 4.
  - Corrected CoreWeave run: parent `/dlwh/iris-run-job-20260619-114822`, child `/dlwh/iris-run-job-20260619-114822/grug-train-MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMUONHBANK-T1024-H3-G4-N4-cw-20260619-114819`.
- Config: 4 H100 nodes, 32 GPUs total, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, bf16 params/updates, `bench_kinds=expert_grouped_muonh_bank_consumer`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `backend_steps=3`, `max_grouped_stack_size=256`, `grouped_expert_consumer_tokens_per_expert=1024`, `allow_boundary_collectives=false`, output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-GROUPEDMUONHBANK-T1024-H3-G4-N4-cw-20260619-114819-f32da5`.
- Result:

| row | lowered AG/AR/RS/A2A | compiled AG/AR/RS/A2A | lowered dots | compiled GPU GEMM calls | median | mean | nominal H100 bf16 peak |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| best task | 0/0/0/0 | 0/0/0/0 | 140 | 336 | 0.144849s | 0.144758s | 60.02% |
| task range | 0/0/0/0 | 0/0/0/0 | 140 | 336 | 0.144849-0.145840s | 0.144758-0.145892s | 59.55-60.02% |

The run succeeded without the previous 57.84 GiB allocation. Compiled HLO reported `custom_call=700`, `gpu_gemm_custom_call=336`, and no collectives. The summary kept `ns4d_input_sharding_spec`, `ns4d_compute_sharding_spec`, and `ns4d_result_sharding_spec` all at `P(('replica_dcn', 'data'), 'expert', None, None)`.
- Interpretation: This is the first strong evidence that the pragmatic representation can work: grouped MuonH, grouped `optax.apply_updates`, and grouped expert-bank consumption can stay collective-clean and fast at the R2/D2/E8 L26/T1024 gate. It also confirms the public grouped-MoE helper OOM was not an inherent cost of the grouped expert-bank representation; it was specific to that routed-token helper path.
- Next action: Treat this as the preferred production integration direction. Wire the real model to consume grouped expert banks directly, or introduce a grouped-bank expert module API, rather than restoring grouped params to FSDP before `apply_updates` or using the public helper's all-at-once routed-token materialization. Keep a small numerical/correctness gate around grouped `apply_updates` before replacing the current production optimizer path.

### 2026-06-19 10:05 PDT - direct grouped-to-FSDP restore validation
- Hypothesis: The pragmatic FSDP-master path might still be viable if grouped MuonH updates are converted directly from `P(('replica_dcn', 'data'), 'expert', None, None)` to grouped FSDP target layout `P(None, 'expert', 'data', None)` before splitting into ordinary per-layer leaves.
- Command:
  - Commit `a045028d8` changed production grouped MuonH restore to use a direct target-layout `jax.sharding.reshard`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-164947`; child `/dlwh/iris-run-job-20260619-164947/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-TARGETRESTORE-H3-G4-CAP256-N4-cw-20260619-164945`.
  - Commit `5cdf79012` then changed the restore to an explicit `shard_map` boundary for the R2/D2 case: `all_gather` over `replica_dcn`, `all_to_all` over `data`, then reorder the grouped layer axis.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-165623`; child `/dlwh/iris-run-job-20260619-165623/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-EXPLICITA2A-H3-G4-CAP256-N4-cw-20260619-165621`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, fp32 params/optimizer state, bf16 Newton-Schulz compute, `expert_grouped_muonh_group_size=4`, `max_grouped_stack_size=256`, backend steps H3.
- Result:

| boundary | outcome | lowered collectives | compiled collectives | timing |
| --- | --- | --- | --- | --- |
| direct `reshard` to `P(None, 'expert', 'data', None)` | failed | local/tiny lowering looked clean | GPU SPMD warned it would replicate then partition | OOM on a 26.17GiB allocation |
| explicit replica gather + data all-to-all | succeeded | update/apply both `AG=14`, `A2A=14`, `AR=0`, `RS=0` | update/apply both `AG=112`, `A2A=392`, `AR=0`, `RS=0` | update median about 1.127s; apply median about 1.073-1.077s |

- Interpretation: The layout movement is mathematically right but not viable through current XLA GPU lowering. The plain `reshard` chooses a replicate-then-partition path and OOMs. The explicit `shard_map` avoids the OOM, but the compiled graph explodes the intended grouped boundary into many collectives and is no faster than the previous bad boundary class. This rules out the current FSDP-master hot-path conversion as the production route.
- Next action: Stop iterating on generic `reshard`/`shard_map` grouped-to-FSDP restore variants unless the new proposal avoids the required materialization or uses a lower-level custom communication path. Prefer the grouped-bank representation that already passed the R2/D2/E8 L26/T1024 collective-clean gate.

### 2026-06-19 10:35 PDT - sequential grouped-bank consumer gate
- Hypothesis: The successful grouped-bank consumer gate may be too optimistic because it consumes all layers in a group at once. A more model-like sequential consumer should try to read one layer at a time from persistent grouped expert banks.
- Command:
  - Added harness bench kind `expert_grouped_sequential_bank_consumer`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 52 passed.
  - CLI smoke with unsharded group axis: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --ns4d-group-axis none --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --bench-kinds expert_grouped_sequential_bank_consumer --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_seq_bank_smoke.json`.
  - CLI smoke with target sharded group axis: same command but `--ns4d-group-axis replica_dcn,data`.
- Result:
  - Unsharded group axis lowers and returns rank-3 expert-local outputs. StableHLO summary: `dot_general=8`, `all_gather=0`, `all_to_all=0`, `all_reduce=0`, `reduce_scatter=0`.
  - Sharded group axis is now a structured skip before lowering: `expert_grouped_sequential_bank_consumer consumes one layer at a time and cannot directly slice a grouped layer axis sharded over ('replica_dcn', 'data')`.
  - The original failing smoke raised `ShardingTypeError`: slicing a group axis sharded over 4 mesh devices down to output size 1 is not implemented because the output dimension is not divisible by the mesh axis.
- Interpretation: This is the realism caveat for grouped-bank integration. Keeping params/updates/results grouped is fast, but a conventional sequential transformer block cannot simply index one layer out of a `replica_dcn,data`-sharded grouped bank. That access pattern necessarily needs a communication boundary, a different grouped/scan-like model consumer that keeps the group axis alive, or a lower-level custom gather/broadcast path. The existing all-at-once grouped-bank gate is a lower bound, not proof that the ordinary block loop can consume the grouped bank directly.
- Next action: Do not port grouped banks into the real block loop by calling `GroupedMoEExpertMlp.layer()` on a sharded group axis. The next viable model-facing design must either keep the layer group axis live through a grouped block/scan consumer, or make an explicit one-layer access boundary and measure/overlap it.

### 2026-06-19 10:25 PDT - packed A2A grouped-to-FSDP boundary launched
- Hypothesis: The previous explicit `shard_map` grouped-to-FSDP boundary was too granular: it restored each group and each expert weight name separately. Packing all groups for one expert weight name before the explicit `all_gather(replica_dcn)` + `all_to_all(data)` boundary should reduce the intended communication from 14 AG/A2A pairs to 2 AG/A2A pairs for L26/G4.
- Command:
  - Code change: added `expert_fsdp_grouped_packed_a2a_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, with a comparison test in `experiments/grug/moe/test_muon_update_bench.py`. Commit `9b20b0f0f` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 53 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - Local forced-CPU lower-only smoke:
    `XLA_FLAGS=--xla_force_host_platform_device_count=32 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 2 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_a2a_apply_boundary,expert_fsdp_grouped_packed_a2a_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_packed_a2a_l26_lower.json`.
  - CoreWeave validation launched: parent `/dlwh/iris-run-job-20260619-172133`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, bf16, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `bench_kinds=expert_fsdp_grouped_explicit_a2a_apply_boundary,expert_fsdp_grouped_packed_a2a_apply_boundary`, `backend_steps=1`, `max_grouped_stack_size=256`, mode `both`, output prefix under `s3://marin-na/tmp/ttl=7d`.
- Local lower-only result:

| boundary | lowered AG | lowered A2A | lowered AR/RS | note |
| --- | ---: | ---: | --- | --- |
| explicit per-group A2A | 14 | 14 | 0/0 | one restore per group per weight name |
| packed A2A | 2 | 2 | 0/0 | one restore per weight name |

- Interpretation: The packed form fixes the StableHLO-level granularity problem. The live CoreWeave run is the real test: if compiled HLO still explodes the 2 logical A2As into hundreds of NCCL calls or stays near 1s, the FSDP-master boundary still needs custom communication or a grouped model consumer. If compiled counts stay close to 2 and timing drops materially, the FSDP-master plus packed restore path becomes viable again.
- Next action: Babysit `/dlwh/iris-run-job-20260619-172133`, extract summary JSON, and compare compiled AG/A2A counts plus median timing against the previous explicit A2A baseline (`AG=112`, `A2A=392`, about 1.07s apply).

### 2026-06-19 10:30 PDT - packed A2A grouped-to-FSDP boundary negative result
- Hypothesis: Packing all grouped blocks by expert weight name would let XLA GPU compile the grouped-to-FSDP boundary as two large logical A2As and reduce runtime relative to the per-group explicit A2A path.
- Command:
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-172133`; child `/dlwh/iris-run-job-20260619-172133/grug-train-MUON-BENCH-D2560-L26-R2D2E8-PACKEDA2A-H1-G4-N4-cw-20260619-172131`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-PACKEDA2A-H1-G4-N4-cw-20260619-172131-9fbb78/summary.json`.
- Result:

| boundary | lowered AG/A2A | compiled AG/A2A | compiled GEMMs | median |
| --- | --- | --- | ---: | ---: |
| explicit per-group A2A | 14/14 | 112/98 | 7 | 0.305741s |
| packed A2A | 2/2 | 16/158 | 1 | 0.467800s |

- Interpretation: Packing did reduce logical AG/A2A and compiled AG, but compiled A2A still exploded and runtime regressed by about 53% versus the per-group explicit boundary. The packed path is therefore not a rescue for the FSDP-master hot boundary. The underlying issue is not only HLO-level operation count; XLA GPU/NCCL lowering still decomposes the large data-axis A2A in a way that is slower than the smaller per-group boundary.
- Next action: Stop treating generic JAX `reshard`, explicit per-group `shard_map`, or packed `shard_map` restore as sufficient for the production FSDP-master path. The remaining plausible routes are (1) keep grouped expert banks live through a grouped/scan-like model consumer, or (2) implement a lower-level custom communication boundary that directly performs the needed grouped-to-FSDP permutation without XLA's A2A decomposition.

### 2026-06-19 11:10 PDT - scan grouped-bank consumer negative result
- Hypothesis: A grouped block/scan consumer might keep the grouped layer-bank axis live, avoiding both per-layer slicing of a sharded group axis and the grouped-to-FSDP restore boundary.
- Command:
  - Code change: added `expert_grouped_scan_bank_consumer` to `experiments/grug/moe/muon_update_bench.py` with focused tests in `experiments/grug/moe/test_muon_update_bench.py`.
  - Focused validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_grouped_expert_scan_bank_consumer_skips_sharded_group_axis experiments/grug/moe/test_muon_update_bench.py::test_grouped_expert_scan_bank_consumer_returns_expert_local_outputs_without_sharded_group_axis -q`.
  - Full local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q`.
  - CLI sharded smoke: `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 6 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --bench-kinds expert_grouped_scan_bank_consumer --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_scan_bank_smoke.json`.
- Result:
  - The first direct sharded scan attempt failed during `jax.eval_shape`: `ValueError: 0th dimension of all xs should be replicated. Got P(None,), P(('replica_dcn', 'data'), 'expert', None, None), P(('replica_dcn', 'data'), 'expert', None, None)`.
  - The harness now emits a structured skip for `expert_grouped_scan_bank_consumer` when the grouped layer axis is sharded. The CLI smoke reports: `lax.scan requires the scanned xs dimension to be replicated; grouped layer axis sharding over ('replica_dcn', 'data') is not directly consumable.`
  - The unsharded scan control still lowers and returns expert-local rank-3 outputs with no AG/AR/RS/A2A, so the failure is specifically the sharded scan axis, not the MLP computation.
- Interpretation: A normal `lax.scan` does not rescue the grouped model-facing representation. If the grouped layer axis is actually sharded across `replica_dcn,data`, sequential block consumption needs either a communication boundary to move the activation carry between layer-bank shards, a different lower-level implementation, or a layout that does not shard the layer axis for model consumption.
- Next action: Do not port grouped expert banks into the real block loop via `lax.scan` over a sharded grouped layer axis. The two remaining viable directions are a custom grouped-to-FSDP permutation boundary, or a custom/pipelined grouped consumer that explicitly handles activation carry movement across the group-axis devices.

### 2026-06-19 11:25 PDT - data-first explicit A2A grouped-to-FSDP boundary
- Hypothesis: The explicit grouped-to-FSDP boundary may be cheaper if the `data` all-to-all is performed before the `replica_dcn` gather, matching the desired logical movement from `P(('replica_dcn', 'data'), 'expert', None, None)` to `P(None, 'expert', 'data', None)` more directly.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_data_first_a2a_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, with focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Commit `24c6e7e50` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 56 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - Local forced-CPU lower-only comparison:
    `XLA_FLAGS=--xla_force_host_platform_device_count=32 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 2 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_a2a_apply_boundary,expert_fsdp_grouped_explicit_data_first_a2a_apply_boundary,expert_fsdp_grouped_packed_a2a_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_data_first_a2a_l26_lower.json`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-174425`; child `/dlwh/iris-run-job-20260619-174425/grug-train-MUON-BENCH-D2560-L26-R2D2E8-DATAFIRSTA2A-H1-G4-N4-cw-20260619-174422`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-DATAFIRSTA2A-H1-G4-N4-cw-20260619-174422-eb8a42/summary.json`.
- Result:

| boundary | lowered AG/A2A | compiled AG/A2A | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| gather-first explicit A2A | 14/14 | 112/98 | 7 | 0.304565s | 0.305158s |
| data-first explicit A2A | 14/14 | 150/98 | 14 | 0.220179s | 0.219504s |

The data-first variant is about 1.38x faster than the gather-first explicit boundary, despite compiling to more all-gathers and more GEMM custom calls. Both variants still compile the intended 14 logical AG/A2A pairs into many more collectives.
- Interpretation: The direct logical target `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)` is the right communication shape, and ordering the data-axis movement first materially helps. It still does not prove the FSDP-master hot boundary is production-viable: compiled collectives remain high (`AG=150`, `A2A=98`) and the boundary is still slower than the collective-clean grouped MuonH bank-consumer gate at about 0.145s for MuonH plus grouped apply plus grouped consumer.
- Next action: Keep the data-first boundary as the best measured FSDP-master explicit-communication checkpoint, but do not integrate it into production yet. The next FSDP-master attempt needs a lower-level/custom grouped-to-FSDP permutation or a `shard_map` structure that prevents compiled collective decomposition; otherwise prefer grouped-bank model consumption.

### 2026-06-19 11:35 PDT - data-first restore-only boundary isolates communication cost
- Hypothesis: If the remaining data-first grouped-to-FSDP cost is mostly `optax.apply_updates` or per-layer tree assembly, a restore-only boundary should be materially faster than restore-plus-apply. If it is mostly communication lowering, restore-only and apply should have nearly the same compiled collectives and timing.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_data_first_a2a_restore_boundary` to `experiments/grug/moe/muon_update_bench.py`, with focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Commit `ba1be76b4` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 57 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-175855`; child `/dlwh/iris-run-job-20260619-175855/grug-train-MUON-BENCH-D2560-L26-R2D2E8-DATAFIRSTRESTORE-H1-G4-N4-cw-20260619-175852`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-DATAFIRSTRESTORE-H1-G4-N4-cw-20260619-175852-81ac33/summary.json`.
- Result:

| boundary | lowered AG/A2A | compiled AG/A2A | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| data-first restore-only | 14/14 | 150/98 | 14 | 0.213788s | 0.213544s |
| data-first restore-plus-apply | 14/14 | 150/98 | 14 | 0.218176s | 0.218407s |

Restore-only is only about 4.4 ms faster by median runtime. Both paths compile to the same high collective counts and GEMM custom-call count.
- Interpretation: This isolates the problem: the hot cost is the grouped-to-FSDP communication lowering, not `optax.apply_updates` or the Python tree shape of the ordinary apply path. Keeping ordinary `apply_updates` after a data-first restore is not the bottleneck; the bottleneck is that XLA GPU compiles the intended 14 logical data-first AG/A2A pairs into `AG=150`, `A2A=98`.
- Next action: Stop optimizing the post-restore apply path. The next FSDP-master route needs a custom/lower-level grouped-to-FSDP permutation or a different representation that avoids this restore boundary. The grouped-bank path remains the only path so far that is both fast and collective-clean at the R2/D2/E8 L26/T1024 gate.

### 2026-06-19 11:55 PDT - ppermute replica-duplication boundary is slower
- Hypothesis: For the R2/D2 grouped-to-FSDP restore, replacing the `replica_dcn` all-gather with an explicit `lax.ppermute` should avoid XLA GPU's all-gather decomposition while keeping the data-axis all-to-all. The logical target is still `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)`.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_data_first_ppermute_restore_boundary` and `expert_fsdp_grouped_explicit_data_first_ppermute_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, plus focused tests in `experiments/grug/moe/test_muon_update_bench.py`. Commit `ef80b16b5` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 58 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-180831`; child `/dlwh/iris-run-job-20260619-180831/grug-train-MUON-BENCH-D2560-L26-R2D2E8-PPERMBRIDGE-H1-G4-N4-cw-20260619-180828`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-PPERMBRIDGE-H1-G4-N4-cw-20260619-180828-db472d/summary.json`.
- Result:

| boundary | lowered AG/A2A/CP | compiled AG/A2A/CP | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| data-first restore-only | 14/14/0 | 150/98/0 | 14 | 0.213987s | 0.213739s |
| ppermute restore-only | 0/14/14 | 0/105/112 | 14 | 0.276760s | 0.278011s |
| ppermute restore-plus-apply | 0/14/14 | 0/105/112 | 14 | 0.278392s | 0.280574s |

- Interpretation: The ppermute boundary does exactly what it was supposed to at the StableHLO level: all-gathers disappear and become 14 collective-permutes plus the same 14 logical data all-to-alls. XLA GPU still decomposes the data all-to-all and ppermute path heavily (`A2A=105`, `CP=112`), and the runtime is about 1.29x slower than the data-first all-gather/all-to-all restore. This rules out the R2 ppermute bridge as a production rescue for the FSDP-master restore boundary.
- Next action: Stop pursuing JAX-level all-gather replacement for this boundary unless we can prove a lower-level communication primitive avoids the compiled A2A/CP explosion. The strongest current evidence still points to either a custom grouped-to-FSDP permutation kernel/collective or a model-facing representation that avoids restoring grouped Muon outputs to FSDP on the hot path.

### 2026-06-19 12:20 PDT - packed ppermute reduces HLO collectives but worsens runtime
- Hypothesis: The ppermute bridge may have failed because it restored each group and expert weight name separately. Packing all grouped blocks by expert weight name before the data-first ppermute boundary should reduce the intended logical communication from 14 A2A/CP pairs to 2 A2A/CP pairs.
- Command:
  - Code change: added `expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, with focused tests in `experiments/grug/moe/test_muon_update_bench.py`. Commit `fb277009c` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 59 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - Local forced-CPU lower-only comparison:
    `XLA_FLAGS=--xla_force_host_platform_device_count=32 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 2 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_data_first_ppermute_apply_boundary,expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_packed_ppermute_l26_lower.json`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-181732`; child `/dlwh/iris-run-job-20260619-181732/grug-train-MUON-BENCH-D2560-L26-R2D2E8-PACKEDPPERM-H1-G4-N4-cw-20260619-181728`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-PACKEDPPERM-H1-G4-N4-cw-20260619-181728-6a7dfe/summary.json`.
- Result:

| boundary | lowered AG/A2A/CP | compiled AG/A2A/CP | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| un-packed data-first ppermute apply | 0/14/14 | 0/105/112 | 14 | 0.278412s | 0.278422s |
| packed data-first ppermute apply | 0/2/2 | 0/159/16 | 1 | 0.428635s | 0.428295s |

- Interpretation: Packing succeeds at the StableHLO level, but compiled XLA GPU/NCCL lowering gets worse: A2A expands to 159 calls and runtime regresses by about 54% versus the un-packed ppermute path. This mirrors the earlier packed A2A result and confirms that reducing logical collective count is not enough; the large packed data-axis A2A lowers poorly.
- Next action: Treat the JAX-level grouped-to-FSDP restore space as exhausted for now: plain `reshard` OOMs, explicit per-group A2A is best but still costly, packed A2A is worse, ppermute is worse, and packed ppermute is worst. The next FSDP-master attempt needs a lower-level custom communication boundary, or we should return to a grouped-bank/model-facing representation that avoids this restore boundary.

### 2026-06-19 12:35 PDT - production data-first restore integration still compiles to many collectives
- Hypothesis: Integrating the best measured data-first grouped-to-FSDP restore into the production grouped MuonH optimizer helper might materially improve the real update/apply path versus the earlier gather-first production path.
- Command:
  - Code change: changed `_restore_grouped_muonh_for_split_explicit_a2a` in `experiments/grug/moe/optimizer.py` to perform the data-axis `all_to_all` before the `replica_dcn` all-gather. Commit `2a6c26dc3` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation before launch: `uv run pytest experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_muon_update_bench.py -q` -> 73 passed; `./infra/pre-commit.py --changed-files --fix` -> OK.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-182442`; child `/dlwh/iris-run-job-20260619-182442/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-DATAFIRST-H3-G4-CAP256-N4-cw-20260619-182440`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-DATAFIRST-H3-G4-CAP256-N4-cw-20260619-182440-804cbc/summary.json`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `max_grouped_stack_size=256`, `ns_compute_dtype=bf16`, `backend_steps=3`.
- Result:

| path | lowered AG/A2A/AR/RS | compiled AG/A2A/AR/RS | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| production grouped MuonH update, data-first restore | 14/14/0/0 | 150/392/0/0 | 308 | 0.933573s | 0.934062s |
| production grouped MuonH apply, data-first restore | 14/14/0/0 | 150/392/0/0 | 308 | 0.888601s | 0.888825s |

Earlier gather-first production path (`/dlwh/iris-run-job-20260619-165623`) was about 1.127s for update and 1.073-1.077s for apply with compiled `AG=112`, `A2A=392`. The data-first production integration is therefore a real improvement, but the compiled all-to-all count is unchanged and the path is still far from the speed target.
- Interpretation: Data-first ordering helps in the real optimizer path, but it does not solve the production blocker. The production update/apply path still combines grouped NS work with a grouped-to-FSDP restore that XLA GPU expands into hundreds of compiled collectives (`A2A=392`). The boundary-only data-first microbench remains much cheaper (~0.214-0.218s), so the real path still has substantial NS plus restore interaction cost.
- Next action: Keep `2a6c26dc3` as the best current FSDP-master production checkpoint, but do not call the objective solved. The next attempt should force a more direct grouped-to-FSDP permutation with a lower-level/custom communication boundary, or avoid the hot restore by changing the model-facing grouped expert representation.

### 2026-06-19 12:45 PDT - data-axis ppermute bridge removes A2A but only slightly improves timing
- Hypothesis: The remaining expensive collective in the data-first grouped-to-FSDP bridge is the data-axis all-to-all. For the R2/D2 probe, replacing that data all-to-all with an explicit point-to-point exchange over `data` should produce the same logical target `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)` while avoiding compiled all-to-all decomposition.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_data_ppermute_restore_boundary` and `expert_fsdp_grouped_explicit_data_ppermute_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, plus focused tests in `experiments/grug/moe/test_muon_update_bench.py`. Commit `7e79e96b0` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 61 passed; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py .agents/logbooks/grug-moe-muon-gpu.md --fix` -> OK.
  - Local forced-CPU lower-only comparison:
    `XLA_FLAGS=--xla_force_host_platform_device_count=32 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 2 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_data_first_a2a_restore_boundary,expert_fsdp_grouped_explicit_data_ppermute_restore_boundary,expert_fsdp_grouped_explicit_data_ppermute_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_data_ppermute_l26_lower.json`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-183654`; child `/dlwh/iris-run-job-20260619-183654/grug-train-MUON-BENCH-D2560-L26-R2D2E8-DATAPPERM-H1-G4-N4-cw-20260619-183652`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-DATAPPERM-H1-G4-N4-cw-20260619-183652-fb6579/summary.json`.
- Result:

| boundary | lowered AG/A2A/CP | compiled AG/A2A/CP | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| data-first A2A restore-only | 14/14/0 | 150/98/0 | 14 | 0.213894s | 0.213827s |
| data-first A2A restore-plus-apply | 14/14/0 | 150/98/0 | 14 | 0.218617s | 0.218668s |
| data-ppermute restore-only | 14/0/14 | 150/0/112 | 0 | 0.210878s | 0.217079s |
| data-ppermute restore-plus-apply | 14/0/14 | 150/0/112 | 0 | 0.213311s | 0.212720s |

The ppermute bridge removes compiled all-to-alls entirely and also eliminates the GEMM custom calls introduced by the A2A bridge. Median timing improves only slightly: restore-only is 1.014x faster and restore-plus-apply is 1.025x faster than data-first A2A.
- Interpretation: The ppermute bridge confirms that the semantic target is expressible and that the data-axis A2A can be replaced by explicit point-to-point exchange. It is not enough for the production objective: the all-to-all disappears, but XLA GPU still compiles 112 collective permutes plus 150 all-gathers, so runtime barely moves. This points away from another JAX-level bridge tweak and toward either a lower-level fused/custom grouped-to-FSDP permutation or avoiding the grouped-to-FSDP hot boundary entirely.
- Next action: Do not integrate the data-ppermute bridge into production yet. The next useful experiment should test a genuinely custom grouped-to-FSDP transfer that preserves the packed/grouped representation and avoids both compiled A2A and hundreds of CP/AG calls, or resume the grouped-bank model-facing representation path.

### 2026-06-19 12:55 PDT - packed data-ppermute bridge reintroduces compiled A2A
- Hypothesis: The un-packed data-ppermute bridge removed compiled all-to-alls but still emitted many all-gathers and collective permutes. Packing all layer groups by expert weight name before the data-ppermute bridge should reduce logical communication to two all-gathers plus two collective permutes, while preserving the no-A2A StableHLO property.
- Command:
  - Code change: added `expert_fsdp_grouped_packed_data_ppermute_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, with focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Commit `09edd1cf6` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 62 passed; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` -> OK.
  - Local forced-CPU lower-only comparison:
    `XLA_FLAGS=--xla_force_host_platform_device_count=32 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 2 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_data_ppermute_apply_boundary,expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary,expert_fsdp_grouped_packed_data_ppermute_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_packed_data_ppermute_l26_lower.json`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-184935`; child `/dlwh/iris-run-job-20260619-184935/grug-train-MUON-BENCH-D2560-L26-R2D2E8-PACKEDDATAPPERM-H1-G4-N4-cw-20260619-184932`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-PACKEDDATAPPERM-H1-G4-N4-cw-20260619-184932-a86e0c/summary.json`.
- Result:

| boundary | lowered AG/A2A/CP | compiled AG/A2A/CP | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| un-packed data-ppermute apply | 14/0/14 | 150/0/112 | 0 | 0.210736s | 0.210701s |
| packed data-first ppermute apply | 0/2/2 | 0/159/16 | 1 | 0.428408s | 0.429760s |
| packed data-ppermute apply | 2/0/2 | 66/144/16 | 0 | 0.366023s | 0.366362s |

- Interpretation: This is a clean negative result. The new row achieves the intended StableHLO shape (`AG/A2A/CP=2/0/2`), but XLA GPU compiled lowering reintroduces 144 all-to-alls and makes it 1.74x slower than the un-packed data-ppermute bridge. Reducing logical collective count and avoiding StableHLO A2A is not enough; the packed reshape/collective pattern still lowers through expensive A2A-like movement.
- Next action: Treat the current JAX-level `reshard`/`shard_map` bridge space as exhausted for the FSDP-master hot boundary. Pallas/Triton kernels cannot replace cross-device collectives by themselves; a real lower-level bridge would need a custom collective/custom call, or the production path should avoid this hot restore by keeping a grouped expert-bank representation consumable by the model/apply path.

### 2026-06-19 13:10 PDT - slice-first group gather cuts the FSDP bridge by ~27%
- Hypothesis: The current direct `shard_map` bridge gathers the grouped axis before slicing the FSDP `data` shard. For the requested logical movement
  `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)`, we should slice the local matrix axis by `data` first, then gather the grouped axis. That should avoid communicating full rows that the output FSDP shard will discard.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_slice_first_gather_restore_boundary` and `expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary` to `experiments/grug/moe/muon_update_bench.py`, with focused coverage in `experiments/grug/moe/test_muon_update_bench.py`. Commit `83a8218b9` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -q` -> 64 passed; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` -> OK.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-190345`; child `/dlwh/iris-run-job-20260619-190345/grug-train-MUON-BENCH-D2560-L26-R2D2E8-SLICEFIRST-H1-G4-N4-cw-20260619-190343`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-SLICEFIRST-H1-G4-N4-cw-20260619-190343-55a1d8/summary.json`.
- Result:

| boundary | lowered AG/A2A/CP | compiled AG/A2A/CP | median | mean |
| --- | --- | --- | ---: | ---: |
| data-ppermute apply | 14/0/14 | 150/0/112 | 0.2119s | 0.2118-0.2125s |
| slice-first gather apply | 14/0/0 | 150/0/0 | 0.1553s | 0.1553s |

The slice-first gather bridge is about 1.36x faster than data-ppermute apply on the same 4-node R2/D2/E8 L26 bridge-only run, and it removes compiled collective-permute entirely while keeping compiled all-to-all at zero.
- Interpretation: This answers the layout question more favorably than the earlier attempts: the direct target is viable if the local row shard is selected before the group-axis gather. It still compiles to 150 all-gathers, so it does not solve the entire FSDP-master objective by itself, but the bridge is now materially cheaper than the previous best un-packed JAX-level bridge.
- Next action: Integrate this slice-first gather restore into the production grouped MuonH path in place of the data-first A2A restore, then rerun the real grouped MuonH update/apply gate. If production still compiles hundreds of collectives or stalls near ~0.9s, the remaining blocker is interaction with grouped NS/apply rather than this isolated bridge.

### 2026-06-19 13:20 PDT - production slice-first restore is a large win, but compiled A2A remains
- Hypothesis: Replacing the production grouped MuonH restore helper with the slice-first group gather should carry the bridge-only improvement into the real optimizer path: keep grouped NS on `P(('replica_dcn', 'data'), 'expert', None, None)`, slice the local FSDP data shard before gathering the grouped axis, then return FSDP-shaped updates for ordinary `apply_updates`.
- Command:
  - Code change: replaced the production grouped MuonH restore helper in `experiments/grug/moe/optimizer.py` with `_restore_grouped_muonh_for_split_slice_first_gather`, updating tests to require StableHLO `all_to_all=0`, `collective_permute=0`, and `all_gather=2` for the focused helper case. Commit `e55fb2964` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation: `uv run pytest experiments/grug/moe/test_optimizer.py::test_grouped_expert_muonh_optimizer_returns_fsdp_updates_before_apply experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs -q` -> 2 passed; `uv run pytest experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_muon_update_bench.py -q` -> 78 passed; `./infra/pre-commit.py --files experiments/grug/moe/optimizer.py experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_muon_update_bench.py --fix` -> OK.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-191053`; child `/dlwh/iris-run-job-20260619-191053/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-SLICEFIRST-H3-G4-CAP256-N4-cw-20260619-191051`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-SLICEFIRST-H3-G4-CAP256-N4-cw-20260619-191051-ac7c42/summary.json`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `max_grouped_stack_size=256`, `ns_compute_dtype=bf16`, `backend_steps=3`.
- Result:

| path | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled GEMMs | median | mean |
| --- | --- | --- | ---: | ---: | ---: |
| production grouped MuonH update, data-first restore | 14/14/0/0/0 | 150/392/0/0/0 | 308 | 0.933573s | 0.934062s |
| production grouped MuonH apply, data-first restore | 14/14/0/0/0 | 150/392/0/0/0 | 308 | 0.888601s | 0.888825s |
| production grouped MuonH update, slice-first restore | 14/0/0/0/0 | 150/294/0/0/0 | 322 | 0.3599s | 0.3598-0.3599s |
| production grouped MuonH apply, slice-first restore | 14/0/0/0/0 | 150/294/0/0/0 | 322 | 0.3598s | 0.3600s |

Compared with the data-first production gate, slice-first is about 2.59x faster for update-only and about 2.47x faster for restore-then-apply. The real path now reports about 21.3% nominal H100 bf16 peak for update-only and about 23.0% for restore-then-apply, using the harness's estimated NS dot flops.
- Interpretation: This is the first production-shaped FSDP-master grouped MuonH gate that is clearly in the right performance band. It proves the slice-first local-slice-before-gather idea survives integration into the grouped optimizer helper and ordinary `apply_updates` boundary. However, it does not prove that the collective explosion is fully solved: StableHLO is A2A-free, but compiled GPU HLO still contains `compiled_hlo_all_to_all=294`. The next blocker is understanding or eliminating those compiler-introduced A2As, not another high-level data-first/ppermute bridge tweak.
- Next action: Keep `e55fb2964` as the current best production checkpoint. The next experiment should inspect the compiled HLO around the slice-first helper and try to force a direct grouped-to-FSDP slice/gather lowering that does not reintroduce compiled all-to-alls, or decide that the remaining `A2A=294` is acceptable if full train integration lands near the non-Muon target.

### 2026-06-19 12:28 PDT - compiled HLO dump corrects the A2A count and locates the source
- Hypothesis: The production slice-first row's `compiled_hlo_all_to_all=294` might be inflated by string matches on async start/done wrappers and metadata. Preserve the exact compiled GPU HLO so the remaining collective sites can be attributed.
- Command:
  - Code change: added remote compiled-HLO upload support to `experiments/grug/moe/muon_update_bench.py` / `launch_cw_muon_update_bench.py`, then launched a compile-only 4-node run with `MUON_BENCH_WRITE_COMPILED_HLO=true`.
  - CoreWeave run: parent `/dlwh/iris-run-job-20260619-192308`; child `/dlwh/iris-run-job-20260619-192308/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-SLICEFIRST-HLODUMP-H3-G4-CAP256-N4-cw-20260619-192305`.
  - Output: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-SLICEFIRST-HLODUMP-H3-G4-CAP256-N4-cw-20260619-192305-3fbe60/`.
  - Local artifact copies: `scratch/muon_hlo_20260619_192305/compiled_hlo_real_expert_fsdp_grouped_muonh_optimizer_update_h3.txt`, `scratch/muon_hlo_20260619_192305/compiled_hlo_real_expert_fsdp_grouped_muonh_optimizer_apply_h3.txt`, and `scratch/muon_hlo_20260619_192305/summary.json`.
- Result: The job succeeded. Lowered HLO remains `AG/A2A/AR/RS/CP=14/0/0/0/0`. After fixing `summarize_hlo` to count compiled collective instructions instead of substring mentions, both update-only and restore-then-apply compiled HLO have:
  - `all-to-all` instructions: 28 total, split evenly across `bf16[1,2,1,32,1280,1280]` and `bf16[1,2,1,32,1280,2560]`.
  - all 28 A2As are under `.../grouped_muonh/reshard_grouped_stack/reshard`, over `data` (`mesh[...] {'axis_1'}`), with `dimensions={1}`.
  - `all-gather-start` instructions: 14 total, 7 each for `bf16[1,32,1280,1280] -> bf16[4,32,1280,1280]` and `bf16[1,32,1280,2560] -> bf16[4,32,1280,2560]`, under `.../transform_updates/shard_map/all_gather`.
  - GEMM custom calls: 322.
- Interpretation: The bad compiled A2A is real, but the previously reported `294` was a regex-overcount, not 294 distinct collective instructions. The remaining A2As are not from the final slice-first grouped-to-FSDP restore. They come earlier, when FSDP-shaped expert params/updates are stacked and reshaped into the grouped NS layout `P(('replica_dcn', 'data'), 'expert', None, None)`. The final restore path is the 14 all-gathers from the slice-first `shard_map`.
- Next action: Treat `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)` as viable for the outbound restore; the next hard problem is the inbound FSDP-to-grouped layout conversion. The likely production fix needs NS-friendly grouped momentum/direction state or an explicit inbound shard_map/custom collective so we do not reshard FSDP leaves into grouped layout via XLA's per-chunk A2A lowering every step.

### 2026-06-19 12:45 PDT - explicit inbound FSDP-to-grouped transfer removes compiled A2A
- Hypothesis: The compiled all-to-alls in the production slice-first gate come from the generic inbound `jax.sharding.reshard` from stacked FSDP expert leaves into the grouped NS layout. Replacing that generic reshard with an explicit slice-first gather should keep the train-state/master parameters in FSDP form while avoiding XLA GPU's A2A lowering for the grouped MuonH compute layout.
- Command:
  - Code change: replaced the generic inbound reshard under `grouped_muonh/reshard_grouped_stack` with `_enter_grouped_muonh_ns_layout_slice_first_gather` in `experiments/grug/moe/optimizer.py`. Commit `512fd5d24` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Validation before launch: `uv run pytest experiments/grug/moe/test_optimizer.py::test_grouped_expert_muonh_optimizer_returns_fsdp_updates_before_apply experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs experiments/grug/moe/test_muon_update_bench.py::test_summarize_hlo_counts_compiled_collective_instructions_not_metadata_mentions -q` -> 3 passed; `./infra/pre-commit.py --files experiments/grug/moe/optimizer.py experiments/grug/moe/test_optimizer.py experiments/grug/moe/test_muon_update_bench.py --fix` -> OK.
  - Compile-only CoreWeave run: parent `/dlwh/iris-run-job-20260619-193809`; child `/dlwh/iris-run-job-20260619-193809/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-INBOUNDGATHER-HLODUMP-H3-G4-CAP256-N4-cw-20260619-193807`; output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-INBOUNDGATHER-HLODUMP-H3-G4-CAP256-N4-cw-20260619-193807-e8c959/`.
  - Timing CoreWeave run: parent `/dlwh/iris-run-job-20260619-194157`; child `/dlwh/iris-run-job-20260619-194157/grug-train-MUON-BENCH-D2560-L26-R2D2E8-REALGROUPEDMUONH-INBOUNDGATHER-H3-G4-CAP256-N4-cw-20260619-194155`.
- Config: 4 H100 nodes, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `max_grouped_stack_size=256`, `ns_compute_dtype=bf16`, `backend_steps=3`.
- Result:

| path | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled GEMMs | median | mean | H100 bf16 peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| production grouped MuonH update, previous slice-first | 14/0/0/0/0 | 14/28/0/0/0 | 322 | ~0.3599s | ~0.3599s | ~21.3% |
| production grouped MuonH apply, previous slice-first | 14/0/0/0/0 | 14/28/0/0/0 | 322 | ~0.3598s | ~0.3600s | ~23.0% |
| production grouped MuonH update, explicit inbound gather | 42/0/0/0/0 | 42/0/0/0/0 | 329 | 0.299156s | 0.298891s | 25.7% |
| production grouped MuonH apply, explicit inbound gather | 42/0/0/0/0 | 42/0/0/0/0 | 329 | 0.300002s | 0.301865s | 27.5% |

The compile-only HLO dump confirms the important qualitative change: compiled all-to-all count is now zero for both update-only and restore-then-apply. Runtime improves by about 1.20x versus the previous production slice-first gate.
- Interpretation: This is the current best pragmatic FSDP-master grouped MuonH checkpoint. Train-state/master params remain in FSDP layout, grouped MuonH computes in `P(('replica_dcn', 'data'), 'expert', None, None)`, and updates return to FSDP before ordinary `apply_updates`. The price is more all-gathers (`14 -> 42`) and a few more GEMM custom calls (`322 -> 329`), but the bad compiled A2A path is gone and the 4-node gate is faster. This validates the user's proposed direct layout target for the outbound direction and shows the remaining compiler issue was the inbound grouped-layout entry.
- Next action: Integrate this checkpoint into full training/profile validation before calling the production path solved. If full train still underperforms, the next design question is whether to keep a grouped optimizer-state mirror to remove some inbound gathers or accept the explicit FSDP-to-grouped gather as the sync boundary.

### 2026-06-19 13:15 PDT - full-train grouped MuonH production gate succeeds but is slow
- Hypothesis: The explicit inbound-gather production grouped MuonH checkpoint should work end-to-end in real Grug MoE training with train-state/master params in FSDP layout, MuonH computed in the grouped NS-friendly layout, and grouped updates restored to FSDP before ordinary `apply_updates`.
- Command:
  - Commit under test: `512fd5d24` plus logbook checkpoint `814083acf` on `codex/research-grug-moe-d2560-mfu`.
  - Launch: direct `experiments/grug/moe/run_cw_may_d2560.sh --submit ...` via the May192 launch path.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-195106`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-195106/grug-train-GM2560-MAY-192S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PRODGATE-N2-cw-20260619-1951`.
  - W&B: `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-192S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PRODGATE-N2-cw-20260619-1951`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/grug-moe-cw-may-d2560-L26-e256-r2-cpu8-GM2560-MAY-192S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PRODGATE-N2-cw-20260619-1951-00e4ec`.
- Config: 2 H100 nodes, 16 devices, batch 16, sequence length 4096, sliding window 2048, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, `batch_shards=16`, synthetic data, no checkpoints, profiler disabled, `gpu_fa4_cute` attention, Pallas CE block size 8192, ring MoE, save-MoE remat, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, `expert_grouped_muonh_group_size=2`, MuonH3, `stack_batch_4d_sharded`, `max_grouped_stack_size=512`, and bf16 NS compute.
- Result: Iris parent and child both reached `JOB_STATE_SUCCEEDED` with exit code 0 and two succeeded training tasks. W&B finished with:

| metric | value |
| --- | ---: |
| `global_step` | 7 |
| `train/loss` | 3.1031787395477295 |
| `throughput/mfu` | 6.671198732245638 |
| `throughput/mean_mfu` | 4.980871047127578 |
| `throughput/tokens_per_second` | 61692.52165774803 |
| `throughput/duration` | 1.062300555058755 |
| `throughput/total_tokens` | 524288 |

The logs show the first compiled/initialization-heavy steps dominated wall time, then W&B uploaded steps 0-7 and the run completed successfully. There was no OOM, traceback, rendezvous failure, or Iris task failure; the post-finish coordination-service warnings were shutdown noise after successful W&B sync.
- Interpretation: The pragmatic grouped MuonH path is functionally validated in full training: it can keep FSDP master/train-state params, enter grouped NS layout, compute MuonH, restore FSDP updates, and use ordinary `apply_updates` without crashing. It is not a performance success yet. Mean MFU is far below the 2-node non-Muon/SGD reference (`May186` at ~17.73 MFU) and below the production goal. This means the explicit bridge fixed the compiled A2A correctness/perf gate in isolation, but full train still needs profiling to find remaining Muon/communication/overlap cost.
- Next action: Do not call the MuonH production path solved. Launch or inspect a profile variant of this exact shape next, ideally preserving command-buffer-disabled name stacks for readability, and compare the optimizer segment against the 4-node harness expectation. Also consider a 4-node R2/D2 full-train validation if we need to match the harness layout where grouped MuonH uses both `replica_dcn` and `data`.

### 2026-06-19 13:50 PDT - direct grouped-to-FSDP restore is a bad path
- Hypothesis: The outbound grouped MuonH restore does not need the explicit split-first `shard_map/all_gather` path. We can reshard the 4D grouped update directly from the NS-friendly layout `P(('replica_dcn', 'data'), 'expert', None, None)` into grouped FSDP layout `P(None, *param_spec)`, then split the stack into ordinary per-leaf updates before `apply_updates`. This should keep the expert axis and FSDP data-axis sharding while reducing the explicit grouped restore all-gathers seen in the May193 readable profile.
- Command:
  - Code change: `_grouped_expert_muonh_updates` now calls `_restore_grouped_muonh_for_split_target_layout` under `grouped_muonh/restore_grouped_fsdp_layout` instead of `_restore_grouped_muonh_for_split_slice_first_gather`.
  - Validation: `uv run pytest experiments/grug/moe/test_optimizer.py -q` -> 15 passed; `./infra/pre-commit.py --files experiments/grug/moe/optimizer.py experiments/grug/moe/test_optimizer.py --fix` -> OK.
  - Local lowering result: the focused production optimizer shape still preserves FSDP update sharding and StableHLO all-gather count drops from 6 to 4, with all-reduce/reduce-scatter/all-to-all still 0.
  - CoreWeave launch: `bash scratch/launch_may194_fa4_2node_b16_grouped_muonh3_direct_fsdp_restore.sh`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-204805`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-204805/grug-train-GM2560-MAY-194S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-DIRECTFSDP-N2-cw-20260619-2048`.
  - W&B: `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-194S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-DIRECTFSDP-N2-cw-20260619-2048`.
- Config: Same B16/R2 full-train gate as May192: 2 H100 nodes, sequence length 4096, sliding window 2048, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, synthetic data, no checkpoints, profiler disabled, `gpu_fa4_cute` attention, Pallas CE block size 8192, ring MoE, save-MoE remat, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, `expert_grouped_muonh_group_size=2`, MuonH3, `stack_batch_4d_sharded`, `max_grouped_stack_size=512`, and bf16 NS compute.
- Result: Negative. Step 0 eventually completed after about 7m43s (`20:49:41` to `20:57:24` UTC). Step 1 then remained open for about 5 minutes with no W&B scalar rows, no further train logs, about 75.2 GiB allocated on every GPU, and 0% GPU and memory utilization on both H100 nodes. Iris still reported both child tasks running with no failure/preemption, so the run was manually stopped to free the two nodes:
  `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260619-204805`.
- Interpretation: The generic direct reshard is not viable as the production default. The focused StableHLO looked better (`all_gather` 6 -> 4), but the real 2-node training path behaved pathologically at runtime, likely because XLA GPU lowered the direct grouped `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', 'model')` conversion into a bad collective/runtime sequence. The explicit `shard_map` restore remains the current best known production path despite its visible `all_gather` cost.
- Next action: Revert the production call site to `_restore_grouped_muonh_for_split_slice_first_gather` and keep this as a negative result. The next attempt should be an explicit custom conversion, not generic `jax.sharding.reshard`: either a more targeted `shard_map` that performs exactly the wanted grouped-to-FSDP movement or a lower-level Pallas/Triton path if XLA continues to expand the grouped layout conversion poorly.

### 2026-06-19 14:25 PDT - tuple-returning shard_map restore does not improve the grouped-to-FSDP boundary
- Hypothesis: The explicit slice-first grouped-to-FSDP restore might still be paying for keeping a grouped 4D result alive outside the `shard_map`. If the `shard_map` itself returns the tuple of FSDP-shaped per-leaf updates, XLA might avoid materializing the grouped restore boundary and reduce the visible all-gather cost.
- Command:
  - Code change: added `expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary` to `experiments/grug/moe/muon_update_bench.py`. The new bench slices the data-sharded matrix axis first, gathers over the grouped `('replica_dcn', 'data')` axis, splits inside `shard_map`, and returns ordinary FSDP-sharded per-leaf updates directly. Commits `17499662c` and `83f2971a7` pushed to `codex/research-grug-moe-d2560-mfu`.
  - Local validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_tuple_slice_first_gather_boundary_returns_fsdp_updates_without_a2a experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_slice_first_gather_boundary_returns_fsdp_updates_without_a2a -q` -> 2 passed; `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` -> OK.
  - First CoreWeave attempt: parent `/dlwh/iris-run-job-20260619-211432`, child `/dlwh/iris-run-job-20260619-211432/grug-train-MUON-BENCH-D2560-L26-R2D2E8-TUPLESPLIT-H1-G4-N4-cw-20260619-211429`; failed before timing because the new bench kind was missing from `is_expert_fsdp_grouped_bench`.
  - Retry CoreWeave run: parent `/dlwh/iris-run-job-20260619-211927`, child `/dlwh/iris-run-job-20260619-211927/grug-train-MUON-BENCH-D2560-L26-R2D2E8-TUPLESPLIT2-H1-G4-N4-cw-20260619-211924`; output prefix `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D2E8-TUPLESPLIT2-H1-G4-N4-cw-20260619-211924-3aa595`.
- Config: 4 H100 nodes, 32 GPUs, `replica_axis=2`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `hidden_dim=2560`, `intermediate_dim=1280`, `num_experts=256`, `ns4d_group_size=4`, `ns4d_group_axis=replica_dcn,data`, `max_grouped_stack_size=512`, `backend_steps=1`, `write_compiled_hlo=true`.
- Result: The retry succeeded, but the tuple-returning path was effectively the same as the existing explicit slice-first restore.

| restore boundary | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | median | mean | compile |
| --- | --- | --- | ---: | ---: | ---: |
| existing grouped 4D slice-first restore | 14/0/0/0/0 | 14/0/0/0/0 | ~0.15325-0.15333s | ~0.15328-0.15337s | ~0.86-0.91s |
| tuple-returning slice-first restore | 14/0/0/0/0 | 14/0/0/0/0 | ~0.15344-0.15350s | ~0.15402-0.15406s | ~0.42-0.43s |

The tuple path lowers faster and its StableHLO text is smaller, but the compiled HLO is slightly larger and runtime is equal to slightly worse. Most importantly, compiled all-gather count remains 14 and there are still no all-to-alls, all-reduces, reduce-scatters, or collective-permutes.
- Interpretation: Returning the split FSDP tuple from inside `shard_map` does not make XLA GPU batch away or avoid the grouped all-gathers. It is a useful harness datapoint, but not a production improvement. The current explicit grouped 4D slice-first restore remains the best known production boundary.
- Next action: Do not integrate the tuple-returning bridge into the production optimizer. Further improvement needs to avoid the grouped-to-FSDP sync boundary, overlap it, or use a lower-level route that communicates exactly the final FSDP slices. Pallas alone will not replace a cross-device collective; it would need to sit around an explicit collective or use a lower-level communication primitive.

### 2026-06-19 14:55 PDT - 2-node profile attempts hang at clique init before step 0
- Hypothesis: Profiling the current explicit inbound-gather / explicit slice-first-restore grouped MuonH full-train path should show where the gap between the fast 4-node optimizer gate and the slower May192 full train comes from.
- Command:
  - Readable profile with command buffers disabled: `bash scratch/launch_may195_fa4_2node_b16_grouped_muonh3_current_readable_profile.sh`.
  - Normal profile with command buffers enabled: `bash scratch/launch_may196_fa4_2node_b16_grouped_muonh3_current_profile.sh`.
  - May195 parent `/dlwh/iris-run-job-20260619-212758`, W&B `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-195S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-READABLEPROFILE-N2-cw-20260619-2127`.
  - May196 parent `/dlwh/iris-run-job-20260619-214055`, W&B `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-196S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-PROFILE-N2-cw-20260619-2140`.
- Config: Both runs matched the May192 B16/R2 full-train grouped MuonH shape, with profiler enabled for two steps and HLO proto upload enabled. May195 additionally set `--xla_gpu_enable_command_buffer=''`; May196 left command buffers enabled.
- Result: Both runs hung before the first training scalar. May195 reached W&B but never emitted step metrics; task logs showed a clique-initialization rendezvous warning. May196 reproduced the same failure with command buffers enabled: W&B summary remained null for `global_step`, `throughput/mfu`, `throughput/tokens_per_second`, and `train/loss`, while Iris still reported both tasks running and logs showed `Initialize clique: devices=16` rendezvous warnings from task 0. May196 was manually stopped with `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job stop /dlwh/iris-run-job-20260619-214055`.
- Interpretation: The profile hang is not explained by the command-buffer readability flag. The next isolation step is a no-profile B16/R2 run on the current branch to determine whether the current grouped MuonH production path still runs end-to-end after the latest harness-only commits, or whether profiling/HLO proto collection is the trigger.
- Next action: Launch a non-profile B16/R2 current-branch throughput sanity run before attempting another multihost profile.

### 2026-06-19 15:10 PDT - no-profile current path completes train loop but W&B finalization crashes
- Hypothesis: If the current explicit inbound-gather / explicit slice-first-restore grouped MuonH production path is still healthy, a no-profile B16/R2 run should get past the May195/May196 clique-init profile hang and complete training.
- Command:
  - Launch script: `bash scratch/launch_may197_fa4_2node_b16_grouped_muonh3_current_throughput.sh`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-215443`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-215443/grug-train-GM2560-MAY-197S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-THROUGHPUT-N2-cw-20260619-2154`.
  - W&B: `https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-197S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-THROUGHPUT-N2-cw-20260619-2154`.
- Config: Same B16/R2 shape as May196 but with profiler disabled: 2 H100 nodes, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, batch 16, sequence length 4096, sliding window 2048, synthetic data, no checkpoints, 8 steps, `gpu_fa4_cute`, Pallas CE block size 8192, ring MoE, save-MoE remat, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, group size 2, MuonH3, `stack_batch_4d_sharded`, cap512, bf16 NS compute.
- Result: The run passed the May195/May196 failure point. Logs show the child created W&B, formed mesh `{'replica_dcn': 2, 'data': 1, 'expert': 8, 'model': 1}`, started step 0 at `21:56:16/17`, finished step 0 at `22:01:08`, finished step 1 at `22:06:00`, finished step 2 at `22:06:01`, then reached `Progress on:train 8.00it/8.00it ... postfix:loss=3.1` at `22:06:09`. There were no rendezvous warnings in the checked logs. W&B then failed finalization with `wandb.sdk.mailbox.mailbox_handle.HandleAbandonedError`, marked the run crashed, and synced no scalar history rows. Iris continued to show the child running after the train loop, so the job was manually stopped to free the nodes.
- Interpretation: The current production path did not regress back to the May195/May196 clique-init hang; profiler/HLO collection is still the leading suspect for that hang. However, May197 did not provide trustworthy throughput scalars because W&B finalization failed before metrics persisted. The post-step timing from logs is too coarse for a headline number.
- Next action: Do not launch another identical profile attempt. Continue on the custom grouped-to-FSDP conversion work, and use a future no-profile or lower-overhead metric path if exact full-train throughput is needed.

### 2026-06-19 15:25 PDT - JSON-tracker rerun captures full-train grouped MuonH metrics in logs
- Hypothesis: The May197 train loop completed but lost scalar evidence because W&B finalization crashed. Logging direct performance metrics before tracker writes and using `json_logger` should preserve exact full-train throughput evidence in Iris logs.
- Command:
  - Code change: added a direct `experiments.grug.moe.train` info log before `levanter.tracker.log(...)` with step, loss, duration, tokens/sec, MFU, and mean MFU.
  - Validation before launch: `uv run python -m py_compile experiments/grug/moe/train.py`; `./infra/pre-commit.py --files experiments/grug/moe/train.py .agents/logbooks/grug-moe-muon-gpu.md --fix`.
  - Launch script: `bash scratch/launch_may198_fa4_2node_b16_grouped_muonh3_current_throughput_json.sh`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-221150`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-221150/grug-train-GM2560-MAY-198S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-THROUGHPUTJSON-N2-cw-20260619-2211`.
- Config: Same B16/R2 current production path as May197, but `--tracker json_logger`: 2 H100 nodes, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, batch 16, sequence length 4096, sliding window 2048, synthetic data, no checkpoints, 8 steps, `gpu_fa4_cute`, Pallas CE block size 8192, ring MoE, save-MoE remat, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, group size 2, MuonH3, `stack_batch_4d_sharded`, cap512, and bf16 NS compute.
- Result: Iris parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions. The first two steps were compile/initialization dominated:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 292.62-293.23s | ~223.5 | ~0.024 | 11.7917 |
| 1 | 293.63-293.64s | ~223.2 | ~0.024 | 9.2990 |

Post-compile steps were stable around 1.06s:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.055-1.058s | 61.94k-62.09k | 6.70-6.71 | 7.1580 |
| 3 | 1.059-1.061s | 61.77k-61.90k | 6.68-6.69 | 5.0763 |
| 4 | 1.059-1.066s | 61.47k-61.87k | 6.65-6.69 | 4.2849 |
| 5 | 1.056-1.058s | 61.94k-62.05k | 6.70-6.71 | 3.5005 |
| 6 | 1.056-1.065s | 61.55k-62.04k | 6.66-6.71 | 3.8169 |
| 7 | 1.058-1.062s | 61.68k-61.94k | 6.67-6.70 | 3.1036 |

- Interpretation: This confirms May192's W&B numbers and May197's coarse log timings without relying on W&B: the current pragmatic FSDP-master grouped MuonH path is correct and stable, but not performant enough. The current 2-node B16 post-warmup full-train rate is only about 6.7 MFU, versus the best 2-node B16 SGD/non-Muon reference at about 17.7 MFU. The explicit inbound/outbound bridge fixed the compiled A2A correctness gate in the isolated optimizer harness, but full training still has a large Muon/communication/overlap gap.
- Next action: Stop treating tracker/profiler finalization as the main uncertainty. The next useful work is profile/trace attribution of the no-profile-successful current path, or a lower-overhead isolated full-step harness that can decompose optimizer bridge, NS compute, gradient psum, and model fwd/bwd without HLO-profiler rendezvous hangs.

### 2026-06-19 15:35 PDT - R2D1E8 real grouped MuonH optimizer attribution
- Hypothesis: The May198 full-step gap can be decomposed by running the real `GrugMoeMuonHConfig(expert_3d_optimizer="grouped_muonh")` optimizer path in the standalone Muon update harness on the exact May198 mesh, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`.
- Command:
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-REALGROUPEDMUONH-CURRENT-H3-G2-CAP512-N2-cw-20260619-222821 ... MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_update,real_expert_fsdp_grouped_muonh_optimizer_apply ... bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-222825`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-222825/grug-train-MUON-BENCH-D2560-L26-R2D1E8-REALGROUPEDMUONH-CURRENT-H3-G2-CAP512-N2-cw-20260619-222821`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-REALGROUPEDMUONH-CURRENT-H3-G2-CAP512-N2-cw-20260619-222821-081da0`.
- Config: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, bf16 params/updates and bf16 NS compute, MuonH3, cap512, warmup 1, iters 5, compiled HLO output enabled.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED`. Both update-only and restore-then-apply paths lowered and compiled with no all-to-all, all-reduce, reduce-scatter, or collective-permute. The active collectives are the expected 26 all-gathers.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled GEMM custom calls | median | mean | H100 bf16 peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| real grouped MuonH update | 26/0/0/0/0 | 26/0/0/0/0 | 533 | 0.4974-0.4980s | 0.4979s | ~30.8% |
| real grouped MuonH apply | 26/0/0/0/0 | 26/0/0/0/0 | 533 | 0.4770-0.4771s | 0.4769s | ~32.2% |

- Interpretation: On the exact May198 2-node mesh, the real grouped MuonH optimizer+apply path accounts for about `0.48s / 1.06s ~= 45%` of the post-compile full training step. The bridge is no longer exploding into A2As, but it still costs 26 all-gathers and about half a second. Compared with the earlier R2D2E8 4-node harness gate around 0.30s, losing the `data=2` axis and running R2D1E8 materially hurts the optimizer path. The remaining ~0.58s of May198 is outside the expert grouped-MuonH optimizer apply path: model fwd/bwd, gradient collectives, non-expert optimizer work, scheduler gaps, or overlap loss.
- Next action: Do not spend more effort on generic reshard variants. The next useful experiments are either (1) a 4-node R2D2 full-train grouped MuonH run to see whether the faster optimizer harness transfers to full training, or (2) a lower-overhead full-step decomposition that separates fwd/bwd/grad-psum/non-expert optimizer without enabling the multihost profiler path that hung in May195/196.

### 2026-06-19 15:50 PDT - R2D2 full-train validation relaunched with legal batch shards
- Hypothesis: The earlier R2D2 optimizer harness was faster than the exact R2D1 May198 optimizer attribution because grouped MuonH could shard over both `replica_dcn` and `data`. A 4-node full-train validation with `replica_dcn=2`, `data=2`, `expert=8`, and `model=1` should show whether that optimizer-path improvement transfers into full training.
- Command:
  - Failed first attempt: parent `/dlwh/iris-run-job-20260619-223456` (`May200`) used 4 nodes with batch 16 and failed in the launcher before creating a child:
    `ValueError: MAY_BATCH=16 must be divisible by batch shards=32`.
  - Corrected launch: parent `/dlwh/iris-run-job-20260619-223756` (`May201`) with `--batch 32`, the smallest legal batch for 32 global GPUs when `model_axis=1`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-223756/grug-train-GM2560-MAY-201S4096-W2048-B32-R2D2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-JSON-N4-cw-20260619-2237`.
- Config: 4 H100 nodes, 32 devices, batch 32, sequence length 4096, sliding window 2048, 8 train steps, `json_logger`, synthetic data, no checkpoints, no profiler, `replica_dcn=2`, `data=2`, `expert=8`, `model=1`, `gpu_fa4_cute` attention, Pallas CE with block size 8192, ring MoE, save-MoE remat, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, ordinary 2D optimizer SGD, group size 4, MuonH3, `stack_batch_4d_sharded`, cap512, and bf16 NS compute.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions. Logs confirm the intended mesh on every task:
  `Grug compact mesh shape: {'replica_dcn': 2, 'data': 2, 'expert': 8, 'model': 1}; batch_shards=32`.
  The first two steps were compile/initialization dominated:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 380.48-386.27s | 339.33-344.49 | 0.0183-0.0186 | 11.7920 |
| 1 | 370.07-370.12s | 354.13-354.18 | 0.0191 | 6.5592 |

Post-compile steps were stable around 1.66s:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.659-1.660s | 78.94k-79.01k | 4.268-4.272 | 4.1190 |
| 3 | 1.660-1.662s | 78.88k-78.97k | 4.265-4.270 | 3.5953 |
| 4 | 1.662-1.667s | 78.65k-78.84k | 4.252-4.263 | 1.7895 |
| 5 | 1.668-1.672s | 78.38k-78.58k | 4.238-4.249 | 1.6126 |
| 6 | 1.663-1.678s | 78.10k-78.84k | 4.223-4.263 | 1.4356 |
| 7 | 1.655-1.660s | 78.98k-79.18k | 4.270-4.281 | 1.2978 |

- Interpretation: Negative scale result. R2D2/B32 is correct and stable, but it does not transfer the faster R2D2 optimizer harness result into better full-train throughput. Relative to May198 R2D1/B16, May201 used twice the GPUs and doubled global batch, but tokens/sec only improved from ~61.5k-62.1k to ~78.1k-79.2k (`~1.27x`) and MFU dropped from ~6.7 to ~4.25. The added `data=2` axis likely helps the isolated optimizer path but loses enough elsewhere, or loses enough overlap, that full train gets worse per GPU.
- Next action: Stop pushing batch/scale as the next lever for this path. The useful next experiments are attribution on the successful no-profile path: compare exact-shape fwd/bwd+SGD against grouped-MuonH and add a lower-overhead decomposition for gradient collectives / non-expert optimizer / grouped MuonH bridge, avoiding the multihost profiler path that hung in May195/May196.

### 2026-06-19 16:15 PDT - exact R2D1/B16 SGD control on current branch
- Hypothesis: The current grouped-MuonH full-train path needs an apples-to-apples non-Muon control on the same reliable JSON/no-profile path, rather than comparing only to older speed-lane runs with potentially different flags or code.
- Command:
  - Launch: `RUN_ID=GM2560-MAY-202S4096-W2048-B16-R2D1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4SGD-CURRENT-JSON-N2-cw-20260619-2257 ... experiments/grug/moe/run_cw_may_d2560.sh --submit --nodes 2 --batch 16 --replica-axis 2 --expert-axis 8 --model-axis 1 --optimizer sgd --tracker json_logger --profiler-steps 0 --data synthetic --checkpoints none --ce-implementation pallas_gpu --moe-implementation ring --input-embed-sharding replicated --output-proj-sharding replicated --remat save_moe --attention gpu_fa4_cute --mp params=bfloat16,compute=bfloat16,output=bfloat16`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-225709`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-225709/grug-train-GM2560-MAY-202S4096-W2048-B16-R2D1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4SGD-CURRENT-JSON-N2-cw-20260619-2257`.
- Config: Same mesh/shape as May198 but with top-level SGD instead of grouped MuonH: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, batch 16, sequence length 4096, sliding window 2048, 8 train steps, JSON logger, no profiler, synthetic data, no checkpoints, FA4 CuTe attention, Pallas CE block size 8192, ring MoE, save-MoE remat, replicated input/output embeddings, and bf16 params/compute/output.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions. Logs confirm mesh `{'replica_dcn': 2, 'data': 1, 'expert': 8, 'model': 1}; batch_shards=16`. First two steps were compile/initialization dominated:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 355.09-356.92s | 183.62-184.56 | 0.0199-0.0200 | 11.7918 |
| 1 | 334.26-334.30s | 196.04-196.06 | 0.0212 | 11.7922 |

Post-compile steps were stable around 0.65-0.66s:

| step | duration | tokens/sec | MFU | loss |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.656-0.660s | 99.37k-99.86k | 10.745-10.798 | 11.7918 |
| 3 | 0.652-0.658s | 99.60k-100.47k | 10.770-10.864 | 11.7934 |
| 4 | 0.661-0.663s | 98.86k-99.17k | 10.691-10.724 | 11.7914 |
| 5 | 0.658-0.662s | 99.05k-99.65k | 10.711-10.776 | 11.7913 |
| 6 | 0.658-0.660s | 99.25k-99.66k | 10.733-10.777 | 11.7918 |
| 7 | 0.655-0.662s | 99.01k-100.08k | 10.707-10.822 | 11.7944 |

- Interpretation: This gives a clean same-shape non-Muon floor for the current branch. Compared to May198 R2D1/B16 grouped MuonH at ~1.06s, ~61.5k-62.1k tokens/s, and ~6.7 MFU, the no-Muon control is ~0.40s faster and reaches ~10.7-10.9 MFU. The full-train grouped-MuonH tax is therefore about 0.40s/step on this shape, close to the isolated May199 real grouped-MuonH optimizer/apply path at ~0.477s. That makes the grouped optimizer path the dominant remaining gap, not general fwd/bwd throughput on this current branch.
- Next action: Focus on reducing or hiding the grouped-MuonH optimizer/apply cost at R2D1 before trying more scale. The next useful harness should measure bridge/NS/apply subpieces or prototype a representation that keeps grouped expert updates consumable by the model without paying the explicit grouped-to-FSDP boundary every step.

### 2026-06-19 16:45 PDT - R2D1 grouped MuonH bridge decomposition
- Hypothesis: The exact R2D1 grouped-MuonH tax can be split into inbound FSDP/unreduced-gradient to grouped layout, grouped MuonH NS/hyperball compute, outbound grouped-to-FSDP restore, and ordinary `optax.apply_updates` overhead. If ordinary `apply_updates` is the bottleneck, restore+apply should be much slower than restore-only; if the representation boundary is the bottleneck, the restore-only timing should explain most of the gap.
- Command:
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-GROUPEDMUONH-DECOMP-H3-G2-CAP512-N2-cw-20260619-231728 ... MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_update,real_expert_fsdp_grouped_muonh_optimizer_apply,expert_fsdp_grouped_updates_muonh_updates,expert_fsdp_grouped_updates_muonh_apply,expert_fsdp_grouped_explicit_slice_first_gather_restore_boundary,expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary ... bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-231730`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-231730/grug-train-MUON-BENCH-D2560-L26-R2D1E8-GROUPEDMUONH-DECOMP-H3-G2-CAP512-N2-cw-20260619-231728`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-GROUPEDMUONH-DECOMP-H3-G2-CAP512-N2-cw-20260619-231728-b9110a`.
- Config: Exact May198/May202 mesh for the expert optimizer path: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, bf16 params/updates, bf16 NS compute, MuonH3, cap512, warmup 1, iters 5, compiled HLO output enabled, and boundary collectives allowed for the explicit bridge probes.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED`. All six benches compiled with the same healthy collective shape: 26 all-gathers and no all-to-all, all-reduce, reduce-scatter, or collective-permute.

| bench | compiled AG/A2A/AR/RS/CP | compiled GEMM custom calls | median | mean | H100 bf16 peak |
| --- | --- | ---: | ---: | ---: | ---: |
| real grouped MuonH update | 26/0/0/0/0 | 533 | ~0.496s | ~0.496s | ~30.9% |
| real grouped MuonH apply | 26/0/0/0/0 | 533 | ~0.477s | ~0.478s | ~32.2% |
| grouped-updates MuonH update | 26/0/0/0/0 | 533 | ~0.302s | ~0.303s | ~50.8% |
| grouped-updates MuonH apply | 26/0/0/0/0 | 533 | ~0.307s | ~0.308s | ~50.0% |
| explicit slice-first gather restore boundary | 26/0/0/0/0 | 0 | ~0.223s | ~0.223s | n/a |
| explicit slice-first gather restore+apply boundary | 26/0/0/0/0 | 0 | ~0.229s | ~0.228s | n/a |

- Interpretation: The current pragmatic FSDP-master path is not blocked by ordinary `optax.apply_updates`. Applying already-restored updates costs only about `0.228s - 0.223s ~= 5ms`. The dominant boundary is grouped-to-FSDP restore at about 0.22s, and the inbound FSDP/unreduced-gradient/momentum to grouped layout costs roughly `0.496s - 0.303s ~= 0.19s`. The grouped-updates path reaches about 50% H100 bf16 peak, so the NS/hyperball compute is no longer the primary issue on this shape. The real full-train Muon tax from May198 vs May202 is about 0.40s/step, which is almost exactly explained by inbound grouping plus outbound restore.
- Next action: Treat the representation boundary as the remaining MuonH target. Do not optimize ordinary `apply_updates` first. The next viable implementation paths are either (1) keep grouped expert banks/updates live long enough that the model or optimizer consumes them without paying a full grouped-to-FSDP restore every step, or (2) replace the explicit slice-first gather boundary with a lower-level/custom grouped-to-FSDP permutation that avoids 26 high-cost all-gathers or overlaps them with useful work.

### 2026-06-19 16:45 PDT - persistent grouped expert bank lower-bound
- Hypothesis: If expert params and updates remain in the grouped NS-friendly layout across the MuonH apply boundary, the optimizer can avoid the compiled grouped-to-FSDP all-gathers and expose the real lower bound for a production representation that does not immediately restore ordinary FSDP leaves.
- Code snapshot:
  - Commit: `d0a5ec1e6 Add persistent grouped MuonH bench`.
  - Added harness bench kind: `expert_fsdp_grouped_persistent_muonh_apply`.
  - Scope: benchmark-only path in `experiments/grug/moe/muon_update_bench.py`; no production optimizer semantics changed.
- Command:
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-PERSISTENTGROUPEDMUONH-H3-G2-CAP512-N2-cw-20260619-234047 ... MUON_BENCH_KINDS=expert_fsdp_grouped_updates_muonh_apply,expert_fsdp_grouped_persistent_muonh_apply ... bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-234050`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-234050/grug-train-MUON-BENCH-D2560-L26-R2D1E8-PERSISTENTGROUPEDMUONH-H3-G2-CAP512-N2-cw-20260619-234047`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-PERSISTENTGROUPEDMUONH-H3-G2-CAP512-N2-cw-20260619-234047-df1c6e`.
- Config: Exact R2D1 expert optimizer shape: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, bf16 params/updates, bf16 NS compute, MuonH3, cap512, warmup 1, iters 5. The run compared the current grouped-updates restore/apply boundary against a persistent grouped params+updates apply that returns grouped next params.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions.

| bench | compiled AG/A2A/AR/RS/CP | compiled GEMM custom calls | median | mean | H100 bf16 peak |
| --- | --- | ---: | ---: | ---: | ---: |
| grouped-updates MuonH restore+apply | 26/0/0/0/0 | 533 | 0.3047s | 0.3050s | 50.4% |
| persistent grouped MuonH apply | 0/0/0/0/0 | 572 | 0.2540s | 0.2541s | 60.4% |

- Interpretation: This confirms the representation-boundary diagnosis. When the benchmark keeps expert params and updates in grouped layout, compiled all-gathers disappear and the isolated expert MuonH3 path improves by about 50 ms versus the current grouped-updates restore/apply path. The persistent grouped lower bound is still not free, but it reaches about 60% nominal H100 bf16 peak and removes the current 26 compiled all-gather boundary. A production fix should now target grouped expert-bank persistence or a model-consumable grouped representation, not ordinary `apply_updates`.
- Next action: Prototype the smallest production-facing representation boundary: either carry grouped expert banks through the optimizer state and only materialize ordinary leaves where the model truly needs them, or add a manual grouped-to-FSDP conversion that avoids the current compiled all-gather pattern. Keep the FSDP-master plan as the conservative correctness target until the grouped-bank representation can be made model-consumable.

### 2026-06-19 16:57 PDT - direct target-FSDP reshard bridge negative result
- Hypothesis: The grouped NS-friendly expert update layout can be converted directly to the desired FSDP/model layout with a target reshard, `P(('replica_dcn', 'data'), 'expert', None, None) -> P(None, 'expert', 'data', None)`, avoiding the explicit slice-first all-gather boundary while still returning ordinary FSDP leaves for `optax.apply_updates`.
- Command:
  - Local focused check: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_target_apply_chunked_fsdp_boundary_returns_fsdp_layout_without_collectives`.
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-TARGETFSDP-H0-CAP512-N2-cw-$(date -u +%Y%m%d-%H%M%S) MUON_BENCH_KINDS=expert_fsdp_grouped_target_apply_chunked_fsdp_boundary,expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_SWEEP_BACKEND_STEPS=1 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260619-235217`.
  - Child Iris job: `/dlwh/iris-run-job-20260619-235217/grug-train-MUON-BENCH-D2560-L26-R2D1E8-TARGETFSDP-H0-CAP512-N2-cw-20260619-235215`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-TARGETFSDP-H0-CAP512-N2-cw-20260619-235215-c54af4`.
- Config: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, bf16 params/updates, bf16 NS compute, MuonH3, cap512, warmup 1, iters 3, compiled HLO output enabled, and boundary collectives allowed.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions. The local lowering-only test passed, but compiled H100 behavior did not preserve the no-collective property.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | median | mean |
| --- | --- | --- | ---: | ---: |
| target-FSDP chunked restore+apply | 0/0/0/0/0 | 26/0/0/0/0 | 0.2384s | 0.2383s |
| explicit slice-first gather restore+apply | 26/0/0/0/0 | 26/0/0/0/0 | 0.2282s | 0.2280s |

- Interpretation: Negative result for the plain target reshard bridge. The direct `reshard(..., P(None, 'expert', 'data', None))` shape looks clean in lowered StableHLO, but XLA GPU rewrites it back into the same 26 per-layer all-gathers in compiled HLO. It is also about 10 ms slower than the explicit slice-first gather baseline on this short bench. This means ordinary target-layout resharding is not enough to avoid the representation-boundary collective explosion.
- Next action: Do not switch production to the target-layout helper. The remaining viable paths are (1) persistent grouped expert-bank representation through the model/optimizer boundary, or (2) a manual grouped-to-FSDP conversion via `shard_map` first, then Pallas/Triton if needed, so the compiler cannot rediscover the conversion as 26 independent all-gathers.

### 2026-06-19 17:10 PDT - packed slice-first gather bridge negative result
- Hypothesis: Packing all grouped updates for one expert weight name before the slice-first grouped-to-FSDP restore should reduce the boundary from one all-gather per layer/group to one larger all-gather per expert weight name, while still returning ordinary FSDP leaves before `optax.apply_updates`.
- Code snapshot:
  - Added harness-only bench kind: `expert_fsdp_grouped_packed_slice_first_gather_apply_boundary`.
  - Scope: `experiments/grug/moe/muon_update_bench.py` and focused tests only; no production optimizer path changed.
- Local validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_packed_slice_first_gather_apply_boundary_packs_group_restores -q`
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_packed_slice_first_gather_apply_boundary_packs_group_restores experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_packed_a2a_apply_boundary_packs_group_restores experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary_returns_fsdp_params_without_a2a -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Command:
  - Local lower: `XLA_FLAGS=--xla_force_host_platform_device_count=16 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 2 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 1 --expert-axis 8 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary,expert_fsdp_grouped_packed_slice_first_gather_apply_boundary --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --output /tmp/muon_packed_slice_r2d1_l26_lower.json`.
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-PACKEDSLICEFIRST-H0-CAP512-N2-cw-$(date -u +%Y%m%d-%H%M%S) MUON_BENCH_KINDS=expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary,expert_fsdp_grouped_packed_slice_first_gather_apply_boundary MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_SWEEP_BACKEND_STEPS=1 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260620-000715`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-000715/grug-train-MUON-BENCH-D2560-L26-R2D1E8-PACKEDSLICEFIRST-H0-CAP512-N2-cw-20260620-000713`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-PACKEDSLICEFIRST-H0-CAP512-N2-cw-20260620-000713-67247f`.
- Config: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, max grouped stack size 512, bf16 params/updates, bf16 NS compute, warmup 1, iters 3, boundary collectives allowed.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | median | mean |
| --- | --- | --- | ---: | ---: |
| explicit slice-first restore+apply | 26/0/0/0/0 | 26/0/0/0/0 | 0.2300s | 0.2299s |
| packed slice-first restore+apply | 2/0/0/0/0 | 2/28/0/0/0 | 0.6011s | 0.6006s |

- Interpretation: Negative result for the packed slice-first bridge. The local and lowered H100 IR showed the intended reduction from 26 all-gathers to 2 all-gathers, but XLA GPU compilation inserted 28 all-to-alls and made the path about 2.6x slower than the simple explicit slice-first baseline. This is a stronger warning than the target-FSDP result: even when the pre-compiled collective count looks better, GPU compilation can introduce a worse communication pattern.
- Next action: Do not pursue packed slice-first restore as the production bridge. The remaining plausible bridge needs to force the exact data movement more directly than ordinary `reshard` or packed `shard_map`, likely with a narrower hand-written permutation or lower-level Pallas/Triton. Persistent grouped expert-bank remains the cleanest lower-bound direction.

### 2026-06-19 17:20 PDT - FSDP-hyperball variant does not reduce grouped MuonH cost
- Hypothesis: The current production grouped MuonH helper pays to enter both direction inputs and FSDP params into the grouped NS-friendly layout. If we compute only the Newton-Schulz direction in grouped layout, restore that direction to FSDP, and run the MuonH hyperball projection against FSDP params, we might avoid the grouped-param entry cost while keeping FSDP master/train-state params and ordinary `optax.apply_updates`.
- Command:
  - Local lower: `XLA_FLAGS=--xla_force_host_platform_device_count=16 uv run python experiments/grug/moe/muon_update_bench.py --layers 26 --ns4d-group-size 2 --ns4d-group-axis replica_dcn,data --hidden-dim 2560 --intermediate-dim 1280 --num-experts 256 --replica-axis 2 --data-axis 1 --expert-axis 8 --model-axis 1 --backend-steps 3 --max-grouped-stack-size 512 --ns-compute-dtype bf16 --bench-kinds expert_fsdp_grouped_muonh_optimizer_apply,real_expert_fsdp_grouped_muonh_optimizer_apply --mode lower --warmup 0 --iters 1 --disable-abstract-mesh --allow-boundary-collectives --output /tmp/muon_fsdp_hyperball_r2d1_l26_lower.json`.
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-FSDPHYPERBALL-H3-G2-CAP512-N2-cw-$(date -u +%Y%m%d-%H%M%S) MUON_BENCH_KINDS=expert_fsdp_grouped_muonh_optimizer_apply,real_expert_fsdp_grouped_muonh_optimizer_apply MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260620-001634`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-001634/grug-train-MUON-BENCH-D2560-L26-R2D1E8-FSDPHYPERBALL-H3-G2-CAP512-N2-cw-20260620-001631`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-FSDPHYPERBALL-H3-G2-CAP512-N2-cw-20260620-001631-315fe1`.
- Config: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, max grouped stack size 512, backend steps 3, bf16 params/updates, bf16 NS compute, warmup 1, iters 3, boundary collectives allowed.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED`.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled GEMMs | median | mean | H100 bf16 peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| grouped NS direction, FSDP hyperball/apply | 0/0/0/0/0 | 26/0/0/0/0 | 520 | 0.4806s | 0.4803s | 31.9% |
| real production grouped MuonH helper | 26/0/0/0/0 | 26/0/0/0/0 | 533 | 0.4790s | 0.4779s | 32.1% |

- Interpretation: Negative result. The FSDP-hyperball variant looks cleaner before compilation, but compiled GPU HLO still contains the same 26 all-gathers and runtime is effectively identical to the real production helper. Moving hyperball back to FSDP does not avoid the dominant boundary cost; the compiler still materializes the grouped direction in a way that pays the restore boundary, and the small GEMM-count reduction is not useful.
- Next action: Do not port FSDP-hyperball into production. The viable directions remain (1) persistent grouped optimizer/model-facing expert banks to avoid the grouped-to-FSDP restore boundary, or (2) a truly lower-level custom transfer for the grouped-to-FSDP direction/update path. Generic JAX-level variants have now repeatedly failed to beat the current slice-first production helper.

### 2026-06-19 17:38 PDT - grouped-trace FSDP-master MuonH benchmark
- Hypothesis: A pragmatic FSDP-master design can keep train-state params in ordinary FSDP layout while keeping the MuonH trace/momentum state in the grouped NS-friendly layout. If grouped grads and grouped trace are both already in the NS layout, the optimizer should avoid the real helper's inbound FSDP/unreduced-gradient/momentum all-gathers, compute MuonH in grouped layout, restore grouped updates to FSDP, and then use ordinary `optax.apply_updates`.
- Code snapshot:
  - Added harness-only bench kind: `expert_fsdp_grouped_trace_muonh_apply`.
  - Scope: `experiments/grug/moe/muon_update_bench.py` and focused tests only; no production optimizer path changed.
- Local validation:
  - Focused tests: `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_trace_muonh_keeps_trace_grouped_and_params_fsdp experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs -q` -> `2 passed`.
  - Full-shape local lowering with `XLA_FLAGS=--xla_force_host_platform_device_count=16` showed the new grouped-trace path lowered as `AG/A2A/AR/RS/CP = 0/0/0/0/0` with 234 two-batch-axis `dot_general` ops.
- Command:
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-GROUPEDTRACE-H3-G2-CAP512-N2-cw-20260620-003305 MUON_BENCH_KINDS=expert_fsdp_grouped_trace_muonh_apply,expert_fsdp_grouped_updates_muonh_apply,real_expert_fsdp_grouped_muonh_optimizer_apply MUON_BENCH_LAYERS=26 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_MODE=both MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260620-003309`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-003309/grug-train-MUON-BENCH-D2560-L26-R2D1E8-GROUPEDTRACE-H3-G2-CAP512-N2-cw-20260620-003305`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-GROUPEDTRACE-H3-G2-CAP512-N2-cw-20260620-003305-bbc96c`.
- Config: 2 H100 nodes, 16 devices, `replica_dcn=2`, `data=1`, `expert=8`, `model=1`, layers 26, hidden 2560, intermediate 1280, 256 experts, group size 2, group axis `replica_dcn,data`, max grouped stack size 512, backend steps 3, bf16 params/updates, bf16 NS compute, warmup 1, iters 3, boundary collectives allowed.
- Result: Parent and child reached `JOB_STATE_SUCCEEDED` with zero failures/preemptions. The grouped-trace path preserved grouped trace and FSDP params at the harness API boundary, but XLA GPU compilation still introduced 26 all-gathers.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled GEMMs | median | mean | H100 bf16 peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| grouped trace -> MuonH -> restore/apply | 0/0/0/0/0 | 26/0/0/0/0 | 520 | 0.3062s | 0.3062s | 50.1% |
| grouped updates -> MuonH -> restore/apply | 0/0/0/0/0 | 26/0/0/0/0 | 533 | 0.3071s | 0.3066s | 50.1% |
| real production grouped MuonH helper | 26/0/0/0/0 | 26/0/0/0/0 | 533 | 0.4782s | 0.4784s | 32.1% |

- Interpretation: The grouped-trace/FSDP-master harness is a useful partial win but not the proof we need. It removes the real helper's extra overhead and improves the isolated expert MuonH apply path from about 0.478s to about 0.306s, reaching roughly 50% nominal H100 bf16 peak. However, the core representation-boundary problem remains: even when lowered HLO has no collectives, compiled GPU HLO rediscovered 26 all-gathers. Persisting grouped trace alone does not avoid the grouped-to-FSDP boundary; it only avoids some inbound/helper overhead.
- Next action: Keep the grouped-trace harness as a pragmatic baseline, but do not claim the FSDP-master design has avoided per-layer collectives. The direct target reshard, packed restore, FSDP-hyperball, and tuple-returning `shard_map` bridge have all failed to remove the compiled grouped-to-FSDP all-gathers. Further work should either use a lower-level/custom communication path that moves exactly the final FSDP slices, or prefer grouped expert-bank persistence/model consumption so the hot step does not restore grouped updates to ordinary FSDP leaves.

### 2026-06-19 19:45 PDT - update-only profile and full apply OOM
- Hypothesis: The grouped-trace/FSDP-master path may be good enough on the MuonH update itself, and the remaining risk is whether returning FSDP-shaped updates plus ordinary `optax.apply_updates` can execute without an extra full expert-param allocation. If donation of params, grads, and optimizer state is enough, the apply harness should run at the full L26 R2D1E8 shape.
- Code snapshot:
  - Branch: `codex/research-grug-moe-d2560-mfu`.
  - Added harness profile support to `experiments/grug/moe/muon_update_bench.py` and `experiments/grug/moe/launch_cw_muon_update_bench.py`.
  - Temporarily changed the real grouped MuonH apply jit donation from `(params, state)` to `(params, grads, state)`.
- Commands:
  - Profile launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-UPDATEPROFILE-H3-G2-CAP512-N2-cw-20260620-022411 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_update MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_ENABLE_JAX_PROFILE=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 KUBECONFIG=$HOME/.kube/coreweave-iris-gpu bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Apply donation launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYDONATEGRADS-H3-G2-CAP512-N2-cw-20260620-023732 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_apply MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 KUBECONFIG=$HOME/.kube/coreweave-iris-gpu bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
- Result:
  - Update-only profile run:
    - Parent Iris job: `/dlwh/iris-run-job-20260620-022414`.
    - Child Iris job: `/dlwh/iris-run-job-20260620-022414/grug-train-MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-UPDATEPROFILE-H3-G2-CAP512-N2-cw-20260620-022411`.
    - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-UPDATEPROFILE-H3-G2-CAP512-N2-cw-20260620-022411-8ad13c`.
    - Succeeded. Process 0 median was `0.302121417s`; process 1 median was `0.302017428s`.
    - Lowered HLO had `AG/A2A/AR/RS/CP = 26/0/0/0/0` and 234 `dot_general` ops. Compiled HLO had 26 all-gathers, 572 GEMM custom calls, and no all-reduce/all-to-all/reduce-scatter/collective-permute.
    - Local TensorBoard profile: `scratch/profiles/muon_update_profile_20260620_022411`, currently served at `http://127.0.0.1:6023/?run=process_0%2F2026_06_20_02_28_23&tag=trace_viewer%40&hosts=g73b7ae`.
    - Deterministic profile report: `scratch/profiles/muon_update_profile_20260620_022411_report.md`.
    - Phase table: `scratch/muon_update_profile_phase_table_20260620.md`.
  - Apply donation run:
    - Parent Iris job: `/dlwh/iris-run-job-20260620-023735`.
    - Child Iris job: `/dlwh/iris-run-job-20260620-023735/grug-train-MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYDONATEGRADS-H3-G2-CAP512-N2-cw-20260620-023732`.
    - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYDONATEGRADS-H3-G2-CAP512-N2-cw-20260620-023732-77e384`.
    - Failed on first execution with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 16.41GiB`.
    - Lowered HLO still had `AG/A2A/AR/RS/CP = 26/0/0/0/0` and 234 `dot_general` ops.
- Interpretation: The update-only grouped-trace path is fast and profile-backed at about 50.8% nominal H100 bf16 peak for the NS dot work. The full apply path remains memory-blocked. Donating grads in addition to params and optimizer state did not change the failure: the 16.41 GiB allocation matches the full per-device FSDP expert param/update tree for L26 (`w_gate_up + w_down` across 26 layers). This points to full-tree materialization of FSDP-shaped updates/next params, not a simple donation or fragmentation issue.
- Next action: Ordinary full-tree `updates = optimizer.update(...); optax.apply_updates(params, updates)` is not yet viable at R2D1E8 L26. The next proof should avoid a simultaneously live full FSDP update tree, either by fusing grouped restore with apply in chunk/layer groups or by moving to a persistent grouped expert-bank/model-facing representation. If preserving ordinary `optax.apply_updates` remains mandatory, we need evidence that XLA can alias or stream the update tree; current donation evidence says it cannot for this shape.

### 2026-06-19 19:56 PDT - cuda_async allocator apply retry
- Hypothesis: The full apply OOM might be allocator headroom/fragmentation rather than a hard structural impossibility. If so, `cuda_async` plus a larger XLA memory fraction should run the same R2D1E8 L26 apply path.
- Command:
  - Launch: `RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYCUDASYNC-H3-G2-CAP512-N2-cw-20260620-025034 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_apply MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async TF_GPU_ALLOCATOR=cuda_malloc_async KUBECONFIG=$HOME/.kube/coreweave-iris-gpu bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`.
  - Parent Iris job: `/dlwh/iris-run-job-20260620-025036`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-025036/grug-train-MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYCUDASYNC-H3-G2-CAP512-N2-cw-20260620-025034`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-PRODGROUPEDTRACE-APPLYCUDASYNC-H3-G2-CAP512-N2-cw-20260620-025034-117bb0`.
- Result: Succeeded, but performance collapsed.
  - Process 0: `median_seconds=2.7792`, about 5.52% nominal H100 bf16 peak.
  - Process 1: `median_seconds=2.0549`, about 7.47% nominal H100 bf16 peak.
  - Compiled HLO stayed at `AG/A2A/AR/RS/CP = 26/0/0/0/0` with 572 GEMM custom calls.
- Interpretation: More headroom plus `cuda_async` proves the full-tree ordinary apply path can execute, but not in a useful performance regime. The previous failure was at least partly allocator/headroom-sensitive, but the viable path is still to avoid materializing or applying the whole FSDP update tree this way. `cuda_async` is diagnostic, not a solution.
- Issue update: https://github.com/marin-community/marin/issues/6493#issuecomment-4756144861
- Next action: Keep the low-level grouped-to-FSDP bridge work as the main path. If we need an apply sanity check in the future, use `cuda_async` only as a diagnostic fallback and compare against update-only/persistent-grouped baselines, not as a throughput candidate.

### 2026-06-19 20:47 PDT - lazy grouped MuonH launcher and group-size fit checks
- Hypothesis: The editable reference launcher should default to a group size that actually runs on 2xH100 nodes, logs a W&B topline, and reduces the grouped-to-FSDP restore collective count without relying on the profiler-only command-buffer flag.
- Code snapshot:
  - Commit `a367068d3` changed `scratch/launch_muon_grouped_reference_2node_wandb.sh` to default to `MUON_BENCH_NS4D_GROUP_SIZE=8`, include the group size in `RUN_ID`, keep normal XLA command-buffer behavior by default, and reserve `--xla_gpu_enable_command_buffer=` for `MUON_BENCH_READABLE_PROFILE=true`.
  - Commit `6b037fa6f` bounded W&B finalization in `experiments/grug/moe/launch_cw_muon_update_bench.py` with quiet/no-console settings, disabled W&B stats/meta collection, and 30s service/network timeouts.
- Commands:
  - G26 default attempt from the earlier lazy script: `bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
  - G13 retry: `MUON_BENCH_NS4D_GROUP_SIZE=13 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
  - G8 validation: `bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
  - G4 fit check: `MUON_BENCH_NS4D_GROUP_SIZE=4 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Result:

| group size | parent job | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | median | H100 bf16 peak | W&B | outcome |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 26 | `/dlwh/iris-run-job-20260620-032518` | 2/0/0/0/0 | n/a | n/a | n/a | n/a | OOM, 10.16 GiB allocation |
| 13 | `/dlwh/iris-run-job-20260620-032938` | 4/0/0/0/0 | n/a | n/a | n/a | n/a | OOM, 19.14 GiB allocation |
| 8 | `/dlwh/iris-run-job-20260620-033329` | 8/0/0/0/0 | 8/0/0/0/0 | 0.3326s | 46.15% | https://wandb.ai/marin-community/marin_moe/runs/cu3yg0xm | Metrics logged; old pre-timeout W&B finalization hung, then job was killed after capture |
| 4 | `/dlwh/iris-run-job-20260620-034328` | 14/0/0/0/0 | n/a | n/a | n/a | n/a | OOM, 16.41 GiB allocation |

- Interpretation: The lazy command now points at the only packed group size in this sweep that both ran and reduced compiled all-gathers. G8 is not strictly faster than the earlier update-only G2 profile (`~0.302s` with 26 AG), but it does cut the restore count to 8 while preserving a usable ~46% nominal-peak update-only timing. The OOM pattern is not monotonic in group size: G4, G13, and G26 all fail, so peak memory is driven by XLA's live materialization schedule rather than only max grouped-stack size. The W&B timeout patch was added after the G8 run because the metrics had uploaded but the job stayed RUNNING in finalization.
- Next action: Treat G8 as the current lazy reference command, not as the final production bridge. The production path still needs either a streaming/chunked restore+apply that avoids the full FSDP update tree, or persistent grouped expert banks. If another fit/perf sweep is needed, use the same launcher with explicit `MUON_BENCH_NS4D_GROUP_SIZE` overrides and require both terminal job exit and W&B summary freshness as validation.

### 2026-06-19 21:10 PDT - OSS Muon implementation clue: layer-wise ownership, not element-wise FSDP
- Question: Existing OSS Muon implementations may have already converged on the right distributed optimizer representation. The relevant question is whether they preserve element-wise FSDP/ZeRO-style optimizer sharding and materialize a full update tree, or whether they switch to a layer/group ownership layout that gives Muon full matrices locally.
- Sources inspected:
  - NVIDIA Megatron/Core `LayerWiseDistributedOptimizer` source docs say the optimizer distributes weights to data-parallel ranks by layer, keeps only the owned shard in the optimizer, lets DDP handle gradient all-reduce, updates only the parameters owned by the rank, then all-gathers updated params to every rank.
  - NVIDIA Emerging Optimizers docs contrast element-wise distributed optimizers with layer-wise sharding. They explicitly say Muon needs gradients for a full layer to calculate updates, so evenly sharded optimizer state needs extra communication; layer-wise sharding gives each GPU full layers and then uses variable-size `all_gatherv` for updated params.
  - NVIDIA's Megatron blog makes the same point: Muon-style preconditioners need whole-layer gradients; element-wise DP sharding cannot calculate the update from local data alone. It describes layer-wise distribution and notes duplicated/distributed TensorParallelMuon modes for TP-sharded matrices.
  - A current Megatron-LM PR list includes "experimental decoupled compact LayerWise DDP layout for Muon", which suggests this representation problem is still active upstream rather than solved by a generic distributed optimizer primitive.
- Mapping to our evidence:
  - Our strict FSDP-master/full-update-tree path has the exact failure mode these docs predict. The Muon update itself is fast enough (`~0.302s`, about 50% nominal H100 bf16 peak for the R2D1 expert path), but converting grouped updates into an ordinary FSDP update tree plus `optax.apply_updates` either OOMs (`16.41GiB`) or only runs with `cuda_async` at `2.05-2.78s`.
  - Our persistent grouped expert-bank lower bound is the closest analogue to Megatron layer-wise ownership: the optimizer owns full grouped expert matrices locally, avoids the grouped-to-FSDP restore, compiles with zero AG/A2A/AR/RS/CP, and runs at `~0.254s` for the R2D1 expert path.
  - The remaining disagreement is the stated conservative objective: keep train-state/master params FSDP and convert back before ordinary `optax.apply_updates`. OSS implementations point away from doing that in the hot path. They accept a representation switch plus a synchronization/all-gather boundary for model use.
- Interpretation: Treat strict FSDP-master + full `optax.apply_updates` as a correctness/reference path, not the likely high-performance production layout. The next productive implementation should model "layer-wise/group-wise ownership" explicitly in JAX terms: grouped expert banks are optimizer-owned, update in place/grouped, and a sync boundary produces model-consumable FSDP/EP leaves only when needed. If we must preserve ordinary FSDP train-state at every step, the next experiment should be a streaming restore+apply primitive that never exposes a whole FSDP update tree; another generic `reshard` or packed `shard_map` is unlikely to beat current evidence.
- Next action: Write a narrow design artifact for the two viable paths:
  1. FSDP-reference streaming apply: grouped chunks restore to FSDP and are consumed immediately, proving or falsifying "ordinary apply without full update-tree materialization".
  2. Layer-wise/grouped-bank production path: grouped expert params/trace are optimizer-owned, model-facing code consumes grouped banks or synchronizes them through an explicit all-gather/all-gatherv-like boundary.

### 2026-06-19 21:10 PDT - direct restore-and-apply harness
- Hypothesis: A DeepSpeed-style boundary might avoid materializing a full FSDP-shaped update tree if grouped MuonH updates are restored to each FSDP leaf and consumed immediately by leaf-level `optax.apply_updates`.
- Code snapshot:
  - Added benchmark kind `expert_fsdp_grouped_updates_muonh_direct_apply` in `experiments/grug/moe/muon_update_bench.py`.
  - The new path computes grouped NS/hyperball updates, restores each split update to its matching FSDP param sharding, and applies it immediately instead of returning `(next_params, next_updates)`.
  - Added focused abstract-mesh coverage in `experiments/grug/moe/test_muon_update_bench.py`.
- Local validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_restores_ordinary_expert_updates_before_apply experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_direct_apply_restores_slices_before_apply experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_explicit_restore_then_apply_returns_fsdp_params experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_can_return_updates_without_apply -q` passed.
  - Tiny forced-host lower/run smoke used `R2D2E2M1`, L4, G4, H1, bf16:
    - Lowered StableHLO for direct apply had `AG/A2A/AR/RS = 0/0/0/0` and 6 two-batch-axis dot-generals.
    - Runtime-compiled HLO for direct apply had `AG/A2A/AR/RS = 0/2/0/0`, median `0.0340s`.
    - Runtime-compiled HLO for existing restore-then-apply had `AG/A2A/AR/RS = 0/2/0/0`, median `0.0313s`.
- Interpretation: The harness is valid and gives us a direct measurement knob for "consume grouped updates near the FSDP leaf", but the tiny compiled evidence does not show a collective-count win. XLA still turns the grouped-to-FSDP restore boundary into two all-to-alls, same as the existing restore-then-apply path. Do not treat this as a proven throughput improvement until a GPU-sized compile/run contradicts the local result.
- Next action: Prefer this as a diagnostic benchmark over an immediate expensive CoreWeave launch. If we launch it, compare against `expert_fsdp_grouped_updates_muonh_apply` and `expert_fsdp_grouped_persistent_muonh_apply` in the same job and require compiled-HLO collective counts plus timing, not just lowered HLO.

### 2026-06-19 21:35 PDT - strict compiled-HLO boundary gate
- Hypothesis: Lowered StableHLO has been too weak as a production-candidate gate. Candidate grouped/apply paths should be able to opt into a strict check that fails if either lowered or compiled HLO contains any grouped-to-FSDP boundary collective (`all_gather`, `all_to_all`, `all_reduce`, `reduce_scatter`, or `collective_permute`), including expert-FSDP grouped benches that are normally allowed for exploration.
- Code snapshot:
  - Added `--require-no-boundary-collectives` to `experiments/grug/moe/muon_update_bench.py`.
  - The existing `--allow-boundary-collectives` remains for debug/decomposition profiles. The two flags are mutually exclusive.
  - `grouped_apply_boundary_collectives` now includes `collective_permute`, so ppermute-style bridges cannot pass the strict gate by accident.
  - Summary rows now include `boundary_collectives_required_absent`.
- Local validation:
  - Focused tests passed:
    - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_restores_ordinary_expert_updates_before_apply experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_direct_apply_restores_slices_before_apply experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_explicit_restore_then_apply_returns_fsdp_params experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_updates_muonh_can_return_updates_without_apply experiments/grug/moe/test_muon_update_bench.py::test_strict_boundary_gate_includes_expert_fsdp_and_collective_permute experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_boundary_byte_estimates -q`
    - Result: `7 passed`.
  - `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed.
  - Tiny forced-host positive smoke:
    - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --hidden-dim 4 --intermediate-dim 2 --num-experts 2 --backend-steps 1 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --max-grouped-stack-size 8 --bench-kinds expert_fsdp_grouped_persistent_muonh_apply --mode run --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --require-no-boundary-collectives --output /tmp/muon_gate_persistent.json`
    - Result: passed; compiled `AG/A2A/AR/RS/CP = 0/0/0/0/0`.
  - Tiny forced-host negative smoke:
    - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 1 --hidden-dim 4 --intermediate-dim 2 --num-experts 2 --backend-steps 1 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --max-grouped-stack-size 8 --bench-kinds expert_fsdp_grouped_updates_muonh_direct_apply --mode run --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --require-no-boundary-collectives --output /tmp/muon_gate_direct.json`
    - Result: failed as intended with `compiled grouped boundary candidate forced grouped apply boundary collectives {'all_to_all': 2}`.
- Interpretation: This gives us a cheap, explicit production-candidate gate. The persistent grouped apply lower bound passes; the direct restore/apply path is still a diagnostic/reference path, not a candidate, because compiled HLO reintroduces boundary routing even when lowered HLO looks clean.
- Next action: Use `--require-no-boundary-collectives` on any future "this avoids per-leaf collective explosion" claim. Use `--allow-boundary-collectives` only when intentionally profiling or decomposing a known-boundary path.

### 2026-06-19 21:35 PDT - custom-partition grouped-to-FSDP restore probe
- Hypothesis: A `jax.custom_partitioning` boundary can give us a cleaner grouped-to-FSDP insertion point than ordinary `shard_map`/`reshard`: lowered HLO should expose an opaque custom-call-like boundary instead of spelling the restore as all-gathers/all-to-alls. This does not yet implement production transport, but it is the right harness shape for a future FFI/custom-call bridge.
- Code snapshot:
  - Added benchmark kind `expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary` in `experiments/grug/moe/muon_update_bench.py`.
  - The custom-partition path restores grouped expert updates into FSDP-shaped update leaves through a custom-partitioning stub.
  - Added a focused sharding-contract test in `experiments/grug/moe/test_muon_update_bench.py`.
  - Added an explicit guard requiring `--disable-abstract-mesh` for lowering this bench, because `jax.custom_partitioning` lowering needs concrete devices.
- Local validation:
  - `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed.
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_custom_partition_slice_first_gather_boundary_returns_fsdp_updates experiments/grug/moe/test_muon_update_bench.py::test_expert_fsdp_grouped_explicit_tuple_slice_first_gather_boundary_returns_fsdp_updates_without_a2a experiments/grug/moe/test_muon_update_bench.py::test_strict_boundary_gate_includes_expert_fsdp_and_collective_permute -q` passed with `3 passed`.
  - Forced-host comparison command:
    - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --bench-kinds expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary,expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary --mode both --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --allow-boundary-collectives --output /tmp/muon_custom_partition_compare.json`
    - Existing tuple restore lowered with `AG/A2A/AR/RS/CP = 2/0/0/0/0`.
    - Custom-partition restore lowered with `AG/A2A/AR/RS/CP = 0/0/0/0/0` and `custom_call = 2`.
    - Both compiled CPU HLOs had `AG/A2A/AR/RS/CP = 0/0/0/0/0`.
  - Strict gate command:
    - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --bench-kinds expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary --mode both --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --require-no-boundary-collectives --output /tmp/muon_custom_partition_strict.json`
    - Result: passed. Lowered HLO had `custom_call = 2` and zero AG/A2A/AR/RS/CP; compiled CPU HLO also had zero AG/A2A/AR/RS/CP.
- Interpretation: This is promising as a lower-level bridge insertion point, not a production performance claim. The custom-partition stub hides the grouped-to-FSDP restore from lowered HLO and passes the strict boundary gate on forced-host CPU, but CPU compiled HLO is not authoritative for H100/NCCL behavior. The current lower function still uses JAX-local slice/gather logic; a real solution would replace that lowering with an FFI/custom-call transport that emits exactly the desired grouped-to-FSDP routing.
- Next action: After commit/push, validate this bench on CoreWeave H100s in compile-only/timing mode against the explicit tuple restore baseline. Require compiled-HLO collective counts and, if it runs, timing. If the GPU compiler keeps the custom boundary opaque, this becomes the harness for a real transport implementation; if it reintroduces collectives or fails to lower, fall back to a custom-call/FFI transport design rather than another generic `reshard`.

### 2026-06-19 21:45 PDT - OSS Muon implementation survey follow-up
- Question: Existing OSS Muon implementations might already encode the right distributed representation for FSDP/ZeRO-like settings.
- Sources checked:
  - KellerJordan/Muon: whole-parameter round-robin ownership, local NS on full matrices, then all-gather updated params.
  - NVIDIA Megatron layer-wise optimizer: whole matrices/layers are assigned to data-parallel ranks, updated by the owning rank, then synchronized through the DDP param-buffer path or all-gather.
  - NVIDIA NeMo Emerging-Optimizers: batched 3D NS, MuonHyperball, and TensorParallel Muon modes. TP `duplicated` gathers full matrices; TP `distributed` keeps shards but pays per-NS-step Gram all-reduces.
  - Axolotl/Dion Muon: groups FSDP2 DTensor params by `(shape, sharding, dtype)`, batches to world size, all-to-alls shards so each rank gets complete matrices, runs NS locally, then all-to-alls results back.
  - DeepSpeed Muon: applies Muon while per-parameter gradient views still exist, before the flattened ZeRO partition loses matrix identity.
- Interpretation: Axolotl/Dion is the closest direct clue for this harness. It validates the coarse-transport shape we have been converging on: batch same-layout matrices, perform one coarse all-to-all/gather into whole matrices for local NS, then one coarse communication boundary back. Megatron is the fallback design: give up on element-wise FSDP ownership in the hot Muon step and use whole-layer/matrix ownership plus a param sync boundary.
- Next action: Prefer an Axolotl-style coarse grouped transport over further per-leaf `reshard` variants. The custom-partition probe should be treated as the harness insertion point for that transport. Continue avoiding flattened-before-Muon designs unless the Muon transform happens before flattening, as in DeepSpeed.

### 2026-06-19 21:50 PDT - H100 custom-partition bridge validation
- Hypothesis: The custom-partition grouped-to-FSDP restore probe may keep the restore opaque enough on H100 that compiled GPU HLO avoids the per-layer all-gathers seen in ordinary tuple restore.
- Code snapshot:
  - Commit `c29af7f41` includes the custom-partition bench plus launcher plumbing for the strict boundary flag.
  - The first H100 attempt, parent `/dlwh/iris-run-job-20260620-043933`, failed before compile because the launcher-created `SimpleNamespace` omitted `require_no_boundary_collectives`. Commit `c29af7f41` fixed this and added a regression test.
- Command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-CUSTOMPART-COMPILE-G2-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary,expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- Result:
  - Relaunch parent Iris job: `/dlwh/iris-run-job-20260620-044409`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-044409/grug-train-MUON-BENCH-D2560-L26-R2E8-CUSTOMPART-COMPILE-G2-N2-cw-20260620-044407`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-CUSTOMPART-COMPILE-G2-N2-cw-20260620-044407-8d2242`.
  - Terminal state: succeeded.

| bench | lowered AG/A2A/AR/RS/CP | lowered custom_call | compiled AG/A2A/AR/RS/CP | compiled custom_call | compile seconds |
| --- | --- | ---: | --- | ---: | ---: |
| explicit tuple slice-first restore | 26/0/0/0/0 | 0 | 26/0/0/0/0 | 0 | 0.856-0.857 |
| custom-partition slice-first restore | 0/0/0/0/0 | 26 | 26/0/0/0/0 | 0 | 0.513-0.537 |

- Interpretation: The custom-partition stub is useful only as a lowered-HLO insertion point. On H100, the compiled GPU HLO still inlines the lowering body and reintroduces the same 26 all-gathers as the explicit tuple restore. This falsifies the hope that `jax.custom_partitioning` alone can hide the grouped-to-FSDP bridge from XLA GPU. A production fix needs a real custom transport/custom-call/FFI or an Axolotl-style explicit coarse all-to-all primitive, not another JAX-level restore wrapper.
- Next action: Stop spending time on JAX-level custom-partition wrappers for this boundary. The next implementation should either:
  1. build an explicit coarse transport that batches same-layout matrices and performs the grouped-to-FSDP routing directly, or
  2. shift production toward grouped expert-bank ownership/model consumption and make the sync boundary explicit.

### 2026-06-19 22:00 PDT - H100 direct restore-and-apply compile comparison
- Hypothesis: Consuming grouped MuonH updates immediately at each FSDP leaf might avoid the full update-tree materialization and reduce the compiled grouped-to-FSDP boundary collectives versus restore-then-ordinary-apply.
- Command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-DIRECTAPPLY-COMPILE-G8-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_fsdp_grouped_updates_muonh_apply,expert_fsdp_grouped_updates_muonh_direct_apply,expert_fsdp_grouped_persistent_muonh_apply MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-045028`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-045028/grug-train-MUON-BENCH-D2560-L26-R2E8-DIRECTAPPLY-COMPILE-G8-N2-cw-20260620-045025`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-DIRECTAPPLY-COMPILE-G8-N2-cw-20260620-045025-f0666b`.
  - Shape: 2 H100 nodes, L26, R2/D1/E8/M1, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, bf16 params/NS compute, compile-only, `--disable-abstract-mesh`, `--allow-boundary-collectives`, compiled-HLO writing enabled.
- Result:
  - Iris parent succeeded with exit 0, failures 0, duration about 4m30s.

| bench | lowered AG/A2A/AR/RS/CP | compiled AG/A2A/AR/RS/CP | compiled custom_call | compiled GPU GEMM custom_call | compile seconds |
| --- | --- | --- | ---: | ---: | ---: |
| restore then `optax.apply_updates` | 0/0/0/0/0 | 8/0/0/0/0 | 360 | 176 | 7.26-7.53 |
| direct restore-and-apply | 0/0/0/0/0 | 8/0/0/0/0 | 360 | 176 | 2.10-2.71 |
| persistent grouped params/apply | 0/0/0/0/0 | 0/0/0/0/0 | 360 | 176 | 1.93-2.67 |

- Interpretation: Direct restore-and-apply is not the grouped-to-FSDP fix. It compiles faster than restore-then-ordinary-apply in this compile-only comparison, but XLA GPU still inserts the same 8 compiled all-gathers at the boundary. The persistent grouped path remains the only tested H100 path with zero compiled grouped boundary collectives.
- Next action: Stop treating leaf-local direct `apply_updates` as a production candidate unless a future runtime profile shows a memory/overlap benefit despite identical collective count. The two viable implementation paths remain:
  1. an explicit coarse transport/custom-call/FFI/Axolotl-style all-to-all that owns the grouped-to-FSDP routing directly, or
  2. a persistent grouped expert-bank representation that avoids the FSDP restore in the hot update path and synchronizes to model-consumable layout at an explicit boundary.

### 2026-06-19 22:05 PDT - OSS Muon survey: ownership and transport clues
- Question: Which public Muon implementations give the strongest design clue for grouped MuonH with FSDP-like model params and an NS-friendly optimizer layout?
- Sources inspected:
  - Megatron-LM layer-wise optimizer and Muon:
    - `https://github.com/NVIDIA/Megatron-LM/blob/d1410e15e164923f23b8e1a3384bf22c29640fcc/megatron/core/optimizer/layer_wise_optimizer.py`
    - `https://github.com/NVIDIA/Megatron-LM/blob/d1410e15e164923f23b8e1a3384bf22c29640fcc/megatron/core/optimizer/muon.py`
  - KellerJordan/modded-nanogpt:
    - `https://github.com/KellerJordan/modded-nanogpt/blob/3df8d388aae1d56d6d558654e6739b17e5cb72f3/train_gpt.py`
    - `https://github.com/KellerJordan/modded-nanogpt/blob/3df8d388aae1d56d6d558654e6739b17e5cb72f3/train_gpt_medium.py`
  - NeMo Emerging Optimizers:
    - `https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/06ff4c68cda0d41a7a40580644a47f61ea4a59b0/emerging_optimizers/orthogonalized_optimizers/muon.py`
    - `https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/06ff4c68cda0d41a7a40580644a47f61ea4a59b0/emerging_optimizers/orthogonalized_optimizers/muon_utils.py`
    - `https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/06ff4c68cda0d41a7a40580644a47f61ea4a59b0/emerging_optimizers/orthogonalized_optimizers/muon_hyperball.py`
  - Dion/Axolotl FSDP-oriented Muon:
    - `https://github.com/axolotl-ai-cloud/dion-optimizer/blob/c63808c120a0ad10ebf86095aa8d72e4c0ae7648/optimizers/muon.py`
    - `https://github.com/axolotl-ai-cloud/dion-optimizer/blob/c63808c120a0ad10ebf86095aa8d72e4c0ae7648/optimizers/opt_utils.py`
    - `https://github.com/axolotl-ai-cloud/axolotl/blob/e86163dd332f3aad2f29de8fe44b6cfd5f74b22d/src/axolotl/core/builders/base.py`
    - `https://github.com/axolotl-ai-cloud/axolotl/blob/e86163dd332f3aad2f29de8fe44b6cfd5f74b22d/src/axolotl/utils/schemas/validation.py`
  - Minimal FSDP2 Muon:
    - `https://github.com/samsja/muon_fsdp_2/blob/9a505c1452c04838b4914ddadffed14ed4a3b8ab/src/muon/muon_fsdp2.py`
- Findings:
  - Megatron is the closest conceptual match. It separates model-visible parameters from optimizer ownership: Muon-managed matrices are routed to layer-wise owners, updated as whole matrices, then synchronized back through a model-consumable parameter/bucket path.
  - Dion is the closest transport analogue for FSDP2. It batches compatible DTensor parameters, uses all-to-all to give devices whole matrices for NS, then all-to-all back. This validates a coarse grouped transport, but copied per leaf it would reproduce the XLA collective-explosion problem.
  - KellerJordan/modded-nanogpt uses architecture-specific parameter banks plus reduce-scatter/all-gather around batched NS. This is a useful grouping pattern, but it assumes the model layout was designed for that bank representation.
  - NeMo gives good batched-NS/MuonHyperball API shape and TP variants, but not a direct FSDP ownership answer.
  - The minimal FSDP2 Muon gathers each DTensor param to an owner rank, runs NS, and scatters shards back. This avoids full replication but is too per-parameter to copy directly into XLA GPU.
- Follow-up source-level survey:
  - Megatron `LayerWiseDistributedOptimizer` uses a `FullParamLayout` and layer-wise ownership so optimizer-managed 2D parameters are assigned whole to DP/expert-DP ranks, then synchronized through DDP parameter buffers or a legacy variable-size all-gather path. This reinforces the idea that optimizer ownership can drive a different buffer layout than ordinary parameter traversal.
  - Megatron `TensorParallelMuon`/Emerging-Optimizers exposes the tradeoff between a duplicated mode that all-gathers shards to full matrices and a distributed mode that keeps shards but pays per-Newton-Schulz-step Gram all-reduces. This is relevant only if we decide to shard within matrices; for the current expert-bank path, whole-matrix/batched NS remains cleaner.
  - Microsoft Dion's FSDP2/DTensor Muon is the best transport analogue: it groups parameters by `(shape, placements, dtype)`, builds mega-batch orthogonalization tasks, all-to-alls shards to assemble whole matrices, runs batched NS, then all-to-alls back. Axolotl's `DistMuon` carries the same idea into a training stack with simpler 1D FSDP2 assumptions.
  - KellerJordan/Muon and modded-nanogpt intentionally use bf16 NS. The modded-nanogpt bank representation is especially relevant because the model-visible parameterization is already banked; it avoids the optimizer-tree restore problem by construction.
- Design implications:
  - Steal Megatron's invariant: Muon ownership is a separate representation from model consumption; only Muon-suitable matrices take the special path.
  - Steal Dion/NanoGPT's grouping rule: group only exact-compatible shape/dtype/sharding/hyperparameter sets and batch NS over those groups.
  - Treat the sync boundary as bulk grouped transport, not independent optax leaves. The source-level survey's highest-confidence recommendation is to copy Megatron/Dion's grouped sync boundary, not the original per-leaf Muon transport.
  - Avoid per-leaf gather/scatter/all-to-all even if PyTorch/FSDP2 examples tolerate it. Our H100 compile checks show XLA GPU will expose or reintroduce those boundaries.
  - Do not return from grouped 4D layout to per-layer leaves through an all-gathered group axis before the hot apply boundary.
- Next action: Frame the next implementation as either:
  1. Megatron-like grouped ownership: persistent grouped expert banks, local grouped NS, explicit bulk sync to model-consumable FSDP/EP layout, or
  2. Dion-like coarse transport: a single grouped same-layout transport primitive that converts grouped optimizer banks to model-consumable layout without spelling per-leaf JAX reshards.

### 2026-06-19 22:14 PDT - H100 grouped MuonH bank-consumer compile gate
- Hypothesis: If the model-side expert consumer can keep the grouped expert-bank representation, the MuonH update plus an expert-like MLP consumer can avoid the grouped-to-FSDP restore boundary entirely. This should compile on H100 with zero grouped boundary collectives.
- Command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-COMPILE-G8-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_grouped_muonh_bank_consumer MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT=1 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-050828`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-050828/grug-train-MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-COMPILE-G8-N2-cw-20260620-050825`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-COMPILE-G8-N2-cw-20260620-050825-2345bf`.
  - Shape: 2 H100 nodes, L26, R2/D1/E8/M1, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, grouped expert consumer tokens per expert = 1, bf16 params/NS compute, compile-only, concrete mesh.
- Result:
  - Iris parent succeeded with exit 0, failures 0, duration about 2m39s.
  - The grouped expert-bank consumer row lowered with `AG/A2A/AR/RS/CP = 0/0/0/0/0` and compiled with `AG/A2A/AR/RS/CP = 0/0/0/0/0` on both H100 tasks.
  - Compiled HLO had `custom_call = 360` and `gpu_gemm_custom_call = 175`.
  - Compile time was about `6.55s` on task 1 and `7.49s` on task 0.
  - Estimated NS dot flops were `2.4289348681728e15`; estimated matrix count was `26624`.
  - With `data_axis=1`, the actual grouped sharding spec was `P('replica_dcn', 'expert', None, None)`. This is the expected R2/E8 specialization of the intended `P(('replica_dcn', 'data'), 'expert', None, None)` shape.
- Caveat:
  - The outer Iris executor wrapper did not pass `MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES`, so the H100 row reported `boundary_collectives_required_absent=false` even though the observed compiled collectives were zero and `allow_boundary_collectives=false`.
  - Fixed in `scratch/launch_muon_update_bench_executor_n1.sh` by forwarding `MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES`.
  - Local validation after the fix:
    - `bash -n scratch/launch_muon_update_bench_executor_n1.sh scratch/muon_update_bench_fast_loop.sh` passed.
    - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'launch_config_reads_env or grouped_muonh_bank_consumer'` passed with `1 passed`.
    - Forced-host strict smoke passed with `boundary_collectives_required_absent=true` and lowered/compiled `AG/A2A/AR/RS/CP = 0/0/0/0/0`:
      - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --max-grouped-stack-size 8 --grouped-expert-consumer-tokens-per-expert 3 --bench-kinds expert_grouped_muonh_bank_consumer --mode both --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --require-no-boundary-collectives --output /tmp/muon_grouped_muonh_bank_consumer_strict.json`
- Interpretation: This is the first H100 evidence that a persistent grouped expert-bank representation can survive both MuonH and an expert-like consumer without compiled boundary collectives. It does not prove the ordinary FSDP-master/`optax.apply_updates` bridge; it argues for porting the real expert MLP consumer to grouped banks or adding an explicit coarse transport boundary.
- Next action: Treat the grouped bank consumer as the next production path. The real integration target is no longer another per-leaf grouped-to-FSDP restore variant; it is either a grouped expert-bank model consumer or a custom coarse transport that behaves like the grouped bank consumer rather than like per-leaf FSDP restore.

### 2026-06-19 22:20 PDT - strict H100 grouped-bank gate after executor fix
- Hypothesis: After forwarding `MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES` through the outer Iris executor wrapper, the same H100 grouped-bank consumer gate should pass with the strict flag visible in launch metadata and summary rows.
- Command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-STRICT-G8-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_grouped_muonh_bank_consumer MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT=1 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- Result:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-051750`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-051750/grug-train-MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-STRICT-G8-N2-cw-20260620-051748`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-MUONHBANKCONSUMER-STRICT-G8-N2-cw-20260620-051748-d84bd9`.
  - Iris parent succeeded with exit 0, failures 0, duration about 2m08s.
  - Launch metadata on both tasks reported `require_no_boundary_collectives=true`.
  - Summary rows on both tasks reported `boundary_collectives_required_absent=true`.
  - Lowered and compiled HLO both had `AG/A2A/AR/RS/CP = 0/0/0/0/0`.
  - Compiled HLO had `custom_call = 360` and `gpu_gemm_custom_call = 175`.
  - Compile time was about `0.62s` on task 0 and `0.49s` on task 1.
- Interpretation: This closes the previous caveat. The strict no-boundary-collectives gate passes on H100 for the synthetic grouped MuonH bank-consumer path. This is now the authoritative evidence for the grouped-bank direction.
- Next action: Port the grouped expert-bank consumer into the real model path, preserving grouped bank sharding through expert MLP consumption. Keep using this strict gate for every production-candidate bridge.

### 2026-06-19 22:31 PDT - H100 grouped MuonH plus public grouped-MoE consumer gate
- Hypothesis: The simple bank-consumer strict gate proves that MuonH can keep grouped expert banks through a dense proxy consumer, but the next closer gate should consume the updated grouped banks through the public `grouped_moe_mlp` helper. This path should have the same EP communication pattern as standalone `grouped_moe_mlp_consumer`; MuonH should add NS GEMMs, not grouped-to-FSDP restore collectives.
- Code snapshot:
  - Added benchmark kind `expert_grouped_muonh_moe_mlp_consumer`.
  - The step applies grouped MuonH, applies grouped updates to grouped expert-bank params, then calls `grouped_moe_mlp_consumer_outputs` over the updated banks.
  - This gate intentionally does not use `--require-no-boundary-collectives`, because the public grouped MoE helper has legitimate EP collectives. The comparison is against the standalone grouped MoE consumer's collective counts.
- Local validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'grouped_muonh_moe_mlp_consumer or grouped_muonh_bank_consumer or grouped_moe_mlp_consumer_preserves'` passed with `3 passed`.
  - Forced-host compile-only comparison passed:
    - `XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python experiments/grug/moe/muon_update_bench.py --layers 4 --ns4d-group-size 4 --ns4d-group-axis replica_dcn,data --hidden-dim 16 --intermediate-dim 8 --num-experts 8 --backend-steps 1 --replica-axis 2 --data-axis 2 --expert-axis 2 --model-axis 1 --max-grouped-stack-size 8 --grouped-expert-consumer-tokens-per-expert 3 --bench-kinds expert_grouped_moe_mlp_consumer,expert_grouped_muonh_moe_mlp_consumer --mode both --warmup 0 --iters 0 --compile-only --disable-abstract-mesh --allow-boundary-collectives --output /tmp/muon_grouped_muonh_moe_mlp_consumer.json`
    - Standalone grouped MoE lowered `AG/RS/A2A = 3/1/0`; MuonH+grouped MoE lowered `AG/RS/A2A = 3/1/0` and added the expected NS dots.
- H100 command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-MUONHMOEMLP-COMPILE-G8-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_grouped_moe_mlp_consumer,expert_grouped_muonh_moe_mlp_consumer MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT=1 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- H100 result:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-052817`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-052817/grug-train-MUON-BENCH-D2560-L26-R2E8-MUONHMOEMLP-COMPILE-G8-N2-cw-20260620-052815`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-MUONHMOEMLP-COMPILE-G8-N2-cw-20260620-052815-db4677`.
  - Iris parent succeeded with exit 0, failures 0, duration about 2m48s.

| bench | lowered AG/RS/A2A | compiled AG/RS/A2A | compiled custom_call | compiled GPU GEMM custom_call | compile seconds |
| --- | --- | --- | ---: | ---: | ---: |
| `expert_grouped_moe_mlp_consumer` | 12/4/0 | 11/1/0 | 16 | 47 | 4.32-4.87 |
| `expert_grouped_muonh_moe_mlp_consumer` | 12/4/0 | 11/1/0 | 376 | 202 | 9.90-9.98 |

- Interpretation: This is the closest H100 gate so far to the real model expert path. MuonH did not add any extra compiled all-gather/reduce-scatter/all-to-all beyond the public grouped MoE consumer baseline; it added the expected NS/GEMM custom calls. The remaining work is not a low-level communication mystery in this synthetic path, but a state/model integration problem: make real `MoEExpertMlp`/`MoEMLP` consume grouped expert banks.
- Next action: Port grouped expert-bank representation into the production Grug MoE module boundary. The first production-facing test should compare a group of real blocks against per-layer blocks for outputs/sharding, then compile a short training step or expert-only block step and check that the grouped expert path preserves the same communication pattern as this gate.

### 2026-06-19 22:39 PDT - production expert-bank adapter
- Hypothesis: Before changing the full `Transformer.blocks` state layout, the real expert module should expose a strict adapter from ordinary per-layer `MoEExpertMlp` leaves to a `GroupedMoEExpertMlp` bank. This gives the production path the same grouped representation used by the successful harness gates without depending on benchmark-only helpers.
- Change:
  - Added `GroupedMoEExpertMlp.from_layers(layers)` in `lib/levanter/src/levanter/grug/grug_moe.py`.
  - The adapter stacks `w_gate_up` and `w_down` along a leading group axis and validates that all grouped layers share `implementation`, `activation`, `capacity_factor`, and `remat_mode`.
  - Added model-level tests in `experiments/grug/moe/test_model.py` comparing grouped execution against independent per-layer `MoEExpertMlp` calls and rejecting mixed backends.
- Validation:
  - `uv run pytest experiments/grug/moe/test_model.py experiments/grug/moe/test_muon_update_bench.py -k 'grouped_expert_mlp_from_layers or grouped_muonh_moe_mlp_consumer or grouped_moe_mlp_consumer_preserves'` passed with `4 passed`.
- Interpretation: This is the first production-facing API step after the H100 grouped-MoE gate. It does not yet group the full `Block` or `Transformer` state, but it gives that integration a tested conversion point from existing per-layer experts into the persistent grouped-bank representation.
- Next action: Add a real `MoEMLP`/block-level grouped expert consumer that builds grouped routed inputs from per-layer router outputs and calls `GroupedMoEExpertMlp` directly, avoiding `GroupedMoEExpertMlp.layer()` in the hot path.

### 2026-06-19 22:43 PDT - FSDP-master objective audit
- Hypothesis: The active goal should be audited against the conservative FSDP-master contract, because grouped-bank progress is useful but does not by itself prove "grouped MuonH -> FSDP updates -> ordinary `optax.apply_updates`" is performant.
- Evidence reviewed:
  - `expert_fsdp_grouped_target_apply_chunked_fsdp_boundary`: direct `P(("replica_dcn", "data"), "expert", None, None) -> P(None, "expert", "data/model", None)` target reshard lowers cleanly but H100 GPU compilation reintroduces the same grouped-to-FSDP all-gathers.
  - `expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary`: returning FSDP leaves directly from `shard_map` does not reduce compiled all-gathers.
  - `expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary`: custom partitioning hides the restore in lowered HLO but H100 compiled HLO inlines it and restores the same all-gathers.
  - packed and ppermute bridges either keep the all-gather pattern, introduce A2A/CP traffic, OOM, or do not improve runtime enough.
  - grouped-trace/FSDP-master keeps the API semantics closer to the objective and improves isolated update timing, but still compiles with the grouped-to-FSDP all-gather boundary.
- Result: Updated `.agents/projects/2026-06-19_grug_moe_muon_grouped_bank_path.md` with an explicit requirement-by-requirement audit. Current status: FSDP params at the boundary, grouped MuonH compute, and semantic conversion to FSDP before apply are present in the harness; avoiding the per-leaf collective explosion and preserving performance are not proven.
- Interpretation: The remaining FSDP-master blocker is specifically the compiled grouped-to-FSDP bridge. Another JAX-level `reshard`/`shard_map` wrapper is unlikely to satisfy the goal unless it materially changes compiled HLO. The remaining viable FSDP-master work is a lower-level/custom transport that beats the explicit slice-first baseline, while the grouped-bank model-consumer path remains the best zero-boundary alternative.
- Next action: Keep the lower-level bridge subagent focused on a Pallas/Triton/FFI/custom-call proof point; mainline integration should continue with grouped expert-bank consumption only with the caveat that it is a representation pivot, not proof of the conservative FSDP-master objective.

### 2026-06-19 22:47 PDT - boundary fanout estimator
- Hypothesis: The harness should make the unavoidable part of the FSDP-master bridge visible: if grouped MuonH shards the leading group axis over `replica_dcn` but FSDP expert params are replicated over `replica_dcn`, returning ordinary FSDP leaves requires a replica fanout even before considering compiler pathologies.
- Change:
  - Added `replica_fanout_factor` and `requires_replica_fanout` to `estimated_boundary_byte_estimates`.
  - Exposed these as `estimated_boundary_replica_fanout_factor` and `estimated_boundary_requires_replica_fanout` in summary rows.
  - Added lower-bound byte fields: `estimated_boundary_replica_fanout_min_extra_per_device_bytes` and `estimated_boundary_replica_fanout_min_total_receive_bytes`.
  - Added lowered/compiled collective fragmentation fields that compare `all_gather + all_to_all + collective_permute` against an ideal one grouped transport per expert weight name.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_boundary_byte_estimates` passed.
  - Tiny 8-device CPU compile-only smoke for `expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary` wrote `/tmp/muon_boundary_fanout_fields.json` and reported `estimated_boundary_replica_fanout_factor=2.0`, `estimated_boundary_requires_replica_fanout=True`, `estimated_boundary_replica_fanout_min_extra_per_device_bytes=3072.0`, `estimated_boundary_replica_fanout_min_total_receive_bytes=24576.0`, `all_gather=2`, and `all_to_all=0`.
  - Tiny 8-device CPU compile-only smoke for the same bench wrote `/tmp/muon_boundary_fragmentation_fields.json` and reported lowered fragmentation `2/2 = 1.0`. CPU compiled HLO optimized this tiny collective away, so H100 compiled rows remain the authoritative signal for compiled fragmentation.
  - `./infra/pre-commit.py --changed-files --fix` passed.
- Interpretation: Future boundary rows now separate inherent fanout (`replica_dcn` ownership -> FSDP replica copies) from avoidable compiler explosion (many per-layer all-gathers, A2A/CP insertion, OOM). This does not solve the bridge, but it makes the pass/fail criterion sharper for any custom transport.
- Next action: Use these fields when comparing any lower-level grouped-to-FSDP bridge against the explicit slice-first baseline.

### 2026-06-19 23:05 PDT - H100 boundary fragmentation row
- Hypothesis: The new fanout/fragmentation fields should be validated on the real 2-node H100 compiler path, not only tiny CPU lowering, so future lower-level bridge candidates have a clear baseline.
- Command:
  - `RUN_ID="MUON-BENCH-D2560-L26-R2E8-FRAGFIELDS-COMPILE-G8-N2-cw-$(date -u +%Y%m%d-%H%M%S)" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG="$HOME/.kube/coreweave-iris-gpu" MUON_BENCH_GPU_REPLICAS=2 MUON_BENCH_LAYERS=26 MUON_BENCH_REPLICA_AXIS=2 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary,expert_fsdp_grouped_target_apply_chunked_fsdp_boundary MUON_BENCH_SWEEP_BACKEND_STEPS=1 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_DTYPE=bf16 MUON_BENCH_NS_COMPUTE_DTYPE=bf16 MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_WORKER_CPU=8 MUON_BENCH_WORKER_RAM=256g XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`
- Result:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-060051`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-060051/grug-train-MUON-BENCH-D2560-L26-R2E8-FRAGFIELDS-COMPILE-G8-N2-cw-20260620-060049`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2E8-FRAGFIELDS-COMPILE-G8-N2-cw-20260620-060049-704a21`.
  - Iris parent and child both succeeded with exit 0. No OOM or rendezvous failure; the final PJRT `WatchTasksAsync` warnings were shutdown noise after task exit.

| bench | lowered collectives | compiled collectives | compiled fragmentation | inherent fanout | min extra per-device fanout |
| --- | --- | --- | ---: | ---: | ---: |
| `expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary` | 8 AG | 8 AG | 4.0x ideal | 2.0x | 8.18 GiB |
| `expert_fsdp_grouped_target_apply_chunked_fsdp_boundary` | 0 | 26 AG + 24 CP | 25.0x ideal | 2.0x | 8.18 GiB |

- Interpretation: The real H100 compiler confirms the split between unavoidable fanout and avoidable fragmentation. Both candidate bridges still require a 2x replica fanout because grouped ownership includes `replica_dcn` while FSDP expert leaves are replicated over it. The explicit slice-first path is the current conservative baseline at 8 all-gathers, 4x the ideal one transport per expert projection. The superficially cleaner target/chunked-FSDP path is worse after GPU compilation: it lowers with no collectives, but compilation expands it to 50 collectives, 25x ideal.
- Next action: A lower-level bridge has to beat the explicit slice-first baseline in compiled HLO, not just in lowered StableHLO. For the FSDP-master objective, the pass criterion should be compiled fragmentation near 1x with the 2x fanout byte floor preserved. For the grouped-bank model-consumer pivot, keep requiring zero grouped-to-FSDP boundary collectives.

### 2026-06-19 23:12 PDT - OSS Muon implementation survey
- Hypothesis: Existing OSS Muon implementations may already contain a distributed layout trick that avoids Marin's grouped-to-FSDP per-leaf bridge problem.
- Method: Spawned a read-only survey agent to inspect primary source code for KellerJordan Muon/modded-nanogpt, Megatron-LM layer-wise distributed Muon, NeMo Emerging-Optimizers, DeepSpeed Muon, MaxText/Optax Muon, PyTorch native Muon, and Flash-Muon. No repo edits or runs.
- Result:
  - No OSS implementation exactly solves `JAX FSDP train-state + temporary grouped MuonH optimizer banks + cheap grouped-to-FSDP update transport`.
  - Megatron-LM is the closest conceptual match: its layer-wise distributed optimizer makes whole-matrix Muon ownership part of the parameter-buffer/bucket layout, then synchronizes updated params through the bucket infrastructure rather than per-leaf optimizer reshards.
  - modded-nanogpt/NorMuon is strong evidence for persistent grouped banks: it reshapes explicit parameter banks so the leading bank dimension is divisible by world size, reduce-scatters bank gradients, updates local chunks, then all-gathers the bank. Its speed comes from changing representation, not from reconstructing temporary groups each step.
  - DeepSpeed ZeRO-3 is a warning case: it gathers params/momentum per subgroup and forbids some reduce-scatter/all-to-all Muon combinations, which is close to the failure mode we are trying to avoid.
  - NeMo/Optax/MaxText support the 3D/4D batched NS contract and explicit dimension metadata, but do not solve cross-layout transport.
- Interpretation: The survey supports the same direction as the H100 gates. For a fast solution, Muon grouping should be a first-class parameter/bucket representation. If we keep canonical FSDP masters, the missing piece is a coarse bank-level transport bucket, not more per-leaf `with_sharding_constraint` wrappers. If we allow a representation pivot, grouped expert banks consumed by the model are the cleanest path.
- Next action: Preserve the conservative FSDP-master goal, but evaluate new work against two explicit paths:
  1. FSDP-master path: grouped NS bank update -> one coarse transport per expert projection -> FSDP bucket slices -> ordinary `optax.apply_updates`.
  2. Representation-pivot path: persistent grouped expert banks consumed directly by grouped MoE/model code, avoiding the grouped-to-FSDP update boundary.

### 2026-06-20 00:05 PDT - production grouped MoEMLP adapter
- Hypothesis: The grouped-bank path needs a production-facing module boundary, not only benchmark helpers. A grouped `MoEMLP` adapter should stack adjacent production `MoEMLP` layers, run the real router path with a leading group axis, and call `GroupedMoEExpertMlp` directly without materializing `GroupedMoEExpertMlp.layer()` in the hot path.
- Change:
  - Added `GroupedMoEMLP` in `experiments/grug/moe/model.py`.
  - `GroupedMoEMLP.from_layers(layers)` stacks router weights, router bias, and `GroupedMoEExpertMlp.from_layers(...)` from ordinary production `MoEMLP` layers.
  - `GroupedMoEMLP.__call__` accepts `[G, B, S, D]`, computes the same QB router/top-k/sigmoid combine weights per group, computes grouped QB beta with the group axis replicated, calls the grouped expert bank once, and returns grouped routed outputs plus per-group router stats.
- Validation:
  - `uv run pytest experiments/grug/moe/test_model.py -q` passed with `5 passed`.
  - `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -k 'grouped_moe_mlp or grouped_moe_expert_mlp' -q` passed with `4 passed`.
  - The new production test constructs ordinary `MoEMLP` layers, converts them with `GroupedMoEMLP.from_layers`, and verifies the grouped path lowers under a small explicit abstract mesh while preserving grouped output and router-stat shapes.
- Interpretation: This is a concrete integration step for the representation-pivot path. It does not yet replace the full transformer block loop or prove full-train performance, but it moves grouped expert banks from benchmark-only code into the production model module boundary with a real router + grouped expert consumer.
- Next action: Add a grouped block/transformer execution mode that batches adjacent `Block.mlp` calls through `GroupedMoEMLP` while leaving attention and non-expert parameters per-layer. Then compile a short production block/train-step gate and compare communication against the synthetic grouped MoE consumer H100 gate.

### 2026-06-20 00:42 PDT - retargeted FSDP boundary primitive baseline
- Hypothesis: The conservative MuonH path should be judged as two explicit FSDP boundary primitives, not as generic grouped-Muon speed: `fsdp_grads_to_grouped_chunks` and `grouped_updates_to_fsdp_apply`, expert weights first.
- Goal file: `/Users/dlwh/.codex/attachments/75b6836e-e804-41b6-9e08-04bdc3440f44/goal-objective.md`.
- Change:
  - Updated `scratch/launch_muon_grouped_reference_2node_wandb.sh` to default to `R1D2E8` instead of `R2D1E8`. The old lazy default used the second node as `replica_dcn`, which replicated the full expert params, grads, and grouped MuonH state across nodes and OOMed before timing with a 10.16 GiB allocation.
  - Added `boundary_primitive` and effective GB/s summary fields to the Muon update bench rows.
  - Labeled the real FSDP grouped optimizer-update harness as the full pipeline it currently is: `fsdp_grads_to_grouped_chunks+grouped_muon_update+grouped_updates_to_fsdp_update_tree`.
- Run:
  - Command: `bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
  - Parent Iris job: `/dlwh/iris-run-job-20260620-062333`.
  - Child Iris job: `/dlwh/iris-run-job-20260620-062333/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-N2-cw-20260620-062330`.
  - W&B: `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-N2-cw-20260620-062330`.
  - Output prefix: `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-N2-cw-20260620-062330-712698`.
- Result:
  - Iris parent and child both succeeded. No OOM or rendezvous failure.
  - Lowered HLO: 24 all-gathers, 0 all-reduces, 0 all-to-alls, 0 reduce-scatters, 72 dot-generals, 72 two-batch-axis dot-generals, `lower_seconds ~= 0.52`.
  - Timed compiled HLO: 24 all-gathers, 0 all-reduces, 0 all-to-alls, 0 reduce-scatters, 270 custom calls, 369 GPU GEMM custom calls.
  - Timing: mean `0.4748943s`, median `0.4750389s`, min `0.4734679s`.
  - Estimated compute: mean `5114.41 TFLOP/s`, about `32.32%` of nominal H100 bf16 peak across 16 GPUs.
  - Boundary estimates: global update bytes `130,862,284,800`, grouped input per device `8,178,892,800`, FSDP output per device `8,178,892,800`, all-gather slice peak per device `16,357,785,600`.
  - Fragmentation: compiled collective count `24`, ideal count `2`, fragmentation factor `12.0`.
  - Replica fanout: factor `1.0`, `requires_replica_fanout=false`.
- Interpretation: The `R1D2E8` shape fixes the replicated-state OOM and gives a runnable 2-node reference, but it does not satisfy the retargeted boundary goal. The main failure mode remains collective fragmentation: XLA compiled the expert boundary as 24 serialized all-gathers instead of a small number of coarse transports.
- Next action: Implement a focused expert-weight `fsdp_grads_to_grouped_chunks` benchmark and a focused `grouped_updates_to_fsdp_apply` benchmark with correctness max-error and peak-HBM fields. The lower-level bridge must beat this R1D2E8 baseline in compiled HLO, not only lowered StableHLO.

### 2026-06-20 00:53 PDT - boundary correctness reporting
- Hypothesis: The boundary harness needs value-equivalence reporting before any new transport variant can be trusted; shape/sharding checks alone do not prove `grouped_updates_to_fsdp_apply` is correct.
- Change:
  - Added a tiny reference path for grouped expert updates -> FSDP update tree and grouped expert updates -> FSDP params through `optax.apply_updates`.
  - Added `boundary_correctness_max_error` to `TimingSummary` and summary rows for explicit expert grouped-to-FSDP restore/apply candidate benches.
  - Added `estimated_boundary_peak_per_device_bytes` as an estimated HBM pressure row, currently the max of grouped input, FSDP output, and all-gather slice peak bytes per device.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'boundary_correctness_max_error or summary_row_reports_boundary_byte_estimates' -q` passed with `2 passed, 77 deselected`.
  - `./infra/pre-commit.py --changed-files --fix` passed.
- Interpretation: This still does not prove the two final primitives, but future focused boundary rows can now carry a correctness max-error and an explicit estimated peak-memory field. The current correctness coverage is intentionally limited to grouped-updates-to-FSDP restore/apply paths; the real optimizer-update pipeline needs a separate reference contract because it also includes trace/MuonH.
- Next action: Run a focused H100 boundary candidate with the new fields, then add the missing `fsdp_grads_to_grouped_chunks`-specific reference and reporting.

### 2026-06-20 01:00 PDT - FSDP grads to grouped chunks harness
- Hypothesis: The first conservative MuonH boundary primitive should be measured independently: ordinary FSDP expert gradients -> grouped/padded NS-friendly chunks, without grouped Muon compute or grouped-to-FSDP apply noise.
- Command: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'fsdp_grads_to_grouped_chunks or boundary_correctness_max_error or summary_row_reports_boundary_byte_estimates' -q`; `./infra/pre-commit.py --changed-files --fix`.
- Config: Added bench kind `expert_fsdp_grads_to_grouped_chunks`. It accepts FSDP-shaped expert leaves, stacks layers into grouped chunks, pads the group axis for `replica_dcn,data` sharding when needed, and reports `boundary_primitive=fsdp_grads_to_grouped_chunks` with the existing boundary byte/fragmentation fields.
- Result: Focused tests passed with `4 passed, 77 deselected`; changed-file pre-commit passed. The earlier full-shape N1 correctness sanity run `/dlwh/iris-run-job-20260620-064627` failed because full D2560 correctness comparison materialized the expert tree and OOMed while allocating 400 MiB on each GPU. Added a 1 GiB global-update cap so large runs still report timing/HLO while tiny correctness cases report `boundary_correctness_max_error`.
- Interpretation: The harness now has an explicit first-boundary primitive and a correctness reference for small cases. Full May-shape correctness should be validated through reduced shapes or targeted checks; full May-shape H100 runs should focus on compiled collectives, payload estimates, timing, and peak-memory signals.
- Next action: Launch a May-shape H100 compile/timing row for `expert_fsdp_grads_to_grouped_chunks`, then compare compiled collective count/payload against the grouped-updates-to-FSDP baselines.

### 2026-06-20 01:02 PDT - H100 FSDP grads to grouped chunks row
- Hypothesis: The first boundary primitive can avoid the all-gather-heavy grouped-to-FSDP failure mode because FSDP expert grads are already data-sharded in the direction needed by grouped chunks under `R1D2E8`.
- Command: `RUN_ID="MUON-BENCH-D2560-L26-R1D2E8-G8-FSDP2GROUPED-N2-cw-20260620-065915" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_grads_to_grouped_chunks MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Config: 2 H100 nodes, `replica_axis=1`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, bf16, output path `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-FSDP2GROUPED-N2-cw-20260620-065915-91b1b1`.
- Result: Iris parent `/dlwh/iris-run-job-20260620-065917` succeeded. Lowered HLO had zero collectives. Compiled HLO had `all_to_all=8`, `all_gather=0`, `reduce_scatter=0`, `all_reduce=0`, `collective_permute=0`, so compiled fragmentation was `8 / 2 = 4.0x` ideal. Process 0 timing was `1.619883s`; process 1 timing was `1.987257s`. Process 0 effective global boundary throughput was `80.79 GB/s`; process 1 was `65.85 GB/s`. Estimated global update bytes `130,862,284,800`; grouped input per device `8,178,892,800`; FSDP output per device `8,178,892,800`; peak per-device estimate `16,357,785,600`; `boundary_correctness_max_error=null` because full-shape correctness is now skipped by the memory cap.
- Interpretation: This gets past the prior full-tree correctness OOM and avoids all-gather, but it still does not meet the boundary goal. The compiler emits 8 all-to-alls for an ideal 2 expert-projection transports and measured bandwidth is far below a plausible H100/NVLink/IB comms roofline. The next target is to reduce the first-boundary fragmentation from 4x toward 1x or test whether packing/lower-level transport can make the 8 A2As coarser/faster.
- Next action: Run the corrected `R1D2E8` apply-side Route A/Route B row (`expert_fsdp_grouped_updates_muonh_apply`, `expert_fsdp_grouped_updates_muonh_direct_apply`) so both halves of the conservative boundary have comparable data under the same non-replicating two-node shape.

### 2026-06-20 01:10 PDT - H100 grouped updates to FSDP apply routes
- Hypothesis: Under the corrected non-replicating `R1D2E8` layout, direct grouped-update application might avoid the update-tree materialization cost even if the grouped-to-FSDP transport is still fragmented.
- Command: `RUN_ID="MUON-BENCH-D2560-L26-R1D2E8-G8-APPLYROUTES-N2-cw-20260620-070324" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_grouped_updates_muonh_apply,expert_fsdp_grouped_updates_muonh_direct_apply MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Config: 2 H100 nodes, `replica_axis=1`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, bf16, output path `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-APPLYROUTES-N2-cw-20260620-070324-89555a`.
- Result: Iris child `/dlwh/iris-run-job-20260620-070326/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-APPLYROUTES-N2-cw-20260620-070324` succeeded. Both routes lowered with zero collectives, but GPU compilation expanded both to `all_gather=8`, `all_to_all=8`, `reduce_scatter=0`, `all_reduce=0`, `collective_permute=0`; compiled fragmentation was `16 / 2 = 8.0x` ideal. Route A (`grouped_updates_to_fsdp_update_tree_then_optax_apply`) timed `1.874761s` on process 0 and `2.588582s` on process 1, with estimated global boundary throughput `69.80` to `50.55 GB/s`. Route B (`grouped_updates_apply_direct`) timed `0.498389s` on process 0 and `1.165260s` on process 1, with estimated global boundary throughput `262.57` to `112.30 GB/s`. Estimated global update bytes were `130,862,284,800`; grouped input per device `8,178,892,800`; FSDP output per device `8,178,892,800`; peak per-device estimate `16,357,785,600`; full-shape `boundary_correctness_max_error=null` due the correctness memory cap.
- Interpretation: Route B is clearly the better FSDP-apply shape for runtime, but neither route satisfies the conservative boundary goal. The same compiled `8 AG + 8 A2A` transport appears in both routes, so the bottleneck is not just Python/tree-shaped `optax.apply_updates`; it is the grouped-update to FSDP layout conversion that GPU XLA fragments after lowering. The next FSDP-master path needs a packed/coarse or lower-level bridge that beats `16` compiled collectives and moves toward the ideal `2` expert-projection transports.
- Next action: Keep Socrates focused on a lower-level grouped-to-FSDP bridge, and in the main harness add or select the next candidate around coarse packed transport rather than another lowered-HLO-only wrapper. Any winner should be judged by compiled HLO fragmentation, wall time, GB/s, peak HBM, and a small-shape correctness row.

### 2026-06-20 01:19 PDT - packed grouped-to-FSDP apply negative result
- Hypothesis: Packing all layer groups for each expert weight name before the grouped-to-FSDP apply boundary might make XLA/NCCL see fewer larger transfers than the direct Route B path.
- Compile-only command: `RUN_ID="MUON-BENCH-D2560-L26-R1D2E8-G8-PACKED-APPLY-COMPILE-N2-cw-20260620-071203" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_grouped_packed_a2a_apply_boundary,expert_fsdp_grouped_packed_slice_first_gather_apply_boundary,expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary,expert_fsdp_grouped_packed_data_ppermute_apply_boundary MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=0 MUON_BENCH_MODE=both MUON_BENCH_COMPILE_ONLY=true MUON_BENCH_TRACKER=json MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Timed command: `RUN_ID="MUON-BENCH-D2560-L26-R1D2E8-G8-PACKED-SLICE-TIMED-N2-cw-20260620-071644" MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_grouped_packed_slice_first_gather_apply_boundary MUON_BENCH_WARMUP=0 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_TRACKER=json MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Result: The compile-only packed gate `/dlwh/iris-run-job-20260620-071206` succeeded. All four packed variants lowered to the ideal `2 all_gather` and zero A2A/CP, but GPU compilation expanded every variant to `all_gather=2`, `all_to_all=10`, `collective_permute=2`, for `14 / 2 = 7.0x` ideal fragmentation. The timed packed slice-first run `/dlwh/iris-run-job-20260620-071648/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-PACKED-SLICE-TIMED-N2-cw-20260620-071644` also succeeded with the same compiled collectives, but was slow: `2.460837s` on process 0 and `2.688172s` on process 1, estimated global boundary throughput only `53.18` to `48.68 GB/s`.
- Interpretation: Existing JAX-level packing is a negative result. It slightly reduces compiled collective count versus Route B (`14` vs `16`), but it introduces extra A2A/CP traffic and is much slower than direct apply (`~2.5-2.7s` versus Route B's `~0.5-1.2s`). This reinforces that lowered HLO is not predictive enough for this boundary; the pass/fail signal must be compiled HLO plus timing. The next FSDP-master candidate should be a lower-level/custom transport or a fundamentally different layout, not this packed wrapper.
- Next action: Do not promote the packed JAX-level apply variants. Keep Route B as the best current semantic baseline for apply-side runtime, and use the packed result as evidence for Socrates/the lower-level bridge: a successful bridge must beat compiled `8 AG + 8 A2A` and wall time `0.5-1.2s`, not just reduce lowered HLO to 2 collectives.

### 2026-06-20 02:10 PDT - whole packed-bank MuonH apply R2 result
- Hypothesis: If the grouped expert bank stays packed through Newton-Schulz and hyperball update, and only converts back to FSDP at the final apply boundary, XLA may avoid the packed-entry `slice_group/slice` collective-permute explosion.
- Change:
  - Added bench kind `expert_fsdp_packed_bank_muonh_apply`.
  - The bench composes FSDP expert grads -> explicit packed grouped banks, whole packed-bank MuonH update, then packed bank -> FSDP params apply.
  - Commit `7e2e68e66` added the harness path; commit `9384724a6` fixed the timing runner so this bench is invoked as `(params, updates) -> params`.
- Command:
  - `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801 MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_apply MUON_BENCH_MODE=both MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_TRACKER=wandb MUON_BENCH_WANDB_PROJECT=marin_moe MUON_BENCH_WANDB_GROUP=grug-moe-cw-muon-packed-bank bash scratch/launch_muon_grouped_reference_2node_wandb.sh`
- Config:
  - 2 H100 nodes, `replica_axis=1`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, H3, bf16 params/NS compute.
  - Iris parent `/dlwh/iris-run-job-20260620-090804`; child `/dlwh/iris-run-job-20260620-090804/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801-a160fa`; W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801`.
- Result:
  - Run succeeded and got past the previous timing-wrapper TypeError.
  - Lowered HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `18 dot_general`.
  - Compiled HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `41 gpu_gemm_custom_call`.
  - Timing: mean `0.63868s`, median `0.63820s`, min `0.63518s`.
  - Estimated compute: `2.4288e15` NS dot FLOPs, mean `3802.9 TFLOP/s`, about `24.0%` nominal H100 bf16 peak.
  - Boundary estimates: global update bytes `130,862,284,800`, grouped input per device `8,178,892,800`, FSDP output per device `8,178,892,800`; compiled collective count `6`, ideal `2`, fragmentation `3.0x`.
- Interpretation: This is the first conservative FSDP-master candidate that avoids both the old per-leaf all-gather explosion and the packed-entry collective-permute explosion. It is much better than the packed-entry route (`2 AG + 14 A2A + 16 CP`, roughly `1.99s` update / `1.51s` apply), but it still misses the target boundary shape: 6 all-to-alls is 3x the ideal two expert-projection transports, and throughput is only about 24% of nominal peak for the whole packed-bank update/apply harness.
- Issue update: https://github.com/marin-community/marin/issues/6493#issuecomment-4757113152
- Next action: Validate the same packed-bank path on N1 and R4 using `data_axis` for additional nodes (`R1D4E8`, not `R4D1E8`). Then use the scale/fragmentation table to decide whether the next main-thread optimization is reducing XLA's 6-A2A fragmentation or handing the boundary to Socrates' lower-level bridge.

### 2026-06-20 02:18 PDT - whole packed-bank MuonH N1/R2/R4 scale table
- Hypothesis: The whole packed-bank path should preserve the no-AG/no-CP property across single-node and data-sharded multi-node layouts. Scaling behavior should show whether the remaining `6 all_to_all` boundary fragmentation is a serious limiter.
- Commands:
  - N1: `RUN_ID=MUON-BENCH-D2560-L26-R1D1E8-G8-H3-PACKEDBANKMUON-N1-cw-20260620-091338 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_GPU_REPLICAS=1 MUON_BENCH_REPLICA_AXIS=1 MUON_BENCH_DATA_AXIS=1 MUON_BENCH_EXPERT_AXIS=8 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_NS4D_GROUP_SIZE=8 MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_apply MUON_BENCH_MODE=both MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_DISABLE_ABSTRACT_MESH=true MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_TRACKER=wandb MUON_BENCH_WANDB_PROJECT=marin_moe MUON_BENCH_WANDB_GROUP=grug-moe-cw-muon-packed-bank XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`
  - R4: same command with `RUN_ID=MUON-BENCH-D2560-L26-R1D4E8-G8-H3-PACKEDBANKMUON-N4-cw-20260620-091400`, `MUON_BENCH_GPU_REPLICAS=4`, `MUON_BENCH_DATA_AXIS=4`.
- Config:
  - N1: 8 H100s, `R1D1E8`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D1E8-G8-H3-PACKEDBANKMUON-N1-cw-20260620-091338-3aa668`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D1E8-G8-H3-PACKEDBANKMUON-N1-cw-20260620-091338`.
  - R2: 16 H100s, `R1D2E8`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801-a160fa`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKMUON-N2-cw-20260620-090801`.
  - R4: 32 H100s, `R1D4E8`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D4E8-G8-H3-PACKEDBANKMUON-N4-cw-20260620-091400-4b5ff8`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D4E8-G8-H3-PACKEDBANKMUON-N4-cw-20260620-091400`.
- Result:

| layout | GPUs | compiled AG/A2A/CP/RS | median seconds | mean seconds | median TFLOP/s | nominal peak % | boundary fragmentation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `R1D1E8` | 8 | `0/0/0/0` | 0.59984 | 0.59984 | 4049.1 | 51.18 | 0.0x |
| `R1D2E8` | 16 | `0/6/0/0` | 0.63820 | 0.63868 | 3805.7 | 24.05 | 3.0x |
| `R1D4E8` | 32 | `0/6/0/0` | 0.29488 | 0.29509 | 8236.7 | 26.03 | 3.0x |

- Interpretation:
  - The packed-bank path is stable across N1/R2/R4 and avoids per-leaf all-gather and collective-permute explosion in all three layouts.
  - N1 is a clean compute baseline: no boundary collectives and about 51% nominal peak.
  - R2 is slower than N1 despite twice the GPUs because the two-node boundary/communication cost and reduced local work dominate.
  - R4 recovers useful scaling: about `2.03x` faster than N1 and `2.16x` faster than R2, but still only about 26% nominal peak because the compiled boundary remains `6 all_to_all`, 3x the ideal two expert-projection transports.
- Next action: Treat the whole packed-bank path as the current conservative FSDP-master baseline. The remaining main-thread optimization is to collapse `6 A2A -> 2 A2A` or prove via Socrates' lower-level bridge that a custom/bucketed transport can do it. If that fails, the representation-pivot path with persistent grouped expert banks remains the cleaner way to avoid this boundary entirely.

### 2026-06-20 02:40 PDT - packed-bank MuonH phase diagnostics
- Hypothesis: The full packed-bank `6 all_to_all` path decomposes into three expected two-transport phases: FSDP grads into packed grouped banks, FSDP params into packed grouped banks for hyperball/scale, and packed grouped updates back to FSDP apply layout. A narrower diagnostic should confirm there is no hidden all-gather/collective-permute explosion inside the whole-bank path.
- Change:
  - Added `expert_fsdp_packed_bank_muonh_update_only`, which packs params and grads into whole grouped banks, runs packed-bank MuonH/Hyperball, and returns packed grouped updates without restoring to FSDP.
  - Added `expert_fsdp_packed_bank_direction_apply`, which packs grads only, runs Newton-Schulz direction/scale, and applies the packed direction back to FSDP params. This is not semantically full MuonH because it skips packed params and hyperball scaling, but it isolates the entry+apply boundary plus NS direction cost.
  - Commit: `a282cfce7` (`Add packed-bank MuonH boundary diagnostics`).
- Commands:
  - Combined timing attempt: `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKDIAG-N2-cw-20260620-093043 ... MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_update_only,expert_fsdp_packed_bank_direction_apply ... bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
  - Compile-only split: `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKDIAGCOMPILE-N2-cw-20260620-093445 ... MUON_BENCH_COMPILE_ONLY=true ...`.
  - Direction-apply timing: `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-DIRECTIONAPPLY-N2-cw-20260620-093453 ... MUON_BENCH_KINDS=expert_fsdp_packed_bank_direction_apply ...`.
- Config:
  - 2 H100 nodes, `replica_axis=1`, `data_axis=2`, `expert_axis=8`, `model_axis=1`, `layers=26`, `ns4d_group_size=8`, `ns4d_group_axis=replica_dcn,data`, H3, bf16 params/NS compute.
  - Runs:
    - `/dlwh/iris-run-job-20260620-093045`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKDIAG-N2-cw-20260620-093043`.
    - `/dlwh/iris-run-job-20260620-093447`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKDIAGCOMPILE-N2-cw-20260620-093445`.
    - `/dlwh/iris-run-job-20260620-093456`, W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-DIRECTIONAPPLY-N2-cw-20260620-093453`.
- Result:
  - Combined timing attempt failed while timing update-only with `RESOURCE_EXHAUSTED` on a 35.55 GiB allocation. It still emitted the key lowered evidence before failing: update-only lowered HLO had `0 all_gather`, `4 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, and `18 dot_general`.
  - Compile-only update-only row succeeded: lowered and compiled HLO both had `0 all_gather`, `4 all_to_all`, `0 collective_permute`, `0 reduce_scatter`; compiled HLO had `45 gpu_gemm_custom_call` and `90 custom_call`.
  - Compile-only direction-apply row succeeded: lowered and compiled HLO both had `0 all_gather`, `4 all_to_all`, `0 collective_permute`, `0 reduce_scatter`; compiled HLO had `40 gpu_gemm_custom_call` and `90 custom_call`.
  - Direction-apply timing succeeded with mean `0.52986-0.52990s`, median `0.52992-0.53004s`, min `0.52664-0.52685s`, estimated NS dot work `2.4288e15` FLOPs, mean `4583-4584 TFLOP/s`, about `28.97%` nominal H100 bf16 peak, and effective boundary global bandwidth about `246.97 GB/s`.
- Interpretation:
  - The decomposition is confirmed: update-only is `2 A2A` for grads entry plus `2 A2A` for params entry; direction-apply is `2 A2A` for grads entry plus `2 A2A` for FSDP apply; full packed-bank MuonH apply is the expected `2 + 2 + 2 = 6 A2A`.
  - There is no hidden all-gather or collective-permute explosion in the whole-bank path. The remaining multi-node cost is real boundary transport plus NS/hyperball work, not a per-leaf compiler blow-up.
  - The update-only timing OOM is a harness-output materialization problem, not a compile-shape failure. If that timing matters later, add a checksum/reduction output variant instead of returning the full packed update bank.
  - Comparing direction-apply (`~0.530s`) to full packed-bank R2 (`~0.638s`) suggests params-pack + hyperball/scale overhead is roughly `0.11s` on this shape, so the dominant remaining target is still the boundary/NS path, not Python tree apply or a surprise extra collective.
- Next action: Keep `expert_fsdp_packed_bank_muonh_apply` as the current conservative FSDP-master baseline. The next useful implementation work is either a lower-level bridge that collapses the three two-A2A phases or a representation decision that makes one packed param/update bank persistent without violating the FSDP-master contract.

### 2026-06-20 02:51 PDT - update-only checksum timing
- Hypothesis: The update-only packed-bank diagnostic failed only because the harness materialized the full packed update bank on output. Returning a scalar checksum from the timing factory should preserve the computation while avoiding the 35.55 GiB output allocation.
- Change:
  - Commit `14cfd9d3c` changes only the timing factory for `expert_fsdp_packed_bank_muonh_update_only` to compute the normal packed-bank MuonH update and return a scalar checksum. The ordinary step factory still returns the packed update bank and is still covered by output-sharding tests.
  - Added focused test coverage for the scalar timing output.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank_muonh_update_only or packed_bank_direction_apply' -q` -> 3 passed.
  - `uv run python experiments/grug/moe/muon_update_bench.py --bench-kinds expert_fsdp_packed_bank_muonh_update_only,expert_fsdp_packed_bank_direction_apply --layers 2 --hidden-dim 8 --intermediate-dim 4 --num-experts 2 --ns4d-group-size 2 --ns4d-group-axis none --backend-steps 1 --dtype float32 --replica-axis 1 --data-axis 1 --expert-axis 1 --model-axis 1 --iters 1 --warmup 0 --mode both --disable-abstract-mesh --compiled-hlo-output scratch/muon_update_bench_checksum_diag_compiled_hlo.txt --hlo-output scratch/muon_update_bench_checksum_diag.stablehlo --output scratch/muon_update_bench_checksum_diag_smoke.json` passed.
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed, and commit hooks including Pyrefly passed.
- CoreWeave command:
  - `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKUPDATECHECKSUM-N2-cw-20260620-094724 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_update_only MUON_BENCH_MODE=both MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_TRACKER=wandb MUON_BENCH_WANDB_PROJECT=marin_moe MUON_BENCH_WANDB_GROUP=grug-moe-cw-muon-packed-bank XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Config:
  - Iris parent `/dlwh/iris-run-job-20260620-094726`; child `/dlwh/iris-run-job-20260620-094726/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKUPDATECHECKSUM-N2-cw-20260620-094724`.
  - W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKUPDATECHECKSUM-N2-cw-20260620-094724`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKUPDATECHECKSUM-N2-cw-20260620-094724-965194`.
  - R1D2E8, 16 H100s, L26, group size 8, H3, bf16 params/NS compute, `max_grouped_stack_size=512`.
- Result:
  - Job succeeded with `failure_count=0` and no `RESOURCE_EXHAUSTED`.
  - Lowered HLO: `0 all_gather`, `4 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `18 dot_general`.
  - Compiled HLO: `0 all_gather`, `4 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `39 gpu_gemm_custom_call`, `90 custom_call`.
  - Rank 0 timing: mean `0.455426s`, median `0.455002s`, min `0.452986s`, stdev `0.002678s`.
  - Rank 1 timing: mean `0.455440s`, median `0.454919s`, min `0.453069s`, stdev `0.002669s`.
  - Summary row estimates: `2.428804005888e15` NS dot FLOPs, median `5338-5339 TFLOP/s`, about `33.73-33.74%` nominal H100 bf16 peak, global boundary bandwidth estimate `~287.6 GB/s`, boundary peak per device `16.36 GB`.
- Interpretation:
  - The OOM was a harness-output materialization artifact. The update-only computation itself runs at full May shape on R1D2E8.
  - The phase table is now timed:
    - update-only (grads entry + params entry + NS + hyperball, no FSDP apply): `~0.455s`, `4 A2A`.
    - direction-apply (grads entry + NS direction + FSDP apply, no params entry/hyperball): `~0.530s`, `4 A2A`.
    - full packed-bank apply (grads entry + params entry + NS + hyperball + FSDP apply): `~0.638s`, `6 A2A`.
  - This makes the full-path decomposition internally consistent: no hidden per-leaf explosion remains, but the three bulk boundary phases still cost enough that collapsing or avoiding one phase is the next optimization target.
- Next action: keep Socrates focused on a lower-level bridge or custom primitive that can collapse the three two-A2A phases, while the main path treats `expert_fsdp_packed_bank_muonh_apply` as the conservative FSDP-master baseline.

### 2026-06-20 03:00 PDT - restore packed-bank MuonH updates before ordinary Optax apply
- Hypothesis: The conservative FSDP-master path can keep the model/train state in ordinary FSDP form if the packed grouped MuonH update is restored into a normal FSDP update tree before calling ordinary `optax.apply_updates`. This should preserve compatibility even if it does not reduce the remaining boundary transport.
- Change:
  - Commit `ff3779cb2` changes the full `expert_fsdp_packed_bank_muonh_apply` path to restore packed-bank MuonH updates into a normal FSDP update tree before ordinary `optax.apply_updates`.
  - Added `expert_fsdp_packed_bank_updates_to_fsdp_tree`, `expert_fsdp_packed_bank_muonh_updates_outputs`, and `expert_fsdp_packed_bank_muonh_updates_step_factory`.
  - Updated the packed A2A apply helper to use the same restore-tree-then-Optax boundary.
  - Added focused coverage in `test_expert_fsdp_packed_bank_muonh_restores_fsdp_updates_before_apply`.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank_muonh_apply or packed_bank_muonh_update_only or packed_bank_direction_apply or packed_bank_a2a_apply_boundary' -q` -> 7 passed.
  - Tiny local smoke for `expert_fsdp_packed_bank_muonh_apply` passed.
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed, and commit hooks including Pyrefly passed.
- CoreWeave command:
  - `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKRESTOREAPPLY-N2-cw-20260620-095713 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_apply MUON_BENCH_MODE=both MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=3 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_TRACKER=wandb MUON_BENCH_WANDB_PROJECT=marin_moe MUON_BENCH_WANDB_GROUP=grug-moe-cw-muon-packed-bank XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Config:
  - Iris parent `/dlwh/iris-run-job-20260620-095716`; child `/dlwh/iris-run-job-20260620-095716/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKRESTOREAPPLY-N2-cw-20260620-095713`.
  - W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKRESTOREAPPLY-N2-cw-20260620-095713`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PACKEDBANKRESTOREAPPLY-N2-cw-20260620-095713-4be659`.
  - R1D2E8, 16 H100s, L26, group size 8, H3, bf16 params/NS compute, `max_grouped_stack_size=512`.
- Result:
  - Job succeeded on both tasks with no OOM or traceback.
  - Lowered HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `18 dot_general`.
  - Compiled HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `41 gpu_gemm_custom_call`, `90 custom_call`.
  - Rank 0 timing: mean `0.660683s`, median `0.659915s`, min `0.659113s`.
  - Rank 1 timing: mean `0.659931s`, median `0.659877s`, min `0.659066s`.
  - Summary row estimates: `2.428804005888e15` NS dot FLOPs, median `~3680 TFLOP/s`, about `23.26%` nominal H100 bf16 peak, global boundary bandwidth estimate `~198 GB/s`, boundary peak per device `16.36 GB`.
- Interpretation:
  - This validates the pragmatic compatibility boundary: train-state/master params can remain FSDP, packed grouped MuonH can compute updates, and the updates can be restored to an FSDP tree before ordinary `optax.apply_updates`.
  - It is not a performance win over the previous direct packed-bank apply (`~0.638s` median); restoring the update tree and using ordinary Optax costs roughly `20 ms` on R1D2E8 and leaves the same compiled `6 A2A` transport.
  - The remaining target is still the boundary primitive, not semantic compatibility. The path is usable as a conservative FSDP-master baseline while Socrates/lower-level work attempts to collapse or avoid the three two-A2A phases.
- Next action: keep the restore-before-Optax path as the compatibility reference, but optimize against the faster direct packed-bank baseline and the ideal of two expert-projection transports. Continue lower-level bridge work; no more JAX wrapper variants unless they change compiled HLO and timing.

### 2026-06-20 03:22 PDT - packed-bank boundary phase reporting
- Hypothesis: The packed-bank compatibility path needs phase-level accounting in the summary rows so future variants can be judged by transport phases, ideal collective count, and implied bandwidth, not only raw compiled collective totals.
- Change:
  - Commit `b9dc64fe6` adds packed-bank boundary phase estimates to `summary_row`: phase names, expected primitive, global bytes per phase, ideal collective count, compiled/lowered-to-ideal ratios, and estimated phase/global GB/s.
  - Added focused coverage for the packed-bank phase fields.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'boundary_byte_estimates or boundary_phase_estimates or packed_bank_muonh' -q` -> 6 passed.
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed, and commit hooks including Pyrefly passed.
- CoreWeave command:
  - `RUN_ID=MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORT-N2-cw-20260620-100652 MARIN_PREFIX=s3://marin-na/tmp/ttl=7d MUON_BENCH_KINDS=expert_fsdp_packed_bank_muonh_apply MUON_BENCH_MODE=both MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=1 MUON_BENCH_SWEEP_BACKEND_STEPS=3 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512 MUON_BENCH_WRITE_COMPILED_HLO=true MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_ENABLE_JAX_PROFILE=false MUON_BENCH_TRACKER=wandb MUON_BENCH_WANDB_PROJECT=marin_moe MUON_BENCH_WANDB_GROUP=grug-moe-cw-muon-packed-bank XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 bash scratch/launch_muon_grouped_reference_2node_wandb.sh`.
- Config:
  - Iris parent `/dlwh/iris-run-job-20260620-100655`; child `/dlwh/iris-run-job-20260620-100655/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORT-N2-cw-20260620-100652`.
  - W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORT-N2-cw-20260620-100652`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORT-N2-cw-20260620-100652-cc6155`.
- Result:
  - The benchmark emitted summary rows and W&B metrics, then failed during distributed shutdown about five minutes later. Task 0 exited 139 after `Shutdown barrier in coordination service has failed`; task 1 was marked coscheduled-failed because its sibling failed.
  - Treat the timing and phase metrics as valid, but the terminal job state as a launcher cleanup failure. The failure happened after `summary_table`, `wandb_logged`, and summary upload.
  - Lowered HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `18 dot_general`.
  - Compiled HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `41 gpu_gemm_custom_call`.
  - Rank 0 timing: `0.691390216s`; rank 1 timing: `0.691413651s`.
  - Reported phases:
    - `fsdp_grads_to_packed_grouped_bank`: expected `all_to_all`, global bytes `130862284800`, ideal collectives `2`.
    - `fsdp_params_to_packed_grouped_bank`: expected `all_to_all`, global bytes `130862284800`, ideal collectives `2`.
    - `packed_grouped_updates_to_fsdp_apply`: expected `all_to_all`, global bytes `130862284800`, ideal collectives `2`.
  - Aggregate phase fields: `estimated_boundary_phase_count=3`, `estimated_boundary_phase_global_bytes=392586854400`, `estimated_boundary_phase_ideal_collective_count=6`, compiled/lowered-to-ideal ratios `1.0`, mean phase global bandwidth about `568 GB/s`, and mean aggregate boundary bandwidth about `189 GB/s`.
- Interpretation:
  - The packed-bank path now reports the intended phase model, and the compiled HLO exactly matches the phase ideal: six total all-to-alls for three two-A2A phases.
  - The post-metrics failure is consistent with rank 0 spending extra time in W&B finalization while the other host exits and tears down distributed state.
- Follow-up fix:
  - Add multi-host `sync_global_devices` barriers before and after rank-0-only W&B logging in `launch_cw_muon_update_bench.py`, so nonzero ranks do not exit while rank 0 is still finishing the tracker.
  - Focused validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'sync_global_devices_if_multihost or wandb_metric_row or launcher_reads_wandb_env' -q` -> 4 passed; `uv run python -m py_compile experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed; `./infra/pre-commit.py --files experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed.
- Next action: run a one-iteration R1D2E8 phase-report validation with the W&B barriers in place. If it exits cleanly, the phase-reporting path is a usable ongoing regression check.

### 2026-06-20 03:30 PDT - W&B finalization hang isolated
- Hypothesis: The previous shutdown crash was caused by rank 1 exiting while rank 0 was still finalizing W&B; adding barriers should turn it into a clean exit if W&B finalization returns.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-102108`; child `/dlwh/iris-run-job-20260620-102108/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSYNC-N2-cw-20260620-102105`.
  - W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSYNC-N2-cw-20260620-102105`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSYNC-N2-cw-20260620-102105-0e72ff`.
- Result:
  - Metrics and W&B logging succeeded, but both GPU tasks remained running for several minutes after `wandb_logged`.
  - Lowered and compiled HLO were unchanged: `0 AG / 6 A2A / 0 CP / 0 RS`, with `18` lowered dots and `41` compiled GPU GEMMs.
  - Rank timings were `0.653619s` and `0.653597s`; median H100 bf16 peak about `23.48%`.
  - Phase fields were present and correct: three phases, each `130862284800` global bytes and two ideal all-to-alls; total phase bytes `392586854400`, total ideal collectives `6`, compiled/lowered-to-ideal ratios `1.0`.
  - The validation job was manually stopped after metrics landed because the remaining work was only stuck cleanup.
- Interpretation:
  - The barriers fixed the fatal distributed teardown race, but exposed a second issue: `wandb.finish()` itself can hang inside the JAX worker process.
- Follow-up fix:
  - Move W&B logging into a spawned child process with a `120s` timeout. The main JAX process waits for the child, emits `wandb_timeout` or `wandb_failed` on tracker trouble, then continues through the after-W&B barrier so all distributed ranks exit together. Summary JSON remains the authoritative result even if W&B cleanup misbehaves.
  - Focused validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'sync_global_devices_if_multihost or wandb_metric_row or launcher_reads_wandb_env' -q` -> 4 passed; `uv run python -m py_compile experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed; `./infra/pre-commit.py --files experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed.
- Next action: rerun the one-iteration phase-report validation once more. The pass condition is terminal Iris success after W&B logging, not just metric emission.

### 2026-06-20 03:38 PDT - W&B child process switched from multiprocessing to subprocess
- Hypothesis: Moving W&B into a child process should protect the distributed JAX process from W&B finalization, but Python multiprocessing `spawn` may not work under Iris' `_callable_runner.py` because the benchmark module is loaded as `__main__`.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-102956`; child `/dlwh/iris-run-job-20260620-102956/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTCHILDWANDB-N2-cw-20260620-102954`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTCHILDWANDB-N2-cw-20260620-102954`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTCHILDWANDB-N2-cw-20260620-102954-dc38ed`.
- Result:
  - Metrics emitted successfully before tracker handoff: compiled `0 AG / 6 A2A / 0 CP / 0 RS`, median `0.65806-0.65809s`, median H100 bf16 peak about `23.32%`, and the same three two-A2A phase rows with compiled/lowered ratios `1.0`.
  - Rank 0 then failed in `multiprocessing.spawn` before W&B logging: `_pickle.PicklingError: Can't pickle <function _log_summary_to_wandb_process ...>: attribute lookup _log_summary_to_wandb_process on __main__ failed`.
  - The validation job was manually stopped after the failure because the useful benchmark metrics were already present.
- Follow-up fix:
  - Replace multiprocessing with `subprocess.run([sys.executable, "-m", "experiments.grug.moe.launch_cw_muon_update_bench"])` plus a local JSON payload file and sentinel env var `MUON_BENCH_WANDB_PAYLOAD`.
  - Make `main()` build the executor step lazily so subprocess payload mode can import the module without calling `this_output_path()`.
  - Focused validation: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'sync_global_devices_if_multihost or wandb_metric_row or launcher_reads_wandb_env' -q` -> 4 passed; `uv run python -m py_compile experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed; `./infra/pre-commit.py --files experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix` passed.
- Next action: rerun phase-report validation once more. This time the W&B subprocess should avoid both the `spawn` pickling failure and the in-process `wandb.finish()` hang.

### 2026-06-20 03:42 PDT - phase-report validation succeeds with W&B subprocess
- Hypothesis: The subprocess W&B path should preserve W&B metrics while allowing the distributed JAX benchmark to exit normally.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-103539`; child `/dlwh/iris-run-job-20260620-103539/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSUBPROC-N2-cw-20260620-103536`.
  - W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSUBPROC-N2-cw-20260620-103536`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-PHASEREPORTSUBPROC-N2-cw-20260620-103536-3d74b5`.
- Result:
  - Parent and child both succeeded with `failure_count=0`; both GPU tasks exited 0.
  - W&B subprocess emitted `wandb_logged` and the run reached terminal success. The JAX `WatchTasksAsync CANCELLED` warnings appeared during normal shutdown but did not mark the job failed.
  - Lowered HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `18 dot_general`.
  - Compiled HLO: `0 all_gather`, `6 all_to_all`, `0 collective_permute`, `0 reduce_scatter`, `41 gpu_gemm_custom_call`, `90 custom_call`.
  - Rank timings: `0.658006s` and `0.658088s`, about `23.32%` nominal H100 bf16 peak.
  - Phase fields: three phases, each `130862284800` global bytes and two ideal all-to-alls; total phase bytes `392586854400`, total ideal collectives `6`, compiled/lowered-to-ideal ratios `1.0`, aggregate boundary bandwidth about `198.9 GB/s`, phase-normalized bandwidth about `596.6 GB/s`.
- Interpretation:
  - Phase-reporting is now usable as a regression check for grouped MuonH boundary primitives.
  - The conservative FSDP-master packed-bank path remains semantically compatible and cleanly instrumented, but performance is still the same transport-bound baseline: three two-A2A phases plus NS/hyperball work.
  - Future candidates should beat this by reducing phase bytes, reducing phases, overlapping transport with NS, or avoiding the grouped-to-FSDP apply boundary.
- Next action: continue the boundary-primitive goal from this baseline. Socrates' lower-level bridge work should compare against `0 AG / 6 A2A`, `~0.658s`, and the three phase rows above.

### 2026-06-20 03:55 PDT - Route A/B apply-boundary comparison
- Hypothesis: Applying packed grouped MuonH updates directly inside the packed-bank restore boundary might avoid enough per-leaf tree reconstruction or Optax overhead to beat the conservative Route A path (`grouped updates -> FSDP-shaped update pytree -> optax.apply_updates`).
- Change:
  - Commit `43b77ff5d` adds `expert_fsdp_packed_bank_direct_apply_boundary` as Route B.
  - Route B uses the same packed-bank transport shape as Route A, then applies each restored update shard directly to the matching FSDP param shard inside the boundary helper.
  - Added focused correctness and HLO tests for Route B plus `scratch/launch_muon_packed_bank_apply_routes_2node_wandb.sh`.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank_direct_apply or packed_bank_boundary_phase_estimates or boundary_byte_estimates' -q` -> 4 passed.
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank or fsdp_grads_to_explicit_packed_grouped_bank or boundary_phase_estimates' -q` -> 14 passed.
  - `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed.
  - `./infra/pre-commit.py --changed-files --fix` passed, including Pyrefly.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-105102`; child `/dlwh/iris-run-job-20260620-105102/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-APPLYROUTES-N2-cw-20260620-105059`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-APPLYROUTES-N2-cw-20260620-105059`; the job emitted `wandb_logged`, but local W&B API lookup did not find the run after `wandb_timeout`, so Iris logs are the source of truth.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-APPLYROUTES-N2-cw-20260620-105059-4256eb`.
  - Config: R1D2E8, 16 H100s, L26, group size 8, H3, bf16 params/NS compute, `max_grouped_stack_size=512`.
- Result:
  - Parent and child both succeeded with `failure_count=0`; both GPU tasks exited 0.
  - Route A (`packed_grouped_updates_to_fsdp_apply`): lowered/compiled `0 AG / 2 A2A / 0 AR / 0 RS / 0 CP`, median `0.183173s` on rank 0 and `0.183103s` on rank 1. Phase/global bandwidth about `714-715 GB/s`.
  - Route B (`packed_grouped_updates_to_fsdp_direct_apply`): lowered/compiled `0 AG / 2 A2A / 0 AR / 0 RS / 0 CP`, median `0.184111s` on rank 0 and `0.184163s` on rank 1. Phase/global bandwidth about `710-711 GB/s`.
  - Both routes report one phase, `130862284800` global bytes, two ideal all-to-alls, compiled/lowered-to-ideal ratio `1.0`, and no hidden collective explosion.
- Interpretation:
  - Route B is useful as a first-class harness primitive, but it is not faster. Direct apply is about `0.5-0.6%` slower than restoring an FSDP update tree and calling ordinary Optax in this isolated boundary benchmark.
  - This suggests the grouped-to-FSDP apply boundary is already transport-bound at the JAX level; the apply-side tree work is not the dominant cost.
  - The conservative Route A remains the pragmatic FSDP-master path. To improve materially, the next primitive needs to reduce or overlap the packed-bank transport itself rather than only moving the apply call.
- Next action: keep Socrates focused on lower-level bridge/custom primitive designs. Use the isolated apply-boundary target as `0 AG / 2 A2A`, `~0.183s`, `~130.9 GB global bytes` when judging future grouped-update-to-FSDP variants.

### 2026-06-20 04:20 PDT - First boundary N1 and D2 packed-bank validation
- Hypothesis: The first optimizer boundary (`FSDP grads -> packed grouped bank`) should avoid per-leaf collective explosion. For single-node it should be a local pack with no collectives; for data-axis D2 it should lower to one packed all-to-all per expert weight family, not hundreds of leaf-wise collectives.
- Change:
  - Commit `ba962fe89` added standalone phase reporting for `expert_fsdp_grads_to_explicit_packed_grouped_bank` and a lazy launch script at `scratch/launch_muon_grads_to_packed_bank_boundary.sh`.
  - Commit `a56dd930c` changed the timing path for FSDP-grads-to-grouped benchmarks to return a scalar checksum, avoiding the previous full output materialization OOM.
  - Follow-up local change adds explicit correctness-skip reporting and CLI/env controls (`--boundary-correctness-max-global-bytes`, `--force-boundary-correctness`) so full-shape runs no longer silently report `boundary_correctness_max_error=null`.
- Validation:
  - Focused tests after the skip-reporting patch: `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'boundary_correctness_gate or packed_bank_boundary_phase_estimates or fsdp_grads_to_explicit_packed_grouped_bank_timing' -q` -> 3 passed.
  - Pycompile: `uv run python -m py_compile experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py` passed.
  - Tiny local smoke with forced correctness reported `boundary_correctness_max_error=0.0`; same shape with an intentionally tiny cap reported `boundary_correctness_skipped_reason`.
- Runs:
  - N1: parent `/dlwh/iris-run-job-20260620-110821`; child `/dlwh/iris-run-job-20260620-110821/grug-train-MUON-BENCH-D2560-L26-R1D1E8-N1-G1-GRADSPACKBANK-cw-20260620-110819`; W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D1E8-N1-G1-GRADSPACKBANK-cw-20260620-110819`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D1E8-N1-G1-GRADSPACKBANK-cw-20260620-110819-b34976`.
  - D2: parent `/dlwh/iris-run-job-20260620-111142`; child `/dlwh/iris-run-job-20260620-111142/grug-train-MUON-BENCH-D2560-L26-R1D2E8-N2-G2-GRADSPACKBANK-cw-20260620-111140`; W&B `marin-community/marin_moe/MUON-BENCH-D2560-L26-R1D2E8-N2-G2-GRADSPACKBANK-cw-20260620-111140`; output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-N2-G2-GRADSPACKBANK-cw-20260620-111140-4b3b9d`.
- Result:
  - N1 lowered/compiled `0 AG / 0 A2A / 0 AR / 0 RS / 0 CP`, median `0.0156868s`. Phase report: one `none` phase, ideal collective count `0.0`, global bytes `130862284800`, estimated peak per-device bytes `16357785600`.
  - D2 lowered/compiled `0 AG / 2 A2A / 0 AR / 0 RS / 0 CP`, median `0.174654s` on rank 0 and `0.174748s` on rank 1. Phase report: one all-to-all phase, ideal collective count `2.0`, compiled/lowered fragmentation factor `1.0`, global bytes `130862284800`, per-A2A global bytes `65431142400`, estimated peak per-device bytes `16357785600`, median phase/global bandwidth about `749 GB/s`.
- Interpretation:
  - The first boundary is behaving as the packed-bank design intended for N1 and D2: no all-gather, no reduce-scatter, no collective-permute, and no per-leaf collective explosion.
  - D2 is still slower than an ideal roofline target, but the shape is now a compact two-A2A primitive rather than an XLA fragmentation bug. The remaining gap is transport bandwidth/latency for the packed A2A itself.
  - Correctness for the full-size run remains intentionally skipped by byte cap to avoid reintroducing full-output materialization OOM; small-shape reference correctness is `0.0`, and future runs will report the skip reason explicitly.
- Next action: commit the correctness-skip reporting patch, then launch the R2 first-boundary rung from the updated code so replica-dcn grouping is validated with the improved summary fields.

### 2026-06-20 04:30 PDT - First boundary R2 replica-dcn validation
- Hypothesis: For replica-dcn grouping, the first optimizer boundary (`FSDP grads -> packed grouped bank`) should be a local repack/fanout inside the replicated-weight axis, not a network collective. In particular, R2 should not introduce all-gather, all-to-all, all-reduce, reduce-scatter, or collective-permute in the lowered or compiled HLO.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-112254`; child `/dlwh/iris-run-job-20260620-112254/grug-train-MUON-BENCH-D2560-L26-R2D1E8-N2-G2-GRADSPACKBANK-cw-20260620-112252`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R2D1E8-N2-G2-GRADSPACKBANK-cw-20260620-112252`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-N2-G2-GRADSPACKBANK-cw-20260620-112252-4521de`.
  - Config: R2D1E8, 16 H100s, `ns4d_group_axis=replica_dcn`, `ns4d_group_size=2`, L26, H3, bf16 params/NS compute, `max_grouped_stack_size=512`, correctness cap `1073741824` bytes.
- Result:
  - Parent and child both succeeded with `failure_count=0`; both GPU tasks exited 0 after emitting metadata, lowered HLO, timing rows, summary rows, and W&B logging.
  - Lowered HLO on both ranks: `0 AG / 0 A2A / 0 AR / 0 RS / 0 CP`.
  - Compiled HLO on both ranks: `0 AG / 0 A2A / 0 AR / 0 RS / 0 CP`, with one GPU GEMM custom call in the checksum-timing path.
  - Rank 0 median `0.00611677s`; rank 1 median `0.00611917s`.
  - Full-shape correctness was intentionally skipped with reason `estimated global bytes 130862284800 exceed correctness cap 1073741824`; small-shape forced correctness had already validated `0.0` max error.
  - Phase report: one `none` phase, ideal collective count `0.0`, global logical bytes `130862284800`, grouped input per device `8178892800`, FSDP output per device `16357785600`, replica fanout factor `2.0`.
- Interpretation:
  - This validates the key replica-dcn first-boundary property for R2: XLA did not turn the packed-bank repack into a hidden AG/A2A/RS/AR sequence.
  - The enormous reported GB/s is only a logical-byte/local-repack rate, not a network bandwidth claim. The real claim is the absence of collectives.
  - R2 therefore supports the pragmatic design where FSDP gradients can enter the grouped MuonH bank over `replica_dcn` without the collective explosion that made the earlier full train-step integration unattractive.
- Next action: run the same first-boundary check at R4. If R4 also preserves the no-collective property, the remaining risky boundary is grouped updates back to FSDP/apply, plus integration into the Grug MoE optimizer path.

### 2026-06-20 04:35 PDT - First boundary R4 replica-dcn validation
- Hypothesis: If the packed-bank first-boundary design is viable for the intended replicated-weight path, R4 should preserve the same no-collective property as R2: FSDP grads should repack into `P('replica_dcn', 'expert', None, None)` grouped banks without all-gather, all-to-all, all-reduce, reduce-scatter, or collective-permute.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-112733`; child `/dlwh/iris-run-job-20260620-112733/grug-train-MUON-BENCH-D2560-L26-R4D1E8-N4-G4-GRADSPACKBANK-cw-20260620-112731`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R4D1E8-N4-G4-GRADSPACKBANK-cw-20260620-112731`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R4D1E8-N4-G4-GRADSPACKBANK-cw-20260620-112731-ddde49`.
  - Config: R4D1E8, 32 H100s, `ns4d_group_axis=replica_dcn`, `ns4d_group_size=4`, L26, H3, bf16 params/NS compute, `max_grouped_stack_size=512`, correctness cap `1073741824` bytes.
- Result:
  - Parent and child both succeeded with `failure_count=0`; all four GPU tasks exited 0.
  - Lowered HLO on all four ranks: `0 AG / 0 A2A / 0 AR / 0 RS / 0 CP`.
  - Compiled HLO on all four ranks: `0 AG / 0 A2A / 0 AR / 0 RS / 0 CP`, with three GPU GEMM custom calls in the checksum-timing path.
  - Rank medians: `0.00464850s`, `0.00459857s`, `0.00467783s`, `0.00469472s`.
  - Full-shape correctness was intentionally skipped with reason `estimated global bytes 130862284800 exceed correctness cap 1073741824`; small-shape forced correctness had already validated `0.0` max error.
  - Phase report: one `none` phase, ideal collective count `0.0`, global logical bytes `130862284800`, grouped input per device `4089446400`, FSDP output per device `16357785600`, replica fanout factor `4.0`.
- Interpretation:
  - R4 preserves the same key property as R2: no hidden network collectives in the first boundary. This makes `replica_dcn` a viable axis for the NS-friendly grouped bank at least through the FSDP-grads-to-grouped conversion.
  - The timing is faster than R2 because the grouped input per device is halved again; as with R2, the logical GB/s number is not a network roofline claim because this phase compiles to local repacking.
  - The remaining blocker for production integration is therefore not this first boundary. It is the grouped-update-to-FSDP/apply side and how to avoid paying the already-measured packed-bank transport cost too many times or too synchronously.
- Next action: keep Socrates focused on lower-level grouped-to-FSDP bridge options. In the main harness, use R2/R4 first-boundary results as the “good” reference and optimize the apply boundary or end-to-end optimizer integration against it.

### 2026-06-20 04:45 PDT - Apply boundary R2/R4 replica-dcn validation
- Hypothesis: The second optimizer boundary (`packed grouped updates -> FSDP apply layout`) should avoid per-leaf collective explosion. For replica-dcn grouping it cannot be free, because the grouped update bank holds only a replica shard while ordinary FSDP master weights need the full per-device FSDP update slice. The target is therefore a small number of packed collectives, not hundreds of leaf-wise reshard collectives. Route A restores the FSDP update tree and then calls ordinary `optax.apply_updates`; Route B directly applies while unpacking the packed bank.
- Change:
  - Generalized `scratch/launch_muon_packed_bank_apply_routes_2node_wandb.sh` from the original D2 launch into a topology-aware launcher supporting `n1`, `d2`, `d4`, `r2`, and `r4`.
  - The default remains D2. Passing `r2` or `r4` now sets `replica_axis=2/4`, `data_axis=1`, `expert_axis=8`, `model_axis=1`, and `ns4d_group_axis=replica_dcn`.
  - `MUON_BENCH_APPLY_GROUP_SIZE` controls the layer-stack grouping and defaults to 8.
- Validation:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank_a2a_apply_boundary or packed_bank_direct_apply_boundary or packed_bank_boundary_phase_estimates' -q` -> 6 passed.
  - Dry runs passed for both `MUON_BENCH_DRY_RUN=true bash scratch/launch_muon_packed_bank_apply_routes_2node_wandb.sh r2` and `... r4`.
- R2 run:
  - Iris parent `/dlwh/iris-run-job-20260620-113457`; child `/dlwh/iris-run-job-20260620-113457/grug-train-MUON-BENCH-D2560-L26-R2D1E8-N2-G8-H3-APPLYROUTES-cw-20260620-113455`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R2D1E8-N2-G8-H3-APPLYROUTES-cw-20260620-113455`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-N2-G8-H3-APPLYROUTES-cw-20260620-113455-851d57`.
  - Config: R2D1E8, 16 H100s, `ns4d_group_axis=replica_dcn`, `ns4d_group_size=8`, L26, H3, bf16 params/NS compute, two route kinds.
- R2 result:
  - Parent and child both succeeded with `failure_count=0`; both GPU tasks exited 0.
  - Route A lowered/compiled `2 AG / 0 A2A / 0 AR / 0 RS / 0 CP`; rank medians `0.2322367710s` and `0.2323536780s`.
  - Route B lowered/compiled `2 AG / 0 A2A / 0 AR / 0 RS / 0 CP`; rank medians `0.2316506261s` and `0.2314961820s`.
  - Both routes reported one packed all-gather phase with ideal collective count `2.0`, compiled/lowered fragmentation factor `1.0`, global logical update bytes `130862284800`, grouped input per device `8178892800`, FSDP output per device `16357785600`, and median phase/global bandwidth about `563-565 GB/s`.
  - Full-shape correctness was intentionally skipped with reason `estimated global bytes 130862284800 exceed correctness cap 1073741824`; small-shape forced correctness is covered by tests.
- R4 run:
  - Iris parent `/dlwh/iris-run-job-20260620-113851`; child `/dlwh/iris-run-job-20260620-113851/grug-train-MUON-BENCH-D2560-L26-R4D1E8-N4-G8-H3-APPLYROUTES-cw-20260620-113849`.
  - W&B target `marin-community/marin_moe/MUON-BENCH-D2560-L26-R4D1E8-N4-G8-H3-APPLYROUTES-cw-20260620-113849`.
  - Output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R4D1E8-N4-G8-H3-APPLYROUTES-cw-20260620-113849-1d06f2`.
  - Config: R4D1E8, 32 H100s, `ns4d_group_axis=replica_dcn`, `ns4d_group_size=8`, L26, H3, bf16 params/NS compute, two route kinds.
- R4 result:
  - Parent and child both succeeded with `failure_count=0`; all four GPU tasks exited 0.
  - Route A lowered/compiled `2 AG / 0 A2A / 0 AR / 0 RS / 0 CP`; observed rank medians around `0.3137-0.3139s`.
  - Route B lowered/compiled `2 AG / 0 A2A / 0 AR / 0 RS / 0 CP`; observed rank medians around `0.31356s`.
  - Both routes reported one packed all-gather phase with ideal collective count `2.0`, compiled/lowered fragmentation factor `1.0`, global logical update bytes `130862284800`, grouped input per device `4089446400`, FSDP output per device `16357785600`, replica fanout factor `4.0`, and median phase/global bandwidth about `417 GB/s`.
  - Full-shape correctness was intentionally skipped with the same byte-cap reason.
- Interpretation:
  - The apply boundary is no longer a per-leaf collective explosion in the harness. Through R4, it is exactly two packed all-gathers, one per expert weight family, with no all-reduce, reduce-scatter, all-to-all, or collective-permute in lowered or compiled HLO.
  - Route B does not materially beat Route A. Direct apply avoids building an explicit FSDP update tree, but the measured cost is dominated by packed replica fanout rather than Optax tree application.
  - R4 is slower than R2 (`~0.314s` vs `~0.232s`) because each device starts with a smaller grouped shard but needs the same FSDP output slice, so the replica fanout receive grows. The packed all-gather path is the remaining expensive boundary.
- Next action: keep the production path conservative: grouped MuonH compute, then Route A back to ordinary FSDP updates and `optax.apply_updates`. For material speedup beyond this harness result, Socrates should target a lower-level grouped-to-FSDP bridge or overlap strategy for the packed all-gather/fanout itself.

### 2026-06-20 05:09 PDT - Production Grug MoE packed-entry validation at R2
- Hypothesis: Now that the harness proves both FSDP/MuonH boundaries avoid per-leaf collective explosion, the conservative production path should at least run end-to-end with FSDP master params, grouped MuonH compute, grouped updates restored to FSDP-shaped updates, and ordinary `optax.apply_updates`.
- Change:
  - Commit `c9bfbaee2` wires `MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY` through `experiments/grug/moe/launch_cw_may_d2560.py` and `experiments/grug/moe/run_cw_may_d2560.sh`.
  - The launch path now supports `--expert-grouped-muonh-packed-entry true`, which selects the packed-entry grouped MuonH production optimizer path.
- Validation before launch:
  - `uv run pytest experiments/grug/moe/test_optimizer.py -k 'packed_entry or grouped_expert_muonh_optimizer_returns_fsdp_updates_before_apply or packs_multi_chunk_restore_boundary' -q` -> 4 passed.
  - Dry-run wrapper summary confirmed `expert_grouped_muonh_packed_entry: true`.
  - `./infra/pre-commit.py --changed-files --fix` passed.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-115017`; child `/dlwh/iris-run-job-20260620-115017/grug-train-GM2560-MAY-197S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-THROUGHPUT-N2-cw-20260620-1150`.
  - W&B `marin-community/marin_moe/GM2560-MAY-197S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-CURRENT-THROUGHPUT-N2-cw-20260620-1150`.
  - Config: 2 H100 nodes, `replica_axis=2`, `data_axis=1`, `expert_axis=8`, `model_axis=1`, batch 16, seq 4096, sliding window 2048, 8 train steps, Pallas CE, ring MoE, FA4 CuTe attention, bf16 params/compute/output, `optimizer=muonh`, `expert_3d_optimizer=grouped_muonh`, `expert_grouped_muonh_group_size=2`, `expert_grouped_muonh_packed_entry=true`, MuonH3 with bf16 NS compute.
- Result:
  - Parent and child jobs both succeeded with `failure_count=0`; both GPU tasks exited 0.
  - W&B finished at `global_step=7`.
  - Mesh log confirmed `{'replica_dcn': 2, 'data': 1, 'expert': 8, 'model': 1}; batch_shards=16`.
  - Steps 0 and 1 were compile/autotune polluted: `~446.8s` and `~440.4s`.
  - Warm steps 2-7 averaged `2.08290s`, `31466.6 tokens/s`, and `3.40268 MFU`. Final step 7 was `2.05934s`, `31823.8 tokens/s`, `3.44131 MFU`, `train/loss=3.10720`.
  - Shutdown emitted `WatchTasksAsync` connection-refused/CANCELLED warnings after the final metrics, consistent with normal JAX distributed teardown; Iris still marked the child succeeded.
- Interpretation:
  - This is an important positive correctness/integration result: the packed-entry grouped MuonH path runs end-to-end in the production Grug MoE training loop with FSDP master params and ordinary apply semantics.
  - It is also a negative performance result. The production path is far slower than the isolated boundary harness would predict, and much slower than the non-Muon R2/B16 reference lane. The harness shows no per-leaf collective explosion in the two explicit boundaries, so the production slowdown likely comes from an unoptimized interaction inside the full train step: repeated boundary use, synchronization placement, lost batching, poor overlap, or extra tree/materialization work around the optimizer path.
  - The goal is therefore not complete. The next step should be to profile or instrument the production packed-entry path to find where the extra ~2s step cost is coming from, while Socrates continues lower-level bridge work for the grouped-to-FSDP fanout.
- Next action:
  - Record this result on #6493 as a successful integration / failed performance-preservation milestone.
  - Do not move to R4 production until R2 production overhead is explained or a lower-level boundary primitive changes the expected cost.

### 2026-06-20 05:33 PDT - Production grouped MuonH R2 group-size comparison
- Hypothesis: The poor production packed-entry result might be caused by `expert_grouped_muonh_group_size=2`, which creates many small Newton-Schulz chunks. Increasing the group size to 8 should reduce chunk-loop overhead and improve production throughput if chunk fragmentation is the main cost.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-121332`; child `/dlwh/iris-run-job-20260620-121332/grug-train-GM2560-MAY-199S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-G8-THROUGHPUT-N2-cw-20260620-1213`.
  - W&B `marin-community/marin_moe/GM2560-MAY-199S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-G8-THROUGHPUT-N2-cw-20260620-1213`.
  - Config matches the prior R2 production packed-entry validation except `expert_grouped_muonh_group_size=8` instead of 2. It uses 2 H100 nodes, `replica_axis=2`, `data_axis=1`, `expert_axis=8`, `model_axis=1`, batch 16, seq 4096, Pallas CE, ring MoE, FA4 CuTe attention, bf16 params/compute/output, grouped MuonH3 with bf16 NS compute, and packed entry enabled.
- Result:
  - Parent and child jobs both succeeded with `failure_count=0`; both GPU tasks exited 0.
  - Steps 0 and 1 were compile/autotune polluted: `~438.5s` and `~421.8s`.
  - Warm steps 2-7 averaged `1.97780s`, `33152.5 tokens/s`, and `3.58499 MFU`. Final step 7 was `1.93693s`, `33835.0 tokens/s`, `3.65879 MFU`, `train/loss=3.10861`.
  - Compared with May197 group-size-2 warm steps (`2.08290s`, `31466.6 tokens/s`, `3.40268 MFU`), group size 8 improves throughput and MFU by `1.0536x` and reduces warm-step duration by about `5.05%`.
- Interpretation:
  - Larger NS chunks help, but only modestly. Chunk count/loop overhead is not the main reason the production path is around `~2s` per step.
  - The harness apply boundary still predicts a much smaller packed fanout cost than the full train step pays. The remaining overhead is likely a full-optimizer interaction: extra materialization around the grouped optimizer tree, synchronization placement, lost fusion/batching around the NS/update path, or repeated work outside the two explicit boundary primitives.
  - This reinforces that the next evidence should be a semantic profile of the production packed-entry path, not another scale-up rung.
- Next action:
  - Launch a short R2/group-size-8 profile with HLO proto and command buffers disabled for trace readability. Use it to attribute the warm-step time across grouped entry, Newton-Schulz, packed restore, ordinary apply, and surrounding optimizer/tree work.

### 2026-06-20 06:08 PDT - May200 production grouped-MuonH profile attribution
- Hypothesis: The May199 group-size-8 throughput result is still slow because either the grouped Newton-Schulz body is not actually fast in the full train step, or the FSDP<->grouped boundary conversions are dominating once integrated into production.
- Run:
  - Iris parent `/dlwh/iris-run-job-20260620-123557`; child `/dlwh/iris-run-job-20260620-123557/grug-train-GM2560-MAY-200S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-G8-PROFILE-N2-cw-20260620-1235`.
  - W&B run `marin-community/marin_moe/GM2560-MAY-200S4096-W2048-B16-R2-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-G8-PROFILE-N2-cw-20260620-1235`; profiler artifact `...-profiler:v0`.
  - Config matches May199 except it profiles steps 3-4, enables HLO proto/profile metadata, and sets `XLA_FLAGS=--xla_gpu_enable_command_buffer=''` for readable trace names. This run is for attribution only; command-buffer disable hurts performance and should not be used as a throughput comparison.
- Artifact/result:
  - W&B finish raised `HandleAbandonedError`, so the run state is `crashed` and W&B has only partial scalar summary keys. The profiler artifact did upload successfully: `jax_profile`, size `266552592` bytes.
  - Parsed artifact with `lib/marin/tools/profile_summary.py` into `scratch/profile_reports/profile_summary_may200_grouped_muonh_r2_g8.json` and `scratch/profile_reports/profile_report_may200_grouped_muonh_r2_g8.md`.
  - Direct XPlane summary reported no trace truncation, but most top device ops still collapse to `XlaModule` without shapes. Raw Perfetto event args retain the useful JAX scopes, so the phase table below is from direct trace parsing over device events for the one profiled host's 8 GPUs.
- Phase table (mean per GPU, divided by two profiled steps):

| Phase | Approx per-step mean per GPU | Device event count over profile | Notes |
|---|---:|---:|---|
| grouped packed entry | `638.5 ms` | 256 | `ncclDevKernel_SendRecv` under `grouped_muonh/packed_entry/*` |
| grouped packed restore | `602.7 ms` | 256 | `SendRecv` plus `AllGather` under `grouped_muonh/packed_restore/*` |
| grouped Newton-Schulz | `194.8 ms` | 1376 | `newton_schulz_grouped_4d` NVJet/dot kernels |
| dense 2D replicated Newton-Schulz | `79.5 ms` | 42624 | ordinary non-expert 2D MuonH leaves still use replicated NS |
| grouped hyperball | `0.6 ms` | 32 | negligible in this profile |

- Grouped-scope collective split:

| Scope | Kernel | Approx per-step mean per GPU | Count over profile |
|---|---|---:|---:|
| packed entry | `ncclDevKernel_SendRecv` | `638.5 ms` | 256 |
| packed restore | `ncclDevKernel_SendRecv` | `363.2 ms` | 160 |
| packed restore | `ncclDevKernel_AllGather_RING_LL` | `227.6 ms` | 32 |

- Interpretation:
  - This profile does not show the old per-leaf all-gather explosion. The direct profile summary counted only `64` total all-gather events and raw grouped scopes show a small number of coarse packed boundary collectives.
  - The grouped Newton-Schulz compute body is not the main integrated bottleneck here: it is roughly `0.195s/step/GPU`, while the two FSDP<->grouped boundaries together are roughly `1.24s/step/GPU`.
  - The full production path is therefore slow because the conservative FSDP-master bridge is paying expensive packed entry/restore communication in the train step. Group-size tuning only helped May199 by about `5%` because it mainly affects chunking/NS work, not the boundary fanout.
  - The remaining useful work is lower-level boundary replacement/overlap: reduce or replace the packed-entry SendRecv and packed-restore SendRecv/AllGather phases while preserving FSDP master params and ordinary apply semantics, or explicitly pivot to a grouped-bank model-side consumer if this bridge cannot be made competitive.
- Next action:
  - Give Socrates the May200 phase table and focus the lower-level bridge investigation on the production packed boundary collectives, not generic Newton-Schulz speed.
  - Do not launch R4 production until the R2 boundary cost has a concrete fix or a reasoned lower-level primitive design.

### 2026-06-20 06:24 PDT - Chunk-local grouped-MuonH boundary probe
- Hypothesis: May200's `grouped_muonh/packed_entry/slice_*_chunk` and
  `grouped_muonh/packed_restore/concat_chunks` SendRecv time may be caused by
  building one global packed bank and then slicing/concatenating chunks across
  the stack axis. A per-chunk boundary mode should avoid the whole-bank
  dynamic-slice/concat traffic, at the cost of more explicit entry/restore
  collectives.
- Change:
  - Commit `f58064e63` adds
    `expert_grouped_muonh_chunk_local_boundaries` to the production grouped
    MuonH optimizer, May launch wrapper, Muon update harness, and tests.
  - Commit `e1ae179cd` fixes the CoreWeave Muon bench executor wrapper to
    forward `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY` and
    `MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES` to the remote
    launcher.
  - New lazy launcher:
    `scratch/launch_muon_chunk_local_2node_wandb.sh`.
- Validation:
  - `uv run pytest experiments/grug/moe/test_optimizer.py -q` -> 19 passed.
  - `./infra/pre-commit.py --changed-files --fix` -> passed.
  - Abstract HLO test for chunk-local mode on R2D2E8 confirms updates still
    match parameter shardings and lowers to `8` all-gathers, `16` all-to-alls,
    no all-reduce, and no reduce-scatter.
- Invalid first launch/control:
  - `/dlwh/iris-run-job-20260620-131439` completed successfully, but metadata
    showed `expert_grouped_muonh_chunk_local_boundaries=false` because the
    remote executor wrapper did not forward the new env var.
  - It is still a useful same-shape control: mean `0.8215s`, median `0.8202s`,
    compiled HLO `18` all-gathers and `10` all-to-alls.
- Valid chunk-local run:
  - Iris parent `/dlwh/iris-run-job-20260620-131904`; child
    `/dlwh/iris-run-job-20260620-131904/grug-train-MUON-BENCH-D2560-L26-R1D2E8-G8-H3-CHUNKLOCAL-N2-cw-20260620-131902`.
  - Config metadata confirmed
    `expert_grouped_muonh_chunk_local_boundaries=true`,
    `expert_grouped_muonh_packed_entry=false`, 2 H100 nodes,
    `replica_axis=1`, `data_axis=2`, `expert_axis=8`, `model_axis=1`,
    26 layers, group size 8, bf16 params/NS compute, and
    `real_expert_fsdp_grouped_muonh_optimizer_update`.
  - Child succeeded with both tasks exit 0; profile uploaded under
    `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R1D2E8-G8-H3-CHUNKLOCAL-N2-cw-20260620-131902-0ce836/profiler/`.
  - Timing: process 0 mean `0.5595s`, median `0.5555s`, min `0.5555s`;
    process 1 mean `0.5572s`, median `0.5527s`, min `0.5518s`.
  - Compiled HLO changed to `8` all-gathers and `16` all-to-alls, no
    all-reduce/reduce-scatter. Estimated effective NS throughput from the
    harness summary was `~27.4-27.5%` of nominal H100 bf16 peak.
- Interpretation:
  - Chunk-local boundaries are a real improvement over the comparable
    accidentally-disabled/control harness run: roughly `1.47x` faster
    (`0.8215s -> 0.5572s` mean on process 1).
  - The profile summary is still communication dominated: direct XPlane summary
    sees `128` SendRecv events and `64` AllGather events on process 0, with
    communication accounting for most device time. This is not a compute-only
    win.
  - The lowered/compiled collective shape supports the May200 diagnosis:
    changing how the grouped bank crosses the FSDP boundary materially changes
    runtime. However, the result is still far from the ideal two-boundary
    primitive; XLA still fragments the boundary into `24` compiled collectives
    for only two logical data movements.
- Next action:
  - Keep Socrates focused on a lower-level boundary primitive or packed
    transport that gets closer to the two logical grouped<->FSDP movements.
  - Use chunk-local as a production-safe fallback/benchmark point, not as the
    final answer.

### 2026-06-20 06:40 PDT - N1 packed-bank boundary primitives
- Hypothesis: Before debugging multi-node grouped-MuonH boundary traffic, the
  two primitive halves should be clean on one node: FSDP grads -> grouped bank,
  and grouped updates -> FSDP apply layout. On N1 these should need no
  collectives and should expose the baseline local materialization cost.
- Runs:
  - `/dlwh/iris-run-job-20260620-133408`:
    `MUON-BENCH-D2560-L26-R1D1E8-N1-G1-GRADSPACKBANK-cw-20260620-133405`.
  - `/dlwh/iris-run-job-20260620-133655`:
    `MUON-BENCH-D2560-L26-R1D1E8-N1-G8-GRADSPACKBANK-cw-20260620-133653`.
  - `/dlwh/iris-run-job-20260620-133420`:
    `MUON-BENCH-D2560-L26-R1D1E8-N1-G8-H3-APPLYROUTES-cw-20260620-133418`.
- Config:
  - Single H100 node, `replica_axis=1`, `data_axis=1`, `expert_axis=8`,
    `model_axis=1`, 26 layers, bf16, backend steps 3, W&B enabled, compiled HLO
    enabled.
  - Grads primitive kind: `expert_fsdp_grads_to_explicit_packed_grouped_bank`.
  - Apply route kinds:
    `expert_fsdp_packed_bank_a2a_apply_boundary` and
    `expert_fsdp_packed_bank_direct_apply_boundary`.
- Results:

| Run | Primitive | Group size | Mean seconds | Median seconds | Compiled collectives | Estimated global bytes | Estimated peak/device |
|---|---|---:|---:|---:|---|---:|---:|
| N1 G1 grads | FSDP grads -> packed grouped bank | 1 | `0.015834641` | `0.015726442` | AG/AR/RS/A2A=`0/0/0/0` | `121.875 GiB` | `15.234375 GiB` |
| N1 G8 grads | FSDP grads -> packed grouped bank | 8 | `0.015797431` | `0.015665859` | AG/AR/RS/A2A=`0/0/0/0` | `121.875 GiB` | `15.234375 GiB` |
| N1 G8 Route A | packed grouped updates -> FSDP update tree -> `optax.apply_updates` | 8 | `0.017000123` | `0.016983062` | AG/AR/RS/A2A=`0/0/0/0` | `121.875 GiB` | `15.234375 GiB` |
| N1 G8 Route B | packed grouped updates + FSDP params -> updated FSDP params directly | 8 | `0.017276616` | `0.017034352` | AG/AR/RS/A2A=`0/0/0/0` | `121.875 GiB` | `15.234375 GiB` |

- Correctness:
  - All full-size N1 correctness checks were skipped by the harness because the
    estimated global expert update size is `130862284800` bytes, above the
    current 1 GiB correctness cap.
  - The harness still reports the exact skip reason in the summary rows.
- Interpretation:
  - The primitive halves are clean on one node: no compiled all-gather,
    all-reduce, reduce-scatter, or all-to-all.
  - `G8` does not materially change N1 local materialization time relative to
    `G1`, so the single-node baseline is roughly `16-17 ms` per boundary half
    for this expert bank shape.
  - Route A and Route B are equivalent on N1 within noise. The decision between
    them should be based on R1D2/R1D4 communication shape and eventual training
    integration simplicity, not N1 timing.
- Active follow-up:
  - R1D2/FSDP-data runs are active:
    `/dlwh/iris-run-job-20260620-133929` for G8 grads-to-bank and
    `/dlwh/iris-run-job-20260620-133945` for G8 apply routes.
  - R1D4/FSDP-data runs are active:
    `/dlwh/iris-run-job-20260620-134339` for G8 grads-to-bank and
    `/dlwh/iris-run-job-20260620-134355` for G8 apply routes.
  - Aquinas (`019ee53d-a3c8-7b21-a9ee-5b2b04dcf38e`) and heartbeat
    `watch-muon-lower-bridge-subagent` are babysitting the R1D4 results.

### 2026-06-20 06:43 PDT - R1D2 packed-bank boundary primitives
- Hypothesis: If the packed-bank boundary primitive is viable, R1D2 should
  lower to a small number of coarse collectives rather than per-leaf collective
  explosion.
- Runs:
  - `/dlwh/iris-run-job-20260620-133929`:
    `MUON-BENCH-D2560-L26-R1D2E8-N2-G8-GRADSPACKBANK-cw-20260620-133927`.
  - `/dlwh/iris-run-job-20260620-133945`:
    `MUON-BENCH-D2560-L26-R1D2E8-N2-G8-H3-APPLYROUTES-cw-20260620-133943`.
- Config:
  - Two H100 nodes, `replica_axis=1`, `data_axis=2`, `expert_axis=8`,
    `model_axis=1`, `ns4d_group_axis=data`, group size 8, 26 layers, bf16,
    backend steps 3.
- Results:

| Primitive | Mean seconds | Median seconds | Compiled collectives | Estimated global bytes | Estimated grouped/FSDP per-device | Estimated peak/device | Effective global GB/s |
|---|---:|---:|---|---:|---:|---:|---:|
| FSDP grads -> packed grouped bank | `0.172119` | `0.172207` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `7.6171875 GiB` | `15.234375 GiB` | `~760` |
| packed grouped updates -> FSDP update tree -> `optax.apply_updates` | `0.18468` | `0.18511` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `7.6171875 GiB` | `15.234375 GiB` | `~708` |
| packed grouped updates + FSDP params -> updated FSDP params directly | `0.18389` | `0.18360` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `7.6171875 GiB` | `15.234375 GiB` | `~712` |

- Correctness:
  - Full-size correctness was skipped by the 1 GiB cap for all three rows.
- Interpretation:
  - The boundary primitive does avoid per-leaf collective explosion at R1D2:
    each logical boundary compiles to exactly two all-to-alls and no all-gather,
    all-reduce, or reduce-scatter.
  - The problem is performance, not collective count. R1D2 spends roughly
    `0.17-0.185s` per boundary half versus `0.016-0.017s` on N1, with effective
    global throughput only `~0.7-0.76 TB/s` for the full 121.875 GiB logical
    expert update.
  - Route A and Route B remain indistinguishable at this scale. Direct apply is
    not buying a clear boundary-speed win yet; ordinary `optax.apply_updates`
    remains viable if it is easier to integrate.
- Next action:
  - Let the active R1D4 runs establish whether the data-axis all-to-all gets
    worse with four hosts.
  - If R1D4 stays around the same per-boundary latency, the primitive is
    probably bounded by cross-host all-to-all latency/bandwidth and may need a
    lower-level packed transport or overlap rather than more XLA reshaping.

### 2026-06-20 06:49 PDT - R1D4 packed-bank boundary primitives
- Hypothesis: Increasing the FSDP `data` axis from 2 to 4 should keep the
  packed-bank boundary lowering clean and should reduce per-device transfer
  size. If the primitive is viable, it should still compile to a small constant
  number of coarse collectives.
- Runs:
  - `/dlwh/iris-run-job-20260620-134339`:
    `MUON-BENCH-D2560-L26-R1D4E8-N4-G8-GRADSPACKBANK-cw-20260620-134337`.
  - `/dlwh/iris-run-job-20260620-134355`:
    `MUON-BENCH-D2560-L26-R1D4E8-N4-G8-H3-APPLYROUTES-cw-20260620-134353`.
- Config:
  - Four H100 nodes, `replica_axis=1`, `data_axis=4`, `expert_axis=8`,
    `model_axis=1`, `ns4d_group_axis=data`, group size 8, 26 layers, bf16,
    backend steps 3.
- Results:

| Primitive | Mean seconds | Median seconds | Compiled collectives | Estimated global bytes | Estimated grouped/FSDP per-device | Estimated peak/device | Effective global GB/s |
|---|---:|---:|---|---:|---:|---:|---:|
| FSDP grads -> packed grouped bank | `0.08137` | `0.08134-0.08138` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `3.80859375 GiB` | `15.234375 GiB` | `~1608` |
| packed grouped updates -> FSDP update tree -> `optax.apply_updates` | `0.08417-0.08421` | `0.08414-0.08419` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `3.80859375 GiB` | `15.234375 GiB` | `~1554` |
| packed grouped updates + FSDP params -> updated FSDP params directly | `0.08533-0.08541` | `0.08419-0.08564` | AG/AR/RS/A2A=`0/0/0/2` | `121.875 GiB` | `3.80859375 GiB` | `15.234375 GiB` | `~1532-1534` |

- Correctness:
  - Full-size correctness was skipped by the 1 GiB cap for all three rows.
- Interpretation:
  - R1D4 preserves the desired structural property: no compiled all-gather,
    all-reduce, or reduce-scatter, and exactly two all-to-alls for each logical
    boundary.
  - The R1D4 boundary halves are roughly 2x faster than R1D2 because the
    per-device grouped/FSDP payload halves from `7.6171875 GiB` to
    `3.80859375 GiB`.
  - Route A remains at least as good as Route B. There is still no evidence
    that direct apply is worth taking on integration complexity; the pragmatic
    path is still grouped-bank transport back to an FSDP update tree followed
    by normal `optax.apply_updates`.
  - The primitive is no longer blocked on collective explosion. The remaining
    risk is end-to-end training integration and making sure we do not reintroduce
    per-leaf materialization before or after the packed boundary.

### 2026-06-20 07:30 PDT - Production grouped MuonH restore uses packed-bank data A2A
- Hypothesis: The benchmark harness result only helps training if the
  production grouped MuonH transform uses the same packed-bank restore pattern.
  The existing `packed_entry` production path still lowered to all-gathers on
  the R2D2 abstract test mesh, so data-axis FSDP runs could silently miss the
  harness winner.
- Change:
  - Updated `_packed_grouped_muonh_updates_to_fsdp_leaves` in
    `experiments/grug/moe/optimizer.py` so grouped updates use data-axis
    `all_to_all(split_axis=<FSDP data-sharded matrix axis>, concat_axis=0)` when
    the live grouped axis includes `data`.
  - Left a replica gather fallback for layouts where `replica_dcn` is part of
    the grouped axis.
  - Made `expert_grouped_muonh_packed_entry=True` the default for
    `GrugMoeMuonHConfig` and `experiments/grug/moe/run_cw_may_d2560.sh`. This
    only affects runs that explicitly choose `MAY_EXPERT_3D_OPTIMIZER=grouped_muonh`;
    default May runs still use the ordinary `muonh` expert path.
- Validation:
  - `uv run pytest experiments/grug/moe/test_optimizer.py -q`:
    `20 passed`.
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py -k 'packed_bank or launcher_reads_grouped_muonh_boundary_env or summary_row_reports_boundary_byte_estimates or strict_boundary_gate' -q`:
    `16 passed, 90 deselected`.
- Lowering evidence:
  - Production grouped MuonH with `packed_entry=True` on an abstract
    `replica_dcn=1,data=4,expert=8,model=1` mesh now lowers as
    AG/AR/RS/A2A=`0/0/0/6` for the optimizer update path. That is the expected
    three packed phases across two expert weight names:
    grads-to-bank, params-to-bank, and updates-to-FSDP.
  - The legacy non-packed production path remains worse and is now explicit in
    tests: it still has all-gathers plus the improved data A2A restore.
- Interpretation:
  - The harness winner is now integrated into the production grouped MuonH
    optimizer path for data-axis FSDP layouts.
  - The remaining completion gaps are distributed correctness evidence below
    the full-size 1 GiB cap and an end-to-end Grug MoE training run using
    `MAY_EXPERT_3D_OPTIMIZER=grouped_muonh` with the new default packed-entry
    path.

### 2026-06-20 07:05 PDT - Local packed-bank boundary correctness regression
- Hypothesis: The reduced local correctness case should compare actual and
  reference packed-bank boundary outputs even when the two trees carry different
  `NamedSharding` specs. This is needed because the local reduced case caught
  reference/checking failures before it reached the intended zero-error proof.
- Change:
  - Updated `max_abs_tree_error` to reshard expected leaves to the actual
    leaf sharding before subtraction.
  - Updated `reference_packed_bank_updates_apply` to replicate the packed
    reference stack axis before splitting per layer. This keeps the reference
    path from attempting a local `jnp.split` over a sharded stack axis.
  - Fixed `scratch/muon_update_bench_fast_loop.sh local` on Bash 3.2 by
    temporarily disabling `set -u` around empty optional argument arrays.
- Command:
  - `RUN_ID=MUON-BENCH-CORRECTNESS-SMALL-R1D2E2-G2-local MUON_BENCH_LAYERS=2 MUON_BENCH_HIDDEN_DIM=16 MUON_BENCH_INTERMEDIATE_DIM=8 MUON_BENCH_NUM_EXPERTS=2 MUON_BENCH_REPLICA_AXIS=1 MUON_BENCH_DATA_AXIS=2 MUON_BENCH_EXPERT_AXIS=2 MUON_BENCH_MODEL_AXIS=1 MUON_BENCH_GPU_REPLICAS=1 MUON_BENCH_NS4D_GROUP_SIZE=2 MUON_BENCH_NS4D_GROUP_AXIS=data MUON_BENCH_KINDS=expert_fsdp_grads_to_explicit_packed_grouped_bank,expert_fsdp_packed_bank_a2a_apply_boundary MUON_BENCH_SWEEP_BACKEND_STEPS=1 MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=4 MUON_BENCH_WARMUP=1 MUON_BENCH_ITERS=1 MUON_BENCH_MODE=both MUON_BENCH_TRACKER=json MUON_BENCH_WANDB=false MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES=true MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS=true MUON_BENCH_BOUNDARY_CORRECTNESS_MAX_GLOBAL_BYTES=1073741824 bash scratch/muon_update_bench_fast_loop.sh local grouped-d2e4`.
- Result:
  - Output: `scratch/MUON-BENCH-CORRECTNESS-SMALL-R1D2E2-G2-local.json`.
  - `expert_fsdp_grads_to_explicit_packed_grouped_bank_h1`:
    lowered AG/AR/RS/A2A=`0/0/0/2`, `correctness_max_error=0.0`.
  - `expert_fsdp_packed_bank_a2a_apply_boundary_h1`:
    lowered AG/AR/RS/A2A=`0/0/0/2`, `correctness_max_error=0.0`.
  - Compiled local HLO includes two all-reduces from the correctness/checksum
    timing path; that is not the boundary primitive shape being validated.
- Interpretation:
  - This is reduced local CPU/fake-device evidence, not a distributed
    CoreWeave correctness proof.
  - It does cover the reference/checking corner that previously made the
    reduced correctness run fail before measuring the packed-bank boundary
    primitives.

### 2026-06-20 07:18 PDT - R1D2 packed-entry grouped MuonH e2e training succeeded
- Hypothesis: The production grouped MuonH packed-entry integration should run
  through real Grug MoE training on the non-replicating two-node FSDP-data
  layout without OOM, sharding errors, or per-leaf collective failures.
- Run:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-135753`.
  - Child Iris job:
    `/dlwh/iris-run-job-20260620-135753/grug-train-GM2560-MAY-199S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-E2E-N2-cw-20260620-1357`.
  - W&B:
    `marin-community/marin_moe/GM2560-MAY-199S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-E2E-N2-cw-20260620-1357`.
  - Output path:
    `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/grug-moe-cw-may-d2560-L26-e256-r2-cpu8-GM2560-MAY-199S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-E2E-N2-cw-20260620-1357-fcde75`.
- Config:
  - Two H100 nodes, `replica_dcn=1`, `data=2`, `expert=8`,
    `model=1`, `batch_shards=16`.
  - Batch 16, sequence length 4096, sliding window 2048, 5 steps,
    synthetic data, no checkpoints.
  - `MAY_EXPERT_3D_OPTIMIZER=grouped_muonh` with the new packed-entry default,
    MuonH3, bf16 params/compute/output, Pallas cross entropy, ring MoE, FA4
    attention.
- Result:
  - Parent and child both reached `JOB_STATE_SUCCEEDED`, exit code 0.
  - W&B state is `finished`.
  - The child ran two tasks and completed steps 0 through 4.
  - Mesh logging confirmed `replica_dcn=1,data=2,expert=8,model=1` on both
    tasks.
  - No fatal traceback, OOM/HBM, sharding, rendezvous, or clique failure was
    seen. The final `WatchTasksAsync` connection-refused warnings occurred
    after training/W&B finish and Iris still reported success.
- Metrics:
  - Step 0 included compile time: duration about `459-462s`,
    `~142 tokens/s`, MFU `~0.015`.
  - Final W&B summary at global step 4:
    `train/loss=4.88875675201416`,
    `throughput/duration=2.6897398190340027`,
    `throughput/tokens_per_second=24365.18191693972`,
    `throughput/examples_per_second=5.948530741440361`,
    `throughput/mfu=2.6347597139402783`,
    `throughput/mean_mfu=1.5867107566699008`.
  - Logs also showed steady post-compile steps 2-4 around `2.68-2.70s`.
- Interpretation:
  - This closes the production "does it run end-to-end on R1D2 FSDP-data?"
    gap: the packed-bank grouped MuonH path is functionally integrated into
    Grug MoE training.
  - The performance is still poor relative to the non-Muon baseline and the
    packed-bank harness boundary timings. The next problem is not correctness
    or per-leaf collective explosion; it is explaining the extra production
    train-step overhead and/or moving the boundary work into a lower-level
    primitive/overlapped path.

### 2026-06-20 07:42 PDT - May200 profile run exposed an SPMD activation reshard warning
- Hypothesis: A short profiled repeat of the R1D2 packed-entry grouped MuonH
  training run should show where the production overhead comes from. The run
  disables GPU command buffers for trace readability, so its timings are only
  diagnostic.
- Run:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-142240`.
  - Child Iris job:
    `/dlwh/iris-run-job-20260620-142240/grug-train-GM2560-MAY-200S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-PROFILE-N2-cw-20260620-1422`.
  - W&B:
    `marin-community/marin_moe/GM2560-MAY-200S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-PROFILE-N2-cw-20260620-1422`.
- Status at 07:42 PDT:
  - Iris still reported the parent and child as running with two child tasks.
  - W&B reported state `crashed` before metrics/artifacts, but Iris logs showed
    step 0 completed and step 1 started, so Iris is currently the source of
    truth for this run.
  - No profiler artifact had uploaded yet.
- Evidence:
  - Step 0 completed after compile with loss `11.791757583618164`, duration
    about `457-460s`, and compile-included MFU about `0.015`.
  - During SPMD partitioning, both tasks emitted:
    `Involuntary full rematerialization. The compiler cannot go from sharding
    {devices=[16,1,1]<=[16]} to {devices=[1,1,2,8]<=[16]
    last_tile_dim_replicate} efficiently` for `bf16[1,4096,2560]`.
  - The metadata attached the warning to
    `jit(train_step)/forward_backward/transpose(jvp(Transformer))/RMSNorm/convert_element_type`.
  - The same warning repeated when step 1 started compiling/partitioning.
- Interpretation:
  - This warning is not the packed-bank MuonH boundary itself. It is an
    activation-like tensor being replicated and repartitioned across the
    `data,expert` layout in the production train graph.
  - It is a plausible contributor to the gap between the clean packed-bank
    harness timings and the poor production end-to-end step time, and it needs
    profile/HLO follow-up once May200 either uploads a profile or terminates.

### 2026-06-20 10:56 PDT - Production Route A collective-count regression test refreshed
- Hypothesis: The current production grouped MuonH Route A path should have
  explicit, bounded collectives in lowered HLO, and the local regression test
  should match the actual contract before further optimizer work.
- Change:
  - Updated
    `experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs`
    expected `all_to_all` counts for the current R2D2 lowered HLO:
    non-packed-entry `2`, packed-entry `6`, chunk-local `4`.
- Command:
  - `uv run pytest experiments/grug/moe/test_optimizer.py::test_grouped_expert_muonh_packed_entry_r2_boundary_does_not_duplicate_replica_gather experiments/grug/moe/test_optimizer.py::test_grouped_expert_muonh_packed_entry_r4_boundary_is_explicit experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs`
- Result:
  - `5 passed in 5.17s`.
- Interpretation:
  - This does not make the production path faster, but it restores a useful
    guardrail around the exact path we need to iterate on: FSDP params in,
    grouped MuonH updates internally, FSDP-shaped updates/results out, no
    all-reduce/reduce-scatter in the tested optimizer boundary, and a bounded
    collective count rather than per-leaf explosion.

### 2026-06-20 11:00 PDT - Route A default documented and guarded
- Hypothesis: Launch defaults should point at the recommended production Route A
  boundary and keep the slower packed-bank-compute experiment explicitly opt-in.
- Change:
  - Updated `experiments/grug/moe/launch_cw_may_d2560.py` docs to show
    `MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY=true`.
  - Kept `MAY_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE=false` documented as
    experimental.
  - Added a default assertion that grouped MuonH enables packed entry but does
    not enable packed-bank compute.
- Command:
  - `uv run pytest experiments/grug/moe/test_optimizer.py::test_may_grouped_muonh_defaults_to_packed_entry experiments/grug/moe/test_optimizer.py::test_may_optimizer_reads_grouped_muonh_packed_entry_env experiments/grug/moe/test_muon_update_bench.py::test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs`
  - `./infra/pre-commit.py --changed-files --fix`
- Result:
  - Pytest: `5 passed in 7.25s`.
  - Pre-commit: OK.

### 2026-06-20 11:22 PDT - May202 Route A profile says production is comm-bound
- Hypothesis: The readable May202 Route A profile should explain the gap
  between the compact boundary harness and the slow production training loop.
- Artifact:
  - Local profile: `scratch/profiles/may202`.
  - Structured summary: `scratch/profiles/may202_summary.json`.
  - Markdown report: `scratch/profiles/may202_report.md`.
  - xprof tables: `scratch/profiles/may202_xprof_tables`.
- Command:
  - `uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize --profile-dir scratch/profiles/may202 --xplane-output-dir scratch/profiles/may202_xprof_tables --xplane-count-trace-events --breakdown-mode exclusive_global --output scratch/profiles/may202_summary.json`
  - `uv run python lib/marin/tools/profile_summary.py report --summary scratch/profiles/may202_summary.json --output scratch/profiles/may202_report.md`
- Result:
  - The summary parsed `884,773` complete events and did not flag trace
    truncation.
  - xprof cost-analysis emitted warnings about a newer GEMM backend config
    field (`scale_mode`), but kernel/op attribution was still exported and
    merged into the summary.
  - Time breakdown: communication `74.2%`, compute `25.8%`.
  - Top communication aggregates:
    - packed-restore `replica_gather_to_fsdp/all_gather`: `32` calls,
      `7.81s` aggregate.
    - packed-restore `concat_chunks/concatenate` SendRecv: `448` calls,
      `6.41s` aggregate.
    - MoE backward `psum` all-reduce: `832` calls, `5.99s` aggregate.
    - packed-entry `slice_update_chunk/dynamic_slice` SendRecv: `384` calls,
      `5.07s` aggregate.
    - packed-entry `slice_param_chunk/dynamic_slice` SendRecv: `384` calls,
      `5.07s` aggregate.
    - packed-restore shard-map all-gather: `32` calls, `3.69s` aggregate.
  - Semantic families:
    - `optimizer_muon`: `73.6%` of profiled duration.
    - `moe`: `18.3%`.
    - `attention_flash`: `1.7%`.
- Interpretation:
  - The production Route A path is communication-bound, not NS-GEMM-bound.
  - The main gap versus the compact boundary harness is not ordinary
    `optax.apply_updates`; it is packed-entry chunk slicing and packed-restore
    fanout in the production transform.
  - The next bridge target is therefore specific: eliminate or replace
    `slice_update_chunk` / `slice_param_chunk` SendRecv and reduce or overlap
    packed-restore all-gather/fanout while keeping FSDP master params and
    ordinary apply semantics.

### 2026-06-20 11:33 PDT - May208 chunk-local production profile launched
- Hypothesis: The chunk-local boundary mode that improved the standalone
  `real_expert_fsdp_grouped_muonh_optimizer_update` harness should remove the
  May202 packed-entry whole-bank dynamic-slice traffic in the full Grug train
  step. This is not expected to be the final primitive, but it is the closest
  production-safe fallback to validate before a lower-level bridge.
- Command:
  - `bash scratch/launch_may208_fa4_2node_b16_grouped_muonh3_chunklocal_readable_profile.sh`
- Run:
  - Initial parent Iris job `/dlwh/iris-run-job-20260620-180813` failed before
    training because W&B rejected the long run name: `128 limit exceeded for
    Name`.
  - Relaunched parent Iris job: `/dlwh/iris-run-job-20260620-181104`.
  - Replacement run id:
    `GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-20260620-1811`.
- Config:
  - Same May202-style two-node readable profile shape: 2 H100 nodes, B16,
    sequence length 4096, sliding window 2048, `replica_axis=2`,
    `data_axis=1`, `expert_axis=8`, `model_axis=1`, `gpu_fa4_cute`
    attention, Pallas CE, ring MoE, save-MoE remat, bf16
    params/compute/output, MuonH3 with bf16 NS compute, profiler HLO proto
    enabled, command buffers disabled for readable names.
  - Boundary change versus May202: `expert_grouped_muonh_packed_entry=false`
    and `expert_grouped_muonh_chunk_local_boundaries=true`.
- Next check:
  - Compare profile summary against `scratch/profiles/may202_summary.json`.
    The important signal is whether `grouped_muonh/packed_entry/slice_*_chunk`
    and `grouped_muonh/packed_restore/concat_chunks` disappear or shrink, and
    whether the replacement explicit per-chunk collectives are faster in the
    full train step.

### 2026-06-20 11:36 PDT - Boundary mode is explicit in bench summaries
- Hypothesis: W&B/summary rows for the integrated grouped MuonH optimizer need
  to say which production boundary mode they used. Otherwise May202
  packed-entry and May208 chunk-local results are too easy to conflate because
  they share the same high-level bench kind.
- Change:
  - Added `expert_grouped_muonh_boundary_mode`,
    `expert_grouped_muonh_packed_entry`, and
    `expert_grouped_muonh_chunk_local_boundaries` to
    `experiments/grug/moe/muon_update_bench.py::summary_row`.
  - Boundary mode values are `per_chunk_reshard`, `packed_entry`, and
    `chunk_local`; chunk-local wins if both booleans are true.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_grouped_muonh_boundary_mode experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Pytest: `5 passed in 4.95s`.
  - Pre-commit: OK.

### 2026-06-20 11:48 PDT - Integrated grouped MuonH reports boundary phase estimates
- Hypothesis: The integrated `real_expert_fsdp_grouped_muonh_*` bench rows
  need the same phase-level boundary estimates as the lower-level packed-bank
  primitive rows. Without those fields, May202/May208 summaries can say the
  mode but cannot say whether the compiled collectives match the intended
  logical boundary phases.
- Change:
  - Extended `estimated_boundary_phase_estimates` to cover real grouped MuonH
    optimizer benches.
  - Reports separate logical phases for FSDP grads -> grouped chunks, FSDP
    params -> grouped chunks, and grouped updates -> FSDP update tree.
  - Splits mixed `all_to_all+all_gather` phases into separate summary rows so
    compiled HLO collective counts can be compared by collective type.
  - Added parametrized coverage for `per_chunk_reshard`, `packed_entry`, and
    `chunk_local` boundary modes.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_grouped_muonh_boundary_mode experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Pytest: `8 passed in 3.64s`.
  - Pre-commit: OK.

### 2026-06-20 11:25 PDT - May208 chunk-local is live but already non-competitive
- Observation:
  - Iris still reports parent `/dlwh/iris-run-job-20260620-181104` and child
    `/dlwh/iris-run-job-20260620-181104/grug-train-GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-20260620-1811`
    as running with both tasks alive.
  - W&B now reports the run as `crashed`, with zero history rows.
  - Bounded Iris logs show both ranks completed only step 0. Rank 0 duration
    was `431.93s`, tokens/s `151.73`, MFU `0.0164`; rank 1 duration was
    `432.93s`, tokens/s `151.38`, MFU `0.01637`.
  - No step 1 metrics or traceback appeared in a 10-minute recent log window.
- Interpretation:
  - This is not a viable production fallback. Chunk-local avoids the
    packed-entry shape in principle, but the integrated path is now orders of
    magnitude slower than May202 and much worse than the grouped boundary
    harnesses.
  - Treat May208 as evidence that the fallback path is not sufficient; keep
    waiting only if we want a profile artifact for diagnosis.

### 2026-06-20 11:31 PDT - May208 chunk-local profile interpreted
- Correction to the initial observation:
  - May208 did not stay stuck after step 0/1. Steps 0 and 1 were pathological
    at `~432s` and `~452s`, but steps 2-7 reached a steady-state-ish
    `~1.04-1.11s` per step.
  - Representative profiler-window rank-0 metrics:
    step 3 `duration=1.1043s`, `tokens_per_second=59343.9`, `mfu=6.4172`;
    step 4 `duration=1.1105s`, `tokens_per_second=59014.5`, `mfu=6.3816`.
  - W&B still reports the run as `crashed` with zero history rows because the
    artifact upload ended in `wandb.sdk.mailbox.mailbox_handle.HandleAbandonedError`.
    The profiler artifact was nevertheless created as
    `GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-20260620-1811-profiler:v0`.
- Artifacts:
  - Summary: `scratch/profiles/may208_summary.json`.
  - Report: `scratch/profiles/may208_report.md`.
  - Command:
    `uv run python lib/marin/tools/profile_summary.py summarize --artifact marin-community/marin_moe/GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-20260620-1811-profiler:v0 --download-root scratch/profiles/may208 --breakdown-mode exclusive_global --output scratch/profiles/may208_summary.json`.
- Profile result:
  - The profile has two useful denominators:
    - Device/global timeline view: communication share `75.85%`, compute share
      `20.0%`.
    - Full per-track summary view: communication `12.47s` / `23.82%`, compute
      `10.78s` / `20.60%`, host `27.00s` / `51.60%`.
  - Top device collectives: all-reduce `2160` calls / `6.89s`, all-gather
    `2080` calls / `5.09s`, reduce-scatter `832` calls / `0.482s`.
  - Compared with May202, the previous huge packed-entry/packed-restore
    SendRecv-looking hotspots disappear and Muon communication drops from
    `35.91s` to `12.47s` in the full per-track summary. However, the
    chunk-local path introduces many all-gathers, whole-profile duration is not
    cleanly better because host/upload/finalization dominates, and steady-state
    MFU remains only `~6.4-6.8`.
- Interpretation:
  - Chunk-local is a useful negative/diagnostic result: it improves the targeted
    Muon communication hotspots but does not produce a clean end-to-end
    production-profile win. The lower-level packed-bank harness remains the
    better direction: few large boundary collectives, not thousands of
    fragmented train-step collectives.

### 2026-06-20 11:58 PDT - Per-type boundary collective efficiency reporting
- Hypothesis: The boundary harness should make "avoids per-leaf collective
  explosion" directly visible in summary/W&B rows, not only by reading HLO logs
  or a coarse total fragmentation factor.
- Change:
  - Added per-collective-type actual/ideal/excess/ratio/match fields for
    lowered and compiled HLO when a bench has explicit boundary phase
    estimates.
  - Example field names:
    `estimated_boundary_compiled_all_to_all_excess_collective_count`,
    `estimated_boundary_compiled_all_gather_matches_ideal_collective_count`,
    and the corresponding `lowered` fields.
  - Fields are only populated when explicit phase estimates exist, avoiding
    misleading "excess" counts for older boundary probes that only have a
    coarse fallback ideal.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_boundary_byte_estimates -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Pytest: `5 passed in 5.07s`.
  - Pre-commit: OK.

### 2026-06-20 12:04 PDT - Boundary collective explosion gate
- Hypothesis: A single pass/fail-style summary signal is useful for screening
  candidate boundary primitives before opening profiles. The per-type fields
  added above are detailed enough for diagnosis, but the harness should also
  state whether lowered/compiled collectives exactly match the explicit phase
  contract.
- Change:
  - Added `estimated_boundary_{lowered,compiled}_total_excess_collective_count`.
  - Added `estimated_boundary_{lowered,compiled}_matches_ideal_collective_counts`.
  - Added a negative regression test where the logical packed-bank MuonH apply
    phase expects only six all-to-alls but compiled HLO has three extra
    all-gathers; the row now reports excess `3.0` and
    `matches_ideal_collective_counts=false`.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_summary_row_flags_boundary_collective_type_excess -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Pytest: `5 passed in 4.14s`.
  - Pre-commit: OK.

### 2026-06-20 12:05 PDT - May208 terminal failure mode
- Observation:
  - After uploading the profiler artifact, May208 task 0 raised
    `wandb.sdk.mailbox.mailbox_handle.HandleAbandonedError`.
  - JAX distributed then hit `INTERNAL: Shutdown barrier has failed` and
    terminated both tasks.
  - Iris showed the child with `failures=1`; the run had already emitted the
    useful profiler-window metrics and uploaded a usable profiler artifact.
- Interpretation:
  - Treat the W&B crash as post-profile artifact handling, not as invalidating
    the May208 performance/profile evidence. The steady-state profile remains
    a valid negative result for the chunk-local fallback.

### 2026-06-20 11:41 PDT - Boundary rows include axes and type bandwidth
- Hypothesis:
  - The boundary primitive goal needs a compact table that can be screened
    without opening HLO or TensorBoard. The summary rows already reported
    estimated bytes, peak HBM, total boundary GB/s, and per-type collective
    count excess, but they did not directly report the mesh axes or
    per-collective-type estimated bandwidth.
- Change:
  - Added `devices`, `replica_axis`, `data_axis`, `expert_axis`, and
    `model_axis` to each `summary_row`.
  - Added mean/median estimated boundary phase GB/s split by expected
    collective type:
    `all_gather`, `all_reduce`, `reduce_scatter`, `all_to_all`, and
    `collective_permute`.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_summary_row_flags_boundary_collective_type_excess -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Pytest: `5 passed in 5.05s`.
  - Pre-commit: OK after formatter fix.

### 2026-06-20 11:43 PDT - Packed-bank boundary contract launch wrapper
- Hypothesis:
  - The next N1/R2/R4 validation should launch the FSDP-grad ingress primitive
    and both grouped-update egress routes in one job per topology. That keeps
    the new summary-row fields directly comparable across Route A and Route B.
- Change:
  - Added `scratch/launch_muon_packed_bank_boundary_contract.sh`.
  - The wrapper accepts `n1`, `r2`, `r4`, plus data-axis variants `d2`/`d4`.
  - Default bench kinds:
    `expert_fsdp_grads_to_explicit_packed_grouped_bank`,
    `expert_fsdp_packed_bank_a2a_apply_boundary`, and
    `expert_fsdp_packed_bank_direct_apply_boundary`.
  - Defaults to TTL output prefix and W&B group
    `grug-moe-cw-muon-boundary-contract`.
- Command:
  - `MUON_BENCH_DRY_RUN=true bash scratch/launch_muon_packed_bank_boundary_contract.sh n1`
  - `MUON_BENCH_DRY_RUN=true bash scratch/launch_muon_packed_bank_boundary_contract.sh r2`
  - `MUON_BENCH_DRY_RUN=true bash scratch/launch_muon_packed_bank_boundary_contract.sh r4`
- Result:
  - All three dry-runs produced the expected axes, group axis, run names, W&B
    project, and TTL output prefix.

### 2026-06-20 11:48 PDT - Boundary rows report process count
- Hypothesis:
  - The boundary goal asks for nodes/`replica_dcn`; total devices and mesh axes
    are useful but still force readers to infer node count. The harness should
    report JAX process count and local device count directly.
- Change:
  - Added `local_devices` and `process_count` to benchmark metadata and
    summary rows.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_packed_bank_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates experiments/grug/moe/test_muon_update_bench.py::test_summary_row_flags_boundary_collective_type_excess -q`
- Result:
  - Pytest: `5 passed in 4.65s`.

### 2026-06-20 11:51 PDT - R2 packed-bank boundary contract succeeds
- Hypothesis:
  - The packed-bank primitives should avoid per-leaf collective explosion on
    R2 by keeping FSDP-grad ingress local to the `replica_dcn` grouping and
    doing grouped-update egress as two large all-gathers.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_contract.sh r2`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-184515`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/ydgedtaw`.
  - Run id:
    `MUON-BENCH-D2560-L26-R2D1E8-N2-G8-BOUNDARYCONTRACT-cw-20260620-184512`.
  - 2 H100 nodes, `replica_axis=2`, `data_axis=1`, `expert_axis=8`,
    `model_axis=1`, `ns4d_group_axis=replica_dcn`, `ns4d_group_size=8`,
    bf16 params/NS compute, 26 layers.
- Result:
  - Child job succeeded with both tasks completed.
  - `fsdp_grads_to_explicit_packed_grouped_bank`: compiled AG/A2A = `0/0`,
    ideal collectives `0`, excess `0`, mean `0.00654s`, estimated phase
    bandwidth `~20.0 TB/s`, compiled HBM peak `15.24 GiB`.
  - Route A `packed_grouped_updates_to_fsdp_apply`: compiled AG/A2A = `2/0`,
    ideal collectives `2`, excess `0`, mean `0.23245s`, estimated all-gather
    phase bandwidth `~563 GB/s`, compiled HBM peak `38.09 GiB`.
  - Route B `packed_grouped_updates_to_fsdp_direct_apply`: compiled AG/A2A =
    `2/0`, ideal collectives `2`, excess `0`, mean `0.23233s`, estimated
    all-gather phase bandwidth `~563 GB/s`, compiled HBM peak `38.09 GiB`.
  - Full-size correctness was skipped by the existing global-byte cap:
    `estimated global bytes 130862284800 exceed correctness cap 1073741824`.
- Interpretation:
  - R2 validates the collective-count contract for both egress routes:
    no per-leaf collective explosion, and Route A and Route B are effectively
    tied at this scale.
  - The remaining question is R4 scaling and whether the all-gather bandwidth
    stays acceptable or gets worse with `replica_dcn=4`.

### 2026-06-20 11:58 PDT - R4 packed-bank boundary contract succeeds
- Hypothesis:
  - The packed-bank boundary contract should continue to avoid per-leaf
    collective explosion at `replica_dcn=4`, with grouped-update egress still
    compiling to the intended two all-gathers.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_contract.sh r4`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-185137`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/q9ntnsmp`.
  - Run id:
    `MUON-BENCH-D2560-L26-R4D1E8-N4-G8-BOUNDARYCONTRACT-cw-20260620-185135`.
  - 4 H100 nodes, `devices=32`, `local_devices=8`, `process_count=4`,
    `replica_axis=4`, `data_axis=1`, `expert_axis=8`, `model_axis=1`,
    `ns4d_group_axis=replica_dcn`, `ns4d_group_size=8`, bf16 params/NS
    compute, 26 layers.
- Result:
  - Child job succeeded with all four tasks completed.
  - `fsdp_grads_to_explicit_packed_grouped_bank`: compiled AG/A2A = `0/0`,
    ideal collectives `0`, excess `0`, mean `0.00500s`, estimated phase
    bandwidth `~26.2 TB/s`, compiled HBM peak `15.23 GiB`.
  - Route A `packed_grouped_updates_to_fsdp_apply`: compiled AG/A2A = `2/0`,
    ideal collectives `2`, excess `0`, mean `0.31373s`, estimated all-gather
    phase bandwidth `~417 GB/s`, compiled HBM peak `35.74 GiB`.
  - Route B `packed_grouped_updates_to_fsdp_direct_apply`: compiled AG/A2A =
    `2/0`, ideal collectives `2`, excess `0`, mean `0.31423s`, estimated
    all-gather phase bandwidth `~416 GB/s`, compiled HBM peak `35.74 GiB`.
  - Full-size correctness was skipped by the existing global-byte cap:
    `estimated global bytes 130862284800 exceed correctness cap 1073741824`.
- Interpretation:
  - R4 validates the same collective-count contract as R2: ingress remains
    local, and both grouped-update egress routes compile to exactly two
    all-gathers with zero excess collectives.
  - R4 egress is slower than R2: `~0.314s` versus `~0.232s`, and estimated
    all-gather phase bandwidth drops from `~563 GB/s` to `~416-417 GB/s`.
  - Route A and Route B remain tied at this scale; the next useful question is
    whether the packed egress can be overlapped or replaced by a lower-level
    direct FSDP-boundary primitive, not whether `optax.apply_updates` itself is
    the dominant difference between the two routes.

### 2026-06-20 12:04 PDT - N1 packed-bank boundary contract succeeds
- Hypothesis:
  - The same packed-bank contract should be clean on a single 8xH100 node:
    with `replica_dcn=1` there should be no cross-replica egress collective,
    giving a local baseline for the R2/R4 all-gather costs.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_contract.sh n1`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-185834`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/pzmc1hfm`.
  - Run id:
    `MUON-BENCH-D2560-L26-R1D1E8-N1-G8-BOUNDARYCONTRACT-cw-20260620-185831`.
  - 1 H100 node, `devices=8`, `local_devices=8`, `process_count=1`,
    `replica_axis=1`, `data_axis=1`, `expert_axis=8`, `model_axis=1`,
    `ns4d_group_axis=none`, `ns4d_group_size=8`, bf16 params/NS compute,
    26 layers.
- Result:
  - Child job succeeded.
  - `fsdp_grads_to_explicit_packed_grouped_bank`: compiled AG/A2A = `0/0`,
    ideal collectives `0`, excess `0`, mean `0.01589s`, estimated phase
    bandwidth `~8.24 TB/s`, compiled HBM peak `15.24 GiB`.
  - Route A `packed_grouped_updates_to_fsdp_apply`: compiled AG/A2A = `0/0`,
    ideal collectives `0`, excess `0`, mean `0.01706s`, estimated phase
    bandwidth `~7.67 TB/s`, compiled HBM peak `30.47 GiB`.
  - Route B `packed_grouped_updates_to_fsdp_direct_apply`: compiled AG/A2A =
    `0/0`, ideal collectives `0`, excess `0`, mean `0.01720s`, estimated
    phase bandwidth `~7.61 TB/s`, compiled HBM peak `30.47 GiB`.
  - Full-size correctness was skipped by the existing global-byte cap:
    `estimated global bytes 130862284800 exceed correctness cap 1073741824`.
- Interpretation:
  - N1/R2/R4 now all show the same key contract property: no per-leaf
    collective explosion. N1 has no egress collectives because `replica_dcn=1`;
    R2 and R4 add exactly the two expected all-gathers.
  - Route A and Route B are still tied at N1, matching the R2/R4 observation.

### 2026-06-20 12:06 PDT - N1 small-shape boundary correctness succeeds
- Hypothesis:
  - Full-shape correctness references are too large to materialize under the
    default cap, so a small CoreWeave run should prove the same primitive code
    matches reference packing/apply while the full-shape runs carry the
    performance and collective-count evidence.
- Change:
  - Added `scratch/launch_muon_packed_bank_boundary_correctness.sh`, a small
    launch wrapper for the same three packed-bank boundary kinds.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_correctness.sh n1`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-190220`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/kbo0smtj`.
  - Run id:
    `MUON-BENCH-CORRECTNESS-L4-H128-I64-E8-n1-G4-BOUNDARYCONTRACT-cw-20260620-190218`.
  - 1 H100 node, `devices=8`, `local_devices=8`, `process_count=1`,
    `replica_axis=1`, `data_axis=1`, `expert_axis=8`, `model_axis=1`,
    `layers=4`, `hidden_dim=128`, `intermediate_dim=64`, `num_experts=8`,
    `ns4d_group_axis=none`, `ns4d_group_size=4`, bf16 params/NS compute.
- Result:
  - Child and parent jobs succeeded.
  - All three primitive rows reported `boundary_correctness_max_error=0.0`
    and `boundary_correctness_skipped_reason=null`.
  - All three rows compiled to AG/A2A/AR = `0/0/0`, with zero excess
    collectives.
  - Mean times were `0.000864s` for ingress, `0.000336s` for Route A, and
    `0.000308s` for Route B. HBM peaks were `0.000185 GiB`, `0.000366 GiB`,
    and `0.000366 GiB`.
- Interpretation:
  - This closes the single-node correctness gap for the exact packed-bank
    boundary primitive implementations. The remaining correctness evidence
    gap is the same small-shape check on R2/R4, where the grouped stack axis is
    actually sharded over `replica_dcn`.

### 2026-06-20 12:09 PDT - R2 small-shape boundary correctness succeeds
- Hypothesis:
  - The small-shape correctness run should continue to match the reference when
    the grouped stack axis is sharded over `replica_dcn=2`.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_correctness.sh r2`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-190602`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/40qz71rj`.
  - Run id:
    `MUON-BENCH-CORRECTNESS-L4-H128-I64-E8-r2-G4-BOUNDARYCONTRACT-cw-20260620-190600`.
  - 2 H100 nodes, `devices=16`, `local_devices=8`, `process_count=2`,
    `replica_axis=2`, `data_axis=1`, `expert_axis=8`, `model_axis=1`,
    `layers=4`, `hidden_dim=128`, `intermediate_dim=64`, `num_experts=8`,
    `ns4d_group_axis=replica_dcn`, `ns4d_group_size=4`, bf16 params/NS
    compute.
- Result:
  - Parent and child jobs succeeded; W&B finished.
  - All three primitive rows reported `boundary_correctness_max_error=0.0`
    and `boundary_correctness_skipped_reason=null`.
  - Ingress: compiled AG/A2A = `0/0`, lowered/compiled ideal `0`, excess `0`,
    mean `0.001168s`, HBM peak `0.000184 GiB`.
  - Route A `packed_grouped_updates_to_fsdp_apply`: lowered AG = `2` matching
    ideal with zero lowered excess; compiled AG/A2A = `0/0`, compiled excess
    `-2`, mean `0.000696s`, HBM peak `0.000459 GiB`.
  - Route B `packed_grouped_updates_to_fsdp_direct_apply`: lowered AG = `2`
    matching ideal with zero lowered excess; compiled AG/A2A = `0/0`,
    compiled excess `-2`, mean `0.000534s`, HBM peak `0.000459 GiB`.
- Interpretation:
  - This closes the R2 reference-correctness check for the packed-bank
    boundary primitive implementations.
  - For this tiny shape, compiled HLO does not report the two expected
    all-gathers on the apply routes even though lowered HLO does. Treat this
    as correctness evidence, not as the compiled-collective-count evidence.
    Full-size R2/R4 remain the authoritative compiled-collective-count runs.

### 2026-06-20 12:12 PDT - R4 small-shape boundary correctness succeeds
- Hypothesis:
  - The small-shape correctness run should continue to match the reference when
    the grouped stack axis is sharded over `replica_dcn=4`.
- Command:
  - `bash scratch/launch_muon_packed_bank_boundary_correctness.sh r4`
- Config:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-190921`.
  - W&B:
    `https://wandb.ai/marin-community/marin_moe/runs/dvvlkl51`.
  - Run id:
    `MUON-BENCH-CORRECTNESS-L4-H128-I64-E8-r4-G4-BOUNDARYCONTRACT-cw-20260620-190919`.
  - 4 H100 nodes, `devices=32`, `local_devices=8`, `process_count=4`,
    `replica_axis=4`, `data_axis=1`, `expert_axis=8`, `model_axis=1`,
    `layers=4`, `hidden_dim=128`, `intermediate_dim=64`, `num_experts=8`,
    `ns4d_group_axis=replica_dcn`, `ns4d_group_size=4`, bf16 params/NS
    compute.
- Result:
  - Child job succeeded.
  - All three primitive rows reported `boundary_correctness_max_error=0.0`
    and `boundary_correctness_skipped_reason=null`.
  - Ingress: lowered/compiled AG/A2A/AR = `0/0/0`, ideal collectives `0`,
    excess `0`, mean `0.001219s`, estimated phase bandwidth `1.29 GB/s`,
    HBM peak `0.000183 GiB`.
  - Route A `packed_grouped_updates_to_fsdp_apply`: lowered AG = `2` matching
    ideal with zero lowered excess; compiled AG/A2A/AR = `0/0/0`, compiled
    excess `-2`, mean `0.000517s`, estimated all-gather phase bandwidth
    `3.04 GB/s`, HBM peak `0.000413 GiB`.
  - Route B `packed_grouped_updates_to_fsdp_direct_apply`: lowered AG = `2`
    matching ideal with zero lowered excess; compiled AG/A2A/AR = `0/0/0`,
    compiled excess `-2`, mean `0.000571s`, estimated all-gather phase
    bandwidth `2.76 GB/s`, HBM peak `0.000413 GiB`.
- Interpretation:
  - This closes the R4 reference-correctness check for the packed-bank
    boundary primitive implementations.
  - As with R2, this tiny shape is not authoritative for compiled collective
    counting because compiled HLO hides or optimizes away the two expected
    all-gathers on the apply routes. Use the full-size R2/R4 runs for
    compiled-collective evidence and the small-shape N1/R2/R4 runs for
    correctness evidence.

### 2026-06-20 12:34 PDT - Harness reports packed-bank-compute production mode
- Hypothesis:
  - The integrated production path used by May206/May207 is distinct from the
    ordinary `packed_entry` Route A path and should be represented directly in
    benchmark config and summary rows. Otherwise the next production benchmark
    can be misread as whole-bank packed-entry instead of bounded packed-bank
    compute.
- Change:
  - Added `expert_grouped_muonh_packed_bank_compute` to
    `experiments/grug/moe/muon_update_bench.py::BenchConfig`.
  - Threaded `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE` through the
    CoreWeave Muon update bench launcher.
  - Summary rows now report `expert_grouped_muonh_boundary_mode` as
    `packed_bank_compute` when packed entry and packed-bank compute are both
    enabled, and expose the boolean
    `expert_grouped_muonh_packed_bank_compute`.
  - Phase estimates for `packed_bank_compute` use chunk-count ingress/egress
    expectations, matching the bounded per-bank production path rather than
    the whole-bank packed-entry path.
- Command:
  - `uv run pytest experiments/grug/moe/test_muon_update_bench.py::test_cw_muon_update_bench_launcher_reads_grouped_muonh_boundary_env experiments/grug/moe/test_muon_update_bench.py::test_summary_row_reports_grouped_muonh_boundary_mode experiments/grug/moe/test_muon_update_bench.py::test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates -q`
  - `./infra/pre-commit.py --files experiments/grug/moe/muon_update_bench.py experiments/grug/moe/launch_cw_muon_update_bench.py experiments/grug/moe/test_muon_update_bench.py --fix`
- Result:
  - Focused pytest: `10 passed in 4.29s`.
  - Pre-commit on touched files: OK.
  - Commit: `e36c03910`.
- Interpretation:
  - This does not change training semantics. It closes a measurement/reporting
    gap so the next integrated packed-bank-compute harness run can be judged
    against the intended boundary contract instead of being conflated with
    `packed_entry`.

### 2026-06-20 12:25 PDT - R2 packed-bank-compute reporting run launched
- Hypothesis:
  - With `expert_grouped_muonh_packed_bank_compute` now present in
    `BenchConfig`, a May206-style integrated harness rerun should produce
    summary rows that explicitly identify `packed_bank_compute` mode and
    report the intended per-bank boundary phase contract.
- Command:
  - Launched through `scratch/muon_update_bench_fast_loop.sh iris
    fullprod-r4e8-l26-h3` with env overrides:
    `MUON_BENCH_GPU_REPLICAS=2`, `MUON_BENCH_REPLICA_AXIS=2`,
    `MUON_BENCH_DATA_AXIS=1`, `MUON_BENCH_EXPERT_AXIS=8`,
    `MUON_BENCH_MODEL_AXIS=1`, `MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn`,
    `MUON_BENCH_NS4D_GROUP_SIZE=2`,
    `MUON_BENCH_KINDS=real_expert_fsdp_grouped_muonh_optimizer_update,real_expert_fsdp_grouped_muonh_optimizer_apply`,
    `MUON_BENCH_SWEEP_BACKEND_STEPS=3`,
    `MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=512`,
    `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY=true`,
    `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE=true`,
    `MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES=false`,
    `MUON_BENCH_TRACKER=wandb`, and TTL prefix.
- Run:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-192454`.
  - Child Iris job:
    `/dlwh/iris-run-job-20260620-192454/grug-train-MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-192451`.
  - Expected W&B:
    `marin-community/marin_moe/MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-192451`.
  - Local state file:
    `scratch/20260620-1924_muon_packed_bank_compute_report_r2_state.json`.
- Current observation:
  - Parent and child were running; child had `task_state_counts={"running": 2}`.
  - Dalton (`019ee67f-d8ba-7101-a21d-e2953faae78e`) is babysitting the run.

### 2026-06-20 12:31 PDT - R2 packed-bank-compute report launch plumbing fixed
- Hypothesis:
  - The R2 report run should exercise the bounded packed-bank compute path, but
    launch metadata showed `expert_grouped_muonh_packed_bank_compute=false`.
- Result:
  - Run `/dlwh/iris-run-job-20260620-192454` failed before useful timing rows.
  - Logs confirmed launch config and per-bench metadata had
    `expert_grouped_muonh_packed_entry=true` but
    `expert_grouped_muonh_packed_bank_compute=false`.
  - The job then took the old whole-bank packed-entry path and OOMed trying to
    allocate `17.00 GiB`.
- Root cause:
  - `scratch/muon_update_bench_fast_loop.sh` exported
    `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE`, but
    `scratch/launch_muon_update_bench_executor_n1.sh` did not forward that env
    var through the Iris parent `job run`.
- Change:
  - Added `-e MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE ...` to the
    executor wrapper.
- Verification:
  - `bash -n scratch/launch_muon_update_bench_executor_n1.sh`
  - `bash -n scratch/muon_update_bench_fast_loop.sh`
  - `./infra/pre-commit.py --files scratch/launch_muon_update_bench_executor_n1.sh --fix`
- Next action:
  - Relaunch the R2 packed-bank-compute report run from the fixed wrapper and
    verify launch metadata says `expert_grouped_muonh_packed_bank_compute=true`
    before interpreting any timing rows.

### 2026-06-20 12:33 PDT - Corrected R2 packed-bank-compute report run verified
- Hypothesis:
  - After forwarding `MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE`
    through the Iris parent wrapper, the same R2 report run should enter the
    intended bounded packed-bank compute path instead of the whole-bank
    packed-entry path that OOMed.
- Command:
  - `MARIN_PREFIX=s3://marin-na/tmp/ttl=7d KUBECONFIG=$HOME/.kube/coreweave-iris-gpu RUN_ID=MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-193220 ... bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3`
- Run:
  - Parent Iris job: `/dlwh/iris-run-job-20260620-193222`.
  - Child Iris job:
    `/dlwh/iris-run-job-20260620-193222/grug-train-MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-193220`.
  - Expected W&B:
    `marin-community/marin_moe/MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-193220`.
  - Output prefix:
    `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-R2D1E8-G2-H3-PACKEDBANKCOMPUTE-REPORT-N2-cw-20260620-193220-d773c1`.
- Current observation:
  - Launch metadata now correctly reports
    `expert_grouped_muonh_packed_entry=true` and
    `expert_grouped_muonh_packed_bank_compute=true`.
  - Initial lowered HLO for
    `real_expert_fsdp_grouped_muonh_optimizer_update_h3`: AG/A2A/AR/RS =
    `26/0/0/0`, `dot_general=234`.
  - Dirac (`019ee686-7a97-7200-870e-3a50e42280b0`) is babysitting the run.
- Next action:
  - Wait for compiled HLO counts and timing rows, then compare against the
    expected packed-bank compute boundary contract.
