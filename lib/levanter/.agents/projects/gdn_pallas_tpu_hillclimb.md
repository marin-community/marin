# GDN Pallas TPU Hill-Climb Log

Append-only running log for `lib/levanter/src/levanter/layers/gated_deltanet.py` TPU optimization work.

## Goal

Increase training MFU for the Gated DeltaNet TPU implementation without changing model semantics.

## Loop Contract

Each iteration should include:

1. one optimization hypothesis,
2. one code change set,
3. TPU correctness validation,
4. one profiled training run,
5. one commit if validated.

## Known Constraints (as of 2026-01-06)

- Strict lower-triangular inversion is a TPU hotspot.
- Pallas TPU kernels do not support dynamic slice indexing in-kernel, requiring static indexing/segmentation.

## Macro Move Menu

To avoid local-minimum micro-tuning, every **performance** iteration should explicitly pick one of these categories
as the headline hypothesis:

1) **Pipelined chunk loop inside one kernel (`pltpu.emit_pipeline`)**
2) **TPU vector-layout fixes (singleton last-axis, transpose fusion, etc.)**
3) **BF16-input / FP32-accum MXU policy + `dot_general` everywhere**
4) **V/K tiling and parallelism re-map (add grid axes; reduce per-program state size)**
5) **Kernel decomposition (FLA-style multi-kernel pipeline) or partial offloading to XLA**
6) **Triangular solve/inversion redesign (hierarchical blocks / preconditioning)**

See `docs/recipes/optimize_gdn_pallas_tpu.md` for details and guardrails.

## Entry Template

```markdown
### Iteration <N> - <short title>

- Date: <UTC timestamp>
- Commit: <sha>
- Hypothesis:
- Change summary:
- Correctness checks:
  - Command:
  - Result:
- Profile run:
  - Command:
  - Job ID:
  - Trace location:
- Hotspots observed:
- MFU/throughput delta:
- Next hypothesis:
```

## Iterations

### Iteration 0 - Infra bootstrap

- Date: 2026-02-18
- Commit: 4879e0379
- Hypothesis: Standardized scripts/docs and lightweight profile entrypoint reduce iteration overhead for future optimization passes.
- Change summary: Added `scripts/gdn/gdnctl.py`, tiny profile experiment, recipe/docs, and unattended Codex loop harness.
- Correctness checks:
  - Command: N/A (infra-only change)
  - Result: N/A
- Profile run:
  - Command: N/A
  - Job ID: N/A
  - Trace location: N/A
- Hotspots observed: N/A
- MFU/throughput delta: N/A
- Next hypothesis: Use new loop to target one kernel bottleneck per commit.

### Iteration 1 - Loop hardening + trace validation

- Date: 2026-02-18
- Commit: 4879e0379
- Hypothesis: The loop must run reliably under TPU queue contention; adding safe tiny-profile defaults and a first-class dev TPU profile path will make each iteration deterministic.
- Change summary:
  - Fixed `ray-test`/`ray-profile` command and submission-id parsing issues in `scripts/gdn/gdnctl.py`.
  - Defaulted unattended Codex loop to `gpt-5.3-codex` + `model_reasoning_effort=xhigh`.
  - Added safe tiny-profile defaults for v5p-8 (`batch_size=8`, shorter profile window) in `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py` and CLI defaults.
  - Added `dev-tpu-profile` subcommand in `scripts/gdn/gdnctl.py` to bypass Ray queueing.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvin-gdn-loop --tests both --no-sync`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run scripts/ray/dev_tpu.py --cluster us-central1 --tpu-name calvin-gdn-loop execute --no-sync -e EQX_ON_ERROR=nan -e WANDB_MODE=online -e MARIN_PREFIX=gs://marin-us-central1 -e GDN_PROFILE_SIZE=130m -e GDN_PROFILE_NUM_STEPS=8 -e GDN_PROFILE_PROFILE_START_STEP=2 -e GDN_PROFILE_PROFILE_NUM_STEPS=3 -e GDN_PROFILE_BATCH_SIZE=8 -e GDN_PROFILE_RUN_NAME_PREFIX=gdn_loopcheck -- "uv pip uninstall --python .venv/bin/python torchvision || true && .venv/bin/python -m experiments.speedrun.hackable_transformer_gdn.tiny_profile --force_run_failed true"`
  - Job/Run: W&B run `gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5`
  - Trace location: `.profiles/wandb/gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5-profiler-v0/plugins/profile/2026_02_18_12_05_06/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate):
  - `while`: `218.985 ms` total; major loops mapped to `lib/levanter/src/levanter/layers/gated_deltanet.py:1861` and `lib/levanter/src/levanter/layers/gated_deltanet.py:2361`.
  - `custom-call`: `182.564 ms` total; dominant entries are `shard_map.1068-1072` from `lib/levanter/src/levanter/layers/gated_deltanet.py:2361` and `shard_map.1063-1067` from `lib/levanter/src/levanter/layers/gated_deltanet.py:1315`.
  - Large non-GDN training cost remains in logits path (`fusion.321`, source in Equinox/JAX jit; `long_name` includes `bf16[2,4096,128256]` dot-general outputs).
- MFU/throughput delta: N/A (infra-validation iteration; no kernel math change yet).
- Next hypothesis: reduce GDN segment scan overhead by fusing segment boundaries/state handoff so line-2361 and line-1861 while/custom-call blocks execute fewer large-loop iterations per step.

### Iteration 2 - Unroll flash segment scans

- Date: 2026-02-18T12:55:20Z
- Commit: 1d74d11ac
- Hypothesis: Unrolling the segment-level `lax.scan` loops in the flash TPU forward/backward path will remove `while` overhead and improve MFU.
- Change summary:
  - Added `_GDN_SEGMENT_SCAN_UNROLL = 4` in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Applied `unroll=_GDN_SEGMENT_SCAN_UNROLL` to both segment scans at `gated_deltanet.py:1862` (forward) and `gated_deltanet.py:2041` (backward).
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Result: Ray job `ray-run-calvinxu-levanter-20260218-123907` succeeded; `49 passed, 40 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_unroll4_i1 --no-wait`, then `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260218-124457 --show-logs --tail 600`
  - Job ID: `ray-run-calvinxu-bash-20260218-124457`
  - Trace location:
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_unroll4_i1_130m_ch128_seg16_20steps-12a667`
    - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_18_04_51_05/perfetto_trace.json.gz`
- Hotspots observed (TPU XLA Ops aggregate from downloaded Perfetto trace):
  - `while` category dropped from `1751.883 ms` in baseline run `gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5` to `0.000 ms` in this run.
  - `custom-call` remains dominant at `1465.997 ms`; largest GDN sources are `gated_deltanet.py:2374` (`964.671 ms`) and `gated_deltanet.py:1316` (`571.327 ms`), both from shard-map pallas calls.
  - Non-GDN top ops are still logits/haliax heavy (`conditional.2`, `select_reduce_fusion`, `fusion.6073`).
- MFU/throughput delta (vs baseline run `gdn_loopcheck_130m_ch128_seg16_8steps-5ecaf5`):
  - `throughput/mfu`: `4.1533 -> 4.2092` (`+1.34%`).
  - `throughput/tokens_per_second`: `134358.54 -> 136165.61` (`+1.35%`).
  - `throughput/duration`: `0.24388s -> 0.24065s` (`-1.33%`).
- Next hypothesis: reduce remaining GDN custom-call cost at `gated_deltanet.py:2374`/`1316` by increasing useful work per pallas call (fewer shard-map launches per training step).

### Iteration 3 - Increase flash segment scan unroll to 8

- Date: 2026-02-18T13:54:59Z
- Commit: 4645d0210
- Hypothesis: Increasing segment-level scan unroll from `4` to `8` in flash TPU forward/backward should slightly reduce residual scan overhead and improve MFU without changing kernel memory shape.
- Change summary:
  - Changed `_GDN_SEGMENT_SCAN_UNROLL` from `4` to `8` in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - No other kernel or model changes.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Result: Ray job `ray-run-calvinxu-levanter-20260218-133500` succeeded; `49 passed, 40 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_unroll8_i2_ray --no-wait`, then `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260218-134259 --show-logs --tail 600`
  - Job ID: `ray-run-calvinxu-bash-20260218-134259`
  - Trace location:
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_unroll8_i2_ray_130m_ch128_seg16_20steps-44ec2b`
    - W&B profiler artifact: `run-gdn_unroll8_i2_ray_130m_ch128_seg16_20steps-44ec2b-profiler:v0`
    - Downloaded trace: `.profiles/wandb/gdn_unroll8_i2_ray/plugins/profile/2026_02_18_05_49_17/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate from downloaded Perfetto trace):
  - `custom-call`: `183.251 ms` total (dominant category).
  - Largest GDN sources remain `gated_deltanet.py:2374` (`120.561 ms`) and `gated_deltanet.py:1316` (`71.415 ms`).
  - `while`: `0.000 ms` (still eliminated after prior unroll work).
  - Non-GDN top ops remain `conditional.2`, `select_reduce_fusion`, and `fusion.6073`.
- MFU/throughput delta (vs prior unroll-4 run `gdn_unroll4_i1_130m_ch128_seg16_20steps-12a667`):
  - `throughput/mfu`: `4.2092 -> 4.2562` (`+1.12%`).
  - `throughput/tokens_per_second`: `136165.61 -> 137688.28` (`+1.12%`).
  - `throughput/duration`: `0.24065s -> 0.23799s` (`-1.11%`).
- Next hypothesis: `custom-call` at `gated_deltanet.py:2374`/`1316` dominates; target fewer shard-map launches or more work per launch in those pallas calls.

### Iteration 4 - Single forward super-segment pallas call

- Date: 2026-02-18T07:40:10Z
- Commit: 8cf1cca9c
- Dominant bottleneck carried in: `custom-call` at `gated_deltanet.py:2374`/`1316` from Iteration 3 trace (`183.251 ms` total on TPU:0 XLA Ops aggregate).
- Candidate shortlist (estimated upside / risk):
  1. Full-sequence super-segment for both forward and backward (`+10-20%`, high vmem risk).
  2. Associative blockwise state composition to break serial segment dependencies (`>20%`, very high implementation risk).
  3. WY-style decomposition into reusable prep + state/output kernels (`+8-15%`, medium/high complexity and memory-traffic risk).
- Selected hypothesis: collapse segment-level forward launches to one large pallas custom-call (more work per launch, fewer launches), while preserving backward correctness via segment-boundary states.
- Change summary:
  - Updated forward flash path to execute one `_gdn_chunk_segment_fwd_pallas` call over all padded chunks and emit segment-start states for backward.
  - Extended forward TPU pallas kernel/specs to output segment-boundary start states (`SegStartStride`) used by backward.
  - Kept backward on bounded segment scan to avoid the full-super-segment backward vmem blowup.
  - During development, full forward+backward super-segment attempt failed with scoped vmem OOM (`RESOURCE_EXHAUSTED`) in job `ray-run-calvinxu-bash-20260218-151423`; stale job was explicitly stopped via `uv run scripts/ray/cluster.py --cluster us-central1 stop-job ray-run-calvinxu-bash-20260218-151423`.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Result: Ray job `ray-run-calvinxu-levanter-20260218-152558` succeeded; `49 passed, 40 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_supersegfwd_i1_ray --no-wait`, then `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260218-153121 --tail 80`
  - Job ID: `ray-run-calvinxu-bash-20260218-153121`
  - Trace location:
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_supersegfwd_i1_ray_130m_ch128_seg16_20steps-51e61a`
    - W&B profiler artifact: `run-gdn_supersegfwd_i1_ray_130m_ch128_seg16_20steps-51e61a-profiler:v0`
    - Downloaded trace: `.profiles/wandb/gdn_supersegfwd_i1_ray/plugins/profile/2026_02_18_07_38_54/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate, compared to Iteration 3 baseline trace):
  - `custom-call`: `183.251 ms -> 188.818 ms` (`+3.04%`), still dominant.
  - Dominant GDN source from chunk flash entry call remained: `gated_deltanet.py:2374 -> 2375`, `120.561 ms -> 121.882 ms` (`+1.10%`).
  - Secondary GDN shard-map hotspot worsened: `gated_deltanet.py:1316 -> 1335`, `71.415 ms -> 77.171 ms` (`+8.06%`).
  - `while` remained effectively eliminated (`0 ms` in both runs).
- MFU/throughput delta (vs Iteration 3 run `gdn_unroll8_i2_ray_130m_ch128_seg16_20steps-44ec2b`):
  - `throughput/mfu`: `4.2562 -> 4.1910` (`-1.53%`).
  - `throughput/tokens_per_second`: `137688.28 -> 135577.14` (`-1.53%`).
  - `throughput/duration`: `0.23799s -> 0.24169s` (`+1.56%`).
- Assessment: **low-impact / regression**. MFU gain is below 3% (negative), and dominant hotspot is unchanged (`custom-call` in the same GDN callsites).
- Next hypothesis: escalate to a radical backward redesign that changes algorithmic decomposition (e.g., blockwise associative state propagation or a two-stage backward that avoids large per-call gradient tensors) so we can safely reduce both forward and backward launch count without vmem blowups.

### Iteration 5 - Backward state tape with segmented forward launches

- Date: 2026-02-18T16:55:00Z
- Commit: e21104682
- Dominant bottleneck carried in: `custom-call` from `jit__train_step` remained dominant in Iteration 4 (`4531.684 ms` in XProf `op_profile` by-program view), with biggest GDN sources at `gated_deltanet.py:2375` and `gated_deltanet.py:1335`.
- Candidate shortlist (estimated upside / risk):
  1. Full super-segment state tape (forward all chunks + backward no recompute) (`+10-20%`, high scoped-vmem risk).
  2. Segmented-forward state tape + backward no-recompute (keep segment launch sizing, change backward dataflow) (`+8-15%`, medium implementation risk).
  3. Blockwise associative state composition in backward (`>20%`, very high algorithmic/verification risk).
- Selected hypothesis: implement option (2) to remove backward forward-recompute while keeping forward launches segment-bounded to avoid scoped-vmem blowups.
- Change summary:
  - Added per-chunk forward state tape output from TPU pallas forward kernel (`Schunkstarts_ref`) and threaded it through custom VJP residuals.
  - Replaced backward in-kernel forward-recompute with direct `S_prev` tape consumption (`Sprev_chunks_ref`) in `_gdn_chunk_segment_bwd_kernel_tpu`.
  - Updated flash forward path to run a segmented `lax.scan` of `_gdn_chunk_segment_fwd_pallas` (instead of one all-chunks super-segment launch), concatenating chunk-start tape for backward.
  - Preserved segment scan unroll (`_GDN_SEGMENT_SCAN_UNROLL = 8`) and segment-bounded backward launch structure.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_chunkstarts_segfwd_i2_dev`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_chunkstarts_segfwd_i2_dev_130m_ch128_seg16_20steps-2c11a3`
  - Trace location: `.profiles/wandb/gdn_chunkstarts_segfwd_i2_dev/plugins/profile/2026_02_18_16_49_38/perfetto_trace.json.gz`
  - XProf extraction path: compared `run-...-profiler:v0` `xplane.pb` via `xprof.convert.raw_to_tool_data(..., tool="op_profile", group_by="program")` on dev TPU host.
- Hotspots observed (before/after vs Iteration 4 run `gdn_supersegfwd_i1_ray_130m_ch128_seg16_20steps-51e61a`):
  - `custom-call`: `4531.684 ms -> 4284.468 ms` (`-5.46%`), still dominant.
  - GDN line hotspot: `gated_deltanet.py:2375 -> 2337`: `2359.334 ms -> 2246.928 ms` (`-4.76%`).
  - GDN line hotspot: `gated_deltanet.py:1335 -> 1329`: `1852.074 ms -> 1713.839 ms` (`-7.46%`).
  - Secondary shift: `all-gather` increased `562.782 ms -> 755.833 ms` (`+34.30%`), now clearly the #2 category behind `custom-call`.
- MFU/throughput delta (vs Iteration 4 run):
  - `throughput/mfu`: `4.1910 -> 4.3196` (`+3.07%`).
  - `throughput/tokens_per_second`: `135577.14 -> 139737.23` (`+3.07%`).
  - `throughput/duration`: `0.24169s -> 0.23450s` (`-2.98%`).
- Assessment: moderate win; this clears prior scoped-vmem failure path and improves MFU above the 3% threshold, but the dominant hotspot category is still `custom-call`.
- Next hypothesis: pursue a more radical launch/dataflow reduction in the remaining GDN custom-call path (e.g., associative/blockwise backward decomposition or fewer larger pallas calls that keep scoped-vmem bounded).

### Iteration 5B - Recursive block solve attempt (infra blocked)

- Date: 2026-02-19T18:24:09Z
- Commit: none (failed attempt)
- Dominant bottleneck carried in: `custom-call` remained dominant in the latest available dev trace (`178.524 ms` on TPU:0 XLA Ops aggregate), with top GDN sources at `gated_deltanet.py:2337` (`115.855 ms`) and `gated_deltanet.py:1329` (`71.412 ms`).
- Candidate shortlist (estimated upside / risk):
  1. Recursive block solve without chunk-size inverse materialization (`+10-20%`, medium risk).
  2. Two-stage WY-style prep/apply decomposition in pallas (`+12-25%`, high risk, memory-traffic risk).
  3. Associative blockwise triangular dependency composition (`>20%`, very high algorithmic risk).
- Selected hypothesis: option (1), replacing explicit strict-lower inversion with equivalent block-recursive solve/transpose-solve decomposition.
- Change summary:
  - Implemented a recursive solve path in `gated_deltanet.py` and rewired forward/backward chunk kernels to use solve + transpose-solve instead of explicit full inverse materialization.
  - Reverted the speculative code changes after infra failure so the tree does not retain unvalidated kernel edits.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Result: failed immediately with `Error: SSH configuration for dev-tpu-calvinxu-gdn not found` (allocation missing).
  - Allocation command attempted: `uv run scripts/ray/dev_tpu.py --cluster us-east5-a --tpu-name calvinxu-gdn allocate --tpu-type v5p-8`
  - Blocker: allocation timed out with `ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.`
- Profile run:
  - Command: not run (blocked on dev TPU allocation).
  - Job ID: N/A
  - Trace location: N/A
- Hotspots observed: unchanged (no new trace); carry-in dominant hotspot remains GDN `custom-call`.
- MFU/throughput delta: N/A (infra-blocked; no validated run).
- Next hypothesis: retry the recursive block-solve redesign once dev TPU allocation succeeds; if allocation remains unstable, coordinate cluster-side remediation before further kernel iterations.

### Iteration 6 - Replace explicit triangular inversion with blockwise solves

- Date: 2026-02-19T19:15:26Z
- Commit: 2db3ad589
- Dominant bottleneck carried in: `custom-call` remained dominant in Iteration 5 (`4284.468 ms` in XProf `op_profile` by-program), with largest GDN shard-map pallas callsites under `jit(_train_step)/.../HackableDecoderLayer/.../pallas_call`.
- Candidate shortlist (estimated upside / risk):
  1. Replace explicit `(I - A)^-1` construction with direct blockwise solve + transpose-solve (`+10-20%`, medium/high numerical + backward-derivation risk).
  2. Two-stage WY-style chunk decomposition to reduce per-chunk solve pressure (`+15-30%`, high implementation complexity risk).
  3. Associative blockwise scan across chunks/segments to collapse launch count (`>20%`, very high algorithmic + vmem risk).
- Selected hypothesis: implement option (1), removing explicit strict-lower triangular inversion from forward/backward hot paths and solving directly for required RHS/adjoint RHS.
- Change summary:
  - Replaced `_invert_I_minus_strict_lower_doubling` usage with `_solve_I_minus_strict_lower_blockwise` in both `_gdn_chunk_segment_fwd_kernel_tpu` and `_gdn_chunk_segment_bwd_kernel_tpu`.
  - Added `_solve_I_minus_strict_lower_transpose_blockwise` for adjoint solves in backward (no explicit inverse materialization).
  - Kept exact nilpotent-doubling semantics in base blocks and recursive block decomposition for larger tiles.
  - Fixed TPU Pallas lowering regression by removing `jnp.flip`/`rev` from transpose solve and using direct recursive upper-triangular block solve.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both --no-sync`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Failed command attempts:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_blocksolve_i6_dev --no-sync` failed with `FileNotFoundError` for `gs://marin-us-east5-a/...`.
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_blocksolve_i6_dev2 --marin-prefix gs://marin-us-central1 --no-sync` failed region check (`us-central1` path on `us-east5` VM).
  - Successful command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_blocksolve_i6_dev3 --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_blocksolve_i6_dev3_130m_ch128_seg16_20steps-8f5e31`
  - Trace location: `.profiles/wandb/gdn_blocksolve_i6_dev3/plugins/profile/2026_02_19_19_09_35/perfetto_trace.json.gz`
  - XProf extraction path: parsed `run-...-profiler:v0` `xplane.pb` via `.venv/bin/python` on dev TPU host with `xprof.convert.raw_to_tool_data.xspace_to_tool_data(..., tool="op_profile", params={"group_by":"program"})`.
- Hotspots observed (before/after vs Iteration 5 run `gdn_chunkstarts_segfwd_i2_dev_130m_ch128_seg16_20steps-2c11a3`, XProf `op_profile` by-program):
  - `jit__train_step` total: `7566.979 ms -> 8568.784 ms` (`+13.24%`).
  - `custom-call`: `4284.468 ms -> 5286.068 ms` (`+23.38%`), still dominant.
  - `all-gather`: `755.833 ms -> 756.352 ms` (`+0.07%`, effectively unchanged).
  - Dominant custom-call children remained the same shard-map GDN paths and worsened:
    - `shard_map.2936 and its duplicate(s)`: `1013.209 ms -> 1536.945 ms` (`+51.69%`).
    - `shard_map.2930 and its duplicate(s)`: `784.315 ms -> 1194.791 ms` (`+52.34%`).
    - `shard_map.2898 and its duplicate(s)`: `1028.278 ms -> 927.985 ms` (`-9.75%`).
- MFU/throughput delta (vs Iteration 5 run):
  - `throughput/mfu`: `4.3196 -> 3.9850` (`-7.75%`).
  - `throughput/tokens_per_second`: `139737.23 -> 128914.05` (`-7.75%`).
  - `throughput/duration`: `0.23450s -> 0.25418s` (`+8.40%`).
- Assessment: **low-impact / regression**. MFU gain is below 3% (negative) and dominant hotspot is unchanged (`custom-call` in the same GDN shard-map path).
- Next hypothesis: escalate to a more radical decomposition that reduces GDN custom-call launch count directly (for example, associative block transition composition across chunks/segments with fewer large pallas calls and a backward that consumes composed transitions instead of per-segment shard-map kernels).

### Iteration 7 (loop 1/20) - Remove trailing singleton layout in hot TPU Pallas g/b paths

- Date: 2026-02-20T05:44:30Z
- Commit: 0ff31bd21
- Dominant bottleneck carried in: `custom-call` remained dominant from Iteration 6 (`5286.068 ms` in XProf `op_profile` by-program), with top GDN shard-map calls under `jit(_train_step)/.../HackableDecoderLayer/.../pallas_call`.
- Candidate shortlist (estimated upside / risk):
  1. **Macro Move A**: remove `(..., Ct, 1)` singleton layouts for `g_cum/beta/dg/db` in forward+backward segmented Pallas calls (`+10-18%`, medium risk from spec/layout mismatch bugs).
  2. **Macro Move B**: systematic `jnp.matmul(..., x.T)` to `lax.dot_general` migration via one helper in fwd+bwd (`+8-15%`, medium/high risk from broad math-path churn).
  3. **Macro Move D**: full-sequence `pltpu.emit_pipeline` forward stage-axis kernel keeping recurrent state in VMEM (`+20%+`, very high implementation and correctness risk).
- Selected macro-move category: **A) Fix vector-layout pathologies**.
- Selected hypothesis: eliminate trailing singleton tensor layouts in the two dominant segmented Pallas kernels by keeping `g_cum/beta/dg/db` as rank-4 `(..., Ct)` tensors end-to-end (no `[..., None]` expansion and no `[..., 0]` squeeze).
- Change summary:
  - Updated forward segmented Pallas `in_specs` to load `g_cum` and `beta` as rank-4 blocks `(1,1,Seg,Ct)` instead of `(1,1,Seg,Ct,1)`.
  - Updated backward segmented Pallas `in_specs/out_specs` similarly so `dg`/`db` are rank-4 outputs `(B,H,Seg,Ct)`.
  - Removed forward/backward local-call singleton expansion (`gcum5 = g[..., None]`, `b5 = b[..., None]`) and corresponding trailing-dimension squeeze in segment-scan backward return path.
  - Adjusted shard-map output specs to reuse rank-4 `g_spec`/`b_spec` directly.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_rank4gb_i1_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_rank4gb_i1_dev_130m_ch128_seg16_20steps-f2d508`
  - W&B profiler artifact: `run-gdn_rank4gb_i1_dev_130m_ch128_seg16_20steps-f2d508-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_rank4gb_i1_dev/plugins/profile/2026_02_20_05_37_54/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate from downloaded Perfetto trace, compared to Iteration 6 baseline `gdn_blocksolve_i6_dev3`):
  - `custom-call`: `220.258 ms -> 223.589 ms` (`+1.51%`), still dominant.
  - `all-gather`: `31.403 ms -> 32.668 ms` (`+4.03%`).
  - Dominant GDN custom-call tf_ops unchanged and slightly slower:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `142.338 ms -> 143.558 ms` (`+0.86%`).
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `64.435 ms -> 66.546 ms` (`+3.28%`).
  - Source line numbers shifted due edit (baseline `gated_deltanet.py:2388`/`1376` now `2357`/`1368`), but the same two shard-map GDN callsites remain dominant.
- MFU/throughput delta (vs Iteration 6 run `gdn_blocksolve_i6_dev3_130m_ch128_seg16_20steps-8f5e31`):
  - `throughput/mfu`: `3.9850 -> 3.9889` (`+0.10%`).
  - `throughput/tokens_per_second`: `128914.05 -> 129039.34` (`+0.10%`).
  - `throughput/duration`: `0.25418s -> 0.25394s` (`-0.10%`).
- Assessment: **low-impact**. Gain is below 3% and dominant hotspot is unchanged (`custom-call` in the same GDN shard-map path).
- Next hypothesis: escalate to a more radical launch/dataflow move (Macro Move D or F), specifically a full-sequence pipeline kernel (`emit_pipeline`) or FLA-style 2-kernel split (solve kernel + recurrent kernel) to reduce GDN custom-call count and increase per-call arithmetic intensity.

### Iteration 8 (loop 2/20) - FLA Experiment A: split forward into solve + recurrent kernels

- Date: 2026-02-20T06:44:00Z
- Commit: a4ee55408
- Dominant bottleneck carried in: GDN shard-map custom calls remained dominant in Iteration 7 TPU:0 XLA Ops aggregate (`shard_map/custom-call` `220.522 ms` total), with top callsites at `gated_deltanet.py:2357` (transpose/jvp path, `143.558 ms`) and `gated_deltanet.py:1368` (jvp path, `66.546 ms`).
- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: full-sequence `emit_pipeline` kernel to keep recurrent state in VMEM across chunks (`+20-35%`, very high implementation risk).
  2. **Macro Move F (Experiment A)**: 2-kernel split of forward segment path into chunk-local solve kernel + recurrent apply kernel (`+10-20%`, high risk from extra HBM traffic/launch overhead).
  3. **Macro Move E**: tile recurrent state update by V-blocks (`+12-25%`, high risk from gradient path complexity and sharding changes).
- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: implement Experiment A directly by splitting forward segmented kernel into:
  - Kernel 1 (solve/prep): compute chunk-local `v_pseudo` and `k_cumdecay`.
  - Kernel 2 (recurrent/apply): consume those tensors with recurrent `S_prev` to produce `out`, `S_end`, and `chunk_starts`.
- Change summary:
  - Replaced monolithic `_gdn_chunk_segment_fwd_kernel_tpu` with two TPU Pallas kernels:
    - `_gdn_chunk_segment_prepare_kernel_tpu` + `_gdn_chunk_segment_prepare_pallas`
    - `_gdn_chunk_segment_recurrent_fwd_kernel_tpu` + `_gdn_chunk_segment_recurrent_fwd_pallas`
  - Kept segmented scan API and backward kernel structure unchanged so custom VJP wiring/correctness remained stable.
  - Updated forward wrapper `_gdn_chunk_segment_fwd_pallas` to orchestrate the two-kernel decomposition each segment.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - First run: one transient failure in `test_gdn_layer_backward_matches_hf[False]` (`max_abs ~4.1e-05`); second run passed fully.
  - Final result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_split2kernel_i2_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_split2kernel_i2_dev_130m_ch128_seg16_20steps-474c1f`
  - W&B profiler artifact: `run-gdn_split2kernel_i2_dev_130m_ch128_seg16_20steps-474c1f-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_split2kernel_i2_dev/plugins/profile/2026_02_20_06_37_50/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate, compared to Iteration 7 baseline `gdn_rank4gb_i1_dev_130m_ch128_seg16_20steps-f2d508`):
  - `shard_map` custom-call bucket: `220.522 ms -> 229.021 ms` (`+3.85%`), still dominant.
  - Backward-dominant call remained unchanged:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `143.558 ms -> 143.560 ms` (`+0.00%`).
  - Forward closed-call path worsened after split:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `66.546 ms -> 75.052 ms` (`+12.78%`).
    - Source moved from one line to two kernels (`gated_deltanet.py:1368` -> `gated_deltanet.py:1305` + `gated_deltanet.py:1483`), confirming extra work/traffic in the split path.
  - Other categories were effectively flat (`all-gather` `32.668 -> 32.684 ms`, `fusion` `46.341 -> 45.997 ms`).
- MFU/throughput delta (vs Iteration 7 run):
  - `throughput/mfu`: `3.9889 -> 3.8997` (`-2.24%`).
  - `throughput/tokens_per_second`: `129039.34 -> 126155.06` (`-2.24%`).
  - `throughput/duration`: `0.25394s -> 0.25974s` (`+2.29%`).
- Assessment: **low-impact / regression**. MFU regressed and dominant hotspot category remained unchanged.
- Next hypothesis: escalate to a more radical launch-reduction move, specifically Macro Move D with `pltpu.emit_pipeline` over full chunk axis (no Python unrolled chunk loops) and a matching backward pipeline so forward/backward shard-map call count drops instead of increasing.

### Iteration 9 - FLA Experiment B: V-tiled recurrent kernels (reverted)

- Date: 2026-02-20T09:04:15Z
- Commit: none (failed attempt)
- Loop session/local index: `1/20`
- Starting commit: `3abf4d1112ce53c4f52664fa115268b407bc004c`
- Dominant bottleneck carried in: GDN shard-map `custom-call` path remained dominant from Iteration 8 TPU:0 XLA Ops (`custom-call` `232.081 ms` on TPU:0 XLA Ops thread), with top callsites:
  - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` (`143.560 ms`)
  - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` (`75.052 ms`)
- Candidate shortlist (estimated upside / risk):
  1. **Macro Move E / FLA Experiment B**: tile recurrent forward+backward kernels over V blocks (`+10-25%`, high risk from reduction correctness and duplicated K-only compute per V block).
  2. **Macro Move D**: full-sequence `pltpu.emit_pipeline` recurrent kernel over chunk axis (`+20-35%`, very high implementation and validation risk).
  3. **Macro Move B**: global `dot_general` conversion in backward hotspot (`+8-15%`, medium risk; less launch-structure impact than E/D).
- Selected macro-move category: **E) Tile the state/output along V**.
- Selected hypothesis: run segmented recurrent forward/backward on `grid=(NH, V_blocks)` so each program holds `K x V_tile` state, with backward emitting per-V partials reduced in the wrapper.
- Change summary:
  - Implemented V-tiling for `_gdn_chunk_segment_recurrent_fwd_pallas` and `_gdn_chunk_segment_bwd_pallas` with tiled `BlockSpec` index maps and `V_tile=64` policy for `V_pad>=128`.
  - Implemented backward partial accumulation path (`dq/dk/dg/db` reduced across `V_blocks`; `dv/dS_start` merged by tiled reshape/transposes).
  - Fixed an initial shape-store bug in tiled backward outputs and re-ran TPU tests to green.
  - Reverted the kernel code after profiling because end-to-end throughput/MFU regressed beyond policy threshold; tree intentionally left without speculative kernel changes.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Final result after fix: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_vtile_recur_i1_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_vtile_recur_i1_dev_130m_ch128_seg16_20steps-90a7d2`
  - W&B profiler artifact: `run-gdn_vtile_recur_i1_dev_130m_ch128_seg16_20steps-90a7d2-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_vtile_recur_i1_dev/plugins/profile/2026_02_20_08_59_14/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops thread `pid=3, tid=3`, compared to Iteration 8 baseline trace `.profiles/wandb/gdn_split2kernel_i2_dev/plugins/profile/2026_02_20_06_37_50/perfetto_trace.json.gz`):
  - `custom-call`: `232.081 ms -> 132.090 ms` (`-43.08%`), still dominant category.
  - `all-gather`: `32.684 ms -> 20.479 ms` (`-37.34%`).
  - Same dominant GDN callsites were faster (hotspot improved, not moved):
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `143.560 ms -> 71.773 ms` (`-50.01%`).
    - `jvp(...)/closed_call/shard_map/pallas_call:` `75.052 ms -> 52.524 ms` (`-30.02%`).
- MFU/throughput delta (vs Iteration 8 run `gdn_split2kernel_i2_dev_130m_ch128_seg16_20steps-474c1f`):
  - `throughput/mfu`: `3.8997 -> 3.8252` (`-1.91%`).
  - `throughput/tokens_per_second`: `126155.06 -> 123743.20` (`-1.91%`).
  - `throughput/duration`: `0.25974s -> 0.26481s` (`+1.95%`).
- Assessment: **low-impact / regression** under current governance. Despite lower per-hotspot trace times, end-to-end MFU regressed by >1% and dominant hotspot category remained `custom-call`, so this attempt is marked failed and code reverted.
- Next hypothesis (escalation): take a more radical launch/dataflow redesign that removes duplicated chunk-local K-only work and reduces backward launch pressure, e.g. Macro Move D (`emit_pipeline` full-sequence recurrent state carry) or a backward decomposition that computes chunk-local factors once and applies recurrent updates in a separate V-tiled stage.

### Iteration 10 - Macro Move B: transpose-fused forward flash matmuls

- Date: 2026-02-20T10:10:00Z
- Commit: 2c8d3c8d
- Loop session/local index: `2/20`
- Dominant bottleneck carried in: GDN `custom-call` remained dominant in the Iteration 8 baseline trace (`232.081 ms` on TPU:0 XLA Ops thread), with the same backward/forward shard-map pallas callsites at `143.560 ms` and `75.052 ms`.
- Selected macro-move category: **B) transpose fusion via `dot_general`**.
- Selected hypothesis: remove explicit transpose-materialization from hot flash forward matmul paths by extending `_mxu_matmul_f32` and routing through transpose-fused dot variants.
- Change summary:
  - Added transpose-fusion support in `_mxu_matmul_f32`.
  - Updated hot forward/solve callsites in `lib/levanter/src/levanter/layers/gated_deltanet.py` (around lines `1017`, `1188`, `1272`, `1438`, `1453`) to use the fused path.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_dotfuse2_i2_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_dotfuse2_i2_dev_130m_ch128_seg16_20steps-457c69`
  - Trace location: `.profiles/wandb/gdn_dotfuse2_i2_dev/plugins/profile/2026_02_20_10_04_56/perfetto_trace.json.gz`
- Hotspots observed (vs Iteration 8 baseline trace):
  - `custom-call`: `232.081 ms -> 232.092 ms` (flat; unchanged dominant hotspot).
- MFU/throughput delta (vs Iteration 8 run):
  - `throughput/mfu`: `3.8997 -> 3.9135` (`+0.35%`).
  - `throughput/tokens_per_second`: `126155.06 -> 126600.07` (`+0.35%`).
  - `throughput/duration`: `0.25974s -> 0.25883s` (`-0.35%`).
- Assessment: **low-impact**. Gain is below 3%, dominant hotspot unchanged.
- Next hypothesis: target backward-dominant pallas call directly, where most residual cost remains.

### Iteration 11 - Macro Move B: transpose-fused backward flash matmuls (reverted)

- Date: 2026-02-20T10:43:43Z
- Commit: 17619a4b0
- Loop session/local index: `3/20`
- Dominant bottleneck carried in: same GDN shard-map `custom-call` path as Iteration 10, with backward-side transpose/jvp callsite still dominant.
- Selected macro-move category: **B) transpose fusion via `dot_general`**.
- Selected hypothesis: extend transpose-fusion deeper into backward hot paths to reduce backward custom-call time.
- Change summary:
  - Applied additional transpose-fused matmul rewrites in backward/adjoint paths.
  - Reverted kernel code after profiling due material end-to-end regression; commit records failed attempt.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_dotfusebwd_i3_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_dotfusebwd_i3_dev_130m_ch128_seg16_20steps-84ddf0`
  - Trace location: `.profiles/wandb/gdn_dotfusebwd_i3_dev_130m_ch128_seg16_20steps-84ddf0/plugins/profile/2026_02_20_10_39_53/perfetto_trace.json.gz`
- Hotspots observed (vs loop baseline `gdn_loopgate_iter002_...-51ecc9`):
  - `custom-call`: `232.093 ms -> 276.800 ms` (`+19.26%`).
  - Backward dominant GDN callsite: `143.559 ms -> 188.271 ms` (`+31.14%`).
  - Forward dominant GDN callsite: `75.051 ms -> 75.052 ms` (flat).
- MFU/throughput delta:
  - `throughput/mfu`: `3.8574 -> 3.6081` (`-6.46%`).
  - `throughput/tokens_per_second`: `124787.53 -> 116721.09` (`-6.46%`).
- Assessment: **failed attempt / regression**. Speculative kernel code reverted; log-only failure commit retained.
- Next hypothesis: move away from broad transpose-fusion tuning toward launch/dataflow reductions that shrink backward custom-call wall time.

### Iteration 12 - FLA Experiment A: reuse forward solve tape in backward

- Date: 2026-02-20T11:22:01Z
- Commit: 51c47da95
- Loop session/local index: `4/20`
- Starting commit: `64b706211e460717bcea452c0ce09debdc444743`
- Dominant bottleneck carried in: GDN `custom-call` remained dominant in the latest baseline trace (`232.093 ms` on TPU:0 XLA Ops aggregate), with top callsites:
  - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` (`143.559 ms`)
  - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` (`75.051 ms`)
- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F (Experiment A extension)**: persist forward solve outputs (`v_pseudo`, `k_cumdecay`) and consume them in backward to remove duplicate per-chunk solve recompute (`+10-20%`, medium/high HBM-tape risk).
  2. **Macro Move D**: full-sequence `emit_pipeline` recurrent kernel over chunks (`+20-35%`, very high implementation/validation risk).
  3. **Macro Move E / Experiment B**: staged V-tiled backward decomposition (chunk-local adjoint + recurrent apply) (`+15-30%`, high complexity/reduction risk).
- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: extend Experiment A end-to-end by reusing forward solve/prep tensors in backward, so backward keeps the recurrent pass but no longer recomputes chunk-local solve outputs.
- Change summary:
  - Threaded forward prepare outputs (`v_pseudo`, `k_cumdecay`) through `_chunk_gated_delta_rule_flash_pallas_impl(..., return_prepare_tape=True)` into custom-VJP residuals.
  - Extended `_gdn_chunk_segment_bwd_pallas` and `_gdn_chunk_segment_bwd_kernel_tpu` input specs/signatures to accept the solve tape per chunk.
  - Removed backward forward-solve recompute path (`rhs_all` + `_solve_I_minus_strict_lower_blockwise`) and consumed taped `v_pseudo`/`k_cumdecay` directly while preserving transpose-solve adjoint math.
  - Kept segmentation/launch structure intact so the iteration isolates decomposition/dataflow impact.
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - First run: one transient tolerance miss in `test_gdn_layer_backward_matches_hf[False]` (`max_abs 1.3156328e-05` vs `atol=1e-05`).
  - Rerun result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_prep_tape_i4_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_prep_tape_i4_dev_130m_ch128_seg16_20steps-e4c03f`
  - W&B profiler artifact: `run-gdn_prep_tape_i4_dev_130m_ch128_seg16_20steps-e4c03f-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_prep_tape_i4_dev_130m_ch128_seg16_20steps-e4c03f/plugins/profile/2026_02_20_11_15_48/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops aggregate, compared to baseline trace `.profiles/wandb/gdn_loopgate_iter002_130m_ch128_seg16_20steps-51ecc9/plugins/profile/2026_02_19_23_03_33/perfetto_trace.json.gz`):
  - `custom-call`: `232.093 ms -> 174.246 ms` (`-24.92%`), still dominant.
  - Dominant backward GDN callsite got faster (same hotspot, not moved):
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `143.559 ms -> 85.705 ms` (`-40.30%`).
    - Source in current run maps to `gated_deltanet.py:2614` (baseline source `gated_deltanet.py:2500`).
  - Forward closed-call GDN callsite effectively unchanged:
    - `jvp(...)/closed_call/shard_map/pallas_call:` `75.051 ms -> 75.049 ms` (`-0.00%`).
  - `all-gather`: `32.643 ms -> 32.647 ms` (`+0.01%`).
- MFU/throughput delta (vs baseline run `gdn_loopgate_iter002_130m_ch128_seg16_20steps-51ecc9`):
  - `throughput/mfu`: `3.8574 -> 4.3954` (`+13.95%`).
  - `throughput/tokens_per_second`: `124787.53 -> 142190.31` (`+13.95%`).
  - `throughput/duration`: `0.26259s -> 0.23045s` (`-12.24%`).
- Assessment: **high-impact win**. This iteration directly accelerated the same dominant backward custom-call hotspot rather than shifting bottlenecks, and MFU improved well above governance thresholds.
- Next hypothesis: push the remaining `~75 ms` forward closed-call pallas path with a bolder launch/dataflow move (Macro Move D full-sequence pipeline or Macro Move E/F staged recurrent decomposition) to reduce shard-map custom-call pressure further.

### Iteration 13 - Revert Iteration 6 blockwise solve/inversion rewrite

- Date: 2026-02-20T12:41:15Z
- Commit: 4668d57aa
- Hypothesis: Iteration 6's blockwise solve replacement may still be suppressing end-to-end MFU despite later wins; reverting it should recover additional throughput if that regression source persists.
- Change summary:
  - Reverted commit `2db3ad589` kernel math path in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Restored explicit inverse-based chunk solve path (`_invert_I_minus_strict_lower_doubling` + matmul) while preserving later architectural changes (including Iteration 12 forward-prep tape consumption in backward).
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both --no-sync`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_revert_i6_i13_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_revert_i6_i13_dev_130m_ch128_seg16_20steps-72bcb2`
  - W&B profiler artifact: `run-gdn_revert_i6_i13_dev_130m_ch128_seg16_20steps-72bcb2-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_revert_i6_i13_dev/plugins/profile/2026_02_20_12_39_21/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops thread `pid=3, tid=3`):
  - Versus Iteration 12 trace (`gdn_prep_tape_i4_dev`):
    - `custom-call`: `174.246 ms -> 185.840 ms` (`+6.65%`).
    - Backward dominant GDN callsite: `85.705 ms -> 91.478 ms` (`+6.74%`).
    - Forward dominant GDN callsite: `75.049 ms -> 80.879 ms` (`+7.77%`).
  - Versus Iteration 6 trace (`gdn_blocksolve_i6_dev3`):
    - `custom-call`: `220.258 ms -> 185.840 ms` (`-15.63%`).
- MFU/throughput delta:
  - Versus Iteration 12 run `gdn_prep_tape_i4_dev_...-e4c03f`:
    - `throughput/mfu`: `4.3954 -> 4.2823` (`-2.57%`).
    - `throughput/tokens_per_second`: `142190.31 -> 138531.44` (`-2.57%`).
    - `throughput/duration`: `0.23045s -> 0.23654s` (`+2.64%`).
  - Versus Iteration 6 run `gdn_blocksolve_i6_dev3_...-8f5e31`:
    - `throughput/mfu`: `3.9850 -> 4.2823` (`+7.46%`).
    - `throughput/tokens_per_second`: `128914.05 -> 138531.44` (`+7.46%`).
    - `throughput/duration`: `0.25418s -> 0.23654s` (`-6.94%`).
- Assessment: **partial recovery, not a new champion**. Reverting Iteration 6 materially improves over the Iteration 6 state, but underperforms Iteration 12 by ~2.6% MFU, so Iteration 12’s gain is not just an artifact of Iteration 6 regression.
- Next hypothesis: keep Iteration 12 tape-reuse path and target the remaining backward and forward shard-map pallas callsites with launch/dataflow reductions (Macro Move D/E), not a full rollback of Iteration 6-era follow-on changes.

### Iteration 14 - Re-apply Iteration 6 blockwise solve path (restore Iteration 12 baseline)

- Date: 2026-02-20T13:09:08Z
- Commit: 7f9b19a4c
- Hypothesis: Continue optimization from the strongest known baseline (Iteration 12) instead of the Iteration 13 rollback branch state.
- Change summary:
  - Re-applied commit `2db3ad589` by reverting the Iteration 13 rollback commit (`git revert 4668d57aa`).
  - Restored `lib/levanter/src/levanter/layers/gated_deltanet.py` to the same kernel code as Iteration 12 (`git diff 51c47da95..7f9b19a4c -- lib/levanter/src/levanter/layers/gated_deltanet.py` is empty).
- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both --no-sync`
  - Result: `87 passed, 2 skipped`.
- Profile run:
  - Not re-run in this reset iteration because the kernel code is identical to Iteration 12.
  - Active baseline profile/metrics remain Iteration 12 run `gdn_prep_tape_i4_dev_130m_ch128_seg16_20steps-e4c03f`.
- Assessment: **baseline reset**. This restores the known-better code path and should be treated as the launch point for subsequent optimization iterations (15+).
- Next hypothesis: target the remaining forward closed-call shard-map `custom-call` hotspot with a macro-move change (D/E/F), keeping the Iteration 12 backward tape reuse intact.

### Iteration 15 - Macro Move D / FLA Experiment A extension: lane-safe full-sequence forward recurrent pipeline (reverted)

- Date: 2026-02-20T16:33:35Z
- Commit: none (failed attempt)
- Loop session/local index: `4/20`
- Starting commit: `09a067c4db98eed262f22ca4a151d0f32ac7b0ab`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1/plugins/profile/2026_02_20_13_35_37/perfetto_trace.json.gz`, TPU XLA Ops `pid=3, tid=3`):
  - `custom-call`: `174.229 ms` (dominant category).
  - Top GDN callsites:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `85.704 ms`
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `75.049 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: make full-sequence recurrent forward `emit_pipeline` lane-safe for `head_dim=64` by staging as `(..., K, Ct)` / `(..., V, Ct)` (`+10-20%`, high compile/layout risk).
  2. **Macro Move E**: lane-packed full-sequence backward recurrence (`+15-30%`, high risk from padded overcompute and prior regressions).
  3. **Macro Move F**: FLA-style backward decomposition (chunk-local adjoint precompute + recurrent `dS` apply) (`+20-35%`, very high implementation/tape-I/O risk).
- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: keep the existing 2-kernel forward decomposition (prepare + recurrent) but rewrite the full-sequence recurrent pipeline staging layout so the path runs at `K=V=64` without last-axis 64-lane slice failures.

- Change summary:
  - Reworked `_in_specs_chunk_fullseq_recurrent_fwd_tpu` and `_gdn_chunk_fullseq_recurrent_fwd_pipeline_kernel_tpu` to lane-safe staged layouts:
    - `q/k/k_cumdecay`: `(..., K, Ct)`
    - `v_pseudo/out`: `(..., V_pipe, Ct)` with `V_pipe = round_up(V, 128)`
  - Updated `_gdn_chunk_fullseq_recurrent_fwd_pallas` wrapper to transpose/pad staged tensors in/out of the kernel while preserving external tensor contracts.
  - Enabled full-sequence recurrent forward path for chunk tiles with `Ct >= 128` (instead of requiring `K_pad,V_pad >= 128`).
  - Reverted speculative kernel changes after profile regression; tree intentionally left without speculative code changes.

- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped, 1 warning`.

- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_emitpipe_lanefwd_i4_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_emitpipe_lanefwd_i4_dev_130m_ch128_seg16_20steps-cda47e`
  - W&B profiler artifact: `run-gdn_emitpipe_lanefwd_i4_dev_130m_ch128_seg16_20steps-cda47e-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_emitpipe_lanefwd_i4_dev_130m_ch128_seg16_20steps-cda47e/plugins/profile/2026_02_20_16_29_57/perfetto_trace.json.gz`

- Hotspots observed (TPU XLA Ops `pid=3, tid=3`, vs baseline trace `.profiles/wandb/gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1/plugins/profile/2026_02_20_13_35_37/perfetto_trace.json.gz`):
  - `custom-call`: `174.229 ms -> 183.223 ms` (`+5.16%`), still dominant.
  - Dominant backward GDN callsite remained flat (hotspot unchanged):
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `85.704 ms -> 85.695 ms` (`-0.01%`).
  - Forward closed-call GDN callsite improved:
    - `jvp(...)/closed_call/shard_map/pallas_call:` `75.049 ms -> 60.653 ms` (`-19.18%`).
  - But new/expanded GDN shard-map pallas work offset the gain:
    - `jvp(...)/HackableDecoderLayer/shard_map/pallas_call:` `10.419 ms -> 33.827 ms` (`+224.68%`), source moved to `gated_deltanet.py:1755`.
  - `all-gather`: `32.731 ms -> 26.438 ms` (`-19.23%`).

- MFU/throughput delta (vs baseline run `gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1`):
  - `throughput/mfu`: `4.41497 -> 4.36303` (`-1.18%`).
  - `throughput/tokens_per_second`: `142823.53 -> 141143.16` (`-1.18%`).
  - `throughput/duration`: `0.22943s -> 0.23216s` (`+1.19%`).

- Assessment: **low-impact / regression (failed attempt)**. MFU regressed beyond the 1% regression threshold and dominant hotspot category remained `custom-call`.
- Why this did not unlock a large speedup: the lane-safe full-sequence forward path reduced one forward closed-call pallas site but introduced additional shard-map pallas time, leaving the dominant backward custom-call unchanged and raising total custom-call wall time.
- Next bold hypothesis (escalation): implement a true FLA Experiment A backward decomposition (separate chunk-local adjoint/precompute kernel from recurrent `dS` apply kernel), then pipeline only the recurrent stage with `emit_pipeline` so launch count drops without creating extra forward shard-map pressure.

### Iteration 16 - Macro Move B: transpose-fused backward/forward flash matmuls (reverted)

- Date: 2026-02-20T18:42:09Z
- Commit: none (failed attempt)
- Loop session/local index: `2/20`
- Starting commit: `c26b8f65fb05fc4bfadef88d3190769005d94d1f`
- Dominant bottleneck carried in (latest baseline trace `.profiles/wandb/gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1/plugins/profile/2026_02_20_13_35_37/perfetto_trace.json.gz`):
  - `custom-call` / `shard_map` GDN pallas path remained dominant, with backward-side shard-map calls still the largest contributor.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move B**: transpose-fuse hot flash matmuls (especially backward) through a single `dot_general` helper to avoid explicit transpose materialization (`+10-18%`, medium regression risk from broad math-path churn).
  2. **Macro Move D**: emit-pipeline backward recurrent chunk loop to remove Python-unrolled in-kernel reverse loops (`+15-30%`, high lane-layout/compile risk for `head_dim=64`).
  3. **Macro Move F (Experiment A/B)**: split backward into chunk-local adjoint-precompute + recurrent `dS` apply stage (`+20-35%`, very high implementation risk, highest upside).
- Selected macro-move category: **B) Replace transpose-matmul patterns with fused `dot_general`**.
- Selected hypothesis: extend `_mxu_matmul_f32` with transpose flags and route hot forward/backward chunk matmuls through it so transposes are expressed in dot dimension numbers rather than explicit `x.T` operands.

- Change summary:
  - Added transpose-aware matmul helper path (`transpose_a` / `transpose_b`) in `_mxu_matmul_f32`.
  - Rewrote hot forward and backward flash-kernel matmuls to use the transpose-fused helper, including chunk backward adjoint paths.
  - Reverted speculative kernel code after profiling regression; tree intentionally left without speculative code changes.

- Correctness checks:
  - Dev TPU attempt (per directive) failed due missing managed allocation:
    - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Error: `SSH configuration for dev-tpu-calvinxu-gdn not found`.
  - Fallback Ray validation:
    - Command: `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5-a --tpu auto --tests both`
    - Ray job: `ray-run-calvinxu-levanter-20260220-182819`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bwd_dotfuse_i2_ray --no-wait`
  - Ray job: `ray-run-calvinxu-bash-20260220-183305`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_bwd_dotfuse_i2_ray_130m_ch128_seg16_20steps-9d58b3`
  - W&B profiler artifact: `run-gdn_bwd_dotfuse_i2_ray_130m_ch128_seg16_20steps-9d58b3-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_bwd_dotfuse_i2_ray_130m_ch128_seg16_20steps-9d58b3/plugins/profile/2026_02_20_10_39_01/perfetto_trace.json.gz`

- Hotspots observed (Perfetto XLA Ops aggregate, compared to baseline trace `.profiles/wandb/gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1/plugins/profile/2026_02_20_13_35_37/perfetto_trace.json.gz`):
  - `custom-call`: `1393.866 ms -> 1563.605 ms` (`+12.18%`), still dominant and worse.
  - `shard_map` bucket: `1370.957 ms -> 1540.689 ms` (`+12.38%`).
  - `all-gather`: `381.254 ms -> 380.746 ms` (`-0.13%`, effectively flat).
  - Same dominant GDN shard-map callsites slowed:
    - top shard-map long-name bucket: `34.360 ms -> 43.059 ms` (`+25.32%`).

- MFU/throughput delta (vs baseline run `gdn_emitpipe_i1_dev_130m_ch128_seg16_20steps-a499c1`):
  - `throughput/mfu`: `4.41497 -> 4.12845` (`-6.49%`).
  - `throughput/tokens_per_second`: `142823.53 -> 133554.71` (`-6.49%`).
  - `throughput/duration`: `0.22943s -> 0.24535s` (`+6.94%`).

- Assessment: **low-impact / regression (failed attempt)**. MFU regressed materially and the dominant hotspot got slower; no promotion under current governance.
- Why this did not unlock a large speedup: broad transpose-fused rewrites in the backward/forward flash path increased the dominant shard-map/custom-call wall time instead of reducing it.
- Next bold hypothesis (escalation): move to a true decomposition/launch redesign (Macro Move F Experiment A/B), splitting backward into chunk-local adjoint-precompute and a separate recurrent `dS` apply stage (optionally V-tiled / pipelined) to reduce dominant shard-map pressure rather than algebraically rewriting the same fused kernel.
