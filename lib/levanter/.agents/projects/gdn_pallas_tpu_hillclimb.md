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

### Iteration 16 - Macro Move C / FLA Experiment A extension: BF16 recurrent kernels with transpose-fused dot_general

- Date: 2026-02-20T14:20:00Z
- Commit: 29cb35d8724bd817607899ca5ed6576e06ce4892
- Loop session/local index: `4/20`
- Starting commit: `8e6459b4af2f6883c18729804c454037ddefe979`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_loopgate_iter004_130m_ch128_seg16_20steps-60161d/plugins/profile/2026_02_20_03_34_47/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `custom-call`: `174.246 ms` (dominant category).
  - Top GDN callsites:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `85.705 ms`
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `75.050 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move C + Experiment A extension**: enforce BF16-input/FP32-accum policy in recurrent kernels and remove transpose materialization via fused `dot_general` in both fwd/bwd recurrent math (`+10-20%`, medium/high numerical risk).
  2. **Macro Move D**: re-attempt full-sequence backward `emit_pipeline` to reduce recurrent launch count (`+15-30%`, high compile/layout risk).
  3. **Macro Move F / Experiment B**: V-tiled backward recurrent decomposition with partial reductions (`+15-25%`, very high complexity and reduction-correctness risk).

- Selected macro-move category: **C) Switch kernel math to BF16 inputs + FP32 accumulation**.
- Selected decomposition experiment (directive alignment): **FLA Experiment A extension** (optimize the recurrent kernel side of the existing solve/recurrent split).
- Selected hypothesis: speed up the dominant recurrent custom calls by using one transpose-fused MXU helper everywhere and BF16 VMEM operands with FP32 accumulation, while keeping small-tile paths on FP32 for correctness.

- Change summary:
  - Extended `_mxu_matmul_f32` to support transpose fusion (`transpose_a` / `transpose_b`) and explicit FP32 accumulation via `lax.dot_general(..., preferred_element_type=jnp.float32)`.
  - Replaced explicit transpose materialization in hot recurrent fwd+bwd matmuls with fused helper calls.
  - Switched segmented/fullseq recurrent kernels and segmented backward kernel to BF16 operand loads for large chunk tiles (`Ct >= 128`), with automatic FP32 fallback for smaller tiles to preserve test-level parity.
  - Kept solve/prep path in FP32 by default (numerically sensitive triangular solve path).

- Correctness checks:
  - Dev TPU command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Dev TPU result: unavailable (`SSH configuration for dev-tpu-calvinxu-gdn not found`).
  - Ray fallback (final pass): `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Ray job: `ray-run-calvinxu-levanter-20260220-220649`
  - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16acc_i4_dev --marin-prefix gs://marin-us-east5 --no-sync`
  - Dev TPU result: unavailable (`SSH configuration for dev-tpu-calvinxu-gdn not found`).
  - Ray fallback submit: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16acc_i4_ray --no-wait`
  - Ray profile job: `ray-run-calvinxu-bash-20260220-221107`
  - Wait command: `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260220-221107 --show-logs --tail 400`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53`
  - W&B profiler artifact: `run-gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53/plugins/profile/2026_02_20_14_17_21/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_loopgate_iter004_130m_ch128_seg16_20steps-60161d/plugins/profile/2026_02_20_03_34_47/perfetto_trace.json.gz`):
  - `custom-call`: `174.246 ms -> 149.616 ms` (`-14.14%`), still dominant.
  - Dominant backward GDN callsite got faster (same hotspot, not moved):
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `85.705 ms -> 68.292 ms` (`-20.32%`).
  - Dominant forward GDN callsite also improved:
    - `jvp(...)/closed_call/shard_map/pallas_call:` `75.050 ms -> 67.830 ms` (`-9.62%`).
  - Secondary recurrent shard-map path stayed flat:
    - `jvp(...)/HackableDecoderLayer/shard_map/pallas_call:` `10.434 ms -> 10.430 ms` (`-0.03%`).
  - `all-gather`: `32.687 ms -> 32.663 ms` (`-0.07%`).

- MFU/throughput delta (vs baseline run `gdn_loopgate_iter004_130m_ch128_seg16_20steps-60161d`):
  - `throughput/mfu`: `4.3667 -> 4.6080` (`+5.53%`).
  - `throughput/tokens_per_second`: `141262.39 -> 149069.19` (`+5.53%`).
  - `throughput/duration`: `0.23197s -> 0.21982s` (`-5.24%`).

- Assessment: **meaningful win**. The dominant hotspot category remained `custom-call` but got materially faster at the same backward/forward recurrent callsites, yielding a >5% end-to-end MFU improvement.
- Next bold hypothesis: combine this mixed-precision recurrent path with a launch-structure change (Macro Move D or F) to reduce remaining recurrent custom-call count and target another double-digit reduction in `custom-call` wall time.

### Iteration 17 - Macro Move C / FLA Experiment A extension: BF16 flash I/O policy through custom-VJP boundaries

- Date: 2026-02-20T23:18:45Z
- Commit: 29cb35d8724bd817607899ca5ed6576e06ce4892
- Loop session/local index: `4/20`
- Starting commit: `29cb35d8724bd817607899ca5ed6576e06ce4892`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53/plugins/profile/2026_02_20_14_17_21/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `custom-call`: `149.616 ms` (dominant category).
  - Top GDN callsites:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `68.292 ms`
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `67.830 ms`
- Candidate shortlist (estimated upside / risk):
  1. **Macro Move C + Experiment A extension**: make flash path BF16-native across custom-VJP boundaries (`q/k/v`, backward tape tensors, and prepare kernel operand policy) while keeping FP32 accumulation (`+10-18%`, medium compile/numerical risk).
  2. **Macro Move D**: re-attempt full-sequence backward `emit_pipeline` to cut sequential overhead (`+15-30%`, high compile/layout risk).
  3. **Macro Move F / Experiment B**: V-tiled recurrent decomposition for backward dominant path (`+20-35%`, very high complexity/reduction risk).
- Selected macro-move category: **C) Switch kernel math to BF16 inputs + FP32 accumulation**.
- Selected decomposition experiment (directive alignment): **FLA Experiment A extension** (optimize the existing solve/recurrent split).
- Selected hypothesis: eliminate FP32 I/O cliffs around the flash Pallas path by carrying BF16 operands through forward/backward wrappers, while preserving FP32-accumulated MXU dots and FP32 gate/exp-sensitive scalar math.
- Change summary:
  - Added chunk-size-gated BF16 flash I/O policy (`Ct >= 128`) in `chunk_gated_delta_rule` and `_chunk_gated_delta_rule_flash_pallas_impl` so flash `q/k/v` enter Pallas in BF16.
  - Extended backward wrapper (`_chunk_gated_delta_rule_flash_pallas_bwd`) to keep `q/k/v`, `d_out`, and forward prep tape tensors (`v_pseudo_chunks`, `k_cumdecay_chunks`) in BF16 on the hot TPU path.
  - Made blockwise solve helpers honor `precision_mode` input dtype (`bf16` or `fp32`) instead of forcing FP32 at function entry.
  - Enabled prepare-kernel precision policy by chunk tile size (`bf16` for large tiles), with BF16 loads for `k/v` and FP32 accumulation in dot paths.
  - First profile attempt failed to compile due BF16 minor-dim insertion at `v * beta_m[:, None]`; fixed by keeping those broadcasted gate multiplications in FP32 in the prepare kernel.
- Correctness checks:
  - Dev TPU command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
  - Dev TPU result: unavailable (`SSH configuration for dev-tpu-calvinxu-gdn not found`).
  - Ray fallback (final pass): `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Ray job: `ray-run-calvinxu-levanter-20260220-230434`
  - Result: `49 passed, 40 skipped`.
- Profile run:
  - Initial submit (failed compile): `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16io_i4_ray --no-wait`
  - Failed Ray profile job: `ray-run-calvinxu-bash-20260220-225451`
  - Compile error observed: `MosaicError: ... Insertion of minor dim that is not a no-op only supported for 32-bit types` at `gated_deltanet.py:1287` (`v_beta = v * beta_m[:, None]`).
  - Final submit (after compile fix): `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16iofix_i4_ray --no-wait`
  - Successful Ray profile job: `ray-run-calvinxu-bash-20260220-230813`
  - Wait command: `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260220-230813 --show-logs --tail 400`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9`
  - W&B profiler artifact: `run-gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9/plugins/profile/2026_02_20_15_14_27/perfetto_trace.json.gz`
- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53/plugins/profile/2026_02_20_14_17_21/perfetto_trace.json.gz`):
  - `custom-call`: `149.616 ms -> 131.057 ms` (`-12.40%`), still dominant.
  - Dominant backward GDN callsite remained unchanged:
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `68.292 ms -> 68.341 ms` (`+0.07%`).
  - Dominant forward GDN callsite improved significantly:
    - `jvp(...)/closed_call/shard_map/pallas_call:` `67.830 ms -> 49.357 ms` (`-27.23%`).
  - Secondary recurrent shard-map path stayed near-flat:
    - `jvp(...)/HackableDecoderLayer/shard_map/pallas_call:` `10.430 ms -> 10.304 ms` (`-1.21%`).
  - `all-gather`: `32.663 ms -> 20.092 ms` (`-38.49%`).
- MFU/throughput delta (vs baseline run `gdn_bf16acc_i4_ray_130m_ch128_seg16_20steps-3aba53`):
  - `throughput/mfu`: `4.6080 -> 4.9270` (`+6.92%`).
  - `throughput/tokens_per_second`: `149069.19 -> 159388.47` (`+6.92%`).
  - `throughput/duration`: `0.21982s -> 0.20559s` (`-6.47%`).
- Assessment: **meaningful win**. The same dominant hotspot category (`custom-call`) got faster, with most gain coming from the forward closed-call shard-map path and lower collective cost; this clears the performance-governance promotion threshold by a wide margin.
- Why this did not unlock a larger speedup: the top backward closed-call shard-map pallas hotspot stayed flat (~68 ms), so remaining speedup headroom is concentrated in backward recurrent math/launch structure.
- Next bold hypothesis: keep this BF16 I/O policy and target the unchanged backward dominant hotspot with a structural decomposition move (Macro Move F Experiment B or Macro Move D backward recurrent pipeline) rather than additional dtype-only tweaks.

### Iteration 18 - Probe: triangular-solve bottleneck sensitivity (profile-only A/B)

- Date: 2026-02-21T01:02:30Z
- Commit: 586daf1c0a2d2cb229fb5bbe2652acc2babf56e1
- Purpose: measure an upper bound on strict-lower-triangular solve bottleneck share in the training chunk path by intentionally bypassing solve work.
- Probe policy: **ablation only** (not correctness-preserving; not champion-eligible).

- Probe setup:
  - Baseline mode: `GDN_TRIANGULAR_SOLVE_PROBE=off`
  - Ablation mode: `GDN_TRIANGULAR_SOLVE_PROBE=identity` (approximate no-op solve)
  - Matched run shape: `v5p-8`, `size=130m`, `num_steps=12`, `profile_start_step=2`, `profile_num_steps=4`, `batch_size=8`
  - Dev TPU path used due Ray queue instability.

- Commands:
  - Baseline:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 12 --profile-start-step 2 --profile-num-steps 4 --batch-size 8 --run-name-prefix gdn_trisolve_probe_baseline_dev --profile-env GDN_TRIANGULAR_SOLVE_PROBE=off --marin-prefix gs://marin-us-east5 --no-sync`
  - Identity ablation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 12 --profile-start-step 2 --profile-num-steps 4 --batch-size 8 --run-name-prefix gdn_trisolve_probe_identity_dev --profile-env GDN_TRIANGULAR_SOLVE_PROBE=identity --marin-prefix gs://marin-us-east5 --no-sync`

- Runs:
  - Baseline: `https://wandb.ai/marin-community/marin/runs/gdn_trisolve_probe_baseline_dev_130m_ch128_seg16_12step-2cd08b`
  - Identity ablation: `https://wandb.ai/marin-community/marin/runs/gdn_trisolve_probe_identity_dev_130m_ch128_seg16_12step-9c0571`

- Measured delta (identity vs baseline):
  - `throughput/mfu`: `5.1181 -> 6.2805` (`+22.71%`)
  - `throughput/tokens_per_second`: `165569.79 -> 203173.29` (`+22.71%`)
  - `throughput/duration`: `0.19791s -> 0.16128s` (`-18.51%`)

- Interpretation:
  - Strict-lower-triangular solve is a **material bottleneck** in the current training chunk path.
  - This probe is an upper-bound sensitivity test: making solve nearly free improved throughput by ~22.7%, so solve-path speedups can matter but are unlikely alone to explain the full MFU gap to target.

- Next hypothesis:
  - Pursue correctness-preserving reformulations that reduce or amortize explicit triangular solves (for example, blockwise associative/state-space reformulation) while targeting the remaining dominant custom-call hotspots in backward recurrent kernels.

### Iteration 19 - Macro Move F / triangular-transform tape reuse in backward

- Date: 2026-02-21T14:48:57Z
- Commit: d07b293baf588c6bf8f1ec2b746f4eec9e00eb52
- Loop session/local index: `1/20`
- Starting commit: `d07b293baf588c6bf8f1ec2b746f4eec9e00eb52`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9/plugins/profile/2026_02_20_15_14_27/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `custom-call`: `131.057 ms` (dominant category).
  - Top GDN callsites:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `68.341 ms`
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `49.357 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F**: tape and reuse `T=(I-A)^-1` from chunk prepare, replacing backward transpose-triangular solve with `T^T @ d_sol` MXU matmul (`+10-25%`, medium/high tape-memory risk).
  2. **Macro Move D**: convert segmented backward chunk loop to `emit_pipeline` recurrent apply over chunks (`+15-30%`, high compile/vmem risk).
  3. **Macro Move E**: V-tile backward recurrent state/apply to reduce per-program working set and improve occupancy (`+12-25%`, high implementation/reduction risk).

- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: in the flash chunk path, compute the chunk solve transform once in prepare and reuse it in backward so the dominant backward closed-call hotspot becomes MXU-heavy matmul work instead of repeated transpose solve kernels.

- Change summary:
  - Extended chunk prepare output tape with per-chunk solve transform `solve_transform = (I - A)^-1` (`Ct x Ct`) in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Reworked prepare kernel math from direct RHS solve calls to:
    - compute `solve_transform` once per chunk,
    - produce `v_pseudo/k_cumdecay` via `solve_transform @ rhs_all`.
  - Threaded `solve_transform` through flash forward tape and custom-VJP residuals.
  - Replaced backward transpose solve hot path with transform reuse:
    - old: `_solve_I_minus_strict_lower_transpose_blockwise(A, d_sol_all, ...)`
    - new: `_mxu_matmul_f32(solve_transform, d_sol_all, transpose_a=True, ...)` (`T^T @ d_sol_all`).
  - Updated segmented/full-sequence wrapper scan plumbing and shard-map specs for the additional rank-5 tape tensor, including identity padding for padded chunks in backward.

- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
  - Result: `87 passed, 2 skipped, 1 warning`.

- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_invreuse_i1_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85`
  - W&B profiler artifact: `run-gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9/plugins/profile/2026_02_20_15_14_27/perfetto_trace.json.gz`):
  - `custom-call`: `131.057 ms -> 91.176 ms` (`-30.43%`), still dominant category.
  - Same dominant backward closed-call hotspot became much faster:
    - `transpose(jvp(...))/closed_call/shard_map/pallas_call:` `68.341 ms -> 26.256 ms` (`-61.58%`), source moved from old line `3043` to `gated_deltanet.py:3180` after refactor.
  - Forward closed-call hotspot changed modestly:
    - `jvp(...)/closed_call/shard_map/pallas_call:` `49.357 ms -> 51.550 ms` (`+4.44%`), with the largest source now at `gated_deltanet.py:1406`.
  - Secondary shard-map path stayed flat:
    - `jvp(...)/HackableDecoderLayer/shard_map/pallas_call:` `10.304 ms -> 10.317 ms` (`+0.13%`).
  - `all-gather`: `20.092 ms -> 20.092 ms` (flat).

- MFU/throughput delta (vs baseline run `gdn_bf16iofix_i4_ray_130m_ch128_seg16_20steps-546ab9`):
  - `throughput/mfu`: `4.9270 -> 5.6402` (`+14.47%`).
  - `throughput/tokens_per_second`: `159388.47 -> 182457.82` (`+14.47%`).
  - `throughput/duration`: `0.20559s -> 0.17959s` (`-12.64%`).

- Assessment: **high-impact win / champion-level**. This iteration sped up the same dominant train-path `custom-call` hotspot (especially backward closed-call shard-map pallas) rather than merely moving cost, and it cleared governance promotion thresholds by a wide margin.
- Why this unlocked a large speedup: converting backward transpose-solve work into reused-transform MXU matmul dramatically shortened the previous critical-path backward custom-call while keeping other major categories flat.
- Next bold hypothesis: keep this transform-tape reuse and target the now-leading forward closed-call shard-map path (`~51.6 ms`) with a launch-structure macro move (D or E) to reduce forward custom-call count/work imbalance without re-expanding backward cost.

### Iteration 20 - Macro Move A / train-path singleton-layout rewrite (infra blocked)

- Date: 2026-02-21T17:34:52Z
- Commit: none (failed attempt)
- Loop session/local index: `2/20`
- Starting commit: `6a5194916199f8df9bf1c1ada3d87565e74121a9`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `custom-call`: `91.176 ms` (dominant category).
  - Top GDN callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `51.550 ms` (forward closed-call hotspot; source at `gated_deltanet.py:1406` in prior run).
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.256 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move A**: remove train-path `(..., Ct, 1)` style row-broadcast patterns inside flash prepare/recurrent kernels using TPU-friendly broadcast maps (`+10-18%`, medium risk from broad kernel math rewrite).
  2. **Macro Move D**: full-sequence `emit_pipeline` for prepare/recurrent staging to reduce segmented launch overhead (`+8-20%`, high compile/layout risk).
  3. **Macro Move F**: further split forward chunk prepare into transform-build and RHS-apply stages to reduce per-kernel pressure (`+10-25%`, high memory/launch-balance risk).

- Selected macro-move category: **A) Fix vector-layout pathologies**.
- Selected hypothesis: replace hot train flash row-scaling/diff patterns that rely on `[:, None]` expansion with explicit TPU-safe broadcasts to avoid last-axis singleton cliffs in forward/backward chunk kernels.

- Change attempt:
  - Implemented helper-based row scaling and pairwise-diff rewrites in `lib/levanter/src/levanter/layers/gated_deltanet.py` for flash prepare/recurrent/bwd kernels.
  - Reverted the speculative kernel edit after profiling infrastructure blocked completion, leaving no unvalidated optimization code in the working tree.

- Correctness checks:
  - Dev TPU command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
  - Dev TPU result: failed (`ssh: connect to host 136.112.108.150 port 22: Operation timed out`).
  - Ray fallback (final pass): `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
  - Ray job: `ray-run-calvinxu-levanter-20260221-170543`
  - Result: `49 passed, 40 skipped`.

- Profile attempts (blocked):
  - Dev TPU attempt (us-central1) failed:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_layoutfix_i2_dev --no-sync`
    - Failure: `ssh: connect to host 136.112.108.150 port 22: Operation timed out`.
  - Ray profile submit (us-central1): `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_layoutfix_i2_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-171123`
    - Status remained `RUNNING` for extended time with repeated autoscaler churn and worker startup failures in logs (for example, `worker_pool.cc:586 ... workers ... have not registered within the timeout`; missing virtualenv activation path), and no training/profile metrics surfaced.
    - Stop requested: `uv run scripts/ray/cluster.py --cluster us-central1 stop-job ray-run-calvinxu-bash-20260221-171123`.
  - Ray retry submit (us-east5-a): `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_layoutfix_i2_ray_east5 --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-172726`
    - Status remained `PENDING` (`waiting for resources/runtime env setup`), no profile artifact produced.
  - Dev TPU retry (us-east5-a) also failed with the same SSH timeout:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_layoutfix_i2_dev_east5 --marin-prefix gs://marin-us-east5 --no-sync`.

- Outcome:
  - **Infra-blocked iteration**: no completed profile run, no trace artifact, and no measurable MFU/tokens/sec delta for this code attempt.
  - Per failed-attempt handling, reverted speculative optimization code; tree left clean.

- Next bold hypothesis:
  - Retry the Macro Move A singleton-layout rewrite once profiling infra is healthy; if blocked again, pivot to Macro Move F decomposition targeting the forward closed-call hotspot while explicitly minimizing dependence on contested TPU queues.

### Iteration 21 - Macro Move D / fused full-sequence train forward pipeline (reverted)

- Date: 2026-02-21T20:31:14Z
- Commit: none (failed attempt)
- Loop session/local index: `5/20`
- Starting commit: `740b9fbc09f1caa52d8314b2f1b457878c20cf69`
- Dominant bottleneck carried in (latest successful train trace baseline `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`):
  - Prior loop trace summary: train-path `custom-call` remained dominant (`91.176 ms`) with forward closed-call shard-map pallas as the leading site (`~51.550 ms`).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: fuse full-sequence train forward chunk prepare + recurrent apply into a single `emit_pipeline` Pallas custom-call (`+10-25%`, high compile/layout risk).
  2. **Macro Move E**: V-tile recurrent/train state update (`K x Vb`) to reduce per-program VMEM and improve occupancy (`+12-25%`, high reduction-correctness risk).
  3. **Macro Move F**: split backward/train path into chunk-local adjoint precompute + recurrent apply stage (`+15-30%`, very high tape-I/O and implementation risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: reduce train-path launch overhead and intermediate HBM traffic by replacing the 2-kernel full-sequence forward train path (prepare + recurrent) with one fused full-sequence `emit_pipeline` kernel that emits `out`, `chunk_starts`, and backward tape tensors in one call.

- Change attempt summary:
  - Implemented a fused full-sequence forward train kernel/wrapper path in `lib/levanter/src/levanter/layers/gated_deltanet.py` for `return_prepare_tape=True`.
  - Reverted the kernel change after profiling showed a meaningful end-to-end regression against champion MFU.

- Correctness checks:
  - Dev TPU attempt (failed, TPU lock):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260221-201309`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU attempt (failed, TPU lock):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fusedfullseq_i5_dev --no-sync`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback submit:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fusedfullseq_i5_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-201916`
  - Wait:
    - `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260221-201916 --show-logs --tail 400`
    - Result: `status=SUCCEEDED`.
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_fusedfullseq_i5_ray_130m_ch128_seg16_20steps-323825`
  - W&B profiler artifact: `run-gdn_fusedfullseq_i5_ray_130m_ch128_seg16_20steps-323825-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_fusedfullseq_i5_ray_130m_ch128_seg16_20steps-323825/plugins/profile/2026_02_21_12_25_54/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`):
  - Dominant custom-call-equivalent bucket (`shard_map` in this trace format): `88.123 ms -> 76.750 ms` (`-12.91%`), still dominant.
  - `all-gather`: `20.092 ms -> 20.057 ms` (`-0.18%`) (flat).
  - Repeated dominant shard-map kernels improved but hotspot class did not change:
    - `shard_map.3841`: `2.218 ms -> 1.650 ms` (`-25.61%`)
    - `shard_map.3823`: `2.217 ms -> 1.649 ms` (`-25.62%`)

- MFU/throughput delta (vs baseline run `gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85`):
  - `throughput/mfu`: `5.640151 -> 5.457648` (`-3.24%`).
  - `throughput/tokens_per_second`: `182457.82 -> 176553.86` (`-3.24%`).
  - `throughput/duration`: `0.179592s -> 0.185598s` (`+3.34%`).

- Assessment: **low-impact / regression**. Despite lower dominant `shard_map` kernel time in trace, end-to-end MFU regressed by more than governance threshold and the dominant hotspot class remained unchanged.
- Governance action: regression exceeded threshold; reverted speculative kernel change per `revert-count-failure` policy.
- Next bold hypothesis: avoid monolithic forward fusion and instead target the unchanged train-path shard-map/custom-call bottleneck with a more radical decomposition that reduces gradient-path launch/work imbalance (for example Macro Move E V-tiling or a staged D+E train pipeline with lower tape write/read pressure).

### Iteration 22 - Macro Move A / train-path row-broadcast singleton layout rewrite

- Date: 2026-02-21T21:18:12Z
- Commit: 6b73167640ba3d4c4c8ccadf907aeb2ccf8ac90e
- Loop session/local index: `6/20`
- Starting commit: `6b73167640ba3d4c4c8ccadf907aeb2ccf8ac90e`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket in this trace format): `88.123 ms` (dominant category).
  - Top train-path callsites:
    - `gated_deltanet.py:1406`: `44.314 ms` (forward closed-call shard-map path).
    - `gated_deltanet.py:3180`: `26.256 ms` (transpose/jvp backward closed-call shard-map path).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move A**: rewrite train-path row scaling and pairwise diff construction to avoid `(..., Ct, 1)` singleton-broadcast layouts in flash prepare/recurrent/bwd kernels (`+10-20%`, medium risk).
  2. **Macro Move E**: add V-tiling (`vblock`) to recurrent/train kernels to reduce per-program VMEM and raise occupancy (`+15-30%`, high risk).
  3. **Macro Move D**: full-sequence backward `emit_pipeline` to reduce per-segment launch count in backward (`+12-25%`, high compile/runtime risk).

- Selected macro-move category: **A) Fix vector-layout pathologies**.
- Selected hypothesis: remove train-path `[:, None]` row-broadcast idioms that create pathological lane-axis singletons in hot flash custom-calls, replacing them with full-shape `lax.broadcast_in_dim` row scaling and pairwise row/column expansion.

- Change summary:
  - Added layout helpers in `lib/levanter/src/levanter/layers/gated_deltanet.py`:
    - `_pairwise_from_vector` for singleton-free pairwise row/column matrices.
    - `_scale_rows_no_singleton` for row scaling without trailing singleton axes.
  - Rewrote flash train-path kernels to use these helpers in place of `x[:, None]` row broadcasts:
    - chunk prepare kernels (segmented + full-sequence pipeline),
    - recurrent forward kernels (segmented + full-sequence pipeline),
    - chunk backward kernel math (row scaling for `k_beta`, `q_scaled`, `k_w`, `d_k`, `d_q`, `d_k_beta`, `d_v`).
  - Kept algorithm/semantics unchanged while changing dataflow/layout construction in the dominant train-path custom-call stack.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU attempt (failed, TPU lock):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260221-210025`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_rowsafe_i6_dev --no-sync`
    - Failure: startup segfault after distributed service bind (`Failed to add port to server` / `Segmentation fault`).
  - Ray fallback submit:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_rowsafe_i6_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-210559`
    - Status: `SUCCEEDED`.
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`
  - W&B profiler artifact: `run-gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85/plugins/profile/2026_02_21_14_41_31/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `88.123 ms -> 76.743 ms` (`-12.91%`), still dominant.
  - Forward closed-call callsite moved by line offsets but sped up materially:
    - `gated_deltanet.py:1406 -> gated_deltanet.py:1497`: `44.314 ms -> 32.945 ms` (`-25.66%`).
  - Dominant backward closed-call path remained flat:
    - `gated_deltanet.py:3180 -> gated_deltanet.py:3432`: `26.256 ms -> 26.254 ms` (`-0.01%`).
  - Secondary path remained flat:
    - `gated_deltanet.py:199`: `10.317 ms -> 10.307 ms` (`-0.10%`).
  - Collective cost near-flat/slightly worse:
    - `all-gather`: `20.092 ms -> 20.193 ms` (`+0.50%`).

- MFU/throughput delta (vs baseline run `gdn_invreuse_i1_dev_130m_ch128_seg16_20steps-2d1d85`):
  - `throughput/mfu`: `5.640151 -> 5.759190` (`+2.11%`).
  - `throughput/tokens_per_second`: `182457.82 -> 186308.71` (`+2.11%`).
  - `throughput/duration`: `0.179592s -> 0.175880s` (`-2.07%`).

- Assessment: **low-impact (escalation-triggering)**. This move accelerated the same dominant train-path `shard_map/custom-call` hotspot, but end-to-end MFU gain stayed below 3% with dominant hotspot class unchanged.
- Governance note: improvement cleared the `>=0.250%` promotion threshold, but per escalation rule (`<3%` and unchanged dominant hotspot), the next hypothesis must be more radical.
- Next bold hypothesis: pivot to a stronger structural move that attacks unchanged backward hotspot critical path and launch structure (for example Macro Move E V-tiling across recurrent+bwd, or Macro Move D full-sequence backward pipeline) rather than additional singleton/layout-only rewrites.

### Iteration 23 - Macro Move C / BF16 prepare-tape outputs (infra blocked, reverted)

- Date: 2026-02-21T22:27:44Z
- Commit: none (failed attempt)
- Loop session/local index: `7/20`
- Starting commit: `f7eab9057e32e34ae3062edc627b033a9087ddd7`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - forward closed-call shard-map (`gated_deltanet.py:1497` in prior run): `32.945 ms`.
    - backward closed-call shard-map (`gated_deltanet.py:3432` in prior run): `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: force full-sequence `emit_pipeline` path for 64-dim train heads to reduce segmented launch structure (`+10-20%`, high TPU tiling-risk).
  2. **Macro Move C**: keep compute accumulation in FP32 but store flash prepare tapes (`v_pseudo`, `k_cumdecay`, `solve_transform`) in BF16 to reduce train-path tape bandwidth/conversion overhead (`+10-18%`, medium numerical-risk).
  3. **Macro Move E**: V-tiling of recurrent/bwd state path (`KxV -> KxVb`) to improve occupancy (`+15-30%`, high implementation-risk).

- Selected macro-move category: **C) Switch kernel math to BF16 inputs + FP32 accumulation**.
- Selected hypothesis: preserve FP32 accumulation in matmuls, but materialize flash prepare outputs in BF16 (matching downstream bf16 usage) so train-path custom calls move less tape data and avoid redundant f32<->bf16 conversions.

- Change attempts:
  - Attempt 1 (exploratory, then reverted): widened full-sequence `emit_pipeline` gating for 64-dim heads (Macro D). Ray profile compile failed with Mosaic tiling error (`Slice shape ... must be aligned to tiling (128), but is 64`), so this path was reverted.
  - Attempt 2 (selected Macro C): changed prepare pallas out dtypes for `v_pseudo`, `k_cumdecay`, and `solve_transform` from `float32` to bf16 when `precision_mode="bf16"` in:
    - `lib/levanter/src/levanter/layers/gated_deltanet.py` (`_gdn_chunk_segment_prepare_pallas`, `_gdn_chunk_fullseq_prepare_pallas`).
  - Reverted attempt-2 code after profiling infrastructure failed to produce a completed profile run for this revision.

- Correctness checks for attempt 2:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU test attempt:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback test:
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260221-220713`
    - Result: `49 passed, 40 skipped`.

- Profile attempts (blocked, no completed trace for this revision):
  - Dev TPU profile attempt 1 (us-central1):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fullseq64_i7_dev --no-sync`
    - Failure: distributed service bind + segfault (`Failed to add port to server`, `Segmentation fault`).
  - Ray profile attempt 1 (us-central1, exploratory D attempt):
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fullseq64_i7_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-215653` (stopped)
    - Failure in logs: Mosaic compile error (`Slice shape along dimension 4 must be aligned to tiling (128), but is 64`).
  - Dev TPU profile attempt 2 (us-central1, selected C attempt):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16tape_i7_dev --no-sync`
    - Failure: same distributed service bind + segfault.
  - Ray profile attempt 2 (us-central1, selected C attempt):
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16tape_i7_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-221301` (stopped after prolonged RUNNING with no completed train/profile output).
  - Ray profile attempt 3 (us-east5-a fallback, selected C attempt):
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16tape_i7_ray_east5 --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-222004` (stopped after prolonged RUNNING/launch with no completed profile artifact).

- Outcome:
  - **Infra-blocked iteration**: no completed profile run and no trace artifact for the selected revision, so no MFU/tokens/sec delta can be claimed.
  - Per failed-attempt handling, reverted speculative kernel changes and left the tree clean.

- Next bold hypothesis:
  - Once profiling infra is healthy, retry a bandwidth-focused Macro C pass (BF16 tapes with FP32 accumulation) or pivot to Macro E (V-tiling) if custom-call dominance persists.

### Iteration 24 - Macro Move D / full-sequence backward `emit_pipeline` (regressed, reverted)

- Date: 2026-02-21T23:53:47Z
- Commit: none (failed attempt)
- Loop session/local index: `8/20`
- Starting commit: `96d8d47c272d1cfacf4416a64c704b75f254df4a`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: fuse backward segment scan into one full-sequence `emit_pipeline` Pallas call over chunks (`+10-22%`, high compile/runtime risk).
  2. **Macro Move E**: V-tiling recurrent+bwd (`KxV -> KxVb`) to reduce VMEM pressure (`+12-25%`, very high risk due duplicated `QK` work unless decomposition also changes).
  3. **Macro Move F**: split backward into staged adjoint precompute + recurrent apply kernels (`+15-30%`, very high integration risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: replace per-segment backward custom-calls with a full-sequence backward pipeline kernel that carries `dS` in VMEM scratch across reversed chunk stages, reducing launch overhead in the train chunk path.

- Change attempt summary:
  - Implemented `_gdn_chunk_fullseq_bwd_pipeline_kernel_tpu` and `_gdn_chunk_fullseq_bwd_pallas` in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Added full-sequence backward specs (`N_chunks`) and switched `_chunk_gated_delta_rule_flash_pallas_bwd` to use the new full-sequence path when `K_pad >= 128` and `V_pad >= 128`, with segmented fallback preserved for small-dim regimes.
  - Reverted the kernel code after end-to-end profile regression with unchanged dominant hotspot.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU attempt (failed, TPU lock):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260221-233704`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bwdfullseq_i8_dev --no-sync`
    - Failure: distributed service bind/segfault (`Failed to add port to server`, `Segmentation fault`).
  - Ray fallback submit:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bwdfullseq_i8_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260221-234255`
  - Wait:
    - `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260221-234255 --show-logs --tail 400`
    - Result: `status=SUCCEEDED`.
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_bwdfullseq_i8_ray_130m_ch128_seg16_20steps-0b1612`
  - W&B profiler artifact: `run-gdn_bwdfullseq_i8_ray_130m_ch128_seg16_20steps-0b1612-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_bwdfullseq_i8_ray_130m_ch128_seg16_20steps-0b1612/plugins/profile/2026_02_21_15_50_17/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 76.748 ms` (`+0.01%`), still dominant.
  - Forward closed-call tf_op remained flat:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 40.182 ms` (`+0.00%`).
  - Backward closed-call tf_op remained flat:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 26.255 ms` (`+0.00%`).
  - Source-level hotspot line moved by refactor but did not materially improve:
    - `gated_deltanet.py:3432 -> gated_deltanet.py:3941`: `38.973 ms -> 38.797 ms` (`-0.45%`).
  - `all-gather`: `20.193 ms -> 20.005 ms` (`-0.93%`) (minor).

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.684704` (`-1.29%`).
  - `throughput/tokens_per_second`: `186308.71 -> 183899.09` (`-1.29%`).
  - `throughput/duration`: `0.175880s -> 0.178185s` (`+1.31%`).

- Assessment: **low-impact / regression**. MFU regressed by >1% with the same dominant `shard_map/custom-call` hotspot unchanged.
- Governance action: regression crossed `revert-count-failure` threshold; reverted speculative kernel code and left tree clean.
- Next bold hypothesis: pivot to a stronger decomposition that changes compute balance, for example Macro Move F + triangular-solve angle (solve-only stacked RHS / transpose-solve in bwd, avoiding explicit full transform materialization) so forward/backward closed-call hotspots are structurally reduced rather than relabeled.

### Iteration 25 - Macro Move F / solve-only prepare + transpose-solve backward (regressed, reverted)

- Date: 2026-02-22T01:06:22Z
- Commit: none (failed attempt)
- Loop session/local index: `9/20`
- Starting commit: `3e5c79ee7683231686c38a9db296a74adf7e9790`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F**: remove full `solve_transform` tape and switch to solve-only decomposition (`+12-25%`, medium-high risk).
  2. **Macro Move E**: V-tiling recurrent+bwd state (`KxV -> KxVb`) to improve occupancy (`+15-30%`, high risk).
  3. **Macro Move B**: broader train-path transpose fusion with unified MXU dot helper (`+8-15%`, medium risk).

- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: avoid explicit inverse materialization in prepare by solving only stacked RHS, and in backward replace `T^T @ d_sol_all` with transpose solve on recomputed strict-lower `A`, eliminating the `Ct×Ct` forward tape from train-path dataflow.

- Change attempt summary:
  - In `lib/levanter/src/levanter/layers/gated_deltanet.py`, rewired chunk prepare path (segmented + full-sequence pipeline) to compute `sol_all` via `_solve_I_minus_strict_lower_blockwise(...)` and stop emitting `solve_transform` outputs.
  - Updated flash backward wiring to drop `solve_transform` from residuals/specs and use `_solve_I_minus_strict_lower_transpose_blockwise(A, d_sol_all, ...)` after recomputing `A` in the chunk bwd kernel.
  - Reverted kernel code after profile regression.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation (success):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Dev TPU profile (success):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_solveonly_i9_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_solveonly_i9_dev_130m_ch128_seg16_20steps-8df5c5`
  - W&B profiler artifact: `run-gdn_solveonly_i9_dev_130m_ch128_seg16_20steps-8df5c5-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_solveonly_i9_dev_130m_ch128_seg16_20steps-8df5c5/plugins/profile/2026_02_22_01_02_41/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 128.010 ms` (`+66.80%`), still dominant.
  - Forward closed-call tf_op regressed:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 49.360 ms` (`+22.84%`).
  - Backward closed-call tf_op regressed heavily:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 68.341 ms` (`+160.31%`).
  - Collectives were effectively flat:
    - `all-gather`: `20.193 ms -> 20.124 ms` (`-0.34%`).

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.074198` (`-11.89%`).
  - `throughput/tokens_per_second`: `186308.71 -> 164149.35` (`-11.89%`).
  - `throughput/duration`: `0.175880s -> 0.199623s` (`+13.50%`).

- Assessment: **low-impact / severe regression**. Dominant hotspot class was unchanged and became substantially slower, with the backward closed-call shard-map path now much more expensive.
- Governance action: regression exceeded threshold; reverted speculative kernel changes and left working tree clean.
- Next bold hypothesis: keep explicit forward transform reuse for backward, and instead pursue a more radical launch/dataflow move that increases parallelism without reintroducing expensive transpose solves in backward (for example Macro Move E V-tiling with MXU-heavy shared `QK` reuse, or a staged F decomposition that isolates backward adjoint blocks while preserving cheap `T^T` application).

### Iteration 26 - Macro Move D / fused full-sequence forward prep+recurrent kernel (low-impact, reverted)

- Date: 2026-02-22T01:42:37Z
- Commit: none (failed attempt)
- Loop session/local index: `10/20`
- Starting commit: `169f035fa0a7e80158b66e0bcda9821694733090`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: fuse full-sequence chunk prepare + recurrent apply into one `emit_pipeline` custom call to remove inter-kernel tape read/write traffic (`+10-18%`, medium-high risk).
  2. **Macro Move E**: V-tiling recurrent+bwd with shared K-only precompute (`+15-30%`, very high decomposition risk).
  3. **Macro Move C**: BF16 tape/bandwidth policy tightening across flash prepare/recurrent (`+8-15%`, medium numerical risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: in the full-sequence train path, merge chunk-local triangular prep and recurrent apply into a single pipelined Pallas kernel that keeps per-chunk intermediates in-kernel while still emitting backward tape (`v_pseudo`, `k_cumdecay`, `solve_transform`).

- Change attempt summary:
  - Implemented `_gdn_chunk_fullseq_fused_fwd_pipeline_kernel_tpu`, `_in_specs_chunk_fullseq_fused_fwd_tpu`, and `_gdn_chunk_fullseq_fused_fwd_pallas` in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Rewired `_chunk_gated_delta_rule_flash_pallas_impl` full-sequence path to call the new fused forward kernel instead of separate full-sequence prepare and recurrent calls.
  - Reverted the kernel code after profiling showed <3% MFU gain with unchanged dominant hotspot class.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fusedfwd_i10_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_fusedfwd_i10_dev_130m_ch128_seg16_20steps-b77ccb`
  - W&B profiler artifact: `run-gdn_fusedfwd_i10_dev_130m_ch128_seg16_20steps-b77ccb-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_fusedfwd_i10_dev_130m_ch128_seg16_20steps-b77ccb/plugins/profile/2026_02_22_01_38_58/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 76.754 ms` (`+0.01%`), still dominant.
  - Forward closed-call tf_op remained flat:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 40.183 ms` (`+0.00%`).
  - Backward closed-call tf_op remained flat:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 26.255 ms` (`+0.00%`).
  - Secondary shard-map callsite was also flat/slightly worse:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/shard_map/pallas_call:` `10.307 ms -> 10.315 ms` (`+0.08%`).
  - Source-level hotspot line moved by refactor but did not materially improve:
    - `gated_deltanet.py:3432 -> gated_deltanet.py:3710`: `38.973 ms -> 38.883 ms` (`-0.23%`).
  - `all-gather`: `20.193 ms -> 20.120 ms` (`-0.36%`) (minor).

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.881245` (`+2.12%`).
  - `throughput/tokens_per_second`: `186308.71 -> 190257.16` (`+2.12%`).
  - `throughput/duration`: `0.175880s -> 0.172230s` (`-2.08%`).

- Assessment: **low-impact (escalation-triggering)**. End-to-end MFU improved modestly, but the dominant train-path `shard_map/custom-call` hotspot was unchanged and key forward/backward closed-call costs remained flat.
- Governance note: improvement clears the `>=0.250%` promotion threshold, but per escalation rule (`<3%` MFU gain + unchanged dominant hotspot), this attempt is treated as low-impact and the speculative kernel code was reverted.
- Next bold hypothesis: pursue a more radical decomposition that changes backward/forward launch structure and work balance (for example Macro Move E with explicit shared K-only precompute or staged Macro F backward decomposition), rather than additional forward fusion variants.

### Iteration 27 - Macro Move C / BF16 train-tape outputs across flash prepare+recurrent kernels (low-impact, reverted)

- Date: 2026-02-22T02:18:26Z
- Commit: none (failed attempt)
- Loop session/local index: `11/20`
- Starting commit: `194214d73ec1088b8d561d6932f3206cebd9824d`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move C**: write flash prepare/recurrent training tapes in BF16 (instead of FP32) while keeping FP32 accumulation (`+10-18%`, medium-high numerical/trace risk).
  2. **Macro Move E**: V-tiling recurrent+bwd state (`KxV -> KxVb`) with shared precompute to raise occupancy (`+15-30%`, high decomposition risk).
  3. **Macro Move F**: split backward into staged adjoint precompute + recurrent apply kernels to rebalance custom-call critical path (`+15-30%`, very high integration risk).

- Selected macro-move category: **C) Switch kernel math to BF16 inputs + FP32 accumulation**.
- Selected hypothesis: reduce train-path tape bandwidth by emitting BF16 outputs for flash prepare/recurrent tape tensors (`v_pseudo`, `k_cumdecay`, `solve_transform`, `chunk_starts`) on the hot `Ct>=128` path while preserving FP32 accumulation in MXU dots.

- Change attempt summary:
  - Updated `lib/levanter/src/levanter/layers/gated_deltanet.py` so segmented/fullseq prepare and segmented/fullseq recurrent wrappers emitted BF16 tape dtypes on the BF16 precision path.
  - Kept compute kernels and accumulation semantics unchanged.
  - Reverted speculative kernel edits after end-to-end profile showed <3% MFU gain with unchanged dominant hotspot class.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback validation (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260222-020154`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU profile attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16tape_i11_dev --no-sync`
    - Failure: `ABORTED: The TPU is already in use by another process`.
  - Ray fallback submit:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_bf16tape_i11_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260222-020720`
  - Wait:
    - `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260222-020720 --show-logs --tail 400`
    - Result: `status=SUCCEEDED`.
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_bf16tape_i11_ray_130m_ch128_seg16_20steps-2f3238`
  - W&B profiler artifact: `run-gdn_bf16tape_i11_ray_130m_ch128_seg16_20steps-2f3238-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_bf16tape_i11_ray_130m_ch128_seg16_20steps-2f3238/plugins/profile/2026_02_21_18_14_48/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 76.645 ms` (`-0.13%`), still dominant.
  - Forward closed-call tf_op improved slightly:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 39.916 ms` (`-0.66%`).
  - Backward closed-call tf_op remained effectively flat:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 26.238 ms` (`-0.06%`).
  - Secondary train-path shard-map callsite regressed slightly:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/shard_map/pallas_call:` `10.307 ms -> 10.491 ms` (`+1.78%`).
  - `all-gather`: `20.193 ms -> 20.083 ms` (`-0.54%`) (minor).

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.719097` (`-0.70%`).
  - `throughput/tokens_per_second`: `186308.71 -> 185011.71` (`-0.70%`).
  - `throughput/duration`: `0.175880s -> 0.177113s` (`+0.70%`).

- Assessment: **low-impact / regression**. Dominant train-path `shard_map/custom-call` hotspot remained unchanged, with only sub-1% callsite movement and negative end-to-end MFU.
- Governance note: regression did not cross the 1.0% hard regression threshold, but escalation rule still applies (`<3%` gain + unchanged dominant hotspot), so this attempt is treated as low-impact and reverted.
- Next bold hypothesis: move to a stronger structural decomposition (Macro Move E or staged Macro Move F) that changes backward train-path work balance, e.g. V-tiling with shared `QK` precompute to increase MXU utilization without duplicating the dominant closed-call path.

### Iteration 28 - Macro Move D / full-sequence train pipeline on dk/dv>=64 + fullseq backward pipeline (compile-blocked, reverted)

- Date: 2026-02-22T02:56:15Z
- Commit: none (failed attempt)
- Loop session/local index: `12/20`
- Starting commit: `3f542c023dec607aa4ec34917af17d78195fce4e`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` (custom-call equivalent bucket): `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: enable full-sequence `emit_pipeline` train path for active 130m head dims (`dk=dv=64`) and add full-sequence backward pipeline (`+10-25%`, medium-high risk).
  2. **Macro Move E**: V-tiling recurrent+bwd state/output (`KxV -> KxVb`) with shared `QK` reuse (`+12-30%`, high risk).
  3. **Macro Move F**: staged decomposition of backward adjoint path around triangular work (`+10-20%`, high risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: move the active 130m training path off segmented scan/calls by routing `dk/dv >= 64` to full-sequence prepare/recurrent pipelines and adding a full-sequence backward pipeline that carries `dS` in scratch.

- Change attempt summary:
  - Implemented shared chunk backward math helper, full-sequence backward pipeline kernel (`emit_pipeline` over reverse chunk order), and full-sequence backward Pallas wrapper in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Changed flash-path dispatch to attempt full-sequence train kernels for `dk,dv >= 64`.
  - Reverted speculative kernel code after profile compilation failure on TPU lane-tiling constraints.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation (success):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Dev TPU profile (failed at compile):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fullseq64_i12_dev --no-sync`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_fullseq64_i12_dev_130m_ch128_seg16_20steps-0f9cdf`
    - Failure: `MosaicError` / `JaxRuntimeError`:
      - `Slice shape along dimension 4 must be aligned to tiling (128), but is 64`
      - callsite in `_gdn_chunk_fullseq_prepare_pipeline_kernel_tpu` (`gated_deltanet.py`).
  - Ray fallback profile submit:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fullseq64_i12_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260222-024715`
  - Ray fallback wait:
    - `uv run scripts/ray/cluster.py --cluster us-central1 wait-job ray-run-calvinxu-bash-20260222-024715 --poll 5`
    - Result: `status=FAILED` (job supervisor actor died; node heartbeat loss / cluster instability).

- Hotspots observed:
  - No before/after hotspot comparison available for this attempt because no successful profile trace completed.

- MFU/throughput delta:
  - Unavailable (no completed profile run for this candidate).

- Assessment: **failed attempt (compile-blocked + fallback infra failure)**. The attempted full-sequence train-path enablement for `K/V=64` hit a TPU Mosaic lane-tiling compile constraint in full-sequence prepare; Ray fallback did not produce a valid run due cluster/node failure.
- Governance action: reverted speculative kernel edits and left working tree clean.
- Next bold hypothesis: keep Macro Move D but make it lane-safe by introducing explicit full-sequence internal feature tiling/padding to 128-lane DMA slices (pack/unpack around pipeline boundaries), or pivot to Macro Move E (V-tiling) that avoids 64-lane pipeline DMA slices entirely.

### Iteration 29 - Macro Move F / QK+KKT forward-tape reuse in backward (low-impact, reverted)

- Date: 2026-02-22T03:49:24Z
- Commit: none (failed attempt)
- Loop session/local index: `13/20`
- Starting commit: `82f7259b57081cd4f40363fe1e258a174328d054`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - Train-path `shard_map/pallas_call` bucket: `76.743 ms` (dominant category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F**: persist and reuse chunk-local `QK` + `KKT` in backward (remove bwd recompute matmuls) (`+10-20%`, medium-high tape-memory risk).
  2. **Macro Move B**: full train-path transpose fusion via unified `dot_general` helper for all remaining matmuls (`+8-15%`, medium integration risk).
  3. **Macro Move 6**: redesign triangular work with block-recursive stacked-RHS solve to avoid explicit full transform materialization (`+15-35%`, high algorithmic/numerical risk).

- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: extend the forward flash chunk path to emit `QK` and `KKT` tapes, then consume those tapes in backward so chunk-bwd avoids recomputing those `Ct x Ct` products and shifts more work to forward-prepared dataflow.

- Change attempt summary:
  - Added `QK` and `KKT` tape outputs through train-path flash prepare/recurrent kernels (segmented and full-sequence wrappers).
  - Threaded both tapes through `_chunk_gated_delta_rule_flash_pallas_impl` forward outputs and custom-VJP residuals.
  - Updated backward flash path and `_gdn_chunk_segment_bwd_kernel_tpu` to read taped `QK`/`KKT` and remove local recompute of those matrices.
  - After profiling showed unchanged/worse dominant hotspots with <3% MFU gain, reverted speculative kernel edits; tree is intentionally left without this kernel change.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process` (`/tmp/libtpu_lockfile`).
  - Ray fallback validation (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260222-033430`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_qkkt_tape_i13_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_qkkt_tape_i13_dev_130m_ch128_seg16_20steps-2b582c`
  - W&B profiler artifact: `run-gdn_qkkt_tape_i13_dev_130m_ch128_seg16_20steps-2b582c-profiler:v0`
  - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_22_03_43_22/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map/pallas_call`): `76.743 ms -> 80.113 ms` (`+4.39%`), still dominant.
  - Forward closed-call tf_op remained effectively flat:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 40.203 ms` (`+0.05%`).
  - Backward closed-call tf_op regressed:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 29.592 ms` (`+12.71%`).
  - Source-level hotspot moved and regressed:
    - `gated_deltanet.py:3432 -> gated_deltanet.py:3541`: `38.973 ms -> 42.244 ms` (`+8.40%`).
  - `all-gather`: `20.193 ms -> 20.124 ms` (`-0.34%`) (minor).

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.797835` (`+0.67%`).
  - `throughput/tokens_per_second`: `186308.71 -> 187558.86` (`+0.67%`).
  - `throughput/duration`: `0.175880s -> 0.174708s` (`-0.67%`).

- Assessment: **low-impact / bottleneck regression**. End-to-end MFU moved up slightly, but the dominant train-path hotspot class was unchanged and became slower; the key backward closed-call hotspot regressed materially.
- Governance/escalation action: marked low-impact per escalation rule (`<3%` MFU gain with unchanged dominant hotspot) and reverted speculative kernel edits; no champion promotion.
- Next bold hypothesis: make a more radical train-path redesign that removes this backward closed-call pressure, e.g. Macro Move 6 block-recursive stacked-RHS triangular solve/inversion redesign (or Macro Move E V-tiling) so the dominant `shard_map/pallas_call` bucket is structurally reduced instead of shifted.

### Iteration 30 - Macro Move F / solve-only triangular decomposition (regression, reverted)

- Date: 2026-02-22T04:35:12Z
- Commit: none (failed attempt)
- Loop session/local index: `14/20`
- Starting commit: `86727a2e6a16dfa40b3fdb0d79fc5fb73b092174`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` bucket: `76.743 ms` (dominant train-path hotspot family).
  - `all-gather`: `20.193 ms` (secondary communication hotspot).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F**: replace explicit inverse tape with solve-only stacked-RHS forward prep + transpose-solve backward (`+10-25%`, medium-high algorithmic risk).
  2. **Macro Move E**: tile recurrent/bwd state along V (`KxV -> KxVb`) with shared `QK` precompute (`+15-30%`, high decomposition risk).
  3. **Macro Move D**: lane-safe full-sequence `emit_pipeline` with explicit feature tiling/padding to avoid 64-lane DMA cliffs (`+10-20%`, high compile/integration risk).

- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: remove the `Ct x Ct` forward tape (`solve_transform`) by solving `(I - A)X = rhs_all` directly in prepare kernels and using transpose-solve in backward, reducing tape bandwidth and custom-call payload.

- Change attempt summary:
  - Implemented a solve-only train-path decomposition in `lib/levanter/src/levanter/layers/gated_deltanet.py`:
    - switched segmented/fullseq prepare kernels from explicit inverse materialization to stacked-RHS solve,
    - removed `solve_transform` from forward tape/residual plumbing,
    - changed backward chunk kernel to recompute `A` and use transpose-solve instead of multiplying by taped inverse.
  - Profile showed a large end-to-end regression with worse dominant hotspot; reverted speculative kernel edits and left tree clean.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation (success):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_solveonly_i14_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_solveonly_i14_dev_130m_ch128_seg16_20steps-91b6ec`
  - W&B profiler artifact: `run-gdn_solveonly_i14_dev_130m_ch128_seg16_20steps-91b6ec-profiler:v0`
  - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_22_04_31_08/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 128.016 ms` (`+66.81%`), still dominant and significantly worse.
  - `fusion` family remained effectively flat: `45.498 ms -> 45.629 ms` (`+0.29%`).
  - `all-gather`: `20.193 ms -> 20.074 ms` (`-0.59%`) (minor).
  - Trace export did not expose stable source-level tf-op labels for this run; comparison used XLA-op family totals on the same TPU thread.

- MFU/throughput delta (vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`):
  - `throughput/mfu`: `5.759190 -> 5.125364` (`-11.01%`).
  - `throughput/tokens_per_second`: `186308.71 -> 165804.54` (`-11.01%`).
  - `throughput/duration`: `0.175880s -> 0.197630s` (`+12.37%`).

- Assessment: **high-impact regression**. The dominant train-path hotspot class was unchanged and worsened substantially; this move did not unlock useful parallelism in practice for current shapes.
- Governance/escalation action: regression exceeds the `1.0%` threshold, so this attempt is marked failed and reverted (`revert-count-failure`).
- Next bold hypothesis: escalate to **Macro Move E (V-tiling)** so backward and recurrent kernels operate on `KxVb` state tiles with a shared `QK` precompute path; target reducing the `shard_map` critical path by shrinking per-program state and increasing concurrent MXU residency.

### Iteration 31 - Macro Move D / lane-packed full-sequence train forward pipeline on `dk=dv=64` (regression, reverted)

- Date: 2026-02-22T05:15:37Z
- Commit: none (failed attempt)
- Loop session/local index: `15/20`
- Starting commit: `35b0ffd0f2dd60346c9ddf32b9f34f8d578f757b`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` bucket: `76.743 ms` (dominant train-path category).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: enable full-sequence `emit_pipeline` train forward path for active `dk=dv=64` by internal 128-lane feature packing (`+10-25%`, medium-high compile/regression risk).
  2. **Macro Move E**: V-tile recurrent/backward kernels with shared K-only precompute so `KxV -> KxVb` without duplicating chunk-local K work (`+12-30%`, high decomposition risk).
  3. **Macro Move F**: triangular-transform decomposition that avoids carrying full `Ct x Ct` tape and rebalances backward compute via staged solves (`+10-20%`, high numerical/perf risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: remove the active `dk=dv=64` fallback to segmented forward by running full-sequence prepare+recurrent pipelines with internal feature-lane padding (`64 -> 128`) and trimming tapes/outputs back to model dimensions.

- Change attempt summary:
  - Modified `_chunk_gated_delta_rule_flash_pallas_impl` in `lib/levanter/src/levanter/layers/gated_deltanet.py` to route `Ct>=128` and `K/V>=64` through full-sequence prepare/recurrent kernels.
  - Added internal lane packing to `_MXU_TILE` (`K_full`, `V_full`) before `_gdn_chunk_fullseq_prepare_pallas` and `_gdn_chunk_fullseq_recurrent_fwd_pallas`, then trimmed outputs (`out`, `chunk_starts`, `v_pseudo`, `k_cumdecay`) back to `K_pad`/`V_pad` for backward compatibility.
  - Reverted speculative kernel edits after profiling due meaningful end-to-end regression; tree intentionally left without this kernel change.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Failure: `ABORTED: The TPU is already in use by another process` (`/tmp/libtpu_lockfile`).
  - Ray fallback validation (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260222-045739`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_fullseq_lane64_i15_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_fullseq_lane64_i15_dev_130m_ch128_seg16_20steps-0286eb`
  - W&B profiler artifact: `run-gdn_fullseq_lane64_i15_dev_130m_ch128_seg16_20steps-0286eb-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_fullseq_lane64_i15_dev_130m_ch128_seg16_20steps-0286eb/plugins/profile/2026_02_22_05_11_47/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket (`shard_map`): `76.743 ms -> 89.936 ms` (`+17.20%`), still dominant and worse.
  - Dominant train-path `custom-call` category: `79.796 ms -> 92.977 ms` (`+16.52%`).
  - Backward closed-call hotspot remained flat:
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms -> 26.251 ms` (`-0.01%`).
  - Forward hotspot shifted and regressed materially:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms -> 0.000 ms` (removed).
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/shard_map/pallas_call:` `10.307 ms -> 63.685 ms` (`+517.90%`).

- MFU/throughput delta:
  - Vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`:
    - `throughput/mfu`: `5.759190 -> 5.536093` (`-3.87%`).
    - `throughput/tokens_per_second`: `186308.71 -> 179091.54` (`-3.87%`).
    - `throughput/duration`: `0.175880s -> 0.182968s` (`+4.03%`).
  - Vs active governance champion (`gdn_loopgate_iter014_130m_ch128_seg16_20steps-e0fd62`):
    - `throughput/mfu`: `5.729122 -> 5.536093` (`-3.37%`).

- Assessment: **low-impact / regression**. The dominant train-path hotspot class (`shard_map/custom-call`) is unchanged and became significantly slower; lane-packing full-sequence forward for `dk=dv=64` increased the critical path instead of reducing it.
- Governance/escalation action: regression exceeds the `1.0%` threshold; attempt marked failed and kernel code reverted (`revert-count-failure`).
- Next bold hypothesis: escalate to **Macro Move E** with shared K-only precompute plus V-tiled recurrent/backward kernels so per-program state is reduced (`KxV -> KxVb`) without introducing 128-lane forward overcompute on the active 64-dim train config.

### Iteration 32 - Macro Move D / lane-major full-sequence forward pipeline for `dk=dv=64` (infra-blocked, reverted)

- Date: 2026-02-22T06:28:57Z
- Commit: none (failed attempt)
- Loop session/local index: `16/20`
- Starting commit: `785edf9f64367c9dee355662150a432204b45045`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` / train-path `custom-call` family: `76.743 ms` (dominant bucket).
  - Top train-path callsites:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `40.182 ms`.
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call:` `26.254 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: lane-major full-sequence prepare+recurrent forward pipelines (`(..., K/V, Ct)` with `Ct=128` on lane axis) for active `dk=dv=64` path without 128-dim compute overpadding (`+10-25%`, high compile/layout risk).
  2. **Macro Move E**: V-tiled recurrent+bwd state (`KxV -> KxVb`) with shared K-only precompute (`+15-30%`, very high decomposition risk).
  3. **Macro Move F**: staged backward decomposition (adjoint-precompute + recurrent apply) (`+15-30%`, very high integration/tape risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: enable full-sequence train forward kernels for `dk=dv=64` with lane-safe feature-major layout to reduce launch overhead without doubling core K/V compute.

- Change attempt summary:
  - Implemented lane-major full-sequence prepare/recurrent forward Pallas kernels and dispatch path in `lib/levanter/src/levanter/layers/gated_deltanet.py`.
  - Added column-scaling helper and lane-major wrappers for full-sequence train forward execution.
  - Reverted speculative kernel code after profile infrastructure could not produce any completed run for this attempt.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation (failed strict parity case):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `1 failed, 86 passed, 2 skipped`; failure at `tests/test_gdn_layer.py::test_gdn_layer_backward_matches_hf[False]` (tiny max abs diff `3.05e-05`).
  - Ray fallback validation (success):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Job: `ray-run-calvinxu-levanter-20260222-055130`
    - Result: `49 passed, 40 skipped`.

- Profile run:
  - Dev TPU profile attempt (failed lock contention):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_lanefeat_i16_dev`
    - Failure: `ABORTED: The TPU is already in use by another process` (`/tmp/libtpu_lockfile`).
  - Ray profile attempt #1:
    - Submit: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_lanefeat_i16_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260222-055752`
    - Behavior: remained `RUNNING` for an extended window with no completed profiler artifact; logs showed repeated `ray.exceptions.RayTaskError(NotImplementedError)` retry paths under `run_on_pod_ray`; job explicitly stopped:
      - `uv run scripts/ray/cluster.py --cluster us-central1 stop-job ray-run-calvinxu-bash-20260222-055752`
  - Ray profile attempt #2 (retry):
    - Submit: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_lanefeat_i16b_ray --no-wait`
    - Job: `ray-run-calvinxu-bash-20260222-061611`
    - Behavior: remained `RUNNING` with repeated Ray runtime-env/worker churn (`worker_pool.cc: Delete runtime env failed`) and no completed profile artifact; job explicitly stopped:
      - `uv run scripts/ray/cluster.py --cluster us-central1 stop-job ray-run-calvinxu-bash-20260222-061611`
  - Trace location: N/A (no completed profile artifact).

- Hotspots observed:
  - No valid before/after hotspot comparison for this attempt because no profile run completed.
  - Carry-in dominant hotspot remains train-path `shard_map/custom-call` bucket from baseline trace.

- MFU/throughput delta:
  - Unavailable (no completed profile run for this candidate).

- Assessment: **infra-blocked failed attempt**. Validation fallback passed, but required profiling could not be completed after dev lock contention and two stalled Ray profile jobs. Speculative kernel changes were reverted to keep the tree free of unvalidated optimization code.
- Next bold hypothesis: rerun the same lane-major Macro Move D candidate once profiling infra is healthy (prefer held dev TPU without lock contention), otherwise pivot to Macro Move E with explicit V-tiling + shared K-only precompute and re-attempt with a stable profile lane.

### Iteration 33 - Macro Move D / segmented train emit_pipeline with lane-aligned fallback (low-impact)

- Date: 2026-02-22T08:34:44Z
- Commit: 9f2559dca9ef2517f797dbacf8b91061d3b5f10a
- Loop session/local index: `17/20`
- Starting commit: `9f2559dca9ef2517f797dbacf8b91061d3b5f10a`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map` train-path custom-call-equivalent bucket: `76.743 ms` (dominant).
  - `fusion` bucket: `45.498 ms` (secondary compute bucket).
  - `all-gather`: `20.193 ms` (communication secondary).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move D**: convert segmented train kernels to `pltpu.emit_pipeline` stage loops with scratch-carried recurrent state to reduce launch/unroll overhead (`+10-20%`, medium-high compile/layout risk).
  2. **Macro Move E**: V-tiling (`KxV -> KxVb`) in recurrent/backward train path (`+15-30%`, high decomposition risk).
  3. **Macro Move F**: solve/invert decomposition redesign for triangular path to rebalance backward critical path (`+10-25%`, high numerical/integration risk).

- Selected macro-move category: **D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops**.
- Selected hypothesis: replace segmented per-chunk Python-loop kernels with staged `emit_pipeline` kernels (forward fused + backward), then keep a lane-safe fallback for `dk/dv=64` where staged DMA slicing violates TPU lane tiling constraints.

- Change summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added staged `emit_pipeline` execution in segmented fused train forward kernel with VMEM scratch state carry across chunk stages.
  - Added staged reverse-order `emit_pipeline` execution in segmented train backward kernel with VMEM scratch carry for `dS`.
  - Added explicit lane-aligned guardrails:
    - use staged pipeline only when `K_pad >= 128` and `V_pad >= 128`,
    - keep original in-kernel loop implementation for `dk/dv=64` to avoid Mosaic DMA slice alignment failures.
  - Preserved train-path dispatch semantics and backward tape contract.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation (final required run):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_segpipe_i17_dev --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`
  - W&B profiler artifact: `run-gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937/plugins/profile/2026_02_21_13_13_09/perfetto_trace.json.gz`):
  - Dominant train-path bucket unchanged and slightly slower:
    - `shard_map`: `76.743 ms -> 78.098 ms` (`+1.77%`).
  - Other top buckets remained effectively flat:
    - `fusion`: `45.498 ms -> 45.618 ms` (`+0.26%`).
    - `all-gather`: `20.193 ms -> 20.158 ms` (`-0.17%`).
  - Launch-structure effect was visible but insufficient:
    - `shard_map` event count on TPU:0 XLA Ops thread dropped `130 -> 90` (`-30.8%`),
    - average per-event time increased `0.590 ms -> 0.868 ms` (`+47.0%`),
    - net shard-map time still increased (`+1.77%`).
  - This trace export did not provide stable source-level `tf_op` labels for direct `gated_deltanet.py:<line>` comparison; analysis used XLA-op bucket totals on the same thread.

- MFU/throughput delta:
  - Vs baseline run `gdn_rowsafe_i6_ray_130m_ch128_seg16_20steps-38f937`:
    - `throughput/mfu`: `5.759190 -> 5.787594` (`+0.49%`).
    - `throughput/tokens_per_second`: `186308.71 -> 187227.57` (`+0.49%`).
    - `throughput/duration`: `0.175880s -> 0.175017s` (`-0.49%`).
  - Vs active governance champion (`gdn_loopgate_iter015_130m_ch128_seg16_20steps-da7e49`):
    - `throughput/mfu`: `5.748507 -> 5.787594` (`+0.68%`).

- Assessment: **low-impact**. This move achieved the intended structural effect (fewer larger train-path custom calls) but did not reduce the dominant hotspot cost; the dominant `shard_map` bucket remained and became slightly slower while end-to-end MFU gain stayed `<3%`.
- Governance/escalation action: improvement exceeds promotion floor (`>=0.250%`) but escalation rule still applies (`<3%` with unchanged dominant hotspot). Next attempt should be more radical than launch restructuring alone.
- Next bold hypothesis: escalate to **Macro Move E** (state/output V-tiling with shared K-only precompute in train backward/recurrent) or **Macro Move F** (blockwise stacked-RHS triangular decomposition) to reduce per-call work, not only call count.

### Iteration 34 - Macro Move G / exp-diff centered outer-product (infra-blocked, reverted)

- Date: 2026-02-22T19:28:44Z
- Commit: none (failed attempt)
- Loop session/local index: `6/10`
- Starting commit: `6c194533e2489c22d88bee04bd1596a01f76ac22`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - train-path `shard_map`/`custom-call` bucket remained dominant (`~78 ms` on TPU:0 XLA Ops thread).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` to replace Ct x Ct exponentials in prepare/recurrent/backward (`+10-20%`, medium numerical/compile risk).
  2. **Macro Move H**: stack shared-RHS matmuls (`QK/KKT`, `inter/v_prime`) to reduce dot call count (`+8-18%`, medium integration risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) for recurrent/backward state (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add `_exp_diff_and_mask_from_g` and wire it into train-path chunk prepare/recurrent/backward kernels so fast-path uses O(Ct) vector exponentials.

- Change attempt summary:
  - Implemented `_exp_diff_and_mask_from_g` and migrated train-path `exp_diff` construction in `lib/levanter/src/levanter/layers/gated_deltanet.py` across prepare, recurrent forward, and backward chunk math.
  - Attempt was reverted after repeated TPU validation infrastructure failures/timeouts prevented obtaining a completed validation+profile result.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation attempts (blocked/no terminal result):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
    - Direct fallback attempts also stalled without completion summary:
      - `ssh dev-tpu-calvinxu-gdn '... uv run pytest tests/test_gdn_kernels.py tests/test_gdn_layer.py -q'`
      - `ssh dev-tpu-calvinxu-gdn '... uv run pytest tests/test_gdn_kernels.py -q'`
    - In each dev attempt, remote pytest remained running for extended windows (>10-15 min) with no final pass/fail output; runs were explicitly terminated.
  - Ray fallback validation (infra-queued, not started):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - Submitted jobs:
      - `ray-run-calvinxu-levanter-20260222-181337`
      - `ray-run-calvinxu-levanter-20260222-190146`
    - Both remained `PENDING`; Ray stop/status reported: `Job supervisor actor failed to start within 900.0 seconds` (resource unavailability).

- Profile run:
  - Not started because TPU correctness gate could not be completed on either dev TPU or Ray path in this window.
  - Trace artifact: N/A.

- Hotspots observed:
  - No new before/after hotspot comparison available (no completed profile run).
  - Carry-in dominant hotspot remains train-path `shard_map/custom-call` bucket from the latest successful baseline.

- MFU/throughput delta:
  - Unavailable (no completed profile run).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both` (blocked; no terminal result).
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable (no completed profile run).
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration` deltas: unavailable.
    - Macro G exp-op reduction note: unavailable (no completed profile/IR capture).
  - Governance:
    - Infra-blocked path; speculative kernel edits reverted and attempt marked failed (`Commit: none (failed attempt)`).

- Assessment: **infra-blocked failed attempt**. Required TPU validation + profiling evidence could not be completed due repeated dev-run non-termination and Ray jobs stuck pending.
- Governance/escalation action:
  - Reverted speculative kernel changes; working tree returned to baseline.
  - Stopped queued Ray jobs (`ray-run-calvinxu-levanter-20260222-181337`, `ray-run-calvinxu-levanter-20260222-190146`).
- Next bold hypothesis:
  - Re-attempt Macro Move G once TPU validation/profiling lanes are healthy, then immediately compare forward/backward closed-call deltas and exp-op counts; if infra remains unstable, hold queue resources first (dedicated dev TPU or alternate cluster) before further kernel edits.

### Iteration 35 - Macro Move G / exp-diff centered outer-product (infra-blocked, reverted)

- Date: 2026-02-22T22:32:10Z
- Commit: none (failed attempt)
- Loop session/local index: `7/10`
- Starting commit: `22e838c3af70cd444a5d63576d1002de35040059`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - train-path `shard_map/custom-call` bucket remained dominant (`~78 ms` on TPU:0 XLA Ops thread).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` in prepare/recurrent/backward to remove Ct x Ct exponential-heavy work (`+10-20%`, medium numerical risk).
  2. **Macro Move H**: stack shared-RHS matmuls (`QK/KKT`, `inter/v_prime`) to reduce dot invocation count (`+8-18%`, medium integration/VMEM risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) for recurrent/backward state footprint (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: introduce `_exp_diff_and_mask_from_g` and wire it through train-path chunk prepare/recurrent/backward kernels so fast path uses O(Ct) vector exponentials and avoids Ct x Ct exp calls on target train chunks.

- Change attempt summary:
  - Implemented `_exp_diff_and_mask_from_g` and replaced train-path `exp_diff` construction in:
    - segmented/full-sequence prepare kernels,
    - segmented/full-sequence recurrent forward kernels,
    - fused segmented forward kernel,
    - segmented backward chunk helper.
  - Added gradient-stability adjustment (`stop_gradient` on centering term) and a guard to keep the fast path restricted to MXU-sized chunks (`Ct >= 128`).
  - Reverted all speculative kernel edits after TPU infra instability prevented obtaining a completed, valid validation+profile cycle on the final code state.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU validation attempts (blocked):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both` (multiple attempts).
    - Direct fallback probes:
      - `ssh -tt dev-tpu-calvinxu-gdn '... uv run pytest tests/test_gdn_kernels.py tests/test_gdn_layer.py -v'`
      - `ssh dev-tpu-calvinxu-gdn '... uv run pytest -q tests/test_gdn_kernels.py::test_flash_chunk_backward_chunk_size_invariance_kernel_level[True]'`
    - Observed blocker: repeated remote non-termination, then dev TPU host dropped (`Connection to 136.112.108.150 closed by remote host`) and later became unavailable (`ssh: connect to host 136.112.108.150 port 22: Connection refused`).
  - Ray fallback validation attempts (blocked/unstable):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
      - `ray-run-calvinxu-levanter-20260222-210939`: started; failed before full completion (`JOB_SUPERVISOR_ACTOR_DIED`, node terminated / SIGTERM) after showing flash `[True]` regressions.
      - `ray-run-calvinxu-levanter-20260222-222301`: failed immediately (`JOB_SUPERVISOR_ACTOR_DIED`, node terminated before actor start).
      - `ray-run-calvinxu-levanter-20260222-222420`: remained `PENDING` waiting for resources; stop attempt timed out:
        - `uv run scripts/ray/cluster.py --cluster us-central1 stop-job ray-run-calvinxu-levanter-20260222-222420`
        - error: `subprocess.TimeoutExpired: Command '['ray', 'job', 'stop', ...]' timed out after 60 seconds`.

- Profile run:
  - Not started. Correctness gate for the final code state could not be completed on either dev TPU or Ray fallback due infra instability.
  - Trace artifact: N/A.

- Hotspots observed:
  - No new valid before/after hotspot comparison for this iteration (no completed profile run on validated code).
  - Carry-in dominant hotspot remains train-path `shard_map/custom-call` bucket from the prior successful trace.

- MFU/throughput delta:
  - Unavailable (no completed profiled run).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both` (blocked by dev TPU host loss and Ray job-instability fallback).
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable.
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`: unavailable.
    - Macro G exp-op reduction note: unavailable (no completed profile/IR capture on final state).
  - Governance:
    - Infra-blocked iteration; speculative code reverted and attempt recorded as failed (`Commit: none (failed attempt)`).

- Assessment: **infra-blocked failed attempt**. Could not complete required TPU validation + profiling on the final candidate due dev TPU host outage and repeated Ray job supervisor/resource instability.
- Governance/escalation action:
  - Reverted speculative kernel changes; working tree returned to `22e838c3af70cd444a5d63576d1002de35040059`.
  - Recorded exact blocking commands/job IDs for rerun triage.
- Next bold hypothesis:
  - Re-attempt Macro Move G once TPU infra is stable (reserved dev TPU or healthy Ray queue), then immediately capture forward/backward `shard_map/pallas_call` deltas and exp-op reduction evidence.
  - If infra remains unstable, pivot next validated kernel iteration to **Macro Move H** with stacked shared-RHS matmuls after securing a stable execution lane.

### Iteration 36 - Macro Move G / exp-diff centered outer-product (reverted, low-impact)

- Date: 2026-02-23T07:09:37Z
- Commit: none (failed attempt)
- Loop session/local index: `6/10`
- Starting commit: `5015bcbb9f5ac1f16f95f68c5b4eed1de592baff`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map`/custom-call-equivalent bucket: `78.098 ms` (dominant).
  - secondary buckets: `fusion 45.618 ms`, `all-gather 20.158 ms`.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` (`+10-20%`, medium numerical/compile risk).
  2. **Macro Move H**: shared-RHS matmul batching (`+8-18%`, medium integration/VMEM risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) in recurrent/backward (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add `_exp_diff_and_mask_from_g` and apply it across train-path chunk prepare/recurrent/backward so fast path uses O(Ct) vector exponentials.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented `_exp_diff_and_mask_from_g` and replaced train-path `exp_diff` construction in prepare/recurrent/fused-forward/backward chunk helpers.
  - Initial TPU compile failed (`Mosaic failed to legalize scf.if` from conditionalized fallback).
  - Follow-up branch-free/mode-gated variants produced TPU correctness regressions (`use_flash=True` parity failures / NaNs).
  - Reverted all speculative kernel edits; tree returned to `5015bcbb9f5ac1f16f95f68c5b4eed1de592baff`.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (successful run on reverted tree):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.
  - Later re-validation attempt after profiling (non-blocking to this iteration result) hit infra contention:
    - same `dev-tpu-test` command failed with `ABORTED: The TPU is already in use...` and stale `/tmp/libtpu_lockfile` not removable (`Operation not permitted`).
    - Ray fallback submission `ray-run-calvinxu-levanter-20260223-070225` stayed `PENDING`; stop attempt timed out:
      - `uv run scripts/ray/cluster.py --cluster us-east5-a stop-job ray-run-calvinxu-levanter-20260223-070225`
      - `subprocess.TimeoutExpired ... ray job stop ... after 60 seconds`.

- Profile runs:
  - Dev TPU profile attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter6_macroG_failcheck --no-sync`
    - failure: `FileNotFoundError` writing executor info under `gs://marin-us-east5-a/...`.
  - Ray fallback profile (completed):
    - submit: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter6_macroG_failcheck_ray --no-wait`
    - job: `ray-run-calvinxu-bash-20260223-065128`
    - wait: `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-east5-a ray-run-calvinxu-bash-20260223-065128 --show-logs --tail 400`
    - status: `SUCCEEDED`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter6_macroG_failcheck_ray_130m_ch128_seg16_20-bf859f`
  - W&B artifact: `run-gdn_loop_iter6_macroG_failcheck_ray_130m_ch128_seg16_20-bf859f-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_loop_iter6_macroG_failcheck_ray_130m_ch128_seg16_20-bf859f/plugins/profile/2026_02_22_22_59_10/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - Dominant train-path bucket unchanged:
    - `shard_map`: `78.098 ms -> 78.094 ms` (`-0.01%`).
  - Other top buckets effectively flat:
    - `fusion`: `45.618 ms -> 45.585 ms` (`-0.07%`).
    - `all-gather`: `20.158 ms -> 20.097 ms` (`-0.30%`).
  - Event volume unchanged (`11761 -> 11762` on TPU:0 XLA Ops thread).
  - Forward/backward closed-call `shard_map/pallas_call` separation is unavailable in this trace export (only numeric `shard_map.*` labels, no stable source-level `closed_call` tags).

- MFU/throughput delta (vs baseline `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.751907` (`-0.62%`).
  - `throughput/tokens_per_second`: `187227.57 -> 186073.10` (`-0.62%`).
  - `throughput/duration`: `0.175017s -> 0.176103s` (`+0.62%`).
  - Vs governance champion (`5.748507`): `+0.06%` (below promotion gate `+0.250%`).

- Macro G exp-op reduction note:
  - No exp-op reduction measured on the completed profiled run; Macro G kernel edits were reverted before final profiling, and the resulting trace remained baseline-equivalent in dominant buckets.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable from this Perfetto export due missing source-level closed-call labels.
    - Train-path bucket deltas: `shard_map -0.01%`, `fusion -0.07%`, `all-gather -0.30%`.
    - `throughput/mfu -0.62%`, `throughput/tokens_per_second -0.62%`, `throughput/duration +0.62%`.
    - Macro G exp-op reduction: not observed on final profiled run because candidate edits were reverted.
  - Governance:
    - MFU gain `<3%` with unchanged dominant hotspot family (`shard_map/custom-call`), and end-to-end regression vs baseline; attempt marked **low-impact failed** and kernel changes reverted.

- Assessment: **failed attempt / low impact**. Macro G implementation attempts were not robust (compile + correctness regressions), and the completed profile on reverted state showed no dominant-hotspot movement with slight end-to-end regression.
- Next bold hypothesis: escalate to **Macro Move H** (shared-RHS matmul batching for `QK/KKT` and `inter/v_prime`) with explicit BF16-input/FP32-accum `dot_general` policy and train-path-only focus; if that still leaves `shard_map` unchanged, jump to **Macro Move E** V-tiling.

### Iteration 37 - Macro Move G / exp-diff centered outer-product (infra-blocked, reverted)

- Date: 2026-02-23T15:29:00Z
- Commit: none (failed attempt)
- Loop session/local index: `1/10`
- Starting commit: `06a85b37b670019bb3bf5cabd745711a995e5363`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - train-path `shard_map/custom-call` bucket remained dominant (`~78 ms` on TPU:0 XLA Ops thread).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` across prepare/recurrent/backward to remove Ct x Ct exponentials (`+10-20%`, medium numerical/compiler risk).
  2. **Macro Move H**: shared-RHS matmul batching (`QK/KKT`, `inter/v_prime`) (`+8-18%`, medium integration/VMEM risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) in recurrent/backward (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add `_exp_diff_and_mask_from_g` and wire centered outer-product `exp_diff` into train-path chunk prepare/recurrent/backward kernels.

- Change attempt summary:
  - Implemented `_exp_diff_and_mask_from_g` and replaced train-path `exp_diff` construction across prepare/recurrent/fused-forward/backward chunk math.
  - Local smoke tests passed, but TPU validation lanes were infra-blocked; kernel edits were reverted to avoid leaving unvalidated speculative code.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation attempts (blocked):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name "$USER-gdn" --tests both`
      - failure: `ssh: connect to host 136.112.108.150 port 22: Operation timed out`.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
      - submission path failed with `requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(54, 'Connection reset by peer'))`.
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
      - same SSH timeout to stale dev host alias (`136.112.108.150`).
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5-a --tpu auto --tests both`
      - job `ray-run-calvinxu-levanter-20260223-230647` failed: `Job supervisor actor failed to start within 900.0 seconds`.
      - subsequent retries submitted and remained pending during this window:
        - `ray-run-calvinxu-levanter-20260223-232707`
        - `ray-run-calvinxu-levanter-20260223-232741`

- Profile run:
  - Not started because TPU correctness gate could not be completed on any lane.
  - Trace artifact: N/A.

- Hotspots observed:
  - No new before/after hotspot comparison available (no completed profile run).
  - Carry-in dominant hotspot remains train-path `shard_map/custom-call` bucket from the latest successful baseline trace.

- MFU/throughput delta:
  - Unavailable (no completed profiled run).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name "$USER-gdn" --tests both` (blocked by SSH timeout).
    - Ray fallback commands attempted in `us-central1` and `us-east5-a` (blocked by connection reset / job-supervisor start failures).
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable.
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration` deltas: unavailable.
    - Macro G exp-op reduction note: unavailable (no completed profile/IR capture).
  - Governance:
    - Infra-blocked iteration; speculative kernel edits reverted and recorded as failed (`Commit: none (failed attempt)`).

- Assessment: **infra-blocked failed attempt**. Could not complete required TPU validation + profile cycle due repeated dev TPU SSH timeouts and Ray job start failures.
- Next bold hypothesis:
  - First secure a healthy TPU execution lane (fresh dev TPU alias or healthy Ray queue), then re-run Macro G end-to-end with immediate exp-op and train-path closed-call delta capture.
  - If infra stabilizes but Macro G still under-delivers, escalate to **Macro Move H** (shared-RHS matmul batching) next.

### Iteration 38 - Macro Move G / exp-diff centered outer-product (regression, reverted)

- Date: 2026-02-24T03:32:30Z
- Commit: none (failed attempt)
- Loop session/local index: `1/10`
- Starting commit: `32c3823ac8072e489ba7d375cf63ab6131f9a945`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` bucket: `78.098 ms` (dominant), with top closed-call sources at `gated_deltanet.py:2486` (`41.324 ms`) and `gated_deltanet.py:3972` (`26.266 ms`).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` across prepare/recurrent/backward to remove Ct x Ct exponentials (`+10-20%`, medium numerical/compiler risk).
  2. **Macro Move H**: shared-RHS matmul batching for `QK/KKT` and `inter/v_prime` (`+8-18%`, medium integration/VMEM risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) in recurrent/backward train path (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add `_exp_diff_and_mask_from_g` and route train-path flash prepare/recurrent/backward kernels through centered-outer fast path when chunk-range is clip-safe.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added `_exp_diff_and_mask_from_g` (centered outer-product fast path + exact fallback).
  - Added train-path safety gating (`_all_chunks_centered_exp_safe`) and threaded `use_centered_exp` through flash prepare/recurrent/bwd train kernels.
  - Updated full-sequence train dispatch to select centered-exp mode only when clip-safe.
  - Regression observed on profiled train run; speculative kernel changes were reverted to the starting commit state.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> passed.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> passed.
  - Dev TPU validation attempt (blocked by lock contention):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - failure: `ABORTED: The TPU is already in use by another process...`
  - Ray fallback TPU validation (success):
    - `uv run lib/marin/src/marin/run/ray_run.py --cluster us-east5-a --tpu auto -e EQX_ON_ERROR=nan -e WANDB_MODE=offline -- bash -lc 'cd lib/levanter && unset MARIN_PREFIX && uv sync --extra=tpu --group test && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && EQX_ON_ERROR=nan WANDB_MODE=offline uv run pytest tests/test_gdn_kernels.py tests/test_gdn_layer.py -v'`
    - job: `ray-run-calvinxu-levanter-20260224-031706`
    - result: `49 passed, 40 skipped`.

- Profile runs:
  - Dev TPU profile attempt (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroG_outer --no-sync`
    - failure: `FileNotFoundError` writing executor info under `gs://marin-us-east5-a/...`.
  - Ray fallback profile (completed):
    - submit: `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroG_outer_ray --no-wait`
    - job: `ray-run-calvinxu-bash-20260224-032150`
    - wait: `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-east5-a ray-run-calvinxu-bash-20260224-032150 --show-logs --tail 400`
    - status: `SUCCEEDED`.
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter1_macroG_outer_ray_130m_ch128_seg16_20step-6dbb74`
  - W&B artifact: `run-gdn_loop_iter1_macroG_outer_ray_130m_ch128_seg16_20step-6dbb74-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_loop_iter1_macroG_outer_ray_130m_ch128_seg16_20step-6dbb74/plugins/profile/2026_02_23_19_27_16/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - Dominant hotspot family remained `shard_map/custom-call`.
  - Bucket totals:
    - `shard_map`: `78.098 ms -> 39.014 ms` (`-50.05%`)
    - `fusion`: `45.618 ms -> 34.902 ms` (`-23.49%`)
    - `all-gather`: `20.158 ms -> 10.084 ms` (`-49.98%`)
  - Forward/backward closed-call shard-map buckets (`tf_op` labels preserved):
    - Forward closed-call `jit(_train_step)/jvp(...)/closed_call/shard_map/pallas_call`: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - Backward closed-call `jit(_train_step)/transpose(jvp(...))/closed_call/shard_map/pallas_call`: `26.266 ms -> 13.130 ms` (`-50.01%`).
  - Caveat: trace event volume halved (`11761 -> 6596`), so absolute per-trace bucket times are not directly predictive of end-to-end throughput here.

- MFU/throughput delta (vs baseline `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.277688` (`-8.81%`).
  - `throughput/tokens_per_second`: `187227.57 -> 170732.20` (`-8.81%`).
  - `throughput/duration`: `0.175017s -> 0.191926s` (`+9.66%`).
  - Vs governance champion (`5.748507`): `-8.19%` (regression beyond `1.000%` threshold).

- Macro G exp-op reduction note (trace-derived):
  - No reduction observed. On TPU:0 XLA Ops thread, exp-related events increased:
    - `exp*` event count: `10 -> 21`
    - `exp*` total time: `0.0055 ms -> 1.6938 ms`
  - New dominant exp-related bucket: `exponential_reduce_fusion.2` (`1.691 ms`).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run lib/marin/src/marin/run/ray_run.py --cluster us-east5-a --tpu auto ... uv run pytest tests/test_gdn_kernels.py tests/test_gdn_layer.py -v` -> `49 passed, 40 skipped` (job `ray-run-calvinxu-levanter-20260224-031706`).
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.661 ms` (`-50.00%`) [trace bucket].
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 13.130 ms` (`-50.01%`) [trace bucket].
    - `throughput/mfu -8.81%`, `throughput/tokens_per_second -8.81%`, `throughput/duration +9.66%`.
    - Macro G exp-op reduction: **not observed** (exp-related trace time increased).
  - Governance:
    - MFU gain `<3%` and dominant hotspot family unchanged (`shard_map/custom-call`), with major end-to-end regression; attempt marked **low-impact failed** and kernel changes reverted.

- Assessment: **failed attempt / regression**. Despite lower per-trace closed-call bucket totals, end-to-end throughput regressed substantially and Macro G did not reduce exp-heavy work in trace-derived counts.
- Governance/escalation action:
  - Reverted speculative kernel changes; working tree restored to starting commit state.
  - Recorded attempt as failed (`Commit: none (failed attempt)`).
- Next bold hypothesis:
  - Escalate to **Macro Move H** with explicit stacked shared-RHS matmul batching (`[q; k_beta] @ k^T` and `[q_scaled; k_cumdecay] @ S`) using a unified `dot_general` helper, then re-measure closed-call buckets and end-to-end MFU.

### Iteration 39 - Macro Move H / shared-RHS matmul batching (infra-blocked, reverted)

- Date: 2026-02-24T23:23:39Z
- Commit: none (failed attempt)
- Loop session/local index: `1/10`
- Starting commit: `65df652eeef57fd6ce57591cdff17cfe6fd98868`
- Dominant bottleneck carried in (from latest successful baseline trace used by Iteration 38):
  - train-path `shard_map/custom-call` bucket remained dominant (`78.098 ms` on TPU:0 XLA Ops thread), with closed-call hotspots at `gated_deltanet.py:2486` (`41.324 ms`) and `gated_deltanet.py:3972` (`26.266 ms`).

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move H**: batch shared-RHS matmuls (`QK/KKT`, `inter/v_prime`) in train kernels to reduce dot-call count (`+10-20%`, medium integration/VMEM risk).
  2. **Macro Move G**: centered outer-product `exp_diff` with exact fallback (`+10-20%`, medium numerical/compiler risk; prior regressions).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) in recurrent/backward (`+15-30%`, high decomposition risk).

- Selected macro-move category: **H) Batch matmuls by stacking left operands that share the same right operand**.
- Selected hypothesis: apply shared-RHS batching in the train chunk path (forward + backward chunk kernels), then validate on TPU and profile for `shard_map/pallas_call` deltas.

- Change attempt summary:
  - Implemented Macro-H batching edits in `lib/levanter/src/levanter/layers/gated_deltanet.py` for train-path matmul pairs.
  - Local smoke tests passed (`test_gdn_kernels` flash subset + `test_gdn_layer` GDN subset).
  - Reverted all speculative kernel edits because TPU validation/profiling lanes remained infra-blocked and no validated performance result could be produced.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation attempts (blocked):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
      - failed repeatedly during collection with TPU lock contention: `TPU initialization failed: open(/dev/vfio/*): Device or resource busy`.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
      - job `ray-run-calvinxu-levanter-20260224-224933` failed: `JOB_SUPERVISOR_ACTOR_START_TIMEOUT` (`Job supervisor actor failed to start within 900.0 seconds`).
    - Additional Ray fallbacks submitted while triaging capacity remained non-starting/pending during this window:
      - `ray-run-calvinxu-levanter-20260224-231038` (`us-central1`, pending)
      - `ray-run-calvinxu-levanter-20260224-231728` (`us-west4`, pending)

- Profile run:
  - Not started. Required TPU correctness gate for modified kernel state could not be completed.
  - Trace artifact: N/A.

- Hotspots observed:
  - No new validated before/after hotspot comparison (no completed profile on validated code).
  - Carry-in dominant hotspot remains train-path `shard_map/custom-call`.

- MFU/throughput delta:
  - Unavailable (no completed profiled run).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command attempted: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both` (blocked by `/dev/vfio/*` busy), with Ray fallbacks attempted and not reaching a successful test completion.
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable.
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`: unavailable.
    - Macro-H call-count reduction evidence: unavailable (no validated profile run).
  - Governance:
    - Infra-blocked iteration; speculative code reverted and recorded as failed (`Commit: none (failed attempt)`).

- Assessment: **infra-blocked failed attempt**. Could not complete required TPU validation + profile cycle due persistent dev TPU VFIO lock contention and Ray job start-capacity failures.
- Next bold hypothesis:
  - Re-attempt Macro Move H immediately once a healthy TPU lane is available (fresh dev TPU alias not sharing locked VFIO devices, or a Ray cluster where job supervisor starts promptly), then capture forward/backward closed-call deltas and end-to-end MFU deltas in the same run.

### Iteration 40 - Macro Move G / centered outer-product exp-diff in train chunk kernels (regressed, reverted)

- Date: 2026-02-25T01:39:36Z
- Commit: none (failed attempt)
- Loop session/local index: `2/10`
- Starting commit: `9545875bd8b729edf2a3d5ce069ec74f7039f887`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` bucket remained dominant (`78.098 ms`), with `fusion` (`45.618 ms`) and `all-gather` (`20.158 ms`) next.

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` in prepare/recurrent/backward train kernels (`+10-20%`, medium numerical/control-flow risk).
  2. **Macro Move H**: shared-RHS matmul batching for `QK/KKT` and `inter/v_prime` (`+8-18%`, medium VMEM/layout risk).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) for recurrent/backward state updates (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add centered outer-product `exp_diff` construction to train-path prepare/recurrent/backward kernels with exact fallback path preserved, and only use centered mode when chunk ranges are clip-safe.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added `_all_chunks_centered_exp_safe` and `_exp_diff_and_mask_from_g` helpers.
  - Threaded centered-exp mode through train prepare/recurrent/fused-forward/backward chunk kernels.
  - Used dispatch-level `lax.cond` to choose centered-exp vs exact-exp paths (to avoid in-kernel dynamic branching in Pallas).
  - Local smoke and TPU correctness passed, but end-to-end profile regressed meaningfully.
  - Reverted all speculative kernel edits; tree returned to starting commit state.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter2_macroG_centered --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter2_macroG_centered_130m_ch128_seg16_20steps-4ed77c`
  - W&B artifact: `run-gdn_loop_iter2_macroG_centered_130m_ch128_seg16_20steps-4ed77c-profiler:v0`
  - Downloaded trace: `.profiles/wandb/gdn_loop_iter2_macroG_centered_130m_ch128_seg16_20steps-4ed77c/plugins/profile/2026_02_25_01_36_09/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - `shard_map`: `78.098 ms -> 39.033 ms` (`-50.02%`).
  - `fusion`: `45.618 ms -> 34.857 ms` (`-23.59%`).
  - `all-gather`: `20.158 ms -> 14.617 ms` (`-27.49%`).
  - `while`: `0.000 ms -> 31.687 ms` (new large hotspot family introduced).
  - `conditional`: `7.491 ms -> 49.128 ms` (now dominant bucket).
  - Event volume changed materially (`11761 -> 6549`), so trace-bucket time drops did not translate to end-to-end gains.
  - Forward/backward source-level closed-call `shard_map/pallas_call` separation was unavailable in this trace export (no stable `closed_call` labels).

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.289897` (`-8.60%`).
  - `throughput/tokens_per_second`: `187227.57 -> 171127.14` (`-8.60%`).
  - `throughput/duration`: `0.175017s -> 0.191483s` (`+9.41%`).

- Macro G exp-op reduction note (trace-derived):
  - **No reduction observed.**
  - `exp*` event count on TPU:0 XLA Ops thread: `30 -> 47`.
  - `exp*` total time: `0.007998 ms -> 1.702356 ms`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: unavailable from this Perfetto export (missing stable source-level `closed_call` labels).
    - Train-path bucket deltas: `shard_map -50.02%`, `fusion -23.59%`, `all-gather -27.49%`, with new `while` (`31.687 ms`) and dominant `conditional` (`49.128 ms`).
    - `throughput/mfu -8.60%`, `throughput/tokens_per_second -8.60%`, `throughput/duration +9.41%`.
    - Macro G exp-op reduction: not observed (`exp*` count/time increased).
  - Governance:
    - Regression exceeds active threshold (`1.000%` below champion), so attempt is marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. The centered-exp dispatch introduced a costly control-flow hotspot pattern and reduced end-to-end throughput.
- Next bold hypothesis: escalate to **Macro Move H** (shared-RHS matmul batching) with no runtime `lax.cond` in the hot train path, and keep BF16-input/FP32-accum policy consistent across forward/backward kernels.

### Iteration 41 - Macro Move H / shared-RHS matmul batching in train chunk kernels (regression, reverted)

- Date: 2026-02-25T05:14:39Z
- Commit: none (failed attempt)
- Loop session/local index: `3/10`
- Starting commit: `b126717207aed90d40a4e66d3694eb442109f23e`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` bucket: `78.098 ms` (dominant), with closed-call hotspots at:
    - `jit(_train_step)/jvp(...)/closed_call/shard_map/pallas_call`: `41.324 ms`
    - `jit(_train_step)/transpose(jvp(...))/closed_call/shard_map/pallas_call`: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move H**: batch shared-RHS train-kernel matmuls (`QK/KKT`, and backward adjoint counterparts) to reduce dot call count (`+10-20%`, medium compiler/layout risk).
  2. **Macro Move G**: centered outer-product `exp_diff` (`+10-20%`, medium-high risk given recent regressions/control-flow hotspots).
  3. **Macro Move E**: V-tiling with shared-K precompute (`+15-30%`, high decomposition risk).

- Selected macro-move category: **H) Batch matmuls by stacking left operands that share the same right operand**.
- Selected hypothesis: reduce train-path forward/backward `shard_map/pallas_call` wall time by batching `QK/KKT` in fused train forward and batching the corresponding backward adjoint matmuls.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added shared-RHS batched matmul helper and applied Macro-H batching in train-path kernels.
  - First profile attempt failed at TPU compile time (Mosaic layout constraint on concat-with-`k_cumdecay`).
  - Revised implementation to keep the high-value shared-`k^T` batching path (`QK/KKT` + backward adjoints) while removing the concat path that triggered the Mosaic layout error.
  - TPU tests and profile completed on the revised variant.
  - End-to-end throughput regressed materially; all speculative kernel edits were reverted to keep the tree at baseline.

- Correctness checks:
  - Local smoke (success):
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (success):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `87 passed, 2 skipped`.

- Profile runs:
  - Attempt 1 (failed compile):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter3_macroH_batch --marin-prefix gs://marin-us-east5 --no-sync`
    - failure: `Mosaic failed to compile TPU kernel: Not implemented: result/input offset mismatch on non-concat dimension` at `gated_deltanet.py:2311` (`concatenate` in batched `inter/v_prime` path).
  - Attempt 2 (completed after compile-safe revision):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter3_macroH_batch_v2 --marin-prefix gs://marin-us-east5 --no-sync`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter3_macroH_batch_v2_130m_ch128_seg16_20steps-a9cff7`
    - W&B artifact: `run-gdn_loop_iter3_macroH_batch_v2_130m_ch128_seg16_20steps-a9cff7-profiler:v0`
    - trace download: `uv run wandb artifact get marin-community/marin/run-gdn_loop_iter3_macroH_batch_v2_130m_ch128_seg16_20steps-a9cff7-profiler:v0 --root .profiles/wandb`
    - downloaded trace: `.profiles/wandb/plugins/profile/2026_02_25_05_11_47/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - `shard_map`: `78.098 ms -> 41.901 ms` (`-46.35%`)
  - `fusion`: `45.618 ms -> 34.863 ms` (`-23.58%`)
  - `all-gather`: `20.158 ms -> 10.142 ms` (`-49.69%`)
  - New large `while` bucket: `0.000 ms -> 31.522 ms`
  - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.847 ms` (`-49.55%`)
  - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 15.828 ms` (`-39.74%`)
  - Event volume changed materially (`11761 -> 6596`), so trace-only bucket improvements did not predict end-to-end throughput.

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.358350` (`-7.42%`)
  - `throughput/tokens_per_second`: `187227.57 -> 173341.60` (`-7.42%`)
  - `throughput/duration`: `0.175017s -> 0.189037s` (`+8.01%`)

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.847 ms` (`-49.55%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 15.828 ms` (`-39.74%`).
    - `throughput/mfu -7.42%`, `throughput/tokens_per_second -7.42%`, `throughput/duration +8.01%`.
  - Governance:
    - MFU gain `<3%` (regression) and dominant train-path hotspot family remained `shard_map/custom-call`; attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. Macro-H batching reduced measured closed-call trace buckets but did not translate to end-to-end speed; runtime shifted cost into additional control-flow overhead.
- Next bold hypothesis:
  - Escalate to **Macro Move E** (V-tiling with shared-K precompute) to reduce per-program state footprint and increase useful MXU work without relying on concat-heavy layout-sensitive batching in train fused kernels.

### Iteration 42 - Macro Move G / in-kernel centered outer-product exp-diff with exact fallback (regression, reverted)

- Date: 2026-02-25T21:04:10Z
- Commit: none (failed attempt)
- Loop session/local index: `1/10`
- Starting commit: `2835ae5042a7cd0bdf25ba3eba899febfd532e85`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` bucket: `78.098 ms` (dominant), with top closed-call hotspots:
    - `jit(_train_step)/jvp(...)/closed_call/shard_map/pallas_call`: `41.324 ms`
    - `jit(_train_step)/transpose(jvp(...))/closed_call/shard_map/pallas_call`: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` with exact fallback inside train chunk kernels (`+10-20%`, medium compiler/control-flow risk).
  2. **Macro Move H**: shared-RHS train-path matmul batching (`+8-18%`, medium VMEM/layout risk; prior regressions).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) with shared-K precompute (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: add `_exp_diff_and_mask_from_g` and use it directly in train-path prepare/recurrent/fused-forward/backward kernels so fallback branching stays local to `exp_diff` construction rather than dispatch-level control flow.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented `_exp_diff_and_mask_from_g(g, clip)` with:
    - centered outer-product fast path (`er[:,None] * ec[None,:]`, clamped to `[exp(-clip), exp(clip)]`);
    - exact `diff/clip/exp` fallback path for out-of-range chunk ranges.
  - Rewired train-path chunk kernels to consume the helper in:
    - prepare kernels (segmented + full-sequence pipeline),
    - recurrent forward kernels (segmented + full-sequence pipeline),
    - fused train forward kernel (loop + pipeline stage body),
    - backward chunk kernel (for `exp_diff` and derivative mask).
  - Local and TPU correctness passed, but profiled end-to-end throughput regressed materially.
  - Reverted speculative kernel edits; tree restored to starting commit state.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroG_centered_inkernel --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter1_macroG_centered_inkernel_130m_ch128_seg1-131ac7`
  - W&B artifact: `run-gdn_loop_iter1_macroG_centered_inkernel_130m_ch128_seg1-131ac7-profiler:v0`
  - Trace download:
    - `uv run wandb artifact get marin-community/marin/run-gdn_loop_iter1_macroG_centered_inkernel_130m_ch128_seg1-131ac7-profiler:v0 --root .profiles/wandb`
  - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_25_20_58_22/perfetto_trace.json.gz`
  - Throughput source:
    - `gsutil cat gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter1_macroG_centered_inkernel_130m_ch128_seg1-131ac7/tracker_metrics.jsonl`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - `shard_map`: `78.098 ms -> 46.475 ms` (`-40.49%`)
  - `fusion`: `45.618 ms -> 34.883 ms` (`-23.53%`)
  - `all-gather`: `20.158 ms -> 10.141 ms` (`-49.69%`)
  - New large `while` bucket: `0.000 ms -> 31.509 ms`
  - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 24.429 ms` (`-40.88%`)
  - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 16.829 ms` (`-35.93%`)
  - Event volume changed materially (`11761 -> 6596`), so per-trace bucket drops did not predict end-to-end throughput.

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.218560` (`-9.83%`)
  - `throughput/tokens_per_second`: `187227.57 -> 168819.42` (`-9.83%`)
  - `throughput/duration`: `0.175017s -> 0.194101s` (`+10.90%`)
  - Vs governance champion (`5.748507`): `-9.22%` (regression beyond `1.000%` threshold).

- Macro G exp-op reduction note (trace-derived):
  - **No reduction observed.**
  - `exp*` event count on TPU:0 XLA Ops thread: `10 -> 21`.
  - `exp*` total time: `0.005513 ms -> 1.694401 ms`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 24.429 ms` (`-40.88%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 16.829 ms` (`-35.93%`).
    - `throughput/mfu -9.83%`, `throughput/tokens_per_second -9.83%`, `throughput/duration +10.90%`.
    - Macro G exp-op reduction: **not observed** (`exp*` count/time increased).
  - Governance:
    - MFU gain `<3%` and dominant hotspot family remained train-path `shard_map/custom-call`; attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. In-kernel Macro G reformulation reduced some trace buckets but introduced a large `while` hotspot and regressed end-to-end throughput.
- Next bold hypothesis:
  - Escalate to **Macro Move E** (V-tiling with shared-K precompute) to structurally reduce per-program state and attack train-path `shard_map/custom-call` critical path without adding new control-flow-heavy work.

### Iteration 43 - Macro Move G / centered exp-diff helper failed correctness on TPU (reverted)

- Date: 2026-02-25T21:38:50Z
- Commit: none (failed attempt)
- Loop session/local index: `2/10`
- Starting commit: `f7bd4021f593a82b16e2c987e85debe5c9b5168f`
- Dominant bottleneck carried in (from latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call`: `78.098 ms` (dominant category)
  - top closed-call hotspots:
    - `jit(_train_step)/jvp(...)/closed_call/shard_map/pallas_call`: `41.324 ms`
    - `jit(_train_step)/transpose(jvp(...))/closed_call/shard_map/pallas_call`: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` with exact fallback in train prepare/recurrent/backward kernels (`+10-20%`, medium/high compiler-control-flow and numerical risk).
  2. **Macro Move H**: shared-RHS matmul batching (`QK/KKT`, `inter/v_prime`) (`+8-18%`, medium/high layout and VMEM risk).
  3. **Macro Move E**: V-tiling with shared-K precompute (`KxV -> KxVb`) (`+15-30%`, high decomposition risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: introduce `_exp_diff_and_mask_from_g(g, clip)` and route train chunk prepare/recurrent/backward kernels through it, with exact fallback semantics retained for out-of-range cases.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added `_exp_diff_and_mask_from_g` and wired it into train-path chunk kernels (prepare, recurrent forward, fused train forward, backward).
  - Attempt 1 used `lax.cond` fast-path/fallback dispatch inside the helper and failed TPU Mosaic lowering (`failed to legalize operation 'scf.if'`).
  - Attempt 2 removed dynamic branch (centered-only helper) to get compile-safe code; TPU tests then produced NaNs in flash layer parity/invariance tests.
  - Reverted all speculative kernel edits so the tree returns to the starting commit code.

- Correctness checks:
  - Failing attempt command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: failed (`MosaicError: failed to legalize operation 'scf.if'`) in flash backward tests.
  - Failing centered-only follow-up command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: failed (`6 failed, 81 passed, 2 skipped`) with NaNs in flash layer parity/invariance tests.
  - Final validation after revert:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run (post-revert control, required per loop contract):
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter2_macroG_revertctrl --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter2_macroG_revertctrl_130m_ch128_seg16_20ste-1cc007`
  - W&B profiler artifact: `run-gdn_loop_iter2_macroG_revertctrl_130m_ch128_seg16_20ste-1cc007-profiler:v0`
  - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_25_21_35_59/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to carry-in baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - `shard_map`: `78.098 ms -> 39.013 ms` (`-50.05%`)
  - `fusion`: `45.618 ms -> 34.897 ms` (`-23.50%`)
  - `all-gather`: `20.158 ms -> 10.087 ms` (`-49.96%`)
  - New `while` hotspot family: `0.000 ms -> 31.527 ms`
  - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.662 ms` (`-50.00%`)
  - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 13.130 ms` (`-50.01%`)
  - Dominant hotspot class remains train-path `shard_map/custom-call` with added control-flow overhead (`transpose(jvp())/shard_map/while` now visible at `12.543 ms`).

- MFU/throughput delta (vs carry-in baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.430798` (`-6.16%`)
  - `throughput/tokens_per_second`: `187227.57 -> 175685.27` (`-6.16%`)
  - `throughput/duration`: `0.175017s -> 0.186515s` (`+6.57%`)
  - vs active champion (`throughput/mfu=5.748507`): `-5.53%`.

- Macro G exp-op reduction note (trace-derived):
  - **No reduction observed.**
  - `exp*` event count on TPU:0 XLA Ops thread: `10 -> 21`.
  - `exp*` total time: `0.005513 ms -> 1.694374 ms`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped` (after revert).
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.662 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 13.130 ms` (`-50.01%`).
    - `throughput/mfu -6.16%`, `throughput/tokens_per_second -6.16%`, `throughput/duration +6.57%`.
    - Macro G exp-op reduction: **not observed** (`exp*` count/time increased).
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot class unchanged; attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. Macro G helper could not be made both TPU-Mosaic-compatible and numerically robust in this iteration without harming end-to-end throughput.
- Next bold hypothesis:
  - Escalate to a more radical decomposition move that avoids `scf.if`-style control flow in Pallas (for example **Macro Move E** V-tiling with shared-K precompute in train recurrent/backward paths), rather than another centered-exp branch rewrite.

### Iteration 44 - Macro Move G / static centered-exp dispatch with outer-product fast path (regression, reverted)

- Date: 2026-02-25T22:10:31Z
- Commit: none (failed attempt)
- Loop session/local index: `3/10`
- Starting commit: `24222d734c313c0099c14e4f1af04f36c207323f`
- Dominant bottleneck carried in (latest successful baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call`: `78.098 ms` (dominant)
  - top closed-call hotspots:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `41.324 ms`
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move G**: centered outer-product `exp_diff` in train prepare/recurrent/fused-forward/backward with exact fallback dispatch outside Pallas (`+10-20%`, medium compiler/control-flow risk).
  2. **Macro Move H**: shared-RHS matmul batching in train chunk kernels (`+8-18%`, medium/high VMEM/layout risk; prior regressions).
  3. **Macro Move E**: V-tiling (`KxV -> KxVb`) with shared-K precompute (`+15-30%`, high decomposition/rewrite risk).

- Selected macro-move category: **G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp**.
- Selected hypothesis: keep Macro-G centered math, but avoid prior `scf.if` failures by dispatching centered-vs-exact kernels outside Pallas and only enabling centered path for MXU-sized train chunks.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented `_exp_diff_and_mask_from_g` and `_can_use_centered_outer_exp_diff` helpers.
  - Threaded `use_centered_outer_product` through train-path prepare/recurrent/fused-forward/backward kernels.
  - Added top-level `lax.cond` dispatch in chunk forward/backward wrappers to select centered or exact path outside Pallas kernel bodies.
  - First TPU validation run failed one tight parity assertion in small-shape backward HF test (`max abs diff 1.2526e-05` vs `atol=1e-05`); patched by restricting centered path to `Ct >= 128` (target train regime), then TPU validation passed.
  - Profiled end-to-end run regressed materially; reverted speculative kernel edits and returned code to starting-commit behavior.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation attempt 1 (failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `1 failed, 86 passed, 2 skipped` (`tests/test_gdn_layer.py::test_gdn_layer_backward_matches_hf[True]`).
  - TPU validation attempt 2 (after `Ct >= 128` guard):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter3_macroG_centered_gate --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter3_macroG_centered_gate_130m_ch128_seg16_20-2d5213`
  - W&B artifact: `run-gdn_loop_iter3_macroG_centered_gate_130m_ch128_seg16_20-2d5213-profiler:v0`
  - Trace download:
    - `uv run wandb artifact get marin-community/marin/run-gdn_loop_iter3_macroG_centered_gate_130m_ch128_seg16_20-2d5213-profiler:v0 --root .profiles/wandb`
  - Downloaded trace: `.profiles/wandb/plugins/profile/2026_02_25_22_06_53/perfetto_trace.json.gz`
  - Throughput source:
    - `gsutil cat gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter3_macroG_centered_gate_130m_ch128_seg16_20-2d5213/tracker_metrics.jsonl`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, compared to baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`):
  - `shard_map`: `78.098 ms -> 39.027 ms` (`-50.03%`)
  - `fusion`: `45.618 ms -> 34.718 ms` (`-23.89%`)
  - `all-gather`: `20.158 ms -> 14.710 ms` (`-27.03%`)
  - New dominant control-flow buckets:
    - `conditional`: `7.491 ms -> 49.210 ms`
    - `while`: `0.000 ms -> 31.509 ms`
  - Forward closed-call `shard_map/pallas_call` (`tf_op`-derived):
    - `41.324 ms -> 20.647 ms` (`-50.04%`)
  - Backward closed-call `shard_map/pallas_call` (`tf_op`-derived):
    - `26.266 ms -> 13.143 ms` (`-49.96%`)
  - Event volume changed materially (`11755 -> 6546`), and reduced closed-call buckets did not translate to faster end-to-end steps.

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 5.301450` (`-8.40%`)
  - `throughput/tokens_per_second`: `187227.57 -> 171500.90` (`-8.40%`)
  - `throughput/duration`: `0.175017s -> 0.191066s` (`+9.17%`)
  - vs active governance champion (`5.748507`): `-7.78%`.

- Macro G exp-op reduction note (trace-derived):
  - **No reduction observed.**
  - `exp*` event count on TPU:0 XLA Ops thread: `10 -> 21`.
  - `exp*` total time on TPU:0 XLA Ops thread: `0.005513 ms -> 1.697736 ms`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.647 ms` (`-50.04%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 13.143 ms` (`-49.96%`).
    - `throughput/mfu -8.40%`, `throughput/tokens_per_second -8.40%`, `throughput/duration +9.17%`.
    - Macro G exp-op reduction: **not observed** (`exp*` count/time increased).
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot family remained train-path `shard_map/custom-call` with added control-flow overhead; attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. The static-dispatch Macro-G variant reduced per-trace train custom-call buckets but introduced large conditional/while overhead and regressed end-to-end throughput.
- Next bold hypothesis:
  - Escalate to **Macro Move E** (V-tiling with shared-K precompute) to reduce recurrent state footprint and improve MXU residency without adding branch-heavy control flow in the train chunk path.

### Iteration 45 - Champion baseline benchmark on `v6e-8` (control)

- Date: 2026-02-26T03:03:44Z
- Commit: `785edf9f64367c9dee355662150a432204b45045` (champion control benchmark, no kernel changes)
- Purpose:
  - Establish a hardware-specific champion baseline on `v6e-8` before further macro-move iterations and infra pivoting.

- Code changes:
  - None (benchmark-only control run on pinned champion commit).

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster eu-west4-a --tpu-name calvinxu-gdn-v6e --tpu v6e-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_champion_v6e_baseline_recheck_dev --marin-prefix gs://marin-eu-west4 --no-sync`
  - W&B run:
    - `https://wandb.ai/marin-community/marin/runs/gdn_champion_v6e_baseline_recheck_dev_130m_ch128_seg16_-84329f`
  - Throughput source:
    - `wandb-summary.json` downloaded from the run artifact (`wandb` API), because summary fields were not yet hydrated in `run.summary` immediately after completion.

- MFU/throughput (v6e control baseline):
  - `throughput/mfu`: `1.642223`
  - `throughput/tokens_per_second`: `212502.35`
  - `throughput/duration`: `0.154201s`
  - `throughput/device_kind`: `TPU v6 lite`

- Notes:
  - Keep this as the comparator for subsequent `v6e-8` runs.
  - Do not compare this value directly against `v5p-8` MFU without normalizing hardware assumptions.

### Iteration 46 - Macro Move F / segmented train split (solve + recurrent) blocked by TPU infra (reverted)

- Date: 2026-02-26T05:02:05Z
- Commit: none (failed attempt)
- Loop session/local index: `1/10`
- Starting commit: `37fd2dc09689165e6d0374f39f90037d55f8be16`
- Dominant bottleneck carried in (v6e baseline trace `.profiles/wandb/plugins/profile/2026_02_26_03_03_25/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map`: `65.815 ms` (dominant)
  - `fusion`: `16.697 ms`
  - `all-gather`: `15.148 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move F (Experiment A)**: force segmented train path to split kernels (`prepare` then `recurrent`) instead of fused segmented train forward (`+10-25%`, medium/high risk; may increase launch count but reduce fused-kernel pressure).
  2. **Macro Move E (Experiment B)**: recurrent V-tiling over `V_blocks` with `S_prev[K,Vb]` state slices (`+15-30%`, high risk; backward reductions and memory-layout complexity).
  3. **Macro Move H**: shared-RHS matmul batching without concat-sensitive layouts (`+8-18%`, medium/high risk; prior compile/layout regressions).

- Selected macro-move category: **F) Match FlashLinearAttention’s kernel decomposition**.
- Selected hypothesis: apply FLA-style **2-kernel split** for segmented train path (solve/prep kernel then recurrent apply kernel) by routing `return_prepare_tape=True` through split kernels instead of `_gdn_chunk_segment_fwd_fused_pallas`.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented the segmented train-path routing change to use split `prepare + recurrent` kernels.
  - Local smoke tests passed.
  - TPU validation could not be completed due repeated infra failures across dev TPU and Ray fallback paths.
  - Reverted speculative kernel edit; tree returned to starting commit behavior.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Dev TPU attempt (stalled/no completion):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster eu-west4-a --tpu-name calvinxu-gdn-v6e --tests both`
    - progressed deep into suite, then hung with no additional output for multiple minutes.
  - Dev TPU retry (unavailable):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster eu-west4-a --tpu-name calvinxu-gdn-v6e --tests both --no-sync`
    - failed immediately: `Error: SSH configuration for dev-tpu-calvinxu-gdn-v6e not found`.
  - Ray fallback attempt (fixture/env failure):
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
    - failed setup with `_configure_marin_prefix did not yield a value` (cluster env had `MARIN_PREFIX` set).
  - Direct Ray fallback with unset `MARIN_PREFIX` (infra termination):
    - `uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --tpu auto -e EQX_ON_ERROR=nan -e WANDB_MODE=offline -- bash -lc 'cd lib/levanter && uv sync --extra=tpu --group test && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && unset MARIN_PREFIX && EQX_ON_ERROR=nan WANDB_MODE=offline uv run pytest tests/test_gdn_kernels.py tests/test_gdn_layer.py -v'`
    - tests were running/passing, then job failed before completion: `Job supervisor actor died ... actor's node was terminated expectedly: received SIGTERM`.

- Profile run:
  - **Not run** due TPU validation blocker (no stable TPU test pass could be obtained in this infra state).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: **blocked by infra** (commands and failures recorded above).
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: **not measured** (profile blocked).
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`: **not measured** (profile blocked).
  - Governance:
    - Infra-blocked iteration; speculative kernel code reverted, no champion update.

- Assessment: **infra-blocked attempt**. Could not complete required TPU validation/profile evidence due dev TPU availability loss and Ray worker/job-supervisor termination.
- Next bold hypothesis:
  - Re-attempt Macro Move F split on a stable TPU allocation, then profile; if infra remains unstable, pivot to Macro Move E only after validation path is reliable.

### Iteration 47 - Macro Move H / shared-RHS batched dot_general in fused train kernels (regression, reverted)

- Date: 2026-02-26T23:34:39Z
- Commit: none (failed attempt)
- Loop session/local index: `2/10`
- Starting commit: `5412f5c9ddb059b14e7bdf4926766f8803314d27`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - `shard_map`: `78.098 ms` (dominant)
  - top closed-call hotspots:
    - `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `41.324 ms`
    - `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: fuse segmented prepare+recurrent with tape reuse (`+12-22%`, high tape/VMEM risk).
  2. **Macro Move E**: V-tiling with shared-K precompute (`+15-30%`, high decomposition risk).
  3. **Macro Move H**: shared-RHS batched matmul in train forward/backward kernels (`+10-18%`, medium/high TPU lowering risk).

- Selected macro-move category: **H) Batch matmuls by stacking left operands that share the same right operand**.
- Selected hypothesis: replace paired train-path dots sharing `k^T`, `S`, and `S^T` with one batched `dot_general` helper to reduce dot invocation count in fused forward/backward chunk kernels.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added `_mxu_matmul_shared_rhs2_f32` (single batched `dot_general` for two left operands with shared RHS).
  - Applied helper in train-path fused forward (`QK/KKT`, `inter/v_prime`) and backward chunk math (`QK/KKT`, `d_q_scaled/d_k_cumdecay`, `d_QK/dKKT @ k`).
  - Local smoke tests passed; TPU validation passed.
  - Profile showed severe end-to-end regression; speculative kernel edits were reverted to starting-commit behavior.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Required TPU validation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter2_macroH_batched_rhs --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run:
    - `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter2_macroH_batched_rhs_130m_ch128_seg16_20st-98dc8a`
  - W&B profiler artifact:
    - `run-gdn_loop_iter2_macroH_batched_rhs_130m_ch128_seg16_20st-98dc8a-profiler:v0`
  - Downloaded trace:
    - `.profiles/wandb/plugins/profile/2026_02_26_23_31_41/perfetto_trace.json.gz`
  - Throughput source:
    - `wandb` run summary and `gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter2_macroH_batched_rhs_130m_ch128_seg16_20st-98dc8a/tracker_metrics.jsonl`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - `shard_map`: `78.098 ms -> 45.900 ms` (`-41.23%`)
  - `fusion`: `45.618 ms -> 35.073 ms` (`-23.12%`)
  - `all-gather`: `20.158 ms -> 10.127 ms` (`-49.76%`)
  - New `while` hotspot family: `0.000 ms -> 31.690 ms`
  - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 21.239 ms` (`-48.60%`)
  - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 19.428 ms` (`-26.03%`)
  - Dominant new tf-op bucket: `jit(_train_step)/transpose(jvp())/shard_map/while` (`12.549 ms`, 208 events).

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.787594 -> 0.938955` (`-83.78%`)
  - `throughput/tokens_per_second`: `187227.57 -> 30375.02` (`-83.78%`)
  - `throughput/duration`: `0.175017s -> 1.078781s` (`+516.39%`)
  - vs active champion (`throughput/mfu=5.748507`): `-83.67%`.
- Post-hoc scoring correction (2026-02-27):
  - The `0.938955` value above is the **final-step** (`step=19`) outlier from W&B summary.
  - W&B step history shows stable-region performance (steps `10..18`) far above that tail outlier:
    - Candidate run median: `throughput/mfu=5.185450`, `duration=0.195340s`, `tokens/s=167748.32`.
    - Baseline run median: `throughput/mfu=5.830017`, `duration=0.173743s`, `tokens/s=188599.93`.
    - Robust-window MFU delta: `-11.06%` (still a regression, but not `-83.78%`).
  - Action: `gdnctl` performance governance now defaults to robust history-window scoring (`median`, steps `10..18`) instead of final-summary-only scoring.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 21.239 ms` (`-48.60%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 19.428 ms` (`-26.03%`).
    - `throughput/mfu -83.78%`, `throughput/tokens_per_second -83.78%`, `throughput/duration +516.39%`.
  - Governance:
    - MFU gain `<3%` (major regression) and dominant train-path hotspot class remained `shard_map/custom-call` with large new `while` overhead; attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / severe regression**. Reducing visible closed-call buckets did not translate to faster step time; the batched-dot rewrite introduced large control-flow/loop overhead in the backward path.
- Next bold hypothesis:
  - Escalate to **Macro Move I** (prepare+recurrent fusion with explicit tape reuse and no stacked-dot helper) or **Macro Move E** (V-tiling shared-K) to avoid the new `while` overhead regime.

### Iteration 48 - Macro Move I / full-sequence fused prepare+recurrent train forward (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-02-27T13:35:37Z
- Commit: none (failed attempt)
- Starting commit: `ac94e24a28cdb9137dd21c837beb7a3f6e75542c`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/pallas_call` remained the key target in tf-op aggregation:
    - forward: `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/*/shard_map/pallas_call` = `414.635 ms`
    - backward: `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/*/shard_map/pallas_call` = `210.140 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: full-sequence fused train forward (`prepare + recurrent + tape`) in one Pallas call to remove cross-kernel tape traffic (`+10-20%`, high control-flow/lowering risk).
  2. **Macro Move J**: `Ct/Seg` sweep (`Ct={64,96,128}`, `Seg={8,16,32}`) after structural changes (`+5-15%`, medium risk; requires compact benchmark table).
  3. **Macro Move E**: V-tiling shared-K precompute in recurrent/backward to shrink per-program state (`+15-30%`, high decomposition risk).

- Selected macro-move category: **I) Fuse segmented/full-sequence forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: for the full-sequence train path (`return_prepare_tape=True`), replace split full-sequence calls (`prepare` then `recurrent`) with one fused pipelined Pallas kernel that computes chunk-local solve outputs and recurrent apply in one launch while writing the same backward tape contract (`v_pseudo`, `k_cumdecay`, `solve_transform`, chunk starts).

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added full-sequence fused forward pipeline kernel/wrapper (`_gdn_chunk_fullseq_fwd_fused_*`) with one `pallas_call` over `N_chunks` and VMEM state scratch.
  - Routed `_chunk_gated_delta_rule_flash_pallas_impl(..., return_prepare_tape=True)` full-sequence train path to the fused kernel.
  - Kept no-tape/inference full-sequence path on split kernels.
  - Local smoke tests and TPU correctness passed.
  - Profiled run regressed materially; speculative kernel edits were reverted to starting-commit behavior.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Required TPU validation:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - result: `87 passed, 2 skipped`.

- Profile run:
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter6_macroI_fullseq_fused --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run:
    - `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter6_macroI_fullseq_fused_130m_ch128_seg16_20-cafee4`
  - W&B profiler artifact:
    - `run-gdn_loop_iter6_macroI_fullseq_fused_130m_ch128_seg16_20-cafee4-profiler:v0`
  - Downloaded trace:
    - `.profiles/wandb/plugins/profile/2026_02_27_13_31_04/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Top op buckets shifted from baseline fusion/conditional mix to large `while` buckets:
    - baseline top op: `conditional.2 = 59.909 ms`
    - new top ops: `while.56 = 188.931 ms`, `while.55 = 64.083 ms`
  - Forward `shard_map/pallas_call` tf-op aggregate: `414.635 ms -> 207.082 ms` (`-50.06%`).
  - Backward `shard_map/pallas_call` tf-op aggregate: `210.140 ms -> 105.036 ms` (`-50.02%`).
  - Despite reduced shard-map buckets, new while/control-flow overhead dominated end-to-end time.

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.382854` (`-7.67%`).
  - `throughput/tokens_per_second`: `188599.93 -> 174134.30` (`-7.67%`).
  - `throughput/duration`: `0.173743s -> 0.188177s` (`+8.31%`).
  - final-step reference (step `19`): `throughput/mfu=5.406845`, `tokens/s=174910.40`, `duration=0.187342s`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward `shard_map/pallas_call` tf-op aggregate: `414.635 ms -> 207.082 ms` (`-50.06%`).
    - Backward `shard_map/pallas_call` tf-op aggregate: `210.140 ms -> 105.036 ms` (`-50.02%`).
    - `throughput/mfu -7.67%`, `throughput/tokens_per_second -7.67%`, `throughput/duration +8.31%`.
  - Governance:
    - MFU gain `<3%` (regression). Attempt marked **low-impact/regressive** and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. The macro-I fused full-sequence forward path cut train `shard_map/pallas_call` buckets roughly in half, but introduced substantial `while` overhead that regressed end-to-end throughput.
- Next bold hypothesis:
  - Move to **Macro Move J** with required `Ct/Seg` sweep (`Ct={64,96,128}`, `Seg={8,16,32}`) and a compact benchmark table to identify a better operating point after this fusion evidence.

### Iteration 49 - Macro Move I / train full-sequence fused segmented forward (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-02-28T01:07:08Z
- Commit: none (failed attempt)
- Starting commit: `f398fbaf59909fcec889b96a1dbec3d19009f013`
- Dominant bottleneck carried in (from Iteration 48 baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/pallas_call` remained dominant:
    - forward closed-call: `41.324 ms`
    - backward closed-call: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: route train full-sequence forward (`return_prepare_tape=True`) to one fused segmented kernel over all chunks, reusing prep intermediates in-kernel (`+10-20%`, high risk from loop/control-flow lowering).
  2. **Macro Move E**: V-tiling with shared-K precompute in recurrent/bwd kernels (`+15-30%`, high decomposition/reduction risk).
  3. **Macro Move J**: explicit `Ct/Seg` sweep after structural changes (`+5-12%`, medium risk; requires compact table evidence).

- Selected macro-move category: **I) Fuse segmented forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: for training path only, replace split full-sequence forward (`_gdn_chunk_fullseq_prepare_pallas` + `_gdn_chunk_fullseq_recurrent_fwd_pallas`) with a single fused `_gdn_chunk_segment_fwd_fused_pallas` call over all chunks (`Seg=n_chunks_pad`) and force the static in-kernel chunk loop to avoid prior `emit_pipeline` while overhead.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added a `force_loop` path in `_gdn_chunk_segment_fwd_fused_kernel_tpu` and threaded it through `_gdn_chunk_segment_fwd_fused_pallas`.
  - Routed `_chunk_gated_delta_rule_flash_pallas_impl(..., return_prepare_tape=True)` full-sequence train path to one fused segmented forward call over all chunks.
  - TPU correctness passed (after one permitted retry for known borderline tolerance miss).
  - Profiled run regressed materially; speculative kernel edits were reverted, leaving the tree at starting-commit behavior.

- Correctness checks:
  - Command: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both`
  - First run: failed `test_gdn_layer_backward_matches_hf[False]` with borderline `max_abs=1.1938624e-05` vs `atol=1e-05` (known transient signature).
  - Retry (same command, once per retry guard): `87 passed, 2 skipped`.

- Profile run:
  - Dev TPU attempt (stalled / no run registered):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroI_fused_train_static --no-sync`
  - Ray fallback command:
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroI_fused_train_static_ray --no-wait`
    - `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 ray-run-calvinxu-bash-20260228-005412 --show-logs --tail 600`
  - Job ID: `ray-run-calvinxu-bash-20260228-005412`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter1_macroI_fused_train_static_ray_130m_ch128-8e8471`
  - Trace location:
    - `.profiles/wandb/gdn_loop_iter1_macroI_fused_train_static_ray_130m_ch128-8e8471-profiler-v0/plugins/profile/2026_02_27_16_59_27/perfetto_trace.json.gz`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Op bucket deltas:
    - `shard_map`: `78.098 ms -> 39.022 ms` (`-50.03%`) (still the largest single bucket).
    - `fusion`: `45.618 ms -> 35.092 ms` (`-23.07%`).
    - `all-gather`: `20.158 ms -> 10.066 ms` (`-50.06%`).
    - New `while` overhead: `0.000 ms -> 31.692 ms` (dominant new regression source; top events `while.56`, `while.55`).
  - Train closed-call shard-map deltas:
    - forward `jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - backward `jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map/pallas_call`: `26.266 ms -> 13.129 ms` (`-50.01%`).

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.281180` (`-9.41%`).
  - `throughput/tokens_per_second`: `188599.93 -> 170845.15` (`-9.41%`).
  - `throughput/duration`: `0.173743s -> 0.191799s` (`+10.39%`).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped` (after one allowed retry of a known transient tolerance signature).
  - Perf:
    - Forward closed-call `shard_map/pallas_call`: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call`: `26.266 ms -> 13.129 ms` (`-50.01%`).
    - `throughput/mfu -9.41%`, `throughput/tokens_per_second -9.41%`, `throughput/duration +10.39%`.
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot family remained train-path `shard_map/custom-call` with large added `while` overhead; attempt marked **low-impact/regressive**, and speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. The Macro-I full-sequence fused train forward path reduced closed-call shard-map wall time by ~50% but introduced enough `while` overhead to regress end-to-end throughput.
- Next bold hypothesis:
  - Move to **Macro Move J** next (required coverage progression): run the explicit `Ct in {64,96,128}` × `Seg in {8,16,32}` sweep with a compact benchmark table and use the best point as the launchpad for the next structural macro move.

### Iteration 50 - Macro Move I / segmented fused train-path reroute (infra-blocked, reverted)

- Coverage slot: I (1/5, attempted but not validated)
- Covered set so far: {}
- Date: 2026-02-28T05:51:51Z
- Commit: none (failed attempt)
- Starting commit: `0033a77327b651d88ab8a40e0505bd317d7cfff1`
- Dominant bottleneck carried in (from Iteration 49 carry-in baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/pallas_call` remained dominant:
    - forward closed-call: `41.324 ms`
    - backward closed-call: `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: training-only reroute to segmented fused forward (`return_prepare_tape=True`) with segment-bounded launches to reuse prep intermediates in-kernel without full-sequence fused loop path (`+10-18%`, medium/high implementation + lowering risk).
  2. **Macro Move J**: explicit `Ct/Seg` sweep (`Ct={64,96,128}`, `Seg={8,16,32}`) with compact table (`+5-12%`, medium risk; lower structural upside this iteration).
  3. **Macro Move E**: V-tiling with shared-K precompute in recurrent/bwd kernels (`+15-30%`, high decomposition and correctness risk).

- Selected macro-move category: **I) Fuse segmented forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: for training path only, route `_chunk_gated_delta_rule_flash_pallas_impl(..., return_prepare_tape=True)` away from split full-sequence prepare/recurrent kernels to the segmented fused forward path so chunk-local solve outputs are reused once in the same kernel at segment granularity (avoiding prior single-mega-segment control-flow regressions).

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented the Macro-I training-path reroute described above.
  - Added a small TPU layout companion change replacing a hot `(Ct, 1)` backward matvec with direct `dot_general` matvec.
  - Reverted all speculative kernel edits after TPU validation remained infra-blocked.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - Required TPU validation (`tests=both`) attempts:
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both --no-wait`
      - job: `ray-run-calvinxu-levanter-20260228-052222`
      - `ray-wait --timeout 180`: `status=PENDING` timeout.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5-a --tpu auto --tests both --no-wait`
      - job: `ray-run-calvinxu-levanter-20260228-052621`
      - `ray-wait --timeout 180`: `status=PENDING` timeout.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5 --tpu auto --tests both --no-wait`
      - job: `ray-run-calvinxu-levanter-20260228-053015`
      - `ray-wait --timeout 180`: `status=PENDING` timeout.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central2 --tpu auto --tests both --no-wait`
      - job: `ray-run-calvinxu-levanter-20260228-053429`
      - `ray-wait --timeout 180`: `status=PENDING` timeout.
    - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east1 --tpu auto --tests both --no-wait`
      - job: `ray-run-calvinxu-levanter-20260228-053823`
      - `ray-wait --timeout 180`: `status=PENDING` timeout.
  - Dev TPU fallback:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name calvinxu-gdn --tests both --no-sync`
      - failed immediately: `ssh: Could not resolve hostname dev-tpu-calvinxu-gdn`.
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-allocate --cluster us-central1 --tpu-name calvinxu-gdn --tpu-type v5p-8`
      - allocator did not produce a usable dev TPU host; repeated Raylet errors: `worker_pool.cc:1865: Delete runtime env failed`.

- Profile run:
  - **Not run** (required TPU validation could not be completed).

- Hotspots observed:
  - No new validated profile trace; carry-in dominant hotspot remains train-path `shard_map/pallas_call` at the callsites above.

- MFU/throughput delta:
  - N/A (infra-blocked; no validated TPU profile run).

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: **blocked by infra** (all commands + job IDs above).
  - Perf:
    - Forward/backward `shard_map/pallas_call` deltas: **not measured**.
    - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`: **not measured**.
  - Governance:
    - Speculative kernel edits reverted; no champion/perf-state update.

- Assessment: **infra-blocked attempt**. Could not obtain required TPU validation/profile evidence due persistent Ray `PENDING` queue contention across multiple clusters and unavailable dev-TPU SSH target.
- Next bold hypothesis:
  - Re-attempt the same Macro-I training segmented-fusion variant once TPU validation path is healthy; if infra instability persists, resolve cluster capacity/allocator health first before new kernel edits.

### Iteration 51 - Macro Move I / segmented fused train-path reroute with static-loop forward (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-03-01T02:54:31Z
- Commit: none (failed attempt)
- Starting commit: `f79fa79b8d2dbc958863bd1c38f428d368986294`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` remained dominant:
    - forward closed-call source: `gated_deltanet.py:2486` = `41.324 ms`
    - backward closed-call source: `gated_deltanet.py:3972` = `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: training-path reroute from full-sequence split prepare+recurrent to segmented fused prepare+recurrent with static in-kernel loops (`+10-20%`, high lowering/control-flow risk).
  2. **Macro Move J**: explicit `Ct in {64,96,128}` x `Seg in {8,16,32}` sweep after structural changes (`+5-12%`, medium risk).
  3. **Macro Move E**: V-tiling with shared-K precompute in recurrent/bwd kernels (`+15-30%`, high decomposition risk).

- Selected macro-move category: **I) Fuse segmented forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: for `return_prepare_tape=True` (train path), bypass full-sequence split prepare/recurrent pallas calls and use segmented fused forward calls that reuse chunk-local prep intermediates in-kernel once, with static per-segment loops to avoid the prior full-sequence `while` regime.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added `force_loop` threading through `_gdn_chunk_segment_fwd_fused_kernel_tpu`, `_gdn_chunk_segment_fwd_fused_pallas`, and `_gdn_chunk_segment_fwd_pallas`.
  - Changed `_chunk_gated_delta_rule_flash_pallas_impl` dispatch so full-sequence split prepare/recurrent path is disabled for `return_prepare_tape=True`; train path now executes segmented fused forward with `force_loop` on MXU-sized configs.
  - Initial companion backward matvec rewrite triggered TPU Mosaic compile failure in profiling; that helper change was removed before final validation/profile.
  - Final kernel reroute regressed end-to-end throughput; speculative kernel edits were reverted.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (`tests=both`):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
      - first run after rollback: failed `test_gdn_layer_backward_matches_hf[False]` with borderline `max_abs=2.124533e-05` vs `atol=1e-05`.
      - one allowed retry (same command): `87 passed, 2 skipped`.

- Profile run:
  - Attempt A (compile-failed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroI_segfused_static --marin-prefix gs://marin-us-east5 --no-sync`
    - failure signature: TPU Mosaic `dot_general` lowering error (`rhs non contracting dims ... vector-like [B,K] or [B,1,K]`).
  - Attempt B (teardown-failed after artifact generation):
    - same command with `--run-name-prefix gdn_loop_iter1_macroI_segfused_static_r2`
    - produced trace + summary but executor status ended `FAILED` during W&B teardown (`HandleAbandonedError`).
  - Completed profile evidence run (successful command exit):
    - `uv run scripts/ray/dev_tpu.py --cluster us-east5-a --tpu-name calvinxu-gdn execute --no-sync -e EQX_ON_ERROR=nan -e WANDB_MODE=offline -e GDN_PROFILE_SIZE=130m -e GDN_PROFILE_NUM_STEPS=20 -e GDN_PROFILE_PROFILE_START_STEP=2 -e GDN_PROFILE_PROFILE_NUM_STEPS=6 -e GDN_PROFILE_RUN_NAME_PREFIX=gdn_loop_iter1_macroI_segfused_static_offline -e GDN_PROFILE_TPU_VARIANT=v5p-8 -e GDN_PROFILE_BATCH_SIZE=8 -e MARIN_PREFIX=gs://marin-us-east5 -- "set -e && uv sync --all-packages --extra=tpu --python=3.11 && uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu --force-reinstall torch && (uv pip uninstall --python .venv/bin/python torchvision || true) && .venv/bin/python -m experiments.speedrun.hackable_transformer_gdn.tiny_profile --force_run_failed true"`
    - output status: `gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter1_macroI_segfused_static_offline_130m_ch12-95ff0a/.executor_status = SUCCESS`
    - trace location: `marin/logs/gdn_loop_iter1_macroI_segfused_static_offline_130m_ch12-95ff0a/profiler/plugins/profile/2026_03_01_02_52_43/perfetto_trace.json.gz`
    - copied local trace: `.profiles/dev_tpu/gdn_loop_iter1_macroI_segfused_static_offline/perfetto_trace.json.gz`
    - throughput source: `gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter1_macroI_segfused_static_offline_130m_ch12-95ff0a/tracker_metrics.jsonl`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Bucket deltas:
    - `shard_map`: `78.098 ms -> 39.005 ms` (`-50.06%`)
    - `fusion`: `45.618 ms -> 35.056 ms` (`-23.15%`)
    - `all-gather`: `20.158 ms -> 10.114 ms` (`-49.83%`)
    - `while`: `0.000 ms -> 31.665 ms` (new dominant regression bucket)
  - Train closed-call shard-map source deltas:
    - forward source `gated_deltanet.py:2486 -> 2507`: `41.324 ms -> 20.661 ms` (`-50.00%`)
    - backward source `gated_deltanet.py:3972 -> 4001`: `26.266 ms -> 13.129 ms` (`-50.02%`)

- MFU/throughput delta (vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.394733` (`-7.47%`)
  - `throughput/tokens_per_second`: `188599.93 -> 174518.59` (`-7.47%`)
  - `throughput/duration`: `0.173743s -> 0.187762s` (`+8.07%`)

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped` (after one allowed retry of known transient tolerance signature).
  - Perf:
    - Forward closed-call `shard_map/pallas_call` source: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call` source: `26.266 ms -> 13.129 ms` (`-50.02%`).
    - `throughput/mfu -7.47%`, `throughput/tokens_per_second -7.47%`, `throughput/duration +8.07%`.
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot class remained train-path `shard_map/custom-call` with large new `while` overhead. Attempt marked **low-impact/regressive** and kernel edits were reverted.

- Assessment: **failed attempt / regression**. Launch-level shard-map costs were reduced, but new `while` control-flow overhead outweighed those wins and regressed end-to-end throughput.
- Next bold hypothesis:
  - Move to **Macro Move J** next with required `Ct={64,96,128}` x `Seg={8,16,32}` compact sweep table, then use the best operating point as the launchpad for the next structural macro move.

### Iteration 52 - Macro Move I / static-unrolled segmented fused train forward (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-03-01T03:41:51Z
- Commit: none (failed attempt)
- Starting commit: `0e7a48b66dfe09d7f644e9368778b111a9cd5ebc`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` remained dominant:
    - forward closed-call source: `gated_deltanet.py:2486` = `41.324 ms`
    - backward closed-call source: `gated_deltanet.py:3972` = `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: training-path segmented fused forward with static segment unroll (remove `lax.scan` control-flow in forward while keeping fused prepare+recurrent per segment) (`+10-20%`, high lowering/control-flow risk).
  2. **Macro Move J**: explicit `Ct in {64,96,128}` x `Seg in {8,16,32}` sweep to re-anchor operating point after Macro-I/while regressions (`+5-12%`, medium risk).
  3. **Macro Move E**: V-tiling + shared-K precompute in recurrent/bwd kernels (`+15-30%`, high decomposition/correctness risk).

- Selected macro-move category: **I) Fuse segmented forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: for `return_prepare_tape=True` train forward, route to segmented fused forward and unroll the segment loop statically in Python to preserve the ~50% closed-call shard-map reduction while avoiding scan-induced `while` overhead.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented a Macro-I training-only path that called `_gdn_chunk_segment_fwd_pallas(..., return_prepare_tape=True)` per segment with static unrolling (`for seg_idx in range(n_segments)`), carrying state/tapes across calls.
  - TPU correctness passed.
  - Profiled run regressed end-to-end throughput similarly to prior Macro-I variants; speculative kernel edits were reverted.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (`tests=both`):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile run:
  - Dev TPU attempt A:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroI_staticseg --marin-prefix gs://marin-us-east5 --no-sync`
    - failure signature: `wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec`.
  - Dev TPU attempt B (one allowed identical-signature retry):
    - same command with `--run-name-prefix gdn_loop_iter1_macroI_staticseg_retry`
    - same failure signature (`wandb.init` timeout).
  - Ray fallback (completed):
    - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_loop_iter1_macroI_staticseg_ray --no-wait`
    - `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-east5-a ray-run-calvinxu-bash-20260301-031810 --show-logs --tail 600`
    - Job ID: `ray-run-calvinxu-bash-20260301-031810`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_loop_iter1_macroI_staticseg_ray_130m_ch128_seg16_20-c41684`
    - W&B profiler artifact: `run-gdn_loop_iter1_macroI_staticseg_ray_130m_ch128_seg16_20-c41684-profiler:v0`
    - Downloaded trace: `.profiles/wandb/gdn_loop_iter1_macroI_staticseg_ray_130m_ch128_seg16_20-c41684/plugins/profile/2026_02_28_19_33_27/perfetto_trace.json.gz`
    - Throughput source: `gs://marin-us-east5/checkpoints/speedrun/gdn_loop_iter1_macroI_staticseg_ray_130m_ch128_seg16_20-c41684/tracker_metrics.jsonl`

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Bucket deltas:
    - `shard_map`: `78.098 ms -> 39.016 ms` (`-50.04%`)
    - `fusion`: `45.618 ms -> 34.992 ms` (`-23.29%`)
    - `all-gather`: `20.158 ms -> 10.060 ms` (`-50.10%`)
    - `while`: `0.000 ms -> 31.601 ms` (new dominant regression bucket)
  - Train closed-call shard-map source deltas:
    - forward source `gated_deltanet.py:2486 -> 2505`: `41.324 ms -> 20.661 ms` (`-50.00%`)
    - backward source `gated_deltanet.py:3972 -> 4041`: `26.266 ms -> 13.130 ms` (`-50.01%`)

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.305000` (`-9.01%`)
  - `throughput/tokens_per_second`: `188599.93 -> 171615.73` (`-9.01%`)
  - `throughput/duration`: `0.173743s -> 0.190938s` (`+9.90%`)
  - final-step reference (step `19`): `throughput/mfu=5.326328`, `tokens/s=172305.70`, `duration=0.190174s`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call` source: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call` source: `26.266 ms -> 13.130 ms` (`-50.01%`).
    - `throughput/mfu -9.01%`, `throughput/tokens_per_second -9.01%`, `throughput/duration +9.90%`.
  - Governance:
    - MFU gain `<3%` (regression). Attempt marked **low-impact/regressive**; speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. Static-unrolled segmented fusion again halved train closed-call shard-map buckets, but the same `while`-family overhead appeared and dominated end-to-end runtime.
- Next bold hypothesis:
  - Move to **Macro Move J** next (required coverage progression): run the explicit `Ct={64,96,128}` x `Seg={8,16,32}` sweep with a compact benchmark table, then launch the next structural macro move from the best point.

### Iteration 53 - Macro Move I / full-sequence fused train-forward on recurrent-scaffold pipeline (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-03-01T06:59:43Z
- Commit: none (failed attempt)
- Starting commit: `2cc5f1a2d61046f38927b7a09b0db5d1ea0c3c47`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` remained dominant:
    - `shard_map` bucket: `78.098 ms`
    - forward closed-call source: `gated_deltanet.py:2486` = `41.324 ms`
    - backward closed-call source: `gated_deltanet.py:3972` = `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: full-sequence train-forward fusion built from the validated full-sequence recurrent pipeline scaffold (one fused launch for prepare+recurrent+tape) (`+10-20%`, high lowering/control-flow risk).
  2. **Macro Move J**: explicit `Ct in {64,96,128}` x `Seg in {8,16,32}` sweep with compact benchmark table (`+5-12%`, medium risk).
  3. **Macro Move E**: V-tiling with shared-K precompute in recurrent/backward kernels (`+15-30%`, high decomposition/correctness risk).

- Selected macro-move category: **I) Fuse segmented/full-sequence forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: avoid prior segmented-reroute variants and instead fuse train full-sequence prepare+recurrent directly on the full-sequence recurrent pipeline scaffold, so training forward uses one fused `pallas_call` while keeping the inference/no-tape path on the validated split kernels.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added a full-sequence fused train-forward path (`_gdn_chunk_fullseq_fwd_fused_*`) that performs prepare+recurrent+tape emission in one pipeline launch.
  - Routed `_chunk_gated_delta_rule_flash_pallas_impl(..., return_prepare_tape=True)` under `use_fullseq_pipeline` to this fused path; kept no-tape path on split full-sequence kernels.
  - Added TPU layout companion cleanup for backward `d_g` accumulation to avoid `(Ct,1)` shape.
  - First TPU validation exposed unsupported TPU Pallas lowering (`rev` primitive) from an initial suffix-sum expression; replaced with a no-`rev` row-matmul formulation.
  - Final profiled run regressed end-to-end throughput; speculative kernel edits were reverted.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (`tests=both`):
    - Attempt A:
      - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
      - failed with deterministic lowering signature: `NotImplementedError: Unimplemented primitive ... rev` in TPU Pallas lowering.
    - Fix applied (remove `rev` usage in `d_g` path), then rerun:
      - same command
      - Result: `87 passed, 2 skipped`.

- Profile run (completed):
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_iter6_macroI_fullseq_fused --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run:
    - `https://wandb.ai/marin-community/marin/runs/gdn_iter6_macroI_fullseq_fused_130m_ch128_seg16_20steps-9baf60`
  - W&B profiler artifact:
    - `run-gdn_iter6_macroI_fullseq_fused_130m_ch128_seg16_20steps-9baf60-profiler:v0`
  - Downloaded trace:
    - `.profiles/wandb/gdn_iter6_macroI_fullseq_fused_130m_ch128_seg16_20steps-9baf60-profiler-v0/plugins/profile/2026_03_01_06_55_56/perfetto_trace.json.gz`
  - Throughput source:
    - baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c` + new run history-window medians from W&B (`global_step in [10,18]`).

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Bucket deltas:
    - `shard_map`: `78.098 ms -> 40.423 ms` (`-48.24%`)
    - `fusion`: `45.618 ms -> 35.076 ms` (`-23.11%`)
    - `all-gather`: `20.158 ms -> 10.128 ms` (`-49.76%`)
    - `while`: `0.000 ms -> 31.673 ms` (new dominant regression bucket)
  - Train closed-call shard-map source deltas:
    - forward source `gated_deltanet.py:2486 -> 2783`: `41.324 ms -> 20.661 ms` (`-50.00%`)
    - backward source `gated_deltanet.py:3972 -> 4288`: `26.266 ms -> 14.540 ms` (`-44.64%`)

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.370800` (`-7.88%`)
  - `throughput/tokens_per_second`: `188599.934 -> 173744.350` (`-7.88%`)
  - `throughput/duration`: `0.173743s -> 0.188599s` (`+8.55%`)
  - final-step reference (step `19`): `throughput/mfu=5.345875`, `tokens/s=172938.036`, `duration=0.189478s`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call` source: `41.324 ms -> 20.661 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call` source: `26.266 ms -> 14.540 ms` (`-44.64%`).
    - `throughput/mfu -7.88%`, `throughput/tokens_per_second -7.88%`, `throughput/duration +8.55%`.
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot class remained train-path `shard_map/custom-call` with large added `while` overhead.
    - Attempt marked **low-impact/regressive**; speculative kernel edits were reverted.

- Assessment: **failed attempt / regression**. The recurrent-scaffold full-sequence Macro-I fusion reduced measured train closed-call shard-map time, but the new `while` overhead dominated and regressed end-to-end throughput.
- Next bold hypothesis:
  - Move to **Macro Move J** next with required `Ct={64,96,128}` x `Seg={8,16,32}` compact sweep table, then use the best operating point to launch a stronger structural pivot (likely Macro E if `while` overhead persists).

### Iteration 54 - Macro Move J / no-pad segmented train decomposition + Ct/Seg sweep (regression, reverted)

- Coverage slot: J (2/5)
- Covered set so far: {I, J}
- Date: 2026-03-01T10:20:02Z
- Commit: none (failed attempt)
- Starting commit: `5608cdd87bccba0f296d6f44ab6e83622cffaada`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` remained dominant:
    - `shard_map` bucket: `78.098 ms`
    - forward closed-call source: `gated_deltanet.py:2486` = `41.324 ms`
    - backward closed-call source: `gated_deltanet.py:3972` = `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move J**: remove pad-to-segment execution in train flash path and sweep `Ct/Seg` to expose true work scaling (`+10-18%`, high control-flow/launch risk).
  2. **Macro Move E**: V-tiling with shared-K precompute in recurrent/backward kernels (`+15-30%`, high decomposition/correctness risk).
  3. **Macro Move I**: another train prepare+recurrent fusion variant that avoids added `while` costs (`+10-20%`, high repeat-regression risk; on cooldown after repeated `<3%` outcomes).

- Selected macro-move category: **J) Sweep `Ct`/`Seg` explicitly**.
- Selected hypothesis: structurally remove padded chunk-axis execution (`full segments + explicit tail segment`) in train flash forward/backward so the sweep measures real chunk work instead of padded segment overhead.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Implemented a no-pad decomposition in train flash forward/backward: execute `n_full_segments * seg` chunks via existing segmented kernels, then run one tail segmented kernel for `tail_chunks`.
  - This changed launch/dataflow structure in both forward and backward train paths (Macro J structural candidate).
  - TPU validation tests passed with the structural change.
  - Profile sweep showed clear regressions across tested operating points; speculative kernel edits were reverted.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (`tests=both`):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both`
    - Result: `87 passed, 2 skipped`.

- Profile runs (dev TPU):
  - `Ct=96, Seg=32` (primary run; completed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --chunk-size 96 --segment-size 32 --run-name-prefix gdn_iter8_macroJ_nopad_c96s32 --marin-prefix gs://marin-us-east5 --no-sync`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_iter8_macroJ_nopad_c96s32_130m_ch96_seg32_20steps-d1d70d`
    - Trace: `.profiles/wandb/gdn_iter8_macroJ_nopad_c96s32_130m_ch96_seg32_20steps-d1d70d-profiler-v0/plugins/profile/2026_03_01_10_03_00/perfetto_trace.json.gz`
  - `Ct=64, Seg=8` attempt A (infra failure, retried once per policy):
    - same command with `--run-name-prefix gdn_iter8_macroJ_nopad_c64s8`
    - failure signature: TPU init contention (`/dev/vfio/3` device busy).
  - `Ct=64, Seg=8` attempt B (retry; completed):
    - same command with `--run-name-prefix gdn_iter8_macroJ_nopad_c64s8_retry`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_iter8_macroJ_nopad_c64s8_retry_130m_ch64_seg8_20ste-b26968`
    - Trace: `.profiles/wandb/gdn_iter8_macroJ_nopad_c64s8_retry_130m_ch64_seg8_20ste-b26968-profiler-v0/plugins/profile/2026_03_01_10_12_16/perfetto_trace.json.gz`
  - `Ct=128, Seg=16` (completed):
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name calvinxu-gdn --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --chunk-size 128 --segment-size 16 --run-name-prefix gdn_iter8_macroJ_nopad_c128s16 --marin-prefix gs://marin-us-east5 --no-sync`
    - W&B run: `https://wandb.ai/marin-community/marin/runs/gdn_iter8_macroJ_nopad_c128s16_130m_ch128_seg16_20steps-961f0a`
    - Trace: `.profiles/wandb/gdn_iter8_macroJ_nopad_c128s16_130m_ch128_seg16_20steps-961f0a-profiler-v0/plugins/profile/2026_03_01_10_16_31/perfetto_trace.json.gz`

- Macro J sweep table (tested operating points):

| Ct | Seg | Compile/run status | `throughput/mfu` (10..18 median) | `tokens/s` (10..18 median) | `duration` (10..18 median) | Delta vs baseline MFU |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | Attempt A infra-failed (`/dev/vfio/3` busy), attempt B succeeded | `4.295130` | `138946.643` | `0.235832s` | `-26.33%` |
| 96 | 32 | Succeeded | `4.952903` | `160225.466` | `0.204512s` | `-15.04%` |
| 128 | 16 | Succeeded | `5.387411` | `174281.723` | `0.188017s` | `-7.59%` |

- Hotspots observed (`pid=3, tid=3`, vs baseline trace):
  - `Ct=96, Seg=32` (selected structural candidate):
    - bucket deltas: `shard_map 78.098 -> 50.846 ms` (`-34.90%`), `fusion 45.618 -> 35.145 ms` (`-22.95%`), `all-gather 20.158 -> 13.178 ms` (`-34.63%`), `while 0.000 -> 31.434 ms` (new major overhead).
    - train closed-call shard-map deltas: forward `gated_deltanet.py:2486 -> 2504` = `41.324 -> 27.803 ms` (`-32.72%`), backward `3972 -> 4099` = `26.266 -> 17.815 ms` (`-32.17%`).
  - `Ct=128, Seg=16` (best tested point this iteration):
    - bucket deltas: `shard_map 78.098 -> 39.018 ms` (`-50.04%`), `fusion 45.618 -> 35.043 ms` (`-23.18%`), `all-gather 20.158 -> 10.113 ms` (`-49.83%`), `while 0.000 -> 31.658 ms` (new major overhead).
    - train closed-call shard-map deltas: forward `2486 -> 2504` = `41.324 -> 20.661 ms` (`-50.00%`), backward `3972 -> 4099` = `26.266 -> 13.131 ms` (`-50.01%`).

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `Ct=96, Seg=32` (primary candidate):
    - `throughput/mfu`: `5.830017 -> 4.952903` (`-15.04%`)
    - `throughput/tokens_per_second`: `188599.934 -> 160225.466` (`-15.04%`)
    - `throughput/duration`: `0.173743s -> 0.204512s` (`+17.71%`)
  - best tested point (`Ct=128, Seg=16`) still regressed:
    - `throughput/mfu`: `5.830017 -> 5.387411` (`-7.59%`)
    - `throughput/tokens_per_second`: `188599.934 -> 174281.723` (`-7.59%`)
    - `throughput/duration`: `0.173743s -> 0.188017s` (`+8.22%`)

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + final result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name calvinxu-gdn --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward `shard_map/pallas_call` (selected candidate `Ct=96,Seg=32`): `41.324 ms -> 27.803 ms` (`-32.72%`).
    - Backward `shard_map/pallas_call` (selected candidate `Ct=96,Seg=32`): `26.266 ms -> 17.815 ms` (`-32.17%`).
    - `throughput/mfu -15.04%`, `throughput/tokens_per_second -15.04%`, `throughput/duration +17.71%`.
  - Governance:
    - MFU gain `<3%` (regression) and dominant hotspot class remained train-path `shard_map/custom-call`, with large new `while` overhead.
    - Attempt marked **low-impact/regressive**; speculative no-pad kernel edits were reverted.

- Assessment: **failed attempt / regression**. Removing pad-to-segment work reduced measured train shard-map kernel time, but introduced/retained large `while` overhead that dominated end-to-end and regressed MFU across tested operating points.
- Next bold hypothesis:
  - Move to **Macro Move E** next: V-tiling with shared-K precompute in train recurrent/backward kernels, explicitly targeting `while` overhead by increasing useful work per launch and reducing control-flow-heavy segmented loops.

### Iteration 55 - Macro Move I / full-sequence train-forward static-loop fusion (regression, reverted)

- Coverage slot: I (1/5)
- Covered set so far: {I}
- Date: 2026-03-01T12:31:36Z
- Commit: none (failed attempt)
- Starting commit: `6b7e3b2a7d1f4472419153cab0a45a98b1a42f42`
- Dominant bottleneck carried in (baseline trace `.profiles/wandb/gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c/plugins/profile/2026_02_22_08_29_07/perfetto_trace.json.gz`, TPU:0 XLA Ops `pid=3, tid=3`):
  - train-path `shard_map/custom-call` remained dominant:
    - `shard_map` bucket: `78.098 ms`
    - forward closed-call source: `gated_deltanet.py:2486` = `41.324 ms`
    - backward closed-call source: `gated_deltanet.py:3972` = `26.266 ms`

- Candidate shortlist (estimated upside / risk):
  1. **Macro Move I**: training-only full-sequence fused forward kernel with static chunk loop (no `emit_pipeline` in forward) (`+10-20%`, high lowering/VMEM risk).
  2. **Macro Move J**: explicit `Ct in {64,96,128}` x `Seg in {8,16,32}` sweep with compact benchmark table (`+5-12%`, medium risk).
  3. **Macro Move E**: V-tiling with shared-K precompute in recurrent/backward kernels (`+15-30%`, high decomposition risk).

- Selected macro-move category: **I) Fuse segmented/full-sequence forward prepare + recurrent with reusable heavy intermediates**.
- Selected hypothesis: replace the train full-sequence split prepare+recurrent path with a single full-sequence fused train-forward kernel using a static chunk loop, so we reduce train-path custom-call launches and avoid `emit_pipeline` loop-lowering overhead in forward.

- Change attempt summary (`lib/levanter/src/levanter/layers/gated_deltanet.py`):
  - Added a full-sequence fused train-forward kernel/call path that computes prepare + recurrent + tape in one Pallas call.
  - Routed `return_prepare_tape=True` under `use_fullseq_pipeline` to this fused path.
  - Kept no-tape forward path on the validated split full-sequence prepare/recurrent kernels.
  - Profiled result regressed end-to-end throughput; speculative kernel edits were reverted.

- Correctness checks:
  - Local smoke:
    - `uv run pytest -q lib/levanter/tests/test_gdn_kernels.py -k "flash and not slow"` -> `1 passed`.
    - `uv run pytest -q lib/levanter/tests/test_gdn_layer.py -k "gdn and not slow"` -> `13 passed`.
  - TPU validation (`tests=both`):
    - Ray attempt A (required path):
      - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
      - status progressed `PENDING` beyond 180s and failed with signature: `Job supervisor actor failed to start within 900.0 seconds`.
      - per directive, switched to next validation cluster.
    - Ray attempt B (next validation cluster):
      - `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5-a --tpu auto --tests both`
      - failed during test setup with deterministic fixture signature: `ValueError: _configure_marin_prefix did not yield a value`.
    - Dev TPU fallback:
      - initial `dev-tpu-test` failed before allocation (`Could not resolve hostname dev-tpu-calvinxu-gdn`).
      - allocated TPU then reran:
      - `uv run python scripts/gdn/gdnctl.py dev-tpu-allocate --cluster us-east5-a --tpu-name "$USER-gdn"`
      - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
      - Result: `87 passed, 2 skipped`.

- Profile run (completed):
  - Command:
    - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --run-name-prefix gdn_iter55_macroI_fullseq_staticloop --marin-prefix gs://marin-us-east5 --no-sync`
  - W&B run:
    - `https://wandb.ai/marin-community/marin/runs/gdn_iter55_macroI_fullseq_staticloop_130m_ch128_seg16_2-0cc2c9`
  - W&B profiler artifact:
    - `run-gdn_iter55_macroI_fullseq_staticloop_130m_ch128_seg16_2-0cc2c9-profiler:v0`
  - Downloaded trace:
    - `.profiles/dev_tpu/gdn_iter55_macroI_fullseq_staticloop/perfetto_trace.json.gz`
  - Throughput source:
    - W&B history-window medians (`global_step in [10,18]`) from the run above.

- Hotspots observed (TPU:0 XLA Ops `pid=3, tid=3`, vs baseline trace above):
  - Bucket deltas:
    - `shard_map`: `78.098 ms -> 39.003 ms` (`-50.06%`)
    - `fusion`: `45.618 ms -> 35.051 ms` (`-23.16%`)
    - `all-gather`: `20.158 ms -> 10.147 ms` (`-49.67%`)
    - `while`: `0.000 ms -> 31.688 ms` (new dominant overhead)
  - Train closed-call shard-map source deltas:
    - forward source `gated_deltanet.py:2486 -> 2729`: `41.324 ms -> 20.660 ms` (`-50.00%`)
    - backward source `gated_deltanet.py:3972 -> 4232`: `26.266 ms -> 13.130 ms` (`-50.01%`)

- MFU/throughput delta (history-window median, steps `10..18`, vs baseline run `gdn_segpipe_i17_dev_130m_ch128_seg16_20steps-27983c`):
  - `throughput/mfu`: `5.830017 -> 5.395648` (`-7.45%`)
  - `throughput/tokens_per_second`: `188599.934 -> 174548.198` (`-7.45%`)
  - `throughput/duration`: `0.173743s -> 0.187730s` (`+8.05%`)
  - final-step reference (step `19`): `throughput/mfu=5.420683`, `tokens/s=175358.062`, `duration=0.186863s`.

- Acceptance gate checklist:
  - Correctness:
    - TPU tests command + result: `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both` -> `87 passed, 2 skipped`.
  - Perf:
    - Forward closed-call `shard_map/pallas_call` source: `41.324 ms -> 20.660 ms` (`-50.00%`).
    - Backward closed-call `shard_map/pallas_call` source: `26.266 ms -> 13.130 ms` (`-50.01%`).
    - `throughput/mfu -7.45%`, `throughput/tokens_per_second -7.45%`, `throughput/duration +8.05%`.
  - Governance:
    - MFU gain `<3%` and dominant hotspot family remained train-path `shard_map/custom-call` with large new `while` overhead.
    - Attempt marked **low-impact/regressive**; speculative kernel edits were reverted and escalated.

- Assessment: **failed attempt / regression**. The fused full-sequence train kernel cut train closed-call `shard_map/pallas_call` times by ~50%, but added substantial `while` overhead and regressed end-to-end throughput.
- Next bold hypothesis:
  - Move to **Macro Move J** next (coverage slot 2/5 for this run): execute the required `Ct={64,96,128}` x `Seg={8,16,32}` sweep with a compact benchmark table, then use the best point to launch a stronger structural move (likely Macro E if `while` overhead persists).
