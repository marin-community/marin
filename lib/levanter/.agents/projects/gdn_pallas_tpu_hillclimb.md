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
- Commit: (pending)
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
- Commit: (pending)
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
- Commit: (pending)
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
- Commit: (pending)
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
- Commit: (pending)
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
- Commit: (pending)
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
- Commit: this commit
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
