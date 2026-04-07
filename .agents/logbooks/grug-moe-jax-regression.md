# Grug MoE JAX 0.8.0 vs 0.9.2: Research Logbook

## Scope
- Goal: Determine whether upgrading Marin TPU training from JAX 0.8.0 to 0.9.2 regresses steady-state training throughput for the Grug MoE model.
- Primary metric(s): `throughput/tokens_per_second`, `throughput/gflops_per_second`, `throughput/mfu`, and per-step wall-clock `throughput/duration`.
- Constraints: Compare apples-to-apples on the same TPU type and device count, same Grug MoE model/data/optimizer config, same batch size, same code path, and no concurrent TPU workloads.

## Baseline
- Date: 2026-04-06
- Code refs: `experiments/grug/moe/launch.py`, `experiments/grug/moe/train.py`, `lib/levanter/src/levanter/callbacks/_metrics.py`, `lib/marin/pyproject.toml`, `lib/levanter/pyproject.toml`
- Baseline numbers:
  - `jax==0.8.0` / `jaxlib==0.8.0` / `libtpu==0.0.24`: 3-step smoke run completed on `v4-8`, reached step 3, and saved a checkpoint.
  - `jax==0.9.2` / `jaxlib==0.9.2` / `libtpu==0.0.38`: same smoke run failed on the first training step with a `shard_map` `in_specs` mismatch in `lib/levanter/src/levanter/grug/grug_moe.py`.

## Links
- Umbrella issue: https://github.com/marin-community/marin/issues/4357
- Experiment issue: https://github.com/marin-community/marin/issues/4455
- Local harness: `scripts/grug_moe_jax_bench.py`

## Experiment IDs
- Prefix: `GRUG-JAX`
- Planned runs:
  - `GRUG-JAX-001`: pre-upgrade baseline on `025cacea6` with `jax==0.8.0`
  - `GRUG-JAX-002`: post-upgrade case on `3b1aabc6d` or `8a9a31455` with `jax==0.9.2`

## Problem Statement
- Ahmed's `tpu-dep-hell` branch moves Marin TPU runtime dependencies from the old `vllm-tpu`-pinned JAX stack (`jax==0.8.0`) to forked TPU wheels and `jax==0.9.2`.
- The open question for this thread is whether that runtime shift changes Grug MoE training performance, not whether it changes training quality or correctness.

## Success Metrics
- Produce a reproducible benchmark command for both JAX versions.
- Run both versions on the same TPU type and compare steady-state throughput after warmup.
- State whether there is a regression, improvement, or no material change, with explicit evidence and limitations.

## Initial Hypotheses
- H1: The JAX 0.9.2 stack does not materially regress steady-state Grug MoE training throughput on `v5p-8`.
- H2: If there is a regression, it is more likely to show up in step duration / MFU than in dataloader time, because the code path change is primarily runtime/compiler related.
- H3: First-step compile latency may change even if steady-state throughput does not.

## First Experiment Matrix
- Fixed benchmark case: `experiments/grug/moe` baseline model on `v5p-8`, batch size 32, sequence length 4096, AdamH optimizer, QB enabled, eval disabled, JSON logger enabled, short local run with warmup steps excluded from comparison.
- Axis 1: JAX stack
  - `jax==0.8.0`
  - `jax==0.9.2`
- Constant controls:
  - same Git worktree (`tpu-dep-hell`)
  - same benchmark harness
  - same TPU worker / device count
  - same data path and train mixture
  - same mixed-precision policy
  - same seed

## Stop Criteria
- Stop when both JAX versions have completed the same benchmark and the measured-step throughput deltas are clear enough to classify as `exploratory` or better.
- Escalate if the comparison is blocked by environment-resolution issues or obvious TPU contention.

## Final Classification
- Classification: No Grug MoE training regression on the fixed `jax==0.9.2` stack; stronger `v5p-8` evidence shows a small throughput improvement and no observed short-run quality regression.
- Outcome: After fixing the Grug `shard_map` boundary assumptions and repairing the benchmark harness, both JAX stacks train successfully. On matched one-step quality checks on `v5p-8`, loss/router drift is negligible and total parameter norm matches exactly. On a matched `v5p-8`, `steps=7`, `warmup=2`, `batch_size=32` comparison, `jax==0.9.2` is `4.33%` faster than `jax==0.8.0` in `throughput/tokens_per_second` and `mfu`, with `duration` `4.15%` lower. A confirmatory second `0.9.2` run reproduced the same steady-state number within `0.03%`.
- Confidence: High that the fixed `jax==0.9.2` path is no longer a training blocker and shows no obvious short-run numerical regression; moderate that the stronger `v5p-8` steady-state result is the right throughput readout for this thread.
- Main caveat: The earlier short `v4-8` benchmark pointed the other way (`-3.83%` on `steps=2`, `warmup=1`), but the longer `v5p-8` run shows a clear two-step warmup cliff before steady state. I do not consider the earlier `v4-8` micro-benchmark the best final perf claim.

## Experiment Log
### 2026-04-06 22:03 UTC - Kickoff
- Hypothesis: A short local Grug MoE training harness on a reserved dev TPU can provide an apples-to-apples throughput comparison between JAX 0.8.0 and 0.9.2.
- Command: Context gathering only.
- Config:
  - Worktree: `tpu-dep-hell`
  - Candidate baseline commits: `025cacea6` (pre-upgrade) vs `3b1aabc6d` / `8a9a31455` (post-upgrade)
  - Canonical benchmark target: `experiments/grug/moe`
- Result:
  - `experiments/grug/moe/launch.py` defines the production Grug MoE training shape on `v5p-8`.
  - `experiments/grug/moe/train.py` already logs `throughput/tokens_per_second`, `throughput/gflops_per_second`, `throughput/mfu`, `throughput/duration`, and `throughput/loading_time`.
  - Diff `025cacea6..3b1aabc6d` does not touch Grug training code; it changes dependency/runtime files and vLLM-related plumbing, which makes it a clean JAX-stack comparison point for training.
- Interpretation: The benchmark should vary the JAX stack while keeping Grug code constant. A lightweight local harness is appropriate.
- Next action: Create the experiment issue, allocate a dev TPU, and run the benchmark under both JAX versions.

### 2026-04-06 22:26 UTC - Environment prep on dev TPU
- Hypothesis: A minimal local benchmark harness on a single dev TPU worker is enough to classify the regression.
- Commands:
  - Allocated dev TPU session `codex-tpu-dep-hell-grug-jax-v4` on `v4-8`.
  - Synced the branch to the worker and regenerated Iris protobufs.
  - Seeded a local tokenizer snapshot override from `/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` to avoid Hugging Face auth.
- Config:
  - TPU type: `v4-8`
  - Branch: `tpu-dep-hell` at `8a9a31455`
  - Benchmark harness: `scripts/grug_moe_jax_bench.py`
  - Benchmark command shape:
    - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id <id> --steps 3 --warmup-steps 1 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Result:
  - The worker environment was ready for paired smoke tests on both JAX stacks.
- Interpretation: The remaining variable was the JAX/libtpu runtime itself.
- Next action: Run the branch-pinned `jax==0.9.2` smoke test first.

### 2026-04-06 22:28 UTC - Post-upgrade smoke test (`jax==0.9.2`)
- Hypothesis: The `tpu-dep-hell` branch should at least complete a short Grug MoE training smoke test under the new TPU runtime stack.
- Command:
  - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-smoke-092 --steps 3 --warmup-steps 1 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Config:
  - Runtime: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38`
  - Same model/data/optimizer config as the old-stack control
- Result:
  - The run failed on the first training step before any benchmark report could be produced.
  - Fatal error:
    - `ValueError: in_specs passed to shard_map: P('expert', None, None) does not match the specs of the input: P('expert', 'data', 'model')`
  - Failure site:
    - `lib/levanter/src/levanter/grug/grug_moe.py:550`
- Interpretation:
  - This is not a small throughput regression. On this setup, the upgraded stack fails to train Grug MoE at all.
- Next action:
  - Swap the TPU runtime back to the pre-upgrade versions and rerun the exact same smoke test as a control.

### 2026-04-06 22:31 UTC - Pre-upgrade control (`jax==0.8.0`)
- Hypothesis: If the regression is truly tied to the JAX stack, the same Grug MoE smoke test should run under the old pinned TPU runtime.
- Commands:
  - `/home/ubuntu/marin/.venv/bin/pip install -U jax==0.8.0 jaxlib==0.8.0 libtpu==0.0.24`
  - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-smoke-080 --steps 3 --warmup-steps 1 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Config:
  - Runtime: `jax==0.8.0`, `jaxlib==0.8.0`, `libtpu==0.0.24`
  - Same worker, same branch checkout, same benchmark command, same tokenizer override
- Result:
  - The run completed 3 training steps and saved a checkpoint to `/tmp/grug-moe-jax-bench/grug-jax-smoke-080/checkpoints/step-3`.
  - Sample emitted step metrics:
    - Step 0: `throughput/tokens_per_second = 1233.73`, `throughput/duration = 106.24s`
    - Step 1: `throughput/tokens_per_second = 1309.46`, `throughput/duration = 100.10s`
    - Step 2: `throughput/tokens_per_second = 108246.37`, `throughput/duration = 1.21s`
  - The harness post-processing then failed locally because it expected a JSON logger file at `/tmp/grug-moe-jax-bench-logs/grug-jax-smoke-080.log`, but the tracker output for this run was available on stdout instead.
- Interpretation:
  - The old pinned stack successfully runs Grug MoE training on the same TPU and code path where the new stack fails immediately.
  - The smoke-run timing is not clean enough to make a trustworthy steady-state throughput claim, because the final short run shows obviously noisy step timing and the new stack never reaches steady state anyway.
- Next action:
  - Restore the worker to the branch-pinned `jax==0.9.2` stack and write up the result as a training regression rather than a quantified throughput delta.

### 2026-04-06 22:40 UTC - Wrap-up
- Commands:
  - `/home/ubuntu/marin/.venv/bin/pip install -U jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.38`
- Result:
  - Restored the remote worker to the `tpu-dep-hell` branch runtime versions.
- Interpretation:
  - The experiment is complete.
- Next action:
  - Update the experiment issue with the findings.

### 2026-04-06 23:40 UTC - Root-cause investigation on `jax==0.9.2`
- Hypothesis:
  - The first Grug MoE crash is caused by a `shard_map` boundary that relied on older implicit re-sharding behavior.
- Local code change under test:
  - In `lib/levanter/src/levanter/grug/grug_moe.py`, explicitly re-shard `w_up_gate` and `w_down` to `P("expert", None, None)` before calling `shard_map`.
- Why this hypothesis fit the failure:
  - The MoE module initializes weights as:
    - `w_gate_up`: `P("expert", "data", "model")`
    - `w_down`: `P("expert", "model", "data")`
  - But `moe_mlp()` passed them into `shard_map` with `in_specs=(..., P("expert", None, None), P("expert", None, None))`.
  - Under `jax==0.9.2`, that mismatch is rejected explicitly instead of being silently accommodated.
- Verification run:
  - Re-ran the same smoke benchmark on `jax==0.9.2` after copying the patched `grug_moe.py` to the worker.
- Result:
  - The original failure at `lib/levanter/src/levanter/grug/grug_moe.py:550` disappeared.
  - The run progressed further into training and then failed later in a different file:
    - `lib/levanter/src/levanter/grug/loss.py:132`
    - Error: `in_specs passed to shard_map: P(None, None) does not match the specs of the input: P('data', 'model')`
- Interpretation:
  - This confirms the first failure was caused by missing explicit re-sharding at the `shard_map` boundary.
  - The deeper root cause is broader than one MoE callsite: multiple Grug `shard_map` callsites appear to rely on implicit input re-sharding that older JAX tolerated and `jax==0.9.2` no longer allows.
  - The new stack regression is therefore best described as a `shard_map` boundary-compatibility problem under stricter JAX 0.9 sharding checks.
- Next action:
  - Audit remaining Grug `shard_map` callsites, especially those with hard-coded `in_specs` that under-specify actual input sharding.

### 2026-04-07 00:32 UTC - Version-compatible fix validated on both JAX stacks
- Hypothesis:
  - The `jax==0.9.2` Grug MoE regression is caused by stricter `shard_map` input-spec validation at a small number of boundaries, and an explicit `reshard` fix can preserve compatibility with both `jax==0.8.0` and `jax==0.9.2`.
- Local code changes under test:
  - `lib/levanter/src/levanter/grug/grug_moe.py`
    - import `reshard` from `jax.sharding`
    - explicitly `reshard(w_up_gate, local_expert_spec)` and `reshard(w_down, local_expert_spec)` before the MoE `shard_map`
  - `lib/levanter/src/levanter/grug/loss.py`
    - import `reshard` from `jax.sharding`
    - explicitly `reshard(lm_head, P(None, None))` before the fused-loss `shard_map`
- Why this form matters:
  - The first experimental patch used top-level `jax.reshard(...)`, which fixed `jax==0.9.2` but failed under `jax==0.8.0` because that symbol is not exposed there.
  - Switching to `jax.sharding.reshard` matches existing repo usage and works on both tested stacks.
- Verification runs:
  - `jax==0.8.0` / `jaxlib==0.8.0` / `libtpu==0.0.24`
    - Run ID: `grug-jax-smoke-080-fixed`
    - Result: completed 3 training steps, saved `/tmp/grug-moe-jax-bench/grug-jax-smoke-080-fixed/checkpoints/step-3`, then hit the known harness `FileNotFoundError` for the missing JSON log file.
    - Sample step metrics:
      - Step 0: `throughput/tokens_per_second = 1238.23`, `throughput/duration = 105.85s`
      - Step 1: `throughput/tokens_per_second = 1269.32`, `throughput/duration = 103.26s`
  - `jax==0.9.2` / `jaxlib==0.9.2` / `libtpu==0.0.38`
    - Run ID: `grug-jax-smoke-092-fixed-vcompat`
    - Result: completed 3 training steps, saved `/tmp/grug-moe-jax-bench/grug-jax-smoke-092-fixed-vcompat/checkpoints/step-3`, then hit the same known harness `FileNotFoundError`.
    - Sample step metrics:
      - Step 0: `throughput/tokens_per_second = 1256.42`, `throughput/duration = 104.32s`
      - Step 1: `throughput/tokens_per_second = 1272.98`, `throughput/duration = 102.96s`
- Interpretation:
  - Root cause confirmed: the broken branch relied on implicit re-sharding across `shard_map` boundaries that older JAX tolerated and `jax==0.9.2` rejects.
  - The regression is not a deeper Grug MoE training incompatibility in the new runtime. With explicit re-sharding at the affected boundaries, Grug MoE training succeeds on both JAX stacks.
  - On this short smoke benchmark, the nontrivial training-step durations under the fixed code are very similar across `0.8.0` and `0.9.2`; there is no obvious performance regression in the fixed path from these runs alone.
- Next action:
  - Write up the confirmed root cause and the version-compatible fix on the experiment issue.

### 2026-04-07 01:45 UTC - Harness fixed and green on `jax==0.9.2`
- Hypothesis:
  - The benchmark failures after successful training are caused by harness/reporting assumptions, not by the Grug training path itself.
- Local code changes under test:
  - `scripts/grug_moe_jax_bench.py`
    - attach a dedicated file handler for the JSON tracker so `log_root/<run_id>.log` is guaranteed to exist
    - make git metadata optional when the synced TPU checkout has no `.git`
    - disable checkpointing inside this benchmark harness to avoid wasting disk/time on benchmark-only smoke runs
    - de-duplicate repeated final-step throughput records in report post-processing
- Verification run:
  - Runtime: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38`
  - Command:
    - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-092-green --steps 1 --warmup-steps 0 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Result:
  - The run exited `0`.
  - The expected tracker file was created at `/tmp/grug-moe-jax-bench-logs/grug-jax-092-green.log`.
  - Report summary:
    - Step 0: `throughput/tokens_per_second = 1211.27`, `throughput/duration = 108.21s`, `throughput/mfu = 0.24764`
    - Final loss: `11.8214178085`
    - Final router aux loss: `0.0175585914`
    - Final total parameter norm: `571.0413818359`
- Interpretation:
  - The harness bug is fixed: the benchmark now produces its log file, report block, and successful exit status on the new stack.
  - Checkpointing is not needed for this benchmark and was a real source of spurious TPU-worker failures (`ENOSPC`) during earlier validation attempts.
- Next action:
  - Re-run matched old/new JAX cases for a numerical sanity check and a cleaner stable-step throughput comparison.

### 2026-04-07 02:12 UTC - Matched quality and throughput comparison after harness repair
- Hypothesis:
  - After the explicit `reshard` fixes, the remaining question is whether `jax==0.9.2` changes either short-run training math or stable per-step throughput versus `jax==0.8.0`.
- Commands:
  - One-step math check on old stack:
    - `/home/ubuntu/marin/.venv/bin/pip install -U jax==0.8.0 jaxlib==0.8.0 libtpu==0.0.24`
    - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-080-math --steps 1 --warmup-steps 0 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
  - Stable-step throughput check on old stack:
    - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-080-bench-2step --steps 2 --warmup-steps 1 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
  - Restore new stack:
    - `/home/ubuntu/marin/.venv/bin/pip install -U jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.38`
  - Stable-step throughput check on new stack:
    - `/home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-092-bench-2step --steps 2 --warmup-steps 1 --batch-size 32 --tpu-type v4-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Result:
  - One-step math check:
    - `jax==0.8.0`: loss `11.8214120865`, router aux `0.0175586101`, total param norm `571.0413818359`
    - `jax==0.9.2`: loss `11.8214178085`, router aux `0.0175585914`, total param norm `571.0413818359`
    - Deltas:
      - loss abs diff: `5.72e-6`
      - loss rel diff: `4.84e-7`
      - router aux abs diff: `1.86e-8`
      - total param norm abs diff: `0.0`
  - Stable-step throughput check (`steps=2, warmup=1`, so only step 1 is measured):
    - `jax==0.8.0`: step 1 `1276.66 tok/s`, `102.67s`, `mfu = 0.26101`, final loss `11.6174840927`
    - `jax==0.9.2`: step 1 `1227.73 tok/s`, `106.76s`, `mfu = 0.25100`, final loss `11.6174612045`
    - Deltas (`0.9.2` relative to `0.8.0`):
      - `throughput/tokens_per_second`: `-3.83%`
      - `throughput/duration`: `+3.99%`
      - `throughput/mfu`: `-3.83%`
      - final loss abs diff: `2.29e-5`
  - Negative result:
    - Re-running the older `steps=3, warmup=1` harness shape still showed a noisy terminal-step artifact on this setup, so I did not use that shape for the final throughput claim.
- Interpretation:
  - The repaired benchmark harness is working end to end on both stacks.
  - I do not see evidence of a one-step quality regression from the JAX upgrade after the explicit `reshard` fixes; the math check is effectively identical at this precision.
  - I do see a small exploratory training throughput regression on `jax==0.9.2` on this `v4-8` worker, on the order of `~3.8%`.
- Next action:
  - Update the experiment issue with the cleaned benchmark result and the no-quality-regression note.

### 2026-04-07 15:00 UTC - Stronger `v5p-8` rerun with longer warmup-stable window
- Hypothesis:
  - The short `v4-8` comparison likely under-sampled steady state. A matched `v5p-8` run with a longer window after warmup should give a stronger throughput readout, and a matched one-step check should confirm the training math still agrees across JAX stacks.
- Commands:
  - New-stack one-step quality check:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs /tmp/grug-jax-092-v5p-math.out && mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs && cd ~/marin && /home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-092-v5p-math --steps 1 --warmup-steps 0 --batch-size 32 --tpu-type v5p-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b > /tmp/grug-jax-092-v5p-math.out 2>&1'`
  - New-stack long run:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs /tmp/grug-jax-092-v5p-7step.out && mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs && cd ~/marin && /home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-092-v5p-7step --steps 7 --warmup-steps 2 --batch-size 32 --tpu-type v5p-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b > /tmp/grug-jax-092-v5p-7step.out 2>&1'`
  - Swap to old stack:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='cd ~/marin && /home/ubuntu/marin/.venv/bin/pip install -U jax==0.8.0 jaxlib==0.8.0 libtpu==0.0.24'`
  - Old-stack one-step quality check:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs /tmp/grug-jax-080-v5p-math.out && mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs && cd ~/marin && /home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-080-v5p-math --steps 1 --warmup-steps 0 --batch-size 32 --tpu-type v5p-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b > /tmp/grug-jax-080-v5p-math.out 2>&1'`
  - Old-stack long run:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs /tmp/grug-jax-080-v5p-7step.out && mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs && cd ~/marin && /home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-080-v5p-7step --steps 7 --warmup-steps 2 --batch-size 32 --tpu-type v5p-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b > /tmp/grug-jax-080-v5p-7step.out 2>&1'`
  - Restore new stack and confirm repeatability:
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='cd ~/marin && /home/ubuntu/marin/.venv/bin/pip install -U jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.38'`
    - `gcloud compute tpus tpu-vm ssh marin-tpu-v5p-8-us-central1-a-20260407-1410-d86674dd --zone=us-central1-a --project=hai-gcp-models --worker=0 --quiet --command='rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs /tmp/grug-jax-092-v5p-7step-b.out && mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs && cd ~/marin && /home/ubuntu/marin/.venv/bin/python scripts/grug_moe_jax_bench.py --run-id grug-jax-092-v5p-7step-b --steps 7 --warmup-steps 2 --batch-size 32 --tpu-type v5p-8 --tokenizer /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b > /tmp/grug-jax-092-v5p-7step-b.out 2>&1'`
- Config:
  - TPU: `v5p-8` dev TPU in `us-central1-a`
  - Harness shape: `steps=7`, `warmup=2`, `batch_size=32`
  - Measured window: steps `2..6`
  - Tokenizer: local snapshot override at `/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Result:
  - One-step quality check:
    - `jax==0.8.0`: loss `11.8214359283`, router load-balancing `4.0194091797`, router z-loss `17.5586547852`, total param norm `571.0414428711`
    - `jax==0.9.2`: loss `11.8214445114`, router load-balancing `4.0194029808`, router z-loss `17.5586376190`, total param norm `571.0414428711`
    - Deltas:
      - loss abs diff: `8.58e-6`
      - router load-balancing abs diff: `6.20e-6`
      - router z-loss abs diff: `1.72e-5`
      - total param norm abs diff: `0.0`
  - Long-run throughput:
    - `jax==0.8.0` / `jaxlib==0.8.0` / `libtpu==0.0.24`
      - mean steady-state `throughput/tokens_per_second`: `184264.60`
      - mean steady-state `throughput/duration`: `0.711325s`
      - mean steady-state `throughput/mfu`: `22.57041`
      - final loss: `10.8818731308`
    - `jax==0.9.2` / `jaxlib==0.9.2` / `libtpu==0.0.38`
      - run A mean steady-state `throughput/tokens_per_second`: `192213.01`
      - run A mean steady-state `throughput/duration`: `0.681910s`
      - run A mean steady-state `throughput/mfu`: `23.54401`
      - run B mean steady-state `throughput/tokens_per_second`: `192260.89`
      - run B mean steady-state `throughput/duration`: `0.681741s`
      - run B mean steady-state `throughput/mfu`: `23.54987`
      - confirmatory run spread: `<0.03%` on tokens/s, duration, and MFU
    - New-vs-old delta using the mean of the two `0.9.2` runs:
      - `throughput/tokens_per_second`: `+4.33%`
      - `throughput/duration`: `-4.15%`
      - `throughput/mfu`: `+4.33%`
  - Warmup behavior:
    - On `v5p-8`, steps `0` and `1` are still in the slow compile/warmup regime (`~77s`, then `~72s` on the first `0.9.2` run), but steps `2..6` are stable and repeatable. This longer window is a much better readout than the earlier short `v4-8` benchmark.
- Interpretation:
  - I still do not see evidence of a short-run quality regression after the explicit `reshard` fixes.
  - The stronger `v5p-8` steady-state benchmark does not show a regression on `jax==0.9.2`; it shows a small improvement instead, with repeated `0.9.2` runs matching almost exactly.
  - The earlier `v4-8` `steps=2`, `warmup=1` comparison was too shallow to be the final claim for this thread and likely over-weighted warmup behavior.
- Next action:
  - Update the experiment issue body and add a final milestone comment with the stronger `v5p-8` result.
