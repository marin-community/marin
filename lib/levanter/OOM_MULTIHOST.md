# Multi-Host Sampling OOM Plan (4096 tokens)

## Goal
Generate **up to 4096 new tokens** per prompt in multi-host inference **without OOM**, while keeping behavior deterministic and W&B logging intact.

**Current status:** ragged paged attention is disabled and HF config overrides are fixed. v5p-16 completes **single‑prompt** runs at both 2048 and 4096 tokens; v5p-16 runs with **multiple prompts** and v5p-64 runs at 4096 still exit with status 1 (launch-group mismatch on v5p-64; silent exit on v5p-16).

---

## What’s OOMing (Root Cause)

We are no longer hitting *HBM* OOM. We are hitting **TPU scoped vmem** OOM inside the **ragged paged attention kernel** during **prefill**.

The failure looks like:

```
RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for ragged_paged_attention
Scoped allocation with size 37.54M and limit 16.00M exceeded scoped vmem limit
```

Key observations:
- The error **does not shrink** with `max_prefill_size`, batch size, or page size.
- Even with extremely small configs (batch size 1, prefill size 64), the kernel still allocates ~37.5MB scoped vmem.
- This implies the **TPU ragged paged attention kernel has a fixed scratch allocation** that exceeds the 16MB vmem limit for this model shape.

✅ **Conclusion:** This OOM **cannot be fixed by config alone** while using the TPU ragged paged attention kernel.

---

## Why the “realistic” config triggers it

`config/sampler/sample_llama8b_multihost_real.yaml` has:
- `max_new_tokens: 4096`
- `engine.max_seq_len: 4608` (4096 + prompt length)
- Many prompts (100), even though we batch them

The kernel still allocates the same scratch size for ragged paged attention and fails regardless of prefill batching.

---

## Plan (to support 4096 tokens)

### 1) **Disable the TPU ragged paged attention kernel**
We already added a config option for Llama:

```yaml
model:
  type: llama
  use_tpu_ragged_paged_attention: false
```

This forces the **reference implementation**, which avoids the TPU kernel’s scoped vmem allocation.

✅ **This is the most viable path to keep 4096 tokens today.**

---

### 2) **Keep deterministic batching**
Because 100 prompts × 4096 tokens is large, keep batching enabled:

```yaml
max_prompts_per_batch: 1
```

We can increase this later if memory permits.

---

### 3) **Keep KV layout small but correct**
Ensure the KV cache and sequence sizing remain compatible with 4096 tokens:

```yaml
engine:
  max_seq_len: 4608
  page_size: 64   # smaller pages reduce per-page overhead
  max_pages: 80   # ceil(4608/64)=72, leave some margin
```

---

### 4) **If reference attention is too slow**
If this fallback is too slow:

- Consider lowering to 2048 tokens (if acceptable), or
- Explore kernel alternatives:
  - Non-ragged paged attention (if implemented)
  - A patched ragged kernel with reduced scratch

But today, the practical path is the reference implementation.

---

## Checklist (to run)

1. Confirm config contains:
   - `model.type: llama`
   - `use_tpu_ragged_paged_attention: false`
2. Run with `max_prompts_per_batch: 1`
3. Verify W&B logs throughput + samples

Command:

```
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-64 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real.yaml
```

---

## Status

✅ Multi-host sampling works with short lengths.
✅ With ragged paged attention disabled **and** HF-config-derived shapes, the 4096-token run reaches prefill + decode.
⚠️ The run still exits non-zero after several minutes; root cause unknown (no OOM/traceback observed).

## Worked vs failed (as of 2026-02-03)

**Worked**
- v5p-16, 1 prompt, 2048 tokens (`sample_llama8b_multihost_real_1prompt_2048.yaml`): clean exit, W&B runs `tdn4agwm` (sparkling-puddle-26) and `6nwwv4hm` (ruby-sun-27).
- v5p-16, 1 prompt, 4096 tokens (`sample_llama8b_multihost_real_1prompt.yaml`): clean exit, W&B run `ytbn4l2u` (splendid-energy-31), offline run dir `offline-run-20260203_203843-ytbn4l2u`.

**Failed**
- v5p-64, 4096 tokens (1/20/100 prompts): launch-group mismatch + slice halt, core dumps written (e.g. `/tmp/slice_1770110283.dump`).
- v5p-16, 20 prompts, 4096 tokens (`sample_llama8b_multihost_real_20prompts_4096.yaml`): exit status 1 after decode; no explicit TPU runtime error in logs (W&B run `weou7ot5`, clear-sky-28).
- v5p-16, 500 prompts, 2048 tokens (`sample_llama8b_multihost_real_500prompts_2048.yaml`): exit status 1 after decode; no explicit TPU runtime error captured.
- v5p-16, 5 prompts, 4096 tokens (`sample_llama8b_multihost_real_5prompts_4096.yaml`): exit status 1 after decode; no explicit TPU runtime error captured.

---

## Attempts (chronological)

### Attempt #1: Make config more conservative
- Reduced batch size, `max_prefill_size`, `max_pages`, `page_size`, and `max_seqs`.
- Result: **Still OOM** in TPU ragged paged attention with the same ~37.5MB scoped vmem allocation.

### Attempt #2: Add `use_tpu_ragged_paged_attention: false`
- Added a config toggle to disable TPU ragged paged attention and use the reference implementation.
- Result: **Still OOM**, because the HF checkpoint loader **ignored the model override** and rebuilt the model from HF config.

### Attempt #3: Ensure HF loader respects model overrides
- Updated `sample_lm_multihost.py` to pass `config.model` into `HFCheckpointConverter.load_pretrained(...)`.
- This ensures `use_tpu_ragged_paged_attention: false` actually takes effect.
- **Next step:** re-run with `sample_llama8b_multihost_real.yaml` after this change.

### Attempt #4: Merge HF config with explicit overrides (fix shape mismatch)
- Changed `sample_lm_multihost.py` to build the model config from the HF checkpoint first, then apply only non-default overrides (e.g., `use_tpu_ragged_paged_attention: false`).
- This avoids Llama-2 defaults (num_kv_heads=32) clobbering Llama-3.1-8B GQA shapes (num_kv_heads=8).
- **Run (2026-02-03):** `sample_llama8b_multihost_real.yaml` now loads all four shards, completes initial prefill, and enters decode iterations (no vmem OOM so far). W&B run id: `7ds3e56d` (project `levanter-multihost-sampling`).

### Attempt #5: Long decode run exits non-zero (no OOM/traceback)
- **Observed logs (2026-02-03 ~07:29–07:32):** steady decode iterations (64 tokens/iter, ~1–5s/iter).
- **Outcome:** all workers exited with status 1 around ~07:11–07:32 (container `ExitCode=1`, `OOMKilled=false`).
- **Notes:** no `RESOURCE_EXHAUSTED` or Python traceback found in `docker logs` (only JAX scatter dtype warnings). Root cause still unknown.
- **Next step:** capture full `docker logs` right after failure (or rerun with explicit debug env like `JAX_TRACEBACK_FILTERING=off`) to surface the terminating error.

### Attempt #6: User run shows steady decode then exit (needs full error)
- **User-provided logs (2026-02-03 ~07:29–07:32):** consistent decode iterations across workers at ~1.0–1.3s/iter.
- **Outcome:** job later failed (exit status 1) without a surfaced traceback in the sampled log snippet.
- **Action item:** rerun with full `docker logs` capture or add explicit exception logging around `engine.generate` to record the failing exception.

### Attempt #7: Infra reports remote command failed on all workers
- **User-provided logs (2026-02-03 ~07:29–07:32):** decode continues, then throughput steadily degrades (iter time grows to ~5s; tok/s drops to ~12–13).
- **Outcome:** `gcloud ... tpu-vm ssh --command=docker run ...` reports `Command execution on worker X failed with exit status 1` for all workers, followed by a top-level `Error running command` from the launcher.
- **Notes:** the launcher failure means the remote `docker run` command exited non-zero on each worker; no explicit Python traceback or OOM is shown in the provided snippet.
- **System metrics snapshot:** TPU HBM usage appears low and flat (~15–20%), system memory stays low, and TPU duty cycle drops to 0 when the job exits. This suggests the failure is **not** an HBM OOM, but it **does not rule out** a scoped vmem / kernel error or other runtime exception.
- **Action item:** capture full `docker logs levanter` from a worker immediately after failure, or rerun with `JAX_TRACEBACK_FILTERING=off` / explicit exception logging around `engine.generate` to surface the terminating error.

### Attempt #8: Rerun with full traceback envs still exits silently
- **Run (2026-02-03 ~07:53):** set `JAX_TRACEBACK_FILTERING=off`, `PYTHONFAULTHANDLER=1`, `HYDRA_FULL_ERROR=1`.
- **Outcome:** decode continues, then all workers exit with status 1; launcher reports `Command execution on worker X failed with exit status 1` and retries.
- **Notes:** still no Python traceback in the sampled logs. TPU log envs were set to `TPU_STDERR_LOG_LEVEL=3` and `TPU_MIN_LOG_LEVEL=3`, which likely suppress libtpu/XLA error output.
- **Action item:** rerun with TPU log levels set to 0 (or unset), and capture per‑worker `docker logs` + `docker inspect` immediately after failure to see the real error / exit cause.

### Attempt #9: Monitor captured only the retried run (logs wiped on retry)
- **Monitor (2026-02-03 ~08:09):** captured `docker inspect` and `docker logs` from worker 0 after failure.
- **Outcome:** container was **already restarted** (Running=true); logs only show fresh startup (uv install, W&B init, HF load). No failure context.
- **Root cause:** `launch.py` retries by default, and `setup_vm_docker()` runs `docker rm -f levanter` before each attempt, wiping logs from the failed run.
- **Action item:** rerun with `--retries 0` to preserve the failed container/logs, or add a one‑off run that skips `docker rm -f levanter` so we can capture the error.

### Attempt #10: TPU runtime launch-group mismatch (not OOM)
- **Logs (2026-02-03 ~08:23):** TPU runtime reports `Core halted unexpectedly` and **launch group mismatch**:
  - `An unexpected peer shows up in the launch group with a different launch id than the current group leader.`
  - `Accelerator device halted prematurely ... on-device check-failure.`
  - `Slice health check failed ... Worker connection ... failed.`
- **Interpretation:** this is **not HBM OOM**. A worker (or its compiled program) diverged from the group, causing a slice-wide halt. This can happen if:
  - a worker crashes and restarts mid-run, or
  - **non-deterministic compilation** yields different HLOs across workers.
- **Next step (from TPU error):** enable HLO dumps on **all** workers and verify `before_optimizations.txt` / `after_optimizations.txt` are identical across hosts. If they differ, fix the nondeterminism (config/env/code path divergence).

### Attempt #11: HLO dumps identical; slice still halts
- **Run (2026-02-03 ~08:37–08:43):** launched with `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"` and full TPU logs enabled.
- **Observed:** model loads, prefill + decode proceed; throughput degrades to ~5s/iter, then TPU runtime halts:
  - `INTERNAL: *** Halt is unexpected, all pending programs will fail.`
  - `Slice health check failed ... Worker connection ... failed.`
  - Core dump written: `/tmp/slice_1770108203.dump`.
- **HLO consistency check:** hashes of `before_optimizations.txt` + `after_optimizations.txt` are **identical across all workers** (`64c14be2e0...`), ruling out non‑deterministic compilation as the cause.
- **Interpretation:** likely a worker/runtime failure (ICI/link/host issue) or TPU runtime bug, not HBM OOM and not HLO divergence.
- **Next step:** try recreating the TPU slice (fresh queued resource), or change runtime version; if reproducible on a fresh slice, escalate with TPU support using the core dump.

### Attempt #12: Single-prompt run to isolate multi-prompt effects
- **Run (2026-02-03 ~09:06):** launched `sample_llama8b_multihost_real_1prompt.yaml` (4096 tokens, 1 prompt).
- **Outcome:** no clean success recorded; follow-up v5p-64 4096 runs still halted with launch-group mismatch.

### Attempt #13: 20-prompt run (v5p-64)
- **Run (2026-02-03 ~09:11):** launched `sample_llama8b_multihost_real_20prompts.yaml` (first 20 prompts, 4096 tokens; `max_prompts_per_batch: 1`).
- **Purpose:** determine if the failure scales with the number of sequential `engine.generate` calls.
- **Outcome:** **failed** after reaching steady decode (iter time ~4–5s, ~13–16 tok/s). TPU runtime halted the slice:
  - `INTERNAL: *** Halt is unexpected, all pending programs will fail.`
  - `Slice health check failed ... Worker connection ... failed.`
  - Core dump written: `/tmp/slice_1770110283.dump` on worker 0.
  - **Worker 3** showed the earliest fault: `An unexpected peer shows up in the launch group with a different launch id...` immediately before the slice halt.
  - **HLO hashes** across workers were identical (`64c14be2e0...`), so this is **not** nondeterministic compilation.
  - W&B run was **offline** (run id `v73zwmay`); logs are in `/opt/marin/lib/levanter/wandb/offline-run-20260203_091145-v73zwmay`.
  - Dump copied locally: `lib/levanter/.oom_multihost_logs/2026-02-03_20prompts/slice_1770110283.dump`.

### Attempt #14: v5p-16, 1 prompt, 2048 tokens (success)
- **Run (2026-02-03 ~19:06):** `sample_llama8b_multihost_real_1prompt_2048.yaml`.
- **Outcome:** **success**; W&B run `tdn4agwm` (sparkling-puddle-26). Sample output stored in W&B table.

### Attempt #15: v5p-16, 1 prompt, 2048 tokens (success)
- **Run (2026-02-03 ~19:43–19:47):** same config, online W&B.
- **Outcome:** **success**; W&B run `6nwwv4hm` (ruby-sun-27). Clean exit; only teardown warnings about debug metadata.

### Attempt #16: v5p-16, 500 prompts, 2048 tokens (failed)
- **Run (2026-02-03 ~20:08):** `sample_llama8b_multihost_real_500prompts_2048.yaml`.
- **Outcome:** **failed** after decode iterations; launcher reported `Command execution on worker X failed with exit status 1`. No explicit TPU runtime error captured in logs.

### Attempt #17: v5p-16, 20 prompts, 4096 tokens (failed)
- **Run (2026-02-03 ~20:08–20:24):** `sample_llama8b_multihost_real_20prompts_4096.yaml`.
- **Outcome:** **failed** after decode iterations (~5s/iter, ~12–13 tok/s). Launcher reported `Command execution on worker X failed with exit status 1`. W&B run `weou7ot5` (clear-sky-28).

### Attempt #18: v5p-16, 5 prompts, 4096 tokens (new config)
- **Config created:** `sample_llama8b_multihost_real_5prompts_4096.yaml`.
- **Next step:** run to see whether 4096-token failures persist at 5 prompts.

### Attempt #19: v5p-16, 1 prompt, 4096 tokens (success)
- **Run (2026-02-03 ~20:38–20:44):** `sample_llama8b_multihost_real_1prompt.yaml`.
- **Outcome:** **success**; W&B run `ytbn4l2u` (splendid-energy-31). Offline run dir `offline-run-20260203_203843-ytbn4l2u`.
