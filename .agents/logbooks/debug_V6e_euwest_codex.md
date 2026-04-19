# Debug Logbook: v6e-4 eu-west4 TPU/vLLM Bring-up in Codex

**Goal**: Isolate why Marin's `v6e-4` inference path fails in `europe-west4` during vLLM TPU engine initialization while the same eval stack succeeds in `us-east1` and `us-east5`.

**Branch**: `alignment_function`
**Related logbooks**:
- `.agents/logbooks/validate_bloom.md` (`EXP-006`)
- `.agents/logbooks/debug_V6e_euwest_claude.md`

---

## 2026-04-03: Initial Codex State Capture

### Current status at handoff

- `us-east1` SFT inference succeeded on `v6e-4`
  - job: `/ahmed/bloom-eval-sft-us-east1-v6e4`
  - output: `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d`
- `us-east5` DPO inference succeeded on `v6e-4`
  - job: `/ahmed/bloom-eval-dpo-east5-v6e4-r2`
  - outputs:
    - `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f`
    - `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8`
- the previously failing `beta0.01_lr7.5e-7_seed0` checkpoint was re-run on `us-east5` `v6e-4` and succeeded
  - job: `/ahmed/bloom-eval-dpo-east5-lr75e7-v6e4-r1`
  - output: `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220`
- `eu-west4` remains the only regional failure
  - job: `/ahmed/bloom-eval-dpo-europe-west4-v6e4`
  - failed output prefix: `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220`

### Confirmed failure signature

The eu-west4 failure happens after prompts load and after the native vLLM TPU server starts, but before serving becomes ready:

```text
devices = sorted(devices, key=lambda x: x.coords)
AttributeError
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
RuntimeError: vLLM server process exited before becoming ready.
```

From the eu-west4 task logs:
- node: `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`
- model: `gs://marin-eu-west4/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849`
- prompts: `gs://marin-eu-west4/alignment/gpt-4.1-eval-split/`
- TPU shape: `v6e-4`
- sampling: `n=3`, `temperature=0.7`, `max_tokens=1500`

### Facts established before Codex investigation

1. This is not the earlier checkpoint metadata issue from `us-east5`.
   - the `chat_template` problem was fixed by normalizing `tokenizer_config.json`
   - the same `beta0.01_lr7.5e-7_seed0` checkpoint now succeeds on `us-east5`
2. This is not a generic memory issue with 8B models on `v6e-4`.
   - SFT and three DPO checkpoints now succeed on `v6e-4`
3. The failure is region- or worker-specific to the eu-west4 run, not model-specific.
   - exact same checkpoint succeeds on `us-east5`
   - exact same Marin script and sampling params are used across regions

### Working hypotheses

1. **Raw JAX device objects differ on eu-west4**
   - `tpu_inference` assumes JAX devices expose `.coords`
   - on the eu-west4 worker, the device objects handed back by JAX may not expose that attribute
2. **The eu-west4 worker has a TPU runtime or libtpu difference**
   - same Python wheels can still sit on top of different host TPU runtime behavior
   - the user is skeptical of this, so it needs direct proof rather than inference
3. **One bad worker is enough to explain the failure**
   - if a fresh eu-west4 job on another worker succeeds, then this is likely node-specific
4. **`tpu_inference` has a brittle mesh-init assumption**
   - if `.coords` is not guaranteed, then eu-west4 may just be the first place where that assumption breaks

### Codex debugging plan

0. **Execution constraint**
   - only launch diagnostics on the Iris cluster
   - do not use local jobs for this thread because the local environment does not match the TPU worker environment
   - for each new diagnostic, prefer:
     - an Iris CPU-only job first to validate Python package resolution and imports inside the task image
     - then an Iris TPU job to inspect the TPU runtime / JAX device behavior
1. Submit a tiny diagnostic job to `eu-west4` `v6e-4` that prints:
   - `jax.__version__`, `jaxlib.__version__`
   - device type and available attributes
   - whether `jax.devices()[0]` has `.coords`
   - any TPU runtime / libtpu package metadata visible from Python or `pip`
2. Submit the same diagnostic job to a known-good `us-east5` `v6e-4` worker.
3. Compare the raw device objects between regions before touching Marin code.
4. If `.coords` is absent only in eu-west4, prove it in this logbook and decide whether the next action is:
   - a eu-west4 retry on a fresh worker
   - a code-level fallback around `.coords`
   - or a dependency/runtime pin

### Immediate next action

Launch a minimal cross-region diagnostic pair on the Iris cluster:
1. CPU-only validation jobs in `eu-west4` and `us-east5` to confirm the task image resolves the expected packages
2. matching `v6e-4` TPU jobs in `eu-west4` and `us-east5` to inspect raw JAX device attributes directly

---

## 2026-04-03: EXP-001 — Iris CPU-Only and TPU Runtime Inspection

**Hypothesis**: The eu-west4 failure can be explained by a difference visible from inside the Iris task environment itself, either in installed package state or in the raw JAX device objects returned on TPU workers.

**Diagnostic helper**:
- `experiments/posttrain/inspect_jax_runtime.py`

**Execution rule for this experiment**:
- Iris cluster only
- CPU-only validation first
- TPU jobs second
- same dependency extras as the failing path: `--extra marin:tpu --extra marin:vllm`

**Planned commands**:

CPU-only control jobs:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name debug-v6e-cpu-east5-r1 \
  --extra marin:tpu --extra marin:vllm \
  --cpu 4 --memory 16GB --disk 20GB --region us-east5 \
  -- python experiments/posttrain/inspect_jax_runtime.py --label cpu-us-east5

uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name debug-v6e-cpu-euw4-r1 \
  --extra marin:tpu --extra marin:vllm \
  --cpu 4 --memory 16GB --disk 20GB --region europe-west4 \
  -- python experiments/posttrain/inspect_jax_runtime.py --label cpu-europe-west4
```

TPU jobs:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name debug-v6e-tpu-east5-r1 \
  --extra marin:tpu --extra marin:vllm \
  --tpu v6e-4 --cpu 4 --memory 16GB --disk 20GB --region us-east5 \
  -- python experiments/posttrain/inspect_jax_runtime.py --label tpu-us-east5

uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name debug-v6e-tpu-euw4-r1 \
  --extra marin:tpu --extra marin:vllm \
  --tpu v6e-4 --cpu 4 --memory 16GB --disk 20GB --region europe-west4 \
  -- python experiments/posttrain/inspect_jax_runtime.py --label tpu-europe-west4
```

**What these jobs should answer**:
1. Are the same Python distributions installed in both regions inside the Iris task image?
2. Does `jax.devices()[0]` expose `.coords` on a working `us-east5` TPU worker?
3. Does `.coords` exist or fail in the same way on `eu-west4` TPU workers?
4. Is the issue visible before vLLM starts, which would narrow the bug from “vLLM runtime” to “raw JAX/TPU runtime object shape”?

**Observed results**:

CPU-only Iris jobs:
- `us-east5`: `/ahmed/debug-v6e-cpu-east5-r1`
- `eu-west4`: `/ahmed/debug-v6e-cpu-euw4-r1`

TPU Iris jobs:
- `us-east5`: `/ahmed/debug-v6e-tpu-east5-r1`
- `eu-west4`: `/ahmed/debug-v6e-tpu-euw4-r1`

Package parity:
- both regions resolved the same package set inside Iris tasks:
  - `jax==0.8.0`
  - `jaxlib==0.8.0`
  - `vllm-tpu==0.13.2.post6`
  - `libtpu==0.0.24`
- this rules out a simple Python package mismatch between regions

Working TPU control (`us-east5`):
- TPU node:
  - `marin-tpu-v6e-4-us-east5-b-20260403-2209-607863ea`
- `jax.devices()` succeeded
- returned `jaxlib._jax.Device` objects with:
  - `repr`: `TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)` and peers
  - `.coords` populated for all 4 devices
- JAX client platform version:
  - `PJRT C API`
  - `TFRT TPU v6 lite`
  - `Built on Oct 1 2025 15:38:51 ...`

Failing TPU diagnostic (`eu-west4`):
- TPU node:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`
- `jax.devices()` failed before any device objects were returned
- exact failure:
  - `TPU initialization failed: TPU_RET_CHECK failure`
  - `GetChip(i)->location().index_on_host() == i (0 vs. 1)`
- because JAX backend init itself fails, the earlier vLLM / `tpu_inference`
  `.coords` crash is downstream of a lower-level TPU runtime problem on this
  worker

**Interpretation update**:
- This is **not** a vLLM-only bug.
- This is **not** explained by mismatched Python wheels across regions.
- The bad eu-west4 worker fails at raw TPU/JAX bring-up, while a working
  `us-east5` `v6e-4` worker on the same package versions exposes the expected
  `.coords`.
- Current best explanation: the eu-west4 `v6e-4` TPU worker
  `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88` is bad.

**Immediate next action**:
1. Confirm the TPU VM identity via `gcloud compute tpus tpu-vm list`.
2. Replace the bad eu-west4 TPU worker.
3. Re-run the TPU diagnostic on a fresh eu-west4 `v6e-4` worker before
   re-running the full eval job.

---

## 2026-04-03: EXP-002 — Replace the Original eu-west4 `v6e-4` Worker and Re-test

**Hypothesis**: The original failing eu-west4 worker is individually bad. If it
is deleted and replaced, the same minimal TPU diagnostic should start working.

**Original failing worker confirmed from GCP**:
- `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`

**Action taken**:
- deleted:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`
- replacement provisioned:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab`

**Re-test job**:
- `/ahmed/debug-v6e-tpu-euw4-r2`

**Re-test result**:
- the replacement worker fails with the **same** raw JAX TPU init error:
  - `TPU initialization failed: TPU_RET_CHECK failure`
  - `GetChip(i)->location().index_on_host() == i (0 vs. 1)`

**Interpretation update**:
- deleting the original eu-west4 `v6e-4` worker did **not** fix the issue
- this rules out "one bad node" as the primary explanation
- current evidence is consistent with a broader issue in the eu-west4
  `v6e-4` pool or TPU runtime environment, not Marin code and not a one-off host

**Current best conclusion**:
- `us-east5` `v6e-4`: JAX TPU init works and returns `TpuDevice(..., coords=...)`
- `europe-west4` `v6e-4`: raw `jax.devices()` fails on two separate workers
- therefore the original vLLM / `tpu_inference` `.coords` failure is a symptom
  of a deeper eu-west4 `v6e-4` TPU runtime problem

**Follow-up experiment launched**:
- `/ahmed/debug-v6e8-tpu-euw4-r1`
- purpose: determine whether the issue is specific to eu-west4 `v6e-4`, or all
  eu-west4 TPU bring-up

---

## 2026-04-03: EXP-003 — eu-west4 `v6e-8` Control

**Hypothesis**: If eu-west4 TPU bring-up is broadly broken, a minimal `v6e-8`
diagnostic should fail the same way as eu-west4 `v6e-4`. If it succeeds, the
problem is likely isolated to the eu-west4 `v6e-4` pool.

**Job**:
- `/ahmed/debug-v6e8-tpu-euw4-r1` — SUCCEEDED

**Worker**:
- `marin-tpu-v6e-8-europe-west4-a-20260403-2329-df171a1a`

**Result**:
- `jax.devices()` succeeded on eu-west4 `v6e-8`
- returned 8 `TpuDevice(..., coords=...)` devices
- same package versions as every other diagnostic:
  - `jax==0.8.0`
  - `jaxlib==0.8.0`
  - `vllm-tpu==0.13.2.post6`
  - `libtpu==0.0.24`
- JAX client platform version matches the working path:
  - `PJRT C API`
  - `TFRT TPU v6 lite`
  - `Built on Oct 1 2025 15:38:51 ...`

**Conclusion update**:
- eu-west4 is **not** generically broken for TPU/JAX bring-up
- eu-west4 `v6e-8` works
- eu-west4 `v6e-4` failed on two separate workers with the same raw JAX TPU
  init error
- this narrows the issue to the eu-west4 `v6e-4` slice pool or topology path
  specifically

**Current best diagnosis**:
- working:
  - `us-east5 v6e-4`
  - `eu-west4 v6e-8`
- broken:
  - `eu-west4 v6e-4`
- therefore the failure is not caused by Marin code, not caused by package
  skew, and not explained by a single bad worker

**Immediate next actions**:
1. Avoid eu-west4 `v6e-4` for alignment inference until the underlying TPU pool
   issue is understood or fixed upstream.
2. If eu-west locality matters, test whether eu-west4 `v6e-8` can safely serve
   the same vLLM workload used by the alignment pipeline.
3. Capture this as a TPU pool/runtime issue rather than a Marin eval bug.

---

## 2026-04-03: EXP-004 — Actual eu-west4 Alignment Inference Re-run on `v6e-8`

**Why this experiment exists**:
- The `v6e-8` control proved that raw JAX TPU bring-up works in eu-west4.
- The next missing proof is whether the real alignment inference stack also
  works there when we avoid the broken `v6e-4` slice shape.

**Terminology correction from EXP-003**:
- Do not describe this as a generic "topology" issue.
- The evidence is narrower:
  - `us-east5 v6e-4`: works
  - `eu-west4 v6e-8`: works
  - `eu-west4 v6e-4`: fails on two distinct workers during raw JAX TPU init
- Current best diagnosis: eu-west4 `v6e-4` has a slice-shape-specific TPU/JAX
  runtime failure for this stack.

**Code change before launch**:
- parameterized `experiments/posttrain/eval_llama3_8b_alignment.py` with:
  - `--tpu-type`
  - `--run-label`
- rationale:
  - request `v6e-8` without changing tensor parallelism
  - give the rerun a fresh output prefix instead of reusing the failed
    eu-west4 `v6e-4` path

**Planned command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-dpo-europe-west4-v6e8-r1 \
  --cpu 4 --memory 16GB --disk 10GB --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region europe-west4 \
    --tpu-type v6e-8 \
    --run-label v6e8r1 \
    --target beta001_lr75e7_seed0
```

**Success criterion**:
- the child inference step reaches `JOB_STATE_SUCCEEDED`
- and writes a fresh eu-west4 inference artifact for
  `marin_dpo_beta001_lr75e7_seed0_bloom_speceval`

**Launch observed**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e8-r1`
- child inference job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e8-r1/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1-inference_b19c07ee-cefaac57`
- executor step name:
  - `eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1/inference`
- fresh output prefix:
  - `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1/inference-1cb501`

**Current status**:
- the wrapper job is running normally
- the child inference job is pending only on eu-west4 `v6e-8` capacity:
  - `Scheduler: Insufficient TPUs (need 8, available 0)`
  - `Autoscaler: Waiting for workers in scale group 'tpu_v6e_8-europe-west4-a'`

**Interpretation so far**:
- this is already better than the broken eu-west4 `v6e-4` path
- the rerun reached the real inference launch path and is waiting on capacity,
  not failing during raw JAX TPU initialization
- next checkpoint for this thread is whether the `v6e-8` child job starts and
  completes successfully once capacity arrives

**Final result**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e8-r1` — `JOB_STATE_SUCCEEDED`
- child inference job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e8-r1/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1-inference_b19c07ee-a5e2f75a`
  - `JOB_STATE_SUCCEEDED`
- wrapper noted one preemption during the earlier capacity wait / relaunch path,
  but the final active child completed successfully

**Observed runtime**:
- `marin.inference.vllm_server vLLM environment ready`
- steady inference throughput around `20-21 items/s`
- executor wall time:
  - `711.48s`

**Artifacts confirmed**:
- inference output:
  - `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1/inference-1cb501`
- confirmed files:
  - `shard_00000.jsonl.gz`
  - `shard_00001.jsonl.gz`
  - `artifacts/vllm_metrics.json`
- experiment metadata:
  - `gs://marin-eu-west4/experiments/eval_llama3_8b_alignment-40ea31.json`

**Conclusion update**:
- eu-west4 is viable for this Marin alignment inference path on `v6e-8`
- the previously observed eu-west failure remains isolated to the `v6e-4`
  slice-shape/runtime path
- this rules out "eu-west4 itself is broken" for the workload under test

**Immediate next actions**:
1. Treat `eu-west4 v6e-8` as a known-good workaround for this checkpoint.
2. If we want a durable fix, file or escalate the remaining `eu-west4 v6e-4`
   TPU/JAX bring-up bug separately from the alignment eval work.
3. Proceed to the GPT-4.1 judge stage using the new eu-west4 inference artifact
   if regional locality still matters.

---

## 2026-04-03: EXP-005 — Fresh eu-west4 `v6e-4` Inference Re-try After `v6e-8` Success

**Hypothesis**:
- The earlier eu-west4 `v6e-4` failures may still have been transient bad-worker
  or bad-slice events, despite the two failing diagnostics.
- Since `v6e-4` works in other regions and `v6e-8` now works in eu-west4 for the
  exact same checkpoint and code path, a clean eu-west4 `v6e-4` re-run is worth
  testing before treating the failure as durable.

**Why rerun now**:
- the exact eval stack is now validated end-to-end on eu-west4 `v6e-8`
- all artifacts are already mirrored in-region
- the inference entry point already supports:
  - `--tpu-type`
  - `--run-label`
- so we can run a fresh eu-west4 `v6e-4` retry without reusing the earlier
  failed output prefix

**Planned command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-dpo-europe-west4-v6e4-r2 \
  --cpu 4 --memory 16GB --disk 10GB --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region europe-west4 \
    --tpu-type v6e-4 \
    --run-label v6e4r2 \
    --target beta001_lr75e7_seed0
```

**Success criterion**:
- the child inference step reaches `JOB_STATE_SUCCEEDED`
- and writes a new eu-west4 `v6e-4` inference artifact under the `v6e4r2` label

**Failure criterion**:
- the run reproduces the earlier JAX / vLLM bring-up failure on a fresh eu-west4
  `v6e-4` allocation, which would strengthen the case that this is a durable
  eu-west4 `v6e-4` runtime bug rather than a transient event

**Observed result**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r2` — `JOB_STATE_FAILED`
- child inference job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r2/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r2-inference_9ded15f9-5afd0a44`
  - `JOB_STATE_FAILED`
- child error summary:
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`

**Interpretation update**:
- this fresh eu-west4 `v6e-4` retry did not make it through engine bring-up
- that is now another real inference-path reproduction, not just the earlier
  diagnostics
- current evidence still points to a durable eu-west4 `v6e-4` runtime problem,
  but one more fresh retry is still cheap enough to run

---

## 2026-04-03: EXP-006 — Second Fresh eu-west4 `v6e-4` Re-try

**Hypothesis**:
- if the previous retries were unlucky allocations, another fresh `v6e-4`
  assignment could still succeed

**Planned command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-dpo-europe-west4-v6e4-r3 \
  --cpu 4 --memory 16GB --disk 10GB --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region europe-west4 \
    --tpu-type v6e-4 \
    --run-label v6e4r3 \
    --target beta001_lr75e7_seed0
```

**Launch observed**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r3`
- current state:
  - `JOB_STATE_RUNNING`
  - wrapper task is still in the build/startup phase

**Observed result**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r3` — `JOB_STATE_FAILED`
- child inference job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r3/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r3-inference_9d654aaa-fafb09be`
  - `JOB_STATE_FAILED`
- active TPU worker from vLLM logs:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab`
- same engine / mesh-init failure reproduced again:
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
  - `devices = sorted(devices, key=lambda x: x.coords)`
  - `AttributeError`

**Interpretation update**:
- this is now another full end-to-end reproduction on the real inference path
- it reused the same eu-west4 `v6e-4` worker that had already shown the raw JAX
  TPU bring-up problem earlier
- at this point, repeated blind retries are not buying new information

**Current best conclusion**:
- working:
  - `us-east5 v6e-4`
  - `eu-west4 v6e-8`
- failing repeatedly:
  - `eu-west4 v6e-4`
- and the fresh `v6e-4` retries are continuing to land on the same bad eu-west4
  `v6e-4` worker / slice path

**Immediate next actions**:
1. Stop blind `v6e-4` reruns.
2. If we want to keep pushing eu-west4 `v6e-4`, escalate back to TPU-node /
   pool debugging rather than more inference retries.
3. Otherwise use `eu-west4 v6e-8` or a different region for actual inference.

---

## 2026-04-03: EXP-007 — Exact Reproduction Recipe

**What this reproduces**:
- the eu-west4 `v6e-4` inference-path failure for
  `beta0.01_lr7.5e-7_seed0`
- not just the raw JAX diagnostic
- the real Marin alignment eval inference job

**Exact submit command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-dpo-europe-west4-v6e4-r3 \
  --cpu 4 --memory 16GB --disk 10GB --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region europe-west4 \
    --tpu-type v6e-4 \
    --run-label v6e4r3 \
    --target beta001_lr75e7_seed0
```

**Exact resulting jobs from the observed reproduction**:
- wrapper:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r3`
- child:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r3/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r3-inference_9d654aaa-fafb09be`

**Exact output path for the reproduced run**:
- `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r3/inference-81d5df`

**Exact log command to verify the failure**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job logs \
  /ahmed/bloom-eval-dpo-europe-west4-v6e4-r3 --since-seconds 900
```

**Exact vLLM command from the failing worker logs**:
```text
['/app/.venv/bin/vllm', 'serve',
 'gs://marin-eu-west4/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849',
 '--trust-remote-code',
 '--host', '127.0.0.1',
 '--port', '8000',
 '--load-format', 'runai_streamer',
 '--tensor-parallel-size', '4',
 '--max-model-len', '4096',
 '--gpu-memory-utilization', '0.9']
```

**Exact worker observed in the reproduced failure**:
- `marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab`

**Exact failure markers to grep for**:
```text
node_name=marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
devices = sorted(devices, key=lambda x: x.coords)
AttributeError
RuntimeError: vLLM server process exited before becoming ready.
```

**Observed sequence in the failing run**:
1. prompts load successfully
2. Marin starts native vLLM:
   - `Starting vLLM environment`
   - `Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
3. vLLM starts engine init on the eu-west4 `v6e-4` TPU worker
4. `tpu_inference.utils.make_optimized_mesh()` hits:
   - `devices = sorted(devices, key=lambda x: x.coords)`
   - `AttributeError`
5. APIServer exits with:
   - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
6. Marin wrapper surfaces:
   - `RuntimeError: vLLM server process exited before becoming ready.`

**Control that succeeds**:
- same script, same checkpoint, same prompts, same sampling params, but:
  - `--tpu-type v6e-8`
  - `--run-label v6e8r1`
  - wrapper job `/ahmed/bloom-eval-dpo-europe-west4-v6e8-r1`
- this completed successfully and wrote:
  - `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e8r1/inference-1cb501`

**Minimal claim supported by this repro recipe**:
- eu-west4 `v6e-4` can reproduce a real inference-path vLLM TPU engine bring-up
  failure for this checkpoint and stack
- eu-west4 `v6e-8` on the same stack does not reproduce it

---

## 2026-04-03: EXP-008 — One Final Fresh eu-west4 `v6e-4` Retry

**Exact submit command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-dpo-europe-west4-v6e4-r4 \
  --cpu 4 --memory 16GB --disk 10GB --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region europe-west4 \
    --tpu-type v6e-4 \
    --run-label v6e4r4 \
    --target beta001_lr75e7_seed0
```

**Observed result**:
- wrapper job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r4` — `JOB_STATE_FAILED`
- child inference job:
  - `/ahmed/bloom-eval-dpo-europe-west4-v6e4-r4/eval-marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r4-inference_b0d049bc-d687ad4a`
  - `JOB_STATE_FAILED`
- output path:
  - `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval_v6e4r4/inference-5f8c0d`
- experiment metadata:
  - `gs://marin-eu-west4/experiments/eval_llama3_8b_alignment-510951.json`

**Failure details**:
- active TPU worker from vLLM logs:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab`
- same repeated failure:
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
  - `RuntimeError: vLLM server process exited before becoming ready.`
  - `devices = sorted(devices, key=lambda x: x.coords)`
  - `AttributeError`

**Interpretation update**:
- this final retry adds no new contrary evidence
- it hit the same eu-west4 `v6e-4` worker and reproduced the same failure
- across repeated full inference runs, eu-west4 `v6e-4` remains broken for this
  stack while eu-west4 `v6e-8` remains good

**Current conclusion after final retry**:
- repeated eu-west4 `v6e-4` retries are no longer justified as an experiment
- the next useful action is TPU pool / worker debugging or simply avoiding
  eu-west4 `v6e-4` for this workload
