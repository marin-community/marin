# Debugging log for IFBench vLLM 70B on Iris v5p-8

## Overview

Goal: explain why `meta-llama/Llama-3.3-70B-Instruct` did not serve through the IFBench Iris vLLM runner on `v5p-8`, even though 70B inference is expected to work on that hardware. Fix the launch path or isolate the real infrastructure/runtime issue.

## Initial Status

Previous overnight attempts:
- `v6e-4` 70B launch failed.
- `v5p-8` with tensor parallel 8 failed because the host exposed 4 JAX devices; TP=8 was invalid.
- `v5p-8` with tensor parallel 4 and `max_num_seqs=256` failed before serving.
- `v5p-8` with tensor parallel 4 and `max_num_seqs=64` still failed before serving.
- Final retry with a parent-process heartbeat exposed one bad-node attempt: `Couldn't open iommu group /dev/vfio/1`; Iris retried automatically.
- Replacement attempt emitted startup heartbeats through 08:32 UTC, then exited before vLLM became ready. No rollout object was written. Peak container memory was ~15.6GB, so not a host cgroup OOM.

The unresolved gap is the second attempt's missing vLLM stderr/stdout at process exit. Need a repro that writes `/tmp/vllm_server_*` logs to durable storage.

## Hypothesis 1: launch wrapper loses native vLLM diagnostics on late startup failure

The Marin `VllmEnvironment` records native vLLM stdout/stderr under a temp dir and logs a tail if the subprocess exits before readiness. In the second r4 attempt, Iris logs do not show the final failure tail. Either the parent process was killed, the tail was not reached, or Iris log collection lost the last lines.

## Changes To Make

- Add a dedicated IFBench 70B startup probe script that starts `VllmEnvironment`, waits for readiness, optionally sends one prompt, and always copies vLLM diagnostics to local output and/or GCS in `finally`.
- Run it on Iris `v5p-8` interactive priority with TP=4 and conservative server settings.
- Keep the probe to 1 prompt / startup only, not a 20k rollout.

## Results

### Probe r1: TP=4 v5p-8 reached model load, then failed on disk

Job: `/ahmed/ifbench-vllm-70b-probe-lock-r1`

Durable diagnostics: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_20260427T162702Z`

Findings:
- TPU/JAX startup succeeded on `v5p-8`.
- JAX exposed 4 devices, so TP=4 is the correct tensor-parallel size on this host.
- vLLM initialized the TP=4 mesh and selected the Pallas V1 backend.
- vLLM then failed while Hugging Face CAS downloaded model shards.
- Root error: `No space left on device (os error 28)`.
- HF cache was under `/root/.cache/huggingface`, on the small root overlay, despite the Iris job requesting `--disk 750GB`.

Interpretation: this is not evidence that 70B cannot serve on `v5p-8`. The immediate failure is cache placement. Iris defaults `HF_HOME` to `~/.cache/huggingface`, and Hugging Face is not one of Iris's standard hostPath cache mounts. The requested disk is available as pod ephemeral storage, but the model loader was filling the wrong filesystem.

### Fix for next probe

- Force `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, and `VLLM_ASSETS_CACHE` under `/app/.hf_cache`.
- `/app` is the Iris workdir emptyDir and should be covered by the `--disk 750GB` request.
- Preserve `df`, `mount`, and cache `du` before and after vLLM startup in `status.json`.
- Apply the same cache placement to the production IFBench Iris vLLM runner.

### Probe r2: cache fixed, still failed on host memory

Job: `/ahmed/ifbench-vllm-70b-probe-cache-r2`

Durable diagnostics: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_cache_20260427T163738Z`

Findings:
- The cache fix worked: model shards downloaded under `/app/.hf_cache`, not `/root/.cache/huggingface`.
- `/app` is tmpfs on this Iris image, so the 132GB model cache also counted against container memory.
- The job requested 240GB RAM and failed by host-memory pressure at ~237.6GB / 240GB while vLLM was still starting.

Interpretation: the second failure was a container memory request issue, not a TPU/vLLM incompatibility.

### Probe r3: v5p-8 succeeds with cache fix + 400GB memory

Job: `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`

Durable diagnostics: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v5p8_400gb_20260427T165651Z`

Findings:
- `vllm serve meta-llama/Llama-3.3-70B-Instruct` became ready on `v5p-8` with TP=4, `max_model_len=4096`, and `max_num_seqs=16`.
- The probe served a one-prompt completion successfully.
- Peak host memory was ~281GB under the 400GB request.
- Native logs showed model init, KV cache allocation, and precompile completed without OOM.

Conclusion: 70B inference works on `v5p-8` for this runner when HF cache placement and host-memory request are correct. For a full 20k v5p run, use at least 400GB memory and keep the cache env vars under `/app/.hf_cache`.

## Future Work

- [ ] Decide whether the production IFBench runner should always preserve vLLM diagnostics on startup failure.
- [ ] Launch a full 20k `v5p-8` run only if we want a direct race against the active `v6e-8` run; otherwise keep v5p as the proven fallback.
