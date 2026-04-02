# vLLM Smoke Handoff: Llama 3.1 8B Instruct in `us-east5`

## Current state

- The rebuilt model cache in `us-east5` looks correct at:
  - `gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- Verified good layout:
  - root `config.json`
  - root `model-00001-of-00004.safetensors` ... `model-00004-of-00004.safetensors`
  - `original/params.json`
  - `original/consolidated.00.pth`
  - `original/tokenizer.model`
- Verified bad layout is gone:
  - no root `params.json`
  - no root `consolidated.00.pth`

## Failed job

- Iris job id:
  - `/ahmed/vllm-smoke-llama-3-1-8b-gcs-us-east5-a-20260401-193018`
- Final state:
  - `JOB_STATE_FAILED`
- Failure:
  - `python: can't open file '/app/scratch/vllm_server_100_prompt_smoke.py': [Errno 2] No such file or directory`

## Root cause

- The one-off smoke script exists only locally at:
  - `scratch/vllm_server_100_prompt_smoke.py`
- `scratch/` is git-ignored in the repo:
  - `.gitignore:232:/scratch`
- That means the script was never pushable from that path and therefore was not present in the Iris task bundle under `/app/`.
- The monitoring state file in `scratch/20260401-1931_monitoring_state.json` is also ignored for the same reason.

## Verified commands

### Check failed job status

```bash
uv run iris --config lib/iris/examples/marin.yaml job list --json \
  --prefix /ahmed/vllm-smoke-llama-3-1-8b-gcs-us-east5-a-20260401-193018
```

### Check failed job logs

```bash
uv run iris --config lib/iris/examples/marin.yaml job logs \
  --since-seconds 86400 \
  --max-lines 400 \
  /ahmed/vllm-smoke-llama-3-1-8b-gcs-us-east5-a-20260401-193018
```

## Exact broken submit command

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name vllm-smoke-llama-3-1-8b-gcs-us-east5-a-20260401-193018 \
  --extra marin:tpu \
  --extra marin:vllm \
  --tpu v5p-8 \
  --cpu 8 \
  --memory 32GB \
  --disk 50GB \
  --region us-east5 \
  --zone us-east5-a \
  -- python scratch/vllm_server_100_prompt_smoke.py \
    --model gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
    --max-model-len 4096 \
    --num-prompts 100
```

The only direct problem with that command is the script path under ignored `scratch/`.

## Next-agent recovery instructions

1. Move the smoke script into a tracked path.
   Suggested path:
   - `scripts/vllm_server_100_prompt_smoke.py`
   Alternative:
   - `lib/marin/src/marin/inference/vllm_server_100_prompt_smoke.py`

2. Reuse the existing local script content from:
   - `scratch/vllm_server_100_prompt_smoke.py`

3. Keep the current behavior:
   - start native `vllm serve` through `VllmEnvironment`
   - print vLLM startup logs
   - fail immediately if startup logs contain `Resolved architecture: MistralForCausalLM`
   - send `100` prompts through `/v1/chat/completions`

4. The user said it is fine if the prompt is the same 100 times.
   So the prompt generator can be simplified if desired; it is not important to vary prompts for this check.

5. Commit and push the tracked script before resubmitting the Iris job.
   Do not resubmit from `scratch/`.

6. Resubmit the job with the same resources and GCS model path, but update the Python entrypoint to the tracked script path.

7. Babysit until one of these happens:
   - success and logs confirm the model does not resolve as Mistral
   - failure with a real vLLM/model-loading error worth investigating

## Success condition for the rerun

- Job reaches `JOB_STATE_SUCCEEDED`
- Startup logs do not contain:
  - `Resolved architecture: MistralForCausalLM`
- The script completes the 100-request sweep against:
  - `gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`

## Useful note

- The model cache repair itself appears successful.
- The failed smoke job does **not** currently indicate a model-layout problem.
- It only indicates a bad submit path to an ignored local script.
