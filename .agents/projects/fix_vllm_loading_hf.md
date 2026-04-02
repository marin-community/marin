## Problem

Marin's cached Meta Llama artifacts under `gs://.../models/...` are currently
hybridized: they contain HF-native files like `config.json` and
`model.safetensors.index.json` at the root, but they also contain Meta original
files like `params.json` and `consolidated*.pth` at the root. That root-level
`params.json` is the trigger for vLLM's Mistral-format auto-detection, which is
why cached Llama artifacts can resolve as `MistralForCausalLM` instead of
Llama.

Relevant code and evidence:

- Model download entrypoint: [experiments/models.py](/Users/ahmed/code/marin/.claude/worktrees/cuddly-tumbling-nebula/experiments/models.py#L35)
- Current HF download path mapping:
  [lib/marin/src/marin/download/huggingface/download_hf.py](/Users/ahmed/code/marin/.claude/worktrees/cuddly-tumbling-nebula/lib/marin/src/marin/download/huggingface/download_hf.py#L99)
- Current task construction writes each file to
  `os.path.join(output_path, relative_file_path)`:
  [lib/marin/src/marin/download/huggingface/download_hf.py](/Users/ahmed/code/marin/.claude/worktrees/cuddly-tumbling-nebula/lib/marin/src/marin/download/huggingface/download_hf.py#L302)
- Historical flattening fix landed in `3fdd982b2`; before that the code used
  `file.split("/", 3)[-1]`
- Current vLLM still treats root `params.json` as Mistral format:
  [config.py](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/transformers_utils/config.py#L589)
- Current dense Mistral parser still rewrites to
  `architectures=["MistralForCausalLM"]`:
  [mistral.py](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/transformers_utils/configs/mistral.py#L50)

Current live GCS prefixes in `marin-us-central1` still show the broken root
layout:

- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B--d04e592/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`

## Goals

- Prove or falsify that deleting a broken Meta Llama cache prefix and
  re-downloading it with current Marin code fixes the layout issue in place.
- Start with the fastest relevant pilot:
  `meta-llama/Llama-3.1-8B-Instruct@0e9e39f`.
- Use the existing stable output path model cache scheme; do not introduce a
  new artifact naming scheme unless the pilot fails.
- Stop after the first failing gate and do not widen deletion scope until the
  pilot passes.

Non-goals:

- Removing the RL-local `MistralForCausalLM -> LlamaForCausalLM` shim during
  this experiment series
- Publishing a new `tpu-inference` or `vllm-tpu` dependency pair
- Broadly deleting all model caches before the pilot proves the repair works

## Proposed Solution

Treat this as a cache-repair experiment, not a code-change-first effort.

Current Marin code already preserves `original/*` during Hugging Face downloads.
The pilot should:

1. Snapshot the current broken prefix
2. Delete the entire model prefix on GCS, including `.executor_status`
3. Re-run the existing `experiments/models.py` step to recreate the same stable
   output path
4. Verify that root `params.json` and root `consolidated*.pth` do not return
5. Run a vLLM native smoke test against the repaired GCS path and confirm the
   logs no longer report `Resolved architecture: MistralForCausalLM`

Core executor invocation:

```python
from marin.execution import ExecutorMainConfig, executor_main
from experiments.models import llama_3_1_8b_instruct

executor_main(
    ExecutorMainConfig(prefix="gs://marin-us-central1"),
    steps=[llama_3_1_8b_instruct],
    description="repair llama 3.1 8b instruct cache",
)
```

This keeps the existing stable output path from
[experiments/models.py](/Users/ahmed/code/marin/.claude/worktrees/cuddly-tumbling-nebula/experiments/models.py#L38).
Deleting the entire prefix is required so the executor does not skip the step as
already-succeeded.

## Experiment Plan

### Experiment 0: One-time source/layout proof on local machine

Purpose: verify that current source repos and current Marin code imply the
correct nested layout before touching GCS.

Commands:

```bash
uv run python - <<'PY'
from huggingface_hub import HfFileSystem
from marin.download.huggingface.download_hf import _relative_path_in_source

fs = HfFileSystem()
for repo, rev in [
    ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f"),
    ("meta-llama/Llama-3.1-8B", "d04e592"),
    ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b"),
]:
    print("REPO", repo, rev)
    for path in [
        f"{repo}@{rev}/config.json",
        f"{repo}@{rev}/params.json",
        f"{repo}@{rev}/original/params.json",
        f"{repo}@{rev}/consolidated.00.pth",
        f"{repo}@{rev}/original/consolidated.00.pth",
    ]:
        print(path, fs.exists(path))
    print(_relative_path_in_source(f"{repo}@{rev}/original/params.json", repo))
    print()
PY
```

Success criteria:

- HF repo has no root `params.json`
- HF repo has `original/params.json`
- `_relative_path_in_source(...original/params.json...)` returns
  `original/params.json`

If this fails, stop. Do not delete any GCS prefixes.

### Experiment 1: Fast pilot on 8B-Instruct in `marin-us-central1`

Target prefix:

- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`

#### 1A. Snapshot current broken state

Commands:

```bash
MODEL_PREFIX="gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

gcloud storage ls "${MODEL_PREFIX}/" | sed "s#${MODEL_PREFIX}/##" | sort
gcloud storage ls "${MODEL_PREFIX}/original/" || true
gcloud storage cat "${MODEL_PREFIX}/provenance.json" | jq '.access_time, .links[:10]'
```

Expected baseline:

- root `params.json` exists
- root `consolidated.00.pth` exists
- `original/` does not exist

#### 1B. Delete the entire prefix

Important: delete the whole prefix, not individual files. If `.executor_status`
survives, the executor may skip the step.

Command:

```bash
gcloud storage rm "${MODEL_PREFIX}/**"
```

Post-delete check:

```bash
gcloud storage ls "${MODEL_PREFIX}/" || true
```

Success criterion:

- no objects remain under the prefix

#### 1C. Re-download using the existing stable executor step

Command:

```bash
uv run python - <<'PY'
from marin.execution import ExecutorMainConfig, executor_main
from experiments.models import llama_3_1_8b_instruct

executor_main(
    ExecutorMainConfig(prefix="gs://marin-us-central1"),
    steps=[llama_3_1_8b_instruct],
    description="repair llama 3.1 8b instruct cache",
)
PY
```

Success criterion:

- executor does not skip the step
- prefix is recreated successfully

#### 1D. Verify repaired artifact layout

Commands:

```bash
gcloud storage ls "${MODEL_PREFIX}/" | sed "s#${MODEL_PREFIX}/##" | sort
gcloud storage ls "${MODEL_PREFIX}/original/" | sed "s#${MODEL_PREFIX}/##" | sort

gcloud storage ls "${MODEL_PREFIX}/**" | sed "s#${MODEL_PREFIX}/##" | \
  rg '(^config\.json$|^params\.json$|^consolidated\.00\.pth$|^original/params\.json$|^original/consolidated\.00\.pth$)'

gcloud storage cat "${MODEL_PREFIX}/provenance.json" | \
  jq -r '.links[]' | rg 'original/(params\.json|consolidated\.00\.pth)$'
```

Success criteria:

- root `config.json` exists
- root `params.json` does **not** exist
- root `consolidated.00.pth` does **not** exist
- `original/params.json` exists
- `original/consolidated.00.pth` exists

If root `params.json` or root `consolidated.00.pth` reappears, stop. Do not
delete any additional prefixes. That means something beyond the historical
flattening bug is still rewriting the layout.

#### 1E. vLLM native smoke test against repaired GCS path

Run on a TPU worker or via the usual Ray/Fray path. Start with a single
`v5p-8`.

Command:

```bash
uv run python - <<'PY'
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

MODEL_PREFIX = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

env = VllmEnvironment(
    model=ModelConfig(
        name="llama-3.1-8b-instruct-repair-check",
        path=MODEL_PREFIX,
        engine_kwargs={"load_format": "runai_streamer", "max_model_len": 8192},
    ),
    mode="native",
    host="127.0.0.1",
    port=8000,
    timeout_seconds=3600,
)

with env:
    print("MODEL_ID", env.model_id)
    print("LOG_DIR", env.vllm_server.log_dir)
    print(env.logs_tail(max_lines=400))
PY
```

Success criteria:

- server starts
- `/v1/models` returns successfully
- logs do **not** contain `Resolved architecture: MistralForCausalLM`
- logs do show Llama/HF loading behavior consistent with the repaired layout

If the server still resolves as `MistralForCausalLM`, stop. The cache repair is
insufficient by itself, and the next step is a code fix such as forcing
`config_format="hf"` or filtering `original/**` out of model caches.

### Experiment 2: Repeat on 8B base in the same region

Target prefix:

- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B--d04e592`

Run the same sequence as Experiment 1 with:

- download step: `experiments.models.llama_3_1_8b`
- smoke-test model path updated to the 8B base prefix

Reason:

- same Llama family
- still relatively fast
- validates that the repair is not specific to Instruct packaging

Go/no-go:

- only run if Experiment 1 passes all gates

### Experiment 3: Optional quick sanity on 3.2-1B if RL small-model workflows use it

Target prefix if present in a region:

- `gs://<bucket>/models/meta-llama--Llama-3-2-1B--4e20de3`

Run only the layout verification sequence first:

- snapshot
- delete prefix
- re-download via `experiments.models.llama_3_2_1b`
- verify no root `params.json`

Run the vLLM smoke test only if the model is actively used in RL bring-up.

### Experiment 4: 70B-Instruct after the 8B pilot passes

Target prefix:

- `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`

Current live evidence already shows that this prefix contains mixed layouts,
including a nested duplicate subtree under the same model prefix. This makes
full-prefix deletion even more important.

Run the same sequence as Experiment 1 with:

- download step: `experiments.models.llama_3_3_70b_instruct`
- smoke-test on a suitable TPU target only after layout verification passes

Reason for ordering:

- 70B is much more expensive to re-download and smoke-test
- we should not touch it until the 8B repair path is proven

### Experiment 5: Multi-region rollout

Only run after Experiments 1 and 2 pass, and after Experiment 4 passes if 70B
is required in those regions.

#### 5A. Discover affected buckets

Commands:

```bash
for bucket in $(gcloud storage buckets list --format='value(name)' | rg '^marin-'); do
  echo "=== $bucket ==="
  gcloud storage ls "gs://$bucket/models/" 2>/dev/null | \
    rg 'meta-llama--Llama-3-(2-1B|1-8B|1-8B-Instruct|3-70B-Instruct)'
done
```

#### 5B. Roll out bucket-by-bucket

For each bucket containing a target prefix:

1. snapshot prefix
2. delete full prefix with `gcloud storage rm "${MODEL_PREFIX}/**"`
3. re-download using the same executor step with `ExecutorMainConfig(prefix="gs://<bucket>")`
4. verify repaired layout

Do not batch-delete all regions first. Work bucket-by-bucket so a failure
does not fan out.

## Implementation Outline

1. Run the local source/layout proof to verify current Marin code preserves
   `original/*`.
2. Execute the 8B-Instruct pilot in `marin-us-central1`, deleting the full
   prefix and recreating it via the existing executor step.
3. Validate the repaired object layout before spending TPU time on vLLM.
4. Run a native vLLM smoke test and inspect logs for the resolved architecture.
5. Repeat on 8B base, then optionally 3.2-1B, then 70B-Instruct.
6. Roll the repair out region-by-region only after the pilot passes.

## Notes

- `experiments/models.py` already uses stable overridden output paths, which is
  exactly what we want. The plan is to repair those paths in place.
- Deleting only individual root files is not enough. The executor can skip
  reruns if `.executor_status` survives, and mixed legacy files can persist.
- Do not include `meta-llama--Llama-3-1-8B--main` in the first rollout. It is
  not one of the pinned `experiments/models.py` artifacts and should be handled
  separately after the pinned revisions are repaired.
- The 70B prefix currently contains more corruption than the 8B prefixes,
  including a nested duplicate subtree under the same model path. That is a
  strong signal to delete the full prefix, not to attempt surgical cleanup.

## Future Work

- If any repaired prefix still resolves as `MistralForCausalLM`, implement a
  code hardening change before more cache churn:
  - expose `config_format="hf"` in Marin's vLLM wrapper, or
  - add `hf_urls_glob` support to `experiments/models.py` and exclude
    `original/**` from Meta Llama caches entirely
- Once cache repair is proven across active buckets, remove now-obsolete
  workarounds that only existed for old `Qwen2ForCausalLM` registry behavior.
- Revisit whether the RL-local `MistralForCausalLM` alias patch can be removed
  after both the cache layout and dependency stack are clean.
