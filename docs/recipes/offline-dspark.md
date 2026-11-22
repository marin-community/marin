# Train a DSpark draft from a completed SFT export

`experiments.post_training.dspark_sft` trains a Qwen3-style DSpark draft on
fixed assistant replies. It runs the frozen Grug teacher on the rendered token
IDs to collect hidden states. The trainer applies the teacher's final RMSNorm,
gated norm and output projection to compute the distillation targets.
[Offline DSpark experiment](https://github.com/marin-community/marin/issues/10227#2-experimental-setup)

| SFT teacher | Hidden size | Teacher layers | Draft feature taps | Pre-norm logits tap |
| --- | ---: | ---: | --- | ---: |
| Snowball | 2560 | 26 | 2, 12, 22 | 26 |
| Mochi | 6144 | 48 | 2, 24, 45 | 48 |

Taps count completed decoder blocks. The three feature taps feed the drafter.
The last tap feeds the frozen teacher logits head. The draft has three
full-attention Qwen3 layers, an FFN 1.5 times the teacher's width, a block
size of 8, a rank-256 Markov head and a confidence head. It uses the most
frequent 32,000 supervised token IDs from training, with the corresponding
frozen teacher head rows. The schedule is one epoch capped at the requested
step count, with Muon, linear decay and 0.1 CE plus 0.9 total variation loss.
[Reference training recipe](https://github.com/atutej/spec-experiments/tree/85d1dd3d1f793153d3e6c2aeab3f3c940987fd9d/pipelines/speculator_training)

## Stage the frozen inputs

Use the HF directory of an already completed SFT export, including its matching
tokenizer and chat template. Record the producer artifact's exact identity
(for example `<name>@<version>:<fingerprint>` with the exact HF subdirectory)
or a Hub repo and full commit SHA. Both families use the same typed
`CompletedSFT` input. Mochi requires a completed export supplied by the
caller. The Mochi training preset is a recipe for producing an export.

Stage the selected, versioned SFT mixture as normalized Marin chat Parquet
shards in separate `train/` and `validation/` directories. Each row carries
`messages` and may carry `chat_template_kwargs`. Freeze the intended mixture
proportions and holdout membership before creating manifests. Use shards from
the same storage region as the execution host.

Set `TEACHER_DIR`, `TEACHER_IDENTITY`, `FAMILY` (`snowball` or `mochi`),
`MIX_DIR`, and `MARIN_PREFIX` to the selected identities and absolute local
paths. Set `TEACHER_REPLICAS`, `TRAINER_RANKS`, and `MAX_STEPS` to the intended
single-host allocation and training budget. Reserve enough shared disk for
the prepared Arrow dataset and its temporary build files.

```bash
uv run python - "$TEACHER_DIR" "$MIX_DIR" "$TEACHER_IDENTITY" "$FAMILY" \
  "$TEACHER_REPLICAS" "$TRAINER_RANKS" "$MAX_STEPS" > recipe.json <<'PY'
import hashlib
import json
import sys
from pathlib import Path

teacher, mix = map(Path, sys.argv[1:3])
identity, family = sys.argv[3:5]
teacher_replicas, trainer_ranks, max_steps = map(int, sys.argv[5:])

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

teacher_manifest = teacher / "dspark-teacher.json"
teacher_files = sorted(p for p in teacher.iterdir() if p.is_file() and p != teacher_manifest)
teacher_files += sorted(p for p in (teacher / "chat_templates").rglob("*") if p.is_file())
teacher_manifest.write_text(json.dumps({str(p.relative_to(teacher)): digest(p) for p in teacher_files}))
mix_manifest = mix / "dspark-mix.json"
mix_manifest.write_text(json.dumps({
    split: {str(p.relative_to(mix)): digest(p) for p in sorted((mix / split).rglob("*.parquet"))}
    for split in ("train", "validation")
}))
print(json.dumps({
    "teacher": {
        "family": family,
        "identity": identity,
        "directory": str(teacher),
        "manifest_sha256": digest(teacher_manifest),
    },
    "data": {"directory": str(mix), "manifest_sha256": digest(mix_manifest)},
    "max_steps": max_steps,
    "teacher_replicas": teacher_replicas,
    "trainer_ranks": trainer_ranks,
}, indent=2))
PY

uv run python -m experiments.post_training.dspark_sft --config recipe.json
uv run python -m experiments.post_training.dspark_sft --config recipe.json --stage preflight
uv run python -m experiments.post_training.dspark_sft --config recipe.json --stage prepare
```

The default command prints the artifact graph. Preflight checks the teacher's
shape, required tensors and tokenizer, plus the mix manifest, on CPU. Preparation verifies
the data shard digests, renders with Levanter's SFT chat processor, clips at
32,768 tokens by default, and writes assistant masks. It rejects exact
rendered sequences shared by train and validation. The CLI config accepts
`max_sequence_length`, `max_anchors` (default 12,288) and `seed` (default 42).
Keep the manifests and recipe JSON to reproduce the inputs and training budget.

## Train on an allocated host

Use separate Python environments for Speculators and vLLM. Configure the
Speculators trainer with a compatible CUDA PyTorch release and install from a
clean checkout of the pinned contributor revision.

```bash
git clone https://github.com/atutej/speculators.git "$SPECULATORS_DIR"
git -C "$SPECULATORS_DIR" checkout --detach 77eaa2390cab54ff7e5315345137394ee44b7a98
uv venv --python 3.12 "$TRAINER_ENV"
uv pip install --python "$TRAINER_ENV/bin/python" --torch-backend cu130 \
  -e "$SPECULATORS_DIR" 'torch==2.11.0' 'torchaudio==2.11.0' 'torchvision==0.26.0' \
  'transformers==5.12.1' 'datasets==5.0.1'
```

Install the published Marin GPU vLLM wheel for the host architecture using
the release descriptor. Preserve the direct wheel URL on install for Marin's
startup verifier.

```bash
VLLM_WHEEL=$(uv run python -c 'from marin.external_dependencies import VLLM_GPU_RELEASE; from marin.inference.vllm_release import current_vllm_gpu_wheel, vllm_gpu_wheel_requirement; print(vllm_gpu_wheel_requirement(current_vllm_gpu_wheel(VLLM_GPU_RELEASE)))')
uv venv --python 3.12 "$VLLM_ENV"
uv pip install --python "$VLLM_ENV/bin/python" \
  --index https://download.pytorch.org/whl/cu132 \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match "$VLLM_WHEEL"
```

Save a `runtime.json` with the absolute paths and devices on the allocated host.
For example, a Snowball allocation with two teacher GPUs and one trainer GPU
has this layout. Size the allocation for the specific teacher and context
before running it.

```json
{
  "speculators_checkout": "/shared/speculators",
  "trainer_python": "/shared/trainer-env/bin/python",
  "vllm_python": "/shared/vllm-env/bin/python",
  "teacher_devices": "0,1",
  "trainer_devices": "2",
  "port": 18000,
  "scratch_dir": "/local/nvme/dspark"
}
```

```bash
uv run python -m experiments.post_training.dspark_sft \
  --config recipe.json --stage train --runtime runtime.json
```

The training stage rechecks the teacher weight digests. It runs one teacher
with TP1 and expert parallelism across its assigned GPUs. A loopback vLLM
server captures the four taps through the file connector. Speculators deletes
each temporary activation after use. Choose fast local scratch with room for
concurrent activation files and one training checkpoint. The step publishes
the terminal draft after the full step budget and held-out validation complete.
A failed training attempt can be retried from the same prepared data.

## Serve the draft

The training command prints the versioned `hf/` directory. It contains the
Speculators config, serving weights, vocabulary offsets and
`dspark-export.json` with input identities, package versions, validation
metrics and file digests. Use the same teacher and tokenizer. For Snowball,
pass the following speculative configuration to the promoted Marin wheel.

```json
{
  "method": "dspark",
  "model": "/absolute/path/to/published/hf",
  "num_speculative_tokens": 8,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "standard"
}
```

Mochi shares the checkpoint format. Its ShortConv target still needs a GPU
qualification of hidden-state extraction and speculative decoding, including
acceptance and rejection cases, before using the draft for served requests.
