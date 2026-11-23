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
frozen teacher head rows. Training runs for the lesser of `max_steps` and
one packed epoch. Linear decay uses that same step count. The optimizer is
Muon and the loss weights are 0.1 CE and 0.9 total variation.
[Reference training recipe](https://github.com/atutej/spec-experiments/tree/85d1dd3d1f793153d3e6c2aeab3f3c940987fd9d/pipelines/speculator_training)

## Bind the regional inputs

Use the exact HF directory of a completed Snowball or Mochi SFT artifact,
including its tokenizer and chat template. Record the producer's
`name@version:fingerprint` and HF subdirectory as the teacher identity. For a
staged Hub snapshot, record the repo and full commit SHA. Mochi accepts an
explicit completed export for CPU preflight and data preparation. GPU
training requires ShortConv qualification on the promoted wheel.

Store a JSON object mapping the teacher's relative HF filenames to lowercase
SHA-256 digests at a separate manifest URI. Include all safetensors shards,
the weight index if sharded, the config, and every tokenizer and chat-template
file. Generate the manifest from the completed export in its storage region.

Use a versioned normalized Marin chat mixture with `messages` and optional
`chat_template_kwargs` columns. At the mixture root, publish `dspark-mix.json`
with two keys, `train` and `validation`. Each key maps relative Parquet
filenames to their SHA-256 digests. Set the mixture proportions and holdout
membership when selecting the shards.

Save the recipe as JSON and put it in the same region as the inputs and
Iris worker. For example, a recipe for four GPUs assigns two to the teacher
and two to Speculators. Replace each URI, identity and digest with the
corresponding value for the selected export or mixture.

```json
{
  "teacher": {
    "family": "snowball",
    "identity": "checkpoints/snowball-sft@<version>:<fingerprint>/hf/step-<N>",
    "directory": "s3://<regional-bucket>/<snowball-sft>/hf/step-<N>",
    "manifest_uri": "s3://<regional-bucket>/<recipe-inputs>/teacher.json",
    "manifest_sha256": "<teacher-manifest-sha256>"
  },
  "data": {
    "directory": "s3://<regional-bucket>/<versioned-chat-mix>",
    "manifest_sha256": "<mix-manifest-sha256>"
  },
  "max_steps": 100,
  "teacher_replicas": 2,
  "trainer_ranks": 2
}
```

The recipe accepts `max_sequence_length` (default 32,768 tokens),
`max_anchors` (default 12,288) and `seed` (default 42). Set `MARIN_PREFIX` to a
durable regional output prefix and `RECIPE_URI` to the stored recipe.

```bash
uv run python -m experiments.post_training.dspark_sft --config "$RECIPE_URI"
uv run python -m experiments.post_training.dspark_sft --config "$RECIPE_URI" --stage preflight
```

The default command prints the artifact graph. CPU preflight checks the
teacher config, tensor headers, tokenizer and chat template, plus the
pinned mix manifests. It reports the requested GPU count and whether the
teacher family is supported for training. Planning and preflight check the
promoted vLLM wheel against the recipe's source pin.

## Submit one Iris job

Set `IRIS_TARGET_CLUSTER`, `IRIS_REGION`, `GPU_TYPE`, `GPU_COUNT` and
`DSPARK_DISK` for the worker. Request the GPU count printed by preflight on
one worker. Allow scratch space for the full teacher export, prepared Arrow
data and temporary build files, activations, and the training checkpoint.

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name offline-dspark --target-cluster "$IRIS_TARGET_CLUSTER" --region "$IRIS_REGION" \
  --enable-extra-resources --replicas 1 --gpu "${GPU_TYPE}x${GPU_COUNT}" \
  --cpu 32 --memory 256GB --disk "$DSPARK_DISK" --sync-package marin-core --extra cpu \
  -e MARIN_PREFIX "$MARIN_PREFIX" -e TMPDIR /cache \
  -- python -m experiments.post_training.dspark_sft --config "$RECIPE_URI" --stage train
```

The worker runs the artifact graph and provisions the trainer from pinned
Speculators revision `77eaa2390cab54ff7e5315345137394ee44b7a98`. Marin's managed
vLLM environment provisions the promoted wheel and its CUDA toolchain.
The recipe partitions the worker's assigned GPUs between TP1 expert-parallel
teacher prefill and Speculators training. Teacher prefill uses FlashAttention
and Triton MoE, matching the Snowball rollout preset. The two processes
exchange hidden states over the worker's local filesystem.
The trainer uses Speculators' PyTorch implementation of the weighted CE/TV
loss to avoid a tracing error in the fused loss autotuner.

Preparation verifies the data shard digests, renders assistant masks with
Levanter's SFT chat processor, and rejects exact rendered sequences shared
across splits. For a large mix, `uv run --extra cpu` with `--stage prepare`
builds the dataset in the storage region ahead of training. Runs with
different step budgets, optimizer settings, or GPU counts reuse that
dataset. The worker checks split sizes and counts the packed training
batches before staging the teacher weights.

## Serve the draft

The Iris job prints the versioned `hf/` directory in object storage. It contains the
Speculators config, serving weights, vocabulary offsets and
`dspark-export.json` with input identities, package versions, validation
metrics, actual steps, step budget and file digests. A complete export has
this receipt. Stage the draft for vLLM and use the same teacher and tokenizer.
Launch the promoted Marin wheel for Snowball with
`--attention-backend FLASH_ATTN --moe-backend triton` and pass this JSON to
`--speculative-config`.

```json
{
  "method": "dspark",
  "model": "/absolute/path/to/published/hf",
  "num_speculative_tokens": 8,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "standard"
}
```
