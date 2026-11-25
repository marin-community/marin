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
The last tap feeds the frozen teacher logits head. Snowball uses three
full-attention Qwen3 layers and a 3,840-wide FFN. The default Mochi draft uses
one Qwen3 layer and a 4,096-wide FFN, with 521M weights excluding the shared
teacher embedding. Both drafts use a block size of 8, a rank-256 Markov head
and a confidence head. Training retains the most
frequent 32,000 supervised token IDs from training, with the corresponding
frozen teacher head rows. Training runs for the lesser of `max_steps` and
one packed epoch. Linear decay uses that same step count. The optimizer is
Muon and the loss weights are 0.1 CE and 0.9 total variation.
[Reference training recipe](https://github.com/atutej/spec-experiments/tree/85d1dd3d1f793153d3e6c2aeab3f3c940987fd9d/pipelines/speculator_training)

## Bind the regional inputs

Use a completed Snowball or Mochi SFT export with its tokenizer and chat
template. A Hub export needs only `"source": "hf://org/model@<full commit>"`.
Marin's model download step stages that commit in regional storage. For a
Marin export, give the completed artifact's fixed-version directory and the
HF subdirectory. The runner adopts the existing directory as a dependency.
An archived export with a non-calendar directory can set `teacher.artifact_version`.
Set `teacher.tokenizer_source` to an immutable `hf://org/repo@<full commit>`
when the teacher export stores its tokenizer separately.

Use a completed Marin artifact containing OpenAI-format or Datakit Harmony chat Parquet shards
under `train/` and `validation/`. The shards must provide `messages` and may
provide `chat_template_kwargs`. Set the mixture and holdout membership when
building that artifact.

Save the JSON recipe in the region of the inputs and Iris worker. Each
Marin input directory ends in its fixed version.

```json
{
  "teacher": {
    "family": "snowball",
    "source": "s3://<regional-bucket>/<snowball-sft>/<version>",
    "relative_path": "hf/step-<N>"
  },
  "data": "s3://<regional-bucket>/<chat-mix>/<version>",
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
uv run --extra cpu python -m experiments.post_training.dspark_sft --config "$RECIPE_URI" --stage prepare
```

The default command prints the artifact graph. CPU preflight checks the
teacher config, tensor headers, tokenizer, template, and the train and
validation shard locations. It reports the requested GPU count and trainer
ranks. Run `--stage prepare` on CPU
to stage an uncached Hub teacher. Planning and preflight check the promoted
vLLM wheel against the recipe's source pin.

## Prepare the Mochi agentic smoke mix

The preset binds the retained BF16 Mochi Hero SFT step 400,407 export and the
tokenizer pinned by the Mochi SFT recipe. The default three-step config uses a
small sample of Nemotron tool calls, Nemotron reasoning-off instruction turns,
and AgentTrove conversations. The first user prompt assigns each conversation
to one split. Tool definitions and observations reach the teacher as context,
and the template supervises assistant replies and tool calls. The mix samples
shards, row groups, and rows with a fixed seed. Each split interleaves sources
in proportion to their sampled conversation counts, using up to 256 tool-call,
256 instruction, and 128 AgentTrove conversations per selected shard.
[Mochi export record](https://github.com/marin-community/marin/issues/8089#issuecomment-6841382920)

```bash
export MARIN_PREFIX=s3://marin-us-east-08a/marin
uv run --extra cpu --extra dedup python -m experiments.post_training.dspark_sft \
  --preset mochi-agentic --stage prepare
uv run --extra cpu python -m experiments.post_training.dspark_sft \
  --preset mochi-agentic --stage preflight
```

## Prefill Mochi teacher activations

Submit two four-GB200 tasks. The distributed BF16 teacher runs with TP1 and
DP8/EP8. Each activation file lands on the leader, moves to the S3 scratch
bucket selected by `marin_temp_bucket`, and is removed locally after upload.
The bucket expires the files after three days. A completed cache can be reused
for 48 hours from the start of extraction. Cache identity includes the teacher,
prepared dataset, sample mode, tap layers, sequence limit, and extraction
runtime. You can go directly to draft fitting when a reusable cache exists.
Preflight reports the teacher GPU count as `prefill_gpus` and the fitting GPU
count as `gpus`.

Teacher prefill requests generate one token to return each activation file.
The cache progress log reports completed conversations and prompt tokens per
second, plus the mean time spent uploading an activation file.

Export `CW_KEY_ID` and `CW_KEY_SECRET` before submission so Iris can pass
them to the job as secrets.

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name mochi-dspark-prefill --target-cluster cw-us-east-08a --region us-east-08a \
  --enable-extra-resources --replicas 2 --gpu GB200x4 \
  --cpu 48 --memory 768GB --disk 128GB --timeout 7200 \
  --sync-package marin-core --extra cpu -e TMPDIR /cache -e MARIN_PREFIX "$MARIN_PREFIX" \
  --secret CW_KEY_ID env:CW_KEY_ID --secret CW_KEY_SECRET env:CW_KEY_SECRET \
  -- python -m experiments.post_training.dspark_sft --preset mochi-agentic --stage prefill
```

## Fit a Mochi draft from cached activations

After prefill completes, submit the fitting ArtifactStep on one GPU. Use
matching preset, data, and sample selection for both stages. Speculators reads
the activations as it batches examples. The RunAI loader caps each process's
CPU buffer at 8 GiB. The trainer stages only the teacher's embedding and final
gated head shards.

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name mochi-dspark-fit --target-cluster cw-us-east-08a --region us-east-08a \
  --enable-extra-resources --replicas 1 --gpu GB200x1 \
  --cpu 12 --memory 160GB --disk 128GB --timeout 7200 \
  --sync-package marin-core --extra cpu -e TMPDIR /cache -e MARIN_PREFIX "$MARIN_PREFIX" \
  --secret CW_KEY_ID env:CW_KEY_ID --secret CW_KEY_SECRET env:CW_KEY_SECRET \
  -- python -m experiments.post_training.dspark_sft --preset mochi-agentic --stage train
```

The default smoke trains three updates on 12 training and up to four validation
conversations. Add `--steps 4000` to both stages to train for one packed epoch, capped at
4,000 optimizer steps, over the complete frozen mix. The epoch uses an
8,192-token sequence and up to 1,024 anchors. Use `--data` to provide a
different completed regional mix. Teacher activations take 48 KiB per input
token in temporary object storage. Local scratch holds the files in flight,
the trainer environment, frozen weights, and the draft checkpoint.

Use `--draft-size two-layer` to reproduce the 766M Mochi variant with two
Qwen3 layers and a 6,144-wide FFN. It uses the same teacher, data, vocabulary,
and training budget as the default and publishes a separate checkpoint.
Both sizes share one prepared Arrow artifact and the same teacher activation
cache. The trainer builds its decoder config from the chosen size.
Mochi's one-rack RL presets use the qualified two-layer export from the interleaved
50M mix. [Mochi RL launch defaults](../references/rl-launching.md)

## Scale the Mochi mix to about 50M input tokens

`mochi-agentic-50m` samples up to 2,560 tool-call, 1,536 instruction, and 384
AgentTrove conversations per selected shard. These quotas are in rows, so the
token share also depends on conversation length. The mix uses the same
prompt-group holdout and interleaves both splits. It trains the 521M draft
for one packed epoch capped at 15,000 steps. Prepare on a CPU in `us-central2`, where the
original Nemotron shards live. The CPU job writes the sampled mix and
prepared Arrow data to the training region.

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name mochi-dspark-50m-prepare --region us-central2 \
  --enable-extra-resources --cpu 2 --memory 12GB --disk 48GB --timeout 3600 \
  --sync-package marin-core --extra cpu --extra dedup -e MARIN_PREFIX "$MARIN_PREFIX" \
  --secret CW_KEY_ID env:CW_KEY_ID --secret CW_KEY_SECRET env:CW_KEY_SECRET \
  -- python -m experiments.post_training.dspark_sft --preset mochi-agentic-50m --stage prepare
```

After preparation succeeds, use the prefill and fitting commands with
`--preset mochi-agentic-50m`, fresh job names, and `--timeout 14400`. The
50M mix and its holdout use about 2.4 TiB of temporary object storage for
teacher activations.

## Serve a Mochi draft

Serving Mochi drafts requires ShortConv state to survive rejected proposals.
The promoted wheel needs the upstream fix for that verification path.
[ShortConv verifier fix](https://github.com/marin-community/vllm/pull/140)

For SkyRL, set `DRAFT_HF` to the `hf` path printed by the fit stage and
`DRAFT_SKYRL` to a fresh, durable S3 path in the same region. The draft uses
the policy tokenizer. Generate its model manifest and upload the marker after
the model files are in place.

```bash
(
set -eu
draft_staging=$(mktemp -d)
trap 'rm -rf -- "$draft_staging"' EXIT
uv run fsutil cp -r "$DRAFT_HF" "$draft_staging/model"
uv run python - "$draft_staging" <<'PY'
import sys
from pathlib import Path

from vendor.skyrl.marinskyrl.model_manifest import snapshot_model_manifest

root = Path(sys.argv[1])
manifest = snapshot_model_manifest(root / "model", None, None, tokenizer_mode="policy")
(root / ".marinskyrl-model-manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
print(manifest.identity)
PY
uv run fsutil verified-copy "$draft_staging/model" "$DRAFT_SKYRL"
uv run fsutil cp "$draft_staging/.marinskyrl-model-manifest.json" \
  "$DRAFT_SKYRL/.marinskyrl-model-manifest.json"
)
```

Set `generator.speculative_decoding.model` with `source_uri` set to `DRAFT_SKYRL`
and `source_identity` set to the printed digest. Use the verifier that supplied
the teacher activations. [RL launch options](rl-experiments.md)

## Submit the Snowball Iris job

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

Preparation renders assistant masks with Levanter's SFT chat processor and
rejects sequences shared across splits after rendering. Runs with different
step budgets or GPU counts reuse the dataset. The worker checks split sizes
and counts the packed batches before staging the teacher weights.

## Serve the draft

The Iris job prints the versioned `hf/` directory in object storage. It contains the
Speculators config, serving weights and vocabulary offsets. The runner
records dependency identities, configuration, package versions, actual
steps and validation metrics in the parent artifact's `.artifact.json`.
Stage the draft for vLLM and use the same teacher and tokenizer.
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
