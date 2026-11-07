# Train and serve a Snowball draft on Iris

Train a fresh EAGLE3 draft for Snowball step1903 and serve it on Iris. The draft
is a small model that proposes tokens for Snowball to verify. Supply coding
prompts; Snowball generates the responses used to train the draft.

Prepare prompts, train, then serve the saved draft using the commands below.
Training and serving use separate eight-H100 Iris tasks and need no RL
configuration.
The twelve-problem example checks that the workflow runs; useful acceptance,
answer quality and speedup need separate evaluation.

## Prepare twelve coding problems

Start from a Marin checkout with [Iris access](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) and credentials
for the target and output bucket. Choose an unused run location and keep one
writer per output. Use storage in the worker's region. Snowball is about 134 GB;
request at least 400 GB of worker disk. The recipe creates its CUDA environments
from the normal Marin CPU environment.

```bash
uv sync --package marin-core --extra cpu
export TARGET=s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/
export TOKENIZER=marin-community/marin-tokenizer
export TOKENIZER_REVISION=a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2
export PROMPTS=s3://your-bucket/standalone-draft/prompts.jsonl
export RUN=s3://your-bucket/standalone-draft/run-001
uv run python -m experiments.post_training.standalone_draft prepare --output "$PROMPTS"
```

Preparation writes prompts directly to `PROMPTS`. It downloads
[`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/tree/152bb5e9a29651266b018106053980070a0521a1)
at revision `152bb5e9a29651266b018106053980070a0521a1`. It discards the corpus's
answers, removes duplicate instruction/input pairs after stripping whitespace,
and selects twelve problems by a seed-42 hash order. The adjacent `.source.json`
records source and sample hashes. No SPEED-Bench data is used.

To supply your own prompt JSONL, use one record per continuation:

```json
{"id":"problem-a-trial-0","task":"problem-a","messages":[{"role":"user","content":"Write a Python function that reverses a list."}]}
```

Give each record a unique `id`. Put every trial or alias of the same problem
under the same `task`, so training and validation cannot share that problem.
Supply at least two problems with prompts that fit the selected training length.
The recipe generates one continuation per prompt. It does not execute tools
or collect agent trajectories.

## Train a fresh draft

Submit one interactive task to `cw-us-east-02a`, whose H100 scaling group is
`h100-8x`. Extraction uses TP1/DP8/EP8. The draft fit uses one GPU after
extraction stops; the task still holds all eight GPUs until it exits.

```bash
uv run iris --cluster marin job run --job-name standalone-draft-fit \
  --enable-extra-resources --target-cluster cw-us-east-02a --gpu H100x8 \
  --cpu 16 --memory 512GB --disk 400GB --priority interactive --timeout 3600 \
  --sync-package marin-core --extra cpu \
  -- python -m experiments.post_training.standalone_draft train \
  --target "$TARGET" --tokenizer "$TOKENIZER" --tokenizer-revision "$TOKENIZER_REVISION" \
  --data "$PROMPTS" --output "$RUN"
```

The defaults bound this example to two epochs, 512-token training sequences and
128 generated tokens per problem. Generation uses temperature 1, top-p 0.95 and
top-k 20. The pinned generator has no seed option. `training.json` records those
settings and the exact generated response hash; responses are retained.
The twelve-problem sample produces eight training and four holdout rows.

The command returns the saved draft's URI and weight SHA256. A failed fit must
start again with a new output location. An interrupted upload may leave a
partial directory; serving requires the final `training.json` record and
checks the weight hash.

## Serve the export

The run chooses the exact target, pinned tokenizer and exported draft together:

```bash
uv run iris --cluster marin job run --job-name standalone-draft-serve \
  --enable-extra-resources --target-cluster cw-us-east-02a --gpu H100x8 \
  --cpu 16 --memory 512GB --disk 400GB --priority interactive --timeout 3600 \
  --sync-package marin-core --extra cpu \
  -- python -m experiments.post_training.standalone_draft serve \
  --output "$RUN" --endpoint standalone-draft --timeout-hours 0.5
```

In another terminal, wait for the Iris endpoint to register and mint access:

```bash
uv run iris --cluster marin endpoints list standalone-draft --exact
uv run iris --cluster marin endpoints mint standalone-draft --ttl-hours 0.5
```

Copy the printed URL into `ENDPOINT`. That URL grants access to this
endpoint without an auth header. Send an actual generation request:

```bash
export ENDPOINT='<printed capability URL without a trailing slash>'
curl "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"target","messages":[{"role":"user","content":"Write a Python function to reverse a list."}],"max_tokens":64}'
```

`--timeout-hours 0.5` stops the serving process after half an hour of service.
Serving loads the recorded target and draft together, checks the draft's weight
hash, and uses model ID `target`. EAGLE3 proposes three tokens per step.
Compilation and CUDA graphs stay enabled, with Snowball's native 65,536-token
context. Prompt and output tokens must fit that limit.

## Python API

The [CLI](https://github.com/marin-community/marin/blob/main/experiments/post_training/standalone_draft.py)
calls [`marin.drafting`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/drafting.py)
and Marin's existing `run_iris_service` API.

Python callers can use `train_draft(DraftTrainingConfig(...))`,
`with draft_serving(export_uri) as (model, engine):`.
Pass that pair to `local_inference(model, engine, num_chips=8)` for a scoped HTTP
session, or to `IrisServiceConfig` for a named endpoint. These calls execute
inside the existing allocation; they do not submit more GPU jobs.

## Model and training details

Snowball step1903 is a checkpoint of the Grug mixture-of-experts model. The
target must have 26 layers, hidden size 2560 and native context 65,536.
The recipe checks the tokenizer's bytes against the target, downloads the
target, and builds a small training copy of its embedding and output head.

Extraction collects hidden states after layers `[2,13,23]` and the final
residual at layer `26`, before output normalization and gating. The Trainer
applies the target's RMSNorm and learned sigmoid gate to that residual before
computing the target predictions used for training. Online RL captures use a
different convention and cannot be used as this recipe's input.

The saved draft uses Hugging Face format without embedding weights:
`config.json`, `model.safetensors`, vocabulary maps and `export_manifest.json`.
The maps link draft vocabulary entries to target entries; changing them would
change the meaning of learned weights. `training.json` records model versions,
input and response hashes, the task split, hidden-state convention and final
validation metrics. The prompt and response JSONL are saved beside it.
The recipe writes `training.json` after the export upload completes.

Extraction states, the training copy of the target head and native Trainer
checkpoints stay on the worker and are deleted when the fit exits.

The extraction toolkit is `lukedhlee/grugmoe-eagle3@8b1558baa74cafb6c7440f1dca2ebb9ab71dc054`;
the trainer is `lukedhlee/speculators@6496b4db007de00dd4741349643d23b23e29dce8`
with Torch `2.13.0+cu132`, Transformers `5.16.1` and Datasets `4.8.5`.
Serving uses Marin's supported CUDA release from
[`VLLM_GPU_RELEASE`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/external_dependencies.py),
qualified here at source `01911be34fac8715347962a8d791ca34a879e01e`.
The ordinary Marin CPU environment remains separate. The handoffs between these
environments are files and HTTP requests.

The [training checks](https://github.com/marin-community/marin/pull/10704)
cover the Grug target-head calculation, vocabulary maps, a fresh fit, the saved
export and real speculative HTTP generation.

## Optional warm starts and recovery

These commands train from scratch. A failed fit restarts with a new output
location. For a longer custom fit, use the [toolkit's training and adaptation
commands](https://github.com/lukedhlee/grugmoe-eagle3/tree/8b1558baa74cafb6c7440f1dca2ebb9ab71dc054#2-train-a-base-draft-or-adapt-one)
in a persistent workspace.

A warm start trains an existing draft with a new optimizer schedule. It needs
a compatible export and its original vocabulary maps. Resuming an interrupted
fit also needs the original data and native optimizer, scheduler and progress
files. A serving export has no optimizer state; copying object-store checkpoint
directories alone does not establish a usable recovery state.

## Optional RL refresh

Ongoing RL refresh is optional. Existing SkyRL runs configure
`generator.speculative_decoding` with an EAGLE3 `model` source and a `training`
block, for example:

```yaml
generator:
  speculative_decoding:
    method: eagle3
    model:
      source_uri: s3://your-bucket/standalone-draft/run-001
      source_identity: <model.safetensors SHA256 from export_manifest.json>
    num_speculative_tokens: 3
    training:
      interval_steps: 10
```

Setting `training: null` keeps serving without refresh. Its bounded online
trainer and held-out promotion checks remain owned by
[`marinskyrl.speculative_decoding`](https://github.com/marin-community/marin/blob/main/vendor/skyrl/marinskyrl/speculative_decoding.py)
and SkyRL's trainer. The standalone commands above do not import that lifecycle.

The online trainer runs on one GPU and retains a full-precision (FP32) copy of
the weights and random generator state between updates. It uses completed
rollouts, including proposal attention masks and target states that already passed through output
normalization and gating. These inputs and recovery states differ from the
offline recipe, so the two training loops remain separate.

For an accepted draft, the online trainer uploads recovery state and serving
weights to an immutable revision directory. It writes one atomic `latest.json`
record after upload and size checks. Readers check the source model and weight
size before inference workers install the draft weights through vLLM's
draft-only update API. Held-out promotion, rollback and optimizer restore keep
their existing behavior.

Old draft records that point to `complete.json` are unsupported. Start a new
SkyRL run/checkpoint root when adopting this format. Online drafts live at
`trainer.ckpt_path/drafts`, separate from the standalone `RUN` export. Existing
historical artifacts and policy checkpoints are preserved.
