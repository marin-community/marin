# Train and serve a Snowball draft on Iris

This recipe trains an EAGLE3 draft for Snowball/Grug step1903. It generates the
target's responses, extracts offline states, fits a fresh draft and publishes a
serving export. Training and serving run inside separate eight-H100 Iris tasks.
They do not need an RL policy configuration.

The [CLI](https://github.com/marin-community/marin/blob/main/experiments/post_training/standalone_draft.py)
calls the reusable functions in
[`marin.drafting`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/drafting.py).
The recipe installs the pinned toolkit and trainer, stages the target, builds its
training stub, and handles completed checkpoint uploads and serving exports.
You supply storage access, Iris placement, prompts and an unused output location.
Keep one writer per output. Use storage in the worker's region.

## Prepare twelve coding problems

Start from a Marin checkout with [Iris access](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) and credentials
for the target and output bucket. This target is about 134 GB. Request at least
400 GB of worker disk. The normal Marin CPU environment drives the workflow;
the recipe creates its CUDA serving and training environments itself.

```bash
uv sync --package marin-core --extra cpu
export TARGET=s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/
export TOKENIZER=marin-community/marin-tokenizer
export TOKENIZER_REVISION=a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2
export PROMPTS=s3://your-bucket/standalone-draft/prompts.jsonl
export RUN=s3://your-bucket/standalone-draft/run-001
uv run python -m experiments.post_training.standalone_draft prepare --output "$PROMPTS"
```

Preparation downloads
[`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/tree/152bb5e9a29651266b018106053980070a0521a1)
at revision `152bb5e9a29651266b018106053980070a0521a1`. It discards the corpus's
answers, deduplicates stripped instruction/input pairs, hashes each problem, and
selects twelve by a seed42 hash ordering. The adjacent `.source.json` records
source and sample hashes. No SPEED-Bench data is used.

To supply your own prompt JSONL, use one record per continuation:

```json
{"id":"problem-a-trial-0","task":"problem-a","messages":[{"role":"user","content":"Write a Python function that reverses a list."}]}
```

Use unique `id` values and canonical problem identities in `task`. Trials and
aliases of one problem must share that identity. The trainer holds out problems,
not individual continuations. Supply at least two problems and short enough
prompts for the selected training length. This bounded recipe generates one
continuation per prompt; it does not execute tools or collect agent trajectories.

## Train a fresh draft

Submit one interactive task to `cw-us-east-02a`, whose H100 scaling group is
`h100-8x`. The task uses TP1/DP8/EP8 for extraction and serving; the small offline
draft fit uses one GPU after extraction shuts down.

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
top-k 20. The pinned generator has no seed option. `run.json` records those
settings and the exact generated response hash; frozen responses are retained.
The twelve-problem sample produces eight training and four holdout rows.

The target must have 26 layers, hidden size 2560 and native context 65,536.
The recipe verifies the pinned tokenizer's bytes against the target. Extraction
collects raw auxiliary slots `[2,13,23]` and raw final residual slot `26`.
The trainer applies the target's RMSNorm and learned sigmoid gate to that final
residual before computing teacher logits. Managed RL hidden-state captures use
a different convention; they are not inputs to this offline path.

The returned JSON contains the export URI and its weight SHA256. The durable run
contains `run.json`, prompts, exact target responses, extraction manifests,
offline data/states, the target stub, numeric `checkpoints/`, and `exports/epoch-N/`.
`run.json` records target object sizes/ETags, tokenizer and data hashes, the
task split, runtime pins, vocabulary-map hashes and the selected export identity.

## Recover an interrupted fit

Use the same output location on another worker:

```bash
uv run iris --cluster marin job run --job-name standalone-draft-resume \
  --enable-extra-resources --target-cluster cw-us-east-02a --gpu H100x8 \
  --cpu 16 --memory 512GB --disk 400GB --priority interactive --timeout 3600 \
  --sync-package marin-core --extra cpu \
  -- python -m experiments.post_training.standalone_draft resume --output "$RUN"
```

Resume uses the saved configuration and frozen data. It restores the last
published completed epoch's model, optimizer, scheduler, learning rates and
global progress. Publication selects an epoch only after its files finish
uploading. Recovery excludes later partial uploads and rebuilds the trainer's
`checkpoint_best` symlink locally. Missing optimizer/scheduler files or changed
vocabulary-map bytes fail before training. If training has finished, resume
recreates the serving export without further updates.

Recovery is at an epoch boundary. It does not restore RNG state or unfinished
minibatches, and it does not promise a bitwise identical training trajectory.
The pinned trainer serializes floating model and optimizer tensors as BF16 and
restores optimizer masters as FP32. A run that fails before its first completed
epoch has no resumable checkpoint; start with a new output.

For fine-tuning into a **new** run, add `--initial-draft "$OLD_EXPORT"` to `train`
and choose a new `--output`. Supply a compatible export with its original
`d2t-<size>.npy` and `t2d-<size>.npy` files. The toolkit checks those maps against
the learned tensors, keeps the original vocabulary identity, retargets a private
copy to the current target stub, and starts a new optimizer schedule. A serving
export has no optimizer or scheduler state and cannot serve as a resume point.

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

This calls Marin's existing `run_iris_service` API and registers its normal Iris
endpoint. In another terminal, wait for registration and mint access to it:

```bash
uv run iris --cluster marin endpoints list standalone-draft --exact
uv run iris --cluster marin endpoints mint standalone-draft --ttl-hours 0.5
```

Copy the printed capability URL into `ENDPOINT`. That URL grants access to this
endpoint without an auth header. Send an actual generation request:

```bash
export ENDPOINT='<printed capability URL without a trailing slash>'
curl "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"target","messages":[{"role":"user","content":"Write a Python function to reverse a list."}],"max_tokens":64}'
```

`--timeout-hours 0.5` stops the serving process after half an hour of service.
The export is downloaded and its weight hash checked; the target loads
from its real checkpoint URI. Standard EAGLE3 proposes three tokens per step.
Compilation and CUDA graphs stay enabled, with the target's native 65,536-token
context.

Python callers can use `train_draft(DraftTrainingConfig(...))`,
`resume_draft(run_uri)`, or `with draft_serving(run_uri) as (model, engine):`.
Pass that pair to `local_inference(model, engine, num_chips=8)` for a scoped HTTP
session, or to `IrisServiceConfig` for a named endpoint. These calls execute
inside the existing allocation; they do not submit more GPU jobs.

## Runtime and limits

The extraction toolkit is `lukedhlee/grugmoe-eagle3@8b1558baa74cafb6c7440f1dca2ebb9ab71dc054`;
the trainer is `lukedhlee/speculators@6496b4db007de00dd4741349643d23b23e29dce8`
with Torch `2.13.0+cu132`, Transformers `5.16.1` and Datasets `4.8.5`.
Serving uses Marin's supported CUDA release from
[`VLLM_GPU_RELEASE`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/external_dependencies.py),
qualified here at source `01911be34fac8715347962a8d791ca34a879e01e`.
The ordinary Marin CPU environment remains separate. The handoffs between these
environments are files and HTTP requests.

The small Iris qualification checked finite holdout loss, changed draft weights,
the Grug teacher equation, preserved vocabulary maps, recovery into a new local
directory with serialized optimizer/scheduler/progress restored, and a real HTTP
generation using the new export with standard speculative proposal counters.
The teacher numerical check used BF16 bounds `rtol=0.016`, `atol=0.00001`; target
norm/gate tensors, maps and serialized recovery state were compared exactly.
This establishes compatibility for this target and workload. It does not measure
convergence, useful acceptance rates, quality or speedup. Longer training,
production data preparation and benchmark reproduction need separate validation.

Ongoing RL refresh is optional. Existing SkyRL runs configure
`generator.speculative_decoding` with an EAGLE3 `model` source and a `training`
block, for example:

```yaml
generator:
  speculative_decoding:
    method: eagle3
    model:
      source_uri: s3://your-bucket/standalone-draft/run-001/exports/epoch-1
      source_identity: <model.safetensors SHA256 from run.json>
    num_speculative_tokens: 3
    training:
      interval_steps: 10
```

Setting `training: null` keeps serving without refresh. Its bounded online
trainer and held-out promotion checks remain owned by
[`marinskyrl.speculative_decoding`](https://github.com/marin-community/marin/blob/main/vendor/skyrl/marinskyrl/speculative_decoding.py)
and SkyRL's trainer. The standalone commands above do not import that lifecycle.
