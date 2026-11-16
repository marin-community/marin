# Run Mochi RL with the Long Loop workload

`experiments.post_training.mochi_rl.launch` runs the RL portion of
[Ben's Long Loop recipe](https://github.com/marin-community/marin/tree/9248dd74d5464e75b3c686037c6ee55054292968/experiments/post_training/long_loop)
on a completed schema-3 BF16 Mochi SFT export. It skips SFT and terminal HF
export. The normal path samples, trains, evaluates held-out tasks, saves native
checkpoints and selects a best checkpoint within an RL time budget.
`--smoke` selects a shorter path without evaluation or best selection.
Both entrypoint PRs stay draft while the Long and Quick Loops converge.

## One-rack profile

Check out [draft #11321](https://github.com/marin-community/marin/pull/11321)
and select `--profile one-rack`. It requests 15 four-GPU GB200 hosts in one
mandatory NVLink domain: 48 trainer GPUs at TP1/PP12/CP4/EP4 and one 12-GPU
inference engine at TP1/PP1/DP12/EP12. Hosts request 120 CPUs, `900GB` RAM and
`2TB` disk, at production priority. The existing loader uses an 8-GiB buffer
and four concurrent requests per inference worker. The inference sequence cap
stays at 16; this workload has no new one-rack GPU memory-fit result.

The runtime keeps Torch 2.14.1/CUDA 13.2, TE 2.21 and FA4. Compilation mode 3
stays enabled. CUDA graphs use `NONE` after DP12/EP12 graph replay failed during
weight publication; the performance cost remains unmeasured.
[#11307](https://github.com/marin-community/marin/pull/11307) has merged. The draft
uses a pinned integration of the existing route-quota, selection, budget,
Pi-deadline, sandbox, storage and physical-placement repairs; their landing
owners retain the shared changes. It does not refresh package versions.

## Normal workload and time limits

Both profiles use Pi 0.87.0/MCP 7.0.0, high thinking, 128 turns and a default
16,384-token response cap. Pi's context and compaction limits follow the chosen
profile and response cap. Training uses Hero MuonH with `all_experts`
normalization, eight samples per prompt group, TIS with splicing, router replay
and asynchronous collection with staleness three. Constant-reward groups stay
in the consumed batch and receive zero learner advantages.

The normal batch is 32 groups, split equally between SWE and the combined
reasoning/science route. This follows MA's completed three-rack batch; using it
on one rack is a configuration choice, not a new runtime qualification.
`--batch-groups` accepts an even count and keeps eight samples per group and
equal route quotas. The task families, graders and sampling within each route
remain those of the input mixture.

Evaluation and native checkpointing default to every two updates, following
Ben's cadence. `--eval-interval` and `--checkpoint-interval` change them
independently of the absolute `--max-steps` target. Normal training also evaluates
and saves at the end. Best selection requires all 120 held-out tasks to have
scores and a complete five-evaluation EMA window. Valid zero scores count;
missing scores do not. A newly selected best is saved and protected from the
two-checkpoint rolling prune policy. The selection record is
`best_checkpoint.json` below the checkpoint root. A short or incomplete run can
finish without an eligible best.

The RL-budget callback requests stopping at a completed update boundary after
ten hours measured from the first optimizer completion. It includes subsequent
collection, evaluation and checkpoint work. Startup and collection before that
first update are excluded. The clock restarts in a new process, including after
recovery; the callback's 30-minute drain margin is reported, not enforced.
Budget persistence is unchanged.

The separate Iris GPU-job limit defaults to 48 hours, leaving room for loading
and finalization. The coordinator allows at least 48 further hours for queueing
and setup. `--job-timeout-hours` changes the GPU-job limit. Reaching that limit
ends the attempt without an automatic retry.

## Inputs, task images and credentials

Provide retained model, training and held-out artifact directories and the model's
pinned tokenizer revision. The tokenizer defaults to the model export;
`--tokenizer-uri` selects a separate pinned location. Choose a new immutable
`INPUT_VERSION` whenever any input changes. Adoption records identities but does
not copy inputs or extend their retention.

The training artifact must preserve Ben's SWE and six reasoning/science source
families: Eurus, Nemotron, OpenScience, reasoning_gym, Snowball and TaskTrove.
Keep their original TaskSpecs, graders and source labels. The reference population
has 8,025 SWE and 100,000 nonagentic tasks. The held-out population has 80 Nemotron,
20 structure-closure and 20 SWE tasks. Consumed quotas apply to the two routes;
they do not require equal counts from each nonagentic family.

SWE tasks need their original locked task images with Pi/MCP installed and the
qualified directory permissions for instruction-required package setup. Use
existing [task preparation](../references/task-curation.md) and
[RL launching](../references/rl-launching.md) procedures; this entrypoint does
not build images. The recipe uses the qualified 4-GiB sandbox allowance,
10,800-second task/model/Pi HTTP limits, 14,400-second attempt limit and
18,000-second admission wait. The verifier limit stays at 900 seconds.
These settings preserve the original graders and agent setup responsibility.

Configure Iris and regional S3 access, `DAYTONA_RL_API_KEY`,
`DAYTONA_EVAL_API_KEY`, `DAYTONA_TARGET=us`, `TOGETHER_API_KEY`, and W&B login or
`WANDB_API_KEY`. Supply HF credentials if an input needs private Hub access.
The privileged Harbor container profile requires Iris admin authorization.
Keep inputs and native checkpoints for the full queue, run and recovery window.
Output retention defaults to three days and two rolling checkpoints, plus any
protected best; `--ttl-days` and `--checkpoint-count` change the output policy.
Input retention remains the caller's responsibility.

## Render a fresh normal run

Set the placeholders to retained artifacts and a training-file SHA-256 from its
immutable manifest or a complete hash of that file. These commands render the
configuration and check that digest without submitting a GPU job:

```bash
export MARIN_PREFIX=s3://marin-us-east-08a
MODEL='s3://<regional-bucket>/<retained-schema-3-export>'
TASKS='s3://<regional-bucket>/<retained-training-directory>'
HOLDOUTS='s3://<regional-bucket>/<retained-heldout-directory>'
TRAIN_SHA256='<train.parquet-sha256>'
TOKENIZER_REVISION='<tokenizer-commit>'
INPUT_VERSION='<immutable-input-version>'
FRESH_VERSION='<new-normal-version>'
RESUME_VERSION='<new-continuation-version>'
SMOKE_VERSION='<new-smoke-version>'
LAUNCH=(uv run --frozen --all-packages --extra cpu --extra dedup python -m experiments.post_training.mochi_rl.launch)
BASE=(--model-uri "$MODEL" --tokenizer-revision "$TOKENIZER_REVISION"
      --input-version "$INPUT_VERSION" --data-uri "$TASKS" --data-sha256 "$TRAIN_SHA256")
"${LAUNCH[@]}" "${BASE[@]}" --profile one-rack \
  --validation-data-uri "$HOLDOUTS" --version "$FRESH_VERSION" \
  --max-steps 10000 --config-output mochi-normal.yaml
```

`train.parquet` and `validation.parquet` are the default file names. Use
`--data-file` and `--validation-data-file` for different names. The one-rack
profile can also bind several `--data-uri` inputs without a checksum; a checksum
requires one training artifact. Repeat `--validation-data-uri` for held-out
artifacts whose combined population matches the 120-task gate.

Inspect the source YAML's input identities, Pi limits, route quotas, allocation,
callbacks and checkpoint root, then add `--run` to the same command to submit.
A fresh SFT start has fresh MuonH state. Use a new output version for the new
workload; the old four-turn recipe has a different artifact fingerprint.

## Native continuation and recovery

Set `NATIVE` to a completed native directory at TP1/PP12/CP4/EP4, containing model,
optimizer, trainer and data state, such as `checkpoint_root/global_step_2`:

```bash
NATIVE='<completed-native-checkpoint-directory>'
"${LAUNCH[@]}" "${BASE[@]}" --profile one-rack \
  --validation-data-uri "$HOLDOUTS" --version "$RESUME_VERSION" \
  --resume-uri "$NATIVE" --max-steps 10000 --config-output mochi-resume.yaml
```

`--max-steps` is the absolute optimizer-step target. The first import into a new
output version restores native model/optimizer/counters and starts the selected
data stream from its beginning. Repeating that continuation's command and
version recovers its own latest completed checkpoint and saved data state; the
supplied checkpoint is used only when that version has no completed progress.
Keep the inputs and topology compatible. See
[native checkpoint recovery](../references/rl-launching.md#native-checkpoint-inputs)
for failure behavior. Best-selection history and cadence stay fixed within a
recoverable version. The RL-budget clock restarts as described above.

## Explicit smoke

```bash
"${LAUNCH[@]}" "${BASE[@]}" --profile one-rack --smoke \
  --version "$SMOKE_VERSION" --max-steps 2 --max-new-tokens 1024 \
  --job-timeout-hours 6 --config-output mochi-smoke.yaml
```

Smoke uses the same Pi, task routes, n=8 groups, MuonH and asynchronous path.
It defaults to two groups and two rollout workers, with a one-hour RL budget
after the first optimizer completion. It saves native checkpoints but disables
held-out evaluation and best selection. The response cap stays at 16K unless
explicitly overridden; the command selects the 1K smoke limit. A smoke can still
take hours to collect its first mixed batch. It does not establish the normal
run's 120-score/five-evaluation selection or ten-hour completion.

## Three-rack profile

Check out [draft #11353](https://github.com/marin-community/marin/pull/11353)
for `--profile three-rack`. It keeps the one-rack option and raises the context
window to 262,144 tokens. It requests 64 trainer GPUs at TP1/PP16/CP4/EP4 and
two 64-GPU inference replicas at TP1/PP1/DP64/EP64. Each role occupies a separate
16-host NVLink domain. Rack placement is mandatory.

The trainer retains activation fusion, 256-KiB NCCL buffers and pinned CPU
storage for checkpointed activation inputs. Inference keeps the 0.80 KV-memory
budget and verified expert-block publication with an 1,800-second timeout.
Cross-rack transport uses InfiniBand. The inference sequence cap is 64, following
MA's completed 32-group mixed batch; the one-rack cap stays at 16. Both profiles
keep compilation mode 3 and CUDA graphs `NONE`.

Use the retained inputs and `LAUNCH`/`BASE` arrays above. This profile requires
one training artifact and its SHA-256. These commands render normal fresh,
normal continuation and explicit smoke configurations:

```bash
THREE_FRESH_VERSION='<new-three-rack-normal-version>'
THREE_RESUME_VERSION='<new-three-rack-continuation-version>'
THREE_SMOKE_VERSION='<new-three-rack-smoke-version>'
NATIVE_262K='<completed-TP1-PP16-CP4-EP4-checkpoint-directory>'
"${LAUNCH[@]}" "${BASE[@]}" --profile three-rack \
  --validation-data-uri "$HOLDOUTS" --version "$THREE_FRESH_VERSION" \
  --max-steps 10000 --config-output mochi-262k-normal.yaml
"${LAUNCH[@]}" "${BASE[@]}" --profile three-rack \
  --validation-data-uri "$HOLDOUTS" --version "$THREE_RESUME_VERSION" \
  --resume-uri "$NATIVE_262K" --max-steps 10000 --config-output mochi-262k-resume.yaml
"${LAUNCH[@]}" "${BASE[@]}" --profile three-rack --smoke \
  --version "$THREE_SMOKE_VERSION" --max-steps 2 --max-new-tokens 1024 \
  --job-timeout-hours 6 --config-output mochi-262k-smoke.yaml
```

Native continuation needs a completed checkpoint with matching TP1/PP16/CP4/EP4
geometry. It uses the same absolute target and same-version recovery contract
as the one-rack path. Normal runs use the shared evaluation/checkpoint cadence,
selection gate and ten-hour RL budget, with a separate 48-hour GPU-job limit.
The smoke command selects a six-hour job limit explicitly.

## Runtime evidence

The one-rack Ben workload has CPU configuration coverage only. Earlier one-rack
535B evidence belongs to the older source and workload linked in
[#11321](https://github.com/marin-community/marin/pull/11321).
MA's three-rack source `8b4710feb7d6` applied two fresh 64-group mixed updates at a
102K cap and imported its native checkpoint at later source `3f4a411d3202`.
Source `940be7913eb7` separately proved controlled near-262K backward and one
fresh 32-group mixed update under a 262K cap, whose actual maximum length was
42,380. Its evaluation and saving were disabled. The normal recipe here changes
cadence and profile context limits; eligible best selection, sustained completion
and live recovery are not established by those earlier results.
