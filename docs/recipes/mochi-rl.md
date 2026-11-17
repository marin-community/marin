# Run Mochi RL with the Long Loop workload

`experiments.post_training.mochi_rl.launch` runs the RL portion of
[Ben's Long Loop recipe](https://github.com/marin-community/marin/tree/9248dd74d5464e75b3c686037c6ee55054292968/experiments/post_training/long_loop)
on a completed schema-3 BF16 Mochi SFT export. It skips SFT and terminal HF
export. The normal path samples, trains, evaluates held-out tasks, saves native
checkpoints and selects a best checkpoint when enough valid evaluations have
completed, within an RL time budget.
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

The one-rack runtime keeps Torch 2.14.1/CUDA 13.2, TE 2.21 and FA4. Compilation
mode 3 stays enabled. Its CUDA graphs use `NONE` after DP12/EP12 graph replay failed during
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
reasoning/science route. This follows the completed three-rack reference batch;
using it on one rack is a configuration choice, not a new runtime qualification.
`--batch-groups` accepts an even count and keeps eight samples per group and
equal route quotas. Ben's reference uses 64 groups and staleness two.
The task families, graders and sampling within each route remain those of the
input mixture.

Evaluation and native checkpointing default to every two updates, following
Ben's cadence. `--eval-interval` and `--checkpoint-interval` change their cadence
without changing the absolute `--max-steps` target. Evaluations must fall on
checkpoint steps for same-version recovery: the evaluation interval must be a
multiple of the checkpoint interval, such as six and three. This keeps selection
history from getting ahead of a recovery checkpoint. Normal training also
evaluates and saves at the end.

Best selection requires all 120 held-out tasks to have scores and a complete
five-evaluation EMA window. Valid zero scores count;
missing scores do not. A newly selected best is saved and protected from the
two-checkpoint rolling prune policy. The selection record is
`best_checkpoint.json` below the checkpoint root. The ten-hour budget may end
before five complete evaluations, so no best is selected. Completed selection
history is reused during same-version recovery.

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
Input retention remains the caller's responsibility. One-rack trace capture keeps
the existing limits of 64 MiB per step and 512 MiB per run. After a limit is
reached, later traces are dropped and counted in
`trajectory_retention/dropped_by_bounds`; training continues.

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
It defaults to two groups and uses two rollout workers, with a one-hour RL budget
after the first optimizer completion. `--batch-groups` changes the batch while
smoke collection stays at two workers. It saves native checkpoints but disables
held-out evaluation and best selection. The response cap stays at 16K unless
explicitly overridden; the command selects the 1K smoke limit. A smoke can still
take hours to collect its first mixed batch. It does not establish the normal
run's 120-score/five-evaluation selection or ten-hour completion.

## Three-rack profile

Check out [draft #11353](https://github.com/marin-community/marin/pull/11353)
for `--profile three-rack`. It keeps the one-rack option and the 262,144-token
context window. It requests 64 trainer GPUs at TP1/PP16/CP4/EP4 and eight 16-GPU
inference engines at TP1/PP1/DP16/EP16. The trainer occupies one 16-host NVLink
domain. Four inference engines share each of two further 16-host domains.
The allocation stays at 48 four-GPU hosts and 192 GPUs. Rack placement is mandatory.

The trainer retains activation fusion, 256-KiB NCCL buffers and pinned CPU
storage for checkpointed activation inputs. Cross-rack transport uses InfiniBand.
Inference uses FlashInfer, a 128-sequence limit per engine, a 2,048-token batched
budget, a 0.90 GPU-memory fraction and disabled prefix caching. Compilation mode 3
uses `PIECEWISE` graphs with the existing ShortConv/attention split points and
qualified Inductor options. Routed-expert capture and verified expert-block
publication remain enabled, with a 1,800-second publication timeout. Three-rack
trace capture keeps its 256-MiB per-step and 1-GiB per-run limits.

Normal rollout admission permits up to 128 prompt groups in flight; the trainer
batch stays at 32 groups of eight samples. The native buffer also bounds untrained
groups by `(staleness + 1) * batch_size`. At staleness three, `--batch-groups 6`
admits at most 24 groups. `--smoke` keeps two collection workers and its two-group
admission cap. The one-rack profile retains its EP12 layout, 16-sequence limit,
FlashAttention backend and graphs `NONE`.

Use the retained inputs and `LAUNCH`/`BASE` arrays above. This profile requires
one training artifact and its SHA-256. These commands render normal fresh,
normal continuation and explicit smoke configurations:

```bash
THREE_FRESH_VERSION='<new-ep16-normal-version>'
THREE_RESUME_VERSION='<new-ep16-continuation-version>'
THREE_SMOKE_VERSION='<new-ep16-smoke-version>'
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
geometry. Use a new output version when switching from the earlier two-engine
EP64 layout. It uses the same absolute target and same-version recovery contract
as the one-rack path. Normal runs use the shared evaluation/checkpoint cadence,
selection gate and ten-hour RL budget, with a separate 48-hour GPU-job limit.
The smoke command selects a six-hour job limit explicitly.

## Runtime evidence

The one-rack Ben workload has CPU configuration coverage only. Earlier one-rack
535B evidence belongs to the older source and workload linked in
[#11321](https://github.com/marin-community/marin/pull/11321).

The new inference bundle uses the published BF16 vLLM build
`marin-vllm-gpu-20261113-0ac9dcb157ee`. A matched 64-GB200 serving comparison used
the same nodes, prompts, seeds, 256-token outputs and 2,048 concurrent requests.
Output tokens/s per allocated inference GPU rose from 126 with EP64/graphs `NONE`,
to 293 with EP64/`PIECEWISE`, to 416 with four EP16 groups, to 464 with the full
serving bundle. The 3.68x gain measures serving; it does not isolate FlashInfer's
contribution or measure RL speedup. [Issue #11430](https://github.com/marin-community/marin/issues/11430)
provides the public serving context. Detailed matched-run receipts are not public.
The measured gain requires neither [vLLM #130](https://github.com/marin-community/vllm/pull/130)
nor [#11492](https://github.com/marin-community/marin/pull/11492).

Reference source `612fa39494f9` completed three fresh nonzero RL updates at a
102,400-token context limit. Updated weights were verified and later used for
generation in all eight EP16 engines. Each of two held-out evaluations produced
scores for all 120 tasks, and native checkpoint 3 completed. The first batch's actual maximum attended length was
56,196. That qualification used separate task-setup/runtime repairs and cadence;
this draft keeps its existing setup, evaluation, selection and recovery behavior.
Two eligible evaluations do not complete the five-evaluation selection window.

Separate EP16 serving completed one request with 262,024 prompt-plus-output tokens.
Earlier near-262K trainer backward at source `940be7913eb7` used the previous inference geometry. Its separate
fresh 32-group mixed batch reached 42,380 tokens with evaluation and saving
disabled. This draft keeps the 262,144-token context limit. Integrated near-262K training
with all eight engines, sustained completion and live same-version recovery
remain unqualified by those results. The continuing runtime campaign is separate.
