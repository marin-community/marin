# Run Mochi RL at 65K context

`experiments.post_training.mochi_rl.launch` starts RL directly from a completed
schema-3 BF16 SFT export, or continues a native RL checkpoint. It constructs one
training artifact. The context limit is 65,536 attended tokens across the prompt,
responses and tool observations; masked padding does not count.

The allocation is 15 four-GPU GB200 hosts in `cw-us-east-08a`, at production
priority: 48 trainer GPUs with TP1/PP12/CP4/EP4 and one 12-GPU inference engine
with TP1/PP1/DP12/EP12. Iris keeps this 15-host GB200 gang in one NVLink domain.
The role planner checks counts and parallelism; other layouts need their own
placement check.
Each host requests 120 CPUs, 900 GiB RAM and 2 TiB disk. Loading uses an 8 GiB
buffer and four requests per inference worker. This example uses the qualified
runtime: Torch 2.14.1/CUDA 13.2, TE 2.21 and FA4 on GB200. Compilation mode 3
remains enabled. CUDA graphs use `NONE` because graph replay failed after weight
publication with DP12/EP12 `allgather_reducescatter`. The cause and full-model
performance cost remain unmeasured.

The recipe uses eight-response RLOO-N groups, with two prompts
per update, the reasoning-marker penalty and error masking. It selects Hero
MuonH with `expert_normalization: all_experts`, FP32 masters and pinned CPU
momentum. Supply ordinary mixed-task Parquet artifacts; no length probe is added.
Each response turn can generate up to 8,192 tokens, with at most four turns.
Use `--max-new-tokens 1024` for a short smoke within the same context window.

## Prerequisites and inputs

Planning, Hero MuonH, Harbor harness binding and
[native checkpoint support](../references/rl-launching.md#native-checkpoint-inputs)
are on main. The draft also includes the
[qualified GPU runtime #11307](https://github.com/marin-community/marin/pull/11307)
and [turn-terminator fix #11232](https://github.com/marin-community/marin/pull/11232)
while they remain unmerged. [Draft #11321](https://github.com/marin-community/marin/pull/11321)
pins the combined revision. This is a
runnable example kept in draft while loop unification is discussed.

Provide Iris and regional S3 credentials, W&B access, and `DAYTONA_RL_API_KEY`
and `DAYTONA_EVAL_API_KEY`. Supply HF credentials if any selected input requires
Hub access. Harbor tasks use the configured privileged container profile, which
requires Iris admin authorization. See [RL launching](../references/rl-launching.md).

Choose a completed schema-3 BF16 SFT export with its portable-model manifest,
and a tokenizer artifact pinned to the revision used by that export. Select
ordinary mixed-task Parquet data compatible with this recipe, including its
Harbor harness mapping. The model, tokenizer and tasks must remain available
through the planned run and recovery window. Adoption does not extend storage
retention. Give each changed input set a new immutable `INPUT_VERSION`.

Replace the placeholders below with those artifact locations and identities:

```bash
export MARIN_PREFIX=s3://marin-us-east-08a
MODEL='s3://<regional-bucket>/<retained-schema-3-export>'
TASKS='s3://<regional-bucket>/<retained-mixed-task-directory>'
INPUT_VERSION='<immutable-input-version>'
TOKENIZER_REVISION='<tokenizer-commit>'
SMOKE_VERSION='<new-smoke-version>'
CONTINUATION_VERSION='<new-continuation-version>'
```

## Fresh start and continuation

Start with two updates. Build the graph and write its source launch configuration
without submitting:

```bash
COMMON=(--profile one-rack --model-uri "$MODEL" --tokenizer-revision "$TOKENIZER_REVISION"
        --data-uri "$TASKS" --input-version "$INPUT_VERSION")
uv run python -m experiments.post_training.mochi_rl.launch "${COMMON[@]}" \
  --version "$SMOKE_VERSION" --max-steps 2 --config-output mochi-launch.yaml
```

Inspect the model/data locators, role allocation, context budget and checkpoint
paths, then add `--run` to submit the same graph. The checkpoint callback saves
at training end, so this smoke produces `global_step_2` when it completes. A new
output version starts fresh MuonH state. Repeating an interrupted run with the same version recovers
its latest completed native checkpoint when one exists.

To continue from a completed checkpoint under a new output version, set `NATIVE` to the exact
completed native checkpoint directory, such as your run's
`checkpoints/global_step_2`, then run:

```bash
uv run python -m experiments.post_training.mochi_rl.launch "${COMMON[@]}" \
  --version "$CONTINUATION_VERSION" --max-steps 5 --resume-uri "$NATIVE" \
  --config-output mochi-resume.yaml
```

Inspect it, then add `--run`. `--max-steps` is the absolute stopping step, so it
must exceed the restored step. Native continuation restores model, optimizer,
scheduler and trainer counters. The initial import starts the configured data
from the beginning; recovery restores this continuation's saved data state. It writes new checkpoints
under the new output version. Recover an interrupted continuation by repeating
its command with the same version and inputs, including `--resume-uri`.
See [native checkpoint recovery](../references/rl-launching.md#native-checkpoint-inputs)
for selection and failure behavior. Keep the model, tokenizer, data and topology
compatible.
Find checkpoints under `artifacts.checkpoint_root` in the preview YAML, then
append `/global_step_N` for the completed step.

The GPU job defaults to a 24-hour limit. Set `--job-timeout-hours` for a longer
run. The coordinator timeout is at least 48 hours longer than the GPU job limit
to allow for queueing and setup. If the GPU job reaches its limit, the attempt
ends without an automatic retry; recover from the latest completed checkpoint
as above. Keep inputs and checkpoints until training and any planned recovery
are complete.

Outputs use your user namespace. Recovery checkpoints and traces default to
the regional three-day TTL, with two rolling checkpoints. `--ttl-days` and
`--checkpoint-count` change that policy. Terminal HF export is disabled. Repeat
`--data-uri` for more task artifacts; `--data-file` defaults to `train.parquet`.
`--tokenizer-uri` selects a separate pinned tokenizer artifact. Python callers pass `profile=PROFILES["one-rack"]` and bound model/data artifacts
to `build_run`. The profile holds typed role, Megatron, recipe and execution settings. The reused GPU evidence covers the one-rack layout at its pinned
source. [#11321](https://github.com/marin-community/marin/pull/11321) records that
source and the CPU checks; those checks do not qualify other layouts.

## 262K three-rack example

`experiments.post_training.mochi_three_rack`
([draft #11353](https://github.com/marin-community/marin/pull/11353)) uses the same recipe at a
262,144-token context window. It requests 64 trainer GPUs at TP1/PP16/CP4/EP4
and two 64-GPU inference replicas at TP1/PP1/DP64/EP64. Mandatory NVLink placement
keeps these three roles in separate domains. The trainer retains activation
fusion, 256 KiB NCCL buffers and pinned CPU storage for checkpointed activation
inputs. Compilation mode 3 remains enabled with CUDA graphs `NONE`.

The input is a completed BF16 SFT export; no SFT job is built. Select retained
model, tokenizer and mixed-task inputs as above. Supply the published SHA-256
of `train.parquet`; the launcher checks it before submitting training. These
commands only render configurations:

```bash
DATA_SHA256='<train.parquet-sha256>'
THREE_RACK_VERSION='<new-three-rack-version>'
THREE_RACK_RESUME_VERSION='<new-three-rack-continuation-version>'
NATIVE_262K='<completed-TP1-PP16-CP4-EP4-checkpoint-directory>'
THREE_RACK=(--model-uri "$MODEL" --tokenizer-revision "$TOKENIZER_REVISION"
            --input-version "$INPUT_VERSION" --data-root "$TASKS" --data-sha256 "$DATA_SHA256")
uv run python -m experiments.post_training.mochi_three_rack "${THREE_RACK[@]}" \
  --version "$THREE_RACK_VERSION" --updates 2 --config-output mochi-262k.yaml
uv run python -m experiments.post_training.mochi_three_rack "${THREE_RACK[@]}" \
  --version "$THREE_RACK_RESUME_VERSION" --updates 2 --resume-uri "$NATIVE_262K" \
  --config-output mochi-262k-resume.yaml
```

Use a completed native checkpoint with matching TP1/PP16/CP4/EP4 geometry for
`NATIVE_262K`. Here `--updates` counts additional updates: checkpoint 5 with two
updates sets the final target to 7. Repeat the same version and command after
interruption to recover this continuation's progress. Add `--run` only after
checking the configurations and retained inputs. Responses default to 8,192
tokens; `--max-new-tokens 1024` retains the short-smoke option.
The GPU job limit is six hours, with a 72-hour coordinator limit for queueing and setup.

The older three-rack GPU proof used TE 2.19/FA2 and a 1,024-token response cap.
It established near-262K backward and native continuation. Its later nonzero
gradient proof was an offline replay. The aligned FA4/TE 2.21 example and its
8,192-token response default have CPU validation only. Fresh on-policy learning
at these settings remains unverified. Endurance, task/batch tuning and
performance work belong to the separate Mochi follow-up.
