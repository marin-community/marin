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
buffer and four requests per inference worker. Compilation and piecewise CUDA
graphs remain enabled with the required ShortConv exclusions.

The recipe retains Long Loop's eight-response RLOO-N groups, with two prompts
per update, the reasoning-marker penalty and error masking. It selects Hero
MuonH with `expert_normalization: all_experts`, FP32 masters and pinned CPU
momentum. Supply ordinary mixed-task Parquet artifacts; no length probe is added.
Each response turn can generate up to 8,192 tokens, with at most four turns.

## Prerequisites and inputs

This entrypoint requires [cross-host planning #11299](https://github.com/marin-community/marin/pull/11299),
[Hero MuonH #11301](https://github.com/marin-community/marin/pull/11301), the
[qualified GPU runtime #11307](https://github.com/marin-community/marin/pull/11307),
[Harbor harness binding #11324](https://github.com/marin-community/marin/pull/11324)
and the shared [native checkpoint support](../references/rl-launching.md#native-checkpoint-inputs).
Until these changes merge, use a revision containing all of them. The entrypoint
PR records the combined revision used for validation.

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
COMMON=(--model-uri "$MODEL" --tokenizer-revision "$TOKENIZER_REVISION"
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
scheduler, trainer counter and available rollout state. It writes new checkpoints
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
`--tokenizer-uri` selects a separate pinned tokenizer artifact. Python callers
can supply another topology, Megatron geometry, recipe and execution configuration
to `build_run`; this profile qualifies the one-rack layout only.
