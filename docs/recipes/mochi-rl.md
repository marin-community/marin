# Run Mochi RL at 65K context

`experiments.post_training.mochi_rl.launch` starts RL directly from a completed
schema-3 BF16 SFT export, or continues a native RL checkpoint. It constructs one
training artifact. The context limit is 65,536 attended tokens across the prompt,
responses and tool observations; masked padding does not count.

The allocation is 15 four-GPU GB200 hosts in `cw-us-east-08a`, at production
priority: 48 trainer GPUs with TP1/PP12/CP4/EP4 and one 12-GPU inference engine
with TP1/PP1/DP12/EP12. Check that the allocated hosts share one NVLink domain.
The planner checks counts and parallelism; it does not enforce a rack boundary.
Each host requests 120 CPUs, 900 GiB RAM and 2 TiB disk. Loading uses an 8 GiB
buffer and four requests per inference worker. Compilation and piecewise CUDA
graphs remain enabled with the required ShortConv exclusions.

The recipe retains Long Loop's eight-response RLOO-N groups, with two prompts
per update, the reasoning-marker penalty and error masking. It selects Hero
MuonH with `expert_normalization: all_experts`, FP32 masters and pinned CPU
momentum. Supply ordinary mixed-task Parquet artifacts; no length probe is added.

## Prerequisites and inputs

This entrypoint depends on [cross-host planning #11299](https://github.com/marin-community/marin/pull/11299),
[Hero MuonH #11301](https://github.com/marin-community/marin/pull/11301) and the
[qualified GPU runtime #11307](https://github.com/marin-community/marin/pull/11307).
While they remain unmerged, use a revision containing all three. The entrypoint
PR records the exact integration revision used for validation.

Provide Iris and regional S3 credentials, W&B access, and `DAYTONA_RL_API_KEY`
and `DAYTONA_EVAL_API_KEY`. Supply HF credentials if any selected input requires
Hub access. Harbor tasks use the configured privileged container profile, which
requires Iris admin authorization. See [RL launching](../references/rl-launching.md).

The following private example was checked on November 13, 2026. Its model is the
completed schema-3 export of SFT step 400407, export ID
`a4759d7fccffdb2fa0eb29770fc89097f501b7871f581ba83e00c8d90fe86e1b`.
Its portable-model identity is
`sha256:47a79c09972a1c8c25afb527a412354d81cd27b003726e3901f2c34939253abf`.
The data contains four original mixed tasks, including two Harbor tasks.
Both inputs were written on November 13 under a three-day TTL; treat them as
expiring on November 16. Adoption does not extend their retention. Replace these
URIs and the input version with retained artifacts for later runs.

```bash
export MARIN_PREFIX=s3://marin-us-east-08a
MODEL=s3://marin-us-east-08a/tmp/ttl=3d/romain/mochi-three-rack/2026.11.13.01a1c926.a1/hf-sft
TASKS=s3://marin-us-east-08a/tmp/ttl=3d/romain/mochi-three-rack/2026.11.13.01a1c926.a1/data/normal
INPUT_VERSION=2026.11.13.9400
TOKENIZER_REVISION=3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09
```

Browse the [model](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Ftmp%2Fttl%3D3d%2Fromain%2Fmochi-three-rack%2F2026.11.13.01a1c926.a1%2Fhf-sft)
and [tasks](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Ftmp%2Fttl%3D3d%2Fromain%2Fmochi-three-rack%2F2026.11.13.01a1c926.a1%2Fdata%2Fnormal)
through Filer with Marina login.

## Fresh start and continuation

Build the graph and write its source launch configuration without submitting:

```bash
COMMON=(--model-uri "$MODEL" --tokenizer-revision "$TOKENIZER_REVISION"
        --data-uri "$TASKS" --input-version "$INPUT_VERSION")
uv run python -m experiments.post_training.mochi_rl.launch "${COMMON[@]}" \
  --version 2026.11.13.1 --max-steps 20 --config-output mochi-launch.yaml
```

Inspect the model/data locators, role allocation, context budget and checkpoint
paths, then add `--run` to submit the same graph. A new output version starts
fresh MuonH state. Repeating an interrupted run with the same version recovers
its latest completed native checkpoint when one exists.

To extend a run or recover into a new output version, set `NATIVE` to the exact
completed native checkpoint directory, such as your run's
`checkpoints/global_step_20`, then run:

```bash
uv run python -m experiments.post_training.mochi_rl.launch "${COMMON[@]}" \
  --version 2026.11.13.2 --max-steps 25 --resume-uri "$NATIVE" \
  --config-output mochi-resume.yaml
```

Inspect it, then add `--run`. `--max-steps` is the absolute stopping step, so it
must exceed the restored step. Native continuation restores model, optimizer,
scheduler, trainer counter and available rollout state. It writes new checkpoints
under the new output version. To recover that continuation, select its latest
completed checkpoint with a new version; `--resume-uri` always selects the exact
checkpoint supplied. Keep the model, tokenizer, data and topology compatible.

Outputs use your user namespace. Recovery checkpoints and traces default to
the regional three-day TTL, with two rolling checkpoints. `--ttl-days` and
`--checkpoint-count` change that policy. Terminal HF export is disabled. Repeat
`--data-uri` for more task artifacts; `--data-file` defaults to `train.parquet`.
`--tokenizer-uri` selects a separate pinned tokenizer artifact. Python callers
can supply another topology, Megatron geometry, recipe and execution configuration
to `build_run`; this profile qualifies the one-rack layout only.
