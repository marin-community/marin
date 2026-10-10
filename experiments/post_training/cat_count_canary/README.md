# CatCountCanary

CatCountCanary checks whether pretrained Qwen2.5-0.5B-Instruct learns to
answer prompts asking for exactly N copies of `cat`, through MarinSkyRL's
Megatron trainer. Training uses counts 1–20 except held-out counts
`3, 5, 13, 16`. Evaluation also reports extrapolation counts `24, 28`.
Held-out and extrapolation results do not determine the learning verdict.

The canary uses two Megatron data-parallel ranks on one H100 host and two
vLLM engines on another. Each task requests two H100s and 65 CPUs; the CPU
request places tasks on separate 128-CPU hosts. Four task rollout workers
on the policy host reserve 32 CPUs and use token requests. The model HTTP
endpoint is off in this recipe. Training metrics and
policy-training and rollout spans are enabled explicitly.

`--lane async` uses behavior clipping, permits two steps of rollout staleness
and applies two update epochs per batch. `--lane sync` is a manual experiment
outside the canary and CI. It uses zero staleness,
the regular policy objective and truncated importance sampling (TIS) with a cap of 2.
The two lanes use the same trainer loop, 64 prompts with eight samples each, a full-batch mini-batch,
matching training and forward micro-batches of 16 per GPU, and learning
rate 2e-6.

## Submit a run

From the marin repository root, print the plan without submitting:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset dry
```

Inspect the model, dataset and training artifact versions and the pinned
MarinSkyRL `runtime.launcher_commit`. Use a fresh calendar version and owned
job name for every measured run and after any runtime repin.

Check live capacity before submitting:

```bash
uv run iris --cluster marin rpc controller list-peers
```

Use `cw-rno2a` first. Use `cw-us-east-02a` only when a fresh peer read shows
idle H100 hosts. Follow the repository's
[use-iris guide](../../../.agents/skills/use-iris/SKILL.md) for budget and
scheduled-priority inspection. The requested priority is interactive; inspect
the actual receiving-side priority. The scheduler budget is observational for
this canary. A priority downgrade alone does not invalidate a run.

Submit measured runs with a 4-CPU, 16-GB-memory, 8-GB-disk coordinator. The
shared launcher's default coordinator requests 64 GB of disk, which can
exceed the target cluster's disk quota. This shell function uses the canary's
required coordinator settings for every preset:

```bash
submit_canary() {
  canary_job_name="$1"
  shift
  uv run iris --cluster marin job run \
    --target-cluster cw-rno2a --job-name "$canary_job_name" \
    --priority interactive --cpu 4 --memory 16GB --disk 8GB \
    --enable-extra-resources --extra cpu --timeout 1200 \
    --max-retries 0 --no-wait -- \
    python -m experiments.post_training.cat_count_canary \
    --cluster cw-rno2a --job-timeout-seconds 1200 "$@" --run
}

submit_canary atqamar-cat-count-dry-20260930 \
  --version 2026.09.30.1 --preset dry

submit_canary atqamar-cat-count-async-20260930 \
  --version 2026.09.30.2 --preset gate --lane async

submit_canary atqamar-cat-count-sync-20260930 \
  --version 2026.09.30.3 --preset gate --lane sync
```

The coordinator resolves the pinned model and data dependencies and submits
the training child. Checkpoints and HF export are enabled only with
`--checkpoint` and `--export`; export also enables its prerequisite checkpoint.
Both are off by default and in CI.
The default model reads the immutable rl-canaries Qwen snapshot, and outputs
resolve below `s3://marin-us-east-02a/marin/rl-canaries/cat-count/gpu/runs/`.
The coordinator timeout covers the whole run; `--job-timeout-seconds` bounds
only the training child. Monitor queue
time as part of a separate submission-to-completion deadline. Verify the
job is yours before cancelling it:

```bash
uv run iris --cluster marin job describe "$COORDINATOR_JOB_ID"
uv run iris --cluster marin job cancel "$COORDINATOR_JOB_ID"
```

Cancel an owned run after more than two coordinator preemptions or its
chosen deadline. Stop submissions after two consecutive runs both have a
priority below interactive and more than two coordinator preemptions.
Infrastructure retries use a fresh job name and artifact version. A gate run
that resumes after preemption or retry is inconclusive: its early-stop
baseline is the resumed evaluation, rather than the original step-0 policy.

## Presets and evaluation

| Preset | Async cap | Sync cap | Behavior |
| --- | ---: | ---: | --- |
| `dry` | 1 | 1 | Initial evaluation and one training step. |
| `gate` | 30 | 30 | Stops at sampled training evaluation reward ≥0.65. |

Evaluation runs at step 0 and every five completed steps; `dry` evaluates
every step. It records greedy responses and eight sampled responses per
count at temperature 1. The async gate requires step 0 sampled reward in
[0.10, 0.45] and `eval/sampled/train/avg_score` ≥0.65 by step 30.
`--eval-minimum-score` selects an exploratory sampled-score stopping threshold.
The checked-in gate preset and async spec use 0.65. A stopping step ends
training; checkpoints and HF export require their explicit flags.
The 0.65 threshold comes from a [calibration run](https://wandb.ai/dogml/marin-cat-count-canary/runs/1jr0tjpg)
with a different runtime. This evidence does not establish the threshold for the TaskSession runtime.
The [TaskSession validation](../task_sessions/validation.py) checks training and exports,
but does not recalibrate the canary gate.
Megatron logs `policy/dp_weight_checksum_mismatch` after every optimizer step;
any mismatch fails the spec. The grouped step metric keeps the maximum across
optimizer windows. Greedy, held-out and extrapolation scores are reported only.

Grouped metrics include `eval/{train,heldout,extrapolation}/avg_score`,
`eval/{train,heldout,extrapolation}/environment/exact` and their
`eval/sampled/` equivalents. Per-count exact rates are
`eval/cat_count_n{N}/environment/cat_count/exact`. `pass_at_8` means at least
one sampled response is exact; the environment exact rate is the fraction of
responses that are exact. `eval/sampled/train/avg_score` is the stopping signal. Training logs also report `environment/exact_n{N}` and
`reward/zero_std_group_fraction`.

Exact responses contain lowercase `cat` words separated by single spaces,
with no punctuation or extra text. The scorer removes outer whitespace and
trailing end-of-turn markers. `avg_score` is the shaped reward, not the exact
rate. The [scorer](https://github.com/marin-community/MarinSkyRL/blob/c96f6d25f59959096c3b179a51e67c0af6c99536/skyrl-gym/skyrl_gym/envs/cat_count/reward.py)
defines partial-count rewards and penalties.

`--model` selects `qwen2.5-0.5b-instruct`, `qwen2.5-0.5b`, or `qwen3-0.6b`.
`--batch-size`, `--group-size`, `--micro-train-batch-size`, repeated
`--train-n` and `--seed` select experiment inputs. Sampled evaluation always
uses eight responses, independently of the training group size. `--set`
accepts existing `trainer.*`, `generator.*` and `context_budget.*` fields;
data paths, model paths, role geometry and derived fields are owned by the
launcher. Print an exploratory plan before using the submission function:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30.5 --preset dry \
  --set trainer.policy.optimizer_config.lr=1e-6
```

A changed model, count mix or optimization setting needs its own calibration
and spec. A different seed tests the same recipe. Data artifact names depend
on their counts, row count and seed; training names depend on the full recipe.

## Score the run

Read the coordinator's complete log and find its submitted training child:

```bash
uv run iris --cluster marin job logs "$COORDINATOR_JOB_ID" \
  --no-tail --max-lines 60000
uv run iris --cluster marin job describe "$CHILD_JOB_ID"
uv run iris --cluster marin job logs "$CHILD_JOB_ID" \
  --no-tail --max-lines 60000 > /tmp/cat-count-child.log
```

Confirm the terminal state twice, a minute apart, and record preemptions.
Check that the log includes step 0, every completed training step, final
evaluation and the training-completion marker. Use the actual child duration
for `CHILD_DURATION_SECONDS`.

Check out the exact MarinSkyRL commit named by the marin plan's
`runtime.launcher_commit`. From that checkout's `skyrl-train` directory, select
`ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json`, then run:

```bash
uv run --frozen python -m ci.marin_nightly.gate \
  --log /tmp/cat-count-child.log --spec "$CAT_COUNT_SPEC" \
  --wall-clock-seconds "$CHILD_DURATION_SECONDS"
```

The checker verdict determines success. A process exit alone does not prove
learning. The async gate checks sampled reward and the required training
metrics; synchronous runs are available for manual comparisons.

A pass demonstrates task learning through Iris launch, model staging,
Megatron DP=2, separate-host vLLM, NCCL weight synchronization, task rollout
workers, the policy objective, two-epoch reuse and evaluation. Parameter
checksum comparisons detect reported differences between training ranks.
The canary does not establish loss-scale parity, exported-output equivalence,
GPU checkpoint-resume correctness, or coverage of TP/PP/CP/EP, MoE, multi-turn
tools, Harbor, LoRA or long contexts. Subtle clipping, probability, template
or partial weight-sync errors can still learn.
