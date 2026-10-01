# CatCountCanary

CatCountCanary checks whether pretrained Qwen2.5-0.5B-Instruct learns to
answer prompts asking for exactly N copies of `cat`, through MarinSkyRL's
Megatron trainer. Training uses counts 1–20 except held-out counts
`3, 5, 13, 16`. Evaluation also reports extrapolation counts `24, 28`.
Held-out and extrapolation results do not determine the learning verdict.

Both lanes use two Megatron data-parallel ranks on one H100 host and two
vLLM engines on another. Each task requests two H100s and 65 CPUs; the CPU
request places our tasks on separate 128-CPU hosts. Four Gym rollout workers
on the policy host reserve 32 CPUs. Gym uses token requests; the HTTP
endpoint serves Harbor and is off in this recipe. Training metrics and
policy-training and rollout spans are enabled explicitly.

`--lane async` uses behavior clipping, permits two steps of rollout staleness
and applies two update epochs per batch. `--lane sync` uses zero staleness,
the regular policy objective and TIS with a cap of 2. Both use the same
trainer loop, 64 prompts with eight samples each, a full-batch mini-batch,
matching training and forward micro-batches of 16 per GPU, and learning
rate 2e-6.

## Submit a run

From the marin repository root, print the plan without submitting:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset dry
```

Inspect the model, dataset and training artifact versions and the pinned
MarinSkyRL `launcher_requirement`. Use a fresh calendar version and owned
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
    --enable-extra-resources --extra cpu --timeout 9000 \
    --max-retries 0 --no-wait -- \
    python -m experiments.post_training.cat_count_canary \
    --cluster cw-rno2a --job-timeout-seconds 7200 "$@" --run
}

submit_canary atqamar-cat-count-dry-20260930 \
  --version 2026.09.30.1 --preset dry

submit_canary atqamar-cat-count-async-20260930 \
  --version 2026.09.30.2 --preset gate --lane async

submit_canary atqamar-cat-count-sync-20260930 \
  --version 2026.09.30.3 --preset gate --lane sync

submit_canary atqamar-cat-count-filter-20260930 \
  --version 2026.09.30.4 --preset gate-filter --lane async
```

The coordinator builds model and data dependencies, submits the training
child, and exports the final checkpoint. Its timeout must cover those three
stages; `--job-timeout-seconds` bounds only the training child. Monitor queue
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
| `calibrate` | 30 | 30 | Exploratory training without a default early stop. |
| `gate` | 50 | 25 | Stops at a greedy training evaluation gain of 0.2. |
| `gate-filter` | 50 | 25 | Applies the same learning check with zero-variance groups discarded and replaced. |
| `on-policy` | 30 | 30 | Sets async staleness to zero and uses one update epoch. |

Evaluation runs at step 0 and every five completed steps; `dry` evaluates
every step. It records greedy responses and eight sampled responses per
count at temperature 1. The gate requires improvement in
`eval/train/avg_score` from step 0. Reaching the cap without that gain fails.
`--eval-reward-rise` overrides the preset's margin. The launcher's stop
margin must equal the spec's `min_improvement`. A stopping step writes a
checkpoint and a final HF export.

Grouped metrics include `eval/{train,heldout,extrapolation}/avg_score`,
`eval/{train,heldout,extrapolation}/environment/exact` and their
`eval/sampled/` equivalents. Per-count exact rates are
`eval/cat_count_n{N}/environment/cat_count/exact`. `pass_at_8` means at least
one sampled response is exact; the environment exact rate is the fraction of
responses that are exact. `eval/train/avg_score_improvement` is the stopping
signal. Training logs also report `environment/exact_n{N}` and
`reward/zero_std_group_fraction`.

`--model` selects the instruct model, `qwen2.5-0.5b`, or `qwen3-0.6b`.
`--batch-size`, `--group-size`, `--micro-train-batch-size`, repeated
`--train-n` and `--seed` select experiment inputs. Sampled evaluation always
uses eight responses, independently of the training group size. `--set`
accepts existing `trainer.*`, `generator.*` and `context_budget.*` fields;
data paths, model paths, role geometry and derived fields are owned by the
launcher. Print an exploratory plan before using the submission function:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30.5 --preset calibrate \
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
`launcher_requirement`. From that checkout's `skyrl-train` directory, select
`ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json` or
`ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-sync.json`, then run:

```bash
uv run --frozen python -m ci.marin_nightly.gate \
  --log /tmp/cat-count-child.log --spec "$CAT_COUNT_SPEC" \
  --wall-clock-seconds "$CHILD_DURATION_SECONDS"
```

The checker verdict determines success. A process exit alone does not prove
learning. `gate-filter` uses the async learning spec and additionally requires
a positive `async/dynamic_sampling/discarded_rate` in the complete log.

A pass demonstrates task learning through Iris launch, model staging,
Megatron DP=2, separate-host vLLM, NCCL weight synchronization, Gym rollout
workers, the policy objective, two-epoch reuse, evaluation, checkpoints and
export. The two lanes share a loop, so their evidence is partly shared.
The canary does not establish loss-scale parity, DP parameter equality,
exported-output equivalence, GPU checkpoint-resume correctness, or coverage
of TP/PP/CP/EP, MoE, multi-turn tools, Harbor, LoRA or long contexts. Subtle
clipping, probability, template or partial weight-sync errors can still learn.

## Calibration results

The selected configuration uses learning rate 2e-6. Four seeds in each lane
reached a gain of 0.2 with evaluation every five steps. The cap is the worst
first crossing plus five steps; each run stops at its first crossing.

| Lane | Seed | First crossing | Peak and final eval | Ordinary step median |
| --- | ---: | ---: | ---: | ---: |
| Async | 17 | 20 | 0.806 | 5.36 s |
| Async | 23 | 35 | 0.827 | 5.26 s |
| Async | 31 | 15 | 0.840 | 5.60 s |
| Async | 47 | 45 | 0.826 | 5.06 s |
| Sync | 17 | 5 | 0.799 | 6.23 s |
| Sync | 23 | 15 | 0.809 | 5.73 s |
| Sync | 31 | 20 | 0.856 | 5.64 s |
| Sync | 47 | 10 | 0.809 | 6.11 s |

These measurements use marin commit `c57ae7b8de8a905dcd89f1b021e1ec351e534001`
and MarinSkyRL commit `1f130218e56590075363d59cd94fb97c292d65e0`.
The initial greedy training reward was 0.555–0.605. Async training tasks
took 664–1,119 seconds; sync tasks took 524–765 seconds, including startup
and teardown. Ordinary step medians exclude step 1 and evaluation steps.
Sync seed 31 completed training but its finished worker was lost; its
training data is retained and its export is absent.

Model staging, Ray and vLLM startup happen once per invocation. The dry
step's measured 72 seconds included 5.7 seconds of policy training and
49 seconds of checkpoint work. Checkpoint work recurs at save intervals;
it is not an ordinary-step cost. Evaluation also recurs every five steps.
HF export runs after training.

Sampled policy-GPU memory maxima in seven completed runs ranged from 17.1
to 30.5 GiB per device. These are node telemetry samples, not CUDA allocation
peaks; sync seed 31 has no joined memory receipt.
