# CatCountCanary

`cat_count_canary.py` runs a counting-task RL experiment through MarinSkyRL's
Megatron trainer. Each prompt asks for exactly N copies of `cat`. The default
policy is Qwen2.5-0.5B-Instruct, and the default training counts are
all values from 1 through 20 except held-out counts `3, 5, 13, 16`.
Evaluation reports those held-out counts and extrapolation counts `24, 28`
without using them to stop training or gate learning.

The default `async` lane uses two Megatron policy data-parallel ranks
and two vLLM rollout engines in a two-node Iris task with two H100s per node
in `cw-rno2a`. Use `--cluster cw-us-east-02a` to submit the coordinator
and training job to the other H100 cluster. Both lane selections use the unified trainer through the
standard runtime entrypoint. The default `async` selection allows
rollouts to lag by two policy steps and uses behavior clipping without TIS.
`--lane sync` uses zero staleness with the regular policy objective and TIS.
The lanes share one loop, so running both checks different loss and staleness
settings within that loop. Each task requests 65 CPUs,
which puts the two tasks on separate 128-CPU H100 hosts. The pinned MarinSkyRL runtime
contains the CatCount environment and the production Gym rollout-worker path.
Gym rollouts use token requests inside four rollout workers on the policy node;
the workers reserve 32 of its 65 CPUs. The HTTP endpoint is off because it serves
Harbor. Training metrics, policy-training spans and rollout spans are explicit
in the recipe. Check live H100
capacity, the effective job priority and the user budget before submitting a
run.

## Plan and run

From the marin repository root, print the artifact plan first:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset dry
```

The plan lists the model, parquet data and training artifacts with their
resolved versions. It does not submit a job. Check that its
`launcher_requirement` names a MarinSkyRL commit containing the CatCount
environment. Replace the example date with a fresh calendar version for a
measured run. After inspecting the plan and completing the Iris preflight,
submit the one-step check:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset dry --run
```

`--run` submits an Iris coordinator. The coordinator builds the model and data
dependencies before the training job. A `dev` version is mutable and rebuilds
on each run.

After the one-step check succeeds, use `calibrate` for exploratory settings.
Compare step-0 and final exact-match rates at N=10 and N=20, greedy evaluation
reward, and train reward. A setting qualifies for the gate when two seeds pass
the learning checks, followed by a run that confirms the combined settings.
Record those settings and their margins in the gate specs. The `gate` preset and
its specs must use that same recipe. These commands submit the default gate
recipe in each lane:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset gate --lane async --run

uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset gate --lane sync --run
```

Use `--preset gate-filter --lane async --run` to check dynamic
filtering separately. Inspect each plan without `--run` before submission.

| Preset | Step cap | Behavior |
| --- | ---: | --- |
| `dry` | 1 | Evaluates the initial policy and runs one update. |
| `calibrate` | 30 | Runs a short training comparison. |
| `gate` | 60 | Runs the learning check. |
| `gate-filter` | 60 | Runs the async learning check with zero-variance groups discarded and replaced. |
| `on-policy` | 30 | Sets async staleness to zero and uses one update epoch. |

Each evaluation records one greedy reply and eight sampled replies at
temperature 1 per count. The default evaluation interval is five completed
steps; `dry` evaluates every step. Grouped training, held-out and extrapolation
reward and exact rates are reported for both sampling profiles.
`--eval-reward-rise` requests an early stop when greedy training evaluation
reward improves by that margin over the invocation's initial evaluation.
Omitting it runs to the step cap. Phase B determines the final margin and cap;
the current defaults are exploratory. Checkpoints and the final HF export
use the actual completed step when training stops early.

`--model` selects `qwen2.5-0.5b-instruct` (default), `qwen2.5-0.5b`, or
`qwen3-0.6b`. `--batch-size`, `--group-size`, `--micro-train-batch-size`, repeated `--train-n`, and
`--seed` change the corresponding run inputs. `--job-timeout-seconds` bounds
the Iris training child job (default: 7200 seconds); set it above the measured
healthy run time when testing a hang. For example, this plan adds
N=32 to the training mix and changes the learning rate:

```bash
uv run python -m experiments.post_training.cat_count_canary \
  --version 2026.09.30 --preset calibrate \
  --train-n 1 --train-n 2 --train-n 4 --train-n 7 \
  --train-n 10 --train-n 20 --train-n 32 \
  --set trainer.policy.optimizer_config.lr=1e-6
```

Add `--run` after inspecting this plan to submit the exploratory run. The
calibrated gate specs will apply to the pinned gate recipe with the default model,
training counts, batch size and group size. A changed model, count mix or
optimization setting needs its own calibration and spec. Changing `--seed`
tests another run of the same recipe.

The initial throughput settings are 64 prompts with eight training samples
each, matching training and forward micro-batches of 16 per GPU. Phase B
compares matching micro-batches of 8, 16 and 32 on the new runtime pin and records
step time and memory. Inference running-request and queue metrics remain
active alongside the explicit training telemetry.

`--set` accepts existing `trainer.*`, `generator.*`, and `context_budget.*`
keys. The launcher rejects data paths, model paths, role-plan geometry,
derived context fields, and their parent objects. Change those through the
typed options or the launcher code. Each distinct effective recipe has a
separate artifact name; a repeated recipe and version refers to the same
artifact.

## Read the result

`--run` prints the coordinator job ID. Set `COORDINATOR_JOB_ID` to that ID,
then read its log with the Marin Iris CLI:

```bash
uv run iris --cluster marin job logs "$COORDINATOR_JOB_ID" --no-tail --max-lines 60000
```

The coordinator log has a `[rl-iris] Submitted: <child-job-id>` line. Set
`CHILD_JOB_ID` to that ID. The child job description reports task duration,
state and preemption count. Collect the training child's complete log:

```bash
uv run iris --cluster marin job describe "$CHILD_JOB_ID"
uv run iris --cluster marin job logs "$CHILD_JOB_ID" --no-tail --max-lines 60000 > /tmp/cat-count-child.log
```

Confirm the child reached a terminal state in `job describe`, and check that
`/tmp/cat-count-child.log` covers every completed training step. Its
`WANDB_MIRROR` lines contain step-indexed train and eval metrics. If early or
final steps are missing, fetch the log again with a larger `--max-lines` value
before scoring it.
Initial and periodic evaluations cover the training, held-out and extrapolation
counts. `environment/exact_n{N}` is the exact-match rate on training rollouts;
compare its early and late values for hard N. `eval/cat_count_n{N}/avg_score`
is greedy shaped reward on evaluation replies.
`eval/sampled/cat_count_n{N}/avg_score` and
`eval/sampled/cat_count_n{N}/environment/exact` describe the eight sampled
replies. `pass_at_8` reports whether any of those replies succeeds; the
environment exact metric is the fraction of replies that is exact.
`eval/train/avg_score_improvement` records the rise used for stopping when
`--eval-reward-rise` is set. Compare step 0 with the final evaluation and
report held-out and extrapolation results alongside the training results. `reward/zero_std_group_fraction` is the
share of sample groups with identical rewards. The async lane records
staleness and uses behavior clipping. The sync lane applies TIS; async TIS
diagnostics report importance ratios against rollout probabilities without
applying a TIS weight.

From the MarinSkyRL `skyrl-train` directory, set `CAT_COUNT_SPEC` to
`ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json` for the
default async lane or
`ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-sync.json` for the
sync lane. GPU calibration and these CatCount specs are pending on the round-1 pin.
Once the specs are available, set
`CHILD_DURATION_SECONDS` to the measured child-job duration in seconds, then
score its complete log:

```bash
uv run --frozen python -m ci.marin_nightly.gate \
  --log /tmp/cat-count-child.log --spec "$CAT_COUNT_SPEC" \
  --wall-clock-seconds "$CHILD_DURATION_SECONDS"
```

The `gate-filter` run must pass its learning criteria and report a positive
`async/dynamic_sampling/discarded_rate`. A successful process exit alone does
not establish learning; use the gate result and its per-metric diagnostics.
