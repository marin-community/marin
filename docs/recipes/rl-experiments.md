# RL experiments

Plan, submit and inspect an RL experiment through its model entry. A named preset binds
the task selection, initial weights and tokenizer, GPU layout, evaluation schedule and measurement
window. Start with a plan to confirm the inputs, allocation and output path.

## Which experiment to run

Preset names use `<question>-<length>`. The two loops serve different purposes.

### Short loop

The short loop, also called the quick loop, uses gated presets to compare one change against control runs.

| Preset | What it measures | Default work and estimated cost | Evaluation and gate | When to use it | Ran on GPU |
|---|---|---|---|---| --- |
| `speed-short` | Generation throughput per GPU | 8 H100s, 512 prompts × four answers. 11m42s and 1.56 H100-hours in the four-proposal comparison. | Separate pass on the EvalLite pool. Gated on throughput and EvalLite pool family mean. | Screen a vLLM wheel, kernel or engine option before testing the full training loop with `speed-medium`. | [Direct GPU comparison](https://github.com/marin-community/marin/pull/11975) |
| `data-short` | Fraction of prompts with both passing and failing answers | 8 H100s, 256 candidate training prompts × four answers, 10–15 min and about 2 H100-hours | Scores the sampled candidate tasks. Gated on mixed-answer fraction. | Screen a new `--tasks` selection before training on it with `data-medium`. | CPU plan only |
| `eval-short` | Published model scores | 8 H100s. | Scores the full EvalLite pool without a control comparison. | Evaluate a published checkpoint in a generation-only job. | Ran as written |
| `speed-medium` | Update time and policy agreement | 40 H100s, 10 updates. 20m12s and 13.47 H100-hours in the four-proposal comparison. | Training metrics on updates 2–10. Gated on speed and mismatch. | Check startup, trainer kernels, parallelism, weight synchronization or numerics. Use `learning-medium` to measure quality after a precision change. | [Direct GPU comparison](https://github.com/marin-community/marin/pull/11975) |
| `learning-medium` | EvalLite pool family mean and score gain per GPU-hour | 40 H100s, 25 updates. Wall time and cost are unmeasured. | EvalLite pool at steps 0 and 25. Gated on EvalLite pool family mean. | Compare objectives, optimizers, sampling, batch shape or initial weights. | CPU plan only |
| `data-medium` | Learning on a candidate training selection | 40 H100s, 25 updates. Wall time and cost are unmeasured. | EvalLite pool at steps 0 and 25. Reports scores for the candidate selection. | Train on a `--tasks` selection after it passes the short screen. | CPU plan only |
| `speed-long` | Sustained asynchronous update time and policy agreement | 40 H100s, 20 updates, 45–75 min and 30–50 H100-hours | Training metrics on updates 3–20. Gated on speed and mismatch. | Measure grading, admission or throughput changes over a longer window. | CPU plan only |

### Long loop

The long loop trains a model and evaluates it on held-out tasks.
Use `train-long` for EvalLite or `train-long-checkpoint-eval` to add the agentic cohorts.

| Preset | What it measures | Default work and estimated cost | Evaluation and gate | When to use it | Ran on GPU |
|---|---|---|---|---| --- |
| Snowball `train-long` | Full training and checkpoint quality | 40 H100s, 100 updates by default, hours to days. Check the plan and use the reported GPU-hours for cost. | EvalLite at step 0 and every 10 updates. Evaluation coverage invariants, without a calibrated control comparison. | Train a release candidate with checkpoint recovery and durable HF export. | Ran with a five-step SFT initializer and two RL updates |
| Snowball `train-long-checkpoint-eval` | Full training with agentic checkpoint evaluation | 40 H100s, 100 updates by default, hours to days plus Daytona sandbox costs. | EvalLite and agentic cohorts at each checkpoint evaluation. Coverage invariants without a calibrated control comparison. | Qualify training weights on the [checkpoint evaluation cohorts](https://github.com/marin-community/marin/blob/main/experiments/post_training/README.md#checkpoint-evaluation-cohorts). Requires the [Daytona evaluation account and sandbox capacity](#agentic-checkpoint-evaluation). | CPU plan only |

Mochi RL starts from the production SFT step-400407 export and its pinned Hugging Face tokenizer.
Use `--sft PRESET` to build a starting model or `--init NAME@VERSION` to use a published model.
Mochi’s medium presets run two updates and its long presets run up to 10,000 updates.
A preset ending in `-agentic` trains on agent tasks. Snowball's `train-long-checkpoint-eval` adds agent episodes to checkpoint evaluation.

| Mochi preset | When to use it | Layout | Ran on GPU |
| --- | --- | --- | --- |
| `train-medium` | Short one-turn training check. | 15 nodes × 4 GB200s, one rack | Ran with an explicit initializer and checkpointing enabled |
| `train-medium-agentic` | Short mixed run with tool-using and one-turn tasks. | 15 nodes × 4 GB200s, one rack | [Ran with reduced batches and checkpointing enabled](https://github.com/marin-community/marin/pull/11728) |
| `train-long` | One-turn training with held-out evaluation. | 15 nodes × 4 GB200s, one rack | CPU plan only |
| `train-long-3rack` | One-turn training with dedicated racks for training and inference. | 48 nodes × 4 GB200s, three racks | CPU plan only |
| `train-long-agentic` | Mixed training and held-out evaluation with a single-rack allocation. | 15 nodes × 4 GB200s, one rack | CPU plan only |
| `train-long-agentic-3rack` | Mixed training and held-out evaluation across three racks. | 48 nodes × 4 GB200s, three racks | CPU plan only |

Mochi saves no checkpoint or HF export by default. Scheduled evaluation scores the live policy.
Enable checkpointing if you need to resume. The default Mochi run path is under the cluster's
`tmp/ttl=3d/` prefix. Set `MARIN_PREFIX` to a durable S3 root for runs that may use the full 96-hour time limit.

Both Mochi layouts restrict NCCL to the `ibp` InfiniBand rails. The long presets evaluate the
EvalLite active set as `eval_lite_100`.

| Snowball preset | When to use it | Layout | Checkpoints and export | Ran on GPU |
| --- | --- | --- | --- | --- |
| `train-medium-agentic` | Ten updates with held-out evaluation on one inference engine. | `h100x88`, 11 nodes × 8 H100s | Best eligible policy. | [Ran with reduced batches and a bounded attempt](https://github.com/marin-community/marin/pull/11728) |
| `train-long-agentic` | Up to 10,000 updates with held-out evaluation on 22 inference engines. | `h100x256`, 32 nodes × 8 H100s | Best eligible policy. | CPU plan only |
| `train-medium-tools` | Ten updates on a `--tasks` selection of shell-tool tasks. | `p40`, 5 nodes × 8 H100s | None by default. | [Ran three updates](https://github.com/marin-community/marin/pull/11921) |

When a trained step has the best eligible score, a successful waiting run keeps that native
checkpoint and returns its HF export. When step zero wins, the result points to the starting HF
model and trained checkpoints stay only if `keep_every` or an outstanding export protects them.
Detached launches use the trainer’s rolling checkpoint window.

The 40-H100 `train-long` preset packs samples across two Megatron context ranks. The agentic
campaign layout splits a 102,400-token request into 20,480 packed token positions per context rank.

`train-medium-tools` runs Linux tasks and image graders on Daytona sandboxes in the RL organization, so set
`DAYTONA_RL_API_KEY`.

### Task selection and native resume

Pass `--tasks NAME@VERSION` for a mixed preset. Both task families must be present,
and training rows must be disjoint from the held-out evaluation. The
[selection README](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/README.md#snowball-training-selections)
records the composition of the published mixed selection.

Use `--resume NAME@VERSION` with checkpointing enabled to restore the newest retained recovery
checkpoint and its original model identity from a run that has stopped. A running source's
rolling cleanup can remove the checkpoint after you pin it. The source needs an immutable
version and a retained native checkpoint. `--init` starts from published weights and is mutually
exclusive with `--resume`. Set `SOURCE_RUN` to the source run's artifact name and version from its plan,
and `TASK_SELECTION` to a published mixed selection's `NAME@VERSION`.

```bash
uv run python -m experiments.post_training.mochi_rl \
  --preset train-medium-agentic --version 2026.11.19.1 --checkpoints latest \
  --tasks "$TASK_SELECTION" --resume "$SOURCE_RUN" --plan
```

### Compose SFT and RL

Use `--sft PRESET` on the RL command to build the SFT export and initialize RL from its weights
and tokenizer. Replace `--plan` with `--run` to launch both stages. Use one of `--sft`, `--init`,
or `--resume` to select the initial weights. Gated presets compare with indexed controls and refuse `--sft`.

```bash
MARIN_PREFIX=s3://marin-us-east-02a/marin \
  uv run python -m experiments.post_training.snowball_rl \
  --preset train-long --sft openthoughts-32k-4node --version 2026.11.19.1 --plan
uv run python -m experiments.post_training.mochi_rl \
  --preset train-medium --sft mixture-262k-176node --version 2026.11.19.1 --plan
```

### Convert released agent tasks

Run `uv run python -m experiments.post_training.task_curation.swe_verified_release --source "$INPUT_URI"`
from a clean commit to copy the declared Harbor bootstrap into each released TaskSpec's `setup_commands`.
The converter checks the pinned input and writes versioned Parquet plus a manifest of source,
output, commit, and per-row digests. Pin the derived file and its digest in the mixed selection recipe.

### Task health records

Mixed presets record their installed-agent attempts after RL commits its result. The runner
reports a failure of this step separately from the RL result. To extract from a stopped run,
including a failed run, use
`uv run python -m experiments.post_training.task_curation.task_health extract --cluster CLUSTER --run NAME@VERSION`.
Use `MARIN_PREFIX` if the run used an explicit or temporary storage prefix. The extractor writes
grades, phases, image and setup digests, and output tails for ungraded attempts to
`s3://marin-us-east-02a/marin/rl-data/task-health/attempts/<run>/<version>/<attempt-key>/attempts.parquet`.
The agentic preset's graph writes an `attempts.json` receipt with the snapshot's address below
`rl-data/task-health/attempts/by-run/<run>/<version>` in the run's storage prefix.
The extractor snapshots the run's full attempt store, including trial records from earlier retries
at that location. The key identifies the run location and the stopped native attempt.

From a clean commit, run `skip --selection NAME@VERSION --attempts ADDRESS... --version YYYY.MM.DD.N`
to publish a skip list. A task qualifies after two attempts with a failed setup command or verifier
reward or result error with test output, provided the task has no graded attempt. When two distinct
qualifying tasks share an image and setup, the list includes the other tasks in that environment.
`--release FILE` clears entries for tasks whose content has changed.
Pin the resulting `skip.parquet` under the selection recipe's `exclusions` with format `task_digest`.
Selection matches the exact task revision and counts these exclusions separately from held-out overlap.
Released inputs also pin an attempts record. The probe selection must use the same input digest,
and at least 90% of 50 or more distinct probed tasks must have a graded attempt.

## The loop

1. [Curate tasks](../references/task-curation.md#run-a-campaign) into an admitted corpus.
2. [Publish a selection](../references/task-curation.md#rebuild-a-snowball-training-selection)
   at an immutable version with the required evaluation exclusions.
3. [Run a preset](#which-experiment-to-run): use `data-short` then `data-medium` with
   `--tasks NAME@VERSION`, or choose the preset for the change.
4. Read the [verdict](#compare-with-controls) for a gated run. For `learning-medium`,
   `data-medium`, `train-long` and `train-long-checkpoint-eval`, open the dashboard row in `rl_progress_url`.

## Run a preset

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev --plan
```

Plans and host-side submissions run the native driver checks on CPU before Iris submission.
The rollout worker count times its CPUs per worker must fit the driver task's Iris CPU request.
Static preference training with `trainer.evaluation_runner=training` skips this check because it starts
no inference rollout pool.
A gated submission needs five completed controls for the preset and base version. The launcher
checks requested metric keys against a stored control before submitting. A plan composes the
candidate and prints its version without an Iris submission.

The command invokes the shared runner through the Snowball model entry. Track or stage source files
under `experiments/`, `lib/` and `vendor/`. The full tracked working-tree hash forms part of the run
name, so even a documentation edit can change it. `--version dev` satisfies the standard launcher
interface. The preset version fingerprints the composed launch configuration used by the gate.
The run's artifact version identifies its storage format.

- **Plan and status.** The default mode and `--plan` run the SkyRL driver's configuration checks on CPU
  before printing the graph and status, including the run name, allocation and output path. Repeat the
  plan from the same source with the same inputs to look up an existing run.
  A saved attempt's status can include its job ID, completed update and evaluations.
  Use `compare` to check its verdict against the indexed controls.
- **Submit and wait.** Replace `--plan` with `--run` to submit an Iris coordinator. Add `--wait` to keep
  the terminal report in the current shell. A detached launch prints the path to its publication log.
  Pass `--issue ISSUE_NUMBER` when the experimenter supplies an issue number.
  If tracking is undecided, ask whether to open a new issue, use an existing one, or run without an issue.
  Open an approved issue after confirming the plan. Retries of the same native output
  inherit the issue stored for that output.
- **Resume.** Repeat the original `--run --wait` invocation from the same source and inputs.
  The executor reuses successful artifacts. When training resumes, either train-long preset loads its
  latest retained checkpoint and records the new attempt and parent checkpoint.

Submission without `--wait` returns 0 when accepted. A terminal invocation returns 0 for PASS
or successful ungated completion, 1 for FAIL, 2 for INVALID, 3 for an unexpected error, failed
coordinator or unavailable required verdict, and 4 for refusal before execution.
Set `WANDB_API_KEY` before submission. `WANDB_ENTITY` selects the team and defaults to
`marin-community`. The `marin-rl` project uses the run-index ID as the run name and group, and the
native attempt ID as the W&B ID.

For the Snowball presets in the first table, short presets use eight H100s and training
presets use 40. East is `cw-us-east-02a`. The 40-H100 layout has 32 policy GPUs and eight engine GPUs,
128 prompts × four answers and a 24,576-token window. Natural replies allow 8,192 tokens.
These training presets use staleness 2 and TIS. Speed-long has no evaluation,
checkpoints or export; learner and engine token rates are descriptive telemetry.

## Dashboard publication

Publish a stored attempt after an app release, without starting native work:

```bash
uv run python -m experiments.post_training.snowball_rl publish --preset train-long \
  --run-output "$RUN_OUTPUT" --attempt "$ATTEMPT_ID" --marina-url "$MARINA_ORIGIN"
```

Set `RUN_OUTPUT` to the path in the run's status, `ATTEMPT_ID` to the selected native attempt,
and `MARINA_ORIGIN` to the target Marina origin. Both attempt selectors are required together.
Use `--issue NUMBER` to associate that stored attempt with an issue. Each native output accepts
one issue number across its attempts.
EvalDash links a saved evaluation to its Marina record when the evaluation has a native
attempt reference. Historical attempts retain their own native identity and window.
Per-attempt evaluation summaries use the evaluation results saved by that attempt.
The dashboard's detail view shows each evaluation step's counts and strata means, including
family, AAII category and difficulty, and links to the run's Grafana window.

Presets configured for grader diagnostics save them in
`exports/grader_reports/<attempt_id>/grader_report.json`. The report groups
training rollouts and each named evaluation by task subset. Each task's
`subset_id` label supplies the subset, with the source dataset as a fallback.
The report also records success and coverage by task ID. Success rates use
verified responses and are null when no verdict is available.
Open **Grader stats** from the published attempt's details to see coverage,
failure counts and grading latency. Expand **Individual tasks** for task results.
Exported evaluation responses include `task_id` and `subset_id`.
The trainer logs reporting errors and continues, so an attempt may have a partial
or missing report.

The launcher publishes every prepared attempt and its terminal result using the caller's Marina
login. The index groups attempts by `<model>-<preset>-<user>-<config hash>` from the gate's composed
comparison configuration. Each row includes source and preset provenance, allocation cluster,
status, and any linked issue. Record `rl_progress_url` with the result.
Use `--marina-url ORIGIN` to publish to a local Marina stack; the default is the Marina service.

Repeat the original command with `--wait` from the same source and inputs. The executor reuses
successful steps and runs failed work, then the launcher retries publication from stored outputs.
Errors identify the affected native attempt; missing credentials suggest `iris login`.
Publication errors preserve the experiment verdict and exit code. A current successful gated attempt
requires a report for exactly the current native attempts.

Publish a single stored native attempt with `publish --preset PRESET --run-output PATH --attempt ID`.
The attempt's run index identity carries its control flag. The same command republishes controls and
candidates. See the [dashboard and record contract](https://github.com/marin-community/marin/blob/main/infra/marina/apps/rl_progress/README.md).

## Change one input

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev --plan
uv run python -m experiments.post_training.snowball_rl --preset data-short \
  --tasks "$TASK_SELECTION" --version dev --plan
uv run python -m experiments.post_training.snowball_rl --preset train-long \
  --set trainer.max_steps=100 --version dev --plan
```

`--set` takes a typed dotted SkyRL key and a JSON value; text values may be bare.
JSON objects update nested settings. Unknown keys, wrong value types and `+` prefixes
are refused locally.
Placement, data bindings, retention, evaluation and measurement settings belong to the preset.
Learning and train-long can change task sampling, optimizer, schedule and batch shape.
Data presets require `--tasks` and refuse `--set`.

Snowball generation and disaggregated training presets using the pinned 09.21 Datakit initializer
load the frozen Dr. Doom DSpark draft.
DSpark uses probabilistic drafting and standard verification. Use
`--set generator.speculative_decoding=null` for plain decoding on presets that accept settings.
Snowball uses four proposals. The `speed-short` burst preset uses a 0.80 vLLM GPU memory
fraction. The other presets use 0.75.
To compare another draft against the same policy, set `generator.speculative_decoding.model` to a
JSON object with `source_uri` and `source_identity`. For `hf://` sources, use the full Hub commit SHA.
Use `--set generator.speculative_decoding.num_speculative_tokens=N` to sweep proposal count.
With an alternate initializer, set the complete `generator.speculative_decoding` block to use its
compatible draft. Online draft training remains EAGLE3-only.

Mochi one-rack presets using the pinned step-400407 initializer default to a
766M two-layer DSpark draft with four proposals. An alternate initializer can
supply a compatible draft through `generator.speculative_decoding`. The
one-rack vLLM memory fraction of 0.775 gave
about 119,400 KV tokens per inference GPU in the five-update trial. Use `--set generator.speculative_decoding=null`
to run plain decoding. Also set `--set generator.gpu_memory_utilization=0.75` to
reproduce the previous one-rack serving budget. Three-rack presets use plain decoding.

Set `TASK_SELECTION` to a published `NAME@VERSION`. Custom selections must use the default development panel (`recipe.heldout`)
and record the complete frozen EvalLite in their recipe's `exclusions`. The builder checks those recorded inputs.
Create an immutable candidate through [task curation](../references/task-curation.md#final-task-selection).
Selections with AI-judged tasks are refused before submission.

`--init NAME@VERSION` on `learning-medium` or either train-long preset starts from a published initializer
or a retained RL HF export. The selected initializer supplies both weights and its tokenizer.
Training recovery uses the run's native checkpoint lineage. By default, Snowball's `learning-medium`,
`data-medium`, and `train-long` export the checkpoint with the best eligible evaluation score. If step zero wins,
the terminal receipt points to the initializer's HF checkpoint. A trained winner is converted to
HF under `exports/global_step_<N>/policy`. If no evaluation is eligible, the native attempt completes
with its recovery checkpoint and no HF export. The status shows `none eligible` and the evaluation
invariants. See
[native checkpoint and attempt handling](../references/rl-launching.md) for retention and retries.

Use `--checkpoints latest --checkpoint-every 0 --export latest` to save and export the final
weights. Set a positive cadence to also save periodic checkpoints.
`--checkpoints best` requires a checkpoint cadence and evaluation. Pass `--priority production`
to request Iris's production band. The plan prints the effective checkpoint and export settings
and the requested priority, plus the output root and where it came from. Checkpoint and export
options are fixed for gated presets. Candidates can set `--priority`.

`--replicates N` makes 1–5 copies of a gated preset with one seed. `--control --replicates N`
collects the preset's defaults from a clean checkout. Controls have their collection commit
in their shared names. Collect controls only for an authorized calibration.

## Compare with controls

The gate writes `gate_report.json` and `gate_report.md`. Status prints verdict, effect, reason and report
path, followed by the Markdown tables of objectives, guards, invariants and reported statistics.
JSON comparison rows have `objective`, `guard`, `requirement`, `reported` or `slice` roles.
`run_inputs.json` records the requested settings and inputs for comparison; `run.config_changes` lists
explicit settings and derived differences. The gate estimates GPU-hours from the configured GPU
count and the time between the first recorded workload start and native job completion.

A candidate uses the latest five completed main controls from the Marina run index at the version
of its preset at the branch's main merge base. Submission freezes the exact five native attempt IDs.
If the index has fewer than five, the command reports the base version and available count.
The saved candidate must match its recorded launch inputs. All runs in a comparison use the same
held-out TaskSpec content and evaluation protocol.

Guard tolerance is `t(1 - 0.025/k, df) * sd * sqrt(1/n_run + 1/n_control)`, where `k` is the number of
guards and `sd` is the controls' sample standard deviation. A worse change beyond tolerance fails the guard.
Objectives and guards determine `improves`, `trade-off`, `regression` or `no change`; infrastructure invariants
affect the verdict independently. Slices report candidate/control means and the controls' range.

Add repeatable conditions with `--require 'KEY<=NUMBER'`, `KEY>=NUMBER`, `KEY<=control` or `KEY>=control`.
Numeric bounds apply to every candidate replicate; relative bounds compare against controls and cannot
widen a built-in guard. Unknown control metrics are refused before submission. Train-long and controls
refuse requirements. Requirements change report identity while training names and settings stay fixed.

Run stored comparisons on CPU:

```bash
uv run python -m experiments.post_training.snowball_rl compare \
  --preset learning-medium --candidate "$CANDIDATE" --require 'eval_lite/aai_category/coding/mean>=control'
```

Set `CANDIDATE` to the stored run name or `NAME@VERSION`. Pass `--marina-url ORIGIN` for a local
index. The report records the control version and exact attempts. Reports use the caller's namespace
under the preset's output root.

## Change a preset's default

1. Edit that preset's `tasks`, `model`, `held_out_tasks` or `recipe` in `snowball_rl.py`.
   A preset's `Recipe.config` is its typed override point; merge a `RecipePatch` there for a
   preset-specific default. `_DEFAULT_PATCH` supplies shared settings, including context budget.
   Trainer precision, engine precision and batch invariance use the native recipe keys through `--set`.
2. From the repository root, run the native launcher checks for Snowball presets.

   ```bash
   uv run pytest -q -n 0 experiments/post_training/rl_tests/test_rl.py::test_snowball_presets_plan_native_launches
   ```

3. The SkyRL nightly workflow on main runs each gated preset with its default inputs. A version
   with fewer than five completed controls gets enough runs to reach five. Otherwise the workflow
   starts one run per preset. Snowball's gated presets use Reno H100s. The Mochi speed and learning
   presets each use one GB200 rack.

   To refresh selected controls, dispatch the workflow on `main` with `controls=true` and a space-separated
   `presets` list. With repository write access, comment `/rl-controls snowball/learning-medium mochi`
   on a merged pull request to dispatch the same workflow and receive its run link. Use `model` for
   all gated presets of a model, or `model/preset` for one. Every selector must name a gated target.
   The workflow replies with the reason if it refuses a comment.

## Tasks and evaluation

[EvalLite](../references/task-curation.md#evallite) is the held-out selection evaluated by speed-short,
learning-medium and data-medium; train-long uses the active set. EvalLite evaluations request one
greedy answer per task with natural replies up to 8,192 tokens. Data-short evaluates a sample of 256
candidate training tasks with four sampled answers each; it does not evaluate EvalLite.

Speed-short measures 512 training tasks with four 4,096-token answers before its separate EvalLite pass.
Learning/data medium evaluate at update 0 and 25; train-long evaluates before training and every ten updates.
Speed-medium measures update time over updates 2–10, speed-long over 3–20. Their mismatch guard uses the
selected-token-weighted staleness-0 KL over the measured window. Asynchronous medium runtimes are unmeasured.

Evaluation identity includes stored TaskSpec content and the recorded request. Answers join the manifest
by task ID and count as scored only after successful verification without an exception or error treatment.
Equal-family mean averages answers per task, tasks per family, then families. Slices use family, AAII
category and agentic status. Task selection versions are immutable; follow the EvalLite reference to change one.

The primary evaluation is stored under `eval/`. Additional evaluations use `eval/<name>/`.

## Telemetry and provenance

Keep the native run ID and attempt ID together. The run identifies the experiment. The attempt
identifies one native execution, including a resumed execution. In `resolved-launch.yaml`, these
are `config.run.id` and `config.run.attempt_id`. Pair them with the output path and source revision
when recording a result.

For quick-loop training time, use `elapsed_excluding_eval_seconds` from the
`Training run completed` line in the SkyRL training driver logs. The timer runs from
tracker setup through shutdown and subtracts all inline evaluation time, including
step-zero and final evaluations when enabled.

| Sink | Where to find the run |
|---|---|
| W&B | Open `wandb_url` from status for the attempt's exact project and team. The `marin-rl` project tags runs with model, preset, version, branch and user. |
| Finelog and Grafana | Use the status's Grafana link for the run-index ID and UTC execution window. Finelog uses service `marinskyrl`. Filter by model, cluster and run ID, and use the Iris job ID and attempt timestamps to inspect a retry. |
| Marina RL progress | Open `rl_progress_url` from terminal status. The run-index ID and native attempt identify the row. |
| EvalDash | An offline benchmark run has its own evaluation run ID. Publish the IDs returned in `EvaluationResult.run_ids` on the selected RL attempt to link them from the Marina row. |

The dashboards are available through the [RL Grafana dashboard](https://grafana.oa.dev/d/marin-rl-runs/rl-post-training),
[Marina RL progress](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/) and
[EvalDash](https://marina.oa.dev/evaldash/).

## Agentic checkpoint evaluation

`train-long-checkpoint-eval` uses the pinned
[checkpoint evaluation cohorts](https://github.com/marin-community/marin/blob/main/experiments/post_training/README.md#checkpoint-evaluation-cohorts).
Set `DAYTONA_EVAL_API_KEY` before submitting. The cohorts share SkyRL's
evaluation pool with EvalLite. Daytona builds their task environments in
declarative mode, and Terminus-2 evaluates the same frozen policy as EvalLite.

A checkpoint is eligible when it attempts every task and scores at least 99% of the EvalLite
active set and 90% of each agentic cohort. `eval/all` weights verifier scores by the number
of scored tasks. Per-cohort metrics use `eval/<cohort>/all/`. Each task has one trial.
Terminus-2 can take up to 32 main turns with up to 8,192 reply tokens per request within
the engine's serving window. Each cohort uses its configured verifier deadline.
Infrastructure failures can retry once. A result with unconfirmed sandbox cleanup is unscored.

Agentic cohort results are stored under the named evaluation directories in `exports/`.
Repeat the command to resume from the latest retained checkpoint. The separate
[offline evaluation launcher](../explanations/evaluation.md) evaluates an exported model.

## What a run writes

The status block names the run's artifact output and native job, with its Iris URL and GPU-hours
summed from the recorded checkpoint lineage's native job windows and allocations. Submission prints an open-ended
Grafana link immediately; terminal status prints the recorded native time window. For evaluation
presets, `evaluations` lists scored and attempted answer counts and the means for each weight step.
Train-long checks every scheduled evaluation, requires at least 99% of its tasks to score, and
returns a failed status when this evidence is missing or below the bound. Its status also names the
best checkpoint step, or `none eligible`.

| Location | Contents and reader |
|---|---|
| `resolved-launch.yaml`, `terminal.json` | Native configuration and result; `read_stored_skyrl_attempt(output)` in `marin.rl.skyrl_records` reads the retained attempt without Iris. |
| `run_inputs.json` | The runner's declared task, initializer, settings and requirements. |
| `exports/run-records/` | FineStore attempt records, numeric metrics and per-task evaluations. |
| FineStore `evaluations` table | One task per attempt, evaluation and step, with task content digest, ordered answers, native rewards, verifier bounds, normalized verifier scores, outcomes and canonical strata. `finestore.rl.run_records.read_evaluations` reads it. |
| `exports/dumped_evals/<attempt_id>/...` | JSONL answers carrying attempt, step and final-turn identity; additional evaluations also have their own export subtree. |
| Native checkpoint root | The receipt records recovery checkpoint locations and lineage. Snowball uses 14-day scratch storage. When enabled, Mochi writes checkpoints under its run path. |
| `exports/best_checkpoint.json` | The selected step and its evaluation evidence, written when a checkpoint is eligible. |
| HF policy | The completed receipt identifies the HF model for `--init`, either the initializer or a converted checkpoint under `exports/`. |
| Gate artifact `gate_report.json`, `gate_report.md` | Calibrated comparison, invariants and breakdowns for a gated preset. Train-long has evaluation invariants without a calibrated verdict. |

Evaluation summaries include the native answer-weighted overall mean, task-weighted family,
AAII-category, source, topic and difficulty means, and an AAII-weighted mean. The category weights
are 0.2 scientific reasoning, 0.2 coding, 0.3 General and 0.3 Agents, renormalized over represented
categories; format has zero AAII weight. Missing annotations are `untagged`. These summaries do
not add a gate threshold. A versioned selection's `summary.json` declares its strata schema,
reference models, difficulty bounds and parent; `task_manifest.parquet` retains all parent tasks,
with active, candidate or retired membership. A previous version's score uses that version's active
IDs and TaskSpec digests against the saved per-task results.

## Gaps

| gap | implication and follow-up |
|---|---|
| layout-p40-only | calibrated training gates cover the 40-GPU layout; other layouts need qualification |
| mismatch-is-agreement | mismatch measures policy agreement; exactness claims need an explicit zero-mismatch bound |
| EvalLite coverage | the current panel does not qualify multi-turn tools or long context; see the [EvalLite composition](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/README.md#evallite) |
| no-image-sandbox | image-only tasks need a sandbox backend |

## Add a model entry

The shared `experiments/post_training/rl.py` builds the CLI and executes a `ModelPresets` value.
`rl_presets.py` defines the types and `rl_compose.py` composes launch specifications. `rl_measurements.py`
reads a stored run's metrics, scores and invariants. `rl_scoring.py` computes scores from answers,
and `rl_gate.py` compares a candidate's measurements with indexed controls.

`snowball_rl.py` is Snowball's model entry. Its `SNOWBALL` value contains the presets, the
`p40` and `e8` layouts, immutable model and task references, and evaluation selections.
The layouts declare GPU type, CPU, memory, disk, priority and both job deadlines; their role plans
supply the GPU width. Presets declare seed, W&B entity, checkpoint retention and HF export
independently of dashboard publication. Its `__main__` calls `rl.main(SNOWBALL)`.

To add a model, create an entry file and pass its `ModelPresets` to `rl.main`. Each preset supplies
its own model, tasks, layout and recipe. A preset with held-out evaluations declares
`held_out_tasks` and the evaluation profiles. A `mochi_rl.py` entry fits the same interface with the
model label `mochi`, its own namespace, and Mochi's initializer, resources and evaluation choices.
The preset table's namespace owns run and control paths.

Python callers can pass `init: SkyRLModelSource` to `rl.main(table, argv=..., init=...)` or
`build_experiment(preset, table=..., init=...)`. An `ArtifactHfModel` accepts any `ArtifactStep`;
`tokenizer_uri=None` uses the export's own tokenizer files. An explicit tokenizer reference
and revision are preserved. `--init NAME@VERSION` resolves a published model to the same type;
an unpinned Hub tokenizer reference is resolved to a launch-time commit. Python `init` and
`--init` together are refused. Planning retains an unbuilt initializer's producer dependency
without reading its export. Stored inputs preserve its identity and export selector;
comparison uses the concrete export URI recorded by the candidate. At RL startup, after the
model is built, the checkpoint and loaded tokenizer vocabularies must match before engines load.

The coordinator re-enters the calling module with the exact arguments. Run model entries with
`python -m PACKAGE.MODULE`; callers without module metadata can set `entry_module` on `rl.main`.
Its `forward_env=("NAME", ...)` argument requires and forwards extra environment variables to
the coordinator and native jobs, including terminal export. Configuration and logs contain
names only. A layout's coordinator deadline can be changed with `dataclasses.replace`.

## Evaluate and publish a checkpoint

Use `experiments.post_training.skyrl_evaluation.evaluate_model` to score a model in a separate evaluation
job. Pass a `SkyRLPolicyModel` for an exported RL policy or an `ArtifactHfModel` for an SFT export,
a serving `ModelConfig` with `location=""`, evaluation definitions, a version, and the accelerator
and cluster settings.
The artifact supplies the model URI, identity, and tokenizer. The serving config supplies the engine
and generation settings.

`resolve_evaluations` accepts registered suites such as `standard-local` or comma-separated evaluation
names such as `mmlu-smoke,gsm8k-smoke`. The result gives the evaluation group ID, per-evaluation run IDs,
records prefix, and result paths. Each run writes a record and an archive with per-question results.
An RL policy record includes `rl_attempt` when both the training run ID and attempt ID are available.

Use Snowball's `eval-short` to score published weights with the same EvalLite protocol as the
trainer, in a generation-only job. Use trainer evaluation to track policy scores at the run's
scheduled training steps. `eval-short` runs the full EvalLite pool on one node with eight H100s.
A recorded run used 1.1 H100-hours.

```bash
uv run python -m experiments.post_training.snowball_rl \
  --preset eval-short --init "$MODEL" --version dev --plan
```

Set `MODEL` to a published `NAME@VERSION`. Replace `--plan` with `--run` to submit.

To publish a checkpoint export, enable export and set `trainer.hf_hub_repo_id` in the training recipe.
Gated Snowball presets fix that setting. For a manual checkpoint export use `--hf-hub-repo-id` on
`cloud.iris.export_hf_checkpoint`. A repository ID enables publication. New repositories are public by
default. Select private visibility with `trainer.hf_hub_private: true` or `--hf-hub-private` on a manual
export. The uploader rejects an existing repository with different visibility. It uploads
`.marinskyrl-model-manifest.json` and the files it lists, then checks each file's size and SHA-256 at
the upload commit. With `hf_upload_mode: all`, the step archive receives the same checks as the
repository root.
