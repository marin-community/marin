# RL experiments

Plan, submit and inspect an RL experiment through its model entry file. A named preset binds
the task selection, initial weights and tokenizer, GPU layout, evaluation schedule and measurement
window. Start with a plan to confirm the inputs, allocation and output path.

## Which experiment to run

Each preset name is `<question>-<length>`, with short presets making no training updates,
medium presets making 10–25, and long presets making 20 or more.

| Preset | What it measures | Default work and estimated cost | Evaluation and gate | When to use it |
|---|---|---|---|---|
| `speed-short` | Generation throughput per GPU | 8 H100s, 512 prompts × four answers, 14–17 min and about 2 H100-hours | Separate full EvalLite pass. Gated on throughput and EvalLite family mean. | Screen a vLLM wheel, kernel or engine option before testing the full training loop with `speed-medium`. |
| `data-short` | Fraction of prompts with both passing and failing answers | 8 H100s, 256 candidate training prompts × four answers, 10–15 min and about 2 H100-hours | Scores the sampled candidate tasks. Gated on mixed-answer fraction. | Screen a new `--tasks` selection before training on it with `data-medium`. |
| `eval-short` | Published model scores | 8 H100s. The recorded run used 1.1 H100-hours. | Scores the full EvalLite pool without a control comparison. | Evaluate a published checkpoint in a generation-only job. |
| `speed-medium` | Update time and policy agreement | 40 H100s, 10 updates. Wall time and cost are unmeasured. | Training metrics on updates 2–10. Gated on speed and mismatch. | Check startup, trainer kernels, parallelism, weight synchronization or numerics. Use `learning-medium` to measure quality after a precision change. |
| `learning-medium` | EvalLite family mean and score gain per GPU-hour | 40 H100s, 25 updates. Wall time and cost are unmeasured. | Full EvalLite at steps 0 and 25. Gated on EvalLite family mean. | Compare objectives, optimizers, sampling, batch shape or initial weights. |
| `data-medium` | Learning on a candidate training selection | 40 H100s, 25 updates. Wall time and cost are unmeasured. | Full EvalLite at steps 0 and 25. Shares the `learning-medium` gate and controls. | Train on a `--tasks` selection after it passes the short screen. |
| `speed-long` | Sustained asynchronous update time and policy agreement | 40 H100s, 20 updates, 45–75 min and 30–50 H100-hours | Training metrics on updates 3–20. Gated on speed and mismatch. | Measure grading, admission or throughput changes over a longer window. |
| `train-long` | Full training and checkpoint quality | 40 H100s, 100 updates by default, hours to days. Check the plan and use the reported GPU-hours for cost. | EvalLite's 100 active tasks at step 0 and every 10 updates. Evaluation coverage invariants, without a calibrated control comparison. | Train a release candidate with checkpoint recovery and durable HF export. |

## The loop

1. [Curate tasks](../references/task-curation.md#run-a-campaign) into an admitted corpus.
2. [Publish a selection](../references/task-curation.md#rebuild-a-snowball-training-selection)
   at an immutable version with the required evaluation exclusions.
3. [Run a preset](#which-experiment-to-run): use `data-short` then `data-medium` with
   `--tasks NAME@VERSION`, or choose the preset for the change.
4. Read the [verdict](#compare-with-controls) for a gated run. For `learning-medium`,
   `data-medium` and `train-long`, open the dashboard row in `rl_progress_url`.

## Run a preset

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev --plan
```

A plan without `--require` uses the declared control pin. With `--require`, planning also validates the pin
and checks that control r0 recorded the requested metric. Gated execution and re-scoring require the pin's revision to match
the preset; a mismatch returns exit 4 before submission or INVALID during gate execution. Controls are collected
after merge; `--control` and `train-long` need no pin.

The command invokes the shared runner through the Snowball model file. Track or stage source files
under `experiments/`, `lib/` and `vendor/`. The full tracked working-tree hash forms part of the run
name, so even a documentation edit can change it. `--version dev` satisfies the standard launcher
interface. Each preset's steps use its own `revision`.

- **Plan and status.** The default mode and `--plan` run the SkyRL driver's configuration checks on CPU
  before printing the graph and status, including the run name, allocation and output path. Repeat the
  plan from the same source with the same inputs to look up an existing run.
  For a saved attempt, status can include its job ID, completed update, evaluations and verdict.
- **Submit and wait.** Replace `--plan` with `--run` to submit an Iris coordinator. Add `--wait` for
  the terminal report. Pass `--issue ISSUE_NUMBER` when the experimenter supplies an issue number.
  If tracking is undecided, ask whether to open a new issue, use an existing one, or run without an issue.
  Open a new issue only with the experimenter's explicit approval. The native attempt records the issue when supplied.
- **Resume.** Repeat the original `--run --wait` invocation from the same source and inputs.
  The executor reuses successful artifacts. When training resumes, `train-long` loads its
  latest retained checkpoint and records the new attempt and parent checkpoint.

Submission without `--wait` returns 0 when accepted. A terminal invocation returns 0 for PASS
or successful ungated completion, 1 for FAIL, 2 for INVALID, 3 for an unexpected error, failed
coordinator or unavailable required verdict, and 4 for refusal before execution.
Set `WANDB_API_KEY` before submission. `WANDB_ENTITY` selects the team and defaults to
`marin-community`. Snowball logs to the `marin-snowball-rl` project.

Short presets use eight H100s; training presets use 40.
East is `cw-us-east-02a`. The 40-H100 layout has 32 policy GPUs and eight engine GPUs,
128 prompts × four answers and a 24,576-token window. Natural replies allow 8,192 tokens.
All training presets use staleness 2 and TIS. Speed-long has no evaluation,
checkpoints or export; learner and engine token rates are descriptive telemetry.

## Dashboard publication

Publish a stored attempt after an app release, without starting native work:

```bash
uv run python -m experiments.post_training.snowball_rl publish --preset train-long \
  --output "$RUN_OUTPUT" --attempt "$ATTEMPT_ID" --url "$MARINA_ORIGIN"
```

Set `RUN_OUTPUT` to the path in the run's status, `ATTEMPT_ID` to the selected native attempt,
and `MARINA_ORIGIN` to the target Marina origin. Both attempt selectors are required together.
Pass the complete `EvaluationResult.run_ids` tuple with one
`--evaldash-run-id RUN_ID` per evaluation definition. Providing IDs sets the entire list for the
selected attempt. Publishing without IDs preserves its saved links. Historical attempts retain their
own native identity and window. Unattributed historical JSONL dumps do not supply per-attempt
evaluation summaries.
The dashboard's detail view shows each evaluation step's counts and strata means, including
family, AAII category and difficulty, and links to the run's Grafana window.

After `--run --wait` observes the coordinator finish, the launcher publishes learning-medium,
data-medium and train-long attempts using the caller's Marina login. The row shows the model and
the issue link when the launch supplied a number. Record the finishing status's `rl_progress_url` with the verdict.
Use `--url ORIGIN` to publish to a local Marina stack; the default is the Marina service.

Repeat the original command with `--wait` from the same source and inputs. The executor reuses
successful steps and runs failed work, then the launcher retries publication from stored outputs.
Errors identify the affected native attempt; missing credentials suggest `iris login`.
Publication errors preserve the experiment verdict and exit code. A current successful gated attempt
requires a report for exactly the current native attempts.

Publish learning-medium or data-medium controls from their stored paths:

```bash
uv run python -m experiments.post_training.snowball_rl publish --preset learning-medium --url "$MARINA_ORIGIN"
```

The pins in `snowball_rl.SNOWBALL.controls` must match the preset's current revision; unavailable pins
produce a usage error. Speed presets and data-short publish nothing. See the
[dashboard and record contract](https://github.com/marin-community/marin/blob/2a13c0c0627c2a6b2dd6b44d898edd530b4e1b6d/infra/marina/apps/rl_progress/README.md) for native identity,
recorded measurements and stored gate facts.

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

Set `TASK_SELECTION` to a published `NAME@VERSION`. Custom selections must use the default held-out
development input and record the complete
frozen EvalLite in their recipe's `exclusions`. The builder checks those recorded inputs.
Create an immutable candidate through [task curation](../references/task-curation.md#final-task-selection).
Selections with AI-judged tasks are refused before submission.

`--init NAME@VERSION` on `learning-medium` or `train-long` starts from a published initializer
or a retained RL HF export. The selected initializer supplies both weights and its tokenizer.
Training recovery uses the run's native checkpoint lineage. Learning-medium, data-medium and
train-long publish the checkpoint with the best eligible evaluation score. If step zero wins,
the terminal receipt points to the initializer's HF checkpoint. A trained winner is converted to
HF under `exports/global_step_<N>/policy`. See
[native checkpoint and attempt handling](../references/rl-launching.md) for retention and retries.

`--replicates N` makes 1–5 copies of a gated preset with one seed. `--control --replicates N`
collects the preset's defaults from a clean checkout. Controls have their collection commit
in their shared names. Collect controls only for an authorized calibration.

## Compare with controls

The gate writes `gate_report.json` and `gate_report.md`. Status prints verdict, label, reason and report
path, followed by the Markdown tables of objectives, constraints, invariants and reported statistics.
JSON comparison rows have `objective`, `constraint`, `requirement`, `reported` or `slice` roles.
`gate_inputs.json` records the requested settings and inputs for re-scoring; `run.config_changes` lists
explicit settings and derived differences. Native elapsed time and allocated GPUs determine gain per H100-hour.

Each gated preset uses one control pin in `snowball_rl.SNOWBALL.controls`, with a source commit, revision and
replicate count. Data-medium shares learning-medium's pin. Before submission, the command composes the
candidate on CPU and checks every control against current defaults. During scoring, candidate configuration
must match its recorded inputs. The report states why a comparison is INVALID or an invariant fails.

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
uv run python -m experiments.post_training.snowball_rl rescore \
  --preset learning-medium --candidate "$CANDIDATE" --require 'eval_lite/aai_category/coding/mean>=control'
uv run python -m experiments.post_training.snowball_rl rescore \
  --preset speed-medium --leave-out r0
```

Set `CANDIDATE` to the stored run name or `NAME@VERSION`. Leave-one-out compares one control
replicate against the remaining controls with fixed default inputs.
Reports use the caller's namespace under `MARIN_PREFIX`; controls are read-only.

## Change a preset's default

1. Edit that preset's `tasks`, `model`, `held_out_tasks` or `recipe` in `snowball_rl.py`.
   A preset's `Recipe.config` is its typed override point; merge a `RecipePatch` there for a
   preset-specific default. `_DEFAULT_PATCH` supplies shared settings, including context budget.
   Trainer precision, engine precision and batch invariance use the native recipe keys through `--set`.
2. Bump that preset's `revision`. Data-medium shares learning-medium's settings and revision.
3. From the repository root, check the composed configuration against the golden fixture:

   ```bash
   uv run pytest -s -q -n 0 experiments/post_training/rl_tests/test_rl.py::test_presets_match_revisioned_native_configs
   ```

   Inspect the composed settings and add the new revision's hash under that preset in the
   test's `revisions` table.
   The compared-configuration hash includes actual execution memory and priority.
4. Controls are collected by CI after the default change merges, once per changed default.
   That CI workflow is planned separately; control collection is not a merge gate.
   The manual `--control` command remains available for an explicitly authorized calibration.
   Source, revision and replicate count live in `snowball_rl.SNOWBALL.controls`.
5. Validate a newly collected cohort with `snowball_rl rescore --leave-out rN` for every replicate.
   Every leave-one-out verdict must PASS.

## Tasks and evaluation

[EvalLite](../references/task-curation.md#evallite) is the held-out selection evaluated by speed-short,
learning-medium and data-medium; train-long uses its 100-task subset. EvalLite evaluations request one
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

To score earlier EvalLite versions, choose the run output, native attempt, evaluation name and
checkpoint step. Set `version_roots` to the paths of the published selections you want to compare.
The stored launch supplies the export root, evaluation prefix and expected answers per task.

```python
from finestore.rl.run_records import read_evaluations
from finestore.rl.strata import read_selection_strata
from marin.rl.skyrl_records import read_stored_skyrl_attempts, recorded_launch
from experiments.post_training.rl_scoring import evaluation_protocol, native_prefix, score_eval_lite_version

_, attempts = read_stored_skyrl_attempts(run_output)
launch = recorded_launch(attempts[attempt_id])
answers = evaluation_protocol(launch, evaluation_name)["n_samples_per_prompt"]
rows = read_evaluations(
    launch.export_root,
    attempt_id=attempt_id,
    evaluation=native_prefix(launch, evaluation_name),
    step=step,
)
for root in version_roots:
    panel = read_selection_strata(root)
    if panel is None:
        raise ValueError(f"Missing selection manifest: {root}")
    print(panel.definition.version, score_eval_lite_version(rows, panel, expected_answers=answers))
```

Use a preset's evaluation name, such as `eval_lite_100` for train-long. The primary evaluation is
stored under `eval/`, and named evaluations use `eval/<name>/`. The result reports scored tasks,
missing verified answers and digest mismatches. A version has a score only when all of its active
tasks have matching content and complete verified answers. Evaluate the full retained pool to
include tasks that were active in earlier versions.

## Telemetry and provenance

Keep the native run ID and attempt ID together. The run identifies the experiment. The attempt
identifies one native execution, including a resumed execution. In `resolved-launch.yaml`, these
are `config.run.id` and `config.run.attempt_id`. Pair them with the output path and source revision
when recording a result.

| Sink | Where to find the run |
|---|---|
| W&B | Open the attempt's `wandb_url` from status. The Snowball project is `marin-snowball-rl` in the launch's W&B team. The model tag identifies Snowball runs in that project. |
| Finelog and Grafana | Use the status's Grafana link for the native run and UTC execution window. Finelog uses service `marinskyrl`. Filter by model, cluster and run ID, and use the Iris job ID and attempt timestamps to inspect a retry. |
| Marina RL progress | Open `rl_progress_url` from terminal status. The row is keyed by native run and attempt IDs and carries the model, issue link when supplied, and evaluation summaries. |
| EvalDash | An offline benchmark run has its own evaluation run ID. Publish the IDs returned in `EvaluationResult.run_ids` on the selected RL attempt to link them from the Marina row. |

The dashboards are available through the [RL Grafana dashboard](https://grafana.oa.dev/d/marin-rl-runs/rl-post-training),
[Marina RL progress](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/) and
[EvalDash](https://marina.oa.dev/evaldash/).

## What a run writes

The status block names the run's artifact output and native job. Submission prints an open-ended
Grafana link immediately; terminal status prints the recorded native time window. For evaluation
presets, `evaluations` lists scored and attempted answer counts and the means for each weight step.
Train-long checks every scheduled evaluation, requires at least 99% of its tasks to score, and
returns a failed status when this evidence is missing or below the bound. Its status also names the
best checkpoint step, or `none eligible`.

| Location | Contents and reader |
|---|---|
| `resolved-launch.yaml`, `terminal.json` | Native configuration and result; `read_stored_skyrl_attempt(output)` in `marin.rl.skyrl_records` reads the retained attempt without Iris. |
| `gate_inputs.json` | The runner's declared task, initializer, settings and requirements. |
| `exports/run-records/` | FineStore attempt receipts, numeric metrics and per-task evaluations. `read_skyrl_metrics(output, attempt_id=...)` reads a selected attempt. |
| FineStore `evaluations` table | One task per attempt, evaluation and step, with task content digest, ordered answers, native rewards, verifier bounds, normalized verifier scores, outcomes and canonical strata. `finestore.rl.run_records.read_evaluations` reads it. |
| `exports/dumped_evals/<attempt_id>/...` | JSONL answers carrying attempt, step and final-turn identity; named evaluations also have their own export subtree. |
| Native checkpoint root | Recovery checkpoints in the 14-day TTL bucket. The receipt records their location and lineage; they are separate from durable exports. |
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
| AAII coverage; no-agentic-tasks | scientific reasoning, coding, General and a single-turn Agents stand-in; multi-turn tools and long context need separate qualification |
| no-image-sandbox | image-only tasks need a sandbox backend |

## Add a model entry

The shared `experiments/post_training/rl.py` builds the CLI and executes a `ModelPresets` value.
`rl_presets.py` defines its types; `rl_recipe.py` composes launch specifications; `rl_gate.py`
compares stored results; `rl_scoring.py` reads measurements. These modules contain no model table.

`snowball_rl.py` is Snowball's model entry. Its `SNOWBALL` value contains the presets, the
`p40` and `e8` layouts, immutable model and task references, evaluation selections and control pins.
The layouts declare GPU type, CPU, memory, disk, priority and both job deadlines; their role plans
supply the GPU width. Presets declare seed, W&B entity, checkpoint retention and HF export
independently of dashboard publication. Its `__main__` calls `rl.main(SNOWBALL)`.

To add a model, create an entry file and pass its `ModelPresets` to `rl.main`. Each preset supplies
its own model, tasks, layout and recipe. A preset with held-out evaluations declares
`held_out_tasks` and the evaluation profiles. A `mochi_rl.py` entry fits the same interface with the
model label `mochi`, its own namespace, and Mochi's initializer, resources and evaluation choices.
The model table's namespace owns run and control paths.
The native launch-document census counts `rl_recipe.py`, where `SkyRLSpec` is
constructed; add the new table's builders to `tests/rl/launch_document_census.py` the same way
as Snowball. Use `assert_revisioned_native_configs` in `skyrl_test_support.py` with the table's
pinned revisions and native hashes.

Python callers can pass `init: SkyRLModelSource` to `rl.main(table, argv=..., init=...)` or
`build_run(preset, table=..., init=...)`. An `ArtifactHfModel` accepts any `ArtifactStep`;
`tokenizer_uri=None` uses the export's own tokenizer files. An explicit tokenizer reference
and revision are preserved. `--init NAME@VERSION` resolves a published model to the same type;
an unpinned Hub tokenizer reference is resolved to a launch-time commit. Python `init` and
`--init` together are refused. Planning retains an unbuilt initializer's producer dependency
without reading its export. Stored inputs preserve its identity and export selector;
rescoring uses the concrete export URI recorded by the candidate. At RL startup, after the
model is built, the checkpoint and loaded tokenizer vocabularies must match before engines load.

The coordinator re-enters the calling module with the exact arguments. Run model files with
`python -m PACKAGE.MODULE`; callers without module metadata can set `entry_module` on `rl.main`.
Its `forward_env=("NAME", ...)` argument requires and forwards extra environment variables to
the coordinator and native jobs, including terminal export. Configuration and logs contain
names only. A layout's coordinator deadline can be changed with `dataclasses.replace`.

A control can omit a configuration key when its current value equals the native SkyRL schema
default. An omitted key whose current value differs from that default makes the control stale.
Candidate receipts must match their recorded inputs exactly.

## Evaluate and publish a checkpoint

Use `experiments.post_training.skyrl_evaluation.evaluate_model` to score a model in a separate evaluation
job. Pass a `SkyRLPolicyModel` for an exported RL policy or an `ArtifactHfModel` for an SFT export,
a serving `ModelConfig`, evaluation definitions, a version, and the accelerator and cluster settings.
The artifact supplies the model URI, identity, and tokenizer. The serving config supplies the engine
and generation settings. Use `job_timeout_seconds` to set a deadline for the evaluation job.

`resolve_evaluations` accepts registered suites such as `standard-local` or comma-separated evaluation
names such as `mmlu-smoke,gsm8k-smoke`. The result gives the evaluation group ID, per-evaluation run IDs,
records prefix, and result paths. Each run writes a record and an archive with per-question results.
An RL policy record includes `rl_attempt` when both the training run ID and attempt ID are available.

Use Snowball's `eval-short` to score published weights with the same EvalLite protocol as the
trainer, in a generation-only job. Use trainer evaluation to track policy scores at the run's
scheduled training steps. `eval-short` runs the full EvalLite pool on one node with eight H100s.
A recorded run used 1.1 H100-hours. Its 100-task subset scored 0.57. [EvalLite run](https://iris.oa.dev/#/job/%2Fatqamar%2Fusers-ahmad-checkpoints-snowball-rl-eval-short-224fc25b8e-in-169b86d8-2026.11.19.1-coordinator-5dc8b038)

```bash
uv run python -m experiments.post_training.snowball_rl \
  --preset eval-short --init "$MODEL" --version dev --plan
```

Set `MODEL` to a published `NAME@VERSION`. Replace `--plan` with `--run` to submit.

To publish a checkpoint export, enable export and set `trainer.hf_hub_repo_id` in the training recipe.
Gated Snowball presets fix that setting. For a manual checkpoint export use `--hf-hub-repo-id` on
`cloud.iris.export_hf_checkpoint`. Uploads default to private. Select public visibility with
`trainer.hf_hub_private: false` or `--no-hf-hub-private` on a manual export. The publisher checks
the repository's visibility before uploading. It uploads `.marinskyrl-model-manifest.json` and the
files it lists, then checks each file's size and SHA-256 at the upload commit. Files larger than 64 MiB require LFS
digest metadata. With `hf_upload_mode: all`, the step archive receives the same checks as the
repository root.
