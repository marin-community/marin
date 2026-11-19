# RL experiments

Run an experiment from a named preset. The command binds the versioned task selection,
Datakit SFT weights and tokenizer, GPU layout, evaluation schedule and measurement window.

## Which experiment to run

| change | presets |
|---|---|
| vLLM wheel, kernel or engine option | speed-short, then speed-medium |
| trainer kernel, parallelism, recompute or exact numerics | speed-medium, then learning-medium |
| weight synchronization | speed-medium |
| fp8 experts or KV-cache precision | speed-medium, then learning-medium; a person assesses the trade-off |
| loss, estimator, optimizer, schedule, reward, task sampling or batch shape | learning-medium |
| task selection | data-short, then data-medium |
| SFT initializer | learning-medium with `--init` |
| startup latency | speed-medium, which reports startup |
| asynchronous grading, admission or loop performance | speed-long |
| release training | train-long |
| release training with agentic checkpoint evaluation | train-long-harbor |

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
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
```

A plan without `--require` uses the declared control pin. `--require` validates the pin and requested metric
against control r0 even during planning. Gated execution and re-scoring require the pin's revision to match
the preset; a mismatch returns exit 4 before submission or INVALID during gate execution. Controls are collected
after merge; `--control` and `train-long` need no pin.

The default command and `--plan` print the graph and saved status. Add `--run` to submit an Iris
coordinator, or `--run --wait` to poll every 60 seconds and print the final report. Submission
requires `--issue ISSUE_NUMBER` or `--no-issue` and records that choice on the native attempt.
Repeating the same command reuses completed artifacts. Track or stage source files under
`experiments/`, `lib/` and `vendor/`: their working-tree hash forms part of the run name.
`--version dev` satisfies the standard launcher interface; each preset's steps use its `revision`.

Exit codes are 0 for PASS or a completed ungated run, 1 for FAIL, 2 for INVALID,
3 for an unexpected error, failed coordinator or unavailable required verdict, and 4 for refusal before execution. Submission without
`--wait` returns 0 when accepted. Status lists completed updates, native attempt and job IDs,
W&B URLs and a Grafana dashboard link. Set `WANDB_API_KEY` before submission;
`WANDB_ENTITY` selects the team, with `marin-community` as the default. The project is
`marin-snowball-rl`. The team does not affect run identity.

Short presets use eight H100s; training presets use 40.
East is `cw-us-east-02a`. The 40-H100 layout has 32 policy GPUs and eight engine GPUs,
128 prompts × four answers and a 24,576-token window. Natural replies allow 8,192 tokens.
All training presets use staleness 2 and TIS. Speed-long has no evaluation,
checkpoints or export; learner and engine token rates are descriptive telemetry.

## Dashboard publication

Publish a stored attempt after an app release, without starting native work:

```bash
uv run python -m experiments.post_training.snowball_rl publish --preset train-long \
  --output RUN_OUTPUT --attempt ATTEMPT_ID --url MARINA_ORIGIN
```

Without `--output` and `--attempt`, that command publishes the selected preset's pinned controls.
Both selectors are required together. Pass the complete `EvaluationResult.run_ids` tuple with one
`--evaldash-run-id RUN_ID` per evaluation definition. Providing IDs sets the entire list for the
selected attempt. Publishing without IDs preserves its saved links. Historical attempts retain their
own native identity and window. Unattributed historical JSONL dumps do not supply per-attempt
evaluation summaries.
The dashboard's detail view shows each evaluation step's counts and strata means, including
family, AAII category and difficulty, and links to the run's Grafana window.


After `--run --wait` observes the coordinator finish, the launcher publishes learning-medium,
data-medium and train-long attempts using the caller's Marina login. The row shows the model and
the launch's issue link or explicit opt-out. Record the finishing status's `rl_progress_url` with the verdict.
Use `--url ORIGIN` to publish to a local Marina stack; the default is the Marina service.

Repeat the original command with `--wait` from the same source and inputs. The executor reuses
successful steps and runs failed work, then the launcher retries publication from stored outputs.
Errors identify the affected native attempt; missing credentials suggest `iris login`.
Publication errors preserve the experiment verdict and exit code. A current successful gated attempt
requires a report for exactly the current native attempts.

Publish learning-medium or data-medium controls from their stored paths:

```bash
python -m experiments.post_training.snowball_rl publish --preset learning-medium
```

The pins in `snowball_rl.SNOWBALL.controls` must match the preset's current revision; unavailable pins
produce a usage error. Speed presets and data-short publish nothing. See the
[dashboard and record contract](https://github.com/marin-community/marin/blob/2a13c0c0627c2a6b2dd6b44d898edd530b4e1b6d/infra/marina/apps/rl_progress/README.md) for native identity,
recorded measurements and stored gate facts.

## Change one input

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev
uv run python -m experiments.post_training.snowball_rl --preset data-short \
  --tasks my-selection@2026.11.10.1 --version dev
uv run python -m experiments.post_training.snowball_rl --preset train-long \
  --set trainer.max_steps=100 --version dev
```

`--set` takes a typed dotted SkyRL key and a JSON value; text values may be bare.
JSON objects update nested settings. Unknown keys, wrong value types and `+` prefixes
are refused locally.
Placement, data bindings, retention, evaluation and measurement settings belong to the preset.
Learning and train-long can change task sampling, optimizer, schedule and batch shape.
Data presets require `--tasks` and refuse `--set`.

Custom selections must use the default heldout development input and record the complete
frozen EvalLite in their recipe's `exclusions`. The builder checks those recorded inputs.
Create an immutable candidate through [task curation](../references/task-curation.md#final-task-selection).
Selections with AI-judged tasks are refused before submission.

`--init NAME@VERSION` on a learning or train-long preset selects a published model or kept
HF export. Learning-medium and data-medium retain final HF weights for both controls and
candidates. Durable exports live under `exports/global_step_<N>/policy`.
Train-long resumes its latest checkpoint and keeps HF exports. See
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
  --preset learning-medium --candidate RUN_NAME[@VERSION] --require 'eval_lite/aai_category/coding/mean>=control'
uv run python -m experiments.post_training.snowball_rl rescore \
  --preset speed-medium --leave-out r0
```

Leave-one-out compares one control replicate against the remaining controls with fixed default inputs.
Reports use the caller's namespace under `MARIN_PREFIX`; controls are read-only.

## Change a preset's default

1. Edit that preset's `tasks`, `model`, `eval_lite` or `recipe` in `snowball_rl.py`.
   A preset's `Recipe.config` is its typed override point; merge a `RecipePatch` there for a
   preset-specific default. `_DEFAULT_PATCH` supplies shared settings, including context budget.
   Trainer precision, engine precision and batch invariance use the native recipe keys through `--set`.
2. Bump that preset's `revision`. Data-medium shares learning-medium's settings and revision.
3. From the repository root, print the complete compared-configuration hash:

   ```bash
   uv run pytest -s -q -n 0 experiments/post_training/rl_tests/test_rl.py::test_presets_match_revisioned_native_configs
   ```

   A new revision intentionally fails until its printed hash is added under that preset in
   the test's `revisions` table. Inspect the composed settings before accepting the hash.
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

To compute each EvalLite version for a run, read the stored task rows for one attempt, evaluation
and checkpoint step. Load each version's selection with `read_selection_strata` and pass it to
`score_eval_lite_version`.

```python
from finestore.rl.run_records import read_evaluations
from finestore.rl.strata import read_selection_strata
from experiments.post_training.rl_scoring import score_eval_lite_version

rows = read_evaluations(export_root, attempt_id=attempt_id, evaluation="eval/eval_lite_100/", step=step)
for root in version_roots:
    panel = read_selection_strata(root)
    if panel is None:
        raise ValueError(f"Missing selection manifest: {root}")
    print(panel.definition.version, score_eval_lite_version(rows, panel))
```

Use the evaluation name stored on the rows. A primary evaluation uses `eval/`. The result reports
how many active tasks scored, which are missing verified answers, and which have missing or different
content digests. A version has a score only when every active task has verified answers with matching
content. To compare earlier versions whose active tasks have since retired, evaluate the retained
pool so the run includes those tasks.

Train-long-harbor adds 30 pinned BFCL, 30 SWE-bench Verified and 30 OpenThoughts-TBLite trials
on Daytona at each evaluation. Each sandbox builds its pinned task environment in declarative mode.
Set `DAYTONA_EVAL_API_KEY` and `DAYTONA_TARGET` before submission. Checkpoint choice
requires at least 99% native coverage and 90% in each Harbor cohort; `eval/all` pools native and
Harbor scored rewards. Per-cohort metrics are `eval/harbor/<cohort>/*`; `eval/harbor/wall_seconds` measures the
complete Harbor collection callback. The agent input allowance reserves its output tokens within the
engine serving window. SWE-bench uses the pinned harness's 3,000-second verifier timeout. A trial
can retry once for a phase cancellation, sandbox command timeout or Daytona conflict. Teardown
errors after verification retain the reward and are recorded alongside it. Trial artifacts live under
`exports/inline_harbor/`. A Harbor error stops training; rerunning the same command resumes from
the latest checkpoint. `best_checkpoint.json` records the selected checkpoint evidence. The native result records
the HF export and its model files. This runtime evaluates the current training weights; the separate
[offline evaluation launcher](../explanations/evaluation.md) evaluates an exported model.

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
| `exports/best_checkpoint.json` | The selected native checkpoint step and its evaluation evidence, written when a checkpoint is eligible. |
| HF export under `exports/` | Model configuration, weight index and shards, and tokenizer files. The completed receipt identifies the HF model for `--init`. |
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
| harbor-on-daytona | train-long-harbor needs the eval account, sandbox capacity and agentic runtime qualification |

## Add a model entry

The shared `experiments/post_training/rl.py` builds the CLI and executes a `ModelPresets` value.
`rl_presets.py` defines its types; `rl_recipe.py` composes launch specifications; `rl_gate.py`
compares stored results; `rl_scoring.py` reads measurements. These modules contain no model table.

`snowball_rl.py` is Snowball's model entry. Its `SNOWBALL` value contains the presets, the
`p40` and `e8` layouts, immutable model and task references, evaluation selections and control pins.
The layouts declare GPU type, CPU, memory, disk, priority and both job deadlines; their role plans
supply the GPU width. Presets declare seed, W&B entity, checkpoint retention and HF export
independently of dashboard publication. Its `__main__` calls `rl.main(SNOWBALL)`.

To add a model, create its own entry file with explicit `Preset.model`, `Preset.tasks`,
`Preset.eval_lite`, evaluations and recipe configuration, then pass its `ModelPresets` value to
`rl.main`. The table's namespace owns run and control paths. The shared runner imports no other
model entries. The native launch-document census counts `rl_recipe.py`, where `SkyRLSpec` is
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
