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

## The loop

1. [Curate tasks](../references/task-curation.md#run-a-campaign) into an admitted corpus.
2. [Publish a selection](../references/task-curation.md#rebuild-a-snowball-training-selection)
   at an immutable version with the required evaluation exclusions.
3. [Run a preset](#which-experiment-to-run): use `data-short` then `data-medium` with
   `--tasks NAME@VERSION`, or choose the preset for the change.
4. Read the [verdict](#compare-with-controls) for a gated run.

## Run a preset

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
```

A gated preset without a control pin at its current revision refuses every command, including the plan print,
with exit 4. Controls are collected after merge; `--control` and `train-long` need no pin.

Without `--run`, the command prints the graph and saved status. Add `--run` to submit an Iris
coordinator, or `--run --wait` to poll every 60 seconds and print the final report.
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
