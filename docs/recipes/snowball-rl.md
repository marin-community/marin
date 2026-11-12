# Snowball RL experiments

## The loop

1. [Curate tasks](../references/task-curation.md#run-a-campaign) into an admitted corpus.
2. [Publish a selection](../references/task-curation.md#rebuild-a-snowball-training-selection)
   at an immutable version with the required evaluation exclusions.
3. [Run a preset](#which-experiment-to-run): use `data-short` then `data-medium` with
   `--tasks NAME@VERSION`, or choose the preset for the change.
4. Read the [verdict](#compare-with-controls) for a gated run.

## Run a preset

Run an experiment from a named preset. The command binds the versioned task selection,
Datakit SFT weights and tokenizer, GPU layout, evaluation schedule and measurement window.

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
```

Without `--run`, the command prints the graph and saved status. Add `--run` to submit an Iris
coordinator, or `--run --wait` to poll every 60 seconds and print the final report.
Repeating the same command reuses completed artifacts. Track or stage source files under
`experiments/`, `lib/` and `vendor/`: their working-tree hash forms part of the run name.
`--version dev` satisfies the standard launcher interface; each preset's steps use its `revision`.

Exit codes are 0 for PASS or a completed ungated run, 1 for FAIL, 2 for INVALID,
3 for an unavailable required verdict, and 4 for refusal before execution. Submission without
`--wait` returns 0 when accepted. Status lists completed updates, native attempt and job IDs,
W&B URLs and a Grafana dashboard link. Set `WANDB_API_KEY` before submission;
`WANDB_ENTITY` selects the team, with `marin-community` as the default. The project is
`marin-snowball-rl`. The team does not affect run identity.

| preset | allocation | workload and measurement |
|---|---|---|
| speed-short | 8 H100, east | 512 training tasks × four answers × 4,096 tokens; engine tokens/s/GPU, then full-panel quality; about 15 minutes |
| data-short | 8 H100, east | 256 candidate training tasks × four answers, up to 8,192 tokens; mixed-answer fraction; about 15 minutes |
| speed-medium | 40 H100, east | ten synchronous updates with 4,096-token answers; median update time over updates 2–10 and staleness-0 k3 mismatch; about 40 minutes |
| speed-long | 40 H100, east | 20 asynchronous updates, staleness 2 and TIS; median update time over updates 3–20 and staleness-0 k3 mismatch; 45–75 minutes |
| learning-medium | 40 H100, east | full greedy EvalLite at update 0 and after 25 synchronous RLOO-N updates; final equal-family mean and score gain per H100-hour; about two hours |
| data-medium | 40 H100, east | learning-medium on candidate tasks, using learning-medium controls |
| train-long | 40 H100, east | asynchronous training, default 100 updates; checkpoints, 100-task panel every ten updates and HF export; no verdict |

East is `cw-us-east-02a`. The 40-H100 layout has 32 policy GPUs and eight engine GPUs,
128 prompts × four answers and a 24,576-token window. Natural replies allow 8,192 tokens.
Speed-long and train-long use staleness 2 and TIS. Speed-long has no evaluation,
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
The collected controls retain `snowball-100k@2026.11.04.1`; the updated selection recipe is a
separate unpublished version and must be built before use.

`--init NAME@VERSION` on a learning or train-long preset selects a published model or kept
HF export. Learning-medium and data-medium retain final HF weights for both controls and
candidates. Durable exports live under `exports/global_step_<N>/policy`.
Train-long keeps checkpoints and exports weights by default. Repeating its command resumes
its latest checkpoint under the same run identity. Temporary checkpoints expire after 14 days;
see [checkpoint handling](../references/rl-launching.md).
Records jobs use zero preemption retries. After a preemption, repeat the train-long command
to start a fresh attempt from its latest saved checkpoint.

`--replicates N` makes 1–5 copies of a gated preset with one seed. `--control --replicates N`
collects the preset's defaults from a clean checkout. Controls have their collection commit
in their shared names. Collect controls only for an authorized calibration.

## Compare with controls

The gate writes `gate_report.json` and `gate_report.md`. Status opens with objective values
(control → candidate), percentage change and calibrated noise percentage, followed by constraints,
invariants and the verdict and label. Training score gain per H100-hour uses the run's update-0
and final panel scores, native elapsed time and allocated GPUs. Missing elapsed time makes that
objective unavailable.

JSON stores each comparison once in `comparisons`, with an `objective`, `constraint`,
`requirement`, `reported` or `slice` role. An objective that is also guarded has its `passes`
value on the same row. `run.attempt_ids` identifies the native attempts being compared.
The typed preset, normalized settings, tasks, initializer and requirements are saved in
`gate_inputs.json` beside the native receipt, including for Python `build_run` callers.
Re-scoring reads that file rather than interpreting a command line.

The gate selects exact control names in `snowball_rl_gate.CONTROLS` at the selected preset's
revision. Pins record the collection source and replicate count. Multiple pins pool deviations
from each pin's own mean. A missing pin refuses submission with exit 4 and a collection command.
The submitted inputs and current defaults are composed on CPU before submission, including for
`train-long`. Every control must match those defaults in
all compared keys; every candidate must match its recorded inputs. Submission checks every pinned
control before starting a GPU job and refuses stale configurations with exit 4 and the differing
keys. Config differences found during scoring make the comparison INVALID.
`run.config_changes` lists requested settings, input bindings and derived configuration changes in JSON,
Markdown and status.

For each built-in guard, tolerance is
`t(1 - 0.025/k, df) * sd * sqrt(1/n_run + 1/n_control)`, with fixed `k = len(preset.guards)`.
Candidate and control means must differ in the worse direction by more than that tolerance to
fail the guard. An objective that is not a guard uses `k = len(preset.objectives)` for its
calibrated change. Update time uses the median over measured updates; inconsistent timing-phase
evidence is INVALID. Staleness-0 KL uses the selected-token-weighted mean over the entire
measured window, requires finite observations for nonempty updates, and treats an empty window
as INVALID. Pooled KL is telemetry.

Only objectives and built-in guards determine `improves`, `trade-off`, `regression` or
`no change`. Failed infrastructure invariants affect the verdict independently. Zero-spread
comparisons have an unavailable change; a required calibrated guard with zero spread is INVALID.
Reported metrics have directionless `changed`, `same` or `unavailable` results. Flat slice rows
show scored task count and candidate/control means with the control minimum and maximum.

Add repeatable conditions with `--require 'KEY<=NUMBER'`, `KEY>=NUMBER`, `KEY<=control` or
`KEY>=control`. Numeric bounds apply to every candidate replicate. Each relative requirement
uses `k=1` independently; repeating a built-in condition uses its existing row. Requirements
cannot widen a built-in guard's threshold. A relative requirement with zero spread needs a
numeric bound. Preflight checks the key against control r0's computed metrics. Train-long and
controls refuse requirements. Python callers use `build_run(requirements=(...))`. Requirements
affect report identity, while training names and configurations stay fixed.

All planned updates must finish. Training policy loss, final loss, logged loss statistics and
gradients must be finite, panel coverage must reach 99%, grader infrastructure failures must be
at most 1%, and fixed-length speed replies must have the prescribed length. The grader-error
check uses the conservative upper bound on overlapping error and rollout-failure counts; a
bound above 1% fails. Missing or inconsistent counters fail that invariant. Missing receipts,
provenance, guards, mismatch counts, or incompatible panel/configuration evidence make the
comparison INVALID.

Run stored comparisons on CPU:

```bash
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset learning-medium --candidate RUN_NAME[@VERSION] --require 'eval_lite/aaii_category/coding/mean>=control'
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset speed-medium --leave-out r0
```

Automatic gates use the requirements on the current command. A cached native run retains its
original saved inputs; standalone re-scoring appends requirements to those original inputs.
Leave-one-out uses a control pin's fixed
default inputs and compares that replicate against the remaining controls. Reports use the
caller's namespace under `MARIN_PREFIX`; controls are read-only. Status lists changed RL paths
against the control commit, or gives the reason a local git comparison is unavailable.

## Change a preset's default

1. Edit that preset's `tasks`, `model`, `eval_lite` or `settings` in `snowball_rl_presets.py`.
   Each field starts with a shared default. To promote a winning setting for one preset, give
   that preset `settings=(*_DEFAULT_SETTINGS, "trainer.policy.optimizer_config.lr=2.0e-6")`;
   editing `_DEFAULT_SETTINGS` changes every preset that uses it.
2. Bump that preset's `revision`. Data-medium shares learning-medium's settings and revision.
3. From the repository root, print the actual native/allocation and full-envelope hashes:

   ```bash
   uv run pytest -s -q -n 0 experiments/post_training/snowball_tests/test_snowball_rl.py::test_gated_presets_match_revisioned_native_configs
   ```

   A new revision intentionally fails until its printed pair is added under that preset in
   the test's `revisions` table. Inspect the composed settings before accepting the pair.
   The full-envelope hash includes actual execution memory and priority.
4. Collect controls from a clean checkout of the final source. For example, replace `REVISION`
   with learning-medium's revision and run:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset learning-medium --version REVISION --control --replicates 4 --run --wait
   ```

   Pin that preset's source, revision and count in `snowball_rl_gate.CONTROLS`, preserving the
   other entries. Use the exact short `base_commit` recorded in the controls' result provenance
   (the collection checkout's `git rev-parse --short HEAD`), not a later pin-edit commit.
   Its entry has the form `"learning-medium": (ControlPin("learning-medium", "BASE_COMMIT", "REVISION", 4),)`.
5. Run `uv run python -m experiments.post_training.snowball_rl_gate --preset learning-medium --leave-out r0`
   for each replicate (`r0` through `r3` in this example). Every leave-one-out verdict must PASS.

## Tasks and evaluation

The [EvalLite](../references/task-curation.md#evallite) is the held-out set that
`speed-short`, the learning and data presets and `train-long` evaluate: 1,500 public benchmark
tasks graded on the training machines, and a 100-task fast subset. It shows whether a change moves
scores on tasks the model never trains on, measured against controls on the same panel; it is not a
headline benchmark. To change it, follow that link; a new panel version needs new controls. Named panels request one greedy answer per task, with natural replies
up to `EVAL_REPLY_TOKENS`. Engine defaults are FLASH_ATTN, prefix caching on and batch invariance off.

Evaluation identity compares the stored TaskSpec strings and the request: merged sampling,
answer count, reply/context budget, prompt length, chat template and non-`VLLM_*` task environment.
Optimizer and engine settings may vary. Engine settings, `VLLM_*` environment and vLLM version
are recorded as separate attempt facts.

Answers join the manifest by TaskSpec ID, with attempts and weight steps kept separate.
An answer counts as scored only with successful verification and no exception or error treatment.
Scores are normalized by their declared bounds. Equal-family mean averages scored answers per
task, tasks per family, then families. Slices cover family, AAII category and agentic status
when both kinds occur.

Per-update `timing/grading/{mean,p50,p95,max}` and `timing/setup/{mean,p50,p95,max}` are seconds
over kept, scored answers. Setup covers task/verifier startup and staging; grading excludes
verifier setup. Both exclude cleanup. Reports average each statistic over the measured updates.

Learning/data medium evaluates at update 0 and 25. Train-long evaluates before training and
every ten updates. Speed-short measures its primary throughput before its full-panel evaluation;
quality answers and timing stay outside that throughput window. Metrics and attempt receipts
live in FineStore tables under `exports/run-records/`; primary answers use `exports/dumped_evals/`.
Speed-short quality answers use `exports/eval_lite/dumped_evals/eval_only/`. The configured export
root identifies their actual locations.

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

## Gaps

| gap | implication and follow-up |
|---|---|
| layout-p40-only | calibrated training gates cover the 40-GPU layout; other layouts need qualification |
| mismatch-is-agreement | mismatch measures policy agreement; exactness claims need an explicit zero-mismatch bound |
| AAII coverage; no-agentic-tasks | scientific reasoning, coding, General and a single-turn Agents stand-in; multi-turn tools and long context need separate qualification |
| no-image-sandbox | image-only tasks need a sandbox backend |
