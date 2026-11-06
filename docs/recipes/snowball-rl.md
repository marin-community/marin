# Snowball RL experiments

Run a Snowball experiment from a named preset. The command binds the versioned
curation selection, pinned Datakit SFT weights and tokenizer, GPU layout,
evaluation schedule and measurement window into an artifact graph.

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
```

The command prints the graph and status. Add `--run` to submit its Iris
coordinator. Repeating the same command reuses completed artifacts. Source files
under `experiments/`, `lib/` and `vendor/` must be tracked or staged: the git tree
hash forms part of each run's identity. `--version dev` satisfies the standard
launcher interface; Snowball steps use the fixed `PRESET_REVISION` instead.
Single-answer panel evaluations use greedy sampling and request no rollout logprobs.
Evaluation sampling belongs to the preset revision; controls must match that revision.

Add `--run --wait` to poll every 60 seconds and print the final status and report
table. Its exit code is 0 for PASS or completed train-long/control collection,
1 for FAIL, 2 for INVALID and 3 when submission, the coordinator or missing evidence prevents a verdict. Successful ungated runs also exit 0. Submission without `--wait` returns 0 when accepted.
The status lists the latest completed update and the W&B URL when the tracker has
saved one. Set `WANDB_ENTITY` before submission to choose the W&B team; an unset
value uses `marin-community`. Runs use the `marin-snowball-rl` project.

The W&B team changes neither run identity nor pinned configuration. Grafana run and preset-panel links
appear when the native job start time is available; active jobs link through the present time.

| preset | allocation | workload | measurement |
|---|---|---|---|
| speed-short | 8 H100, east | 512 locally graded training tasks, four answers each, exactly 4,096 tokens; then the full greedy test panel | engine tokens/s/GPU and panel quality; approximately 15 minutes |
| data-short | 8 H100, east | 256 candidate training tasks, four answers, up to 8,192 tokens | fraction of tasks with both correct and incorrect answers |
| speed-medium | 40 H100, east | ten synchronous updates, exactly 4,096 tokens per answer | median update time over updates 2–10 and staleness-0 k3 mismatch |
| speed-long | 40 H100, east | 20 asynchronous updates, staleness 2 and TIS, replies up to 8,192 tokens | median update time and learner/engine tokens/s/GPU over updates 3–20; target 45–75 minutes |
| learning-medium | 40 H100, east | the 1,500-task greedy test panel at update 0 and after 25 synchronous RLOO-N updates | final panel equal-family mean; training metrics over updates 21–25 |
| data-medium | 40 H100, east | learning-medium on candidate tasks | same fixed panel and recipe as learning-medium |
| train-long | 112 H100, Reno; optional 40 H100, east | asynchronous training, default 100 updates | checkpoints, 100-task fast panel every 10 updates, HF export; no verdict |

East means `cw-us-east-02a`; Reno means `cw-rno2a`. The 40-H100 layout has 32
policy GPUs and 8 engine GPUs, 128 prompts×4 answers, a 24,576-token window and 8,192
reply tokens. The 112-H100 layout has 64 policy GPUs and 48 engine GPUs, 512×16,
a 65,536-token window and 32,768 reply tokens. Both use staleness 2 for train-long;
speed-medium, learning-medium and data-medium use staleness 0. East uses sockets; Reno uses InfiniBand.
Speed-long uses the asynchronous 40-H100 recipe without evaluation, checkpoints or
export. It guards throughput and staleness-0 k3 mismatch, and reports phase timings, grading/setup
timers, staleness summaries and selected-token counts by staleness bucket.

## Dashboard publication

Completed learning-medium, data-medium and train-long candidates publish one `rl_progress` record per replicate.
The status row includes `rl_progress_url` after publication is verified. Gated records keep the group
verdict; train-long has no quality verdict. Each uses its own update-0 baseline and keeps the native result.
Gated records below 99% scored coverage cannot rank. Train-long publishes only with full baseline and final coverage. Speed presets and data-short keep their results in the gate report.
For gated presets, re-run the standalone gate command to retry publication after a service failure;
the saved verdict remains available. For train-long, repeat the original command to reuse its completed
native run and retry publication. A resumed train-long run with no update-0 baseline keeps its native
result and attempt record without a dashboard row.
Every native attempt keeps `exports/attempts/<attempt_id>/loop-result.json` under its output,
including failed attempts. This records native status, preset, layout, configuration fingerprint and
available Iris/W&B identity. Runtime task-content and protocol hashes remain unknown, so these records
remain unranked.

## Change one input

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev
uv run python -m experiments.post_training.snowball_rl --preset data-short \
  --tasks snowball-100k@2026.11.04.1 --version dev
uv run python -m experiments.post_training.snowball_rl --preset train-long \
  --layout p40 --set trainer.max_steps=100 --version dev
```

`--set` takes a dotted SkyRL key and YAML value. Unknown keys need `+KEY=VALUE`.
The builder protects placement, data bindings, retention, evaluation callbacks
and measurement settings. Learning experiments can change batch shape, task
sampling, optimizer and schedule; the role plan receives batch-shape changes.
Data presets require `--tasks` and refuse recipe settings. Their immutable
selection must name the same heldout input as the default selection.

`--init NAME@VERSION` on learning-medium or train-long selects another model
artifact or a kept HF export. Learning-medium and data-medium evaluate the initial weights by default.
Learning-medium/data-medium controls keep their final HF weights automatically; candidates do so with
`--keep-weights`. Kept HF exports have no expiry and can initialize later runs with `--init`.
They live under the run artifact's durable `exports/global_step_<N>/policy` path, outside the
14-day temporary checkpoint root.
To evaluate a different heldout panel, score every pinned control on its immutable version:

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --control --rescore-panel PANEL_NAME@VERSION --version dev --run --wait
```

This submits one 8-H100 evaluation job per control and uses its recorded evaluation and engine settings.
Omit experiment overrides and `--replicates`. A learning control without kept weights is refused before
submission. Speed-short uses its recorded initializer. The CPU gate joins only matching completed panel
results; training and throughput measurements come from the original controls. Missing or mismatched
control, model, tokenizer, panel, file or protocol evidence gives INVALID. Re-running the CPU gate after
rescoring creates a report for the completed evidence. Speed-medium and speed-long have no panel to rescore.

Train-long retains checkpoints and exports weights by default. `--resume-from RUN` on train-long continues the
earlier train-long checkpoint root and optimizer state under a new run identity.
The same command retries its own latest checkpoint. Temporary checkpoints expire
after 14 days. Check checkpoint compatibility before resuming across changes to
SkyRL or Marin's RL runtime; see [checkpoint handling](../references/rl-launching.md).
Repeat the original `--init` when continuing a run with custom initial weights.

Use `--replicates N` for 1–5 copies of a gated preset, sharing a seed. Controls use
`--control --replicates N` from a clean checkout and fixed default inputs, with the
commit in their shared artifact names. Data-medium uses learning-medium's control.

## Compare with controls

Calibrated presets write `gate_report.json` and `gate_report.md`. Status shows
`pass`, `fail` or `invalid`, a change label and the report path. Train-long has no
verdict; a crashed run reports `fail` with its Iris job ID.

The gate reads exact runs pinned in `snowball_rl_gate.CONTROLS`. Refresh a pin after
its replicates finish. Pins with the same preset revision pool deviations from each
pin's own mean.
Pins must match the current `PRESET_REVISION`. A candidate with no matching pin returns
JSON INVALID with exit code 2 before GPU submission and prints the control collection
command. The test-panel revision requires a control refresh for every gated preset.
Speed-long's 20-update sizing run took 48.2 minutes with natural replies up to 8,192 tokens.

Guards compare candidate and control replicate means, signed so positive means better.
Tolerance is `t(1 - 0.025/k, df) * sd * sqrt(1/n_run + 1/n_control)`, with `k` guards
and calibrated run-to-run spread `sd`. Each run reduces update time to its median;
other metrics use their mean. A guard fails below negative tolerance. Guards and declared
reported metrics determine `improves`, `trade-off`, `regression` or `no change`.
Declared reported metrics use `k=1`. Other breakdown rows use a separate `k` equal
to each table’s row count; they do not determine the label, and a row whose controls
never vary reports its change as unavailable.

Add repeatable requirements to a gated candidate with `--require 'KEY<=NUMBER'`,
`--require 'KEY>=NUMBER'`, `--require 'KEY<=control'` or `--require 'KEY>=control'`.
For example, `--require 'policy/mismatch/staleness0/log_ratio_abs_max<=0'` requires
zero mismatch on every replicate; `--require 'test_panel/aaii_category/coding/mean>=control'`
adds a calibrated guard. Numeric bounds produce per-replicate invariants. Relative
bounds count with the preset guards; zero control spread requires a numeric bound.
Launch preflight computes control r0's comparison values and refuses any key outside
them, such as a mistyped panel slice. Candidate keys must exist at scoring.
Train-long and control collection refuse requirements. Python callers pass the same
strings as `build_run(requirements=(...))`. Requirements change the report identity;
training names and configs stay fixed. Re-scoring appends `--require` to any recorded
requirements, so completed runs can receive new constraints on CPU.

Add repeatable `--baseline NAME@VERSION` arguments to compare against earlier
finished runs of the same preset and `PRESET_REVISION`. The **Vs baseline** table
uses their mean and replicate count, with the pinned controls' pooled spread and
degrees of freedom. It applies the same guard and reporting tolerances as the
control table. The verdict and change label continue to use the controls.
Repeated references and aliases to the same saved run count once. Baselines must
have matching panel task IDs and valid completed batches. Baselines change the gate identity and leave the training run
identity fixed. The CPU re-score command accepts additional baselines and recovers
those recorded at launch. A baseline that becomes unavailable leaves the control verdict intact;
`vs_baseline` is empty and `vs_baseline_reason` explains why. JSON comparison rows use
`control`, `n_control`, `control_min` and `control_max` for the baseline statistics.
Timer comparisons are unavailable until the test-panel control refresh records timers.

The Breakdown tables show overall results, agentic results when the panel has both
agentic and other tasks, family, AAII category,
difficulty, category × difficulty, family × difficulty and training reward per source.
Each populated mean slice includes its scored task count, initializer normalized score,
control mean and replicate range, candidate mean, signed difference, tolerance and change.
Guard keys appear once; one-answer panels report neither mixed nor all-wrong fractions.
Speed-medium also reports learner and engine tokens per configured GPU-second.
Learning/data-medium reports rejected groups, length stops and reply-length spread;
training reward per source uses keys recorded by every control. Telemetry detail rows
do not determine the change label. Initializer values average candidate replicates;
task counts are unavailable when their scored-task counts differ.

Every planned batch/update must finish, losses and gradients must be finite, panel
coverage must reach 99%, grader infrastructure failures must be at most 1%, and speed-short/speed-medium
throughput replies must have the fixed length. Mismatched pins, preset revisions or panel IDs,
missing receipts/provenance/guards, missing mismatch tokens or zero control variation in a calibrated guard
give `invalid` with a reason.

The report bounds overlapping failure counts: upper ≤ 1% passes; lower > 1% fails.
Each panel uses its own evaluation session's counters. Missing or inconsistent panel
counters leave an unknown 0–100% bound.
A candidate straddling 1% is `invalid` if every other check passes, otherwise `fail`.
An ambiguous control is `invalid`. Both bounds remain in the report.

Medium learning and data reports include update-0 scores and final-minus-initial scores in their
Breakdown tables. Their comparisons require matching update-0 measurements from the controls.

Re-score saved runs on CPU:

```bash
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset learning-medium --candidate RUN_NAME[@VERSION]
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset speed-medium --leave-out r0
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset learning-medium --candidate RUN_NAME --baseline EARLIER_RUN@2026.10.27
```

Reports use the caller's namespace under `MARIN_PREFIX`; controls are read-only.
Host status lists changed RL paths, including uncommitted edits, against the control
commit, or explains missing git metadata. This list does not affect the verdict.

The test-panel recipe pins its qualified eight-answer, temperature-1 initializer pass-rate
artifact. Reports check its task IDs against the panel manifest, then join easy (> 0.7),
learnable (0.1–0.7) and hard (< 0.1) bands and reference initializer scores for reporting only.
Refresh that reference when the initializer revision changes. The standalone
`snowball_pass_rates` command measures the 910-task development panel.

## Tasks and evaluation

The pinned `snowball-100k@2026.11.04.1` selection contains 100,000 training tasks
with local graders. Selections containing AI-judged tasks are refused before GPU submission.
Its held-out development rows and the test panel are excluded from training.
Create candidates through task curation's [final task selection](../references/task-curation.md#final-task-selection),
using admitted sources from a FULL campaign and immutable recipe versions. The manifest provides
task IDs, family, AAII category, agentic flag and grader kind.

The default evaluation artifact, `snowball-test-panel@2026.11.06.1`, contains 1,500 full
tasks and a stratified 100-task fast subset. Its recipe records source parts and filters;
its manifest records family, AAII category, agentic flag, grader kind, source, license and
difficulty. It contains no tasks from the 910-task development panel. `PanelEvaluator`
supplies native evaluation callbacks, sampling settings and versioned validation data sources.

Named panels use one seeded greedy answer per task, with natural replies up to
`EVAL_REPLY_TOKENS` (8,192). Training keeps its own sampling. Learning-medium,
data-medium and train-long use the deterministic engine profile: batch-invariant arithmetic,
TRITON_ATTN with the pinned attention-helper hash, and prefix caching disabled.
Speed-short, speed-medium, speed-long and data-short use the throughput profile:
FLASH_ATTN, prefix caching enabled and batch invariance disabled. Speed-short's quality
pass uses its throughput profile. Controls and candidates use the same profile.

Reports join evaluation answers to the run's task manifest by TaskSpec ID and keep attempts and weight steps separate. An answer counts as scored only with successful verification and no exception or error treatment. Verifier scores are normalized by their declared bounds. The equal-family mean first averages each task's scored answers, then tasks within a family, then families. Scored coverage uses the declared task count of the identity-bound panel version.

Each update records `timing/grading/{mean,p50,p95,max}` and
`timing/setup/{mean,p50,p95,max}` in seconds over its retained, scored TaskSpec answers.
Setup includes task and verifier startup, package environments and verifier staging.
Answers without a task or verifier machine contribute zero setup time.
Grading measures wall time excluding verifier setup; both timers exclude cleanup.
The gate reports the mean of each statistic over the measured updates, so its p95 row is
the average of the update p95 values.

Learning/data medium evaluates `test_panel` at update 0 and after update 25.
Train-long evaluates `test_panel_100` before training and every 10 updates.
Speed-short measures throughput and its grading/setup timers before running the full
`test_panel` in the same engine job. Quality answers and timers do not enter the primary
throughput measurement. Its quality guard is `test_panel/family_mean`, with independent
scored-coverage and grader-error invariants.
Metrics are in `exports/metrics/step-*.json`; per-answer JSONL is under
`exports/dumped_evals/`; speed-short's quality answers are under
`exports/test_panel/dumped_evals/eval_only/`. The canonical `terminal.json` records the result and Iris
job ID. Inspect both the artifacts and job before treating a run as successful.

The `Evaluator` interface supplies configuration, callbacks and required environment names for presets. Harbor-on-Daytona support is a follow-up; internal panels are the current
implementation. The default selection and panels need no remote judge credential.
Each run forwards its required keys through Iris.

## Which experiment to run

| change | presets |
|---|---|
| vLLM wheel, kernel or engine option | speed-short, then speed-medium |
| trainer kernel, parallelism, recompute or exact numerics | speed-medium, then learning-medium |
| weight synchronization | speed-medium |
| fp8 experts or KV-cache precision | speed-medium, then learning-medium; a person assesses the trade-off |
| loss, estimator, optimizer, schedule, reward or task sampling, batch shape | learning-medium |
| task selection | data-short, then data-medium |
| SFT initializer | learning-medium with `--init` |
| startup latency | speed-medium, which reports startup |
| asynchronous loop, grading or admission throughput | speed-long |
| release training | train-long |

## Gaps

| gap | implication and follow-up |
|---|---|
| layout-p40-only | calibrated training gates cover the 40-GPU layout; other training layouts need qualification |
| AAII coverage; no-agentic-tasks | the panel covers scientific reasoning, coding, General and a single-turn Agents stand-in; multi-turn tool use and long-context coverage need separate qualification |
| no-image-sandbox | image-only tasks need the planned sandbox backend |
| no-async-speed-gate | Speed-long has no pinned controls; collect controls before using its throughput verdict. |
