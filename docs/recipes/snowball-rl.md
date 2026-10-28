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

| preset | allocation | workload | measurement |
|---|---|---|---|
| speed-short | 8 H100, east | 512 locally graded training tasks, four answers each, exactly 4,096 tokens | engine tokens/s/GPU; approximately 10 minutes |
| data-short | 8 H100, east | 256 candidate training tasks, four answers, up to 8,192 tokens | fraction of tasks with both correct and incorrect answers |
| speed-medium | 40 H100, east | ten synchronous updates, exactly 4,096 tokens per answer | median update time over updates 2–10 and staleness-0 k3 mismatch |
| learning-medium | 40 H100, east | 25 synchronous RLOO-N updates | final panel equal-family mean; training metrics over updates 21–25 |
| data-medium | 40 H100, east | learning-medium on candidate tasks | same fixed panel and recipe as learning-medium |
| train-long | 112 H100, Reno; optional40 H100, east | asynchronous training, default 100 updates | checkpoints, in-run panel and final 910-task evaluation; no verdict |

East means `cw-us-east-02a`; Reno means `cw-rno2a`. The 40-H100 layout has 32
policy GPUs and 8 engine GPUs, 128 prompts×4 answers, a 24,576-token window and 8192
reply tokens. The 112-H100 layout has 64 policy GPUs and 48 engine GPUs, 512×16,
a 65,536-token window and 32,768 reply tokens. Both use staleness 2 for train-long;
medium runs use staleness 0. East uses sockets; Reno uses InfiniBand.

## Change one input

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev
uv run python -m experiments.post_training.snowball_rl --preset data-short \
  --tasks snowball-100k@2026.10.26 --version dev
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
artifact or a completed train-long export. It also evaluates the initial weights.
Only train-long exports HF weights. `--resume-from RUN` on train-long continues the
earlier train-long checkpoint root and optimizer state under a new run identity.
The same command retries its own latest checkpoint. Temporary checkpoints expire
after 14 days. Check checkpoint compatibility before resuming across changes to
SkyRL or Marin's RL runtime; see [checkpoint handling](../references/rl-launching.md).
Repeat the original `--init` when continuing a run with custom initial weights.

Use `--replicates N` for 1–5 copies of a short or medium preset, sharing a seed. Controls use
`--control --replicates N` from a clean checkout and fixed default inputs, with the
commit in their shared artifact names. Data-medium uses learning-medium's control.
Calibration and comparison verdicts are pending.

## Tasks and evaluation

The pinned `snowball-100k@2026.10.26` selection contains 100,000 training tasks,
100 validation tasks and 910 full-validation tasks. Its local panel graders share
the training package environments. All 910 heldout tasks are excluded from training.
Create candidates through [task curation](../references/task-curation.md), using
the admitted FULL corpus and immutable recipe versions. The manifest provides
task IDs, family, AAII category, agentic flag and grader kind.

`PanelEvaluator` reads the run's own evaluation dumps, joins metadata by TaskSpec
ID and keeps attempts and weight steps separate. The equal-family mean first
averages each task's scored answers, then tasks within a family, then families.
It reports scored coverage separately.

Learning/data medium evaluates all 910 tasks at the end; `--init` adds
update 0. Train-long evaluates the fixed 100-task panel before training and every 10 updates,
then evaluates all 910 tasks on the final HF export in a separate engine-only step.
Metrics are in `exports/metrics/step-*.json`; per-answer JSONL is under
`exports/dumped_evals/`. The canonical `terminal.json` records the result and Iris
job ID. Inspect both the artifacts and job before treating a run as successful.

The `Evaluator` interface separates configuration, callbacks and score rows from
presets. Harbor-on-Daytona support is a follow-up; internal panels are the current
implementation. Training tasks with remote judges require `TOGETHER_API_KEY` in
the submitting environment; every run forwards its required keys through Iris.

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
| release training | train-long |

## Gaps

| gap | implication and follow-up |
|---|---|
| remote judge | Together grades 6,653 training tasks; replace it with a local judge in Phase 1.5 |
| layout-p40-only; no-async-speed-gate | medium gates cover the 40-GPU synchronous layout; add async speed-long in Phase 2 |
| AAII coverage; no-agentic-tasks | current tasks cover only part of AAII; add missing task families and agentic tasks |
| reused-dev-panel | use this panel for iteration; use independent benchmarks for release claims |
| no-image-sandbox | image-only tasks need the planned sandbox backend |
| no-grading-timer | generation includes grader tails; add separate grader timings |
| eval-uses-training-sampling | Panel answers use the training temperature because the engine applies `generator.sampling_params` to native requests. Separate evaluation sampling belongs with the Phase 1.5 panel and new controls. |
| mismatch-is-agreement | agreement can miss shared errors; learning remains a separate check |
| harbor-on-daytona | the Harbor fallback and its evidence run are a follow-up PR |

Speed-short measures engines only. A speed claim also needs speed-medium, whose
mismatch guard compares engine and trainer log-probabilities on trained tokens.
