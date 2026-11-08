# Long-loop prototype

Local implementation in progress: Snowball SFT → async RLOO-N (GS8) → Micro210.
The quick-loop recipe is unchanged. No cluster jobs have been submitted.

`graph.py` binds stage builders through the existing Marin artifact runner. Local
integration tests exercise real artifact execution, predecessor propagation,
committed-cache reuse, and failure blocking. This is not a second scheduler.
`preflight.py` checks measured endpoint and learner projection evidence, including
the dual-EOS and missing-earlier-actions regressions. Evidence collectors still
need connecting to the real serving and learner adapters.

`timing.py` writes durable, per-writer JSONL and renders paired durations, overlap
unions, allocation GPU-hours, failures and incomplete spans. Reports label synthetic
inputs and do not equate work sums with elapsed wall time. Native phase adapters,
full timing coverage, admitted-round boundaries and allocation enforcement remain
unfinished. GPU peak reporting alone is not allocation enforcement.

Inspect the plan without cluster access:

```bash
uv run python -m experiments.post_training.long_loop --profile plumbing --version 2026.11.08
```

Render events:

```bash
uv run python -m experiments.post_training.long_loop --version 2026.11.08 \
  --events /path/to/events.jsonl --report /path/to/report.md --synthetic
```

Omit `--synthetic` only for real measurements. Each worker writes its own file;
pass multiple files to `--events`. Preserve source files and unfinished spans.

## Before a cluster smoke

`--run` deliberately fails before submission. A runnable smoke command cannot yet
be provided: main lacks the completed general-model Micro210 runtime, and the
SFT/handoff source pins and real stage bindings have not been wired here.

The concrete plumbing RL builder in `rl.py` uses the maintained quick-loop
recipe at GS8 and staleness 2 on 40 H100s, interactive priority on RNO2A.
Its model is the supplied preceding handoff. Topology validation passes locally.
Next connect concrete SFT, frozen-router conversion and Micro210 builders. Each must forward the predecessor checkpoint, verify committed payloads,
persist submission identity before launch, reconcile ambiguous attempts without
resubmission, and prove allocation release before advancing. Bind real endpoint
and all-turn projection probes. Add native timing adapters rather than replacing
them with synthetic timers. Then review the exact pinned configuration and command
with the coordinator before any cluster submission.

The proposed plumbing workload remains 15 SFT updates (~1.007B positions), three
async RL updates (64 groups × GS8, staleness 2, 24,576 context), then 21 Micro210
paths. Peak budget: 256 H100 including transition overlap and judge; research
priority. SFT steady work projects to 22 minutes; startup, conversion, RL and eval
must be measured before giving a defensible total ETA. Proposed job prefix:
`long-loop-plumbing-<run-version>`.

All implementation stays in this worktree. Fetch and merge origin/main regularly.
No pull request may be opened without explicit user approval. The aggregate
allocation limit is 256 GPUs; priority is interactive and may not be changed
without explicit user approval.
