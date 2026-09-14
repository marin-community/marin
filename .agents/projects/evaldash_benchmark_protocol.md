# Benchmark protocol in the eval record

Closes the two open #8083 issues that both hinge on one missing piece of the
`record.json` contract: the benchmark's declared protocol.

- #8148: a one-item GSM8K canary is admitted to the leaderboard at 100%,
  because coverage today only measures `n_scored / n_attempted` and the
  record never says how many items the benchmark has.
- #8144: the headline metric comes from a global priority list in
  `archive.PRIMARY_METRIC_PRIORITY` that omits `f1`, so DROP publishes exact
  match, and every headline is classified binary or continuous by name alone.

Both are fixed by declaring the protocol where the eval is defined, writing
it into the record, and having the stats engine read the record instead of
guessing.

Reviewed by ibis on 2026-09-14; the revisions from that review are folded in
below (per-leaf item counts, intended attempted count, Harbor metric name,
metric comparability, legacy capped runs, group inheritance).

## Status of the rest of the megathread (2026-09-14)

| Issue | State |
|---|---|
| #8145, #8142 | closed by #9104 |
| #8212 | #9105 open, green, review required |
| #8147 | #9109 open, green, review required |
| #8143 | #9112 open, green, review required |
| #9070 | #9113 approved, green; #9128 co-hosted judge open |
| #8150 | #8883 approved, stale `agentic-lint` failure, no labels |
| #8148, #8144 | this plan |
| #8215 | unclaimed; separate week-plus feature, not covered here |

## Non-goals

- Pipeline parallelism (#8215). Different subsystem, own plan.
- A `pass@k` metric kind. `pass@1` is a per-item 0/1 mean and stays binary;
  `pass@k` for k > 1 is a per-item continuous score and uses the harness
  standard error. No new interval math is needed for either.
- Multi-headline benchmarks. One declared primary metric per task. Every
  other metric stays in `tasks[].metrics` and in the run detail view.
- Rewriting old records. Records without the new fields still parse. The
  reader's behavior for them is spelled out below.
- Widening intervals for an unknown benchmark size. Admission handles it; a
  flag alone does not change an interval and should not pretend to.

## Design

### 1. Declare the protocol in the eval YAML

`EvalchemyTaskOptions` (`lib/marin/src/marin/evaluation/evalchemy/config.py`)
gains three fields:

```yaml
# experiments/evaluation/configs/evalchemy/drop.yaml
tasks: [drop]
task_options:
  drop:
    num_fewshot: 3
    task_alias: drop_3shot
    generation: true
    primary_metric: f1          # base name; the lm-eval filter is still
                                # chosen by FILTER_PRIORITY at read time
    metric_kind: continuous     # binary | continuous
    # expected_items: 9536      # only for tasks lm-eval does not size
max_tokens: 256
```

Rules, enforced at launch by `evalchemy_run_config` in
`experiments/evaluation/evals.py`:

- `primary_metric` is required for every task of every registry eval. A
  file-backed eval without it fails validation with a message naming the
  task. No silent fallback to the priority list for new runs.
- `metric_kind` defaults from `eval_stats.BINARY_METRICS` membership
  (`acc`, `acc_norm`, `exact_match`, `accuracy`, and add `pass@1`).
  A `primary_metric` outside that set with no `metric_kind` is a validation
  error. An explicit `metric_kind` always wins.
- `expected_items` is the benchmark item count for a task whose harness
  output does not size it (Evalchemy's chat-native benchmarks such as
  `aime24`, `math500`, `olympiadbench`, `humanevalplus`, `mbppplus`; see
  section 2). Required for those tasks, forbidden where lm-eval sizes the
  task, so the two sources can never disagree.
- All 23 registry YAMLs get the declaration in the same PR. The values are
  the ones the priority list picks today, except `drop` (`f1`,
  continuous). Any other mismatch found while filling them in is called out
  in the PR body as a behavior change.
- A declaration on a group task (`mmlu`) applies to every leaf lm-eval
  expands it into. Leaves are not declared separately.

Harbor needs no YAML change. Its aggregate keys are `accuracy`,
`mean_reward`, `solved`, `total`; `reward` exists only per sample.
`HarborDefinition.record_ref_for` writes `primary_metric="accuracy"`,
`metric_kind=binary`, which is what the panel selects today.

### 2. Write it into the record

`EvalTaskRef` (`lib/marin/src/marin/evaluation/records.py`) gains:

```python
primary_metric: str | None = None   # base metric name
metric_kind: MetricKind | None = None
```

`TaskCoverage` gains:

```python
n_benchmark: int | None = None
```

Coverage entries are keyed by leaf task, as today. For each leaf:

- `n_benchmark` is the leaf's item count before any cap.
  - lm-eval leaves: `results.json` carries
    `n-samples: {<leaf>: {original, effective}}` at the pinned Evalchemy
    fork (`f4299045`, lm-eval v0.4.12; `original = len(task.eval_docs)`).
    `original` is the source. It is per leaf; groups have no entry.
  - Chat-native Evalchemy benchmarks and any leaf with no `n-samples`
    entry: the YAML's `expected_items`. Export fails if neither exists,
    which the launch validation in section 1 already prevents.
  - Harbor: the dataset's task count before `task_limit`, which preflight
    already enumerates.
- `n_attempted` becomes the **intended** post-cap count:
  `min(cap, n_benchmark)`, where `cap` is `max_eval_instances` or
  `task_limit`, or `n_benchmark` when there is no cap. It is no longer
  recovered from `max(doc_id) + 1`. That recovery understates the intended
  count after an early stop (intended 100, rows for 0..89, 82 graded:
  90/100 and 82/90 both pass a 90% gate while the real execution coverage
  82/100 fails). The document extent is kept only as a consistency check:
  the export raises if the extent exceeds the intended count.
- `n_scored`, `n_correct`, `n_unanswered`, `errors` are unchanged. With
  #9109, scored agent and passthrough errors stay score-bearing; the
  unscored count is `n_attempted - n_scored`, never a sum over `errors`.
- A leaf lm-eval lists under the group but wrote no rows still gets a
  coverage entry with its `n_benchmark`, `n_attempted`, and `n_scored = 0`,
  so a missing leaf shrinks execution coverage instead of vanishing from
  the denominator.

Export-time validation: `n_benchmark > 0`, `0 <= n_scored <= n_attempted
<= n_benchmark`. A violation raises; a record with inconsistent counts is
never written.

### 3. Read it in the stats engine

`Coverage` (`eval_stats.py`) gains `n_benchmark: int | None` with the same
None-propagation as `n_attempted` in `_mechanism_coverage`: any leaf with an
unknown count makes the benchmark's count unknown. Leaf counts are summed
once per leaf name. Add:

```python
@property
def benchmark_rate(self) -> float | None:
    """n_attempted / n_benchmark, or None when either is unreported."""
```

`measurement_from_record` (`eval_measurements.py`):

- Metric key: among the leaf's metric keys whose `base_metric` equals the
  declared `primary_metric`, choose by `FILTER_PRIORITY`, else the
  alphabetically first. If a leaf that has rows lacks the declared metric,
  the record yields no measurement and the dashboard's gap reason is
  `declared metric <name> not in results`.
- Kind: the declared `metric_kind`. The existing rollup rule that demotes a
  non-integral binary mean to continuous stays.
- `Measurement` gains `declared: bool`. Records with no declaration keep
  the current path (`primary_metric()` from the priority list, kind from
  `BINARY_METRICS`) and are marked `declared=False`. That is the only
  remaining headline use of the priority list.
- Inconsistent counts read from an old record (`n_attempted > n_benchmark`,
  `n_scored > n_attempted`, non-positive `n_benchmark`) set a new
  `ResultFlag.INCONSISTENT_COVERAGE`, which is in `DEFAULT_EXCLUDE_FLAGS`.

**Metric comparability.** A column must rank and subtract one metric. Old
DROP records headline `exact_match`; new ones headline `f1`. The engine
therefore learns each benchmark's protocol from the records themselves:

```python
def declared_protocols(measurements) -> Mapping[str, Protocol]:
    """Per benchmark, the (base metric, kind) of the newest declared record."""
```

`select` takes that mapping. A measurement whose `(base_metric, kind)`
differs from its benchmark's protocol is rejected with reason
`metric exact_match differs from declared f1`. A benchmark with no declared
record at all has no protocol and admits as today. `difference_interval`
asserts both measurements share metric and kind. In practice this empties
the old DROP cells until DROP is re-run under the declaration, which is the
correct outcome: those cells were the wrong number.

**Admission.** `SelectionRequest` gains
`min_benchmark_coverage: float = DEFAULT_MIN_COVERAGE`, and
`_admission_reason` adds, after the existing execution-coverage check:

```python
benchmark_rate = measurement.coverage.benchmark_rate
if benchmark_rate is not None and benchmark_rate < request.min_benchmark_coverage:
    return f"benchmark coverage {benchmark_rate:.3f} below {request.min_benchmark_coverage:.2f}"
if benchmark_rate is None and measurement.item_cap is not None:
    return "capped run with unreported benchmark size"
```

An uncapped run with an unreported benchmark size is admitted, as today. A
capped run is admitted only when the record proves the cap still covered
90% of the benchmark. `min_benchmark_coverage=0` admits a capped run whose
size is known. Legacy capped runs, written before `n_benchmark` existed,
stay detail-only: they never reach the panel under any request. That is the
decision, not an oversight; the only such runs are smokes and canaries.

### 4. Surface it in Evaldash

`infra/marina/apps/evaldash/metrics.py`:

- `panel_request` parses `min_benchmark_coverage` from the query string and
  echoes it in `request`.
- `cell_payload` adds `metric_kind`, `declared`, `n_benchmark`, and
  `benchmark_rate`.
- The run detail's coverage block shows both denominators:
  `attempted 1 of 1319 · graded 1 of 1`.
- `build_panel` reports each column's protocol beside `benchmarks`, so the
  SPA labels the header from the protocol rather than from whichever cells
  happen to be selected.

`infra/marina/apps/evaldash/samples.py` picks a primary metric on its own
today. It reads the record's declaration instead, so the sample browser's
default `correct` column agrees with the headline. The exporter
(`lm_eval_samples._correct`, `_lm_eval_grading`) takes the declared metric
from the run config the runner already holds.

`PanelPage.vue`: show the protocol's metric name under the column header,
which the family picker from #9112 already positions for. Rejected cells
show the new reasons through the existing rejection tooltip; no new UI.

## Tests

Behavioral, per `TESTING.md`:

- `evals.py` validation: a YAML with no `primary_metric` fails naming the
  task; `f1` with no `metric_kind` fails; explicit `metric_kind` overrides;
  `expected_items` on an lm-eval-sized task fails; a chat-native task
  without it fails.
- Evalchemy export: a fixture `results.json` with `n-samples` yields
  `n_benchmark` per leaf and `n_attempted = min(cap, original)`; a group
  with one leaf absent from the samples still lists that leaf with
  `n_scored = 0`; a document extent above the intended count raises.
- Harbor runner: `n_benchmark` equals the dataset task count, `n_attempted`
  equals `min(task_limit, count)`, and the headline is `accuracy`.
- `eval_measurements`: a record with both `exact_match,none` and `f1,none`
  and a declared `f1` headline selects `f1`, pairs `f1_stderr,none`, and is
  continuous. The same record without a declaration selects exact match
  and is `declared=False`. A declared metric absent from the results yields
  no measurement. Inconsistent counts set `INCONSISTENT_COVERAGE`.
- `eval_stats.select`: `n_attempted=1, n_benchmark=1319` is rejected with
  the benchmark-coverage reason; `n_attempted=1300` is admitted; a capped
  run with `n_benchmark=None` is rejected even at
  `min_benchmark_coverage=0`; an uncapped run with `n_benchmark=None` is
  admitted; an old `exact_match` DROP cell is rejected once a declared `f1`
  record exists for DROP; `difference_interval` refuses mixed metrics.
- Evaldash `/panel`: the GSM8K canary from #8148 (`max_eval_instances: 1`)
  no longer displaces a full run; the rejection reason and the column
  protocol are in the payload. `samples.py` defaults to the declared metric.

## Implementation order

One branch, commits in this order so each is reviewable alone:

1. Record and config fields, YAML declarations, launch validation.
2. Producers: Evalchemy export and Harbor runner write `n_benchmark`, the
   intended `n_attempted`, and the metric declaration.
3. Stats engine, measurement reader, protocol comparability.
4. Evaldash API, `samples.py`, and SPA.

Commit locally only. Do not push until the owner says so. When the owner
says to push, open a **draft** PR.

Required checks before handoff: `./infra/pre-commit.py --changed-files --fix`,
`uv run --no-project infra/ci/run_tests.py`, and the Evaldash frontend build.
