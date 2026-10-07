# Task curation

This experiment turns pinned dataset inputs into reviewed tasks, verification
reports and accepted Parquet outputs. GLM reviews content quality; source graders
check executable behavior. A completed procedure can still retain unsupported
or failed grading contracts.

## Current checkpoint

[GOAL.md](../../../GOAL.md) records current status and proposed cleanup for
[PR #9798](https://github.com/marin-community/marin/pull/9798). Full processing has
not launched. R11 samples all 153 sources using 50 concurrent source procedures
and 256 shared Zephyr workers. At 2026-10-07 23:30:39 UTC, Iris confirmed the
driver, coordinator and shared worker job were RUNNING. The latest observed
campaign report, timestamped 23:16:57, contained 50 active sources, 103 queued
and no completed source outcomes. Follow the
[R11 job](https://iris-cw-rno2a.oa.dev/#/job/%2Fpower%2Ftask-curation-shared-sample-r11-original-20261007)
and the report links in GOAL.md for subsequent results.

## Dataset ownership

[sources.py](sources.py) declares the 153 available Atlas sources.
Each declaration module in [datasets/](datasets/) owns its pinned inputs, recipe and grader
binding and exposes `pipeline()`. Binding selects the grader declaration and runtime
requirements during ingestion; the grading runtime reads the serialized TaskSpec.
Shared binding helpers live in
[datasets/shared.py](datasets/shared.py); reusable task implementations live in
[TaskCompendium](../../../lib/taskcompendium/src/taskcompendium/pipeline/README.md).
Declarations are grouped under `skyrl`, `tasktrove` and
`nemotron_ultra/{mopd,rlvr1,rlvr2}`, with short filenames for source variants.

TaskCompendium owns the conversion and review `TaskPolicy`, created by library
`policy()` factories, and the reusable `run_source_pipeline` procedure. Experiments
owns source pins and ArtifactStep bindings: names, versions, dependency handles,
output locations and resources. Only experiment dataset declarations expose
`pipeline()`. [Nemotron inputs](datasets/nemotron_ultra/inputs.py) contains acquisition and
reference-file plumbing shared by the Nemotron declarations.

[sources.py](sources.py) also reads reporting metadata for 203 Atlas entries,
including 50 exclusions. The declaration's `source_key` is the same canonical
name used by the runtime manifest and staged-input maps. `hf_id` names its
upstream Hub repository. Atlas lineage and
current upstream revision remain separate from the recipe's implemented pin.
Excluded entries have no executable catalog builders.

```python
from experiments.post_training.task_curation.sources import rl_data_pipelines

source = rl_data_pipelines()["MarinSkyRL:math500"]
step = source.bind(source_config, source_runtime)
```

`source_config` is a `SourcePipelineConfig`. `source_runtime` is a
`SourceRuntimeConfig` containing explicit grader images, backends, controller URL
and optional staged inputs. Graph construction performs no downloads, inference
or job submission. Executable sources require compatible immutable image pins.

## Sample, review, process and verify

Sample mode draws at most 100 raw records per source before conversion. It can
still scan all selected input bytes. Preparation normalizes the panel, preserves
raw provenance and records conversion failures. GLM reviews eligible tasks against
the source rubric. Source quality uses the raw draw as its denominator; missing
reviews, unsupported conversions and duplicates provide no quality verdict.

Full mode expands sources admitted by a matching terminal sample campaign.
It reuses panel observations and applies the source quality gate before processing
the remainder. Filtering retains an audit view and a separate accepted-task view.
A matching terminal sample with zero KEEP outputs, unsupported native controls,
inconclusive verification and no infrastructure errors selects normalize-only full
processing. Every raw record is normalized and retained in audit sidecars. Sampled
quality judgments are preserved; eligible unreviewed rows defer with
`readiness:unbound_controls`, without further model requests. Missing passing
witnesses alone and samples with any KEEP outputs retain ordinary full processing.

Verification samples accepted tasks and runs available controls twice. Missing
passing witnesses are skipped; unsupported runtimes and grader failures remain
explicit. Content acceptance does not certify executable grading.

Source grader files and scoring contracts retain their pinned upstream behavior.
The pipeline does not repair source tasks, graders, schemas or solutions to make
controls pass. Read verification decisions and readiness fields before training
with an output.

## Shared campaign pool

Run [driver.py](driver.py) inside one Iris driver job. Declare
`pip_packages=["./lib/taskcompendium[pipeline]"]` in its Iris `EnvironmentSpec`
so the coordinator and child workers receive the same dependencies. The campaign
retains one Zephyr context and worker pool across source procedures. Worker count,
coordinator memory and output shard count are explicit. `--concurrent-sources`
sets the admission limit and accepts values of at least ten.

This command builds the sample plan:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.driver \
  --runtime-manifest runtime.json --staged-inputs inputs.json \
  --review-transport direct-chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --max-workers 64 --coordinator-memory 16g \
  --concurrent-sources 10 --normalized-shards 32 \
  --worker-image REGISTRY/WORKER@sha256:DIGEST \
  --mode sample --report-path CAMPAIGN_PREFIX/sample.json
```

Add `--run --base-url URL` to execute with `GLM_BULK_TOKEN` in the driver
environment. For full execution, use `--mode full`, a new `--report-path`, and
`--sample-report CAMPAIGN_PREFIX/sample.json`. The sample report must be terminal
and match the source graph and worker image. Only sample procedures recorded as
`sampled` or `completed` are admitted; other outcomes remain in the full report.
Repeat `--source CANONICAL_NAME` to run a subset in catalog order. The sample
report for full execution must cover that same subset.

## Inference reuse and outputs

Keep `--review-cache` stable across campaigns. Expensive inference is cached by
the complete request: model, messages, generation options, request IDs and declared
model revision. Raw completions retain that identity. Cache misses issue fresh
inference requests. Review retries default to three attempts per run; failed
inference remains unavailable review evidence. An artifact or worker-image change
can rebuild processing while reusing an exact
request's completion. Changed tasks, rubrics or request options require fresh
inference when their request identity changes.

The inference cache is best effort. Unreadable storage or invalid responses
become misses, and changed artifact identities can rebuild outputs.
If a provider response is lost before caching, a retry can repeat inference.

`--recorded-review-bundles` can adopt hash-validated review evidence. Only complete
TaskSpec/rubric matches supply judgments; unmatched tasks use the configured
reviewer. Recorded evidence retains its original provenance and does not certify
grading or replace the exact request cache.

Each source artifact retains `hf`, `normalized`, `analysis`, `verification`,
`accepted`, `report.json` and `telemetry.json`. Sidecars join source locators,
raw/decoded hashes, decisions and verification evidence. Telemetry records phase
wall times and completed Zephyr execution IDs; it does not certify source quality.

See the [task-curation reference](../../../docs/references/task-curation.md) for
runtime manifests, staged inputs, quality thresholds and retry accounting, and the
[TaskCompendium pipeline contract](../../../lib/taskcompendium/src/taskcompendium/pipeline/README.md)
for review evidence and output schemas.
