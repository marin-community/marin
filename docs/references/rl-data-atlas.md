# RL Data Atlas

[Open RL Data Atlas](https://public.applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
to browse the saved task-curation source inventory without signing in. Its UUID
is `fb11c931-5861-4878-8bb5-a964d652b45f`; the stable link opens the current release.

The source definitions in `experiments/post_training/task_curation/` own dataset
metadata, source reviews, and conversion pipelines. The applet build exports
those definitions to `dist/catalog.json`. Publishing reconciles that complete
inventory with the applet's saved sources. Page refreshes use the packaged
catalog; they do not query Hugging Face or GitHub. Updating a dataset revision,
count, classification, or source definition requires rebuilding and publishing
the applet.

Search and filter the table, click column headings to sort, and use the information
button beside a source name to inspect counts, classifications, and evidence links.
**Show deprecated / excluded** reveals sources with no released tasks. Export view
downloads the filtered rows as CSV. The MarinSkyRL and Task Trove tabs identify
source origins; their populations can overlap.

Stable source IDs connect each registry entry to its saved reviews. Dataset
links point to pinned conversion inputs for runnable sources. **Input rows**
counts that selected input population before conversion or curation. TaskTrove
archive entries refer to the pinned `open-thoughts/TaskTrove` parquet inputs;
their counts do not describe the separate curated release. Generators have no
fixed count. Unknown counts are omitted from totals, which may include
overlapping populations. Whole-file parquet counts can be regenerated offline
with the task-curation `count_inputs` command after verifying download revisions.

Families and tags support discovery. Source-specific aliases are searchable tags.
Dataset and verifier references provide the revisions used for review applicability.
A changed revision hides a rating that covers a different revision and preserves
its review link and history. Counts and source information change through the
repository definitions and a rebuilt applet.

Quality links to a sample-based review: green Good, yellow Some issues, red Bad,
or gray Unreviewed/Unrated. Review date records the latest actual judgment time.
Good means the sampled review found no qualifying defects; it does not prove
the verifier correct for every task. Confirmation requires actual native inputs
and a reproducible mismatch between expected and observed behavior, such as
rejecting valid tasks, grading correct answers incorrectly, or supplying judge
examples that violate the parser contract. Suspicions alone do not confirm a
defect. A subsequently reproduced verifier,
task-import, or judge-output defect must receive a GitHub issue and an
evidence-backed supplemental review in the source's review pool. Record the
defect in `catalog_verifier_issues` and demote Good to Some issues. Preserve Bad
ratings, previous judgments, execution traces, and difficulty reports.

An open confirmed defect prevents publishers and refreshes from restoring Good
or displaying a current difficulty estimate. Difficulty collection selects only
Good sources. Clearing the defect requires a validated fix and a new source
review on the applicable dataset and verifier revisions; closing a GitHub issue
or merging a fix alone is insufficient. In that source's `catalog_verifier_issues`
row, set `status` to `resolved` and set `resolution_review_id` to the new
`catalog_reviews.id`, which identifies a whole review collection rather than an
individual judgment. All open confirmed defects must be resolved before Good
can return. Then publish the new source rating.

The resolution collection must include a verified native runtime observation
and a review with `attributes.resolved_verifier_issues` entries recording
`issue_url`, the current `dataset_revision` and `verifier_revision`, and
`fix_validated: true`. Link the actual validation evidence in that review.
The runtime observation uses `method: runtime_execution`, `tests_executed: true`,
and `attributes.verification.status: verified`, produced by actual native
execution rather than a manually assigned score. Validate the repaired behavior
against the saved failing inputs and appropriate incorrect-answer controls.
The historical finding remains in the review pool.

Reuse existing source and task reviews before scheduling fresh quality judgments.
An imported audit with no Atlas rating needs a source synthesis that cites its
findings, preserves disagreements, and states applicability to the current release.
Missing ratings alone do not warrant another solver run or three-judge panel.
Reserve additional judgments and native checks for specific evidence gaps,
changed tasks/verifiers, consequential disagreements, or suspected defects.
Keep unknown historical dates and revisions unknown. Record the inspected
MarinSkyRL reference commit separately from native execution provenance.
The review page displays the reference commit for reused evidence, and marks
superseded opinions while keeping their original content available.

Demotion sets `catalog_sources.difficulty` to null. Existing `difficulty.json`
reports and traces remain in `review_artifacts`; the review page labels them as
historical while a defect is open. Collect and publish a current difficulty
estimate only after the source qualifies as Good again.

The review page shows native outcomes, three independent model judgments per task,
task/source syntheses, the MarinSkyRL commit, and linked evidence. Difficulty shows
model solve rates; its report includes task counts,
sampling scope, checkpoint revisions, and uncertainty. These curated columns are
stored separately and survive refreshes. Changed source data or verifier revisions
hide stale Quality and Difficulty values while retaining the historical review link.
Nemotron component recipes pin the blend revision and row selection; their counts
come from the complete selection audit at that revision.
Reusing a historical review at a later repository revision requires separate
evidence that its data bytes, component selection, and verifier still apply.
The review page links that evidence and preserves the actual judgment date and
executed revision; the catalog displays the pinned input revision.

For sources rated Good, the page shows the quality solver's initial solve
count and the saved [issue #8942](https://github.com/marin-community/marin/issues/8942)
evidence above the difficulty comparison. The quality sample can be smaller
and drawn differently. Three judges assess each attempt; their opinions are not
additional solver attempts. Historical dataset releases and rollout settings remain
separate from current measurements.

The catalog's Difficulty column shows each measured model's solve rate as a bar
with solved/verified counts. The denominator is the saved report's `verified`
count; the atlas does not recompute it from task attempts. Consult the report's
attempt records to determine whether timeouts, missing final answers, and
verifier failures were included in that count. Longer bars mean more tasks solved. Current comparisons
use the same task sample for three fixed models:

| Role | Model | Reasoning setting |
| --- | --- | --- |
| Small | Qwen/Qwen3-Coder-30B-A3B-Instruct | Non-thinking checkpoint |
| Large | Qwen/Qwen3.5-122B-A10B | Thinking enabled |
| Hosted | zai-org/GLM-5.3 on Together | Low reasoning effort |

The `atlas-difficulty-v2-65k16k` protocol gives each model 65,536 total context
tokens, at most 49,152 input tokens, and at most 16,384 output tokens including
reasoning. All three use temperature 0.7, top-p 0.95, top-k 20, min-p 0,
repetition penalty 1, and presence and frequency penalties 0. These explicit
settings prevent checkpoint generation defaults from changing the comparison.
Models retain their native reasoning controls; the shared token budget does not
make those controls equivalent. Nemotron's learned verifiers use Hosted GLM-5.3
with Low reasoning effort across all three arms. Their native output budgets
and saved critic requests and responses appear with the run evidence.

Earlier comparisons display their actual model names and Historical status.
Historical GLM AWQ runs are separate from the current Large Qwen model, and
historical Qwen3.5-9B runs are separate from the current Small model. Filters and
sorting use only current Qwen3.5-122B-A10B solve rates; sources without a current
measurement sort last. Archived reports retain their original settings and traces.
Click the bars to open the source's Difficulty section.
Each model has collapsible run settings and task attempts, including saved model
requests, responses, and native verifier results and logs. These saved artifacts
load when their sections open. Missing artifacts are identified explicitly.
Changed data/verifier revisions or unresolved verifier defects prevent a current
difficulty comparison.

Generation-setting follow-ups show their solve rates and changed settings beside
their earlier comparisons, with links to both reports and traces. The output budget
includes thinking tokens. When a model exhausts it before submitting a final
answer, the report records that failure so the solve rate can be interpreted for
the stated generation setting.
Hosted model follow-ups also record changed checkpoints, generation settings,
and configured LLM verifier judges. They do not isolate the effect of reasoning
effort. Providers may expose a model identifier without an immutable checkpoint
revision. Changes to quality-review code alter the recorded `make_review.py`
file hash even when difficulty execution is unchanged. A saved
`publication-execution-implementation-attestation.json` compares the parsed
Python definitions used by the difficulty task loader, worker launcher, and
MarinSkyRL code identity check, including their referenced definitions and imports.
The original and executed script's whole-file hashes remain recorded.
Matching definitions establish unchanged code within the attestation's recorded
scope. They do not establish that external services or task data were unchanged.
A differing comparison cannot support unchanged-execution provenance; publication
requires validation of the changed execution code.

The review tooling, example configuration, and JSON schema are checked in under
`experiments/rl_data_reviews/`. Copy `review-config.example.json` to a local file,
then set the model endpoint and the solver checkpoint commit in `model.revision`, native checkout and
Python paths, source identity, and local task path. Gym execution requires the
MarinSkyRL runtime dependencies; Harbor execution also requires Harbor and its
configured environment provider, such as local Docker. API keys belong in the
environment variable named by `model.api_key_env`.

From the repository root, create and publish a review with:

```bash
uv run experiments/rl_data_reviews/make_review.py \
  --config /path/to/review-config.json --n 3 --seed 42 --output /path/to/review
uv run experiments/rl_data_reviews/publish_review.py \
  --run-dir /path/to/review --atlas-id 'MarinSkyRL:svamp'
```

Set `source.source_id` to the Atlas population named by `--atlas-id` and
`source.revision` to that dataset's HF commit. `source.tasks_path` points to the
local inputs. Use `source.format: skyrl_prepared` for Parquet or JSON rows with
MarinSkyRL's `prompt`, `env_class`, and verifier arguments; use `harbor_directory`
for native Harbor task directories. `task_manifest` accepts JSON or JSONL records
matching the `Task` dataclass in `make_review.py`, including the source ID and
dataset revision for each record. Preserve the native verifier arguments and
selected subset when preparing these inputs.
The script samples local tasks, runs native Gym or Harbor verifiers, saves solver
and verifier traces, obtains three fresh review sessions from the solver model, each without the other judges' opinions,
and synthesizes the opinions. Each Harbor attempt uses a distinct session name.
Add `--resume` to the first command to reuse completed task outcomes, judge outputs, and syntheses with matching inputs,
configuration, and native code. The publisher validates the collection and
uploads its cited evidence into the applet schema using Marina authentication.
Imported Task Trove dashboard notes and task audits remain separate historical collections; this publisher creates new collections from actual task attempts.

Opening the [authenticated page](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
synchronizes the packaged catalog with the saved inventory. **Refresh sources**
forces that synchronization even if its content hash is unchanged. The public
page reads the saved inventory and shows its last synchronization time.
The complete artifact is validated before any source rows change. Refreshes
retire removed sources, retain their historical review records, and preserve
saved quality, difficulty, and traces for surviving IDs. Concurrent visitors
share a refresh lock.

Source code lives in `infra/marina/applets/rl_data_catalog/`. Persistent data lives
in `catalog_sources`, `catalog_refreshes`, `catalog_reviews`, and `review_artifacts` within the applet's Postgres schema.
Validate and update the same applet from the repository root:

```bash
uv run marina validate infra/marina/applets/rl_data_catalog
atlas_revision=$(uv run marina applets versions fb11c931-5861-4878-8bb5-a964d652b45f --json |
  python -c 'import json, sys; print(json.load(sys.stdin)["current_version"])')
uv run marina publish infra/marina/applets/rl_data_catalog \
  --update fb11c931-5861-4878-8bb5-a964d652b45f --base-version "$atlas_revision"
```

Use the actual current revision reported by `versions` as `--base-version`.
For local testing, use `uv run marina publish infra/marina/applets/rl_data_catalog --local`.

To export the catalog without publishing, run from the repository root:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.export_catalog \
  --output infra/marina/applets/rl_data_catalog/dist/catalog.json
```

The generated file is ignored by Git and regenerated by the applet build command.
It contains schema version 1, source rows, and a SHA-256 revision derived from
those rows. Saved review artifacts remain in Postgres and are not bundled into
the generated catalog. Source edits belong in the task-curation definitions.
