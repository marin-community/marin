# RL Data Atlas

[Open RL Data Atlas](https://public.applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
to browse the latest saved [MarinSkyRL](https://github.com/marin-community/MarinSkyRL) sources, the [Task Trove release](https://huggingface.co/datasets/open-athena/task-trove) manifest, and checked-in release registrations without signing in.
Task Trove packages converted datasets as tasks for the Harbor execution
environment. The catalogs are independent and can share original datasets.
Its UUID is `fb11c931-5861-4878-8bb5-a964d652b45f`; the stable link always opens the current release.

Search and filter the table, including its Environment column, click column
headings to sort, and use the information button beside a source name to inspect
counts, classifications, and pinned evidence links. Names use full repository IDs
and distinguish selected subsets; registry aliases remain searchable.
**Show deprecated / excluded** reveals Task Trove sources with
no released tasks. They are hidden by default. Export view downloads the filtered
rows as CSV.

Source rows are identified by catalog and registry or manifest source key. Each
non-mixed registered dataset or generator appears once in the source inventory. Gym entries
bound to already-listed sources are merged into those source rows. Their counts
are the selected dataset split or configuration counts. An environment used by
several sources remains a filterable attribute of each source; it does not add an
aggregate row. The Environment column and verifier dates follow the environment
selected by that registry source. Distinct configurations, blends, and converted Task Trove releases
remain separate source populations.

The **Canonical source** column identifies the registered source population. For
sources that previously had Mixed interaction, each component has its own row
with the shared parent in Canonical source. Other rows name themselves as their
canonical source. Kind is omitted from the table, filters, details, and CSV.

HH-RLHF expands into four train collections: harmless-base, helpful-base,
helpful-online, and helpful-rejection-sampled. Their counts are corroborated by
[tasksource's per-collection metadata](https://huggingface.co/datasets/tasksource/hh-rlhf).
All support conversational Multi-turn interaction. KTO Mix expands into its
Capybara, Intel Orca, and UltraFeedback components. Counts use the original
[Argilla DPO mix's source frequencies](https://datasets-server.huggingface.co/statistics?dataset=argilla/dpo-mix-7k&config=default&split=train)
with two KTO records per preference pair. Capybara supports Multi-turn; Orca and
UltraFeedback use Single-turn prompts. These are contributions to the selected
canonical population, not the full original component datasets.

The three registered Nemotron Ultra blends expand into component populations.
Each child row names its parent blend in Canonical source. RLVR1, RLVR2, and MOPD counts come from scanning every record of the complete
original JSONL files in [nvidia/Nemotron-RL-Ultra-Training-Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/tree/79f8eda15ea12e1adf7bb14dcb338a29d391b80e)
at revision `79f8eda15ea12e1adf7bb14dcb338a29d391b80e`:
98,424, 99,116, and 85,980 records respectively.
The counted file SHA-256 and record dataset selector are exposed in details.
Counts distinguish ordinary and tool-assisted math, ARC variants, and structured
output variants. Tool-assisted math and Lean refinement use Agentic/Multi-turn.
The component counts within each blend sum exactly to that parent blend's total.

Both RLVR blends contain 13,898 SWE records. Exactly 2,838 match the 366 SWE-Gym
instance IDs present in the blend; the remaining 11,060 are attributed to
SWE-rebench-V2 using [the blend card](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/79f8eda15ea12e1adf7bb14dcb338a29d391b80e/README.md)'s exhaustive two-source SWE composition. Membership
was checked against [SWE-Gym/SWE-Gym at revision
`bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym/tree/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb). This counts blend records, including
multiple trajectory steps for a shared instance; it is not a count of unique SWE
issues.

MOPD has 13,132 SWE records: 2,990 SWE-Gym and 10,142 rebench records. Its
three SWE dataset selectors remain separate, as do GenRM's `hs3_en`, `hs3_multi`,
`hs3_multiturn`, and `safety_en` populations and newer instruction-formatting
variants. The 7,191 SWE records lacking a dataset field are identified by their
explicit SWE agent reference and instance metadata; they form one of the three
selector populations. The same membership rule assigns 1,884 of these records
to SWE-Gym and the remaining 5,307 to rebench. These contributions are included
in the 2,990/10,142 totals; none are omitted.
`hs3_multiturn` uses Alignment/Multi-turn. The indirect prompt-injection component
uses Agentic/Multi-turn and has 2,000 records. All current Nemotron component
counts are exact and included in filtered totals.

No aggregate parent rows remain in the table. A changed upstream revision
invalidates bundled exact counts and uses card estimates until that revision is
audited. The rounded card percentages do not establish exact individual counts.
If a card combines two sources without individual counts, the fallback leaves
those counts unknown and omits them from the tally. Refreshes retire old rows
atomically; a failed composition lookup preserves the previous snapshot.

To rebuild counts from complete local files, run the audit module with an
snapshot directory named by the HF revision SHA. It must contain complete
`rlvr1.jsonl`, `rlvr2.jsonl`, and/or `mopd.jsonl`; other filenames are ignored.
The identifier JSON must contain `revision` (the SWE-Gym repository SHA) and
`rows` (objects with `instance_id`). Use the complete upstream identifier list,
not a sample: 2,438 entries at the pinned SWE-Gym revision. Its shape is
`{"revision": "SWE_GYM_SHA", "rows": [{"instance_id": "INSTANCE_ID"}]}`. The audit is an offline maintenance step; page loads
consume the bundled counts and fetch metadata, rather than scanning task files.

```bash
uv run python -m infra.marina.applets.rl_data_catalog.audit_nemotron \
  --snapshot /path/to/snapshots/HF_REVISION_SHA \
  --swe-gym-identifiers /path/to/swe-gym-identifiers.json \
  --output infra/marina/applets/rl_data_catalog/server/nemotron_counts.py
```

The top line shows one data-source count, the filtered task tally, and upstream
status. The source count counts displayed source/component rows across all
catalogs, excluding deprecated/excluded rows; an expanded parent contributes
one count for each component and has no separate aggregate entry. It stays global when search
filters change. Gym aliases such as `gym/aime` remain searchable, and source details
include the gym entrypoint and registration link. Adapters without a registered
input source are omitted. Generators display **Generated** and have no fixed count.

The task tally above the table sums known finite dataset counts in the currently filtered rows,
including deprecated sources only when they are shown. Task Trove counts released
Harbor tasks. MarinSkyRL counts the selected split, configuration, or source
filter before rollout preparation and deduplication. The details panel and CSV
distinguish exact HF metadata counts, upstream-reported counts, and estimates.
An estimated row and any filtered total containing it display `≈`.
Merging gym rows removes duplicate inventory contributions. Source populations
can still share task records; the tally is not a count of unique tasks. Missing
finite dataset counts are omitted from the sum and listed separately. Generators display **Generated** and contribute no fixed count to the tally.

Counts unavailable in HF's viewer are resolved from the current dataset
card: APPS uses its 5,000-row train split; Eurus uses its 25,276 coding train
problems; Nemotron instruction following uses its 56,339-row RL population;
Nemotron Ultra uses the card's separate blend counts before component expansion. GPQA uses the 198-row
`gpqa_diamond/train` configuration. ASDiv's GitHub README reports 2,305 problems;
ASDiv and Reasoning Gym link to their GitHub repositories rather than nonexistent
HF repositories. Count evidence links record the revision used for the latest
refresh and advance when upstream revisions change.

OpenScience has four configurations and a partial HF viewer. Its displayed count
is the sum of complete configuration counts and HF's estimates for incomplete
configurations: approximately 4,476,918 rows at the September 28 audit. The dataset
card reports approximately six million. The applet exposes this discrepancy in
the count basis and does not treat the 1,200,238 preview rows as the full dataset.
This is a repository-wide count across all four configurations. OpenScience's
native loader requires an explicit `parameters.subset`; task inputs and reviews
retain that selection.

All Agentic sources use Multi-turn interaction, including excluded Task Trove
sources. Task Trove sources use Harbor and Agentic. Other interaction labels
describe the supported conversation structure; a Multi-turn collection can
contain one-turn examples. MarinSkyRL families were audited against upstream
cards/schema and selected loaders on September 28, 2026; each links to its pinned
evidence. These audited family assignments remain fixed until reviewed again.
Type distinguishes verifiable-reward tasks (RLVR), preference alignment, and
agent interaction. Family describes the task domain, such as math-answer or
competitive-programming; interaction labels describe single-turn or multi-turn
conversations. Task Trove families come from its release manifest. Benchmark flags
include explicit card designations such as AIME 1983–2024, APPS, and GPQA as well
as HF's official benchmark tag. A false flag means no designation was found.
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
Nemotron component identities and exact counts survive repository metadata changes
when the selected blend file still matches the complete-file audit's SHA-256.
The audit is bundled in `server/nemotron_counts.py` under the applet directory.
Reusing a historical review at a later repository revision requires separate
evidence that its data bytes, component selection, and verifier still apply.
The review page links that evidence and preserves the actual judgment date and
executed revision; the catalog still reports the latest repository revision date.

For sources rated Good, the page shows the quality solver's initial solve
count and the saved [issue #8942](https://github.com/marin-community/marin/issues/8942)
evidence above the difficulty comparison. The quality sample can be smaller
and drawn differently. Three judges assess each attempt; their opinions are not
additional solver attempts. Historical dataset releases and rollout settings remain
separate from current measurements.

The catalog's Difficulty column shows each measured model's solve rate as a bar
with solved/verified counts. Longer bars mean more tasks solved. Current comparisons
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

## Create and publish a review

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
`source.revision` to that dataset's immutable release revision. `source.tasks_path` points to the
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

Opening the [authenticated page](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/) checks the MarinSkyRL registry repository and Task Trove release
repository heads and always refreshes MarinSkyRL upstream dataset metadata, including
when the MarinSkyRL head is unchanged. The public page reads the latest saved
catalog and shows its last upstream check time; it does not refresh upstream sources.
When the MarinSkyRL repository head
changes, the applet reloads verifier commit history; **Refresh sources** also
refetches metadata when heads have not changed. The applet downloads no task archives.
Counts previously verified for a gated viewer are retained when access fails only
if the dataset revision still matches. A changed revision invalidates that evidence.
Each successful upstream snapshot
replaces that catalog's active rows atomically. A failed refresh preserves its
previous snapshot and shows an error. Concurrent visitors share a refresh lock.
For MarinSkyRL, Last revision is the newer of the HF or GitHub source update and the latest
commit touching its verifier implementation directory. Both dates, their commits,
and the registry date are available in the details panel. Task Trove dates refer
to its release repository, not each original dataset.
A source selects a gym environment whose verifier scores answers or actions.
Verifier edits can change training rewards even when the dataset files are unchanged.

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

On HF rate limits, the backend attempts to read
`projects/hai-gcp-models/secrets/HF_TOKEN_READONLY/versions/1` using Marina's Google identity.
The runtime service account has access scoped to this secret through the grant in
[PR #9501](https://github.com/marin-community/marin/pull/9501). A Marina-authenticated
maintenance refresh can instead send the token in `X-HuggingFace-Token` with its
POST to `api/refresh?force=true`. To verify runtime access, send a Marina-authenticated
POST to `api/refresh?force=true&hf_auth=runtime` without a caller-supplied HF token;
the response reports `hf_authentication: runtime_secret`. The token is used only for HF requests and is
never stored in the applet tables or sent to the frontend. Normal page loads reuse Task Trove snapshots when its release head is unchanged.

## Register a pinned release

Add a `RegisteredSource` to
`infra/marina/applets/rl_data_catalog/server/registered_sources.py`. These entries
appear under **Registered releases** and require no training registration. The
storage host is separate from the execution environment and interaction type.
Set `release` with `ReleaseReference(host, repository, revision)` and `execution`
with `ExecutionContract(environment, type, turns, tools, agent, scoring)`. Set
`verifier_revision` to the bundled release pin or a separately pinned verifier.
HF and Git references require a full commit SHA; Harbor Hub references require
a `sha256:` release digest. Register native release identities without copying
tasks into another repository.

Choose a stable `name` for the source population. Its Atlas ID is
`Registered releases:<name>`. Record the selected populations and their counts
in `counts`; use `all` for a release without named splits. Counts must cover the
complete selected population, including all membership API pages. Link the count
evidence, verifier, license scope, and execution instructions. Declare the tool
policy, agent requirement, and scoring contract explicitly. The registration
describes these requirements; evaluation adapters must enforce them.

Submit registrations and release updates through a Marin PR. After deployment,
refresh reads the bundled metadata without downloading tasks or following
moving upstream tags. Unchanged registrations retain their saved snapshot.
Removing an entry retires its row and preserves historical reviews.

The source's `registration_revision` fingerprints the release reference, counted
populations, execution contract, and verifier revision. Changes to these fields
make prior ratings stale; edits to notes and evidence links preserve applicability.
Before starting a review, copy this value from `api/sources` or the details panel
into `source.registration_revision` in the configuration described in the
[review workflow](#create-and-publish-a-review). Keep
`source.revision` equal to the native dataset commit or package digest. The
publisher checks both identities and archives the run configuration. A missing
or mismatched registration revision cannot publish a current assessment.

Registration establishes no quality or difficulty rating. Publish assessments
separately against the deployed registration. The shared Atlas intake accepts
Git, Hugging Face, and Harbor Hub references; it does not change the Harbor fork's
Git/local-path registry schema or add task download and training adapters.

### PDBThink coordinate tasks

`Registered releases:pdbthink-coordinate-tasks` pins
[PDBThink Coordinate Tasks v1.3.0](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/tree/3734406cb97b1702844319f9a5d860cbbf8fe660):
100,000 tasks across 19 families, with 91,154 train, 4,411 validation, and 4,435
test tasks. The population excludes the frozen PDBThink benchmark's source
entries, exact protein sequences, and associated RCSB 30% clusters, as described
in the [release manifest](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/manifest.json). Publisher validation
is linked; Atlas quality and difficulty remain unreviewed.

PDBThink requires one tool-free response using `CoordinateNoToolsAgent`. Supply
only `prompt.json` to the solver; gold, provenance, tests, and oracle solutions
are evaluator-only. Follow the pinned release's
[execution instructions and native-token budget manifest](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/USAGE.md).
Atlas's default Terminus-2 runner permits tools and requires a separate adapter
before evaluating this population. The
[GLM teacher traces v1.0.0](https://huggingface.co/datasets/open-athena/pdbthink-glm53-teacher-traces/tree/v1.0.0)
retain their task v1.2.0 prompts and scores and do not establish difficulty for the registered
v1.3.0 release.

### Skill2Env

`Registered releases:skill2env` pins the official
[Skill2Env Harbor Hub release](https://hub.harborframework.com/datasets/skill2env/skill2env)
at digest `sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c`.
It contains 7,496 tasks; Harbor revision 2, version `1.0.1`, and the tags `latest`
and `v1.0` refer to that release. Project-owned code is Apache-2.0; third-party
skills and assets retain their own terms.

These tasks use multi-turn terminal interaction. Preserve their resource limits
and network restrictions. The Atlas scoring contract retains the executable
grader's component rewards, reports their arithmetic mean, and requires every
component to reach one within `1e-9` tolerance for a full pass. The paper's
additional LLM rubric reward is separate from those scores and Atlas's quality
audit. The component-reward adapter and offline Terminus tooling in
[PR #9632](https://github.com/marin-community/marin/pull/9632) remain separate
runner work.

[Issue #9630](https://github.com/marin-community/marin/issues/9630) records a
three-task assessment and a confirmed grader defect supporting a prepared
"Some issues" rating. Two attempts exhausted a 50-turn budget. Those results do
not establish corpus-wide difficulty, and the rating is not published by this
registration. Atlas's "Good"-only eligibility rule excludes a source rated
"Some issues" from the standard difficulty comparison. The registration starts
unreviewed; publishing the prepared assessment requires binding its evidence to
the deployed source ID and registration contract.
