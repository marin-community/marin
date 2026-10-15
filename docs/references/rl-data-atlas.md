# RL Data Atlas

[RL Data Atlas](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
reads a Postgres copy of the published [curation catalog](task-curation.md#run-catalog).
Curation owns source declarations, admission decisions, executed controls and
sampled GLM judgments. Atlas displays those records and their task packets.

The checked-in app also reads immutable curriculum-matching releases. The
curriculum pane shows task coverage by subject, group, capability and source.
Its loader and selected release are independent of the sampled grading catalog.

The 95% control pass threshold permits curation to skip validation of unsampled
tasks. Below that threshold, curation validates each task and retains individual
passes. The 80% Good threshold below grades a sample of the retained output.
Curation supplies the tasks and evidence; the grade does not change admission.

```mermaid
flowchart LR
  C[Curated TaskSpec parquet and stage evidence] --> P[Zephyr: sample and prepare packets]
  P --> G[GLM: one judgment per packet]
  G --> K[Typed FineStore catalog and archived packets]
  K --> L[Explicit loader: stage and check]
  L --> D[Select Postgres load]
  D --> A[Atlas: SQL reads]
  T[Public TaskSpec instructions and contracts] --> E[Zephyr: bounded public text and cached CPU Harrier embeddings]
  H[Curriculum anchor embeddings] --> M[Zephyr: exact matches and scores]
  E --> M
  M --> R[Count source outcomes and publish release manifest]
  R --> CS[Zephyr: coverage statistics and review samples]
  CS --> CL[Compact curriculum loader]
  CL --> CA[Postgres: compact statistics, curriculum and samples]
  CA --> A
```

## Reading grades

FULL campaigns grade automatically; SAMPLE opts in with `--assess`.
The default sample covers 100 retained tasks and up to 20 excluded rows.
"Admitted" in the catalog means a task retained in the final TaskSpec parquet
after curation. Excluded rows are sampled separately to inspect filtering decisions.
Admitted opinions, excluded diagnostics and executed grader controls remain
separate. Grading does not change admission. Input populations may overlap.

SQL computes each cohort's verdict counts and label over the entire selected sample:

- Good: at least 80% have usable task evidence and a Good judgment.
- Bad: more than 20% have usable task evidence and a Bad judgment.
- Unknown: neither threshold is met and fewer than 80% have conclusive judgments.
- Some issues: the remaining samples.

A completed or gated pipeline that processes inputs and retains zero tasks is Bad.
An incomplete pipeline without retained tasks remains Unknown. Missing packets
and unavailable judgments count against coverage; they are never dropped from
the denominator. Atlas displays the Good fraction and the reason for the label.
These opinions describe the inspected sample, not an estimate of the whole source.

Large packets retain the task and answer contract, with bounded previews of long
text, collections and stage audits. Previews record original sizes and selected
indices or text prefixes and suffixes; full evidence remains in curation outputs.
The model names any missing fact that prevents a judgment. Omitted private fixture
text alone does not force Unknown.

Assessment keys use ordinary artifact dependencies and explicit versions.
Changing only the rubric or grading model reuses curation. A failed new key has
no current grade; archived evidence remains available. Inventory-only sources
carry a skip reason. The source table shows the latest curation outcome; the
assessment page shows the curation associated with its selected grade.

## Updating Atlas

Source code lives in `infra/marina/applets/rl_data_catalog/`. Its migration
creates the SQL tables; it does not read object storage or run curation.
Publish code when the schema or UI changes, then load data explicitly:

```bash
uv run marina validate infra/marina/applets/rl_data_catalog
uv run marina release-status infra/marina/applets/rl_data_catalog --json
uv run marina applets versions fb11c931-5861-4878-8bb5-a964d652b45f --json
uv run marina release infra/marina/applets/rl_data_catalog --base-version CURRENT_VERSION

uv run python infra/marina/applets/rl_data_catalog/load.py \
  --catalog-root CATALOG_ROOT \
  --applet-id fb11c931-5861-4878-8bb5-a964d652b45f
```

`applet.toml` declares the existing applet's UUID. Releases require the
applet's publisher or a configured operator. By default, source must be a clean
checkout at the current remote main. A Marina image deployment does not publish Atlas. A
default release records its source commit and subtree hash in the immutable package;
`release-status` reports whether that tree matches this checkout.
Use `--allow-dirty` to explicitly release branch or uncommitted source without
verified Git provenance; `release-status` then reports `unrecorded` from a clean checkout.

Coordinate the code release and data load: the new API returns 503 until a
catalog has been selected. Run the loader in the
catalog's storage region. It reads through `CurationCatalog`, sends byte-bounded
SQL batches, checks sample and packet coverage, then selects the load with one
UPDATE. An unchanged snapshot skips loading. One operator runs updates; concurrent
loaders are unsupported. Abandoned and obsolete staging rows are removed on the
next update. Schema changes may require downtime.

Packets are stored separately as JSON text, preserving escaped null characters.
Inventory queries scan structured metadata and verdict rows. Atlas performs no
cloud reads, background refresh or inference. Request/response traces remain
archival references in the assessment record.

For a local preview, start disposable Marina and Postgres, then run the same
loader against a local curation cache using the printed applet ID and origin:

```bash
uv run marina release infra/marina/applets/rl_data_catalog --local --json
uv run python infra/marina/applets/rl_data_catalog/load.py \
  --catalog-root ~/.cache/marin/rl-curation/campaigns/catalog \
  --applet-id PRINTED_ID --url PRINTED_ORIGIN
```

Inspect the inventory, an assessment and its task evidence. Stop the local stack
with Ctrl-C when finished; its database is disposable.

## Curriculum coverage

Open **Curriculum coverage** from the source inventory. The heatmap starts with
all curriculum domains and their topics. Open a subject or group to descend through the
hierarchy, or search names and descriptions across the curriculum. Search returns
at most 100 areas. Select a cell for its task instances and a task for its public
instruction preview, embedding text, cosine scores and alternatives.
Long instruction previews record the original character count and truncation.

Displayed cosine scores are similarities; distance is `1 - similarity`.
The export uses fixed maximum distances of 0.45 for subjects and 0.25 for
capabilities. A task can have a capability match, a subject-only match, no
match within the cutoff, insufficient context, or a skipped embedding.
Alternative subject and capability matches remain available in sampled task details.

The **By curriculum** tab shows all curriculum domains and their topics together.
Open a domain to see its topics and skills on one page, or select **All skills**
to expand the complete curriculum. Counts shade each area; zeros stay visible
to expose gaps. The **By source** tab shows individual dataset columns. Source
family labels and answer-format tags are not curriculum dimensions. Opening
an area also loads its sampled tasks below the map; **View task examples**
jumps directly to that list. Empty areas explain when no examples are available.

Heatmap cells, column denominators and outcome totals describe every task
instance in the exported population. A Zephyr pass computes these counts by
source and curriculum node, including hierarchy rollups. Postgres sums these
small statistics for dataset and train/eval filters. Overview requests do
not scan task or match tables.

Examples combine two independent deterministic samples: up to 100 tasks per
source and 10 per curriculum node, including subjects, groups and capabilities.
The combined sample removes repeated source/task identities. The task list
and details show this bounded review sample; their size does not change the
full-population counts. Curriculum entries, statistics and samples form the
compact serving dataset. The complete task and embedding outputs stay in
object storage.

Converted populations contain successfully converted TaskSpecs before admission.
They retain training/evaluation intent. Selecting a source shows its full
conversion ledger when available. Sources outside the release remain in the
inventory; enable **Show sources without processed tasks** to show their striped
columns.

Run the embedding graph through `--embed coverage`. The `publication` stage
counts assignment outcomes by source, reconciles complete converted populations
against source manifests, and writes `release.json` referencing the assignment Parquet. The subsequent `coverage` stage writes aggregated
statistics, bounded example Parquet and `coverage.json` without rerunning
embeddings or matching. Load both manifests:

```bash
uv run python -m infra.marina.applets.rl_data_catalog.load_curriculum \
  --release IMMUTABLE_RELEASE_ROOT/release.json \
  --coverage IMMUTABLE_COVERAGE_ROOT/coverage.json \
  --applet-id APPLET_ID --url MARINA_ORIGIN
```

The loader reads statistics and sampled task files. It checks sample lineage,
training/evaluation use, curriculum references and sample counts. It reconciles
statistics against complete-population totals in the task release manifest before atomically selecting
the release. Byte-bounded SQL requests keep import work proportional to the
compact export. Failed loads preserve the selected release; the current and
previous releases are retained. Run one curriculum loader per applet at a time.
APIs read Postgres without cloud scans.

Start the local stack as shown in [Updating Atlas](#updating-atlas), then use the
same curriculum-loader command with its printed ID and origin. Stop the stack
with Ctrl-C when finished.

## PDBThink coordinate tasks

`Hugging Face:pdbthink-coordinate-tasks` registers
[PDBThink Coordinate Tasks v1.3.0](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/tree/3734406cb97b1702844319f9a5d860cbbf8fe660)
as an inventory-only source, with no conversion pipeline or current assessment.
The pinned [release manifest](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/manifest.json)
records 100,000 tasks across 19 families: 91,154 train, 4,411 validation and 4,435
test tasks. The [benchmark exclusions](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/README.md#relationship-to-the-benchmark)
cover the frozen PDBThink benchmark's source entries, exact protein sequences
and associated RCSB clusters at 30% sequence identity.

The task protocol requires one tool-free response to displayed protein coordinates.
Expose only `prompt.json` to the solver; gold, provenance, tests and oracle
solutions are evaluator-only. The pinned [usage instructions](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/USAGE.md)
describe `CoordinateNoToolsAgent` and native-token budgets. A curation conversion
must preserve these restrictions and the bundled deterministic verifier, whose
source is in [generator_source.tar.gz](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/generator_source.tar.gz).
Conversion or grader changes must bump the pipeline version used by assessments.

The publisher's [v1.3.0 correction audit](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/3734406cb97b1702844319f9a5d860cbbf8fe660/audits/contract_revision/regressions.json)
is release evidence. The separately published [GLM teacher traces v1.0.0](https://huggingface.co/datasets/open-athena/pdbthink-glm53-teacher-traces/tree/v1.0.0)
retain task-v1.2.0 prompts and scores. Neither supplies a current curation
assessment for this entry. Publish the inventory through a SAMPLE or FULL
campaign and use the [Atlas loader](#updating-atlas) to display it.

## Historical reviews

Historical reviews and raw artifacts remain in their original Postgres tables.
They do not affect current grades. Atlas retains an archive reader; the previous
manual review, source-verification and difficulty workflows are removed.
