# RL Data Atlas

[Open RL Data Atlas](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
to browse the latest MarinSkyRL sources and the Task Trove release manifest. The applet requires Marina authentication.
Task Trove packages converted datasets as tasks for the Harbor execution
environment. The two catalogs are independent and can share original datasets.
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
status. The source count counts displayed source/component rows across both
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
This is a repository-wide count: the SkyRL registry does not select a configuration.

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
The review page shows native outcomes, three independent model judgments per task,
task/source syntheses, the MarinSkyRL commit, and linked evidence. Difficulty can
show paired small/large model solve rates; its report includes task counts,
sampling scope, checkpoint revisions, and uncertainty. These curated columns are
stored separately and survive refreshes. Changed source data or verifier revisions
hide stale Quality and Difficulty values while retaining the historical review link.

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

Opening the page checks the MarinSkyRL registry repository and Task Trove release
repository heads and always refreshes MarinSkyRL upstream dataset metadata, including
when the MarinSkyRL head is unchanged. When the MarinSkyRL repository head
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
