# Snowball curriculum data

Freeze a source-stratified inventory, retain a smaller ranked prefix, and export
versioned rollout contracts. These commands build artifacts locally. Publication
and training use their own admission and runtime gates.

## Selection

`v0` reads admitted TaskSpecs from a pinned catalog snapshot or source manifest.
The recipe specifies source extensions, family weights, difficulty priors and
holdouts. Evaluation sources, matching logical IDs, matching public content and
matching private contracts are removed before quota allocation. Difficulty is an
uncalibrated prior. Reports retain the eligible census, rejected counts, realized
quotas, inclusion probabilities and any shortfall.

```sh
uv run python -m experiments.post_training.snowball_curriculum.v0 \
  --sources sources.json --recipe recipe.json --holdout-sources holdouts.json \
  --target 100000 --seed 0 --output selected
```

Both `tasks.jsonl` and `train.parquet` contain identical normalized records.
Nullable native fields have an explicit Arrow schema. `sources.json`,
`selected-aliases.jsonl`, `selection.json` and `recipe.json` describe the frozen
population. `artifact-manifest.json` inventories every payload by SHA256 and size.

`rebalance` retains a smaller seeded prefix within each original source/difficulty
bin. A requested increase beyond the cached predecessor fails. Reviewed native
imports are deduplicated and excluded against the heldout native token contracts,
then allocated across format/family/control strata. Original serialized
`lowered_task_spec` bytes remain in the selected inventory.

```sh
uv run python -m experiments.post_training.snowball_curriculum.rebalance \
  --previous selected --previous-manifest-sha256 SHA256 --plan rebalance.json --native-train native-train.parquet \
  --native-validation native-validation.parquet --output rebalanced
```

The plan declares `version`, `previous_version`, `seed`, `native_rows`,
`family_quotas`, `difficulty_policy` and `selection_policy`. The predecessor recipe
must contain its catalog URI, snapshot and migration receipt. No task payload is
rewritten during selection or rebalance.

## Current rollout export

The exporter runs in the pinned trainer CPU environment, which supplies
`skyrl_gym.rollout_specs`. Pure selection and annotation modules do not import the
trainer. `Deployment` requires all rollout limits, command timeout, shell tool
configuration and machine choices. Ordinary tasks use the production lowerer and
serialized `shellbox:{settings}` discriminator, restoring `ToolHarness` with the
same command settings in the worker. Assistant-prefix continuation is selected explicitly.

```sh
uv run python -m experiments.post_training.snowball_curriculum.export export \
  --source rebalanced/train.parquet --source-sha256 SHA256 \
  --source-uri rebalanced/train.parquet --source-manifest-sha256 MANIFEST_SHA256 \
  --configuration deployment.json --output runtime-v1
```

Canonical TaskSpecs retain their complete contracts. Native records go through
`taskcompendium.convert.structure_closure.structure_closure_task`: original
conversation events are followed by the supplied assistant text prefix, and the
shared structure-closure grader replaces the historical session grader. The
`assistant_prefix` harness requests continuation of that text using the selected model's
own tokenizer. Historical integer prompt IDs and tokenizer pins remain source
audit data. Historical terminal-marker normalization is absent in the current
grader. This changes native conditioning and grading semantics; historical replay
and numerical equivalence are unverified.

The exporter requires a resolvable, credential-free source artifact URI and its
manifest SHA256, verifies the entire local source inventory before and after
conversion, and records both pins in provenance. Local paths support local
reproduction; publish durable source locators with shared artifacts.

The exporter preserves logical IDs, source identity and private verifier
resources. `source_contract_id` and `source_public_content_id` identify the
original selection/session; explicit values supplied by a panel adapter are
retained. Current `contract_id` and `public_content_id` identify the converted
TaskSpec. `deployment-contract-mapping.jsonl` records source payload, full task,
public/private contract and serialized rollout hashes, with explicit native
semantic changes. Each row has one output; conversion failures abort export.

`derive` creates a fresh artifact for declared grader dependency lock replacements,
verifier startup budgets and outer rollout deadlines. Lock replacement requires
the environment provenance SHA256 and applies to sandbox graders. Host-side judge
environment metadata remains unchanged. Task payload changes beyond that whitelist
fail. The source manifest and artifact remain immutable. Derivation writes
`train.parquet` by default; select `--output-filename validation.parquet` when the
derived artifact will supply validation data.

```sh
uv run python -m experiments.post_training.snowball_curriculum.export derive \
  --source runtime-v1/train.parquet --source-sha256 SHA256 \
  --source-uri runtime-v1/train.parquet --source-manifest-sha256 MANIFEST_SHA256 \
  --configuration derivation.json --output runtime-v2
```

Export performs schema and real harness checks. Production machine capability,
credential resolution, exact TaskDataset admission, prompt budgets and actual
served-model transport require separate checks against the final launch config.
A local artifact manifest supplies file identity; remote publication needs a
verified copy/readback receipt covering every manifest payload.

## Annotation pilot

`pilot freeze` samples converted tasks before curation admission, retaining
holdout exclusions and the original denominator. `pilot annotate` applies a
static quality decision and public summary/tag/difficulty annotation. Model names,
revisions and hosted inference endpoints are explicit inputs. Failed/uncertain
reviews stay in accounting, and rejected tasks are never refilled. A quality keep
decision supplies a prior; it does not establish executability or grader accuracy.

`pilot embed` embeds accepted public summaries with the pinned Harrier model and
retains task-to-vector joins. Annotation requests omit private graders and
resources; the quality stage receives inline contracts with resource metadata
and explicitly unresolved external bytes. Commands default to planning; inference
and embedding require `--execute`. Use each subcommand's `--help` for its required
input and cache paths.
