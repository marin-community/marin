# Capability-driven RL task generation

This project implements the recipe in [Task Generation.md](Task%20Generation.md):
GLM-5.3 designs ten proposals per capability, reviews and repairs the portfolio,
then agent sessions construct tasks against a pinned Marin TaskSpec contract.
Proposal approval is a construction decision; it is never a runtime certificate.
The final recipe must run generation, adversarial testing, and bounded revision
with GLM-5.3 and minimal operator intervention. Development subagents are not
runtime dependencies. The [generate command](docs/unattended_generation.md) now
chains proposal generation and construction with bounded task repair; unattended
recovery and the broader evaluation matrix remain unfinished.

Refinement 003 accepted 100 of the pilot's 340 proposals. Review-only 005 accepted
200 unchanged proposals, with no missing slots; this is review acceptance, not a
measured quality gain. That 200-count remains the immutable historical run result.
Subsequent audits revoked eight exact hashes: two arithmetic failures and six
internal proposal contradictions. A current singleton check through the normal
fail-closed loader leaves 192 structurally loadable review005 entries; none is
runtime certified. The [dated revocation-status sidecar](docs/audits/proposal_review_005_revocation_status_20260918_r2.json)
records the exact hashes and loader result, and `data/revocations.json` enforces
the exclusions.
Real no-tool, ShellSim and Daytona protocol trials have run, including remote
executable grading. The c17 widget-lifecycle task passed the earlier pilot runtime
and semantic-review gates, but its expanded three-run evaluation exposed a
duplicate-key submission receiving full reward. Its two construction repairs are
exhausted; the current version is rejected under expanded evaluation. The
[historical acceptance audit](docs/audits/c17_acceptance_008.json) and
[current disposition](docs/audits/c17_duplicate_member_disposition_003.json)
preserve both results. Pilot acceptance is not training admission.
See [recorded outcomes](docs/experiments.md), [construction repairs](docs/construction_repair.md),
and [intentional partial-credit controls](docs/partial_controls.md).
The [repeated-runtime evaluator](docs/evaluation.md) can freeze and execute three
fresh full suites without changing historical task acceptance. The first completed
campaign retained all three solver successes and the rewarded attack; it correctly
finished requiring review. Broader recipe gaps remain explicit.
The c32 portable-image revalidation subsequently passed its authored oracle,
independent solver, six authored negatives, and three independent attacks. Its
quality review still requires resource-envelope and repeated-evaluation evidence;
it is not accepted for training. See the [terminal audit](docs/audits/c32_image_migration_revalidation_terminal_004.json).

The historical pilot samples one capability from every curriculum in the original
34-curriculum catalog. The expanded `new_catalog.json` contains 1,999 capabilities
across 45 curricula and is the input for future coverage. Each manifest record
includes the entire original capability and a content hash.
See [catalog provenance](docs/catalog_audit.md), [the task contract](docs/task_contract.md),
[cluster execution](docs/infrastructure.md), and [the experimental recipe](docs/recipe.md).

## Run the proposal pilot

From this directory:

```bash
uv sync --group dev
uv run pytest -q
scripts/submit.sh --pilot data/pilot.json --out runs/proposal-pilot-001 \
  --concurrency 256 --tier interactive
```

The launcher uses the existing authenticated Iris/CoreWeave and Orion GLM service
configuration. It stages only the runtime inputs and streams outputs to the
run's CoreWeave prefix. Credentials never belong in task artifacts. Use a new
run name for changed experiments; resume the same run only after confirming its
previous worker is terminal. Small controller/schema unit tests run locally;
inference, agent construction, generated code and rollout experiments run on the cluster.

For larger runs, select `--tier bulk` and use hundreds of concurrent work items.
Prepare the entire catalog with `uv run --frozen -m capability_pipeline.catalog new_catalog.json
--all-json data/new-catalog-all-capabilities.json`, then pass that manifest to `--pilot`.
The expanded catalog yields 1,999 capabilities and 19,990 proposal slots.
The planning stage has one item per capability and generation has ten per
capability. A work slot refills immediately when a request ends. Readiness checks
use actual serving-worker counts; rate limits and missing relay routes produce
infrastructure holds. Do not interpret fleet-wide shared-token demand as this
run's utilization, or lower concurrency without recording an observed failure.

## Inspect a run

Each stage retains the exact request, raw streaming responses, reasoning, usage,
timing, validation result, and input/prompt fingerprint under `items/`. Atomic
results are reusable only for identical requests and are revalidated on resume.
The top-level artifacts are:

- `input_pilot.json` and `run.json`: input identity and experiment settings.
- `plans.json`: ten deliberately differentiated slots per capability.
- `proposals.json`, `proposals-round-N.json`: original and repaired blueprints.
- `reviews-round-N.json`: independent seven-axis reviews and concrete feedback.
- `accepted.json`: proposals with passing individual and portfolio reviews.
- `rejected.json`, `null.json`: visible failures, unresolved repairs and abstentions.
- `report.json`: counts, environment/verifier distribution, missing slots and failures.

Reports keep failures observed during the current run under `failures`. Seeded
experiments retain earlier failures separately under `inherited_failures`, with
their source-report hashes. A recovered proposal can be present even when its
earlier failed generation remains in that history; `missing_slots` describes the
current artifact coverage.

Every proposed slot is accounted for. Missing work and failed requests do not
become null proposals; invalid, incomplete or unreviewed artifacts cannot enter
the accepted set. A model response cut off by its token budget gets one explicit
structural repair attempt with a larger output budget. Semantic repairs receive
a fresh independent review before acceptance. A review-declared missing slot gets
one bounded regeneration from its frozen slot plan and recorded failure context;
the original failure remains in the audit even when the fresh review accepts it.

To begin a bounded build pilot while portfolio repairs continue, prepare
individually promising candidates and run a separate construction review:

```bash
uv run python -m capability_pipeline.admission runs/proposal-pilot-002 \
  --out data/construction-candidates-001.json --count 9
scripts/submit.sh --stage admit --source data/construction-candidates-001.json \
  --out runs/construction-admission-001 --concurrency 256 --tier interactive
```

This review sees all portfolio feedback and retains source hashes and fresh
review history. It applies the same individual quality thresholds; unresolved
required changes block acceptance. Its `accepted.json` permits construction only
and does not certify the original portfolio or runtime behavior.
An initial `reject` stops admission by default. For a specifically audited,
salvageable proposal, pass `-- --repair-rejected` to permit one of the declared
repair rounds to produce a new proposal hash and undergo a fresh review. The
flag and policy are recorded in run metadata; an unchanged or finally rejected
proposal remains rejected, and a repair may still abstain with `null`.

To apply improved review/repair prompts to an existing terminal proposal run,
use `propose --seed-run PATH` with a different output directory. The seed needs
`input_pilot.json`, `plans.json`, `proposals.json` and terminal `report.json`.
The controller verifies the exact pilot, validates inherited artifacts, records
their byte hashes, skips initial generation, and freshly reviews every portfolio.
Previous acceptance is never inherited. For cluster execution, pass a compact
seed directory through `submit.sh --stage propose --source PATH`; the worker
validates those four files for proposal-stage inputs.
Use `-- --repair-rounds 2` to request two bounded repair rounds.

## Quality boundaries

Simple answers, executable graders and rubric judges have distinct validation
requirements. Although the pinned upstream specification prose still calls
LLM-as-judge a stub, the same pinned source implements the native judge in
`taskcompendium/judging.py` and wires it through Harbor's `SemanticVerifier`.
This project's `judge-calibrate-task` command exercises that exact adapter with
private positive, wrong, empty and injection controls. Keep judge tasks pending
until both calibration and the actual private Harbor path pass live tests.
The bounded native-judge protocol probe passed three credentialed model-path
trials through the pinned Harbor verifier; see `docs/judge_probe.md`. That result
establishes transport and elementary discrimination, not task calibration.

Final task publication requires typed TaskSpec validation, public/private resource
isolation, executable positive and negative controls, a real environment lifecycle,
and solver/adversarial rollouts. Preserve `graded`, `extraction_error`,
`invalid_task`, and `infra_error` separately; only a graded outcome has a reward.
The recipe and evidence ledger identify requirements still awaiting live evidence.

Runtime validation and semantic acceptance are separate. The independent
[final quality review](docs/quality_review.md) checks the complete generated task,
its admitted capability and every planned validation condition against a frozen
artifact snapshot. Only `quality_accepted` tasks are exported by the current
controller; `runtime_validated` alone is insufficient. Older experiment snapshots
retain their original controller states and are not retroactively certified.

An approval can be revoked when subsequent evidence exposes a defect. The
hash-bound [revocation registry](docs/revocations.md) blocks those proposal bytes
from construction while permitting them as inputs to a fresh repair review.
Original reviews and reports remain intact as historical evidence.
