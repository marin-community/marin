# Task curation

The source catalog in
[`experiments/post_training/task_curation/sources.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py)
returns 153 available Atlas source definitions. Each definition calls the
corresponding source-family module's `pipeline()` factory under `datasets/skyrl`,
`datasets/tasktrove` or `datasets/nemotron_ultra/{mopd,rlvr1,rlvr2}`. Dataset
modules own their input pins, component selection, conversion policy and runtime
binding. `sources.py` also reads Atlas metadata for all 203 entries, including
50 exclusions.

## Package organization

| Package | Responsibility |
|---|---|
| `experiments/post_training/task_curation/datasets/` | Pinned dataset declarations and experiment-specific input bindings |
| `experiments/post_training/task_curation/sources.py` | Explicit source-name to `pipeline()` catalog |
| `experiments/post_training/task_curation/pipeline.py` | `RlDataPipeline` metadata and artifact binding |
| `experiments/post_training/task_curation/driver.py` and `campaign.py` | Campaign configuration, shared pool and source admission |
| `taskcompendium.datasets` | Reusable dataset-family converters, rubrics and preserved source contracts |
| `taskcompendium.pipeline` | Acquisition, sampling, review, filtering, verification and sidecars |
| `taskcompendium.runtime` | Task execution and environment capture |
| `verifyit` | Common graders and invocation adapters for installed source scorers |
| `shellbox` | Isolated execution, image validation and backend implementations |

A `TaskPolicy` describes a family's normalization, review rubric and checks.
Library factories use `policy()` and return this bundle. A `DatasetRecipe` selects
pinned inputs and intended use, with its conversion policy in `recipe.policy`.
An `RlDataPipeline`
attaches source metadata and constructs the whole-source artifact with
`bind(config, runtime)`. Constructing the catalog performs no downloads, inference,
machine creation or job submission.

TaskCompendium provides reusable components and `run_source_pipeline`, which
executes an explicit recipe and writes data and evidence. Experiments provides
the source declarations and ArtifactStep wrappers: pins, dependency handles,
names, versions, output paths and resources. Only experiment dataset declarations
expose `pipeline()`. Nemotron acquisition and reference-file plumbing lives in
`datasets/nemotron_ultra/inputs.py` beside the dataset declarations. TaskCompendium has no imports
from experiments and constructs no ArtifactSteps.

```python
from experiments.post_training.task_curation.sources import rl_data_pipelines

source = rl_data_pipelines()["MarinSkyRL:math500"]
step = source.bind(source_config, source_runtime)
```

## Source procedure

1. Download or adopt exact pinned inputs and record their identity.
2. Select 100 raw records with a seeded sample, or all records in a smaller source.
3. Convert the sample and perform available cheap checks. Send bounded batches to
   GLM, or adopt hash-validated recorded reviews.
4. Apply the source quality gate. More than 90% known good judgments skips review
   of the remainder. More than 50% known defects rejects the source. Both use the
   entire panel as the denominator; unavailable responses count as neither good
   nor defective. Intermediate quality requires further review. Missing evidence
   can leave the gate incomplete.
5. In full mode, expand admitted sources, normalize their records and apply the
   selected review policy. Analysis and conversion may share a physical pass.
6. Verify a seeded sample of accepted outputs using the available source controls.
   The campaign requests 100 tasks, two attempts and a 0.95 pass fraction.
7. Persist canonical shards, evidence and the source report.

The shared campaign runs source procedures in threads over one Zephyr context.
Sources reuse the same worker pool across phases. `--concurrent-sources` controls
whole-source admission; `--max-workers` controls the shared worker count.
Independent sources continue when one fails, and the campaign report records
those failures.

GLM request batches are bounded by task count and bytes. Requests have bounded
retries. Failed or malformed reviews defer records, retain diagnostics and do
not count as task defects. Exact-query caching includes the complete request and
model revision. Regenerating TaskSpecs does not require repeating identical
expensive inference requests. Sidecars retain separate normalization, review and
verification identities.

Inference cache reuse is best effort. Unreadable storage and invalid cached
responses become misses; changed identities can regenerate outputs.

## Graders and goldens

Conversion preserves source grading semantics, including inline pytest graders.
A broken source test suite is a rollout failure and rejects the task. Conversion
does not repair comparators or rewrite tests to accept the golden.

VerifyIT descriptors and native commands are supported grading contracts. A
source can run its supplied command directly. Image-installed evaluators are
acquired from pinned upstream commits; the experiment's runtime manifest supplies
their resolved image pins. Task resources carry private grading inputs and result
bridges. A native command declares its output path and format:
`reward_file`, `reward_json`, or `score_json` with reward and diagnostic details.
A source-supplied script runs through a native command. Neither execution path
imports the converter. Packaging must preserve the original scorer's decisions. A source with
no available grader declares `source_unavailable`; verification records unsupported
readiness without creating a synthetic grader script.

TaskSpec has no universal golden-script field. Each dataset's control suite finds
its own source-provided witness, such as a private answer, reference program or
TaskTrove `solve.sh`. `/solution/solve.sh` belongs to the TaskTrove layout; it is
not required of every dataset. Missing goldens skip the positive control while
other available checks still run. Negative controls check that an incorrect
candidate fails. Missing controls and unsupported evaluators cannot certify a
working grader.

`EnvironmentRequirements.compatible_backends` declares acceptable Shellbox
backends separately for the agent and private verifier. Verification and RL
rollouts must select a declared backend and satisfy the task's runtime and image
requirements. A sampled pass certifies the backend exercised. QEMU runs inside a
Zephyr worker; Iris gVisor verification creates an isolated Iris job through
`IrisMachineFactory`. Secret bindings belong only to trusted private grading.

Quality acceptance and executable readiness remain separate. A completed source
procedure can retain inconclusive verification evidence. Read the source report's
quality decision, verification decision and grader readiness before choosing data
for executable RL.

## Run a campaign

Run the driver inside an Iris job whose `EnvironmentSpec` includes
`pip_packages=["./lib/taskcompendium[pipeline]"]`. Plan first:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.driver \
  --runtime-manifest runtime.json \
  --review-transport direct-chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --mode sample \
  --max-workers 64 --coordinator-memory 16g --concurrent-sources 10 \
  --normalized-shards 32 --worker-image REGISTRY/WORKER@sha256:DIGEST \
  --report-path CAMPAIGN_PREFIX/sample.json
```

The runtime manifest maps canonical source names to explicit settings. A QEMU entry
contains `backend: "qemu"`, an immutable grader `image`, its `worker_image`, and
the staged `qemu_bundle` path. Iris entries use `backend: "iris-gvisor"` and
an immutable grader image. Pass `--controller-url` when required by the runtime.
`--staged-inputs` adopts exact existing inputs; `--recorded-review-bundles` adopts
reviews with checked content hashes.

Add `--run --base-url PROVIDER_URL` to execute, with `GLM_BULK_TOKEN` configured in the
driver environment. Full execution also requires `--mode full
--sample-report CAMPAIGN_PREFIX/sample.json`. The full-admission guard checks the
sample campaign's source identities and outcomes. A changed grader requires new
verification evidence; an old sampled pass cannot certify it.

## Outputs

Each source artifact keeps these paths together beneath its dataset prefix:

| Relative path | Contents |
|---|---|
| `hf/` | Pinned input manifest and raw locator ledger |
| `normalized/part-*.parquet` | TaskSpecs and conversion outcomes in canonical shards |
| `analysis/part-*.parquet` | Review evidence, decisions and identity fields |
| `analysis/unprocessed-*.parquet` | Records outside an unexpanded sample and gate reasons |
| `verification/part-*.parquet` | Per-task controls and grader readiness |
| `accepted/part-*.parquet` | Rows admitted by the final publication decision |
| `report.json` | Source decisions, revisions, counts and output links |

Sidecars join on `task_id`, `source_locator`, `raw_input_sha256` and decoded
`raw_sha256`. Rejected and deferred rows remain auditable. Successful procedures
remove redundant scratch payloads; failed procedures retain diagnostics.

To add a source, create a declaration under the appropriate `datasets/<family>/` directory with a `pipeline()` factory,
reuse a family converter or add a concrete one in `taskcompendium.datasets`, then
add its explicit entry to `sources.py`. Preserve private answers and fixtures,
pin runtime requirements and run the source through the sample campaign before
full processing. The
[library rubric](https://github.com/marin-community/marin/blob/main/lib/taskcompendium/src/taskcompendium/pipeline/README.md)
describes family and verification requirements.
