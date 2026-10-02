# Task curation

The task curation pipeline downloads pinned source files, normalizes their records,
checks grading contracts, asks GLM for a source-specific quality assessment, and
writes final decisions to Parquet. Every selected input remains in the audit,
including rejected tasks and failed reviews.

The artifact graph lives in
[`experiments/post_training/task_curation/pipeline.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/pipeline.py).
Reusable readers, normalizers, checks and review logic live under
`lib/taskcompendium/src/taskcompendium/pipeline/`.

## Run a source

Plan a ten-record run without downloading data or contacting GLM:

```bash
uv run --with './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.pipeline \
  --source math500 --limit 10 --model-revision YOUR_GLM_REVISION
```

Add `--run`, `--base-url URL` and `--review-cache PATH` to execute. Set
`GLM_BULK_TOKEN` in the execution environment. Artifact outputs use `MARIN_PREFIX`;
set it to the desired S3 prefix or a local directory. The same graph runs in either
location. Credentials are excluded from artifact configuration and fingerprints.

Repeat `--source` to select more sources. `--limit N` caps selected input records
**per source**, including records later rejected. `--all-rows` removes that cap.
Executable source controls also require `--image` with an immutable image digest.
A missing runtime binding remains explicit; static GLM acceptance does not certify
that a grader is executable.

## Stages

1. **Download.** Existing `hf_download`/`raw_download` artifact builders stage the
   declared source files at their pinned revision. Related selections share the
   download when they use the same files. Download identity is independent of N.
2. **Read and limit.** Shared readers decode the staged format and select the
   requested component. The audit uses
   `reshard(1).take_per_shard(N).reshard(64)` before normalization. On one shard,
   `take_per_shard` imposes the overall source limit. This deliberately simple
   implementation can read and reshuffle the complete source before truncation.
   A source with fewer than N records completes with those records.
3. **Audit.** Normalization preserves the public/private boundary and records
   edits or import failures. Duplicate and conflicting references are identified.
   Checks and GLM findings are recorded independently.
4. **Filter.** Policy produces a final keep/reject decision and reasons. Bad,
   conflicting, invalid or unavailable reviews do not admit tasks. Confidence
   thresholds are policy; they do not create a separate human-review queue.
5. **Optional rewrite.** Explicitly selected tasks from the filtered audit receive
   a separate repair rubric. Candidate instructions are checked and reviewed
   again before a new filtering decision. Original evidence and candidate
   lineage remain available.
6. **Merge.** Cross-source canonicalization retains all audit rows, removes exact
   duplicates from accepted views, cuts typed reference conflicts and excludes
   training records that overlap evaluation tasks.

Source provenance uses the pinned dataset, revision, component/split and original
file/record locator. It does not depend on the position in a development sample.
Decoded records that fail normalization get audit rejections. File decoding errors
fail the stage with file context.

## Source families

A source does not need its own Python module. Sources sharing conversion and
review structure belong together, with source-specific criteria beside their
metadata. Each `DatasetRecipe` owns its `RecipeInputs`, normalizer, rubric, intended
use and check suite. `RecipeInputs` declares staged file selection and pinned
`HubDownload` or `UrlDownload` inputs, including auxiliary reference files. The
experiment translates those declarations into artifacts without source-name
acquisition switches.

Examples of source families:

| Family | Shared contract |
|---|---|
| `math_answers` | Ten typed-math sources, including MATH-500 and Hendrycks MATH; named extraction functions retain schema differences |
| `numeric_answers` | AIME24 and SVAMP exact-numeric answers |
| `instruction_tasks` | Direct Nemotron IF and RLVR IFEval records |
| `code_contracts` | APPS, Eurus2 and VerifiableCode with retained test contracts |
| `python_tasks`, `atlas_code`, `executable_tasks` | Archived executable tasks with shared runtime checks |
| `preference_tasks`, `repository_tasks`, `rubric_tasks` | Related source schemas with explicit source-specific criteria |
| `nemotron_ultra/` | Seventy-five selections grouped by reward family; pinned blend and auxiliary inputs |

Family `RECIPES` mappings contain static recipes. Factories remain for sources
that need runtime images or converter adapters. Existing TaskTrove conversion
adapters are supplied by the experiment; library families compose them with
normalization and preserve conversion edits. Grading rules live in `lib/verifyit`: ARC grids, injection actions, schedule and
calendar checks, capture comparison, reference/abstention gates, puzzles, IFEval
and JSON Schema. Taskcompendium adapters extract submissions, validate private
configuration and translate scoring results. Environment capture and isolated
execution remain taskcompendium responsibilities. VerifyIT does not import
TaskSpec or taskcompendium. Shared format readers remain separate. SQL and structured tool actions retain their distinct contracts.

For example, the Python-test family owns the `pymethods` and `pymethods_large`
source definitions. Both use the same converter and common privacy/test criteria.
The former adds scheduling and optimization checks; the latter adds missing public
signatures and class-state checks. Those differences remain in their own rubric
entries. A new variant adds a source entry and any needed criteria, not another
wrapper module.

To add a source:

- Declare its immutable release, split/component and staged file selection in the
  owning family. Reuse a format reader and archive decoder where possible.
- Reuse the family normalizer, or add a concrete extraction function when the row
  shape or grading contract differs. Keep private solutions and tests private.
- Specify source-specific rubric additions, intended train/eval use, and any
  executable controls. Register the family entry with the experiment binding.
- Run the ordinary graph with a small limit. Inspect audit rows and reasons.

Keep distinct contracts explicit. HH pairs, binary KTO labels and generation-based
GenRM prompts need different handling. Interactive calendar episodes differ from
final-schedule JSON. Shared topic alone is not sufficient to share a normalizer.
The Atlas catalog records coverage and exclusion reasons; it is not another
executable source registry.

## Cache identity and retries

Download artifacts identify source bytes. Audit artifacts include source selection,
limit, recipe/stage revisions, rubric and model settings. Filter artifacts include
policy. Bump the relevant family or shared-stage revision when its behavior changes.
Adding an unrelated source does not require hashing the whole package.

Completed Zephyr shards are reused. FineStore stores schema-valid GLM completions
with the expected request identity, keyed by the exact submitted query and model
revision. Task provenance IDs are canonicalized out of cached review queries.
Changing only catalog layout does not require GLM inference; changing a prompt,
rubric, model or supplied environment inventory creates a different query.

A failed mapper can submit its unfinished review batch again. The audit records
the final result for each task. Earlier attempt files are debug logs; they do not
create extra rejected rows after a successful retry.

## Outputs

Each source has audited and filtered artifacts. One canonical artifact contains:

| Directory | Contents |
|---|---|
| `audit/` | Every selected input and its final decision |
| `accepted/` | Kept tasks after cross-source policy |
| `train/` | Accepted tasks intended for training |
| `eval/` | Accepted tasks intended for evaluation |
| `executable/` | Accepted tasks with ready graders, including evaluation tasks |

Training consumers must select training intent as well as any required grader
readiness. Each output is sharded Parquet. The canonical artifact is exposed
without copy-only export stages.

Audit columns include source provenance, `raw_json`, `task_json`, normalization
changes, check results, GLM quality/reference/confidence findings, `filter_status`
and `filter_reasons`. Rewrite columns retain the original task, edits, reason,
parent identity and lineage. `grader_readiness` is separate from quality. Original
source data and opaque evaluator contracts remain available for later binding.

## Instruction repair

Use `--rewrite-plan SOURCE plan.json` alongside the ordinary source selection.
The plan selects task IDs from an earlier audit and supplies a repair rubric:

```json
{
  "task_ids": ["TASK_ID_FROM_AUDIT"],
  "rubric": {
    "id": "structured-instruction-repair",
    "version": "1",
    "criteria": [
      "Remove contradictory formatting boilerplate while preserving the public schema and supplied facts."
    ]
  }
}
```

Repairs apply to a single user instruction. Tools, resources and verifier fields
remain unchanged. A proposal that invents missing information or changes the task
contract must be rejected. Accepted instruction edits receive new candidate IDs;
checks, original-aware quality review and filtering run again. Invalid, unchanged
or unavailable proposals retain the original task and its existing decision.

## Environment evidence

`--environment-inventory SOURCE inventory.json` adds scoped file evidence to that
source's review rubric. `EnvironmentInventory` records the environment identity,
origin, roots, paths and completeness. A declared file list and an observed live
filesystem inventory support different claims; keep that distinction explicit.
Neither establishes that a golden solution or source evaluator ran successfully.
