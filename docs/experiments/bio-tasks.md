# Synthetic biology task generation

The generators in `experiments/post_training/bio_tasks/` implement 121
recipes from [the biology data program](https://github.com/marin-community/marin/issues/9257).
They combine independently sourced real observations with synthetic correctness controls, establish references,
execute separate input-reading oracle solutions, and package tasks for Harbor.
Generation and grading make no model calls. The source inventory retains all 50
inspected repositories, including pinned evidence, verification limitations, and
the original 2026-07-28 downloads/stars/citations. The maintained
[task catalog](bio-task-catalog.md) contains all source assessments, skills, and
format coverage, with the original adoption inventory preserved as a separate file.

The [implemented recipe list](bio-task-recipes.md) records every operation, skill,
format profile, and repository mapping. The 121 recipes span 13 domains:

| Domain | Recipes |
|---|---:|
| sequence | 17 |
| genomic intervals | 10 |
| expression | 15 |
| sequencing reads | 13 |
| variants | 9 |
| phylogeny | 10 |
| assembly and ecology | 8 |
| imaging and spatial | 7 |
| statistics | 8 |
| structures and proteomics | 9 |
| networks | 6 |
| assays and metabolomics | 7 |
| workflow and identifiers | 2 |

Use three instances with distinct inputs and reference targets per recipe: 363 train tasks at this checkpoint.
There are 18 real-data examples and 345 simulated controls. The manifest marks
`corpus_stage=authoring-candidates-and-controls` and `training_ready=false`.
Real examples use the unchanged 27,179-gene, 12-sample GSE60450 count matrix and
experimental structures 1UBQ, 1CRN and 4HHB. Source files are vendored with hashes,
licenses and transformations in `data_sources.json`. Full benchmark lineage screening
and end-to-end workflow validation remain pending. See the
[ID workflow gap analysis](bio-task-catalog.md#id-workflow-coverage-and-input-realism)
for the distinction and the requirements for realistic inputs and artifact outputs.
These are easy and medium tasks; hard compositions and the final difficulty
mixture remain separate work. Evaluation uses the independent benchmarks in the
program issue. No development or test split is generated.

Native formats now include GFF3 and GTF, BED12 and bedGraph, SAM and VCF,
Matrix Market/10x, Newick and NEXUS charsets, PDB and mmCIF, SBML, MGF,
PGM images, PAF, HMMER domain tables, FASTA indexes, and FastQC reports.
The profiles are explicitly bounded, rather than general-purpose parser support.
Generic CSV/JSON summaries do not establish native H5AD, BAM, SRA, BUSCO-output,
Kraken-report, or OME-TIFF coverage. In particular, `sra-spot-export` currently
models biological/technical read routing from a CSV ledger; it does not read an SRA archive.

Every source repository has a scientific-operation mapping in
`repository_coverage.json`. Actual CLI/API execution remains a distinct requirement
for all 50 repositories. A remote CPU run checked 30 packages on three instances
each: 22 passed all three, four had answer mismatches, and four failed execution.
Across the 90 cases, 68 passed, 10 produced mismatching answers, and 12 failed
execution. The other 20 repositories still need reference scripts. Results,
including failures, are recorded in `native_validation.json`. The current generated solver environment contains Python;
it does not yet establish native tool execution. Repository source revisions and
runtime package versions are different provenance fields and must remain separate.

`strand-extraction` v2 uses GFF3 gene/mRNA/exon parents and a multirecord FASTA.
Separate recipes now cover multi-exon GTF splicing and GFF3 CDS phase, retaining
the format distinctions instead of converting every annotation to CSV.

## Build and inspect

From the repository root:

```bash
uv run python -m experiments.post_training.bio_tasks.build \
  --output /tmp/bio-tasks-example \
  --instances-per-recipe 3 \
  --seed 20260923 \
  --base-image python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c \
  --tool-ref 12bbd5d45b1b176167ab3cfe905e06004c2f02e3
```

The output directory must not exist. Generation runs one small reference process
at a time. Each instance must pass its independent oracle, preserve row-order
invariance, and reject empty, malformed, duplicate-ID, missing-ID, and recipe-specific
scientifically wrong answers. Where adjacent instances have different targets,
copied answers must fail. Duplicate input draws are deterministically resampled, with the actual seed and draw
number recorded; 64 unsuccessful draws stop generation. A reference mismatch stops
generation immediately and leaves the manifest marked `incomplete`.

Serve the output directory with `python -m http.server 8757 --directory /tmp/bio-tasks-example`
and open `http://localhost:8757/`. The index groups three examples per recipe, with text search, domain filtering,
format/skill labels, a data-origin filter (real data selected initially), and a separate
50-repository execution-coverage table. A second page tracks all 455 ID task identifiers,
with filters for benchmark and coverage status, recipe examples, explicit gaps and
local reference-check runtimes. Task pages show exact instructions, bounded input previews, expected
outputs, negative controls, metadata, and verifier code. These pages contain answers
and must remain outside solver environments. The bundle contains:

- `harbor/train/{task_id}/`: instructions, environment inputs, and private tests.
- `oracles/{task_id}/solution/`: independent input-reading solutions, stored separately.
- `tasks/part-00000.parquet`: TaskTrove browser columns and task/solution archives.
- `ledger.jsonl`: task identity, lineage, split, input formats, input/archive hashes, and validation results.
- `manifest.json`: source hashes, pinned base image and runtime references, counts,
  and readiness status.
- `source_inventory.json`: all 50 repository assessments and original adoption metadata.
- `data_sources.json`: biological accessions, source licenses, content hashes and transformations.
- `benchmark_coverage.json` and `benchmark-coverage.html`: pinned ID task inventory, mapping gaps and example evidence.
- `native_validation.json`: retained evidence for the completed package reference checks.
- `repository_coverage.json`: explicit recipe mappings and CLI/API execution evidence status.

The Parquet export can be opened with TaskTrove's file browser. These tasks use
Harbor's native `tests/test.sh` entry point in a separate verifier environment;
they do not supply a TaskTrove `verifier.toml` for the generic conversion validator.
TaskTrove/EvalDash service ingestion, attempt joins, and review annotations are
not implemented here. The local pages make the first generated examples inspectable.

Generation lineage IDs track related tasks and support deduplication. Real examples
also carry `biological_sources` and `biological_lineages`; changing a query or seed
does not make the underlying study independent for benchmark exclusion.
Recipe version changes preserve lineage and the initial input seed. Deterministic
resampling records any replacement generation seed explicitly. Perturbations of an existing
dataset must retain that dataset's lineage when additional generation routes are
added. The generator does not reserve development or test tasks.

## Execution boundary

Tasks request a 1,800-second solver limit and offline execution. Inputs are copied
from `environment/inputs/`; references and verifier code live under `tests/`.
Harbor creates a separate verifier sandbox and transfers `/app/answer.json` for
grading. Only that answer artifact is requested. Use Harbor revision
`d072bef08e54050880b484eb81d892944d1d82fb` or a verified compatible version that
supports this handoff and enforces `allow_internet=false` for the selected backend.

The solver image requires Python 3.12, pip, venv, and a shell. The separate verifier
image installs pinned Pydantic, TOMLKit, and TaskTrove verifier code during image
build. Build requires network access; verification does not. Capture the resulting
image digests before teacher collection: a pinned base image and source revision
alone do not freeze all build-time transitive packages.

`validated_locally` means input-reading oracle and negative-control checks passed
on the host. It does not establish container execution, network isolation, or
scientific approval. Both `container_validation` and `scientific_review` remain
`pending`. Before collection, review at least three varied examples per recipe
and validate reference/failing solutions through the selected Harbor backend.

The verifier checks every record ID, field, type, missing value, and quantity.
Integer counts are exact; proportions use the tolerances stated in the prompt.
Booleans and numeric strings are rejected. All checks must pass for reward 1.
A bad reference produces `invalid_task` and removes reward files; file-access
failures produce `infra_error`. Solver answer errors receive reward 0.

Teacher collection and successful-trace export are not part of this command.
The program's five-attempt GLM-5.3 protocol remains in the issue. No teacher or
training job is launched when building or inspecting this corpus.

## Real-package reference checks

The scripts in `native/` run a package CLI or API on the same public inputs and
translate its output into the existing answer contract. A separate reporting
process grades it against the private construction reference. This is an
integration check, not a model judge, replacement oracle, or proof of teacher
package use. The 30 implemented scripts include unvalidated options and version
assumptions; only recorded passing executions count as evidence.

Prepare checks from a validated corpus without installing packages:

```bash
uv run python -m experiments.post_training.bio_tasks.native.prepare \
  --corpus /path/to/validated-corpus --output /path/to/native-checks
```

On an explicitly provisioned worker, `native.remote` accepts `--bundle`,
`--output` and `--micromamba`. It creates one isolated package environment at a
time, retains its explicit dependency list and command logs, and makes no model
calls. Missing scripts remain pending. This worker command is not suitable for
the shared development VM. The currently published environment candidates are
not dependency locks or verified task images; retain resolved environments and
build reproducible solver images after validation.

Stage the bundle and package-manager executable in a writable, executable task
work directory. The worker creates package environments beside the bundle and
points package temporary files there. On Iris's GCP Docker runtime, executing the
package manager from `/tmp` failed with `PermissionError`; placing executable
environments there is unsuitable. Package caches remain outside captured outputs
and are removed when the worker exits.

Prefer available TRC/GCP host CPUs for subsequent package-validation batches,
with bounded CPU, memory, disk, and runtime requests. Use GCS in the worker's
region for staged inputs and retained outputs. Iris can place CPU-only work on
TPU hosts, but its GCP configuration also has an on-demand CPU fallback; record
actual placement before attributing a run to TRC capacity. The first 30-package
batch ran on CoreWeave and its captured outputs were retrieved from S3.

Grade returned results with:

```bash
uv run python -m experiments.post_training.bio_tasks.native.report \
  --bundle /path/to/native-checks --results /path/to/results \
  --output /path/to/native-report.json
```

The report preserves the denominator of 50 repositories and requires three
completed, distinct-input executions with passing biological answers per repo.

The 2026-09-23 run installed all 30 environments and completed 78 of 90 cases.
It used one CPU worker, serial environments, and no automatic retries or model
calls. Private expected answers remained local. The frozen contract rejected
10 completed cases; tolerances were unchanged. See the
[catalog's failure table](bio-task-catalog.md#package-reference-run) for the eight
packages that did not pass all three cases. The earlier five host-package checks
remain in the evidence file as separate runs with their original environments.
