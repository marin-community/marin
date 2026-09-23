# Synthetic biology task generation

The generators in `experiments/post_training/bio_tasks/` implement 115
recipes from [the biology data program](https://github.com/marin-community/marin/issues/9257).
They construct fresh inputs, establish exact references from construction ledgers,
execute separate input-reading oracle solutions, and package tasks for Harbor.
Generation and grading make no model calls. The source inventory retains all 50
inspected repositories, including pinned evidence, verification limitations, and
the original 2026-07-28 downloads/stars/citations. The maintained
[task catalog](bio-task-catalog.md) contains all source assessments, skills, and
format coverage, with the original adoption inventory preserved as a separate file.

The [implemented recipe list](bio-task-recipes.md) records every operation, skill,
format profile, and repository mapping. The 115 recipes span 13 domains:

| Domain | Recipes |
|---|---:|
| sequence | 17 |
| genomic intervals | 10 |
| expression | 11 |
| sequencing reads | 13 |
| variants | 9 |
| phylogeny | 10 |
| assembly and ecology | 8 |
| imaging and spatial | 7 |
| statistics | 8 |
| structures and proteomics | 7 |
| networks | 6 |
| assays and metabolomics | 7 |
| workflow and identifiers | 2 |

Use three instances with distinct inputs and reference targets per recipe: 345 train tasks at this checkpoint.
The manifest marks `corpus_stage=small-authoring-fixtures` and `training_ready=false`.
These are small correctness fixtures, not realistic workflow coverage. See the
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
for all 50 repositories. Biopython, pysam and cutadapt each passed three host
reference checks, recorded in `native_validation.json`; 47 repositories remain
pending. Reference scripts are implemented for 30 repositories. The current generated solver environment contains Python;
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
format/skill labels, and a separate 50-repository execution-coverage table. Task pages show exact instructions, bounded input previews, expected
outputs, negative controls, metadata, and verifier code. These pages contain answers
and must remain outside solver environments. The bundle contains:

- `harbor/train/{task_id}/`: instructions, environment inputs, and private tests.
- `oracles/{task_id}/solution/`: independent input-reading solutions, stored separately.
- `tasks/part-00000.parquet`: TaskTrove browser columns and task/solution archives.
- `ledger.jsonl`: task identity, lineage, split, input formats, input/archive hashes, and validation results.
- `manifest.json`: source hashes, pinned base image and runtime references, counts,
  and readiness status.
- `source_inventory.json`: all 50 repository assessments and original adoption metadata.
- `native_validation.json`: retained evidence for the completed package reference checks.
- `repository_coverage.json`: explicit recipe mappings and CLI/API execution evidence status.

The Parquet export can be opened with TaskTrove's file browser. These tasks use
Harbor's native `tests/test.sh` entry point in a separate verifier environment;
they do not supply a TaskTrove `verifier.toml` for the generic conversion validator.
TaskTrove/EvalDash service ingestion, attempt joins, and review annotations are
not implemented here. The local pages make the first generated examples inspectable.

Lineage IDs track related tasks and support deduplication and benchmark exclusions.
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

Grade returned results with:

```bash
uv run python -m experiments.post_training.bio_tasks.native.report \
  --bundle /path/to/native-checks --results /path/to/results \
  --output /path/to/native-report.json
```

The report preserves the denominator of 50 repositories and requires three
completed, distinct-input executions with passing biological answers per repo.
