# Synthetic biology task generation

The generators in `experiments/post_training/bio_tasks/` implement 140
recipes from [the biology data program](https://github.com/marin-community/marin/issues/9257).
They combine independently sourced real observations with synthetic correctness controls, establish references,
execute separate input-reading oracle solutions, and package tasks for Harbor.
Generation and grading make no model calls. The source inventory retains all 50
inspected repositories, including pinned evidence, verification limitations, and
the original 2026-07-28 downloads/stars/citations. The maintained
[task catalog](bio-task-catalog.md) contains all source assessments, skills, and
format coverage, with the original adoption inventory preserved as a separate file.

The [implemented recipe list](bio-task-recipes.md) records every operation, skill,
format profile, and repository mapping. The 140 recipes span 13 domains:

| Domain | Recipes |
|---|---:|
| sequence | 24 |
| genomic intervals | 10 |
| expression | 17 |
| sequencing reads | 18 |
| variants | 9 |
| phylogeny | 12 |
| assembly and ecology | 8 |
| imaging and spatial | 7 |
| statistics | 11 |
| structures and proteomics | 9 |
| networks | 6 |
| assays and metabolomics | 7 |
| workflow and identifiers | 2 |

The default build contains one task per recipe: 140 authoring examples, comprising
25 real-data candidates and 115 simulated controls. Add another task from a recipe
only when its dataset, study design, modality or scientific decision contributes
meaningful coverage. Deterministic generators can still produce extra validation
cases without adding them to training. The manifest marks
`corpus_stage=authoring-candidates-and-controls` and `training_ready=false`.
The final training release targets real biological data for every task. Synthetic
controls stay outside that release and remain available for verifier development.
Every released task requires a private executable oracle that reads only public
inputs, plus checks of alternative valid outputs and plausible scientific errors.
Real examples use the unchanged 27,179-gene, 12-sample GSE60450 count matrix and
experimental structures 1UBQ, 1CRN and 4HHB, complete annotated phage genomes
NC_001422.1/NC_001416.1/NC_001604.1, 6,000 observed paired reads from ERR266411,
complete curated UniProt globin and COX1 proteins, and the published Mayo PBC baseline
and longitudinal clinical tables.
The read source retains varying per-base qualities in three disjoint 2,000-pair
blocks; these are technical subsets, not biological replicates. Source files are vendored with hashes,
licenses and transformations in `data_sources.json`. Full benchmark lineage screening
and end-to-end workflow validation remain pending. See the
[ID workflow gap analysis](bio-task-catalog.md#id-workflow-coverage-and-input-realism)
for the distinction and the requirements for realistic inputs and artifact outputs.
Select tasks for realistic biological inputs, scientific decisions and connected
analysis stages. Record actual input scale and measured runtime; no difficulty
labels or quotas apply. Evaluation uses the independent benchmarks in the program
issue. No development or test split is generated.

Native formats now include GFF3 and GTF, BED12 and bedGraph, SAM and VCF,
Matrix Market/10x, Newick and NEXUS charsets, PDB and mmCIF, SBML, MGF,
PGM images, PAF, HMMER domain tables, FASTA indexes, and FastQC reports.
The profiles are explicitly bounded, rather than general-purpose parser support.
Generic CSV/JSON summaries do not establish native H5AD, BAM, SRA, BUSCO-output,
Kraken-report, or OME-TIFF coverage. In particular, `sra-spot-export` currently
models biological/technical read routing from a CSV ledger; it does not read an SRA archive.

Every source repository has a scientific-operation mapping in
`repository_coverage.json`. Actual CLI/API execution remains a distinct requirement
for all 50 repositories. 37 packages have passing checks: 34 on three reference cases each,
and IQ-TREE, FastTree and RAxML on one shared observed COX1 alignment. The
first CoreWeave run passed 22 packages; corrections on an existing TRC host passed
12 more, with captured outputs downloaded from regional GCS. Picard quality-yield and
fastp paired-filter checks pass on observed reads. MAFFT and MUSCLE passed three real-protein alignments each; the MAFFT channel
correction and earlier installation failure are retained. Thirteen other repositories
still need scripts. `native_validation.json` indexes separate checksum-pinned
files under `native_validation_runs/`; earlier failures and resolved environments
remain in that history. The three real-data recipes checked with MUSCLE, fastp and Picard now have
image-build contexts with the exact resolved package artifacts and checksums.
The two connected DESeq2 recipes also have locked R image contexts. Other recipes
currently use Python-only environments. Package checks alone do not
establish successful execution in these Harbor images. Repository source revisions and
runtime package versions are different provenance fields and must remain separate.

`strand-extraction` v2 uses GFF3 gene/mRNA/exon parents and a multirecord FASTA.
Separate recipes now cover multi-exon GTF splicing and GFF3 CDS phase, retaining
the format distinctions instead of converting every annotation to CSV.

## Build and inspect

The full GSE81682 candidate has 1,920 cells, 46,078 endogenous features and 92 ERCC
controls. Sparse conversion preserved all 22,590,142 nonzero counts and their
identities exactly. The 76,277,182-byte compressed MatrixMarket file and its
feature/cell tables are preserved in regional GCS; `data_sources.json` records
SHA-256 hashes and retrieval locations. This source is not yet a registered task.
Donor/pool reconciliation, redistribution terms and benchmark-lineage review
remain open. The broad sorting gates do not provide fine cell labels or donors.

Connected DESeq2 differential-expression and GO-enrichment candidates are defined in
`generators/real_rnaseq.py`. Their private fits use unchanged GSE60450 observations;
the public annotation snapshot preserves propagated biological-process membership
from org.Mm.eg.db and GO.db 3.22.0. Six native oracle cases pass on reserved TRC
CPUs, checking 16,659–17,361 fitted genes and 5,898–6,021 GO terms per case. They
are registered as authoring candidates. Both canonical Harbor cases now pass,
and both changed-intermediate controls fail while retaining correct summaries.
The R image selects `OPENBLAS_CORETYPE=HASWELL` before R starts; all four Harbor
runs reported that kernel. This fixes an earlier probability-field mismatch:
on the same TRC host, Zen and Haswell passed while Sandy Bridge failed two genes.
All numerical tolerances remain unchanged. These checks establish execution and
artifact grading; source independence and scientific review remain separate gates.
No benchmark inputs or model calls were used to prepare these references.

`sources/prepare_deseq.R` prepares all contrasts, fitted size factors and frozen
annotations on a CPU worker with R 4.5.3 and DESeq2 1.50.2. After verifying the
worker's output manifest, `sources/vendor_deseq.py` preserves the private reference
files and selects the public BP annotation snapshot. `data_sources.json` records
content hashes, package versions and annotation database hashes. The oracle
executes DESeq2's estimation stages from task inputs; it shares the statistical
engine with the preparation reference. The enrichment cross-check uses independent
SciPy and R probability implementations.

On a CPU host with the pinned R environment on `PATH`:

```bash
python -m experiments.post_training.bio_tasks.sources.validate_rnaseq \
  --output <new-output> --lock <resolved-packages.json> \
  --source-revision <commit> --seed 20260923 --instances 3
```

It checks complete QC, fitted-gene and enrichment tables as well as summaries and
negative controls. R validation requires remote compute on the shared authoring VM.

Build all recipes on a CPU host with the pinned R environment on `PATH` and
`OPENBLAS_CORETYPE=HASWELL` set before starting the build. The
measured native R peak exceeds 1 GiB; do not run the full build on a shared VM
with a 500 MiB workload limit. From the repository root:

```bash
uv run python -m experiments.post_training.bio_tasks.build \
  --output /tmp/bio-tasks-example \
  --instances-per-recipe 1 \
  --seed 20260923 \
  --base-image python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c \
  --tool-ref 12bbd5d45b1b176167ab3cfe905e06004c2f02e3
```

Use `--recipes <id> [<id> ...]` for an explicit smaller build, such as
`--recipes strand-extraction real-fastq-fixed-trim`. The manifest lists selected IDs
and the total registered count; benchmark mappings retain their full inventory,
with example links only for the selected recipes. Omission builds every recipe
and runs every oracle. It does not silently skip missing runtimes.

Recipes can declare `InputFile(sha256, size_bytes)` entries for compressed or
binary observations. Place each original file in a source cache under its full
SHA-256 filename and pass `--source-cache <directory>`. Validation and packaging
check both size and digest, preserve the bytes, and perform no downloads. Use
the source manifest to retrieve the pinned observations before building. The
ledger records every input's size and hash; the inspection page distinguishes
file inputs from inline text. Large source files belong in artifact storage,
with their provenance and hashes in Git. TaskTrove serialization still holds a
complete task bundle in memory, so size remote build workers accordingly.

Task observations live under `setup_files/inputs/`. The pinned Harbor runtime uploads
that directory into each fresh sandbox; `/app/inputs` points to it. Biological inputs
and private references therefore do not change the reusable image build context.
For package-enabled recipes, image building verifies every downloaded artifact
against the successful reference run, installs its explicit package set offline,
and removes the download cache. Agent and verifier execution both disable internet.
Image-build network access is separate from solve-time access.

The output directory must not exist. Generation runs one reference process
at a time. Each instance must pass its executable oracle, preserve row-order
invariance, and reject empty, malformed, duplicate-ID, missing-ID, and recipe-specific
scientifically wrong answers. FASTQ-producing tasks additionally check complete
ordered read IDs, sequences and qualities in submitted native files. Missing,
truncated and altered artifacts fail even when the JSON summary is correct. The
trusted verifier streams files under per-artifact byte limits and rejects symlinks.
Connected analyses can declare complete TSV intermediate artifacts, such as
fitted gene-level results. These contracts check every ID and quantity, declared
numeric tolerances and missingness. Row and column order may vary; wrong values,
duplicate or missing IDs, malformed rows and nonfinite values fail. A correct
final summary cannot compensate for an incorrect required intermediate table.
Integer count matrices can require complete MatrixMarket coordinate artifacts,
plain or gzip-compressed. Verification checks dimensions, every positive count
and its 1-based coordinate; entry order, whitespace and comments may vary.
Duplicate coordinates, explicit zeros, missing entries and changed counts fail.
The verifier streams decoded entries to disk and uses GNU sort with one worker
and a 32 MiB buffer to establish canonical coordinate order. Compressed, decoded
and line-size limits bound input expansion. Matrix tasks allow 300 seconds for
verification; full-study performance validation remains pending. Pair matrices
with checked feature/cell index tables to bind coordinates to biological identities.
Protein alignment tasks accept different aligned FASTA files if they preserve every
input residue and reach 95% of the sum of independently computed optimal pairwise
scores (twice BLOSUM62, gap opening 20, extension 1). Each admitted instance must
pass with a separate center-star solver. This objective verifies an alignment task;
it does not establish a true phylogeny or orthology.
Where adjacent instances have different targets,
copied JSON and native output files must fail against the next instance's contract. Target deduplication includes native artifact content. Duplicate input draws are deterministically resampled, with the actual seed and draw
number recorded; 64 unsuccessful draws stop generation. A reference mismatch stops
generation immediately and leaves the manifest marked `incomplete`.

Serve the output directory with `python -m http.server 8757 --directory /tmp/bio-tasks-example`
and open `http://localhost:8757/`. The index groups tasks by recipe, with text search, domain filtering,
format/skill labels, a data-origin filter (real data selected initially), and a separate
50-repository execution-coverage table. A second page tracks the build's frozen provisional ID task/protocol records and 90 held-out OOD
BioMysteryBench identifiers, with filters for distribution, benchmark and coverage status, recipe examples, explicit gaps and
local reference-check runtimes. Task pages show exact instructions, bounded input previews, expected
outputs, negative controls, metadata, and verifier code. These pages contain answers
and must remain outside solver environments. The bundle contains:

- `harbor/train/{task_id}/`: instructions, environment inputs, and private tests.
- `oracles/{task_id}/solution/`: independent input-reading solutions, stored separately.
- `reference-outputs/{task_id}/`: verified private JSON and native outputs for inspection; never mount for solvers.
- `tasks/part-00000.parquet`: TaskTrove browser columns and task/solution archives.
- `ledger.jsonl`: task identity, lineage, split, input formats, input/archive hashes, and validation results.
- `manifest.json`: source hashes, pinned base image and runtime references, counts,
  and readiness status.
- `source_inventory.json`: all 50 repository assessments and original adoption metadata.
- `data_sources.json`: biological accessions, source licenses, content hashes and transformations.
- `benchmark_coverage.json` and `benchmark-coverage.html`: pinned ID/OOD task inventory, mapping gaps and example evidence.
- `benchmark_tasks/`: per-benchmark task records and workflow-pattern indexes, with public versus advertised counts.
- `benchmark_sources.json` and `benchmark-sources.html`: all 48 agentic benchmark/protocol rows, provisional ID/OOD policy and source-inspection status.
- `native_validation.json` and `native_validation_runs/`: indexed, checksum-pinned package execution evidence.
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

Tasks request a 1,800-second solver limit and offline execution. Inputs are staged
from `setup_files/inputs/`; references and verifier code live under `tests/`.
Harbor creates a separate verifier sandbox and transfers `/app/answer.json` plus
every declared native output artifact for grading. Use Harbor revision
`d072bef08e54050880b484eb81d892944d1d82fb` or a verified compatible version that
supports this handoff and enforces `allow_internet=false` for the selected backend.

The solver image requires Python 3.12, pip, venv, and a shell. The separate verifier
image installs pinned Pydantic, TOMLKit, and TaskTrove verifier code during image
build. Build requires network access; verification does not. Capture the resulting
image digests for the task-dataset release: a pinned base image and source revision
alone do not freeze all build-time transitive packages.

`validated_locally` means input-reading oracle and negative-control checks passed
on the host. It does not establish container execution, network isolation, or
scientific approval. Scientific review remains pending. The committed
[container evidence](../../experiments/post_training/bio_tasks/container_validation.json)
records passing oracle trials and rejected artifact controls at their recorded task
and grader hashes, including complete RNA-seq tables and COX1 trees and distances.
The COX1 validation passed all three cases; its enclosing worker failed on a
concurrent sandbox deletion, and a subsequent check confirmed no owned sandboxes
remained. A new build requires checks against its
own hashes. Before task release, review every selected task and validate its
reference/failing solutions through the selected Harbor backend. Keep additional
null, boundary and invalid-input cases in the validation suite.

The verifier checks every record ID, field, type, missing value, and quantity.
Integer counts are exact; proportions use the tolerances stated in the prompt.
Booleans and numeric strings are rejected. All checks must pass for reward 1.
A bad reference produces `invalid_task` and removes reward files; file-access
failures produce `infra_error`. Solver answer errors receive reward 0.

Teacher collection and successful-trace export are not part of this command.
The program's five-attempt GLM-5.3 protocol remains a later phase in the issue.
Task-dataset completion is based on ID workflow coverage, private executable oracles,
validated environments and scientific contracts, and screened biological provenance. No teacher or
training job is launched when building or inspecting this corpus.

## Real-package reference checks

The scripts in `native/` run a package CLI or API on the same public inputs and
translate its output into the existing answer contract. A separate reporting
process grades it against the private construction reference. This is an
integration check, not a model judge, replacement oracle, or proof of teacher
package use. Only recorded passing executions count as evidence; an unexecuted script or
package import does not establish tool coverage.

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
batch ran on CoreWeave with S3 outputs. Correction runs used a reserved TRC v4 host
in `us-central2`, requested no accelerators, and retrieved their archives from GCS.

Grade returned results with:

```bash
uv run python -m experiments.post_training.bio_tasks.native.report \
  --bundle /path/to/native-checks --results /path/to/results \
  --output /path/to/native-report.json
```

The report preserves the denominator of 50 repositories and requires all supplied
completed, distinct-input executions with passing biological answers per repo.

The 2026-09-23 run installed all 30 environments and completed 78 of 90 cases.
It used one CPU worker, serial environments, and no automatic retries or model
calls. Private expected answers remained local. The frozen contract rejected
10 completed cases; tolerances were unchanged. See the
[catalog's failure table](bio-task-catalog.md#package-reference-run) for the eight
packages that did not pass all requested cases. The earlier five host-package checks
remain in the evidence file as separate runs with their original environments.
