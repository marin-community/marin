# Synthetic biology task generation

The generators in `experiments/post_training/bio_tasks/` implement 12
recipes from [the biology data program](https://github.com/marin-community/marin/issues/9257).
They construct fresh inputs, establish exact references from construction ledgers,
execute separate input-reading oracle solutions, and package tasks for Harbor.
Generation and grading make no model calls. The source inventory retains all 50
inspected repositories, including pinned evidence, verification limitations, and
the original 2026-07-28 downloads/stars/citations. The maintained
[task catalog](bio-task-catalog.md) contains all source assessments, skills, and
format coverage, with the original adoption inventory preserved as a separate file.

| Recipe | Difficulty | Native input profiles | Required distinctions |
|---|---|---|---|
| `strand-extraction` | Easy | FASTA + GFF3 (single exon) | Sequence-ID and exon-parent joins, closed coordinates, reverse complement |
| `interval-overlap` | Easy | BED4 | Half-open coordinates, chromosome identity, overlap threshold, distinct peaks |
| `donor-counts` | Medium | CSV | Cell-ID joins, raw counts, donor replication, type/QC filters, empty donors |
| `cell-fractions` | Medium | CSV | Patient/specimen joins, cohort/QC selection, local barcodes, zero denominators |
| `paired-read-qc` | Medium | FASTQ (Phred+33) + JSON settings | Mate-ID joins, per-mate thresholds, inclusive boundaries |
| `genotype-alleles` | Medium | VCF 4.3 + CSV sample selection | FORMAT/GT, named sample columns, partial missingness, ploidy, called-allele denominator |
| `transcript-tpm` | Medium | CSV | Effective lengths, transcript/gene joins, decoys, TPM versus CPM |
| `alignment-sites` | Easy | Aligned FASTA + CSV inventory | Missing states, eligible sites, variable versus informative columns |
| `tree-branches` | Easy | CSV edge list | Root conventions, internal/terminal edges, branch lengths |
| `busco-summary` | Medium | CSV supplied hits | Ortholog identity, duplicate hit rows, complete/fragmented precedence |
| `taxonomy-counts` | Medium | CSV taxonomy/assignments | Direct/clade counts, ancestor overlap, unclassified denominator |
| `image-measurements` | Medium | JSON arrays | Supplied labels, disconnected pixels, physical pixel centers, axis order |

Author three instances per recipe and expand recipe breadth before repetitions.
These recipes are an initial authoring batch. They do not implement all planned
families, hard compositions, or the final 30/60/10 difficulty mixture. All generated
tasks belong to one train split. Evaluation uses the separate benchmarks in the
program plan. CSV/JSON intermediates do not establish native H5AD, Newick,
BUSCO-output, Kraken-report, or OME-TIFF coverage. Format profiles describe the
specific inputs generated here, not general-purpose parser support.

`strand-extraction` v2 supplies native GFF3 gene, mRNA, and exon records with
`ID`/`Parent` attributes and a multirecord FASTA. It replaces v1's CSV annotations;
lineage and generated transcript sequences are preserved. The profile has one
exon per mRNA. GTF attributes, multi-exon splicing, and CDS phase remain separate
coverage requirements.

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
copied answers must fail. A reference mismatch stops generation and leaves the
manifest marked `incomplete`.

Serve the output directory with `python -m http.server 8757 --directory /tmp/bio-tasks-example`
and open `http://localhost:8757/`. The index groups three examples per recipe and
shows input formats and skills. Task pages show exact instructions, bounded input previews, expected
outputs, negative controls, metadata, and verifier code. These pages contain answers
and must remain outside solver environments. The bundle contains:

- `harbor/train/{task_id}/`: instructions, environment inputs, and private tests.
- `oracles/{task_id}/solution/`: independent input-reading solutions, stored separately.
- `tasks/part-00000.parquet`: TaskTrove browser columns and task/solution archives.
- `ledger.jsonl`: task identity, lineage, split, input formats, input/archive hashes, and validation results.
- `manifest.json`: source hashes, pinned base image and runtime references, counts,
  and readiness status.
- `source_inventory.json`: all 50 repository assessments.

The Parquet export can be opened with TaskTrove's file browser. These tasks use
Harbor's native `tests/test.sh` entry point in a separate verifier environment;
they do not supply a TaskTrove `verifier.toml` for the generic conversion validator.
TaskTrove/EvalDash service ingestion, attempt joins, and review annotations are
not implemented here. The local pages make the first generated examples inspectable.

Lineage IDs track related tasks and support deduplication and benchmark exclusions.
Recipe version changes preserve lineage and input seed. Perturbations of an existing
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
