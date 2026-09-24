# Biology workflow authoring queue

The [full task-design plan](../../experiments/post_training/bio_tasks/workflow_plan.json)
contains **147 proposed questions across 52 scientific workstreams**, with review
assignments for all **2,881 inventoried task, protocol or dataset-definition records
across 30 ID releases**. It records exact task IDs, input designs, required artifacts,
verification approaches and unresolved data/endpoint requirements. These are
proposals; they add **zero validated mappings**. Twelve provisional-ID source leads still
lack task inventories and remain explicit access or enumeration gaps.

Choose the next coherent task by its expected gain in currently missing, distinct
ID benchmark tasks. Recompute after each completed or rejected candidate.
Repository coverage is secondary. The
[machine-readable queue](../../experiments/post_training/bio_tasks/workflow_queue.json)
retains assessed target IDs, conditional gains and overlap; the
[coverage inventories](../../experiments/post_training/bio_tasks/benchmark_tasks/README.md)
remain authoritative for validated mappings.

The plan groups 551 benchmark/family combinations for review. Fifty known
BixBench-Verified aliases reduce the record count to 2,831, or 2,830 after the one
excluded objective. Other shared studies, assays and protocols are not automatically
independent workflows. Workstream membership alone does not show that a proposed
question completes a benchmark endpoint, and the proposed question count is not a
coverage forecast or a cap on authoring.

## Endpoint contracts

The [BiomniBench-DA designs](../../experiments/post_training/bio_tasks/workflow_designs/biomnibench-da.json)
refine all 50 public question endpoints into 45 proposed tasks. Each design specifies
observed inputs, proposed artifact tables and key columns, decisions to freeze, verifier checks and
unresolved data/runtime requirements. These refine the 147 broad questions; they
are not 45 additional tasks or validated mappings. The question-only review found
18 endpoints potentially matched by extending a broad question, 18 with only
component overlap and 14 requiring a new endpoint formulation.

Related endpoints share a design when one study can support the complete analysis.
Expression contrasts, WGCNA and distinct chromatin assays remain separate. Ambiguous
GWAS and survival-proportion questions retain their unresolved estimands. Negative
controls specify incorrect submitted artifacts; random label permutations are not
assumed to change significance. Independent input selection, complete schemas and
native reference execution remain required before authoring is considered ready.

## Solver diagnostics

[Inspect both GLM-5.3 traces](https://codex-main.exe.xyz:8757/solver-checks/20260924/index.html).
The [versioned report](../../experiments/post_training/bio_tasks/solver_diagnostics.json)
retains original grades, artifact comparisons, task hashes and a checksum-pinned
GCS archive. There was one attempt per unchanged task, high reasoning, 65,536 input
tokens, 32,768 output tokens per ordinary request and a 30-minute agent limit.
All 22 recorded inference requests used high reasoning and the output cap.

| Existing task | Agent time | Original reward | Trace and artifact finding |
|---|---:|---:|---|
| Observed PhiX read assembly | 2m32s | 1 | Runs SPAdes and minimap2; all summary fields, FASTA and TSV artifacts pass |
| Replicated RNA population interaction | 5m18s | 0 | Runs the specified DESeq2 interaction model; six correct values appear in six summary objects instead of one. All seven TSVs match in a separate read-only comparison |

The RNA comparison does not change its original grade. Both traces show intended
scientific tool use within the budget; two selected attempts cannot establish
benchmark accuracy, general solvability or training benefit. The server did not
expose its maximum context length or weight revision, and the rendered chat
template was not inspected. No LLM judge or bulk teacher collection was used.

The generator now shows a value-free record example and makes the RNA task's single
`comparison` record explicit. Verification reports artifact checks even when the
summary fails, preserving reward zero. This incurs the usual artifact read/sort
cost even for a malformed summary; existing file bounds and the 60-second verifier
limit (300 seconds for matrix tasks) still apply. Infrastructure failures remain
unscored. Original attempts and grades are unchanged.

Native environment contexts now export `/opt/bio/bin` from `/etc/profile.d/bio.sh`
as well as Docker `ENV`. The pinned Harbor terminal starts `bash --login`; a
fresh container/terminal check of this change is still required before release.
Two model-free follow-up workers failed in the validation launcher before any
sandbox was created (missing local module, then unsupported custom-agent constructor).
Their archives and failure records are retained in the diagnostics report. A corrected
check is prepared; a further launch awaits approval under the Iris repeated-failure
rule. These failures add no task grades and do not change the two GLM results.

## Proposed scientific workstreams

Each row points into the full JSON plan. Each proposed question should become one
coherent task on independent observations; questions can be split or merged after
scientific endpoint and runtime review. The count does not imply that inputs or
oracles are ready. Examples below are proposals, not benchmark answers.

| Workstream | Proposed questions | Example scientific question |
|---|---:|---|
| Observed ortholog evolution (`ortholog-tree-signal`) | 3 | Which observed loci change evolutionary signal between clades after alignment QC? |
| Replicated bulk expression (`rna-seq-contrast-enrichment`) | 4 | Does the condition effect differ between populations under an interaction model? |
| Replicated perturbation screens (`crispr-screen-pathway`) | 2 | Which gene-level hits remain after guide QC and control calibration? |
| Read-backed microbial variation (`bacterial-read-variant`) | 3 | Which variants remain after a complete read-QC, alignment, calling and annotation workflow? |
| Clinical association and response (`clinical-response-model`) | 3 | Which baseline covariates associate with a binary response after prespecified adjustment? |
| Reference and transcript integrity (`transcript-reference-audit`) | 3 | Which transcript sequences and promoter intervals are consistent with strand, exon phase and reference release? |
| Single-cell representation diagnostics (`single-cell-qc-embedding`) | 3 | Which cells and genes remain after explicit QC, and how does normalization affect the representation? |
| Cell identity and expression programs (`single-cell-annotation`) | 3 | Which cell identities are supported by marker and reference evidence, including ambiguous cells? |
| Replicate-aware cell composition (`single-cell-composition`) | 3 | Does a cell population change in abundance across conditions at the donor level? |
| Cell-type expression with biological replication (`single-cell-pseudobulk`) | 3 | Which within-cell-type expression effects survive donor-aware pseudobulk testing? |
| Developmental dynamics and regulons (`single-cell-trajectory`) | 3 | Which state ordering is consistent with measured time and branch-specific gene programs? |
| Expression-supported communication (`cell-communication`) | 2 | Which ligand-receptor associations change between conditions after expression and replicate checks? |
| Measured spatial organization (`spatial-expression`) | 3 | Which anatomical regions show reproducible spatial expression gradients? |
| Spatial reference integration (`spatial-reference-transfer`) | 2 | Can a single-cell reference explain spatial mixture proportions without confusing missing genes with zero expression? |
| Accessibility and regulatory context (`chromatin-regulatory`) | 3 | Which accessible regions and motifs change after sample-aware QC and peak calling? |
| Coverage-aware methylation integration (`methylation-expression`) | 3 | Which methylation differences remain after coverage and sample-level checks? |
| Replicated binding and chromatin contacts (`differential-binding`) | 2 | Which binding differences survive control-aware peak calling and replicated testing? |
| Observed assembly and metagenomic checks (`assembly-qc`) | 3 | How do assembly contiguity, reference agreement and read support differ? |
| Genotype structure and inheritance (`population-genetics`) | 3 | Which population-frequency and differentiation results survive genotype QC? |
| Variant consequence and evidence audit (`variant-evidence`) | 3 | Which coding or splice consequences follow from reference-consistent variant normalization? |
| Observed molecular property prediction (`molecular-property`) | 3 | Which descriptor model predicts a measured assay without scaffold or duplicate leakage? |
| Experimental structure and binding evidence (`ligand-structure-affinity`) | 3 | Which ligand identity and contact claims survive assembly and alternate-location checks? |
| Target prioritization with traceable evidence (`target-evidence`) | 3 | Which targets remain supported after direction-aware cross-source evidence joins? |
| Observed time-to-event analysis (`clinical-survival`) | 2 | Do survival contrasts persist after censoring, baseline eligibility and proportional-hazards checks? |
| Observed assay response and controls (`dose-response`) | 3 | Which dose-response effects remain after control normalization and fit diagnostics? |
| Observed abundance and cross-omics effects (`proteomics-metabolomics`) | 3 | Which paired protein-abundance changes remain after normalization and missingness checks? |
| Observed immune repertoire and state (`immune-repertoire`) | 2 | Which clonotypes and chain-quality differences remain after sample-aware repertoire QC? |
| Measured ecological associations (`ecological-regression`) | 2 | Which trait associations survive phylogenetic, spatial and sampling-effort adjustment? |
| Observed sequence and construct accounting (`sequence-construct`) | 3 | Which annotated sequence, ORF and translated products agree with strand and frame conventions? |
| Frozen database and graph reconciliation (`database-identity-evidence`) | 3 | Which accession and annotation discrepancies are explained by version or species differences? |
| Measured assay validity and longitudinal evidence (`clinical-safety-evidence`) | 2 | Which exposure and biomarker comparisons remain valid after timing, compartment and missingness checks? |
| Single-cell batch and modality integration (`single-cell-integration`) | 3 | Does batch correction preserve measured biological condition differences? |
| Gene programs, feature selection and networks (`gene-program-factorization`) | 3 | Which selected features remain stable after depth, distribution and neighborhood checks? |
| Observed sequence predictor evaluation (`protein-sequence-model-evaluation`) | 4 | Do sequence-model scores predict measured substitution effects beyond a baseline? |
| Measured dynamical-system inference (`observed-dynamics`) | 3 | Which mechanistic model predicts held-out observed trajectories? |
| Observed biological image analysis (`biological-imaging`) | 3 | Which cell or colony measurements remain after segmentation and scale QC? |
| Observed physiological signal analysis (`physiological-timeseries`) | 3 | Which event or condition effects survive signal QC and subject-aware aggregation? |
| Structured evidence and meta-analysis (`evidence-meta-analysis`) | 3 | Which quantitative findings remain after study selection, unit conversion and duplicate-cohort checks? |
| Biological data preparation and statistical communication (`data-statistics-visualization`) | 3 | Which joins and exclusion decisions change the study population and reported denominators? |
| Observed RNA annotation and structure analysis (`small-rna-sequence-structure`) | 3 | Which small-RNA and target-expression changes agree after identifier and condition matching? |
| Matched clinical and multiomic inference (`multimodal-clinical-discovery`) | 3 | Which cross-modal associations persist after patient and timepoint matching? |
| Observed microbial community analysis (`metagenomic-community`) | 3 | Which taxonomic and functional profiles remain after contamination and ambiguity checks? |
| Biological network and model consistency (`network-model-evidence`) | 3 | Which network paths and modules are supported by the measured edge evidence? |
| Observed clonal barcodes and cell lineage (`cell-lineage-barcodes`) | 3 | Which clonal identities remain after read, barcode and allelic-dropout checks? |
| Reference-aware genomic intervals (`genomic-interval-operations`) | 3 | Which regulatory annotations overlap the observed variants when intervals and genome builds are reconciled? |
| Tissue optical transport (`optical-fluence`) | 2 | How does predicted fluence vary across measured tissue conditions under the declared transport model? |
| Read-backed repeat genotyping (`repeat-genotyping`) | 3 | Which repeat alleles are supported by spanning reads after locus and motif reconciliation? |
| Cohort variant and burden analysis (`clinical-variant-cohort`) | 3 | Which cohort differences remain after variant/sample QC and declared covariate adjustment? |
| Repeat expression and transcript splicing (`repeat-expression-splicing`) | 3 | Which repeat families change expression under explicit multimapping and replication rules? |
| Differential chromatin contacts (`chromatin-contact`) | 3 | Which contact changes survive balancing, distance adjustment and replicate-aware testing? |
| Chemical diagram transcription (`chemical-diagram-transcription`) | 2 | Which molecular graphs can be recovered from observed structure diagrams after atom/bond normalization? |
| Biological image reconstruction (`inverse-biological-imaging`) | 2 | How do declared reconstruction methods recover the observed biological volume under the same acquisition model? |

## Endpoint review and selection

Keep every unresolved source category in the denominator. In particular, the 350
SciGym system records require an observed-data suitability review; their current
simulations do not supply real biological observations. BioML assay and cross-validation
definitions need assay-specific label, leakage and equivalence checks. Database-query
IDs need exact resource and evidence contracts. Open-ended scientific plans need
an executable acceptance rule that preserves their meaning; a generic JSON template
cannot supply one. Gated prompts and unavailable source inputs remain unresolved.

The initial ranking below compares only seven assessed candidates. Counts are
conditional targets and can shrink after data or endpoint review. Related protocol
versions add no ranking credit. Review the full portfolio before choosing an
implementation, and use data suitability and effort to break ties.

| Order | Connected workflow | Distinct missing targets | Principal gap before authoring |
|---:|---|---:|---|
| 1 | Multi-locus ortholog alignment and evolutionary-signal comparison | 46 | Independent matched orthologs across two clades; native PhyKIT definitions and aggregation checks |
| 2 | Replicated single/double perturbations across media, with directional pathway overlap | 11 | Independent intervention study with the required factorial design and frozen pathway annotations |
| 3 | Single-cell QC, normalization, embedding and group diagnostics | 8 | Native h5ad, mitochondrial annotation and endpoint-specific representation checks |
| 4 | Observed bacterial reads through alignment and variant comparison | 8 | Independent isolate reads and the required trimming/alignment/calling pipeline |
| 5 | Prespecified binary clinical-response models | 7 | Independent response cohort with the required covariates |
| 6 | Transcript structure and strand-aware reference audit | 6 | Complete matching eukaryotic reference and annotation |
| 7 | Native BAM read-structure and eligibility audit | 2 | Overlaps the broader resequencing candidate; remove these targets if it completes first |

The multi-locus candidate's 46 targets have 13 related BixBench-Verified-50 records,
which require separate protocol checks but add no rank credit. An additional
endpoint needs naturally observed alignments with more than 70% gaps and receives
no credit until suitable data are confirmed. The pinned Medusozoa author mirror
has 20 verified alignment/tree pairs; 16 meet the provisional clade-size requirement
but represent only seven original orthogroups. Native PhyKIT metric definitions and
aggregation remain under review. The existing single-gene COX1 task does not
establish multi-locus coverage.

Keep each task scientifically coherent, based on independent observed inputs,
offline and executable within 30 minutes. Promote a mapping only after required
stages, complete artifacts, an input-reading oracle and container checks pass.
Internal prediction holdouts remain inside a task; the corpus has one train split.

Use at most one bounded worker alongside the lead. The lead owns scientific
design, source suitability, ranking, oracle independence and coverage decisions.
Delegate factual source search, pinned installation and routine execution or
artifact checks to GPT-6 Sol; simpler mechanical work can use GPT-6 Luna. Assign
explicit outputs, allowed actions and a stopping point. Review evidence before
integration, and avoid simultaneous heavy local work.

Completed: the observed PhiX assembly workflow validates an N50-specific CompBioBench
mapping. Its original Drosophila fixture and answer were not used; release-level
scientific and biological-lineage review remain open. The issue's
[opening ledger](https://github.com/marin-community/marin/issues/9257) tracks task,
validated mapping and repository counts separately from this prospective plan.
