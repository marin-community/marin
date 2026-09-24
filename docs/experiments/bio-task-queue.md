# Biology workflow authoring queue

The [question portfolio](../../experiments/post_training/bio_tasks/workflow_portfolio.json)
accounts for all **4,332 inventoried task, protocol or dataset-definition records
across 31 ID releases**. It links 2,480 records to scientific question proposals,
links 50 known BixBench-Verified aliases to their original designs, preserves 350
SciGym systems and 1,451 ScholarQA-Bio questions as individual-endpoint review gaps,
and retains one excluded objective. Eleven provisional-ID source leads still lack task inventories.

The portfolio contains **676 question cards and 635 proposed task groups**, including
25 conditional combinations of question cards. These are planning units, with
substantial differences in source inspection depth. They add zero validated mappings
and do not set a target dataset size. Each combined question retains all member
contracts; incompatible inputs, methods or estimands require separate tasks.

The [workstream plan](../../experiments/post_training/bio_tasks/workflow_plan.json)
organizes these proposals under 147 broad questions across 52 scientific workstreams.
It records exact task IDs, required artifacts, verification approaches and unresolved
input/endpoint requirements. Broad topic membership does not establish coverage.

**Current phase: question planning and consolidation before further implementation.**
The portfolio records a 20-item review order and an implementation assignment
contract. No task or implementation worker is selected. Freeze independent observed
inputs, scientific estimands, native methods and complete deterministic acceptance
for the selected questions before assigning bounded task-creation work.

When authoring resumes, choose the next coherent task by its expected gain in currently missing, distinct
ID benchmark tasks. Recompute after each completed or rejected candidate.
Repository coverage is secondary. The
[machine-readable queue](../../experiments/post_training/bio_tasks/workflow_queue.json)
retains assessed target IDs, conditional gains and overlap; the
[coverage inventories](../../experiments/post_training/bio_tasks/benchmark_tasks/README.md)
remain authoritative for validated mappings.

The plan groups 552 benchmark/family combinations for review. Fifty known
BixBench-Verified aliases reduce the record count to 4,282, or 4,281 after the one
excluded objective. Other shared studies, assays and protocols are not automatically
independent workflows. Workstream membership alone does not show that a proposed
question completes a benchmark endpoint, and the proposed question count is not a
coverage forecast or a cap on authoring.

All 30 original proposed combinations have now received a scientific planning
review. Five were split completely and seven narrowed across the two review passes;
25 combinations remain conditional. Each retained combination records its biological
unit, complete artifact set, deterministic checks, incorrect-output checks and
unresolved input/method decisions. The individual question cards still have uneven
inspection depth, so this is not yet an implementation-ready portfolio.

Compound-response prediction, scGen expression transfer and CRISPR-response prediction
now remain separate. Breast-subtype mutations, post-treatment resistance and
cross-cancer burden also have separate questions. Gastrointestinal segmentation and
AMOS multi-organ segmentation retain their different modality/class requirements;
medical image tasks cannot inherit microscopy inputs. Protein sequence-identity
ranking is separate from database entry/attribute reconciliation.

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

The [BixBench endpoint designs](../../experiments/post_training/bio_tasks/workflow_designs/bixbench.json)
assign all 205 questions to 27 proposed workflows. Full gene, pathway, variant and
patient tables support related endpoints within one observed study. Distinct assays
and model families remain separate. Scientific departures are explicit: normalized
abundances are not converted into artificial read counts, a nonsignificant difference
does not establish equivalence, and static variant counts do not establish mutation
accumulation rates. Such corrected adaptations need endpoint review before any
coverage credit. The designs retain source-study exclusions and question hashes.

The [CompBioBench endpoint designs](../../experiments/post_training/bio_tasks/workflow_designs/compbiobench.json)
assign all 100 questions to 84 proposals, including supporting components for two
software-only endpoints and a subjective shape question. Six explicitly simulated
source questions need independently observed alternatives. Spatial tasks include
both deconvolution and neighborhood analysis; alignment tasks distinguish reads
from templates, and sequence checks include orientation as well as GC content.
Public transcript-model optimization starts from observed sequences and labels
its designed variants separately. Runtime, model weights and method equivalence
remain unresolved. Its JSON and TSV metadata hashes are recorded separately.

These three reviews assign 355 question endpoints. Their 156 detailed proposals
refine or split the 147 broad questions and overlap across benchmarks; they are
neither an independent task count nor a coverage forecast. Cross-benchmark merging
requires matching estimands, usable independent inputs and complete output contracts.
Other releases retain their recorded inspection limits in the full plan.

## BioAgent connected-workflow contracts

The [BioAgent contracts](../../experiments/post_training/bio_tasks/workflow_designs/bioagent.json)
retain all ten existing question IDs and proposals. Rechecking the original prompts
restored source-specific outputs missing from broad method-family descriptions.
Each card now specifies observed inputs, required stages, table keys/columns, native
oracle steps, independent checks, incorrect-output checks and runtime boundaries.
No new task or validated mapping is added.

| Source question | Required endpoint | Input or interpretation limit |
|---|---|---|
| Cross-model expression | Three model/control contrasts, complete pathway universes and a joined three-model result | One study split into three technical subsets is not three biological models. |
| Comparative genomics | Four-genome coding/annotation filters, conserved clusters, consensus functional annotation, alignments and trees | BUSCO alone does not identify all conserved functional clusters; gene trees alone do not establish co-evolution. |
| Recessive pedigree | Observed family segregation, complete transcript consequences and dated evidence | The source plants a disease allele; observed segregation cannot by itself prove disease causality. |
| RNA differential expression | Paired reads through alignment/counting to all DESeq2 effects and tests | Count-only inputs omit the source's upstream workflow. |
| Lineage evolution | Ancestor-relative alleles shared by both descendants and passing the consequence threshold | Unknown ancestral calls cannot be treated as absent alleles. |
| Exome calling | Diploid calls and representation-aware comparison within capture/callable/high-confidence regions | Bacterial inputs are incompatible; a region subset does not establish whole-exome performance. |
| Community profiling | Both observed communities, complete taxonomic profiles and explicit abundance denominators | Two unreplicated samples support description, not population-level treatment inference. |
| Single-cell response | QC, clustering, evidence-based labels and complete within-cell-type condition contrasts | Preserve actual donors and pairing; a matching cluster label omits the expression endpoint. |
| Transcript quantification | Native estimates for all transcripts, effective lengths, units and ambiguity | Real reads do not reveal the source simulation's exact latent molecule counts. |
| Community assembly | Reads to contigs, read-back support, every classification and complete domain/species totals | A single reference virus or one named species does not complete the community workflow. |

Independent source selection, numerical settings and measured runtime remain open
for every card. The two simulated/planted source constructions retain explicit
observed-data adaptation limits. These contracts support planning; source answers
and fixed grader literals are not used to author training inputs or rewards.

## Biomni-Eval1 database-query contracts

The [Biomni-Eval1 designs](../../experiments/post_training/bio_tasks/workflow_designs/biomni-eval1.json)
now specify complete artifacts and incorrect-output checks for all seven DBQA
families, following review of their 50 question stems. Independent resource exports,
redistribution rights, exact schemas and runtime remain unresolved. This review
adds no tasks or validated mappings; the other 383 records retain their existing
inspection limits.

| Family | Source records | Required distinction |
|---|---:|---|
| Gene-set membership | 12 | Resolve species, collection and exact set; membership does not require new enrichment or expression analysis. |
| Clinical variant lookup | 15 | Eight residue-change queries and seven candidate-protein queries require sequence identity and assertion joins; genomic annotation alone is insufficient. |
| Promoter binding | 7 | Preserve strand-relative TSS windows, transcript selection and the specific binding-site track. |
| Interaction lookup | 5 | Source questions ask for predicted database membership; retain separate predicates for predicted and measured interactions. |
| Cytoband location | 4 | Resolve parent/sub-band intervals and the declared gene-location predicate on one assembly/release. |
| miRNA target lookup | 4 | Preserve mature-miRNA arms and prediction-resource membership; RNA folding is a different endpoint. |
| Disease-resource difference | 3 | Compare two complete, compatible resource exports; absent records do not establish biological absence. |

Protein substitutions can match multiple nucleotide variants, and classification
depends on record level and assertion type. The contracts preserve that ambiguity
instead of inventing one genomic allele ([ClinVar query documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/help/)).
Predicted interaction and miRNA target records are not measured outcomes. Any
observed-data adaptation must retain the source lookup endpoint or record the
coverage gap; adding experimental data does not itself establish equivalence.

## Broader question portfolio

The [additional benchmark questions](../../experiments/post_training/bio_tasks/workflow_designs/additional_benchmarks.json)
assign another **483 task/protocol records from 14 releases to 162 proposed question
cards**. Together with the three earlier design files and the separate BioAgent
contracts, these five files assign 848 records across 18 inventoried ID releases. These counts
measure planning assignments, not task coverage or independent training questions.
Each card preserves its source inspection tier; several rely on question stems or
reviewed workflow stages and still need exact endpoint decisions.

The additional file includes EpiBench, VariantBench, ScienceAgentBench,
CORE-Bench, DiscoveryBench, all three TxBench releases, SpatialBench and its long
protocol, scBench and its long protocol, sc-HeurekaBench and BAISBench. It records
exact target IDs, observed-input requirements, required stages, artifacts,
deterministic checks and source-specific scientific decisions. No implementation
worker has been assigned.

Connected components share a question where the input and analysis support it:
six single-cell QC/representation families form one analysis, spatial coordinates
and expression-space QC share another, and caller-to-peptide analysis retains both
somatic-variant entry points. Cross-benchmark consolidation candidates remain
explicit; sharing a workstream does not establish interchangeable methods or data.

The expanded portfolio also includes the following source-specific plans:

| Source | Records | Question cards | Inspection or adaptation limit |
|---|---:|---:|---|
| [BioML-bench](../../experiments/post_training/bio_tasks/workflow_designs/biomlbench.json) | 406 | 25 | Registry/configuration review; 24 definitions appear in the released experiment list |
| [Biomni-Eval1](../../experiments/post_training/bio_tasks/workflow_designs/biomni-eval1.json) | 433 | 20 | Shared protocols plus reviewed sequence/database questions; individual evidence remains unreviewed |
| [BioKGBench](../../experiments/post_training/bio_tasks/workflow_designs/biokgbench.json) | 225 | 3 | Structured graph evidence can be verified; unrestricted literature entailment and absence claims remain gaps |
| [PromptBio-Bench](../../experiments/post_training/bio_tasks/workflow_designs/promptbio-bench.json) | 244 | 94 | Exact versioned patterns and artifact specifications; assay and numeric decisions remain |
| [BixBench3](../../experiments/post_training/bio_tasks/workflow_designs/bixbench3.json) | 20 | 19 | Nineteen endpoint proposals; one excluded objective retained |
| [DrugDiscoveryBench](../../experiments/post_training/bio_tasks/workflow_designs/drugdiscoverybench.json) | 82 | 27 | Public-preview endpoint-stage review; independent evidence and inputs needed |
| [Liu single-cell](../../experiments/post_training/bio_tasks/workflow_designs/liu-single-cell.json) | 63 | 49 | Native method requirements and shared prompt variants retained |
| [Bio-Task Bench](../../experiments/post_training/bio_tasks/workflow_designs/bio-task-bench.json) | 34 | 10 | Related component outputs grouped into connected analyses |
| [BioXArena](../../experiments/post_training/bio_tasks/workflow_designs/bioxarena.json) | 76 | 76 | Catalog/scorers inspected; full prompts and schemas remain inside uninspected data archives |
| [CellBench](../../experiments/post_training/bio_tasks/workflow_designs/cellbench.json) | 50 | 22 | Context-inspired executable adaptations; original open-ended planning quality is not covered |
| [SciGym](../../experiments/post_training/bio_tasks/workflow_designs/scigym.json) | 350 | 3 | Shared observed-dynamics adaptations; zero individual system endpoint assignments |
| [BixBench-Verified-50](../../experiments/post_training/bio_tasks/workflow_designs/bixbench-verified-50.json) | 50 | 0 new | Known aliases; 17 revised question texts retain separate protocol review |
| [ScholarQA-Bio](../../experiments/post_training/bio_tasks/workflow_designs/scholarqabench-bio.json) | 1,451 | 0 | All public question identities/hashes; individual scientific endpoints and deterministic adaptations unreviewed |

[ScholarQA-Bio's public file](../../experiments/post_training/bio_tasks/benchmark_tasks/scholarqabench-bio.json)
contains 1,451 distinct question IDs. Its source protocol produces long-form literature
syntheses with citations; the inspected citation scorer uses a learned attribution
model. Every question remains an explicit endpoint-review gap. Frozen retrieval,
source-table extraction and quantitative evidence reconciliation may supply bounded
components, but do not establish synthesis correctness. Individual question inspection
must determine whether a complete deterministic adaptation is possible.

The remaining source leads now retain publication-level boundaries in the source
inventory. BioASQ requires selecting a specific agentic protocol and release; its
retrieval/exact-answer metrics do not cover ideal-summary quality. LAB-Bench needs a
tool-enabled subset and alias check against Biomni-Eval1. The original BixBench
notebook protocol may overlap existing IDs. Restricted suites and model-judged
scientific opinions remain explicit gaps, with no invented task count or coverage.

BioML's 283 assay-ranking definitions need distinct measured-property and
generalization questions, rather than automatic replication of registry entries.
Its plan retains malformed/duplicated configurations and conflicting descriptions,
including a pressure-prediction metric mismatch. No source configurations or task
titles alone establish valid training data or acceptance rules.

PromptBio's original classifier bundle was split into linear, tree, kernel/neighbor
and imbalance/ensemble questions. Methylation read counts, composition-adjusted
associations and chromosome density also remain separate. Each of its 244 source IDs
is still assigned once. Required packages remain explicit: the Liu PyDESeq2 endpoint
requires native PyDESeq2 execution even if an R DESeq2 cross-check is available.

The 25 retained proposed combinations include ortholog metrics, donor pseudobulk expression,
count contrasts and enrichment, reference-cell classification, paired RNA/protein
prediction, receptor/state analysis and spatial neighborhoods. Every combination
states when one independently observed study can support all its endpoints. For
example, a two-dimensional section cannot supply a three-dimensional neighborhood
endpoint, and a TCR-only study cannot supply BCR V/J outputs. These combinations
remain conditional until the actual inputs and contracts are fixed.

The first review narrowed or split eight combinations after rechecking source questions. The
former donor-pseudobulk group mixed 73 records with different estimands. Sixty-nine
BAISBench/sc-HeurekaBench records now have 16 separate cards for localization,
aging, tissue, genotype, sex interaction, disease state and observed host-response
questions. The paired-condition group retains three exact source records. All
record IDs are preserved; no source answer or multiple-choice option was used.

Paired-assay regression is separate from concentration-response curve fitting.
TCR state analysis is separate from BCR V/J assignment, and protein-ligand contacts
are separate from protein-DNA interfaces. Within-volume 3D neighborhoods are
separate from replicated section comparisons. These decisions increase the number
of proposed questions because one unrelated input collection cannot honestly
complete all those endpoints.

The [shared input-source plan](../../experiments/post_training/bio_tasks/workflow_designs/source_collections.json)
records six candidates and the questions each might support. Four new sources were
reviewed through primary metadata; two reuse existing observed-data assessments.
No new biological payload was downloaded, and no source gained training clearance.

| Input collection | Potential shared work | Scientific boundary |
|---|---|---|
| [Lawlor paired PBMC CITE-seq](https://explore.data.humancellatlas.org/projects/efea6426-510a-4b60-9a19-277e52bfa815) | Paired-condition RNA analysis, RNA/protein prediction, cell characterization | Ten donors; exact layers and per-condition sample coverage still need verification. Healthy stimulation does not supply disease, age or genetic-perturbation outcomes. |
| [GSE252331](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252331) | Alternative RNA/ADT prediction and clinical-state comparisons | Reconcile the deposited subset with the study description; myeloid enrichment complicates cell-abundance interpretation. |
| [GSE271413](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271413) | Receptor QC, sequence annotation and developmental-state analysis | Verify donor/visit identities and sequence availability. It does not establish cancer exhaustion or sorting-gate endpoints. |
| [Fang 3D MERFISH](https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b922) | Native three-dimensional neighborhoods and within-volume expression analysis | Targeted panels and limited biological replication; separate regions cannot be treated as interchangeable animal replicates. |
| Existing GSE60450 counts | Count contrasts, enrichment and native PyDESeq2 comparison | New endpoint and expanded benchmark-lineage checks remain; the count package does not exercise raw-read processing. |
| Existing matched ortholog alignments/trees | Per-locus evolutionary signal | Mirror/deposit identity and full endpoint validation remain; precomputed loci do not exercise ortholog discovery. |

A preliminary accession/DOI search found no explicit matches for the four new
candidates in inspected benchmark metadata. This does not prove cohort independence
from uninspected or gated source inputs. The source plan retains that uncertainty,
exact manifest requirements, rights gaps and the biological limitations above.

Planning priority considers exact endpoint detail, cross-benchmark reuse and
feasibility. Predicted validated gains remain unset. When those contracts are fixed,
rank feasible candidates by conservative gains in missing distinct endpoints,
then benchmark breadth and implementation cost. Keep source aliases, repeated assay
configurations and unreviewed individual cases out of automatic coverage claims.

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

Native environment contexts export `/opt/bio/bin` from `/etc/profile.d/bio.sh`
as well as Docker `ENV`. The corrected model-free Harbor check passed all three
expected cases on the reserved TRC CPU in 3m06s: the native oracle earns reward 1;
a split summary earns 0 while all seven artifacts pass; a split summary with a
corrupted contrast earns 0 and flags exactly `contrast_weights.tsv`. Each fresh
Terminus-2 login terminal resolves `/opt/bio/bin/Rscript` and runs R 4.5.3.
All task sandboxes were deleted. The diagnostics report pins the successful archive
and retains both earlier launcher failures before sandbox creation. No model calls
were made and the original GLM grades remain unchanged.

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

The current ranking compares eight assessed candidates. Counts are conditional
targets and can shrink after data or endpoint review. Related protocol versions add
no ranking credit. Data suitability and effort break ties.

| Order | Connected workflow | Distinct missing targets | Principal gap before authoring |
|---:|---|---:|---|
| 1 | Branch summaries across matched observed ortholog trees | 35 | Full endpoint contracts and source provenance; preliminary native metrics pass |
| 2 | Informative sites, composition and saturation across matched alignments | 11 | Native gap/masking definitions, source provenance and runtime |
| 3 | Replicated single/double perturbations across media, with directional pathway overlap | 11 | Independent intervention study with the required factorial design and frozen pathway annotations |
| 4 | Single-cell QC, normalization, embedding and group diagnostics | 8 | Native h5ad, mitochondrial annotation and endpoint-specific representation checks |
| 5 | Observed bacterial reads through alignment and variant comparison | 8 | Independent isolate reads and the required trimming/alignment/calling pipeline |
| 6 | Prespecified binary clinical-response models | 7 | Independent response cohort with the required covariates |
| 7 | Transcript structure and strand-aware reference audit | 6 | Complete matching eukaryotic reference and annotation |
| 8 | Native BAM read-structure and eligibility audit | 2 | Overlaps the broader resequencing candidate; remove these targets if it completes first |

The [multi-locus source manifest](../../experiments/post_training/bio_tasks/workflow_designs/multilocus_source.json)
pins 30 distinct original orthogroups from the published Medusozoa study. Each has
at least 12 Hydrozoa and eight Scyphozoa tips, and each alignment's complete tip set
matches its published tree. Selection inspected 43 original groups using taxon
counts, without selecting for metric outcomes. The 60 selected files total 1.45 MB.
These observations replace the earlier seven-orthogroup feasibility sample.

Tree and alignment endpoints now have separate candidates. One additional alignment
endpoint requires naturally observed alignments with more than 70% gaps and earns
no ranking credit until suitable observations are confirmed. The tree task proposes
midpoint rooting as a geometric convention and checks every preserved pairwise distance.
It distinguishes PhyKIT's self-including long-branch score from the self-excluded
statistic. Matched loci across clades are paired observations; an unpaired rank-test
p-value cannot establish a biological clade difference.

Dryad explicitly licenses the deposit CC0, but exact mirror-file membership remains
unverified because the small tree archive download returns HTTP 403. The manifest
records both the deposit archive hashes and the mirror file hashes. Candidate
validation proceeds; redistribution and benchmark-study independence remain open.
The local native metric check passes 60 trees, 1,184 tip rows and 15,336 pair rows
with DendroPy rooting. The earlier TRC check detected a Biopython midpoint-rooting
distance change and remains recorded as failed. The source manifest retains the
reproducer, package versions and independent checks. No native audit or source
manifest alone adds task or coverage credit.

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
