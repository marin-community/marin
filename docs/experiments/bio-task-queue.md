# Biology workflow authoring queue

The [question portfolio](../../experiments/post_training/bio_tasks/workflow_portfolio.json)
accounts for all **5,133 inventoried task, protocol or dataset-definition records
across 32 ID releases**. It links 3,058 records to scientific question proposals,
links 50 known BixBench-Verified aliases to their original designs, preserves 350
SciGym systems, 1,451 ScholarQA-Bio questions and 223 LAB-Bench questions as
individual-endpoint review gaps, and retains one excluded objective. Nine provisional-ID
source leads still lack task inventories; each now has an explicit access, identity,
endpoint or scope disposition in the source catalog.

The portfolio contains **702 question cards and 661 proposed task groups**, including
25 conditional combinations of question cards. These are planning units, with
substantial differences in source inspection depth. They add zero validated mappings
and do not set a target dataset size. Each combined question retains all member
contracts; incompatible inputs, methods or estimands require separate tasks.

The [workstream plan](../../experiments/post_training/bio_tasks/workflow_plan.json)
organizes these proposals under 147 broad questions across 52 scientific workstreams.
It records exact task IDs, required artifacts, verification approaches and unresolved
input/endpoint requirements. Broad topic membership does not establish coverage.

All authored tasks use **Harbor** for terminal-based solving and deterministic
artifact grading. Notebook-based benchmarks supply scientific questions only;
there is no notebook execution or submission protocol to implement.

Prioritize realistic scientific outputs with direct executable checks. Defer endpoints
without defensible deterministic scoring, including subjective interpretation or
visual quality. Keep their source IDs and uncovered scope explicit; they do not
block independently ready tasks. Numerical components alone do not earn full
coverage of a source question that also requires an ungraded result.

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

The plan groups 561 benchmark/family combinations for review. Fifty known
BixBench-Verified aliases reduce the record count to 5,083, or 5,082 after the one
excluded objective. Other shared studies, assays and protocols are not automatically
independent workflows. Workstream membership alone does not show that a proposed
question completes a benchmark endpoint, and the proposed question count is not a
coverage forecast or a cap on authoring.

All 30 original proposed combinations have now received a scientific planning
review. Five were split completely and eight narrowed across the endpoint reviews;
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

The count/enrichment proposal now specifies nine BixBench and eight PromptBio
endpoints against their source questions. It retains separate raw-p and adjusted-p
gene masks, tested-gene and genome-wide ORA backgrounds, native GO simplification,
and permutation-based GSEA results. Significant-only top-ten sets differ from the
top five by absolute NES. Raw-p and adjusted-p volcano plots also have separate
numerical tables. Grade those tables and keep figures inspectable; visual endpoints
remain deferred without requiring a special image-comparison pipeline.

The current mouse candidate cannot satisfy the named human KEGG release or a
fibroblast-specific contrast. Those inputs remain explicit requirements of the
conditional group. The inspected clusterProfiler source also shows that semantic
term removal is not connected-component clustering and does not guarantee a single
surviving representative for every removed term. Native similarity neighborhoods,
tie decisions and retained term IDs must be preserved. The full environment and
annotation snapshots still need validation before assignment.

The connected ortholog proposal specifies all 47 source endpoints individually:
within-locus tip/pair summaries, across-locus reductions, ordered rank tests,
paired differences and ratios, strict thresholds, and the separate five-tree panel.
The shared contract fixes protein missing-symbol handling, RCV normalization,
origin-constrained saturation regression and undefined-result statuses. All 11
reviewed PhyKIT source files match the pinned Git revision and downloaded wheel.
Native and self-excluded long-branch scores remain separate; unrounded computation,
midpoint rooting, sample variance and replacement clades are declared adaptations.
The 30-locus candidate still needs rights and study-lineage clearance, native
alignment checks and complete Harbor validation. No coverage is added.

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

## Biomni-Eval1 endpoint contracts

The [Biomni-Eval1 designs](../../experiments/post_training/bio_tasks/workflow_designs/biomni-eval1.json)
specify complete artifacts and incorrect-output checks for all 20 families.
The review covers 100 DBQA/SeqQA question stems and a structural audit of the
remaining 333 prompts across eight source protocols. Individual study and case
evidence remains unreviewed. Independent inputs, redistribution rights, exact
schemas, scientific conventions and runtime remain unresolved; no tasks or
validated mappings are added. SeqQA contracts assign each source ID to its exact
query mode, so a simpler operation cannot silently cover a whole family.

| Family | Source records | Required distinction |
|---|---:|---|
| Gene-set membership | 12 | Resolve species, collection and exact set; membership does not require new enrichment or expression analysis. |
| Clinical variant lookup | 15 | Eight residue-change queries and seven candidate-protein queries require sequence identity and assertion joins; genomic annotation alone is insufficient. |
| Promoter binding | 7 | Preserve strand-relative TSS windows, transcript selection and the specific binding-site track. |
| Interaction lookup | 5 | Source questions ask for predicted database membership; retain separate predicates for predicted and measured interactions. |
| Cytoband location | 4 | Resolve parent/sub-band intervals and the declared gene-location predicate on one assembly/release. |
| miRNA target lookup | 4 | Preserve mature-miRNA arms and prediction-resource membership; RNA folding is a different endpoint. |
| Disease-resource difference | 3 | Compare two complete, compatible resource exports; absent records do not establish biological absence. |
| Restriction digestion | 7 | Four length and three count queries require complete fragment multisets and cleavage geometry, including two-cut enzymes. |
| ORF analysis | 11 | Two threshold counts, six longest-protein queries and three residue queries need explicit ORF, genetic-code and tie rules. |
| Amplicon reconstruction | 7 | Two sequence-target, two length-target and three given-pair queries require separate sequence/length predicates and all possible products. |
| Assembly-primer compatibility | 9 | Candidate-pair selection requires both junctions and the complete product on independent published sequence records. |
| Restriction-cloning compatibility | 8 | Six primer-choice and two enzyme-choice queries preserve end polarity, internal cuts and complete products. |
| GC calculation | 4 | Grade exact base counts and percent before integer rounding, with explicit ambiguity and denominator rules. |
| Translation efficiency | 4 | Sequence-context predictions and measured translation are distinct; an amino-acid translation does not answer the source endpoint. |
| Associated-variant prioritization | 43 | Preserve trait, study, allele and ranking rules; a source prompt supplies candidate rs IDs without defining a unique evidence statistic. |
| Locus-to-gene prioritization | 150 | Retain three 50-record source protocols and distinguish proximity, reported genes, model scores and observed functional support. |
| Screen-effect retrieval | 50 | Match experimental context and effect direction before ranking all candidates; raw guide counts are not supplied by the source prompts. |
| Phenotype-to-gene prioritization | 50 | Rank 8–22 candidate genes using observed profiles and explicit ontology rules; a patient VCF is not a source input. |
| Rare-disease discrimination | 30 | Every source case supplies one gene; the endpoint distinguishes diseases using 1–23 phenotype terms. |
| Delivery-method comparison | 10 | Context-only category choice lacks a defined comparative outcome; retain the observed-assay adaptation gap. |

Protein substitutions can match multiple nucleotide variants, and classification
depends on record level and assertion type. The contracts preserve that ambiguity
instead of inventing one genomic allele ([ClinVar query documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/help/)).
Predicted interaction and miRNA target records are not measured outcomes. Any
observed-data adaptation must retain the source lookup endpoint or record the
coverage gap; adding experimental data does not itself establish equivalence.

The translation-efficiency proposal retains an unresolved observed-data adaptation:
compare prespecified sequence predictions with matched assays in a defined human-cell
context. Ribosome occupancy per mRNA needs a declared estimand and is not automatically
a direct protein-production rate ([primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11326257/)).
Feature-only scoring remains component coverage. Sequence-oracle agreement establishes
the specified computation, not laboratory yield or measured expression.

The 333 structurally audited records contain 319 distinct prompt texts: ten
identical-prompt groups comprise 24 records. Their IDs and hashes remain in the
design file; answer equivalence has not been inspected, so canonical alias and
coverage counts are unchanged. The gene-selection evaluator accepts any nonempty
intersection with its reference list. Authored tasks instead require the declared
selection cardinality and full candidate evidence ([pinned evaluator](https://github.com/snap-stanford/Biomni/blob/400c1f366b96a35ca253e13c9b06c5076af41d65/biomni/eval/biomni_eval1.py)).

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
| [BioKGBench](../../experiments/post_training/bio_tasks/workflow_designs/biokgbench.json) | 225 | 6 | Structured graph evidence can be verified; unrestricted literature entailment and absence claims remain gaps |
| [PromptBio-Bench](../../experiments/post_training/bio_tasks/workflow_designs/promptbio-bench.json) | 244 | 101 | Exact versioned patterns and artifact specifications; assay and numeric decisions remain |
| [BixBench3](../../experiments/post_training/bio_tasks/workflow_designs/bixbench3.json) | 20 | 19 | Nineteen endpoint proposals; one excluded objective retained |
| [DrugDiscoveryBench](../../experiments/post_training/bio_tasks/workflow_designs/drugdiscoverybench.json) | 82 | 27 | Public-preview endpoint-stage review; independent evidence and inputs needed |
| [Liu single-cell](../../experiments/post_training/bio_tasks/workflow_designs/liu-single-cell.json) | 63 | 49 | Native method requirements and shared prompt variants retained |
| [Bio-Task Bench](../../experiments/post_training/bio_tasks/workflow_designs/bio-task-bench.json) | 34 | 10 | Related component outputs grouped into connected analyses |
| [BioXArena](../../experiments/post_training/bio_tasks/workflow_designs/bioxarena.json) | 76 | 76 | All scorers, 16 official chemical/network descriptions and 10 provisional mirror single-cell descriptions inspected; 50 prompts remain uninspected |
| [CellBench](../../experiments/post_training/bio_tasks/workflow_designs/cellbench.json) | 50 | 22 | Context-inspired executable adaptations; original open-ended planning quality is not covered |
| [SciGym](../../experiments/post_training/bio_tasks/workflow_designs/scigym.json) | 350 | 3 | All supplied input structures audited; shared observed-dynamics adaptations, zero biological endpoint assignments |
| [BixBench-Verified-50](../../experiments/post_training/bio_tasks/workflow_designs/bixbench-verified-50.json) | 50 | 0 new | Known aliases; 17 revised question texts retain separate protocol review |
| [ScholarQA-Bio](../../experiments/post_training/bio_tasks/workflow_designs/scholarqabench-bio.json) | 1,451 | 0 | All stems reviewed; 553 possible quantitative components, 898 open synthesis/strategy requests; all complete synthesis contracts unresolved |
| [LAB-Bench literature/database](../../experiments/post_training/bio_tasks/workflow_designs/labbench-literature-database.json) | 801 | 16 + 7 shared | 520 database schemas reuse existing cards; 58 supplementary targets; all 199 literature stems reviewed across 14 patterns, with source evidence and acceptance unresolved |

BioXArena's chemical/network cards now use the descriptions and CSV headers from
two checksum-verified official archives. BACE1 uses log-nM IC50; EGFR and hERG use
negative-log molar potency. Cell Painting starts from processed well profiles,
and the network classification tasks need explicit category and evidence rules.
Multilabel grading masks untested outcomes and excludes constant-label columns;
the proposed verifier must check each denominator. The inventory records conflicts
between descriptions and actual headers, without inspecting biological row values
or private answers. Synthetic-lethality label provenance is unresolved: the prompt
names network and single-gene features but no measured double-perturbation assay.
Independent observed inputs, rights and runnable acceptance contracts remain required.

The ten single-cell cards use a pinned expanded mirror for provisional description
and header review. All 16 chemical/network descriptions match the official archive
bytes, but this does not prove parity for the single-cell descriptions. No biological
rows or matrices were inspected. Primary-source scorer hashes were rechecked.

The designs distinguish six-candidate cell matching from global permutation recovery,
closed-set labels from unknown-cell detection, and classification accuracy from batch
correction. They preserve raw versus normalized modality targets and label provenance.
Two mirror descriptions conflict with CSV headers: cell-type prediction lacks the
claimed donor/sample covariates, while query matching includes an ATAC index described
as training-only. Neither finding establishes leakage without inspecting the values.
The denoising description specifies simulated dropout; that construction is deferred
from real-data authoring. It needs no LLM judge, but a measured-target alternative
has not been selected. These reviews add no validated mappings.

BioKGBench's 225 instruction identities now route to six contracts: protein names
(55), entry presence (45), protein-pair evidence (94), cellular localization (11),
tissue evidence (11) and disease evidence (9). The last three require different
observations and cannot inherit a protein-interaction verifier. Protein-pair evidence
can still conditionally share a question with the 45 Biomni/LAB prediction queries,
provided the outputs preserve measured versus predicted support.

The contracts retain native UniProt history, PSI-MI interaction records and
qualifier-aware GO annotations where applicable. Names are compared under a declared
field schema; database absence needs a complete scoped index or authoritative history.
Localization, tissue detection and disease association each preserve their evidence
limits. Graph, biological-evidence and label fields remain uninspected; the review
adds no validated mapping or literature-entailment guarantee.

[ScholarQA-Bio's public file](../../experiments/post_training/bio_tasks/benchmark_tasks/scholarqabench-bio.json)
contains 1,451 distinct question IDs. Its source protocol produces long-form literature
syntheses with citations; the inspected citation scorer uses a learned attribution
model. All 1,451 trimmed question texts are distinct, which does not establish
independent scientific workflows. All 1,451 stems have been read individually.
Of these, 553 suggest possible quantitative components across 22 analysis categories;
898 retain open synthesis or strategy dispositions. The component routes include
assay and outcome analysis, molecular measurements, community analysis, structural
validation, neural signals, behavioral learning and image measurements. Exact per-ID
routes and counts are in the linked design file. Related existing cards are
component references, not assignments.

The review retains real-input requirements, complete artifacts, incorrect-output
checks and each component's limits. For example, kinetic fitting does not answer
which delivery strategy is best, and comparing animal outcomes does not establish
human efficacy. Relative microbial abundance is not absolute biomass, unassayed
editing sites are not measured negatives, and dye color loss alone does not prove
biodegradation. The new component contracts preserve those distinctions.
All 1,451 complete synthesis endpoints remain gaps. Source evidence, inspiring papers
and answers were not read, and no deterministic citation check substitutes for
scientific entailment. Prioritize executable endpoints in the analytical ID benchmarks
before these unresolved synthesis contracts; use the component ideas where they
strengthen an independently justified workflow.

The nine remaining source leads have explicit dispositions in
[benchmark sources](../../experiments/post_training/bio_tasks/benchmark_sources.json).
They remain provisionally ID and add no invented task IDs or coverage.

| Source | Current planning boundary |
|---|---|
| LABBench2 | Pinned metadata is available; the existing credential lacks the task-data grant. |
| TargetVal | The task constructor selects genes using external scores; requested sample counts do not enumerate the selected identities. |
| BioSecBench Surveillance / Function | Public example descriptions need individual review; restricted suites remain unenumerated. Only benign observed-data analysis can inform authoring. |
| BioSecBench Refusal | Refusal behavior is a different objective from successful observed-data analysis. |
| LifeSciBench | No complete manifest found in the overview/paper; tool-enabled scope and full deterministic acceptance remain unverified. |
| ABC-Bench | No inspected task manifest; physical execution claims need their own evidence. Screening evasion is excluded. |
| ABLE | No inspected task manifest; generated designs do not supply observed biological-performance labels. |
| BioASQ | Synergy 2025 selected for inventory; its official download redirects to login. Retrieval metrics cannot certify expert feedback or ideal summaries. |

BioASQ's [Synergy protocol](https://participants-area.bioasq.org/general_information/TaskSynergy26/)
uses versioned questions, PubMed snapshots and iterative expert feedback. Preserve
edition/round/question identity and answer-readiness state when inspecting a release;
the 2025 README must confirm its specific conventions. An offline feedback replay
would be an explicit adaptation. These source limits do not block planning the
already inventoried analytical workflows.

The LAB-Bench source is scoped to public DbQA (520), LitQA2 (199) and SuppQA (82)
questions. All IDs, question hashes and twelve source subtask labels are inventoried;
other categories and private questions are outside this source row. All 82 supplementary
question stems and citation records have now been reviewed: 58 route to sixteen
provisional Harbor study-audit questions, while 24 retain explicit interpretation,
context or domain-scope gaps. The questions cover cohort/QC reconciliation, model
results, proteomic evidence, assay measurements, structure/imaging metadata and
reported reagent/sequence provenance. They preserve original units, sample hierarchy,
reported-versus-recomputed values and exact evidence locations. Inputs must retain
complete relevant observed study records; answer-only tables are insufficient.
All 520 database question schemas now link to the existing seven Biomni database
contracts, adding no new card. The audit preserves 160 gene-set queries, 160 clinical
variant queries and forty each for promoters, predicted interactions, cytobands, miRNA
targets and disease-resource differences. Clinical queries request 61 benign and 99
pathogenic recorded classifications; eighty provide a reference protein and eighty
depend on full-sequence choices. No source choices or database records were inspected.
All 199 literature question stems now have per-ID scientific-pattern reviews.
The largest groups concern molecular-feature contrasts (34), entity/marker/localization
evidence (28), interaction/mutational evidence (24), phenotype comparisons (21),
assay responses (20) and reported quantities (20). The remaining groups cover sequence,
structure, connectivity, enrichment, method provenance, model comparisons and open
scientific claims. These are review categories, not task counts or equivalence claims.

The literature review specifies independent input requirements, complete outputs,
incorrect-output checks and related existing question cards for each pattern. It
prioritizes measured contrasts, assay analysis, study accounting and structure
comparison for further input planning. Nine mechanistic/general claims and two open
literature-existence questions lack a complete deterministic acceptance contract;
bounded retrieval cannot establish global absence. Every literature endpoint remains
unassigned until its evidence supports a faithful adaptation. The 24 supplementary
context gaps also remain.
Independent input packages and complete acceptance contracts remain open; source
supplements, ideal answers, distractors and key passages were not read. No validated
coverage is added.
All fifty Biomni-Eval1 DbQA stems occur in this release; twenty match multiple source
IDs. DbQA has 328 distinct trimmed stems across 520 records. Candidate choices can
change the task even when the stem matches, so these relations add no alias credit.

The original BixBench notebook source now shares the existing pinned 205-question
inventory with the Harbor source row. The upstream notebook loader and default
configuration select the same dataset and train split; all 205 question IDs and
hashes match. This adds no question or coverage credit. The notebook protocol is
source provenance only; every authored task runs through Harbor. Historical paper
release identity remains unverified. The selected metadata has
59 capsule UUIDs and 54 short-ID groups; the README headline of 60 capsules is not
an observed inventory count. Pinned source links and file hashes are recorded in
[benchmark sources](../../experiments/post_training/bio_tasks/benchmark_sources.json).

BioML's 283 assay-ranking definitions need distinct measured-property and
generalization questions, rather than automatic replication of registry entries.
Its plan retains malformed/duplicated configurations and conflicting descriptions,
including a pressure-prediction metric mismatch. No source configurations or task
titles alone establish valid training data or acceptance rules.

PromptBio's original classifier bundle was split into linear, tree, kernel/neighbor
and imbalance/ensemble questions. Methylation read counts, composition-adjusted
associations and chromosome density also remain separate. Direct inspection of ten
additional pinned task stems separates oncogene-set overlap from enrichment,
expression/essentiality from drug-response correlation, four time-series targets,
and three distinct assay-probability questions. All 244 source IDs remain assigned
once across 101 cards; these are proposals, not generated tasks.

The time-series cards retain the specified LSTM, random forest, seasonal decomposition
and expanding-window regression requirements. Proposed changes to overlapping or
random temporal windows are explicit adaptations; hospital occupancy, admissions
and patient glucose are different outcomes. Assay cards require observed diagnostic,
editing or binding evidence with real denominators. They cannot share invented
binomial counts. Native methods remain explicit elsewhere too: the Liu PyDESeq2
endpoint requires PyDESeq2 execution even if an R DESeq2 cross-check is available.

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
The Lawlor cell-annotation CSV was downloaded and its checksum verified; expression
matrices and raw reads remain uninspected. No source gained training clearance.

| Input collection | Potential shared work | Scientific boundary |
|---|---|---|
| [Lawlor paired PBMC CITE-seq](https://explore.data.humancellatlas.org/projects/efea6426-510a-4b60-9a19-277e52bfa815) | Paired-condition RNA analysis, RNA/protein prediction, cell characterization | Annotation metadata confirms ten donors in all three conditions. Matrix layers and matched cell axes remain unverified; healthy stimulation does not supply disease, age or genetic-perturbation outcomes. |
| [GSE252331](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252331) | Alternative RNA/ADT prediction and clinical-state comparisons | Reconcile the deposited subset with the study description; myeloid enrichment complicates cell-abundance interpretation. |
| [GSE271413](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271413) | Receptor QC, sequence annotation and developmental-state analysis | Verify donor/visit identities and sequence availability. It does not establish cancer exhaustion or sorting-gate endpoints. |
| [Fang 3D MERFISH](https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b922) | Native three-dimensional neighborhoods and within-volume expression analysis | Targeted panels and limited biological replication; separate regions cannot be treated as interchangeable animal replicates. |
| Existing GSE60450 counts | Developmental count contrasts, enrichment and native PyDESeq2 | Verified 27,179 genes and twelve sample identities. Two libraries per population/stage; no documented failed replicate for a QC-exclusion task. Enrichment inputs and new native methods remain unresolved. |
| Existing matched ortholog alignments/trees | Per-locus evolutionary signal | Mirror/deposit identity and full endpoint validation remain; precomputed loci do not exercise ortholog discovery. |

Lawlor's 282,528 annotation rows include 16,382 labelled cells across all 30
donor-condition pairs. Four T-cell populations meet a proposed minimum of 20 cells
in every pair; four monocyte strata are empty and cannot become zero-expression
pseudobulk samples. Donors are the biological replicates, and the conditions are
separate ex vivo samples, not repeated measurements of the same cells.

The final labels and selection used protein evidence. RNA-to-protein prediction
therefore has a separate candidate pool: 53,707 condition/genotype singlets selected
without a cell-type-label filter. Upstream selection dependencies still need review.
That count differs from the paper's intermediate filtering count, so reproducing
the original pipeline remains unresolved. The source plan pins five deposited files
and specifies input boundaries for five question groups; it adds no validated mapping.

A preliminary accession/DOI search found no explicit matches for the four new
candidates in inspected benchmark metadata. This does not prove cohort independence
from uninspected or gated source inputs. The source plan retains that uncertainty,
exact manifest requirements, rights gaps and the biological limitations above.

GSE60450's vendored count bytes match the [licensed public deposit](https://zenodo.org/records/4249555).
Both provider sample sheets agree with GEO metadata after an explicit filename-prefix
conversion. The source plan proposes a luminal developmental contrast using all six
luminal libraries; 16,659 genes pass the declared population count filter. Disease
context and single-cell processing remain outside this input's scope.

The count/enrichment and PyDESeq2 questions now have input and acceptance contracts.
Existing DESeq2 references do not validate shrinkage, GSEA or semantic reduction.
The available GO memberships lack ontology edges, and no KEGG snapshot is supplied.
The predicted/reference gene-set endpoint needs separate inputs. This review adds
no validated benchmark coverage and rejects this source for replicate-QC exclusion.

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

Keep every unresolved source category in the denominator. All 350 supplied SciGym
model structures are now audited: anonymized species with no reactions, kinetic
laws or parameters. The connected proposal requires reconstructing an executable
model and predicting private observed responses. A two-model comparison is a
supporting component. Species counts range from 2 to 786, and biological identities,
independent measured interventions and per-system equivalence remain unresolved.
The source simulations do not supply real biological observations. BioML assay and cross-validation
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
