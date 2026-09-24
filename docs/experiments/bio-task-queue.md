# Biology workflow authoring queue

Choose the next coherent task by its expected gain in **currently missing,
distinct ID benchmark tasks**. Recompute the gain after each completed or rejected
candidate. Repository coverage is secondary. The
[machine-readable queue](../../experiments/post_training/bio_tasks/workflow_queue.json)
records exact target IDs, overlapping protocols, required artifacts, source gaps,
selection rationale and completed work. The
[coverage inventories](../../experiments/post_training/bio_tasks/benchmark_tasks/README.md)
remain authoritative for validated mappings.

The initial ranking below compares assessed candidates. Counts are conditional
targets, not measured coverage: missing data, unsupported endpoints or runtime
limits can reduce them. Related protocol versions receive no additional ranking
credit. Data suitability and implementation effort break ties. Uninspected source
categories remain in the review backlog and can change this ranking.

| Order | Connected workflow | Distinct missing targets | Principal gap before authoring |
|---:|---|---:|---|
| 1 | Multi-locus ortholog alignment and evolutionary-signal comparison | 46 | Independent matched orthologs across two clades; native PhyKIT definitions and aggregation checks |
| 2 | Replicated single/double perturbations across media, with directional pathway overlap | 11 | Independent intervention study with the required factorial design and frozen pathway annotations |
| 3 | Single-cell QC, normalization, embedding and group diagnostics | 8 | Native h5ad, mitochondrial annotation and endpoint-specific representation checks |
| 4 | Observed bacterial reads through alignment and variant comparison | 8 | Independent isolate reads and the required trimming/alignment/calling pipeline |
| 5 | Prespecified binary clinical-response models | 7 | Independent response cohort with the required covariates |
| 6 | Transcript structure and strand-aware reference audit | 6 | Complete matching eukaryotic reference and annotation |
| 7 | Native BAM read-structure and eligibility audit | 2 | Overlaps the broader resequencing candidate; remove these targets if it completes first |

The selected next step is source validation for the multi-locus workflow. Its
46 targets have 13 related BixBench-Verified-50 records, which do not increase the
rank and still require separate protocol checks. A further endpoint requires
naturally observed alignments with more than 70% gaps; it receives no priority
credit until suitable data are confirmed. The existing single-gene COX1 example
does not establish multi-gene or biological-group coverage.

Keep each task scientifically coherent, grounded in independent observations,
offline and executable within 30 minutes. A matching package or component does
not complete a benchmark workflow. Promote a mapping only after its required
stages, complete artifacts, input-reading oracle and container checks pass. No
teacher model API calls or language-model judge are used for task generation or grading.

Use at most one bounded worker alongside the lead. The lead owns prioritization,
scientific workflow design, source suitability, oracle independence and coverage
decisions. Delegate factual source search, pinned installation and routine execution
or artifact checks to GPT-6 Sol; simpler mechanical work can use GPT-6 Luna. Give
each assignment explicit outputs, allowed actions and a stopping point. Review
the evidence before integrating results, and avoid simultaneous heavy local work.
The first assignment is a bounded primary-source search for independent multi-locus
data, with no compute launches or code changes.

Completed: the observed PhiX assembly workflow validates the CompBioBench N50
workflow adaptation. Its original Drosophila fixture and answer were not used;
release-level scientific and biological-lineage review remain open. The issue's
[opening ledger](https://github.com/marin-community/marin/issues/9257) tracks the
current task, mapping and repository counts separately from this prospective queue.
