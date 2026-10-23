# Reference task designs: COVID immune repertoire

These are human-authored target designs for the first Taskforge smoke. They are
not `TaskSpec` objects and are not yet runnable. Each design fixes the intended
solver experience, private evidence, reference interpretation, and judge
controls before we automate proposal authoring and building.

All three designs derive from
`jupyter-book/air_repertoire/clonotype.ipynb` in
`theislab/single-cell-best-practices` at revision
`dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5`. They share split group
`covid-immune-repertoire` and therefore must remain in one dataset split.

The build must replace the notebook's LaminDB reads with immutable,
digest-pinned solver artifacts. The source notebook, its prose conclusions,
private reference tables, rubric, and controls must not be present in the
solver environment.

## 1. Receptor evidence for the MAIT annotation

**Seed:** `covid-mait-gene-usage`
**Notebook anchor:** cells 11–13
**Intended environment:** Python with pandas; no network
**Answer:** an open-ended scientific report graded by `DecisionGrader`

### Solver-visible request

A transcriptome-based annotation pipeline labels a subset of T cells as MAIT
cells. Evaluate whether T-cell-receptor V-gene usage provides independent
support for that label.

Use the supplied cell-level receptor table to:

1. estimate use of `TRAV1-2` within every coarse T-cell label;
2. compare the putative MAIT population with relevant non-MAIT populations;
3. report the magnitude of the enrichment and any denominator or missing-data
   choices that materially affect it; and
4. explain what this receptor evidence supports and what it cannot establish by
   itself.

Provide a concise report with a result table or equivalent quantitative
summary and enough methodological detail to reproduce the comparison.

### Planned solver artifact

`v_gene_counts_by_donor_and_cell_type.csv`, one row per pseudonymous donor and
coarse T-cell label:

```text
donor_id
cell_type_coarse
n_cells
n_primary_vj_called
n_trav1_2
```

These are exact sufficient counts for pooled and donor-aware V-gene usage
comparisons. They preserve missing-call denominators without exposing original
donor identifiers, receptor sequences, expression matrices, or the source
MuData object. Fractions and conclusions are intentionally not supplied.

### Private evidence and reference interpretation

Notebook cell 12 reports the following fractions among the analyzed cells:

| coarse label | TRAV1-2 fraction |
| --- | ---: |
| CD4 | 0.004050 |
| CD8 | 0.025099 |
| Lymph_prolif | 0.036866 |
| MAIT | 0.691238 |
| Treg | 0.007282 |
| gdT | 0.018349 |

A strong answer should find approximately 69.1% usage in the MAIT-labelled
population versus roughly 0.4%–3.7% in the listed alternatives. This is large,
biologically coherent enrichment because MAIT cells commonly use a
semi-invariant alpha chain containing TRAV1-2. It independently supports the
transcriptomic label, but it does not prove that every labelled cell is a MAIT
cell or that every MAIT cell must use TRAV1-2. The report should distinguish
missing receptor calls from observed non-TRAV1-2 calls and avoid treating one
V gene as a complete receptor-specificity assay.

### Judge rubric

| Weight | Criterion | Full-credit behavior |
| ---: | --- | --- |
| 0.30 | Quantitative comparison | Computes group-specific fractions correctly and identifies the approximately 69% versus few-percent separation. |
| 0.25 | Reproducibility | States the unit, grouping, denominator, and handling of missing/unpaired receptors clearly enough to reproduce the result. |
| 0.25 | Scientific interpretation | Connects the enrichment to the semi-invariant MAIT receptor while treating it as supporting evidence rather than definitive proof. |
| 0.20 | Calibration and limitations | Discusses within-label exceptions, incomplete receptor recovery, and the limits of V-gene-only evidence without retracting the supported conclusion. |

**Gate:** An answer that gives no data-derived comparison cannot score above
0.45. An answer claiming that TRAV1-2 alone proves every cell's identity cannot
score above 0.65.

### Calibration controls

- **Known-correct:** recomputes the six fractions, highlights the approximately
  19-fold separation from the largest non-MAIT fraction, states both possible
  missing-call denominators, and gives a calibrated interpretation. Expected:
  high acceptance.
- **Correct but terse:** reports approximately 69% in MAIT and under 4% in each
  comparator but gives little missingness analysis. Expected: partial/high, not
  perfect.
- **Fluent but unsupported:** explains MAIT receptor biology without opening the
  file or reporting a comparison. Expected: below the evidence gate.
- **Materially wrong:** says TRAV1-2 is uniformly distributed or most common in
  conventional CD4 cells. Expected: near rejection.
- **Overclaim:** computes the enrichment correctly but says it proves all labels
  and antigen specificity. Expected: substantial cap.

### Build blockers

- Identify and pin the exact TCR MuData artifact currently loaded as
  `air_repertoire/stephenson2021_tcr_qc.h5mu`.
- Audit the cohort's redistribution terms.
- Record row count, schema, digest, and the exact rule used to choose the primary
  VJ chain.

## 2. CD8 repertoire evenness and COVID-19 severity

**Seed:** `covid-diversity-severity`
**Notebook anchor:** cells 14–16
**Intended environment:** Python with pandas and a small numerical stack; no network
**Answer:** an open-ended scientific report graded by `DecisionGrader`

### Solver-visible request

Assess whether CD8 T-cell repertoire evenness shows a convincing relationship
with clinical severity in this cohort.

Compute or verify a donor-level normalized Shannon evenness measure, summarize
its distribution across severity groups, and decide what conclusion is
warranted. Your analysis must keep donors as the independent observational
units. Discuss within-group variation, sample-size limitations, and at least one
plausible biological or technical source of variation not encoded by the
severity label.

Deliver a concise quantitative report, including a donor-level result table or
plot and the analysis choices needed to reproduce it.

### Planned solver artifact

Prefer `cd8_clonotypes.parquet`, one row per analyzed CD8 cell:

```text
cell_id
patient_id
severity
clone_id
```

If a derived donor table is used for the smoke instead, it must include
`patient_id`, `severity`, `n_cells`, `n_clonotypes`, and `normalized_shannon`,
plus provenance connecting it to the pinned cell-level object. The prompt must
not imply that cells are independent replicates.

### Private evidence and reference interpretation

Notebook cell 15 reports severity-group summaries of donor-level normalized
Shannon evenness:

| severity | median | min | max |
| --- | ---: | ---: | ---: |
| Healthy | 0.986247 | 0.708246 | 0.999646 |
| Asymptomatic | 0.971843 | 0.832017 | 0.997485 |
| Mild | 0.987283 | 0.816396 | 0.998319 |
| Moderate | 0.906214 | 0.808433 | 0.988063 |
| Severe | 0.923813 | 0.915976 | 1.000000 |
| Critical | 0.980684 | 0.960805 | 0.997798 |

The defensible conclusion is not a monotonic severity effect. Most donors have
high evenness, while lower-evenness donors occur in several groups, including
healthy controls. Observed within-group ranges overlap heavily, and donor-level
variation is larger than a simple between-severity pattern. Sample depth,
number of clonotypes, donor counts, latent infections such as CMV, age, and
sampling variability are relevant limitations or confounders. A solver may use
additional defensible donor-level analyses, but cell-level pseudoreplication is
a major error.

### Judge rubric

| Weight | Criterion | Full-credit behavior |
| ---: | --- | --- |
| 0.25 | Donor-level computation | Computes or verifies normalized Shannon evenness per donor and does not treat cells as independent subjects. |
| 0.30 | Evidence synthesis | Shows substantial overlap and lack of a convincing monotonic severity relationship using donor-level summaries or plots. |
| 0.25 | Statistical calibration | Distinguishes absence of convincing evidence from proof of no association and considers group sizes and uncertainty. |
| 0.20 | Biological and technical limitations | Identifies plausible drivers such as latent-virus expansions and sampling depth, tied to this analysis rather than as a generic caveat list. |

**Gate:** Cell-level hypothesis tests that inflate the sample size cannot score
above 0.35. A claimed monotonic severity trend without confronting the reported
overlap cannot score above 0.45.

### Calibration controls

- **Known-correct:** computes donor-level evenness, reports overlapping group
  distributions and non-monotonic medians, visualizes donor points, and reaches
  the calibrated conclusion above. Expected: high acceptance.
- **Alternative correct:** fits a donor-level model or rank-based comparison and
  concludes that the cohort does not support a clear severity effect, while
  describing its low power. Expected: high acceptance even if the method differs
  from the notebook.
- **Partial:** reports group medians correctly but omits donor dispersion or
  confounders. Expected: middle reward.
- **Pseudoreplicated:** treats every CD8 cell as an independent severity sample
  and reports a tiny p-value. Expected: rejection or very low reward.
- **Directionally wrong:** claims evenness falls steadily from healthy to
  critical. Expected: rejection.

### Build blockers

- Materialize donor counts and the exact normalized-Shannon implementation,
  including treatment of unassigned clonotypes.
- Check whether low-cell-count donors require a predeclared inclusion threshold
  or sensitivity analysis.
- Pin and audit the same underlying TCR object as design 1.

## 3. B-cell mutation and isotype evidence for maturation

**Seed:** `covid-bcr-mutation-isotype`
**Notebook anchor:** cells 17–23
**Intended environment:** Python with pandas; no network
**Answer:** an open-ended scientific report graded by `DecisionGrader`

### Solver-visible request

Compare B-cell states using heavy-chain V-region mutation frequency and
heavy-chain isotype. Identify which states show evidence consistent with
somatic hypermutation, class switching, both, or neither, and discuss any
important exceptions.

Support the interpretation with quantitative summaries derived from the
supplied receptor table. Explain why these measurements are informative, but do
not treat either one as a complete reconstruction of lineage history or antigen
experience.

Deliver a concise report with a state-level table or plot and enough detail to
reproduce the analysis.

### Planned solver artifact

`bcr_state_receptors.parquet`, one row per cell or primary heavy chain with at
least:

```text
cell_id
patient_id
cell_type
heavy_c_call
heavy_v_mutation_freq
clone_id
```

The build must specify how multiple heavy chains, absent mutation values, and
isotype subcalls are handled. It should preserve donor identity for sensitivity
checks even though the primary requested summary is by cell state.

### Private evidence and reference interpretation

Notebook cell 22 reports median heavy-chain V-gene mutation frequencies:

| cell state | median mutation frequency |
| --- | ---: |
| B_exhausted | 0.032051 |
| B_immature | 0.000000 |
| B_naive | 0.000000 |
| B_non-switched_memory | 0.032051 |
| B_switched_memory | 0.059295 |
| Plasma_cell_IgA | 0.067308 |
| Plasma_cell_IgG | 0.051282 |
| Plasma_cell_IgM | 0.003205 |
| Plasmablast | 0.044872 |

The notebook's qualitative isotype result is that naive and immature B cells
mainly express IgM/IgD and carry almost no mutations. Switched-memory cells and
IgA/IgG plasma cells show class-switched isotypes and roughly 5%–7% median
V-region mutation. IgM plasma cells are an important exception: plasma-cell
state does not itself imply class switching or high mutation. Non-switched
memory and exhausted B-cell labels show intermediate mutation in this summary.
The final private evidence must include the numeric isotype-by-state table
behind the notebook figure; prose alone is insufficient for calibration.

### Judge rubric

| Weight | Criterion | Full-credit behavior |
| ---: | --- | --- |
| 0.30 | Quantitative mutation analysis | Produces accurate state-level mutation summaries and identifies the near-zero, intermediate, and approximately 5%–7% groups. |
| 0.25 | Isotype analysis | Correctly distinguishes IgM/IgD-dominated unswitched states from IgA/IgG class-switched states using the supplied data. |
| 0.25 | Integrated biological interpretation | Uses mutation and isotype jointly, recognizes IgM plasma cells and other non-simple cases, and avoids reducing maturation to one label. |
| 0.20 | Reproducibility and limitations | States chain, missingness, aggregation, and donor choices and avoids claiming a full clonal history from cross-sectional summaries. |

**Gate:** An answer with no quantitative use of the supplied receptor data
cannot score above 0.45. Reversing the naive-versus-switched mutation pattern
cannot score above 0.25.

### Calibration controls

- **Known-correct:** reproduces state medians and isotype proportions, identifies
  switched-memory and IgA/IgG plasma evidence, and explicitly handles IgM plasma
  cells as an exception. Expected: high acceptance.
- **Alternative correct:** stratifies by donor before pooling and reaches a
  compatible interpretation. Expected: high acceptance even if its numerical
  aggregation differs slightly.
- **Partial:** correctly compares mutation frequencies but ignores isotype or
  the IgM plasma exception. Expected: middle reward.
- **Fluent but generic:** describes affinity maturation without computing any
  supplied-data summary. Expected: below the evidence gate.
- **Materially wrong:** claims naive/immature B cells have the highest mutation
  or that IgM is proof of class switching. Expected: rejection.

### Build blockers

- Extract and retain the numeric isotype-by-state table used by notebook cell
  22; the rendered figure alone is not adequate private evidence.
- Identify and pin the BCR object currently loaded as
  `air_repertoire/stephenson2021_bcr_qc.h5mu` and audit redistribution rights.
- Record the exact Scirpy mutation-field semantics, chain-selection rule, row
  counts, missingness, and object digest.

## Shared smoke acceptance criteria

Before any of these designs becomes a `TaskProposal` or `TaskSpec`:

1. The source and dataset licenses are recorded separately.
2. Every solver artifact has an immutable URI, digest, schema, size, and
   derivation record.
3. A clean environment can complete the analysis without network access.
4. The notebook and private evidence are absent from the solver capsule.
5. Known-correct, partial, generic-but-unsupported, and materially wrong
   responses produce the intended ordering under repeated judge calls.
6. The grader preserves its model, prompt, rubric, reference evidence, and
   quality-level probabilities for audit.
7. No public BixBench or BixBench-Verified-50 question or capsule is used in
   construction or calibration.
