# Wave-nine full curriculum baseline

This is the retired 34-root C-series baseline. The later D-series cross-domain catalog is documented in
[`../wave10/README.md`](../wave10/README.md); counts and identifiers below remain historical results.

Wave nine fills the 12 macro areas absent after wave eight. It uses the same four isolated Sol/high roles documented
in `../workflow.md`: curriculum generation, curriculum-blind task generation, holistic review, and blind-fit
judgment. The immutable historical catalog is
`s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-c1821f8f67c1/curriculum.yaml`; the one-off JSON
inputs and role outputs are stored in the Loom evidence artifact named below.

## Inputs

The wave sampled 16 model-visible tasks per subject from TaskTrove Clean `2026.09.10.9`, pinned by Parquet SHA-256
`26d0613cc017f5564d5bc8af29596efefa42f839f0f38842ffe1e6885d20da66`. The broken `nl2bash` source was excluded.
A curriculum-blind Sol/high role generated 24 tasks for each subject, for 288 frozen tasks total. All blind sets
contain at least two tasks for every inventory guidepost and include entry, representative, and boundary intents.

The evaluation policy allowed answer-hidden MMLU questions for ML/AI Development and Algorithms & Competitive
Programming at revision `c30699e8356da336a370243923dbaf21066bb9fe`. The other ten subjects had no applicable
task-level evaluation probes. Generation, review, blind-task generation, and fit judgment used `gpt-5.6-sol` with
high reasoning effort and provider-controlled sampling.

## Results

| Subject | Version | Nodes/capabilities | Review | E/A/G/I | Fit | Systematic gaps |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Networking | `wave9-4` | 14/12 | 80, `revise`, low | 22/0/2/0 | 22/24 | 0 |
| ML/AI Development | `wave9-3` | 31/27 | 76, `revise`, low | 19/0/5/0 | 19/24 | 1 |
| Speech & Audio Processing | `wave9-2` | 26/22 | 79, `revise`, low | 20/0/4/0 | 20/24 | 1 |
| Algorithms & Competitive Programming | `wave9-3` | 38/34 | 62, `regenerate`, low | 19/0/5/0 | 19/24 | 0 |
| Web Frontend | `wave9-3` | 12/10 | 76, `revise`, low | 23/0/1/0 | 23/24 | 0 |
| Agent & Workflow Automation | `wave9-1` | 8/6 | 91, `pilot_ready`, low | 21/2/1/0 | 23/24 | 0 |
| Research & Scholarship | `wave9-2` | 21/16 | 79, `revise`, low | 17/1/6/0 | 18/24 | 0 |
| Education & Training | `wave9-2` | 17/14 | 80, `revise`, low | 20/0/2/2 | 20/22 | 1 |
| Marketing Sales & Content | `wave9-2` | 25/20 | 82, `revise`, low | 20/0/4/0 | 20/24 | 1 |
| Design & Media Production | `wave9-2` | 26/20 | 76, `revise`, low | 15/0/9/0 | 15/24 | 3 |
| Engineering Design & Geospatial | `wave9-2` | 28/22 | 76, `revise`, low | 15/0/9/0 | 15/24 | 2 |
| Game & Interactive Development | `wave9-2` | 16/13 | 67, `regenerate`, low | 19/0/5/0 | 19/24 | 0 |

E/A/G/I denotes exact, ambiguous, gap, and invalid. A non-passing review remains provisional in the catalog.
`pilot_ready` still requires at least 85 and no blockers.

The 12 graphs add 262 nodes: 216 capabilities and 46 groups. The `wave9-1` catalog contains all 34 macro areas and
all 151 guideposts with 959 nodes: 854 capabilities and 105 groups. All final reviews and fits measure the exact
selected curriculum versions. Only Agent & Workflow Automation is `pilot_ready`; the other 11 subjects are
provisional and retain their blockers for evidence-led iteration.

## What the wave found

The broad inventory seeds a catalog. It does not establish that the resulting capability boundaries transfer. Across
unrelated subjects, reviewers repeatedly found four failure modes:

1. A shared domain noun hid different operations. Examples include calibration measurement versus statistical
   inference, dialog focus management versus composite-widget navigation, and MST construction versus edge
   classification. These require different training samples and correctness contracts.
2. Probes named a dataset, trace, schema, measurement, or artifact without supplying it. A probe that cannot be
   executed on paper is not a useful generator seed, even when the outcome is otherwise coherent.
3. Blind tasks often requested integrated packages spanning several valid capabilities. Those misses are useful
   coverage signals, but treating every composite as a new capability would destroy mutual self-confidence. The wave
   promotes repeated gaps only when they identify one stable central operation.
4. TaskTrove evidence is uneven. It is strong for code-centric and structured-data work, but sparse for direct CAD,
   survey research, creative media production, speech evaluation, marketing execution, and several ML lifecycle
   operations. Evidence confidence therefore remains low for much of this wave.

The holistic score and blind-fit rate continue to measure different axes. A task can have an obvious home while its
capability still mixes non-transferring operations; conversely, a coherent graph can miss a sampled end-to-end
workflow. Neither measurement is converted into the other.

## Stopping rule and follow-up

Each nonpassing subject received one broad repair pass; C22 passed its initial review. The weakest graphs received a
bounded final split/probe correction, but the wave did not recursively optimize against one reviewer or the frozen
blind sample. Every final graph is schema-valid, passes
`Curriculum.check_generation_contract(maximum_depth=4)`, and gives every inventory guidepost an explicit home.
Subjects whose last review is `revise` or `regenerate` remain provisional versioned starting points.

The next useful work is evidence-led iteration:

- sample direct tasks for the branches named in each low-confidence review;
- rerun the same complete-subject review when a subject changes;
- use repeated real-task mapping misses to prioritize subject revisions, while keeping mapping quality outside the
  curriculum acceptance score; and
- periodically compare section density and repeated operation signatures across subjects to merge distinctions that
  do not change sampling or evaluation.

## Reproducibility

The exact role prompts and isolation rules are in `../workflow.md`; the scoring criteria are in `../rubric.md`. The
wave manifest pins the source release, source hashes, role models, prompt versions, blind-task hashes, discovery-task
hashes, and evaluation revision. The Loom evidence artifact contains that manifest, all frozen inputs, selected final
curricula and design audits, and the latest independent review and fit outputs.

Loom artifact `curriculum-wave9-evidence`, revision 4, branch scope `zooq8ec9`, has source JSON SHA-256
`db23ae13e147cc9eac4193274902ec8ec6ad00f451f915006f52621e7724d0c7`. It is associated with session channel
`s2uyqg34`.
