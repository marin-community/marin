# Source-guided curriculum follow-up

This follow-up tests whether public college curricula, textbook exercise structures, and professional standards
should become universal inputs to curriculum generation. It preserves the 45-root cross-domain v1 catalog as the
baseline and regenerates five representative subjects in a separate experiment artifact. The result favors retaining
v1. Source guidance found real omissions and improved blind fit, while direct regeneration weakened capability
boundaries, prerequisites, and probes.

## Method

The sample covers mathematics and statistics (D01), mechanical, aerospace, and marine engineering (D08), medicine
and clinical care (D17), education and learning sciences (D31), and sports, exercise, and recreation (D41). For each
subject, three public sources were frozen before generation. A high-reasoning `gpt-5.6-sol` role normalized operation
families, exercise families, sequence signals, guidepost coverage, and evidence gaps without reading either
curriculum. A fresh generator then received the inventory, the source brief, and the durable rubric, but not v1.

A paired reviewer compared v1 and the source-guided graph in one context. A separate paired fit judge reused the
same 24 frozen, curriculum-blind task instructions from v1 and saw neither the source brief nor either review. This
paired design controls some model variance: absolute fit counts should not be compared to older judgments produced
in another call, but the two graphs in each row share a judge and prompt.

The sources were:

- D01: [MIT Course 18](https://math.mit.edu/academics/undergrad/major/course18/),
  [UC Berkeley Statistics](https://statistics.berkeley.edu/academics/undergrad/major), and
  [OpenStax Introductory Statistics](https://openstax.org/books/introductory-statistics-2e/pages/preface).
- D08: [MIT Mechanical Engineering](https://catalog.mit.edu/degree-charts/mechanical-engineering-course-2/),
  [Georgia Tech Mechanical Engineering](https://catalog.gatech.edu/programs/mechanical-engineering-general-bs/), and
  [ABET engineering criteria](https://www.abet.org/wp-content/uploads/2024/11/2025-2026_EAC_Criteria.pdf).
- D17: [Stanford MD curriculum](https://med.stanford.edu/md/mdhandbook/section-4--discovery-curriculum-overview/section-4-5-required-pre-clerkship-courses.html),
  [Duke Patient FIRST](https://medschool.duke.edu/education/health-professions-education-programs/doctor-medicine-md-program/curriculum/first-year),
  and [AAMC Core EPAs](https://store.aamc.org/downloadable/download/sample/sample_id/63/).
- D31: [UIUC Elementary Education](https://catalog.illinois.edu/undergraduate/education/elementary-education-bs/),
  [UW Education Studies](https://education.washington.edu/academics/program/ba-education-studies), and
  [Educational Psychology](https://open.umn.edu/opentextbooks/textbooks/153).
- D41: [Penn State Kinesiology](https://bulletins.psu.edu/undergraduate/colleges/health-human-development/kinesiology-bs/),
  [UIUC Kinesiology](https://catalog.illinois.edu/undergraduate/ahs/kinesiology-bs/), and the
  [ACSM exercise physiologist outline](https://www.acsm.org/wp-content/uploads/2024/12/ACSM-Certified-Exercise-Physiologist-Exam-Content-Outline.pdf).

## Result

| Subject | Nodes v1 / source | Capabilities v1 / source | Edges v1 / source | Holistic v1 / source | Frozen fit v1 / source | Preferred |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D01 Mathematics & Statistics | 60 / 56 | 49 / 48 | 0 / 8 | 75 / 63 | 13/24 / 18/24 | v1 |
| D08 Mechanical, Aerospace & Marine Engineering | 53 / 55 | 45 / 44 | 0 / 7 | 78 / 64 | 22/24 / 23/24 | v1 |
| D17 Medicine & Clinical Care | 46 / 28 | 36 / 23 | 0 / 0 | 90 / 80 | 21/24 / 20/24 | v1 |
| D31 Education & Learning Sciences | 38 / 33 | 30 / 26 | 0 / 0 | 92 / 78 | 23/24 / 24/24 | v1 |
| D41 Sports, Exercise & Recreation | 38 / 40 | 31 / 34 | 0 / 5 | 87 / 73 | 20/24 / 21/24 | v1 |

The source-guided graphs improved aggregate frozen fit from 99/120 to 106/120 and improved four of five subjects.
They also surfaced useful missing operations: integrated statistical work, experimentation and marine/aerospace system
work, clinical documentation and coordination, education technology and data governance, and athlete assessment and
professional practice.

The holistic reviewer nevertheless preferred v1 in all five comparisons. Mean score fell from 84.4 to 71.6. Direct
regeneration repeatedly turned course or professional-practice groupings into capability catch-alls, copied sequence
signals into prerequisite edges that failed the completed-artifact test, and produced weaker or under-specified
probes. Medicine is the clearest example: the source-guided graph found documentation and coordination work but
collapsed different diagnostic modalities, procedures, and ethics decisions into three non-transferable sections.

The two diagnostics measure different failure modes. Blind fit rewards fewer sampled coverage gaps. Holistic review
rejects sections in which mastery should not transfer. The source-guided graphs often improved the first by worsening
the second. Higher node count was not predictive in either direction.

## Recommendation

Retain the broad inventory-and-rubric generator. Add a source-audit stage:

1. Generate the broad subject graph from the inventory and rubric.
2. Freeze a compact source brief that records operation families, exercise families, candidate progression, direct
   and absent guidepost evidence, and source bias.
3. Compare the brief with the generated graph and request bounded additions, splits, probe repairs, or candidate
   prerequisite audits. Do not ask the model to mirror course headings or chapter order.
4. Re-run holistic review and the existing frozen blind fit after repair.

This pattern preserves the strongest contribution of syllabi, textbooks, and standards—coverage and progression
evidence—without treating them as task clusters. Prioritize it for practice-heavy or weakly evidenced subjects and
for guideposts whose v1 review reports low confidence. A full three-source survey for every root is not yet justified.

## TaskTrove placement check

A final practical audit sampled 24 tasks from TaskTrove Clean `2026.09.10.9`: eight Unix/SuperUser system questions,
eight multi-file repository implementations, and eight SWE-Rebench repository changes. Selection was deterministic
within each predeclared cohort. The known-problematic `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` source was explicitly
excluded.

At the root level, 23 tasks mapped exactly and one ambiguously; all 24 included D02 Computer & Information Sciences.
At the capability level, 20 mapped exactly and four were gaps:

- all eight multi-file implementation tasks mapped to `d02.swe.implementation`;
- all eight real repository changes mapped to repair, evolution, implementation, or language lowering; and
- four of eight Unix/tool tasks mapped to network diagnosis, authentication, authorization, or implementation.

The remaining four tasks involved tmux or file-edit invocation, Windows URL activation, SSH session usage, and zsh
key-map configuration. They expose a coherent missing capability for ordinary operating-environment use and
configuration. Stretching systems implementation or networking to absorb these tasks would violate mutual
self-confidence. The D02 root routed every task; these four tasks expose a missing D02 capability.

The result also illustrates why `subject_domain` and `task_mechanic` are separate mapping projections. Subject domain
correctly routes shell use and bug fixing to D02. The mechanic distinguishes environment configuration, defect
localization, interface evolution, and bounded implementation, revealing whether D02 contains an appropriate
capability instead of treating root membership as a successful mapping.

## Artifacts

- Experiment root: `s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-source-survey-v2-148820c9ecae/`
- Comparison SHA-256: `148820c9ecaee60bd7051788b17d8e2b93110d5b380564e3afa86fd592519c59`
- Evidence archive SHA-256: `973c97ebe42ca84eae06982527bb32cb7f5406ec2ba9188124b2e3aae339b78f`
- Viewer: [revision 3](https://applets.marina.oa.dev/a/67f69132-2ef4-4c9e-b8b5-77cabd126442/v/3/)
- Model: `gpt-5.6-sol`, high reasoning effort, provider-controlled sampling
- Canonical v1 catalog SHA-256: `72a763b98f9ecf7f8f598b788c4f59e7ace213c01a30b403768a8f8f16f55382`

The evidence archive contains the frozen source captures and manifest, source briefs, independent source-guided
curricula, paired reviews, paired blind fits, practical TaskTrove sample and placements, role logs, schemas, one-off
runners, and the machine-readable comparison. The source-guided graphs remain external experimental outputs. The
canonical catalog still contains v1.
