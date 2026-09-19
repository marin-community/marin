# Cross-domain curriculum v1

This wave replaces the retired 34-root C-series inventory with 45 D-series subject roots spanning academic,
professional, service, sports, safety, transport, and practical domains. The final catalog contains subject curricula
only; earlier practice graphs were optional source material when they directly matched a new subject.

## Procedure

Four isolated `gpt-5.6-sol` roles used high reasoning effort and provider-controlled sampling:

1. a generator received one subject inventory and optional directly relevant prior graphs;
2. a curriculum-blind generator received only the subject name and guideposts and wrote
   `max(24, 2 * guidepost_count)` tasks;
3. a holistic reviewer received the curriculum, rubric, inventory, and source provenance but no blind tasks; and
4. a fit judge received only stripped task instructions and the curriculum.

All 45 initial graphs received a holistic review and blind-fit judgment. The initial pass had no `pilot_ready`
subjects, so every graph received one bounded repair using its own review and fit findings, followed by a fresh
complete review and fit judgment against the frozen blind tasks. The run stopped after that repair. D14's repair
regressed blind fit from 22/24 to 9/24 without improving its score, so the catalog retains its stronger v1 graph; the
other 44 subjects use v2.

Every selected subject passes `Curriculum.check_generation_contract(maximum_depth=4)` and schema and cross-reference
validation through `validate_subject_run`. That validation is mechanical, not a quality gate. Only the three
`pilot_ready` subjects named below pass `validate_subject_promotion`. The catalog round-trips through `load_catalog`,
contains exactly D01–D45, and has globally unique section IDs.

## Result

The selected catalog contains:

- 45 subject roots and 356 inventory guideposts;
- 2,252 nodes: 1,853 capabilities and 399 groups;
- 2,662 sampling facets and 16 prerequisite edges;
- 26–97 nodes per subject, median 47;
- 19–85 capabilities per subject, median 39; and
- 993/1,072 blind tasks with an exact or defensibly ambiguous home, or 92.6%.

The final holistic score distribution is 64–93, mean 81.3 and median 80. Three subjects are `pilot_ready`: Food
Science & Nutrition, Communication/Journalism/Library & Information Studies, and Social Work/Counseling & Community
Services. Forty-one remain `revise`, and Computer Hardware, Embedded Systems & Robotics remains `regenerate`. Low
evidence confidence applies to 43 subjects and medium confidence to two; this reflects the intentionally thin v1
evidence base rather than a schema failure.

The repair improved the aggregate diagnostic from 973/1,077 blind fits (90.3%) to 993/1,072 selected fits (92.6%).
The same 1,080 frozen task IDs were judged in both passes: v1 had 930 exact, 43 ambiguous, 104 gap, and three invalid;
the selected pass has 956 exact, 37 ambiguous, 79 gap, and eight invalid. The denominator excludes invalid tasks, so
the percentages are not a fixed-denominator comparison. Across paired tasks, 53 gaps became exact while 28 exact
homes became gaps; seven initially exact tasks were judged invalid, and two initially invalid tasks became exact.
Average holistic score rose from 78.0 to 81.3, and subjects with a repeated systematic gap fell from 13 to seven.
The remaining repeated gaps are recorded as blocking dispositions rather than hidden or converted into task-specific
exception capabilities.

| Subject | Revision | Blocking gap and disposition |
| --- | --- | --- |
| D01 Mathematics & Statistics | v2 | Retain the integrated-statistical-inference gap; do not add a task-specific bundle. |
| D15 Biomedical Engineering & Medical Technology | v2 | Retain the end-to-end device-lifecycle assurance gap pending stronger lifecycle evidence. |
| D16 Basic Biomedical Sciences | v2 | Retain the tissue-architecture integration gap pending coherent operation-family evidence. |
| D17 Medicine & Clinical Care | v2 | Retain the test-interpretation-to-action gap rather than merge laboratory and imaging loops. |
| D28 Law, Regulation & Criminology | v2 | Retain the procedural-execution gap pending jurisdiction-fixed examples. |
| D34 Marketing, Commerce & Consumer Studies | v2 | Retain the strategy-to-production gap rather than create an omnibus workflow capability. |
| D35 History & Archaeology | v2 | Retain the comparative-historiography gap pending evidence that it is one stable operation. |

D14 is the sole selected v1 graph. Both revisions scored 64 and remained `regenerate`; v1 mapped 21 exact, one
ambiguous, two gap, and zero invalid tasks, while v2 mapped nine exact and 15 gap. The selection rule retained v1
because the repair did not improve holistic quality and materially regressed the frozen fit diagnostic. Its remaining
blockers include missing major guidepost operations, several mixed operation families, and under-specified probes.

## Artifacts

- Catalog: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e/curriculum.yaml`
- Catalog SHA-256: `72a763b98f9ecf7f8f598b788c4f59e7ace213c01a30b403768a8f8f16f55382`
- Summary: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e/catalog_summary.json`
- Summary SHA-256: `4eee45d7a4be604dbed4975fb0c4c40b78d3115fc31a77ebbb80fcbb238de949`
- Evidence: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e/evidence.tar.gz`
- Evidence SHA-256: `cd127af6afbffc188e94359d2995ad833e8a54ce8d0962dc036592e0d02cfc74`
- Model: `gpt-5.6-sol`, high reasoning effort
- Prompt versions: `subject-generator-wave10-v1`, `blind-tasks-v1`, `holistic-review-v1`, and `blind-fit-v1`
- Viewer revision: [revision 2](https://applets.marina.oa.dev/a/67f69132-2ef4-4c9e-b8b5-77cabd126442/v/2/)

Version names serve different layers: cross-domain v1 names this experiment; `2026.09.18-cross-domain-v1` is the
catalog's internal version; `2026.09.18.2` is the Marin `ArtifactStep` version; and each selected subject carries a
`wave10-v1` or `wave10-v2` revision. Generation ran on 2026-09-18 from repository commit
`3d9ed5797a7414a02f469d837cd1613e0be45df4` with `codex-cli 0.155.1`. Input hashes are
`ef10281c71ad610a0cd4382f674887ff8f0c0b790af03e79e0400462dfaf00d6` for the inventory,
`7551a2041728c0d1d2106ce988c0f95dce73e8b05d7842926eca85af452ec494` for the rubric, and
`c1821f8f67c1e738cfc9eea30821f7ee4a7e49c6bacce52ef9ba14c5043f93fe` for the source C-series catalog. The one-off
runner SHA-256 is `6c80d27968e2ebe8d39b8abfef2bd56079b624377373777b4da4cd129038e5ef`. Provider-controlled sampling means these
inputs and outputs are auditable but generation is not bit-for-bit reproducible.

The evidence archive contains the run manifest, frozen subject inputs, v1 and v2 curricula, blind tasks, stripped
judge inputs, reviews, fits, blocking-gap dispositions, raw role logs, final summary, and one-off runner. The catalog
payload and task-level artifacts remain external; the repository contains the inventory, rubric, workflow, Artifact
handle, and this concise report.

To inspect the complete evidence locally:

```bash
uv run fsutil cp \
  s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e/evidence.tar.gz \
  /tmp/curriculum-wave10-evidence.tar.gz
echo 'cd127af6afbffc188e94359d2995ad833e8a54ce8d0962dc036592e0d02cfc74  /tmp/curriculum-wave10-evidence.tar.gz' \
  | sha256sum --check
tar -tzf /tmp/curriculum-wave10-evidence.tar.gz
tar -xzf /tmp/curriculum-wave10-evidence.tar.gz -C /tmp
```

The archive's `curriculum_wave10/run_wave10.py`, `finalize_wave10.py`, `run_manifest.json`, and
`catalog_summary.json` preserve the runner, provenance, validation, and metric-recomputation inputs. Role prompts in
the runner define the exact allowed-read sets; these are context-isolated roles using the same model, not independent
models or providers.

## Known limitations and follow-up

The catalog is a useful broad baseline, not 45 fully promoted curricula. Common residual issues are broad integration
capabilities, entry-to-representative jumps, under-specified probes, and uncertain cross-disciplinary boundaries.
The selected graphs contain 16 prerequisite edges that survived the completed-artifact counterfactual. This may
reflect true dependency sparsity or insufficient evidence and should be retested.

The acceptance terms used here are defined in [`../rubric.md`](../rubric.md): mutual self-confidence, epsilon
continuity, and the completed-artifact counterfactual govern capability boundaries and dependencies. Exact home,
defensibly ambiguous, systematic gap, and blocking disposition are defined by the blind-fit and promotion contracts
in [`../workflow.md`](../workflow.md).

The next evidence-quality iteration is a source survey, not another blind optimization loop. For a representative
five-domain comparison, freeze public college course sequences, established textbook contents and exercise families,
and relevant accreditation or professional standards before generation. Use them to amend root guideposts and expose
canonical operations or candidate progression. Course chapters do not define curriculum capabilities or prerequisite
edges: mutual self-confidence, epsilon continuity, executable probes, the completed-artifact counterfactual, holistic
review, and frozen blind fit remain the acceptance tests. If the five-domain comparison yields recurring gains,
apply the survey procedure across all roots in the next inventory version.
