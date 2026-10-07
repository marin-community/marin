# RL data curation: status and remaining work

Deliver a source table covering all 203 Atlas entries: 153 available sources and
50 documented exclusions. Each processed source must link its pinned inputs,
converted tasks, analysis, verification and final report. Full processing has
not launched. [PR #9798](https://github.com/marin-community/marin/pull/9798)
contains the implementation; the [Loom goal](https://loom.oa.dev/s/fa29pypv/artifacts/goal)
contains the runtime diagrams and campaign evidence.

## Current status

This checkpoint documents the implementation pushed at `8caac4518a`, the completed
VerifyIT test-import fix, and proposed cleanup. PR comments are recorded as future
work; this documentation pass does not implement or resolve them.

| Campaign | Scope | Latest observed outcome |
| --- | --- | --- |
| R10 | All 153 sources, up to 100 raw records each | Terminal report: 138 sampled, 1 completed, 7 gated, 7 failed. |
| R11 | All 153 sources; 50 concurrent source procedures; 256 shared Zephyr workers | Running: 50 active, 103 queued, no completed source outcomes observed. |

R10's terminal report was written at **2026-10-07 23:10:22 UTC**. Its 152 source
reports contain 15,130 panel rows: 6,455 KEEP, 2,464 REJECT and 6,211 DEFER.
DEFER withholds a quality or readiness decision; it does not establish a source
defect. Verification was passed for 76 sources, rejected for 3 and inconclusive
for 73. ARC inductive produced no source report. The Iris job ended
`worker_failed` after pod deletion; the campaign report retains its outcomes.

R11 uses code `8caac4518a` on `cw-rno2a`. The driver, coordinator and shared worker
job were confirmed RUNNING at **23:30:39 UTC**; 256/256 workers were RUNNING at
23:17:59. The count-changing report is timestamped 23:16:57. The existing watcher
still observed that report at 23:36:50. These are observation times, not completed
source results. The next hourly check is **2026-10-08 00:14:13 UTC**.

- [R11 Iris job](https://iris-cw-rno2a.oa.dev/#/job/%2Fpower%2Ftask-curation-shared-sample-r11-original-20261007)
- R11 report: `s3://marin-us-east-02a/marin/data/rl/campaign-2026.10.07.r11-original/sample-r11-original-report.json`
- R10 report: `s3://marin-us-east-02a/marin/data/rl/campaign-2026.10.07.r10-native/sample-r10-native-report.json`

## Pipeline and ownership

The catalog in [sources.py](experiments/post_training/task_curation/sources.py)
maps source names to dataset `pipeline()` declarations with source metadata.
Experiments owns pins, acquisition dependencies, runtime bindings and ArtifactStep
wrappers. [TaskCompendium](lib/taskcompendium/README.md) supplies reusable conversion,
review, filtering, verification and reporting components and procedures.
[VerifyIT](lib/verifyit/README.md) owns common graders. Custom graders come from
the task, an installed upstream package, or pinned source injected into the
private environment. Shellbox executes the declared backend and image.

The procedure is pinned acquisition → sample at most 100 raw records → convert
and review with GLM or recorded manual evidence → source gate → process admitted
data → verify sampled outputs with available controls → write shards and reports.
Analysis and conversion may share a physical pass. More than 90% known good
judgments permits skipping remaining per-task review; more than 50% known defects
rejects the source. Both use the raw panel denominator. Provider failures remain
unavailable evidence and do not count as defects. Missing goldens skip their
positive control; other applicable checks still run.

Each source retains `hf/`, `normalized/`, `analysis/`, `verification/`, `accepted/`,
`report.json` and `telemetry.json` under its artifact prefix in us-east02a storage.
Normalized and accepted outputs have an explicit shard count. Inference caching
is best effort and identifies the complete request and model revision. One
shared Zephyr pool serves source procedures and phases. See the
[experiment README](experiments/post_training/task_curation/README.md) and
[pipeline contract](lib/taskcompendium/src/taskcompendium/pipeline/README.md).

## Proposed cleanup

These items require a later implementation pass. The current documentation does
not imply they are complete.

| Area | Proposed work |
| --- | --- |
| Grader ownership | Trace remaining bindings to their original installed package, source file or task command. Keep custom scoring out of TaskCompendium. Check the RewardKit bridge's placement and StackOverflow grader acquisition. |
| Candidate paths | Separate a private native grader's candidate input path from actor `output_paths`; preserve terminal text, JSON and tool-call contracts. Check calendar answer extraction against the original scorer. |
| Runtime integration | Align `native_command` and `source_unavailable` contracts with RolloutEngine's consumer. Standard grading integration exists; these two forms still need alignment. Evaluate existing TaskSpec and VerifyIT interfaces before adding fields or wrappers. |
| Dataset declarations | Replace nested calendar callbacks with an explicit binding; make citation and ARC scorer provenance and invocation understandable. Keep source-specific rubrics beside their declarations where reuse does not justify a library family. |
| Organization | Review `rubric_tasks` and generic policy, adapter, rubric and binder names. Consolidate overlapping source-family helpers without changing source grading behavior. Preserve consistent filenames. |
| Source provenance | Check ARC source pins and acquire OpenSWE from original task inputs rather than an earlier curation output. Record unavailable original evidence explicitly. |
| Image preparation | Let declarations specify a Dockerfile or immutable image reference. Add an artifact stage for staging, building and publishing to GHCR, then feed its resolved digest to verification and rollouts. Current execution requires manually supplied image pins. |
| Cost and visibility | Measure raw-input scanning, cache reads, review requests and sandbox startup separately. Bound inference batches by count and bytes; use existing counters and phase telemetry to explain queued or inconclusive sources. Finestore work is owned separately. |

## Remaining execution work

1. Monitor the existing R11 sample campaign hourly and retain all source outcomes.
   Investigate failures and inconclusive results against their original source
   contracts; distinguish task defects, missing controls and infrastructure errors.
2. In the next implementation pass, split independent dataset, runtime and image
   cleanup among Sol agents. Keep shared interfaces under one owner and merge
   changes before rerunning affected samples. Comment-driven changes are paused
   for this checkpoint.
3. Repeat the all-source sample with fixes until every available source has a
   recorded disposition. Reuse matching ArtifactStep outputs and expensive
   inference where available; rebuilding TaskSpecs is acceptable.
4. Launch full processing for admitted sources, verify available controls, and
   publish the final 203-entry table with counts, quality decisions, grader
   readiness, exclusion reasons and dataset/report links. Completed procedures
   and quality KEEP counts alone do not certify executable RL readiness.
