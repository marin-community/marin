# Research engineering

Research engineering turns an open question into a supported design or
implementation through data exploration, analysis, and experiments. Preserve
both the selected implementation and the work that explains how it was chosen.
Use the `research-engineering` skill to coordinate this workflow.

For example, choosing a quick RL pipeline can involve finding candidate
datasets, inspecting their contents, exploring filters and mixtures, writing
analysis code, and running training and evaluation trials. The record starts
with those investigations, before the first training run.

Use [experiment records](experiment-records.md) for shared evidence standards
and bounded background research. Ordinary implementation, debugging, incidents,
smoke tests, and one-off validation do not require this workflow. Start a
research record when the task grows into an exploration whose alternatives and
results will inform a reusable decision.

## Start with a decision and an issue

State the question, baseline, constraints, and what evidence would justify a
choice. Search prior work and inspect existing code before choosing the next
analysis or experiment. Revise the plan as findings change the decision.

Use an issue in
[`marin-community/marin-logbooks`](https://github.com/marin-community/marin-logbooks)
as the task's entry point. Link any related Marin issue rather than maintaining
two progress narratives. The issue should let a reader understand what was
tried, why it was tried, the interesting results, and what was learned.

Reuse an existing coordinating issue when possible. Starting an authorized
public research program includes creating its coordinating issue through
`file-issue`. Keep private work on its authorized surface until publication is
approved; logbooks is public.

## Preserve two outputs

The logbooks repository archives research branches that share Marin's Git
history. Its default branch remains a small index of task issues, branches,
and notebooks. It is not a synchronization target for all Marin refs.

| Branch | Contents and purpose |
| --- | --- |
| Research branch in `marin-logbooks` | A source tree based on an exact Marin commit, experimental changes, the notebook, and supporting analysis. Work directly on this branch and retain it after the task. |
| Contribution branch in Marin | The selected pipeline, configuration, evaluation code, or reusable fixes, prepared as ordinary PRs against Marin. A task may produce several PRs or conclude without one. |

Read logbooks' current
[`AGENTS.md`](https://github.com/marin-community/marin-logbooks/blob/main/AGENTS.md)
for branch setup, remote names, publication commands, and concurrency rules.
Keep the notebook and supporting scripts in
`logbooks/<issue-number>-<topic>/` within the research branch. Follow the setup
procedure in logbooks' default-branch `AGENTS.md` to create
`logbooks/<issue-number>-<topic>/AGENTS.md` with the archive instructions.
Keep ordinary source edits in their existing Marin locations.

Start from the Marin revision the task needs; record that base and later
upstream merges. Preserve published research commits and incorporate upstream
changes through merges. Remove inherited CI workflows before the first push
and reconcile workflows again after upstream merges; archive CI is opt-in.
Never mirror-push the repository or merge the experimental source tree into
the index branch. Add or update a small index entry on the default branch.

## Keep three levels of detail

The notebook is the detailed working record. Append entries for meaningful
design alternatives, dataset investigations, analyses, experiment attempts,
negative results, and evidence that changes a prior conclusion. Explain:

- the question or hypothesis and why this attempt was worth trying;
- the exact source commit and material command or configuration, including
  dataset revisions, environment or hardware when relevant;
- the result and supporting artifacts, with estimates and exploratory findings
  distinguished from measured results;
- the interpretation, limitations, and next decision.

Preserve intermediate code and alternative configurations needed to understand
or repeat the work. For a launched job, link its execution identity and source
snapshot; a commit alone does not capture uncommitted files or external
dependencies. Record false starts and restarts when they change interpretation
or the next action. Mark invalidated evidence with a later correction.

Keep small figures and analysis outputs with the notebook when useful. Store
large datasets, checkpoints, raw logs, and dense telemetry in their existing
artifact systems, and link them. Omit routine setup repair, command mistakes,
and retries that add no evidence.

Issue comments summarize milestones and material changes of direction. Commit
and push their supporting code and notebook entries first, then link exact
commits and artifacts. Batch entries while working; also push before a
cross-session handoff or stopping. A handoff does not require a new comment.

The issue body holds the current question, conclusion, selected approach,
confidence, open questions, and links to the notebook and contribution PRs.
Update it when that overall state changes. The body and milestone comments
should explain the work without requiring the reader to reconstruct it from
raw notebook entries.

## Promote the selected implementation

Build contribution branches from current Marin and extract the chosen source
changes. Leave notebook files, abandoned variants, and archive CI changes in
logbooks. Split independent reusable fixes into their own PRs when useful.

Before committing the extracted changes, have a separate, fresh reviewer
inspect every piece of the proposed contribution, including code,
configuration, and tests. For each piece, ask: will we use and maintain this
going forward, or is it a historical artifact of the research? Keep material
whose only purpose is to document or reproduce past trials in the research
branch. Producing the final experiment result does not by itself justify
promoting an artifact into Marin.

Validate the extracted code in its contribution branch; results from the
research branch do not establish that the extracted version works.

Finish the issue with the decision, supporting evidence, rejected
alternatives and their reasons, remaining uncertainty, and links to code PRs.
Record whether those PRs are open or merged. Preserve the research branch and
its index entry even if every attempted approach was rejected.
