# Upstream issues for task specs, verifiers, and rollouts

Each Markdown file under `taskcompendium/`, `verifyit/`, and `rolloutengine/` is one GitHub issue body
for `marin-community/marin`. Its first line is `# <title>`. `file_issues.sh` filed them on
2026-10-05 as #9757 to #9763; the numbers are in the index below. `pr-9623-comment.md` is the
review comment posted on PR #9623 the same day (issuecomment-6000468495), covering #9757, #9763, the
staged-grading questions, and the Iris network-policy mismatch. Draft PRs for #9757 to #9762 were opened the same day; their numbers are in the index. #9761 stacks on
#9768. #9763 has no PR: control replay is implemented in Taskforge with a scripted model callable.

Ownership rule used to sort them:

- TaskCompendium (`lib/taskcompendium`) describes one task and runs that task's own grading
  scripts. It does not gain generic verifier kinds.
- verifyit (`lib/verifyit`) owns verifier types (answer modes, judge, test runners, script,
  aggregation) and their Python API.
- RolloutEngine (`lib/rolloutengine`, PR #9623) owns execution: machines, images, workspace
  state, and running the agent and the verifier.
- Task-specific checks, rubrics, weights, and controls belong to the generated tasks and need no
  upstream issue.

Checked against `origin/main` at `9da8fd185d` (TaskCompendium schema `0.20`) and the head of
#9623 at `7eb1dff26f` (schema `0.22`, open, not merged). Each body says whether its gap is open on
main, at the PR head, or both.

## Filing index

| File | Title | Owner | Labels | Depends on / blocks |
|---|---|---|---|---|
| `taskcompendium/verifier-status-and-detail.md` (#9757, draft PR #9769 on `rollout-engine`) | [taskcompendium] Keep verifier status and detail in GradeResult | taskcompendium (reader in rolloutengine) | post-training, agent-generated | Targets code added by #9623; file after it merges or raise on the PR. Blocks checking partial-credit labels with `grade-supplied-state`. Its `VerdictReward` lets task scripts that use the verifyit issues report `invalid_task`. |
| `taskcompendium/json-submission-fence.md` (#9758, draft PR #9766) | [taskcompendium] JSON submissions fail on a fenced reply or numeric answer | taskcompendium | bug, post-training, agent-generated | Independent. Open on main and at the #9623 head. |
| `verifyit/candidate-modes.md` (#9759, draft PR #9765; follow-up #9764) | [verifyit] Grade extracted text candidates for every answer mode | verifyit | post-training, agent-generated | Independent. TaskCompendium picks up new modes through `candidate_spec` with no schema change. |
| `verifyit/judge-output-budget.md` (#9760, draft PR #9768) | [verifyit] Apply output budgets to judge reference and checklist rubrics | verifyit | post-training, agent-generated | Independent. File before `judge-criterion-samples`, which builds on the same call path. |
| `verifyit/judge-criterion-samples.md` (#9761, draft PR #9770 on #9768) | [verifyit] Return per-criterion checklist grades and support repeated judgments | verifyit | post-training, agent-generated | Depends on `judge-output-budget` for per-call detail. Feeds `weighted-gated-aggregation`. |
| `verifyit/weighted-gated-aggregation.md` (#9762, draft PR #9767) | [verifyit] Add weighted and gated reward aggregation | verifyit | post-training, agent-generated | Independent of the judge issues; combined machine-plus-judge rubrics need both. |
| `rolloutengine/grade-supplied-state.md` (#9763) | [rolloutengine] Grade a supplied final state without model inference | rolloutengine | post-training, agent-generated | Depends on #9623 merging. Partial-credit checks also need `verifier-status-and-detail`. |
| `rolloutengine/public-task-machine.md` (not filed) | [rolloutengine] Export the task-machine lifecycle and in-machine shell grading | rolloutengine | post-training, agent-generated | Depends on #9623 merging. Blocks dropping Taskforge's imports of `machines._task_machine` and `grading._shell_grade` (`taskforge.build.sdk`). #9763 would cover the grading half. |

## Drafts not filed

These earlier drafts were written against schema `0.8` on this branch. They were deleted because
the gap is closed, belongs to the generated tasks, or was merged into an issue above.

| Draft | Disposition | Reason |
|---|---|---|
| `action-interface-providers` | already exists | `EnvironmentRequirements.tool_providers` on main; `environment.interaction`, `ExternalVerifierSpec`, and the `TaskSession` protocol at the #9623 head. No accepted task used a provider. |
| `composite-verifier` | split | Machine checks are the task's own scripts run by `ShellVerifierSpec` (#9623 head). Judged criteria and repeated judgments: `verifyit/judge-criterion-samples.md`. Weights, gates, and penalties: `verifyit/weighted-gated-aggregation.md`. Conditional caps and ordinal anchors stay in the task's script. |
| `container-verifier-runtime` | already exists (#9623 head) | `ShellVerifierSpec` has private `files`, `timeout`, `env`, `user`, a reward source (stdout, reward file, exit code), and an optional separate grading `environment` with a pinned `RegistryImage` or `DockerBuild`, plus `collect` and `artifacts`. Main has only the `VerifierSpec.environment_requirements` schema, and verifyit has the `script`, `pytest`, and `stdio` modes that run inside an image. |
| `file-and-state-submissions` | already exists (#9623 head), remainder moved | A shell verifier grades files and final state in the task machine or in copied `artifacts`. Grading an authored workspace without a model moved to `rolloutengine/grade-supplied-state.md`. |
| `grading-detail` | merged | Into `taskcompendium/verifier-status-and-detail.md`. `GradeResult.diagnostics` and `rewards` exist at the #9623 head, but pure answer grading does not fill them. |
| `harbor-chat-usage` | superseded | RolloutEngine replaces the Harbor direct-chat runner for execution. The caller's model adapter owns request options such as `max_tokens`, and `ModelTurn` keeps `stop_reason`, exact token IDs, and `metadata`. The Harbor `ChatAgent` still drops `usage`; this matters only if that path remains supported. |
| `invalid-task-outcome` | merged | Into `taskcompendium/verifier-status-and-detail.md`. verifyit already has `Status.INVALID_TASK`; TaskCompendium's `Outcome` does not. |
| `json-submission-extraction` | rewritten | As `taskcompendium/json-submission-fence.md`. |
| `judge-verifier-budgets` | rewritten | verifyit has a `judge` mode on main. The remaining gap is narrower: `verifyit/judge-output-budget.md`. |
| `multi-step-success-policy` | already exists (#9623 head) | `TaskSpec.stages`, the `staged` verifier with `mean` or `final`, and per-stage `minimum_rewards`, which stop later stages. `final` plus `minimum_rewards` gives conjunctive success for binary stage rewards. With partial-credit stages, a failed stage returns its own partial reward, not 0. |
| `package-manifest` | ours | Harbor packages are no longer the execution path; RolloutEngine reads `TaskSpec` JSON directly. Task identity and provenance hashing (`TaskSpec.model_dump_json` plus `schema_version`) belong in the pipeline's ledger. |
| `resources-with-roles` | already exists | `ResourceGroups` (`all`, `worker`, `oracle`, `verifier`) on main; `environment.files` and `ShellVerifierSpec.files` at the #9623 head. Executing oracle material moved to `rolloutengine/grade-supplied-state.md`. |
| `shellsim-binding` | already exists (#9623 head) | `EnvironmentKind.SHELLSIM`, `ShellSimMachineFactory`, and the default `shell(command)` session. Step and output budgets belong to the machine factory. |
| `structured-answer-verifiers` | rewritten | verifyit already implements `math`, `json-schema`, `xml-elements`, `csv-columns`, and `ifeval`. The remaining gap is candidate (in-memory) grading: `verifyit/candidate-modes.md`. |
| `task-metadata` | ours | `TaskSpec.tags` on main and `TaskSpec.metadata` at the #9623 head carry coverage tags and difficulty. Their vocabulary is pipeline policy. |
| `workspace-state` | already exists | `EnvironmentRequirements.docker_image`, `working_directory`, and `setup_commands` on main; `EnvironmentSpec` (`image`, `workdir`, `files`, `setup`, `healthcheck`) at the #9623 head. |

## Evidence

Bodies cite `experiments/post_training/capability_env_gen/` on branch `mark/autoenv`, which is pushed.
Numbers from `lib/taskforge/.evidence/` (gitignored, local only) and the construct-003 run audit
are copied into the bodies. The construct-003 feature tally is in
`lib/taskforge/.evidence/spec/construct003_accepted_feature_tally.tsv`. Of the 16 accepted tasks,
13 were `code_answer` tasks with a container `script` verifier, 2 were ShellSim tasks with a
container `script` verifier, and 1 used `exact`. All 16 had one step, 10 had verifier-role resources,
and 2 had agent-role resources.
