# Taskforge

Taskforge turns task ideas into validated TaskCompendium `TaskSpec`s. A proposal source writes task
proposals; triage admits them; an author supplies each proposal's builder program, a DAG of steps
whose single sink returns the `TaskSpec`; validation runs solver trials of the task on RolloutEngine;
and review accepts, retries or rejects it by its solve rate. Every stage is a plain function a caller
may call directly, and `loop` coordinates the stages a caller passes through a transition table.

Taskforge is the `marin-taskforge` member of the repository's uv workspace and is locked by the
root `uv.lock`. It depends on the `marin-taskcompendium`, `marin-rolloutengine`, `marin-shellbox`,
`marin-verifyit` and `marin-rigging` workspace members.

## Package layout

```
src/taskforge/
  content_hash.py  canonical JSON and its sha256 digests
  atomic_file.py   atomic file replacement
  ledger/     timed spans and item events to per-item JSONL
  sandbox/    the ShellSim machine factories, and up-front refusals of a task they cannot run
  spec/       TaskSpec assembly: worker files, task-machine requirements, in-process verifyit graders
  llm/        the ModelClient protocol, NoModel, RecordedModel (ledger-attributed calls), sampling policy
  proposal/   the TaskProposal document (model.py) and the ProposalSource protocol (source.py)
  triage/     the triage decision, and `accept`
  builder/    step.py: `@step`, `program()`, the per-item step cache; sdk.py: BuildContext, BuildFailure;
              run.py: `run_program`, `run_build`, the draft; author.py: Revision, `adopt`
  validate/   lowering for RolloutEngine and k solver trials (trials.py, solver.py), the failure
              classifier, attempt files as resumable evidence, the CalibrationSummary, `run_validation`
  review/     the Decision types (Accept, Reject, Retry, Modify) and `decide`, a pure rule table
  loop/       events.py: Stage, Exit, DEFAULT_TRANSITIONS, `transition`, `derive_state`;
              stages.py: the stage Protocols and `Stages`; program.py: `Loop`, `run_idea`, `run_item`;
              policy.py: LoopPolicy and `install_policy`
```

Packages are totally ordered. A package imports only from packages to its left and from external
packages; `tests/test_imports.py` fails on any import that points right:

```
content_hash -> atomic_file -> ledger -> sandbox -> spec -> llm -> proposal -> triage -> builder -> validate -> review -> loop
```

## Stages

The stages, their exits and the default transition table are in `src/taskforge/loop/events.py`; the
Protocol each stage function implements is in `src/taskforge/loop/stages.py`.

## Running

A run over the capability catalog with a scripted solver lives in
`experiments/post_training/capability_driven_envs`. From the repository root:

```bash
uv run --package marin-taskforge --frozen python -m \
    experiments.post_training.capability_driven_envs.driver --root /tmp/capability-run
```

It prints the path of `summary.json`, which lists the terminal of every item under the root, those of
earlier runs with other `--capability` selections included, and every accepted task with its draft
directory and band. A relaunch on the same root resumes every item from its event log; a relaunch
under another policy, session limits included, or with a changed idea under the same id is refused.

A solver model must raise `taskforge.llm.client.GlmUnavailable` for a transient failure; anything
else leaves its trial `UNCLASSIFIED`, which no retry runs again and review rejects.

## Testing

```bash
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

`tests/test_pipeline_walkthrough.py` carries one toy task through every stage, directly and through
the loop, and is the place to start reading.
