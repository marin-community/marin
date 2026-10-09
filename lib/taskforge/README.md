# Taskforge

Taskforge turns task ideas into validated TaskCompendium `TaskSpec`s. A proposal source writes
task proposals, a builder program turns each proposal into a `TaskSpec` with fixed controls,
validation runs solver trials on RolloutEngine, and review accepts, retries or rejects the task by
its solve rate. Review's repairs are refused here: `LoopPolicy` requires every repair budget to be 0.

Every stage runs end to end on ShellSim machines, with the models supplied by the caller.

Taskforge is a standalone uv project with its own `uv.lock` and `.venv`. It depends on
`taskcompendium`, `rolloutengine`, `shellbox`, `verifyit` and `rigging` through path dependencies
on the sibling `lib/` packages.

## Package layout

```
src/taskforge/
  content_hash.py  canonical JSON and its sha256 digests
  atomic_file.py   atomic file replacement
  ledger/     timed spans and item events to per-item JSONL
  sandbox/    MachineFactory selection per host and up-front task refusals (ShellSim)
  spec/       TaskSpec assembly, lowering for RolloutEngine, and fixed controls
  llm/        sampling policy, call records, and where a call is recorded
  proposal/   the TaskProposal document (model.py) and the ProposalSource protocol (source.py)
  triage/     the triage decision
  builder/    builder programs: memoized steps, the Build SDK, the standard template
  validate/   trials, the failure classifier, solver trials, attempt files as resumable evidence,
              calibration
  review/     the Decision contract and the band rules that derive it from a calibration summary
  loop/       the per-item program, its policy, and the event log that item status is derived from
  queue/      the unattended run: its config, the queue over every idea and item, summary.json
docs/         policy.example.json, a committed run config
scripts/      run_queue.py
```

Packages are totally ordered. A package imports only from packages to its left and from external
packages; `tests/test_imports.py` fails on any import that points right:

```
content_hash -> atomic_file -> ledger -> sandbox -> spec -> llm -> proposal -> triage -> builder -> validate -> review -> loop -> queue
```

## What each stage does

- Models: the caller's `queue.job.RunInputs` supplies the builder's model (`builder.sdk.ModelEndpoint`)
  and each solver trial's rollout model.
- Triage accepts every proposal.
- Authoring adopts `builder.template.standard` unchanged as each item's program, so no task can be
  repaired: `LoopPolicy` refuses `max_repairs` or a band rule's `repairs` above 0.
- Machines are ShellSim only, and graders are verifyit graders that run in process.
- Validation runs the solver's trials and no others; the control and adversary events and
  `CalibrationSummary` fields are recorded empty.
- The queue runs on a laptop with a JSONL ledger.

## Running

A run over the capability catalog with scripted models lives in
`experiments/post_training/capability_driven_envs`. From the repository root:

```bash
uv run --project lib/taskforge --frozen python -m \
    experiments.post_training.capability_driven_envs.driver --root /tmp/capability-run
```

It prints the path of `summary.json`, which lists every accepted item with its draft directory
(`task.json`, `lowered.json`, `controls.json`, `provenance.json`) and its solve rate. A relaunch on
the same root resumes every item from its event log.

A model a run is given must raise `taskforge.llm.client.GlmUnavailable` for a transient failure. A
solver model that raises anything else leaves its trial `UNCLASSIFIED`, which no retry runs again, so
the item ends `ABANDONED`; a builder model's other exceptions cost a build revision.

## Testing

```bash
uv run --project lib/taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```
