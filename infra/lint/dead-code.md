# dead-code — detector prompt

## What to look for

Flag code that is unused, stale, or no longer necessary: dead imports and unused parameters; leftover scaffolding, commented-out code, and experiment markers; obsolete branches and stale configuration flags; test cases that no longer validate meaningful behavior after upstream changes.

## Anchor examples

- **Unused variables/parameters**: "Passing the full variable set to every template... `01-pvc.yaml.tmpl` rejects `image/port/remote_log_dir`... Please either pass a per-template variable map or relax the unused-variable check" — K8s templating code accepts parameters it ignores, breaking template validation.

- **Leftover scaffolding/staging code**: "this is just so we can stage it safely... I'm nervous about it so I want to test on the dev-cluster only, and I'll remove this entirely once we've tested. Definitely don't want this knob long-term" — Configuration flags added for rollout testing that should be cleaned up after validation.

- **Code obsoleted by refactors**: "we can delete this, all workers send logs directly now" — Worker provider code still handling legacy log forwarding after the system migrated.

- **Stale test gating**: "test is no longer gated by what it actually needs... `@requires_model` no longer implies that `hf_hub_download()` will work" — Test assumptions invalidated by upstream changes; test still runs but validates the wrong thing.

- **Outdated docstrings**: "Please refresh this docstring. The implementation no longer returns `False` merely because the detector labels a group HOSTILE" — Documentation describing behavior that has changed, creating silent misreads.

- **Vestigial wire / flag-gated parallel implementation**: A branch ships both a legacy code path and its replacement behind a feature flag that defaults off. Helpers, dataclasses, tests, and proto fields exist solely to drive the path that will be deleted. Reviewer language: "two wires shipping together," "this is dead the moment the flag flips," "self-admitted TODO to collapse this."
  ```python
  if os.environ.get("IRIS_RECONCILE_RPC_ENABLED"):
      reconcile_workers_via_reconcile(...)   # new path
  else:
      _reconcile_one(...)                    # legacy path, will be deleted
  # plus: legacy_translator_request, WorkerReconcileDispatch, parallel test doubles, ...
  # TODO(Reconcile-RPC-default): collapse once StartTasks/PollTasks retire
  ```
  Why: "Ship both, flip the flag later" sounds safe but doubles the surface area reviewers must understand and tends to leave dead branches around forever. Either the new path is safe enough to merge alone, or the old path should not have been kept. The presence of a self-admitted "collapse once X retires" TODO is the signal.

- **Within-branch add-then-remove churn (migrations, flags, fields)**: Inside a single branch, one commit adds a schema column / flag / field, another commit drops it. The branch never deployed in the intermediate state, so the additive change is pure churn — the right answer is to delete the additive change, not to stack a removal on top.
  ```text
  migrations/0047_worker_supports_reconcile_rpc.py   # adds column
  migrations/0048_drop_worker_supports_reconcile_rpc.py   # drops the column 0047 just added
  ```
  Why: A migration pair that adds and immediately drops the same schema in the same branch will never run in production; it just adds two migration steps that future readers have to mentally cancel. Same pattern applies to feature flags introduced and removed in the same branch.

- **Speculative abstraction with one user**: A `Union` / `Protocol` / generic helper introduced "in case we add more variants later," but the branch has exactly one variant. Reviewer language: "widen to a Union when a second case lands — but there is no second case," "this Protocol exists only to satisfy one import."
  ```python
  TransitionDelta = AttemptMissingOnWorker  # single concrete variant today; widen later
  class WorkerReconcileResultLike(Protocol): ...  # one implementation, ever
  db_writes: list[...] = field(default_factory=list)  # always []
  ```
  Why: Abstractions cost reader attention up front and pay back only when the second case lands. "We'll widen it later" is cheaper to do at the moment of the second case (where the shape is concrete) than now (where it is speculative).

## False-positive guidance

- **Conditional compatibility shims**: Legacy code paths protected by version checks or feature flags are acceptable if the condition is documented and tied to a known removal date (e.g., `# CRON(2026-06-01) -- remove after all workers updated`).
- **Dead branches in unreachable test code**: Code under `if False:` or inside `@skip` decorators is intentionally inert; mark for cleanup only if the guard is missing.
- **Deferred refactoring in comments**: Comments like `# TODO: consolidate with X` are technical debt, not dead code; flag only if the code is actively harmful or prevents other work.
- **Configuration with no callers in diff**: A new flag with no current usage might be scaffolding for an incoming PR; check git blame and related PRs before flagging.
- **Experimental code in `/experiments/` or gated by feature flags**: Explicitly tentative code should not trigger; prefer architecture reviews at submission time.

## Suggested confidence floor

Flag with high confidence when code is explicitly marked for deletion, no longer called after a known refactor, or actively broken (test gating mismatched to implementation). Lower confidence on vague "redundant" comments without concrete evidence.
