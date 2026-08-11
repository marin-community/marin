# Iris journey coverage

Iris journeys cover chronological behavior across the real controller, service,
SQLite persistence, scheduling, reconciliation, federation, and checkpoint code.
Only execution processes, provider transports, peer transport, and time are
faked. `JourneyWorld` checks Attempt history, terminal-state monotonicity, live
Attempt uniqueness, Job counts, and duplicate launches after every control tick.

The source catalog evaluated all 2,496 pytest families at
`1373230331f63a0a388a7b9944b48672ad844cdc`. The full catalog and rationale are
in the [Iris test cleanup artifact](https://loom.oa.dev/s/kiyyg1qv/artifacts/iris-test-cleanup-plan).

## Catalog closure

| Original disposition | Families | Resolution |
|---|---:|---|
| KEEP | 1,802 | Retained, except four private or tautological cases reclassified and removed. |
| RENAME | 18 | All names now describe the tested subject and outcome. |
| FIX | 408 | 402 were repaired; six implementation-only controller forms were deleted. |
| MIGRATE | 200 | 89 were replaced by journeys; 111 were retained after the boundary review below. |
| DELETE | 68 | Removed from Iris; three GCS lease cases moved to the owning Rigging package. |

The 89 removed migration families map to these collected journeys:

| Source group | Families | Journey replacement |
|---|---:|---|
| Controller replay | 3 | checkpoint, lifecycle, and automatic invariants |
| Direct controller lifecycle | 9 | lifecycle, scheduling, coscheduling, and worker lifecycle |
| Transition chronology | 12 | lifecycle, cancellation, hierarchy, endpoint, and worker lifecycle |
| Multi-backend routing | 3 | routing |
| Preemption | 4 | lifecycle and worker lifecycle |
| Scheduling fairness | 1 | mixed-capacity routing |
| Worker health | 1 | worker lifecycle |
| Public service chronology | 10 | cancellation, hierarchy, public views, and worker lifecycle |
| Reconcile chronology | 3 | worker lifecycle |
| Federation handoff | 2 | federation happy path and execution-created subtree sync |
| Administrative action | 5 | resource actions |
| Endpoint replacement | 1 | endpoints |
| Dry-run mode | 2 | modes |
| Cluster public reads | 6 | lifecycle, cancellation, and public views |
| Chaos scenarios | 16 | lifecycle, scheduling, coscheduling, worker lifecycle, and checkpoint |
| Smoke checkpoint and width | 2 | checkpoint and 128-Task lifecycle |
| Budget spend | 4 | budget |
| JAX endpoint coordination | 5 | public initializer tests plus endpoint restart/replacement |

The other 111 original MIGRATE calls remain intentionally focused:

- 94 controller families retain exact multi-slice resource release, ASSIGNED and
  heartbeat batch semantics, resubmit drain policy, worker threshold behavior,
  federation payload/auth/queue/tombstone/exec contracts, or endpoint relay
  behavior. A broader journey would hide their oracle or require a fake of the
  behavior under test.
- Three WorkerPool families are now two `requires_cluster` process adapters for
  submit/map results and user-exception propagation.
- Fourteen CLI, client, environment, process, provider, and loopback-auth
  families remain live adapters. They test the boundary that an in-process
  journey deliberately fakes.

No removed chronological behavior depends only on a snapshot or private table
assertion. Replay goldens and the random chaos harness are gone; the replacement
stories use addressed faults, bounded convergence, public reads, and recorded
external commands.
