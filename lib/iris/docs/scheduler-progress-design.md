# Scheduler Progress Guarantee Design

**Issue:** [#2742 — Iris: scheduler should guarantee progress](https://github.com/marin-community/marin/issues/2742)

## Problem Statement

Iris inherits dynamic job creation from Ray: a running job can submit child
jobs, which can submit grandchildren, and so on. The current scheduler uses a
simple FIFO queue (`deque`) for task ordering. This means a parent job's child
tasks compete with unrelated top-level submissions on equal footing. If the
cluster is busy, a job tree that needs its children to complete before the
parent can finish may be starved by a burst of independent top-level jobs.

The scheduler must guarantee **progress**: once a job tree starts executing, its
descendant jobs should be prioritized over unrelated work so the tree can run
to completion and release its resources.

## Current State Analysis

### Job Hierarchy (already exists)

`JobName` encodes hierarchy via path components:

```
/root-job                  depth=1 (root)
/root-job/child-1          depth=2
/root-job/child-1/worker   depth=3
/root-job/0                task of root-job
```

Key properties already available on `JobName`:

- `parent -> JobName | None` — the parent job
- `is_root -> bool` — whether this is a top-level job
- `_parts: tuple[str, ...]` — the path components
- `namespace -> str` — the root component (first part)
- `is_ancestor_of(other)` — ancestry check

**Depth** is derivable as `len(self._parts)` for a job, but there is no
explicit `depth` property yet.

### Task Queue (the thing we need to change)

In `ControllerState`, the task queue is a plain FIFO deque:

```python
self._task_queue: deque[JobName] = deque()  # FIFO queue of task IDs
```

Tasks are appended to the back in two places:
1. `_on_job_submitted()` — new job's tasks go to end of queue
2. `_requeue_task()` — retried tasks go to end of queue

`peek_pending_tasks()` iterates the queue in order and returns schedulable
tasks. The scheduler's `find_assignments()` processes this list top-to-bottom:
coscheduled jobs first, then first-fit for the rest.

### Scheduler (pure function)

`Scheduler.find_assignments(pending_tasks, workers) -> SchedulingResult`

The scheduler is a pure function that takes a list of pending tasks and
available workers. It does not own the ordering — it receives tasks in the
order produced by `peek_pending_tasks()`. The scheduling algorithm is:

1. Group tasks by job
2. Handle coscheduled jobs first (all-or-nothing)
3. Handle remaining tasks with first-fit, skipping tasks that don't fit
   (no head-of-line blocking)

### Parent-Child Lifecycle (already exists)

- Child jobs reference their parent via `JobName` hierarchy:
  `job_id = parent_job_id.child(name)` in `IrisClient.submit()`
- `ControllerState.get_children(job_id)` returns direct children
- `_cancel_child_jobs()` recursively cancels descendants when a parent
  terminates non-successfully
- `_terminate_job_tree()` in the service layer does depth-first termination
- The service rejects child job submissions if the parent has terminated

### What Does NOT Exist Yet

1. No priority ordering in the task queue
2. No concept of job "depth" as a scheduling signal
3. No per-user resource quotas
4. No mechanism to deprioritize jobs that exceed quota

## Design

### Core Insight

The depth-first scheduling algorithm can be implemented entirely by changing
**task ordering** in the pending queue. The scheduler itself (`find_assignments`)
does not need to change — it already processes tasks in the order given and
handles head-of-line blocking gracefully (skips tasks that don't fit).

### Priority Model

Tasks are ordered by a composite key:

```
(job_depth DESC, root_job_submitted_at ASC, task_submitted_at ASC)
```

- **job_depth DESC** (higher depth = scheduled first): Child jobs at depth 3
  are scheduled before depth 2, which are scheduled before depth 1. This
  ensures leaf work completes first, allowing parent jobs to make progress.
- **root_job_submitted_at ASC** (older trees first): Among jobs at the same
  depth, prefer tasks belonging to job trees that were submitted earlier.
  This prevents starvation of earlier submissions.
- **task_submitted_at ASC** (FIFO within a group): Tiebreaker within the
  same depth and tree.

### Example

```
Time 0: User A submits /train (depth 1)
Time 1: /train starts, submits /train/eval-1 (depth 2) and /train/eval-2 (depth 2)
Time 2: User B submits /inference (depth 1)
Time 3: /train/eval-1 submits /train/eval-1/score (depth 3)
```

Queue ordering:
```
1. /train/eval-1/score/0    (depth=3, root_submitted=0)
2. /train/eval-1/0          (depth=2, root_submitted=0)
3. /train/eval-2/0          (depth=2, root_submitted=0)
4. /train/0                 (depth=1, root_submitted=0)
5. /inference/0             (depth=1, root_submitted=2)
```

This ensures `/train`'s descendant work completes before `/inference` can
starve it, while `/inference` still gets resources if any are available (the
scheduler skips tasks that don't fit and tries the next one).

### Data Model Changes

#### 1. `JobName.depth` property

```python
@property
def depth(self) -> int:
    """Depth in the job hierarchy. Root jobs have depth 1.
    Tasks inherit their parent job's depth (the task index component
    is not counted as a depth level).
    """
    if self.is_task:
        return len(self._parts) - 1
    return len(self._parts)
```

#### 2. `ControllerJob.root_submitted_at` field

Track when the root ancestor of this job was submitted:

```python
@dataclass
class ControllerJob:
    # ... existing fields ...
    root_submitted_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
```

When a job is submitted in `_on_job_submitted()`:

```python
# Resolve root submission time for priority ordering
if job_id.is_root:
    job.root_submitted_at = event.timestamp
else:
    parent_job = self._jobs.get(job_id.parent)
    if parent_job:
        job.root_submitted_at = parent_job.root_submitted_at
    else:
        # Orphan child (parent not tracked) — use own submission time
        job.root_submitted_at = event.timestamp
```

#### 3. Priority-ordered task queue

Replace the plain `deque` with a sorted structure. Since we need efficient
insertion and iteration-in-order, and the queue is typically small (hundreds
to low thousands of tasks), a sorted list with `bisect.insort` is sufficient.

The sort key for a task is:

```python
def _task_priority_key(self, task_id: JobName) -> tuple[int, int, int]:
    """Priority key for task ordering.

    Returns (negative_depth, root_submitted_ms, submitted_ms) where
    lower values = higher priority (sorted ascending).

    Depth is negated so deeper jobs sort first.
    """
    task = self._tasks.get(task_id)
    job = self._jobs.get(task.job_id) if task else None
    if not task or not job:
        return (0, 0, 0)

    depth = task.job_id.depth
    return (
        -depth,
        job.root_submitted_at.epoch_ms(),
        task.submitted_at.epoch_ms(),
    )
```

### Where Changes Go

| File | Change |
|------|--------|
| `cluster/types.py` | Add `JobName.depth` property |
| `cluster/controller/state.py` | Add `root_submitted_at` to `ControllerJob`; replace `_task_queue` deque with sorted list; add `_task_priority_key()`; update `_on_job_submitted()` to resolve root timestamp; update `_requeue_task()` to insert at correct priority position |
| `cluster/controller/state.py` | Update `peek_pending_tasks()` — no change needed, it already iterates in order |
| `cluster/controller/scheduler.py` | No changes needed — receives tasks in priority order |
| `cluster/controller/controller.py` | No changes needed |

### Implementation of the Priority Queue

The simplest correct approach: keep `_task_queue` as a `list[JobName]` and
use `bisect.insort` with a key function. Python 3.10+ `bisect.insort` supports
a `key` parameter.

```python
import bisect

class ControllerState:
    def __init__(self):
        # ...
        self._task_queue: list[JobName] = []  # sorted by priority

    def _enqueue_task(self, task_id: JobName) -> None:
        """Insert task into priority-sorted queue."""
        key = self._task_priority_key(task_id)
        # bisect.insort uses the key to find the right position
        bisect.insort(self._task_queue, task_id, key=lambda tid: self._task_priority_key(tid))

    def _task_priority_key(self, task_id: JobName) -> tuple[int, int, int]:
        task = self._tasks.get(task_id)
        job = self._jobs.get(task.job_id) if task else None
        if not task or not job:
            return (0, 0, 0)
        depth = task.job_id.depth
        return (-depth, job.root_submitted_at.epoch_ms(), task.submitted_at.epoch_ms())
```

Complexity: O(N) per insertion due to list shift, O(N) for iteration. Since
the queue is bounded by active tasks (typically < 10,000), this is fine.
If it becomes a bottleneck, we can switch to a `SortedList` from the
`sortedcontainers` library.

### Phase 2: Per-User Resource Quotas

The issue identifies a risk: a user who submits many cheap root-level jobs
could dominate the cluster before deeper jobs from other users get a chance.

**Design sketch** (not implemented in phase 1):

1. Add a `user` field to `LaunchJobRequest` (or derive from authentication)
2. Add a `ResourceQuota` configuration per user:
   ```python
   @dataclass
   class ResourceQuota:
       max_running_cpu: int = 0       # 0 = unlimited
       max_running_memory: int = 0
       max_pending_tasks: int = 0
   ```
3. In `peek_pending_tasks()`, after sorting by priority, apply quota filtering:
   tasks from users who have exceeded their running resource quota are moved
   to the end of the list.

This is deliberately deferred to a later spiral. The depth-first algorithm
alone solves the primary progress guarantee problem. Quotas address the
secondary fairness concern.

## Spiral Implementation Plan

### Spiral 1: `JobName.depth` property + test

**Goal:** Add the `depth` property to `JobName` with tests.

Changes:
- `lib/iris/src/iris/cluster/types.py`: Add `depth` property
- `lib/iris/tests/cluster/test_types.py`: Add tests for depth at various levels

```python
# types.py
@property
def depth(self) -> int:
    """Depth in the job hierarchy.

    Root jobs have depth 1. Tasks inherit their parent job's depth.
    /root -> 1
    /root/child -> 2
    /root/child/0 (task) -> 2
    """
    if self.is_task:
        return len(self._parts) - 1
    return len(self._parts)
```

Test:
```python
def test_job_name_depth():
    assert JobName.root("train").depth == 1
    assert JobName.from_string("/train/eval").depth == 2
    assert JobName.from_string("/train/eval/score").depth == 3
    # Task depth equals parent job depth
    assert JobName.from_string("/train/0").depth == 1
    assert JobName.from_string("/train/eval/0").depth == 2
```

### Spiral 2: `root_submitted_at` on `ControllerJob` + wiring

**Goal:** Track root submission timestamp through the job hierarchy.

Changes:
- `lib/iris/src/iris/cluster/controller/state.py`:
  - Add `root_submitted_at: Timestamp` field to `ControllerJob`
  - Update `_on_job_submitted()` to resolve root timestamp from parent chain
  - Update `add_job()` test helper to propagate root timestamp

Test:
```python
def test_child_job_inherits_root_submitted_at(state):
    """Child jobs inherit root_submitted_at from their parent."""
    parent_ts = Timestamp.from_ms(1000)
    submit_job(state, "/parent", make_request(), timestamp_ms=1000)
    submit_job(state, "/parent/child", make_request(), timestamp_ms=2000)

    parent = state.get_job(JobName.from_string("/parent"))
    child = state.get_job(JobName.from_string("/parent/child"))
    assert parent.root_submitted_at.epoch_ms() == 1000
    assert child.root_submitted_at.epoch_ms() == 1000  # inherits from parent
```

### Spiral 3: Priority-ordered task queue

**Goal:** Replace FIFO deque with priority-sorted list. This is the core change.

Changes:
- `lib/iris/src/iris/cluster/controller/state.py`:
  - Change `_task_queue: deque[JobName]` to `_task_queue: list[JobName]`
  - Add `_task_priority_key()` method
  - Add `_enqueue_task()` method using `bisect.insort`
  - Replace all `_task_queue.append(task_id)` calls with `_enqueue_task(task_id)`
  - Update `_mark_remaining_tasks_killed` queue rebuild to use sorted list
  - Update `remove_finished_job` to use `list.remove` instead of `deque.remove`

There are exactly 4 places where tasks are added to the queue:
1. `_on_job_submitted()` — for each new task
2. `_requeue_task()` — for retried tasks
3. `add_job()` — test helper
4. (none others — verified by grep)

Test:
```python
def test_deeper_jobs_scheduled_before_shallow(scheduler, state, job_request, worker_metadata):
    """Tasks from deeper job hierarchy levels are scheduled first."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Submit root job first
    submit_job(state, "/root", job_request(cpu=1))
    # Then submit a child job
    submit_job(state, "/root/child", job_request(cpu=1))

    pending = state.peek_pending_tasks()
    # Child task (depth 2) should come before root task (depth 1)
    assert pending[0].job_id == JobName.from_string("/root/child")
    assert pending[1].job_id == JobName.from_string("/root")

    result = scheduler.find_assignments(pending, state.get_available_workers())
    # With 2 CPU available and 1 CPU per job, both get scheduled
    # but child should be first in assignments
    assert len(result.assignments) == 2
    assert result.assignments[0][0].job_id == JobName.from_string("/root/child")


def test_older_root_tree_preferred_at_same_depth(scheduler, state, job_request, worker_metadata):
    """Among jobs at the same depth, older root trees are scheduled first."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=1))

    # User A's tree submitted at time 1000
    submit_job(state, "/user-a-job", job_request(cpu=1), timestamp_ms=1000)
    # User B's tree submitted at time 2000
    submit_job(state, "/user-b-job", job_request(cpu=1), timestamp_ms=2000)

    pending = state.peek_pending_tasks()
    assert pending[0].job_id == JobName.root("user-a-job")
    assert pending[1].job_id == JobName.root("user-b-job")

    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 1
    assert result.assignments[0][0].job_id == JobName.root("user-a-job")


def test_child_of_older_tree_beats_root_of_newer_tree(scheduler, state, job_request, worker_metadata):
    """A child job from an earlier tree is scheduled before a root job from a later tree."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=1))

    # Old tree: root submitted at t=1000
    submit_job(state, "/old-tree", job_request(cpu=1), timestamp_ms=1000)
    # New tree: root submitted at t=2000
    submit_job(state, "/new-tree", job_request(cpu=1), timestamp_ms=2000)
    # Old tree spawns a child at t=3000
    submit_job(state, "/old-tree/child", job_request(cpu=1), timestamp_ms=3000)

    pending = state.peek_pending_tasks()
    # Child (depth=2) should come first, then old-tree root, then new-tree root
    assert pending[0].job_id == JobName.from_string("/old-tree/child")

    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert result.assignments[0][0].job_id == JobName.from_string("/old-tree/child")


def test_fifo_within_same_depth_and_tree(scheduler, state, job_request, worker_metadata):
    """Tasks at the same depth within the same tree use FIFO ordering."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=1))

    submit_job(state, "/tree/child-a", job_request(cpu=1), timestamp_ms=1000)
    submit_job(state, "/tree/child-b", job_request(cpu=1), timestamp_ms=2000)

    pending = state.peek_pending_tasks()
    assert pending[0].job_id == JobName.from_string("/tree/child-a")
    assert pending[1].job_id == JobName.from_string("/tree/child-b")
```

### Spiral 4: Integration test with Controller

**Goal:** End-to-end test showing progress guarantee with the full controller loop.

Test using `LocalController` or direct `Controller` instance:
```python
def test_controller_progress_guarantee(controller_fixture):
    """A parent job's children are scheduled before unrelated jobs."""
    # Submit parent job that will spawn children
    # Submit unrelated job
    # Verify children are scheduled before unrelated job
    # Verify parent completes
```

### Spiral 5 (future): Per-user resource quotas

Deferred. See Phase 2 section above.

## Test Plan

### Unit Tests (Spiral 1-3)

| Test | What it validates |
|------|-------------------|
| `test_job_name_depth` | `JobName.depth` returns correct values for root, child, grandchild, and task names |
| `test_child_job_inherits_root_submitted_at` | `root_submitted_at` propagated through parent chain |
| `test_orphan_child_uses_own_submitted_at` | Child without tracked parent uses its own timestamp |
| `test_deeper_jobs_scheduled_before_shallow` | Depth-first ordering: children before parents |
| `test_older_root_tree_preferred_at_same_depth` | FIFO among same-depth jobs from different trees |
| `test_child_of_older_tree_beats_root_of_newer_tree` | Depth trumps submission time |
| `test_fifo_within_same_depth_and_tree` | Tiebreaker within same depth and tree |
| `test_requeued_task_maintains_priority` | Retried task re-inserted at correct priority position |
| `test_coscheduled_jobs_respect_priority` | Coscheduled jobs at higher depth processed first |
| `test_priority_queue_with_many_tasks` | Performance: 10,000 tasks insert and iterate correctly |

### Behavioral Tests (Spiral 4)

| Test | What it validates |
|------|-------------------|
| `test_controller_progress_guarantee` | Full scheduling loop respects depth-first ordering |
| `test_unrelated_jobs_not_starved` | Jobs from different trees still get resources when available |

### What NOT to test

- Internal queue data structure details (sorted list vs deque)
- Exact values of priority keys
- Number of internal comparisons

## Risks and Mitigations

### Risk: Starvation of root-level jobs

If a user creates a deeply nested job tree, their leaf tasks will always be
prioritized over new root-level submissions from other users.

**Mitigation:** Per-user resource quotas (Spiral 5). Once a user exceeds their
quota, their tasks are deprioritized regardless of depth.

### Risk: Orphan children with unknown root timestamps

If a child job is submitted but its parent is no longer tracked (e.g., parent
was cleaned up), we cannot determine the root submission time.

**Mitigation:** Use the child's own submission time. This is correct: an
orphan child acts like a new root.

### Risk: Performance with large queues

`bisect.insort` on a list is O(N) per insertion. With 10,000 pending tasks
and 100 insertions per second, this is ~1M operations/second — well within
budget.

**Mitigation:** If profiling shows this is a bottleneck, switch to
`sortedcontainers.SortedList` which provides O(log N) insertion.

### Risk: Task ordering changes break existing tests

The FIFO ordering assumption is implicit in many scheduler tests.

**Mitigation:** Most tests submit jobs sequentially and verify assignments
based on resource fit, not queue position. Tests that depend on FIFO ordering
of same-depth, same-tree tasks will continue to work because the priority key
preserves FIFO within those groups.
