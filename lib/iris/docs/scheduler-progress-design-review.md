# Design Review: Scheduler Progress Guarantee

**Document:** `lib/iris/docs/scheduler-progress-design.md`
**Issue:** [#2742](https://github.com/marin-community/marin/issues/2742)
**Reviewer:** Senior Engineer
**Verdict:** Revision needed (one blocking issue, several non-blocking)

---

## Blocking Issues

### B1: `bisect.insort` with `key=lambda` recomputes keys on existing elements

The design proposes:

```python
bisect.insort(self._task_queue, task_id, key=lambda tid: self._task_priority_key(tid))
```

There are two problems here.

**Problem 1: The `key` parameter is redundant with `_task_priority_key`.**
The `_enqueue_task` method already computes `key = self._task_priority_key(task_id)` on line 234 of the design, then passes a lambda that calls the same function again. The pre-computed `key` variable is unused. This is a copy-paste error -- the code should either use the pre-computed key or drop the local variable. As written, it computes the key for the new element twice (once explicitly, once via the lambda).

**Problem 2: `bisect.insort(a, x, key=fn)` calls `fn` on existing elements during binary search.**
I verified this empirically with CPython 3.12: on a 100-element list, `bisect.insort` calls the key function 8 times total -- once on the new element and 7 times on existing elements. It does NOT cache keys for stored elements. Each call to `_task_priority_key` performs two dict lookups (`self._tasks.get()` and `self._jobs.get()`).

This is not a correctness bug -- the result is correct. But it means the queue is **not maintaining a sorted invariant based on cached keys**. If a task's priority inputs change (e.g., a job's `root_submitted_at` is modified after insertion), the queue becomes silently unsorted. More practically, it means inserting into a queue of N elements costs O(log N) key computations plus O(N) element shifts.

**Recommended fix:** Store `(priority_key, task_id)` tuples in the list and use plain `bisect.insort` without `key=`. This is simpler, faster (no repeated key computation), and makes the sorted invariant explicit:

```python
def _enqueue_task(self, task_id: JobName) -> None:
    key = self._task_priority_key(task_id)
    bisect.insort(self._task_queue, (key, task_id))
```

Then `peek_pending_tasks` iterates `(key, task_id)` pairs and extracts the task_id. This eliminates all re-computation of keys during binary search.

However, this requires `JobName` to be orderable (for tiebreaking when keys are equal). Since `JobName` is `frozen=True` with `slots=True`, it does not have `__lt__` by default. Either:
- Add `order=True` to the `JobName` dataclass (order by `_parts` tuple), or
- Wrap in a tuple `(key, insertion_counter, task_id)` where `insertion_counter` is a monotonically increasing int to guarantee uniqueness and avoid comparing `JobName` objects.

The second approach is cleaner because it does not impose a total order on `JobName` that might not be semantically meaningful.

---

## Non-Blocking Issues

### NB1: Priority model does not distinguish between parent tasks and child jobs at the same depth

Consider the scenario from the review prompt: a root job `/train` has tasks `/train/0`, `/train/1`, etc. It also spawns child jobs `/train/eval-1` and `/train/eval-2`. The design gives:

- `/train/eval-1/0` -- depth 2 (task of a depth-2 job)
- `/train/eval-2/0` -- depth 2
- `/train/0` -- depth 1 (task of a depth-1 job)
- `/train/1` -- depth 1

This ordering is correct for the stated goal: children complete before the parent's own tasks run, which is exactly what you want when the parent blocks on child completion.

But what if the root job `/train` has tasks that need to run *concurrently* with children? For example, if `/train/0` is the "coordinator" task that submits children and then waits for them. In that case, the coordinator task is already running (state RUNNING, not in the pending queue), so it is unaffected by queue ordering. This is fine.

**What about the scenario where `/train`'s own tasks need to run AFTER children complete?** The design handles this correctly: children at depth 2 are prioritized over `/train`'s tasks at depth 1. When the children finish, the parent's pending tasks will naturally be next in line.

No change needed, but the design should explicitly document this interaction pattern for clarity.

### NB2: `_mark_remaining_tasks_killed` rebuilds the queue as a `deque`

The current code at line 1754-1755 of `state.py`:

```python
self._task_queue = deque(tid for tid in self._task_queue if tid not in tasks_to_remove_from_queue)
```

After the change, this must rebuild as a `list`, not a `deque`. The design document mentions this in the "Where Changes Go" table but does not show the code. Since this is a filtering operation (removing elements), the sorted order is preserved and no re-sorting is needed. Just change `deque(...)` to `list(...)`. Simple, but easy to forget.

### NB3: `remove_finished_job` uses `deque.remove` which becomes `list.remove`

At line 1579-1581 of `state.py`:

```python
try:
    self._task_queue.remove(task_id)
except ValueError:
    pass
```

Both `deque.remove` and `list.remove` are O(N), so the change from deque to list does not affect complexity. The behavior is identical. No issue here, but the design should note that this is a no-op change.

### NB4: `_requeue_task` has an `in` check that becomes O(N) on a list

At line 1459 of `state.py`:

```python
if task_id not in self._task_queue:
    self._task_queue.append(task_id)
```

With a `deque`, the `in` check is O(N). With a sorted `list`, it is also O(N). So no regression. But note that `_requeue_task` currently appends to the end, which must be changed to use `_enqueue_task` (priority-ordered insertion) as the design states. The `in` check could be replaced with a `set` membership test if performance matters, but at the expected queue sizes it does not.

### NB5: Orphan child fallback may produce surprising priority ordering

The design states that if a child's parent is not tracked, the child uses its own `submitted_at` as `root_submitted_at`. This means an orphan child submitted at t=5000 will sort *after* a root job submitted at t=1000 at the same depth, which is correct. But if the orphan's parent was part of an older tree (submitted at t=500) that was already cleaned up, the orphan effectively loses its priority inheritance.

In practice, `remove_finished_job` is the only path that removes a parent while children might still exist. The existing `_cancel_child_jobs` cascade means children are killed when the parent terminates non-successfully. For successful termination, children should already be complete. So this edge case should not arise in normal operation.

No change needed, but a comment in the code noting why the orphan fallback is safe would be helpful.

### NB6: Coscheduled jobs at different depths -- interaction with priority ordering

The scheduler processes coscheduled jobs first (before first-fit). The pending task list is ordered by priority, and `find_assignments` iterates `tasks_by_job` which is built from the ordered pending list. However, `tasks_by_job` is a `defaultdict(list)` populated by iterating `pending_tasks` in order -- the *jobs* are encountered in priority order, but the `dict` iteration order reflects insertion order, which is correct.

A coscheduled job at depth 3 will have its tasks appear earlier in `pending_tasks` than a coscheduled job at depth 1. Since `tasks_by_job` preserves insertion order and the coscheduled loop iterates it, the depth-3 coscheduled job gets first shot at workers. This is correct behavior.

No change needed.

### NB7: Race conditions with task state changes

The `_task_queue` is accessed under `self._lock` (an `RLock`). All mutations to the queue (`_on_job_submitted`, `_requeue_task`, `_mark_remaining_tasks_killed`, `remove_finished_job`, `add_job`) and reads (`peek_pending_tasks`) acquire the lock. The `bisect.insort` operation is atomic within the lock scope.

The only potential issue: `peek_pending_tasks` returns a list of `ControllerTask` objects (mutable references). The scheduler's `find_assignments` then reads task/job state *without* holding the lock. This is an existing design choice (the scheduler is a pure function that operates on a snapshot of pending tasks), not introduced by this change. The priority ordering change does not make this worse.

No change needed.

### NB8: The `add_job` test helper must also use priority-ordered insertion

Line 1537 of `state.py`:

```python
self._task_queue.append(task.task_id)
```

The design mentions this in the "4 places where tasks are added" enumeration but does not show the code change. It needs to use `_enqueue_task` like the other insertion sites.

---

## Design Quality Assessment

### AGENTS.md compliance

- **Shallow, functional code:** The design correctly keeps the change localized to task ordering in `ControllerState`. The scheduler remains a pure function. This follows the Iris AGENTS.md preference for shallow, functional code.

- **Avoids over-engineering:** The design correctly defers per-user quotas to a future spiral. The priority model is the simplest thing that solves the stated problem.

- **Tests test stable behavior:** The proposed tests verify observable scheduling outcomes (which tasks get assigned in what order), not implementation details (queue data structure, key values). This is correct per AGENTS.md.

- **Spiral plan:** The design uses spiral implementation, which matches the Iris AGENTS.md guidance.

### Test design

The proposed tests are well-designed:

- `test_deeper_jobs_scheduled_before_shallow` -- tests the core behavior
- `test_older_root_tree_preferred_at_same_depth` -- tests the tiebreaker
- `test_child_of_older_tree_beats_root_of_newer_tree` -- tests that depth trumps time
- `test_fifo_within_same_depth_and_tree` -- tests stability

The "What NOT to test" section correctly excludes implementation details.

**Suggested additions:**

- A test for requeued tasks maintaining priority position (listed in the test plan but no code shown)
- A test that submitting a child job re-sorts the queue correctly (the child's tasks should appear before the parent's tasks even though the parent was submitted first)
- A test for the `_mark_remaining_tasks_killed` path verifying the queue is correctly filtered (not re-sorted) when tasks are killed

### Missing from the design

1. **No discussion of the `add_job` test helper.** It is listed in the enumeration but has no code snippet. This helper is used extensively in tests and must use `_enqueue_task`.

2. **The `_enqueue_task` code snippet has a bug** (the unused `key` local variable and redundant lambda). See B1.

3. **No discussion of `JobName.__lt__`** or tiebreaking for `bisect.insort`. If two tasks have identical priority keys, `bisect.insort` will try to compare the `JobName` objects themselves, which will raise `TypeError` since `JobName` does not implement `__lt__`. This is a latent correctness bug.

---

## Summary

| Category | ID | Summary | Action |
|----------|----|---------|--------|
| Blocking | B1 | `bisect.insort` with `key=` recomputes keys on existing elements and `JobName` lacks `__lt__` for tiebreaking | Store `(key_tuple, counter, task_id)` in list; use plain `bisect.insort` |
| Non-blocking | NB1 | Parent tasks vs child jobs at same depth | Document the interaction pattern |
| Non-blocking | NB2 | `_mark_remaining_tasks_killed` rebuilds as `deque` | Change to `list(...)` |
| Non-blocking | NB3 | `remove_finished_job` uses `.remove()` | No change needed (same complexity) |
| Non-blocking | NB4 | `_requeue_task` `in` check is O(N) | Acceptable at expected scale |
| Non-blocking | NB5 | Orphan child loses priority inheritance | Add explanatory comment |
| Non-blocking | NB6 | Coscheduled jobs at different depths | No change needed (correct) |
| Non-blocking | NB7 | Race conditions | No change needed (pre-existing design) |
| Non-blocking | NB8 | `add_job` test helper needs update | Use `_enqueue_task` |

The design is sound in its core insight (depth-first ordering via task queue priority) and correctly identifies that the scheduler itself needs no changes. The blocking issue is an implementation detail in the `_enqueue_task` method that will cause `TypeError` at runtime when two tasks have identical priority keys.

Fix B1, address the non-blocking items during implementation, and this is ready to build.
