# RL Preemption Support for Manual (No-Ray) Mode

**Status**: done (fix landed, test added)
**Priority**: needed before switching to spot TPUs

## Problem

The manual-mode RL system cannot resume after preemption. The filesystem-based Arrow Flight coordinator persists stale state on GCS that blocks the restarted trainer from publishing weights, causing a permanent deadlock.

## Background: why the filesystem coordinator exists

The `FileSystemArrowFlightCoordinator` was created specifically for no-Ray manual mode (`ON_DEMAND_RL.md`, Change 2). In Ray mode, the coordinator is an in-memory actor — trainer and sampler on the same cluster can access it directly. In manual mode, trainer and sampler are on separate TPU VMs with no shared memory. GCS is the only shared surface, so the coordinator writes a JSON file (`arrow_flight_coordinator.json`) that acts as a service discovery mechanism: "here's where the weights are right now."

The JSON file is NOT a checkpoint. It stores the current weight_id, Flight server addresses, param names, and status. The trainer writes it after every weight publish, the sampler reads it to discover where to fetch weights.

## The deadlock mechanism

1. Trainer runs to step 50, coordinator JSON on GCS has `{weight_id: 50}`
2. Trainer gets preempted (SIGKILL — `mark_failed()` never runs)
3. Coordinator JSON unchanged: `{weight_id: 50, status: "running"}`
4. New trainer starts, loads checkpoint from step 45
5. Trainer calls `serve_weights(-1, initial_model)` — the initial weight publish always uses -1
6. `FileSystemArrowFlightCoordinator.update_server()` checks: `-1 < 50` → **rejects as stale**
7. Sampler never gets new weights, trainer waits for rollouts → **permanent deadlock**

The stale-weight check comment says "use strict < so restarts can overwrite" — but that only handles restarting at the SAME step (50 is not < 50). The initial publish is weight_id=-1, which is always less than any previous step.

## Why Ray mode never had this problem

The Ray coordinator is an `ArrowFlightCoordinator` — a Python object in memory. When the job restarts:
- The old actor is gone (died with the cluster/job)
- New job creates a fresh actor: `_server_info = None`
- `update_server(-1, ...)` checks: `current_weight_id is not None`? No → skips stale check → accepts

The stale check never fires because `_server_info` starts as `None` on every fresh actor. There's no persistent state to conflict with. The filesystem coordinator is the only one where this matters because the GCS JSON file **outlives the process that wrote it**.

## The fix

One line in `FileSystemArrowFlightCoordinator.update_server()`:

```python
# Current (broken on restart):
if weight_id < existing["weight_id"]:

# Fixed:
if weight_id >= 0 and weight_id < existing["weight_id"]:
```

This makes `weight_id=-1` always bypass the stale check. It's the "initial weights" sentinel — always the first thing the trainer publishes on startup. There's no legitimate scenario where you'd want to reject it.

Normal weight updates (0, 1, 2, ...) still get the stale check. Only -1 is special.

### No impact on Ray mode

The fix is only in `FileSystemArrowFlightCoordinator.update_server()`. The Ray `ArrowFlightCoordinator.update_server()` is a separate method on a separate class. Even if someone added the same logic to the Ray actor, it's correct — weight_id=-1 should always be accepted.

### Step-by-step walkthrough with fix

**Normal run (no preemption):**
1. Trainer publishes `weight_id=-1`. Coordinator: `{weight_id: -1}`
2. Sampler picks up weights, generates rollouts
3. Step 0: publishes `weight_id=0`. Check: `0 >= 0 and 0 < -1`? No → accepts
4. Step 1: publishes `weight_id=1`. Check: `1 >= 0 and 1 < 0`? No → accepts
5. ...continues normally

**Preemption at step 50, restart from checkpoint at step 45:**
1. Coordinator on GCS: `{weight_id: 50}`
2. SIGKILL — trainer dies. Coordinator unchanged
3. New trainer loads checkpoint step 45
4. Publishes `weight_id=-1`. Check: `-1 >= 0`? **No** → skips stale check → **accepts**
5. Coordinator reset to `{weight_id: -1}` with new server addresses
6. Sampler picks up new weights, generates rollouts
7. Step 45: publishes `weight_id=45`. Check: `45 >= 0 and 45 < -1`? No → accepts
8. Training resumes from step 45

## Additional preemption issues (lower priority)

### SIGKILL prevents graceful shutdown
Preemption sends SIGKILL. `mark_failed()` never runs. Coordinator stays `status=running` with stale server addresses. The sampler keeps polling dead Flight servers. Mitigation: sampler should have a timeout-based fallback that resets connections after N consecutive failures.

### Replay buffer is in-memory
Lost on preemption. New trainer starts with empty buffer. This is fine — the trainer waits for fresh rollouts from the sampler, which gets weights via the (now-fixed) coordinator.

### Stale rollout files on GCS
Old rollout files remain. The replay buffer's `max_rollout_step_delay=1` rejects most as stale. Not a blocker.

## Files to change

| File | Change | Lines |
|------|--------|-------|
| `arrow_flight.py` | Add `weight_id >= 0` guard in `update_server()` | 1 |
| `test_weight_transfer.py` | Test: coordinator accepts -1 after higher weight_id | ~5 |

## Status

**Fix landed** (2026-03-20):
- `arrow_flight.py`: `weight_id >= 0` guard added to `update_server()` stale check
- `test_weight_transfer.py`: test added for restart-after-preemption scenario (coordinator accepts -1 after higher weight_id, stale check still works for normal updates)

Remaining items (not blocking spot runs):
- Sampler timeout fallback for dead Flight servers (medium priority)
- Stale rollout cleanup on restart (low priority)

### Validated in production (2026-03-21)

Run `exp2039-nb-inflight2` on spot TPUs was preempted at step 111. Relaunched with same run ID:
- Trainer found checkpoint at step 110 on GCS
- Coordinator JSON had `weight_id=111` from previous run
- Trainer published `weight_id=-1` → **accepted** (fix working, no deadlock)
- Old rollouts (steps 104-111) correctly rejected as stale
- Sampler received fresh weights, generated new rollouts
- Trainer resumed training from step 110, reached step 112+ and continuing
- Wandb run resumed successfully
