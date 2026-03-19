# Trainer → Sampler Graceful Shutdown Signal

**Status**: implemented
**Branch**: `on-demand-rl`

## Problem

When the trainer finishes all 500 training steps, it exits cleanly but the sampler keeps running forever — stuck in a loop trying to fetch weights from dead Arrow Flight servers. The sampler's wandb run never closes properly because `RolloutTracker.finish()` is never called.

## Design

Use the existing GCS coordinator JSON as the source of truth for trainer lifecycle. Record an explicit terminal status (not a bare boolean):

- `running` — trainer is active, may publish more weights
- `completed` — trainer exited normally, no more weights coming
- `failed` — trainer crashed, no more weights expected

This lets the sampler distinguish clean completion from failure instead of treating both as "done".

### Why not put the signal in `stop()`?

`TrainWorker.stop()` runs from the unconditional `finally` block in `train()` (line 339-343), so it fires on crashes, startup failures, AND normal completion. Signaling "completed" there would lie to the sampler on crashes.

## Changes (~25 lines across 4 files)

### 1. Add `status` to `ServerInfo` dataclass

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` (~line 55)

```python
@dataclass
class ServerInfo:
    weight_id: int | None
    server_addresses: list[str]
    param_names: list[str]
    status: Literal["running", "completed", "failed"] = "running"
```

### 2. Add `is_done` / `is_failed` to `WeightUpdate`

**File**: `lib/marin/src/marin/rl/weight_transfer/base.py` (~line 50)

```python
@dataclass
class WeightUpdate:
    model: PyTree | None
    state_dict: dict
    weight_id: int
    is_done: bool = False
    is_failed: bool = False
```

### 3. Add `mark_completed()` / `mark_failed()` to abstract server interface

**File**: `lib/marin/src/marin/rl/weight_transfer/base.py` (~line 84)

Add to `WeightTransferServer`:
```python
@abstractmethod
def mark_completed(self) -> None:
    """Signal that training completed normally."""
    pass

@abstractmethod
def mark_failed(self) -> None:
    """Signal that training failed."""
    pass
```

### 4. Coordinator `mark_completed()` / `mark_failed()` methods

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

On `FileSystemArrowFlightCoordinator`:
```python
def _set_status(self, status: str) -> None:
    existing = self._read_metadata() or {}
    existing["status"] = status
    data = json.dumps(existing).encode()
    _gcs_write(self._metadata_path, data)
    logger.info("Set coordinator status to %s", status)

def mark_completed(self) -> None:
    self._set_status("completed")

def mark_failed(self) -> None:
    self._set_status("failed")
```

On `ArrowFlightCoordinator` (Ray actor):
```python
def mark_completed(self) -> None:
    if self._server_info is not None:
        self._server_info.status = "completed"
    logger.info("Marked training as completed in coordinator")

def mark_failed(self) -> None:
    if self._server_info is not None:
        self._server_info.status = "failed"
    logger.info("Marked training as failed in coordinator")
```

### 5. Read `status` in `fetch_server()`

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

Update `FileSystemArrowFlightCoordinator.fetch_server()`:
```python
return ServerInfo(
    weight_id=data["weight_id"],
    server_addresses=data["server_addresses"],
    param_names=data["param_names"],
    status=data.get("status", "running"),
)
```

### 6. `update_server()` always resets status to `running`

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

In `FileSystemArrowFlightCoordinator.update_server()`, include `"status": "running"` in the metadata dict. This avoids stale terminal state if a trainer restarts for the same run directory.

### 7. Implement `mark_completed()` / `mark_failed()` on `ArrowFlightServer`

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

```python
def mark_completed(self) -> None:
    if self._use_filesystem:
        self._coordinator.mark_completed()
    else:
        self._ctx.get(self._coordinator.mark_completed.remote())

def mark_failed(self) -> None:
    if self._use_filesystem:
        self._coordinator.mark_failed()
    else:
        self._ctx.get(self._coordinator.mark_failed.remote())
```

### 8. Signal completion/failure at the right points in `train()`

**File**: `lib/marin/src/marin/rl/train_worker.py`

Do NOT mark completion in `stop()`. Instead, in `train()`:
- After `trainer.train(...)` returns normally (after the `StopTrainerException` catch), call `self.transfer_server.mark_completed()`
- In the `except Exception` path (line 336-338) that logs `TRAIN WORKER CRASHED`, call `self.transfer_server.mark_failed()` before re-raising
- Leave `stop()` as cleanup-only

```python
def train(self):
    try:
        ...
        with (...) as trainer, self.replay_loader:
            ...
            try:
                trainer.train(state, self.data_loader)
            except StopTrainerException:
                pass
        self.transfer_server.mark_completed()
    except StopTrainerException:
        self.transfer_server.mark_completed()
    except Exception:
        logger.exception("TRAIN WORKER CRASHED")
        self.transfer_server.mark_failed()
        raise
    finally:
        try:
            self.stop()
        except Exception:
            logger.exception("Failed to stop train worker during cleanup")
```

### 9. Arrow Flight client returns sentinel on terminal status

**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

In `receive_weights()`, after fetching `server_info`:
```python
if server_info and server_info.status == "completed":
    logger.info("Training complete signal received from coordinator")
    return WeightUpdate(model=None, state_dict={}, weight_id=-1, is_done=True)
if server_info and server_info.status == "failed":
    logger.info("Training failure signal received from coordinator")
    return WeightUpdate(model=None, state_dict={}, weight_id=-1, is_failed=True)
```

### 10. Rollout worker handles terminal updates and closes wandb

**File**: `lib/marin/src/marin/rl/rollout_worker.py`

In `_sync_weights()`, after `_receive_once()`:
```python
update = _receive_once()
if update and update.is_done:
    logger.info("Training complete, stopping rollout worker")
    self._running = False
    return
if update and update.is_failed:
    logger.info("Trainer failed, stopping rollout worker")
    self._running = False
    return
```

In `finally` cleanup of `run()` (~line 835), add `tracker.finish()`:
```python
finally:
    self._running = False
    try:
        if hasattr(self.tracker, 'finish'):
            self.tracker.finish()
    except Exception:
        logger.exception("Failed to finish tracker")
    try:
        self._transfer_client.cleanup()
    ...
```

## Files modified

| File | Lines | What |
|------|-------|------|
| `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` | ~20 | `status` on `ServerInfo`, `mark_completed()`/`mark_failed()` on coordinators and server, `status` in `fetch_server()`/`update_server()`, terminal check in `receive_weights()` |
| `lib/marin/src/marin/rl/weight_transfer/base.py` | ~8 | `is_done`/`is_failed` on `WeightUpdate`, `mark_completed()`/`mark_failed()` on abstract `WeightTransferServer` |
| `lib/marin/src/marin/rl/train_worker.py` | ~4 | Call `mark_completed()` after training, `mark_failed()` on crash |
| `lib/marin/src/marin/rl/rollout_worker.py` | ~8 | Check `is_done`/`is_failed` in `_sync_weights()`, call `tracker.finish()` in cleanup |

## Expected behavior

### Normal completion
1. Trainer finishes all steps → `mark_completed()` → coordinator status becomes `completed`
2. Sampler sees terminal status on next sync → sets `_running=False` → exits main loop
3. `finally` cleanup runs including `tracker.finish()` → wandb run closes
4. Both wandb runs show "finished"

### Trainer crash
1. Trainer raises exception → `mark_failed()` → coordinator status becomes `failed`
2. Sampler sees failure status → stops retrying dead Flight servers → clean shutdown
3. Run is not misreported as success

### Trainer restart
1. Trainer calls `update_server()` with new weights → status resets to `running`
2. Stale terminal state from previous run is overwritten

## Verification

1. Unit test: coordinator round-trip preserves `status`; `update_server()` resets to `running`
2. Unit test: filesystem coordinator can write terminal state before/after weight publish without `fetch_server()` crashing
3. Manual test on `exp2039`: verify sampler exits after trainer completion, rollout wandb run shows "finished"
