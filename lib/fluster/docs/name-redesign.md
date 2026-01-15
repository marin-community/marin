# Job Name Redesign

## Core Principle

**Service dumb, client smart.**

## The `name` Field

There is ONE identifier: `name`.

- `name` is the **complete hierarchical unique identifier**
- `name` is what clients send
- `name` is what the service stores
- `name` is what's used for all lookups

## Client Responsibility

The **client** constructs the full hierarchical name before sending to the service:

```python
# Client code (simplified)
def submit(self, user_name: str, ...):
    # If we're inside a job, prepend our job's name
    if self._current_job_name:
        name = f"{self._current_job_name}/{user_name}"
    else:
        name = user_name  # Root job

    # Send to service
    request = LaunchJobRequest(name=name, ...)
    response = self._client.launch_job(request)
```

## Service Responsibility

The **service** is dumb:

1. Receive `name` from client
2. Check if `name` already exists → reject with `ALREADY_EXISTS`
3. Store job with `name` as identifier
4. Return `name` as `job_id` in response

```python
# Service code (simplified)
def launch_job(self, request, ctx):
    name = request.name

    # Reject duplicates
    if self._state.get_job(name):
        raise ConnectError(Code.ALREADY_EXISTS, f"Job {name} already exists")

    # Store as-is
    job = ControllerJob(job_id=JobId(name), request=request, ...)
    self._state.add_job(job)

    return LaunchJobResponse(job_id=name)
```

## Example Flow

```
User code:
    client.submit(name="worker-0")

Client (inside job "my-exp"):
    → prepends current job name
    → sends name="my-exp/worker-0" to service

Service:
    → receives name="my-exp/worker-0"
    → checks uniqueness
    → stores job with job_id="my-exp/worker-0"
    → returns job_id="my-exp/worker-0"
```

## Hierarchy Examples

```
Root job:     user submits "my-exp-123"         → name="my-exp-123"
Child job:    user submits "worker-0"           → name="my-exp-123/worker-0"
Grandchild:   user submits "task-a"             → name="my-exp-123/worker-0/task-a"
```

## What Changes

### Client (`client.py`)

- `RpcClusterClient.submit()`: Prepend `self._job_id + "/"` to user-provided name
- Track current job name in client context

### Service (`service.py`)

- Remove `uuid.uuid4()` generation
- Use `request.name` directly as `job_id`
- Add duplicate check before storing

### Proto

- `parent_job_id` field becomes informational only (or remove it)
- `name` is the authoritative identifier

## Validation

- Client validates user-provided name doesn't contain `/`
- Service validates `name` is not empty
- Service rejects duplicates
