# Agent Tips

* Use the connect/RPC abstractions to implement and perform RPC calls. DO NOT use httpx or raw HTTP.
* Use scripts/generate-protos.py to regenerate files after changing the `.proto` files.
* Prefer shallow, functional interfaces which return control to the user, vs callbacks or inheritance.

e.g.

class Scheduler:
  def add_job()
  def add_worker():
  def compute_schedule() -> ScheduledJobs:

is preferable to:

class Scheduler:
  def __init__(self, job_creator: JobCreator):
    self.job_creator = job_creator
  def run(self):
    ... self.job_creator.create_job()

It's acceptable to have a top-level class which implements the main loop of
course, but prefer to keep other interfaces shallow and functional whenever
possible.

* Tests should evaluate _behavior_, not implementation. Don't test things that are trivially caught by the type checker. Explicitly that means:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value - they make our code more brittle. Test
_behavior_ instead. You can use mocks as needed to isolate environments (e.g.
mock around a remote API). Prefer "fakes" -- e.g. create a real database but
with fake data -- when reasonable.

## Protocols and Testing

Non-trivial public classes should define a protocol which represents their
_important_ interface characteristics. Use this protocol in type hints for
when the class is used instead of the concrete class.

Test to this protocol, not the concrete class: the protocol should describe the
interesting behavior of the class, but not betray the implementation details.

(You may of course _instantiate_ the concrete class for testing.)

## Dependency Injection (DI-lite)

Iris uses lightweight constructor injection for testability and pluggability.
See `docs/di-cleanup.md` for current state and cleanup tasks.

### When to Inject

Apply DI when:
1. **External environment impact**: Network, filesystem, subprocess, hardware
2. **Simulating failures or time**: Things difficult to control in tests

Skip DI for purely computational code with no side effects.

### Pattern

```python
class FooProvider(Protocol):
    def do_thing(self) -> Result: ...

class OsFooProvider:  # Context-specific name, not "DefaultFooProvider"
    def do_thing(self) -> Result:
        # Real implementation
        ...

class Consumer:
    def __init__(self, foo: FooProvider | None = None):
        self._foo = foo or OsFooProvider()
```

### Naming

- **Protocol**: `FooProvider` (e.g., `BundleProvider`, `ImageProvider`)
- **Production impl**: Context-specific name (e.g., `BundleCache`, `DockerRuntime`, `GcsResolver`)
- **Test fake**: `MockFoo` or `FakeFoo` — prefer fakes with real logic over `Mock(spec=...)`

### Ground Rules

1. Constructor signature: Always `foo: FooProvider | None = None` (optional with default)
2. Prefer fakes with real logic over mocks with stubbed returns
3. One protocol per responsibility — don't combine unrelated operations
4. The injection point should be at the right abstraction level (e.g., inject `Resolver`, not `GcsApi`)

## Imports

Don't use TYPE_CHECKING. Use the real import. If there is a circular dependency:

* Prefer to resolve it with refactoring when sensible
* Otherwise use a protocol if you simply need the type information


## Architecture Notes

### Job vs Attempt

- **Job**: A logical unit of work with a hierarchical `job_id` (e.g., "my-exp/worker-0/task-1")
- **Attempt**: A single execution of a job, identified by `attempt_id` (0, 1, 2...)
- Jobs may be retried on failure/preemption; each retry creates a new attempt
- The controller tracks all attempts for history; workers execute individual attempts
- `(job_id, attempt_id)` uniquely identifies an execution on the worker side

### Key Environment Variables (injected into job containers)

- `IRIS_JOB_ID` - Hierarchical job identifier
- `IRIS_ATTEMPT_ID` - Current attempt number (0-indexed)
- `IRIS_WORKER_ID` - Worker executing the job
- `IRIS_CONTROLLER_ADDRESS` - Controller URL for sub-jobs/actors
- `IRIS_PORT_<NAME>` - Allocated port numbers

## Planning

Prefer _spiral_ plans over _linear_ plans. e.g. when implementing a new feature, make a plan which has step 1 as:

Step 1:
* Add the minimal changes to the `.proto` file
* Add the server stub code
* Add a client wiring
* Add an end-to-end test

Step 2:
* Extend proto with additional field
* Update server code
* Update client code
* Update tests

...

That is _each stage of the plan_ should be a independently testable,
self-contained unit of work. THis is preferable to plans which attempt to make
all of the changes for one area (e.g. all proto changes, then all server
changes, etc.)
