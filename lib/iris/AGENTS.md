# Agent Tips

* Use the connect/RPC abstractions to implement and perform RPC calls. DO NOT use httpx or raw HTTP.
* Use scripts/generate_protos.py to regenerate files after changing the `.proto` files.
* Prefer _shallow_, _functional_ code which returns control quickly to the user, vs callbacks or inheritance.

```
class Scheduler:
  def add_job():
  def add_worker():
  def compute_schedule() -> ScheduledJobs:

class Runner:
  def run_jobs(ScheduledJobs)
```

is preferable to:

```
class Scheduler:
  def __init__(self, job_creator: JobCreator):
    self.job_creator = job_creator
  def run(self):
    ... self.job_creator.create_job()
```

* Tests should test stable behavior, not implementation details.

ABSOLUTELY DO NOT test things that are trivially caught by the type checker.
Explicitly that means:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value - they make our code more brittle.

Test _stable behavior_ instead. You can use mocks as needed to isolate
environments (e.g.  mock around a remote API), but prefer "fakes" -- e.g. create
a real database but with fake data -- when reasonable.

## Documentation

ALWAYS read the docs for the appropriate area.
IF they disagree with the code, ALWAYS add a task to update them.

Documentation should be kept up-to-date as code changes. When implementing new features or making significant changes, update the relevant documentation files:

@README.md - Main overview, CLI reference, and quick start

## Protocols and Testing

Non-trivial public classes should define a protocol which represents their
_important_ interface characteristics. Use this protocol in type hints for
when the class is used instead of the concrete class.

Test to this protocol, not the concrete class: the protocol should describe the
interesting behavior of the class, but not betray the implementation details.

(You may of course _instantiate_ the concrete class for testing.)

## Imports

Don't use TYPE_CHECKING. Use the real import. If there is a circular dependency:

* Prefer to resolve it with refactoring when sensible
* Otherwise use a protocol if you simply need the type information

## RPC/API Accessibility

Any functionality exposed by the worker or controller dashboards must also be
available via RPC. The dashboards should be a friendly interface on top of the
machine accessible RPC API, and should not use internal APIs (except for
efficiency). For example, if we wanted to show the scheduling status for a task,
we should define a new RPC endpoint `/TestSchedule(task_id)` and use that from
the dashboard, rather than creating a scheduler and running it manually.

## Architecture Notes

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

When adding new modules or significant features:
1. Update the README with a brief overview and usage examples
2. Add detailed documentation to the appropriate docs/ file
3. Reference the documentation from this AGENTS.md file

**Key documentation areas:**

| Area | File | Description |
|------|------|-------------|
| Architecture | README.md | High-level architecture, CLI reference, quick start |
| Autoscaler Design | docs/autoscaler-v0-design.md | Technical specification, threading model |
| Thread Safety | docs/thread-safety.md | Thread management, test synchronization best practices |
| Original Design | docs/fray-zero.md | Rationale and design decisions |

## Key Modules

### Autoscaler and VM Management

The autoscaler runs inside the Controller process and manages cloud VMs based on pending task demand. Key files:

```
src/iris/
├── cli/                         # CLI package (cluster, build, run, debug commands)
│   ├── main.py                  # Top-level iris group
│   ├── cluster.py               # Cluster lifecycle, controller, VM ops, dashboard
│   ├── build.py                 # Image build commands
│   ├── debug.py                 # Debugging & validation
│   ├── run.py                   # Command passthrough job submission
│   └── rpc.py                   # Dynamic RPC CLI
├── cluster/
│   ├── controller/
│   │   ├── controller.py        # Controller with integrated autoscaler
│   │   └── main.py              # Controller daemon CLI (serve command)
│   └── vm/
│       ├── autoscaler.py        # Core scaling logic
│       ├── scaling_group.py     # Per-group state tracking
│       ├── gcp.py               # GCP TPU management
│       ├── manual.py            # Pre-existing host management
│       └── config.py            # Config loading + factory functions
```

See [README.md](README.md) for CLI usage and configuration examples.
