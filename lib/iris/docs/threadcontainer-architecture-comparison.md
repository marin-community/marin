# ThreadContainer: Composition vs Inheritance

## Current State (Composition)

Components hold a `ThreadContainer` as a member variable:

```python
class Controller:
    def __init__(self, ..., threads: ThreadContainer | None = None):
        self._threads = threads or ThreadContainer("controller")

    def start(self):
        self._threads.spawn(self._run_scheduling_loop, name="scheduling-loop")

    def stop(self):
        self._threads.stop()

class Worker:
    def __init__(self, ..., threads: ThreadContainer | None = None):
        self._threads = threads or ThreadContainer("worker")
        self._task_threads = self._threads.create_child("tasks")

class ClusterManager:
    def __init__(self, config):
        self._threads = ThreadContainer("cluster")
```

Every component manually delegates `stop()` and `wait()` to `self._threads`.

## Proposed State (Inheritance)

Components subclass `ThreadContainer` directly:

```python
class Controller(ThreadContainer):
    def __init__(self, ..., parent: ThreadContainer | None = None):
        super().__init__("controller")
        if parent:
            parent.add_child(self)
        # ... rest of init ...

    def start(self):
        self.spawn(self._run_scheduling_loop, name="scheduling-loop")

    # stop() inherited from ThreadContainer -- no delegation needed

class Worker(ThreadContainer):
    def __init__(self, ..., parent: ThreadContainer | None = None):
        super().__init__("worker")
        if parent:
            parent.add_child(self)
        self._task_threads = self.create_child("tasks")

class ClusterManager(ThreadContainer):
    def __init__(self, config):
        super().__init__("cluster")
```

## Comparison

### Boilerplate

**Composition**: Every component needs:
- `self._threads = threads or ThreadContainer(name)` in `__init__`
- Delegation in `stop()`: signal wake events, then `self._threads.stop()`
- Delegation in `wait()`: `self._threads.wait()`

**Inheritance**: Components get `stop()`, `wait()`, `spawn()`, `alive_threads()` for free. No delegation layer.

Winner: **Inheritance** -- eliminates ~5 lines of boilerplate per component.

### API Consistency

Today, `ManagedThread` and `ThreadContainer` have similar but not identical interfaces:

| Method       | ManagedThread | ThreadContainer |
|-------------|---------------|-----------------|
| `stop()`    | yes           | yes             |
| `wait()`    | no (`join()`) | yes             |
| `is_alive`  | yes (property)| no              |
| `start()`   | yes           | no (spawn auto-starts) |

With inheritance, components themselves become part of the thread tree. A `Protocol` could unify them:

```python
class Stoppable(Protocol):
    def stop(self) -> None: ...
    def wait(self) -> None: ...
    @property
    def is_alive(self) -> bool: ...
```

This works equally well with composition or inheritance. The question is whether the component **is** a thread container or **has** one.

Winner: **Tie** -- a `Stoppable` protocol works either way.

### Parent-Child Registration

**Composition**: Parent passes its ThreadContainer to children, children create a child container or use it directly:
```python
# ClusterManager passes its container to Controller
controller = Controller(config, threads=self._threads.create_child("controller"))
```

**Inheritance**: Parent registers child explicitly, or child registers itself:
```python
# Option A: parent registers
controller = Controller(config)
self.add_child(controller)

# Option B: child registers via parent arg
controller = Controller(config, parent=self)
```

Winner: **Composition** -- the current `threads=` parameter pattern is cleaner and doesn't require `add_child()` which creates a coupling where the child knows about the parent's type.

### Multiple Inheritance Concerns

**Composition**: No concerns. Components can inherit from whatever they want.

**Inheritance**: If a component needs to inherit from something else, you get multiple inheritance. Today this isn't a problem (Controller, Worker, ClusterManager don't inherit from anything). But it's a constraint on future design.

More concretely: Python MRO means `super().__init__()` calls need careful ordering. Adding ThreadContainer as a base class to an existing hierarchy (e.g., if Worker ever needed to extend a framework class) would be painful.

Winner: **Composition** -- no future constraints.

### stop() Override Complexity

Components don't just call `self._threads.stop()`. They do additional work:

**Controller.stop()**:
1. Wake events (`self._wake_event.set()`, `self._heartbeat_event.set()`)
2. `self._threads.stop()`
3. `self._dispatch_executor.shutdown(wait=True)`
4. `self._autoscaler.shutdown()`

**Worker.stop()**:
1. `self._task_threads.stop()` (children first)
2. `self._server.should_exit = True`
3. `self._threads.stop()`
4. Clean up containers and temp dirs

With inheritance, `stop()` must override `ThreadContainer.stop()` and call `super().stop()` at the right point:

```python
class Controller(ThreadContainer):
    def stop(self):
        self._wake_event.set()
        self._heartbeat_event.set()
        super().stop()  # must come before or after executor shutdown?
        self._dispatch_executor.shutdown(wait=True)
        self._autoscaler.shutdown()
```

The ordering of `super().stop()` relative to component-specific cleanup is subtle and error-prone. With composition, the ordering is explicit -- you call `self._threads.stop()` exactly where it makes sense.

Winner: **Composition** -- explicit ordering is clearer than `super()` placement.

### Encapsulation

**Composition**: ThreadContainer's full API (`spawn`, `spawn_server`, `create_child`, `alive_threads`) is internal to the component. External callers see only `controller.stop()`.

**Inheritance**: External callers can call `controller.spawn(...)`, `controller.create_child(...)`, etc. This leaks thread management internals. You'd need to hide these methods (underscore prefix, `__slots__`, or just documentation), which defeats the purpose.

Winner: **Composition** -- keeps thread management as an implementation detail.

### Testability

**Composition**: Tests can inject a ThreadContainer, or let components create their own. The `threads=` parameter is already used by ClusterManager to pass a shared container:
```python
controller = Controller(config, threads=parent_threads.create_child("controller"))
```

**Inheritance**: Tests interact with the component directly since it IS a ThreadContainer. Slightly less indirection, but also less control -- you can't swap in a mock ThreadContainer.

Winner: **Composition** -- injectable ThreadContainer is more testable.

## Recommendation: Keep Composition

Inheritance saves a few lines of delegation boilerplate but introduces real costs:

1. **Leaks thread management API** to all callers of Controller/Worker/ClusterManager
2. **Makes stop() ordering subtle** via super() placement instead of explicit calls
3. **Constrains future inheritance** (multiple inheritance risk)
4. **Reduces testability** (can't inject a mock container)

The boilerplate cost of composition is small (3-5 lines per component) and concentrated in `__init__` and `stop()`. The clarity benefit of explicit `self._threads.stop()` placement in shutdown sequences is worth it.

### What to Improve Instead

Rather than switching to inheritance, improve the composition pattern:

1. **Add `is_alive` to ThreadContainer** for parity with ManagedThread
2. **Add a `Stoppable` Protocol** so callers can treat ManagedThread and ThreadContainer uniformly when needed
3. **Consider adding `wait()` to ManagedThread** (alias for `join()`) for API symmetry
4. **Keep the `threads=` injection pattern** -- it's clean and testable

### Root vs Child Container Handling

With composition, the pattern is already clear:
- **Root**: `ClusterManager` creates `ThreadContainer("cluster")` and owns it
- **Child**: `ClusterManager` passes `self._threads` (or a child of it) to Controller/Worker via `threads=` parameter
- **Leaf**: Components call `self._threads.create_child()` for sub-groups (e.g., Worker's task threads)

No changes needed here.
