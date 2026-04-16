# Marin Python Coding Standards

This document is the authoritative reference for automated code cleanup workflows
and human reviewers. It consolidates rules from `AGENTS.md`, project guidelines,
and Python best practices into a single, unambiguous checklist.

Every rule includes a rationale and a concrete bad/good example.

---

## 1. Module Structure

### 1.1 Imports

All imports at the top of the file. No mid-function imports
except to guard optional third-party packages. Fix circular imports
by refactoring to a common package instead of hacking around them.

```python
# BAD
def train(config):
    from marin.training.utils import setup_logging  # buried import
    setup_logging()

# GOOD
from marin.training.logging_setup import setup_logging

def train(config):
    setup_logging()
```

No `TYPE_CHECKING` guards. Fix import cycles structurally — extract common
code/interfaces into a separate module, or use `Protocol` at module boundaries when appropriate.

### 1.2 Naming

| Convention | Rule | Example |
|---|---|---|
| Modules | Descriptive nouns. No `*_utils.py`. | `text_cleaning.py` not `text_utils.py` |
| Functions | Verb phrases reflecting return type. | `task_status()` not `probe_task()` |
| Abbreviations | Spell out. | `executor` not `exe` |
| Constants | `UPPER_SNAKE_CASE` at module level. | `MAX_RETRY_COUNT = 3` |

### 1.3 License header

License headers are managed by pre-commit.py. See below.

---

## 2. Functions and Classes

### 2.1 Prefer top-level functions

Use top-level functions when code does not mutate shared state. Reduce deep
inheritance hierarchies. Classes are appropriate when they manage mutable
resources or need polymorphism.

### 2.2 Early returns

Use early exits to reduce nesting.

```python
# BAD
def process(items):
    if items:
        if len(items) > 0:
            result = []
            for item in items:
                result.append(transform(item))
            return result
    return []

# GOOD
def process(items):
    if not items:
        return []
    return [transform(item) for item in items]
```

### 2.3 Function size

Break functions into focused pieces when the pieces are reusable. Follow the
[rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming)):
do not abstract until you have three instances of duplication. Three similar
lines of code is better than a premature abstraction.

### 2.4 No boolean flags for variant behavior

```python
# BAD
class Vllm:
    def __init__(self, docker: bool = False): ...

# GOOD
class NativeVllm: ...
class DockerVllm: ...
```

### 2.5 Accept only what is necessary

Functions should accept the narrowest possible interface. Replace `parallel: bool`
with `num_workers: int`. Normalize inputs to a canonical format once at the
boundary, not throughout the call chain.

### 2.6 Enforced keyword arguments

Use keyword-only arguments (after `*`) for non-obvious parameters and flags.
This prevents positional mistakes and makes call sites self-documenting.

```python
# BAD
def train(config, 128, True, 0.001): ...

# GOOD
def train(config, *, batch_size: int, shuffle: bool, lr: float): ...
```

---

## 3. Types and Data Structures

### 3.1 Typed data over raw dicts

Use `dataclass` or `namedtuple` instead of raw dictionaries for structured data.
Prefer `dataclass(frozen=True)` for immutable value objects — frozen dataclasses
are safer defaults since they prevent accidental mutation and are hashable.

### 3.2 StrEnum over string keys

Use `StrEnum` with `auto()` for string enums. LLMs often generate manual string
assignments; prefer `auto()` to keep values consistent and reduce boilerplate.

```python
# BAD
status = "running"
if task["status"] == "running": ...

# GOOD
from enum import StrEnum, auto

class TaskStatus(StrEnum):
    RUNNING = auto()
    COMPLETED = auto()
```

### 3.3 Protocol for decoupling

Use `Protocol` instead of hard-coupling to concrete types. Do not use
`hasattr` checks or `isinstance` for duck-typing.

### 3.4 No `X | str` unions

Avoid union types like `ModelConfig | str` that require `isinstance` branching.
Pick one input type and normalize at the entry point.

### 3.5 Enum over compound booleans

Replace `is_started and not is_done` with an explicit state enum.
Use StrEnum and "auto" when appropriate for human readability.

---

## 4. Configuration

- No `default_*` wrapper functions that obscure the underlying mechanism.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over environment variables.
- Composition over inheritance: embed sub-configs as fields, do not subclass.
- Use `dataclasses.replace` over mutating config objects in-place.

---

## 5. Error Handling

- Let exceptions propagate by default.
- Only catch exceptions to add meaningful context and re-raise, or to
  intentionally alter control flow.
- Never swallow exceptions.
- Assert liberally. Prefer `raise ValueError("...")` over silent fallbacks.
- No over-protective `try/except` blocks or defensive `None` checks for
  internal code paths.

```python
# BAD
def get_model(name):
    try:
        return registry[name]
    except KeyError:
        return None  # silently swallowed

# GOOD
def get_model(name):
    if name not in registry:
        raise ValueError(f"Unknown model: {name!r}. Available: {sorted(registry)}")
    return registry[name]
```

---

## 6. Logging and I/O

- Use `logging` module, not `print` (except in scripts and interactive debugging).
- Separate computation from I/O: compute a result, then write/upload it.
- Use context managers for resource lifecycle (`open`, connections, temp dirs).
- Use `fsspec.open` for filesystem access. Do not special-case GCS paths.
- Do not copy large artifacts to local filesystem; stream through fsspec.

---

## 7. Comments and Documentation

### 7.1 When to write comments

- Module-level and class-level docstrings explaining purpose and design.
- Non-trivial public functions: Google-style docstrings with args, returns,
  and non-obvious context (do not paraphrase the code).
- Inline comments for subtle logic or non-obvious boolean arguments.

### 7.2 When NOT to write comments

```python
# BAD — restates the code
# Initialize the model
model = Model()

# BAD — obvious from the variable name
# Use in-memory rollout queue
rollout_queue = InMemoryRolloutQueue()

# GOOD — explains WHY
# Flush before checkpoint because the writer holds a file lock
writer.flush()
```

### 7.3 Stale comments

Delete stale comments immediately on discovery. A wrong comment is worse than
no comment.

---

## 8. Dead Code and Cleanup Targets

These are the specific patterns that automated cleanup workflows should look for:

### 8.1 Unused code

- Imports that are never referenced.
- Functions/methods that are never called (check with `rg`).
- Variables assigned but never read.
- Commented-out code blocks (delete, it is in git history).
- `pass` statements in non-empty classes/functions.
- Empty `__init__` methods that only call `super().__init__()`.

### 8.2 Deprecated patterns

- `hasattr` checks used for compatibility shimming.
- `# type: ignore` without a specific error code.
- `noqa` without a specific rule code.
- `TODO` or `FIXME` comments older than 90 days (check with `git blame`).
- Bare `except:` or `except Exception:` without re-raise.

### 8.3 Complexity smells

- Functions longer than 50 lines (candidates for extraction).
- More than 3 levels of indentation (use early returns).
- More than 5 parameters (consider a config dataclass).
- Duplicate code blocks (3+ occurrences → extract).
- Boolean parameters that switch behavior (split into separate functions/classes).

### 8.4 Type annotation gaps

- Public functions missing return type annotations.
- Dataclass fields using `Any` where a concrete type is known.
- `Dict[str, Any]` where a `TypedDict` or dataclass would be appropriate.

### 8.5 Test quality

- Tests with no assertions.
- Tests that assert on private (`_`-prefixed) attributes.
- `time.sleep()` in tests (inject time instead).
- Mocks of internal functions (mock at I/O boundaries only).
- `@pytest.mark.skip` without a linked issue.

### 8.6 Documentation drift

- Docstrings whose parameter lists don't match the function signature.
- README or doc references to renamed/deleted modules.
- Stale examples that no longer run.

---

## 9. Formatting and Lint Rules

Enforced automatically by `./infra/pre-commit.py --all-files --fix`.

---

## 10. Dependency Direction

```
iris → {levanter, zephyr} → marin → haliax
```

Each layer may only import from layers to its right. Never introduce reverse
dependencies. Cleanup workflows should flag any violation.

---

## 11. Testing Standards

- Use `pytest` with fixtures and parameterization. Top-level `def test_*`
  functions, not test classes.
- Prefer integration-style tests that validate observable behavior.
- No tautological tests (asserting that a constructor sets an attribute).
- No mocks except at I/O boundaries. Prefer fakes backed by in-memory state.
- Every test function must contain at least one `assert` or `pytest.raises`.
- Run relevant tests before submitting: `uv run pytest -m 'not slow' <paths>`.

---

## 12. LLM-Generated Code Anti-Patterns

Automated cleanup should specifically watch for these patterns commonly
introduced by code-generation tools:

1. **Over-protective error handling**: `try/except` around code that cannot fail.
2. **Defensive None checks**: `if x is not None` when `x` is never `None`.
3. **Verbose docstrings**: restating function name and parameters without adding
   information.
4. **`__all__` in `__init__.py`**: unnecessary in this codebase.
5. **Boolean dispatch**: `if self.mode == "a": ... elif self.mode == "b": ...`
   instead of separate classes.
6. **Environment variables for configuration**: use explicit parameters.
7. **Unnecessary abstractions**: base classes, factories, or registries for a
   single implementation.
8. **Trailing summaries**: comments like `# End of function` or `# Done`.
