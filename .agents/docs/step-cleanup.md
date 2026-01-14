# ExecutorStep Type Design Cleanup

## Problem Statement

The current `ExecutorStep` design suffers from pervasive type punning throughout the codebase. Config fields can contain either:
- Concrete values (strings, ints, lists)
- `InputName` references (dependencies on other steps)
- `ExecutorStep` objects (implicitly converted to `InputName`)
- `OutputName` references (self-references)
- `VersionedValue` wrappers

This leads to defensive `isinstance` checks scattered across 15+ files:

```python
# From experiments/defaults.py:612
if isinstance(tokenized, InputName | ExecutorStep):
    pretraining_data = lm_data_config(training_set=tokenized, ...)
else:
    pretraining_data = tokenized  # Already a config

# From marin/processing/tokenize/tokenize.py:216
if any(isinstance(x, InputName | ExecutorStep) for x in input_paths):
    return input_paths  # Can't expand paths yet

# From experiments/evals/evals.py:104
if isinstance(step, ExecutorStep):
    model_step_path = output_path_of(step, "hf")
elif isinstance(step, InputName):
    if step.step is None:
        model_step_path = step.name
    else:
        model_step_path = step if step.name is not None else output_path_of(...)
```

### Why This Is Problematic

1. **Type Safety**: Static type checkers can't help - every function accepting paths must handle multiple types
2. **Cognitive Load**: Developers must constantly think "is this resolved yet?"
3. **Error-Prone**: Easy to forget an `isinstance` check, leading to runtime failures
4. **Viral Complexity**: Functions that accept `InputName | str` must propagate that union upward
5. **Testing Difficulty**: Tests must cover both resolved and unresolved code paths

---

## Current Design Analysis

### How It Works Today

```python
@dataclass
class MyConfig:
    input_path: str           # Actually: str | InputName | ExecutorStep
    output_path: str          # Actually: str | OutputName
    batch_size: int           # Actually: int | VersionedValue[int]

step = ExecutorStep(
    name="process",
    fn=process_fn,
    config=MyConfig(
        input_path=output_path_of(upstream_step),  # InputName
        output_path=this_output_path(),            # OutputName
        batch_size=versioned(32),                  # VersionedValue
    ),
)
```

The magic happens in two phases:

1. **Version Computation** (`collect_dependencies_and_version`): Recursively walks config, extracts `InputName` → dependencies, extracts `VersionedValue` → version dict
2. **Config Instantiation** (`instantiate_config`): Replaces `InputName` → actual path, `OutputName` → output path, `VersionedValue` → unwrapped value

### Where Type Punning Causes Pain

| Location | Pattern | Why It Exists |
|----------|---------|---------------|
| `defaults.py:612` | `isinstance(tokenized, InputName \| ExecutorStep)` | Wrapper functions accept both step refs and pre-built configs |
| `tokenize.py:216` | `any(isinstance(x, InputName \| ExecutorStep) for x in paths)` | Can't expand glob patterns on unresolved paths |
| `evals.py:104` | Multi-branch isinstance cascade | Need to extract model name differently per type |
| `speedrun.py:368` | `isinstance(tokenized, InputName \| ExecutorStep)` | Different config construction paths |
| `executor_utils.py:75` | `isinstance(checkpoint_path, InputName)` | Path parsing logic differs |
| `hf_upload.py:83` | `isinstance(input_path, InputName \| ExecutorStep)` | Certificate path construction |

---

## Proposed Solutions

### Option A: Tracer-Based Design (JAX-style)

**Core Idea**: User provides a function that receives "tracer" objects. Inside the function, all values are consistently tracers. The system traces through the function to discover the computation graph.

```python
from marin.execution import step, Tracer

@step(name="process")
def process_step(ctx: StepContext) -> ProcessConfig:
    # All dependencies declared via ctx.require()
    upstream: Tracer[str] = ctx.require(upstream_step)
    tokenizer: Tracer[str] = ctx.require(tokenizer_step / "model")

    # Tracers support path operations
    data_path = upstream / "data"

    return ProcessConfig(
        input_path=data_path,        # Tracer[str], not str
        output_path=ctx.output,      # Tracer[str]
        batch_size=32,               # Plain int
    )

# Usage
process = process_step()  # Returns ExecutorStep
```

**Implementation Sketch**:

```python
@dataclass
class Tracer(Generic[T]):
    """Represents an unresolved value that will be concrete at execution time."""
    _source: ExecutorStep | None
    _subpath: str | None
    _versioned: bool = False

    def __truediv__(self, other: str) -> "Tracer[T]":
        new_subpath = f"{self._subpath}/{other}" if self._subpath else other
        return Tracer(self._source, new_subpath)

    def resolve(self, output_paths: dict[ExecutorStep, str]) -> T:
        if self._source is None:
            return self._subpath
        base = output_paths[self._source]
        return f"{base}/{self._subpath}" if self._subpath else base

class StepContext:
    """Context object passed to step-defining functions."""
    _dependencies: list[ExecutorStep]
    _pseudo_deps: list[ExecutorStep]

    def require(self, step: ExecutorStep | InputName, *, blocking: bool = True) -> Tracer[str]:
        actual_step = step.step if isinstance(step, InputName) else step
        subpath = step.name if isinstance(step, InputName) else None

        if blocking:
            self._dependencies.append(actual_step)
        else:
            self._pseudo_deps.append(actual_step)

        return Tracer(actual_step, subpath)

    @property
    def output(self) -> Tracer[str]:
        return Tracer(None, None)  # Resolved to output_path at execution

def step(name: str, resources: ResourceConfig | None = None):
    """Decorator to define an executor step."""
    def decorator(fn: Callable[[StepContext], ConfigT]) -> Callable[[], ExecutorStep[ConfigT]]:
        def make_step() -> ExecutorStep[ConfigT]:
            ctx = StepContext()
            config = fn(ctx)

            # Config now contains Tracer objects
            # We store the tracers directly - they'll be resolved at execution
            return ExecutorStep(
                name=name,
                fn=_resolve_and_call,  # Wrapper that resolves tracers
                config=config,
                _context=ctx,  # Store for dependency extraction
                resources=resources,
            )
        return make_step
    return decorator
```

**Pros**:
- Consistent types inside step functions (everything is `Tracer[T]`)
- Dependencies explicitly declared via `ctx.require()`
- Similar to JAX/TF patterns developers may know
- Type checker can verify `Tracer[str]` vs `str`

**Cons**:
- More complex mental model
- Breaking change to all existing code
- Tracers need to support all operations users might want (string formatting, path manipulation)
- Runtime overhead from tracing

---

### Option B: Explicit Dependencies with Concrete Values

**Core Idea**: Separate dependencies from config. User function receives only concrete values.

```python
from marin.execution import ExecutorStep, Dep, Output

step = ExecutorStep(
    name="process",
    fn=process_fn,
    deps={
        "input_data": upstream_step / "data",
        "tokenizer": tokenizer_step / "model",
    },
    config=ProcessConfig(
        batch_size=32,
        # No paths here - they come from deps
    ),
)

# The function signature declares what it needs:
def process_fn(
    config: ProcessConfig,
    *,
    input_data: str,      # Concrete path, injected from deps
    tokenizer: str,       # Concrete path, injected from deps
    output_path: str,     # Always provided
):
    # All values are concrete strings
    files = list_files(input_data)  # Works!
    ...
```

**Implementation Sketch**:

```python
@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT]):
    name: str
    fn: Callable
    config: ConfigT
    deps: dict[str, Dep] = field(default_factory=dict)
    resources: ResourceConfig | None = None

    # deps can be:
    # - ExecutorStep (resolved to its output_path)
    # - InputName (resolved to step.output_path / name)
    # - str (hardcoded path, used as-is)

Dep = ExecutorStep | InputName | str

def instantiate_and_call(
    step: ExecutorStep,
    output_path: str,
    output_paths: dict[ExecutorStep, str],
):
    # Resolve all deps to concrete strings
    resolved_deps = {}
    for name, dep in step.deps.items():
        if isinstance(dep, ExecutorStep):
            resolved_deps[name] = output_paths[dep]
        elif isinstance(dep, InputName):
            base = output_paths[dep.step] if dep.step else ""
            resolved_deps[name] = f"{base}/{dep.name}" if dep.name else base
        else:
            resolved_deps[name] = dep

    # Call function with config + resolved deps
    step.fn(step.config, output_path=output_path, **resolved_deps)
```

**Pros**:
- User code ALWAYS works with concrete values
- Clear separation: deps are step references, config is values
- No isinstance checks in user code
- Easy to understand and debug
- Function signatures document dependencies explicitly

**Cons**:
- More verbose step definitions
- Config and deps are split (related data in two places)
- Breaking change to all existing code
- Need to update function signatures

---

### Option C: Hybrid - Typed Config with Automatic Resolution (Recommended)

**Core Idea**: Keep config-based approach but use distinct types and automatic resolution. User functions receive a "resolved" version of their config.

```python
from marin.execution import ExecutorStep, Dep, Output, versioned

@dataclass
class ProcessConfig:
    input_path: Dep[str]       # Explicitly a dependency
    output_path: Output[str]   # Explicitly an output reference
    batch_size: int            # Plain value
    learning_rate: Versioned[float]  # Affects version hash

@dataclass
class ResolvedProcessConfig:
    input_path: str            # Concrete
    output_path: str           # Concrete
    batch_size: int
    learning_rate: float

step = ExecutorStep(
    name="process",
    fn=process_fn,
    config=ProcessConfig(
        input_path=upstream_step / "data",   # Dep[str]
        output_path=this_output(),           # Output[str]
        batch_size=32,
        learning_rate=versioned(0.001),
    ),
)

def process_fn(config: ResolvedProcessConfig):
    # config.input_path is a str, not a Dep
    files = list_files(config.input_path)  # Works!
```

**Implementation Sketch**:

```python
from typing import Generic, TypeVar, get_type_hints, get_origin, get_args

T = TypeVar("T")

class Dep(Generic[T]):
    """A dependency on another step's output. Resolves to T at execution time."""
    def __init__(self, step: ExecutorStep | None, subpath: str | None = None):
        self._step = step
        self._subpath = subpath

    def __truediv__(self, other: str) -> "Dep[T]":
        new_subpath = f"{self._subpath}/{other}" if self._subpath else other
        return Dep(self._step, new_subpath)

    @classmethod
    def hardcoded(cls, path: str) -> "Dep[str]":
        return Dep(None, path)

class Output(Generic[T]):
    """Reference to this step's output path. Resolves to T at execution time."""
    def __init__(self, subpath: str | None = None):
        self._subpath = subpath

    def __truediv__(self, other: str) -> "Output[T]":
        new_subpath = f"{self._subpath}/{other}" if self._subpath else other
        return Output(new_subpath)

class Versioned(Generic[T]):
    """A value that contributes to the step's version hash."""
    def __init__(self, value: T):
        self.value = value

def versioned(value: T) -> Versioned[T]:
    return Versioned(value)

def this_output(subpath: str | None = None) -> Output[str]:
    return Output(subpath)

# Resolution happens automatically
def resolve_config(config: ConfigT, output_path: str, output_paths: dict) -> ConfigT:
    """Recursively resolve Dep/Output/Versioned to concrete values."""
    # Uses dataclass introspection to create resolved copy
    ...
```

**The Key Insight**: By using `Dep[str]`, `Output[str]`, and `Versioned[T]` wrapper types:
1. Type hints are honest about what values are
2. Static analysis tools can distinguish resolved vs unresolved configs
3. No isinstance checks needed in user code - resolution is automatic
4. Type checkers enforce that functions receive resolved configs

**Pros**:
- Config-centric (familiar pattern)
- Type-safe: `Dep[str]` is not `str`
- Automatic resolution - user functions get concrete values
- Incremental migration possible
- Clear documentation of what's a dependency

**Cons**:
- Need wrapper types (`Dep`, `Output`, `Versioned`)
- Some verbosity in type annotations
- Migration effort for existing configs

---

## Syntax Sugar for Option C

To reduce verbosity, we can provide convenient shortcuts:

### Path Operator Overloads

```python
# Instead of:
Dep(upstream_step, "data/train")

# Write:
upstream_step / "data" / "train"  # Returns Dep[str]
```

### Decorator for Step Definition

```python
@executor_step(
    name="process",
    resources=ResourceConfig(gpu_type="a100", num_gpus=8),
)
@dataclass
class ProcessStep:
    input_path: Dep[str]
    output_path: Output[str] = this_output()
    batch_size: int = 32

# Creates ExecutorStep automatically with process_fn as the function
```

### Automatic Config Resolution

```python
from marin.execution import resolved

@resolved  # Marks that this function receives resolved config
def process_fn(config: ProcessConfig):
    # Type checker knows config.input_path is str, not Dep[str]
    ...
```

### Inference of Step Dependencies

```python
# The framework can walk the config and find all Dep[] fields automatically
step = ExecutorStep(
    name="process",
    fn=process_fn,
    config=ProcessConfig(
        input_path=upstream / "data",
        output_path=this_output(),
        batch_size=32,
    ),
)
# Dependencies inferred from Dep[] fields in config
```

---

## Migration Path

### Phase 1: Add New Types (Non-Breaking)

1. Add `Dep[T]`, `Output[T]`, `Versioned[T]` types
2. Make them aliases for existing types initially:
   ```python
   Dep = InputName  # Temporary alias
   Output = OutputName
   Versioned = VersionedValue
   ```
3. Update documentation to prefer new types

### Phase 2: Parallel Support

1. Support both old (`InputName`) and new (`Dep`) in executor
2. Add deprecation warnings for old types
3. Provide codemod script for migration

### Phase 3: Function Signature Updates

1. Update executor to call functions with resolved configs
2. Functions can declare they want resolved configs via type hints
3. Old functions continue to work (receive unresolved configs)

### Phase 4: Remove Old Types

1. Remove `InputName`, `OutputName`, `VersionedValue`
2. Remove isinstance checks from codebase
3. Update all wrapper functions

---

## Comparison Matrix

| Criterion | Current | Option A (Tracer) | Option B (Explicit Deps) | Option C (Typed Hybrid) |
|-----------|---------|-------------------|--------------------------|-------------------------|
| Type Safety | Poor | Good | Good | Excellent |
| Cognitive Load | High | Medium | Low | Low |
| Migration Effort | N/A | High | High | Medium |
| Verbosity | Low | Medium | High | Medium |
| Familiar Pattern | Yes | JAX-like | Django-like | Dataclass-like |
| isinstance checks | Many | Few | None | None |
| Static Analysis | Poor | Good | Good | Excellent |

---

## Recommendation

**Option C (Hybrid with Typed Wrappers)** provides the best balance:

1. **Minimal conceptual change**: Still config-centric, still dataclasses
2. **Maximum type safety**: `Dep[str]` is not assignable to `str`
3. **Automatic resolution**: User functions receive concrete values
4. **Incremental migration**: Can coexist with old code during transition
5. **Clear documentation**: Type hints explain what each field is

The key changes:
- `InputName` → `Dep[str]` (honest about what it is)
- `OutputName` → `Output[str]`
- `VersionedValue[T]` → `Versioned[T]`
- Functions receive resolved configs (all `Dep`/`Output`/`Versioned` replaced with concrete values)

This eliminates the need for isinstance checks because:
1. In config definition: fields are `Dep[str]`, not `str`
2. In function execution: fields are `str` (resolved)
3. Never a union type that requires runtime checking

---

## Appendix: Full Example

```python
from dataclasses import dataclass
from marin.execution import (
    ExecutorStep,
    Dep,
    Output,
    Versioned,
    versioned,
    this_output,
    executor_main,
)

# Config with explicit typing
@dataclass(frozen=True)
class TokenizeConfig:
    train_paths: list[Dep[str]]      # Dependencies on other steps
    validation_paths: list[Dep[str]]
    cache_path: Output[str]          # This step's output
    tokenizer: Versioned[str]        # Affects version hash
    batch_size: int = 1000           # Plain config value

# Step definition (unchanged syntax)
download_step = ExecutorStep(
    name="download/fineweb",
    fn=download_hf,
    config=DownloadConfig(
        id="HuggingFaceFW/fineweb",
        output_path=this_output(),
    ),
)

tokenize_step = ExecutorStep(
    name="tokenize/fineweb",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[download_step / "train"],     # Dep[str]
        validation_paths=[download_step / "val"],  # Dep[str]
        cache_path=this_output(),                  # Output[str]
        tokenizer=versioned("meta-llama/Llama-3-8B"),
        batch_size=2000,
    ),
)

# Function receives resolved config
def tokenize(config: TokenizeConfig):  # Actually receives resolved version
    # config.train_paths is list[str], not list[Dep[str]]
    for path in config.train_paths:
        files = glob(f"{path}/*.jsonl")  # Works! path is a string
        ...

if __name__ == "__main__":
    executor_main(steps=[tokenize_step])
```

The magic: `config: TokenizeConfig` in the function signature means "resolved TokenizeConfig" where all `Dep[T]` → `T`, `Output[T]` → `T`, `Versioned[T]` → `T`.
