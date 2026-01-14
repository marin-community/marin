# ExecutorStep Type Cleanup: Tracer-Based Design

## Problem Statement

The current `ExecutorStep` design suffers from pervasive type punning. Config fields can hold either concrete values OR step references (`InputName`, `ExecutorStep`, `OutputName`). This forces isinstance checks throughout the codebase:

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
    # ... complex branching
```

**Core Problem**: User code must constantly ask "is this resolved yet?" because the same variable can hold either a reference or a concrete value.

---

## Solution: Tracer-Based Design

### Core Principle

Split the world into two distinct phases with distinct types:

| Phase | Types | When |
|-------|-------|------|
| **Step Construction** | `StepRef` (tracer) | When defining pipelines |
| **Step Execution** | Concrete values (`str`, `int`, etc.) | When running on cluster |

In the construction phase, ALL step-related values are `StepRef`. No ambiguity. No isinstance checks.

In the execution phase, ALL values are concrete. The executor resolves everything before calling the function.

### What is StepRef?

`StepRef` is a tracer object that represents "a path that will exist when this step runs." It tracks:
- Which upstream step it depends on (or `None` for output paths)
- What subpath within that step's output
- Whether it's a blocking or non-blocking dependency

```python
@dataclass(frozen=True)
class StepRef:
    """A reference to a path that will be resolved at execution time."""
    _step: ExecutorStep | None  # None means "this step's output"
    _subpath: str | None = None
    _blocking: bool = True

    def __truediv__(self, subpath: str) -> "StepRef":
        """Navigate to subpath: step / "data" / "train" """
        new_subpath = f"{self._subpath}/{subpath}" if self._subpath else subpath
        return StepRef(self._step, new_subpath, self._blocking)

    def nonblocking(self) -> "StepRef":
        """Mark as pseudo-dependency (doesn't block execution)."""
        return StepRef(self._step, self._subpath, blocking=False)
```

### Step Definition with Context

Steps are defined using a function that receives a `StepContext`:

```python
from marin.execution import step, StepContext, StepRef

@step(name="tokenize/fineweb")
def tokenize_fineweb(ctx: StepContext):
    # Declare dependencies - returns StepRef
    download = ctx.require(download_step)

    return TokenizeConfig(
        train_paths=[download / "train"],      # StepRef
        validation_paths=[download / "val"],   # StepRef
        cache_path=ctx.output,                 # StepRef (this step's output)
        tokenizer="meta-llama/Llama-3-8B",     # Plain value
    )

# Create the step
tokenize_step = tokenize_fineweb()
```

### StepContext API

```python
class StepContext:
    """Context for defining a step's dependencies and outputs."""

    def require(self, dep: ExecutorStep | StepRef) -> StepRef:
        """
        Declare a dependency on another step.
        Returns a StepRef that will resolve to that step's output path.
        """
        ...

    def require_nonblocking(self, dep: ExecutorStep | StepRef) -> StepRef:
        """
        Declare a pseudo-dependency (for versioning but doesn't block execution).
        Use for checkpoints from still-running training.
        """
        ...

    @property
    def output(self) -> StepRef:
        """Reference to this step's output path."""
        ...

    def output_subpath(self, subpath: str) -> StepRef:
        """Reference to a subpath within this step's output."""
        ...
```

### Execution: Automatic Resolution

When a step executes, the executor automatically resolves all `StepRef` values to concrete strings:

```python
# The ACTUAL function that runs on the cluster receives resolved config
def tokenize(config: TokenizeConfig):
    # config.train_paths is list[str], not list[StepRef]
    # config.cache_path is str, not StepRef
    for path in config.train_paths:
        files = glob(f"{path}/*.jsonl")  # Works!
```

The executor handles resolution transparently - the function signature is the same, but it receives concrete values.

---

## Detailed Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Add StepRef and StepContext

Create new file: `lib/marin/src/marin/execution/step_ref.py`

```python
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from marin.execution.executor import ExecutorStep

@dataclass(frozen=True)
class StepRef:
    """
    A reference to a path that will be resolved at execution time.

    This is the ONLY type used for paths during step construction.
    At execution time, all StepRefs are resolved to concrete strings.
    """
    _step: ExecutorStep | None  # None = this step's output
    _subpath: str | None = None
    _blocking: bool = True

    def __truediv__(self, subpath: str) -> StepRef:
        """Navigate to subpath."""
        if self._subpath:
            new_subpath = os.path.join(self._subpath, subpath)
        else:
            new_subpath = subpath
        return replace(self, _subpath=new_subpath)

    def nonblocking(self) -> StepRef:
        """Mark as non-blocking dependency."""
        return replace(self, _blocking=False)

    @staticmethod
    def hardcoded(path: str) -> StepRef:
        """Create a reference to a hardcoded path (not part of pipeline)."""
        return StepRef(_step=None, _subpath=path, _blocking=True)

    def resolve(self, output_path: str, output_paths: dict[ExecutorStep, str]) -> str:
        """Resolve to concrete path. Called by executor at execution time."""
        if self._step is None:
            # This step's output or hardcoded path
            base = output_path if self._subpath is None or not self._subpath.startswith("/") else ""
        else:
            base = output_paths[self._step]

        if self._subpath:
            return os.path.join(base, self._subpath) if base else self._subpath
        return base


class StepContext:
    """
    Context object for defining step dependencies.

    Passed to step-defining functions. Tracks dependencies as they're declared.
    """

    def __init__(self):
        self._dependencies: list[ExecutorStep] = []
        self._pseudo_dependencies: list[ExecutorStep] = []
        self._step: ExecutorStep | None = None  # Set after step creation

    def require(self, dep: ExecutorStep | StepRef) -> StepRef:
        """Declare a blocking dependency. Returns StepRef to dep's output."""
        if isinstance(dep, StepRef):
            if dep._step is not None:
                self._dependencies.append(dep._step)
            return dep
        else:
            self._dependencies.append(dep)
            return StepRef(_step=dep)

    def require_nonblocking(self, dep: ExecutorStep | StepRef) -> StepRef:
        """Declare non-blocking dependency (for versioning only)."""
        ref = self.require(dep)
        if ref._step is not None:
            # Move from blocking to non-blocking
            if ref._step in self._dependencies:
                self._dependencies.remove(ref._step)
            self._pseudo_dependencies.append(ref._step)
        return ref.nonblocking()

    @property
    def output(self) -> StepRef:
        """Reference to this step's output path."""
        return StepRef(_step=None)

    def output_subpath(self, subpath: str) -> StepRef:
        """Reference to subpath within this step's output."""
        return StepRef(_step=None, _subpath=subpath)
```

#### 1.2 Add @step Decorator

Add to `step_ref.py`:

```python
from typing import Callable, TypeVar, Generic, ParamSpec
from dataclasses import dataclass
from functools import wraps

ConfigT = TypeVar("ConfigT")

def step(
    name: str,
    *,
    description: str | None = None,
    resources: ResourceConfig | None = None,
    pip_dependency_groups: list[str] | None = None,
):
    """
    Decorator to define an executor step.

    Usage:
        @step(name="tokenize/fineweb")
        def tokenize_fineweb(ctx: StepContext):
            download = ctx.require(download_step)
            return TokenizeConfig(
                train_paths=[download / "train"],
                cache_path=ctx.output,
            )

        # Create the step
        my_step = tokenize_fineweb()
    """
    def decorator(fn: Callable[[StepContext], ConfigT]) -> Callable[[], ExecutorStep[ConfigT]]:
        @wraps(fn)
        def make_step() -> ExecutorStep[ConfigT]:
            ctx = StepContext()
            config = fn(ctx)

            from marin.execution.executor import ExecutorStep

            step_obj = ExecutorStep(
                name=name,
                fn=None,  # Will be set by executor from config type
                config=config,
                description=description,
                resources=resources,
                pip_dependency_groups=pip_dependency_groups,
                _context=ctx,
            )
            ctx._step = step_obj
            return step_obj

        return make_step

    return decorator
```

#### 1.3 Update ExecutorStep

Modify `executor.py`:

```python
@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT]):
    name: str
    fn: ExecutorFunction | None  # None = infer from config
    config: ConfigT
    description: str | None = None
    override_output_path: str | None = None
    pip_dependency_groups: list[str] | None = None
    resources: ResourceConfig | None = None

    # NEW: Context from @step decorator (None for legacy steps)
    _context: StepContext | None = None

    def __truediv__(self, subpath: str) -> StepRef:
        """Allow step / "subpath" syntax."""
        return StepRef(_step=self, _subpath=subpath)

    @property
    def ref(self) -> StepRef:
        """Get a StepRef to this step's output."""
        return StepRef(_step=self)
```

#### 1.4 Update Resolution Logic

Modify `collect_dependencies_and_version` and `instantiate_config` to handle `StepRef`:

```python
def collect_dependencies_and_version(obj: Any) -> _Dependencies:
    """Extract dependencies and version info from config."""
    pseudo_dependencies: list[ExecutorStep] = []
    dependencies: list[ExecutorStep] = []
    version: dict[str, Any] = {}

    def recurse(obj: Any, prefix: str):
        new_prefix = prefix + "." if prefix else ""

        # NEW: Handle StepRef
        if isinstance(obj, StepRef):
            if obj._step is not None:
                index = len(dependencies) + len(pseudo_dependencies)
                if obj._blocking:
                    dependencies.append(obj._step)
                else:
                    pseudo_dependencies.append(obj._step)
                version[prefix] = f"DEP[{index}]" + (f"/{obj._subpath}" if obj._subpath else "")
            else:
                # Output reference or hardcoded path
                version[prefix] = obj._subpath

        # LEGACY: Handle InputName/ExecutorStep (for migration period)
        elif isinstance(obj, ExecutorStep):
            obj = StepRef(_step=obj)
            recurse(obj, prefix)
        elif isinstance(obj, InputName):
            ref = StepRef(obj.step, obj.name, obj.block_on_step)
            recurse(ref, prefix)

        # ... rest of existing logic for dataclasses, lists, dicts
```

### Phase 2: Helper Function Migration

The key insight: helper functions should ONLY accept `StepRef` for step references. No more unions.

#### 2.1 Pattern: Split Functions by Input Type

**Before** (isinstance checks):
```python
def default_train(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    ...
):
    if isinstance(tokenized, InputName | ExecutorStep):
        data = lm_data_config(training_set=tokenized, ...)
    else:
        data = tokenized
```

**After** (separate functions):
```python
def train_from_tokenized(
    ctx: StepContext,
    tokenized: StepRef,  # Always StepRef
    model_config: LmConfig,
    ...
) -> TrainConfig:
    """Build training config from a tokenized dataset step."""
    data = lm_data_config(training_set=tokenized, ...)
    return TrainConfig(data=data, output_path=ctx.output, ...)


def train_from_mixture(
    ctx: StepContext,
    mixture: LMMixtureDatasetConfig,  # Always concrete config
    model_config: LmConfig,
    ...
) -> TrainConfig:
    """Build training config from a pre-built mixture."""
    return TrainConfig(data=mixture, output_path=ctx.output, ...)
```

#### 2.2 Files Requiring Migration

| File | Functions to Update |
|------|---------------------|
| `experiments/defaults.py` | `default_train`, `default_tokenize`, `default_scaling_law_pred`, `_get_tokenizer_for_train`, `_pretraining_data_for_train` |
| `experiments/evals/evals.py` | `extract_model_name_and_path`, `evaluate_levanter_lm_evaluation_harness` |
| `lib/marin/src/marin/speedrun/speedrun.py` | `make_speedrun_step` |
| `lib/marin/src/marin/processing/tokenize/tokenize.py` | `_get_filepaths_to_tokenize`, `_validate_train_urls` |
| `lib/marin/src/marin/utilities/executor_utils.py` | `parse_checkpoint_step` |
| `lib/marin/src/marin/export/hf_upload.py` | HF upload config construction |
| `lib/marin/src/marin/export/levanter_checkpoint.py` | Checkpoint handling |
| `lib/marin/src/marin/scaling_laws/scaling_laws.py` | W&B run ID extraction |

#### 2.3 Migration Strategy for Each File

For each file with isinstance checks:

1. **Identify the union types**: Find all `InputName | ExecutorStep | ConcreteType` patterns
2. **Split into separate functions**: One for `StepRef`, one for concrete types
3. **Update call sites**: Use the appropriate function based on what they have
4. **Remove isinstance checks**: The type system now enforces correctness

### Phase 3: Config Class Updates

#### 3.1 Update Config Type Annotations

Configs should declare which fields are step references:

```python
@dataclass(frozen=True)
class TokenizeConfig:
    train_paths: list[StepRef]      # Was: list[str | InputName]
    validation_paths: list[StepRef]
    cache_path: StepRef             # Was: str | OutputName
    tokenizer: str
    batch_size: int = 1000
```

#### 3.2 Versioned Values

For values that should affect the version hash, we still need `Versioned[T]`:

```python
@dataclass(frozen=True)
class TokenizeConfig:
    train_paths: list[StepRef]
    cache_path: StepRef
    tokenizer: Versioned[str]  # Affects version hash
    batch_size: int = 1000     # Does NOT affect version hash
```

The `Versioned` wrapper is orthogonal to `StepRef` - it marks any value (including plain strings) as contributing to the version hash.

### Phase 4: Experiment Migration

#### 4.1 Update Experiment Files

Convert experiments from old pattern to new:

**Before**:
```python
download = ExecutorStep(
    name="download/fineweb",
    fn=download_hf,
    config=DownloadConfig(
        id="HuggingFaceFW/fineweb",
        output_path=this_output_path(),
    ),
)

tokenize = ExecutorStep(
    name="tokenize/fineweb",
    fn=tokenize_fn,
    config=TokenizeConfig(
        train_paths=[output_path_of(download, "train")],
        cache_path=this_output_path(),
        tokenizer=versioned("llama3"),
    ),
)
```

**After**:
```python
@step(name="download/fineweb")
def download_fineweb(ctx: StepContext):
    return DownloadConfig(
        id="HuggingFaceFW/fineweb",
        output_path=ctx.output,
    )

@step(name="tokenize/fineweb")
def tokenize_fineweb(ctx: StepContext):
    download = ctx.require(download_step)
    return TokenizeConfig(
        train_paths=[download / "train"],
        cache_path=ctx.output,
        tokenizer=versioned("llama3"),
    )

# Create steps
download_step = download_fineweb()
tokenize_step = tokenize_fineweb()
```

### Phase 5: Deprecation and Cleanup

#### 5.1 Deprecate Old Types

```python
# In executor.py
import warnings

class InputName:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "InputName is deprecated. Use StepRef instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing implementation for backwards compat

class OutputName:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OutputName is deprecated. Use ctx.output instead.",
            DeprecationWarning,
            stacklevel=2
        )
```

#### 5.2 Remove Old Patterns

After migration:
1. Remove `InputName`, `OutputName` classes
2. Remove `output_path_of()`, `this_output_path()` functions
3. Remove all isinstance checks
4. Update documentation

---

## API Reference

### New API

```python
from marin.execution import step, StepContext, StepRef, versioned

# Define a step
@step(name="my_step", resources=ResourceConfig(gpu_type="a100"))
def my_step(ctx: StepContext):
    # Declare dependencies
    upstream = ctx.require(other_step)
    checkpoint = ctx.require_nonblocking(training_step / "checkpoints/step-1000")

    return MyConfig(
        input_path=upstream / "data",
        checkpoint_path=checkpoint,
        output_path=ctx.output,
        learning_rate=versioned(0.001),
    )

# Create the step
step_instance = my_step()

# Use in pipeline
executor.run([step_instance])
```

### Comparison

| Old API | New API |
|---------|---------|
| `output_path_of(step, "sub")` | `ctx.require(step) / "sub"` |
| `this_output_path()` | `ctx.output` |
| `OutputName("sub")` | `ctx.output / "sub"` |
| `InputName.hardcoded("path")` | `StepRef.hardcoded("path")` |
| `input.nonblocking()` | `ctx.require_nonblocking(step)` |
| `ExecutorStep(name=..., fn=..., config=...)` | `@step(name=...) def ...(ctx): return config` |

---

## Migration Execution Plan

### Order of Operations

1. **Phase 1**: Add new infrastructure (non-breaking)
   - Add `StepRef`, `StepContext`, `@step` decorator
   - Update executor to handle both old and new patterns
   - **No existing code changes**

2. **Phase 2**: Migrate core utilities
   - `lib/marin/src/marin/execution/` - core executor logic
   - `lib/marin/src/marin/utilities/executor_utils.py`

3. **Phase 3**: Migrate processing modules
   - `lib/marin/src/marin/processing/tokenize/tokenize.py`
   - `lib/marin/src/marin/export/`

4. **Phase 4**: Migrate experiment helpers
   - `experiments/defaults.py` (most complex)
   - `experiments/evals/evals.py`
   - `lib/marin/src/marin/speedrun/speedrun.py`
   - `lib/marin/src/marin/scaling_laws/scaling_laws.py`

5. **Phase 5**: Migrate individual experiments
   - `experiments/pretraining_datasets/`
   - `experiments/tutorials/`
   - All `experiments/exp*.py` files

6. **Phase 6**: Cleanup
   - Remove deprecated types
   - Remove backwards compatibility code
   - Update documentation

### Parallelization Strategy

After Phase 1-2 complete, Phases 3-5 can be parallelized using sub-agents:
- One agent per directory/module
- Each agent handles all files in its scope
- Agents can run concurrently since changes are isolated

---

## Example: Full Migration of defaults.py

### Before (Current Code)

```python
# experiments/defaults.py

def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    use_default_validation: bool = True,
) -> ExecutorStep:
    pretraining_data = _pretraining_data_for_train(tokenized, use_default_validation)

    config = TrainLmOnPodConfig(
        output_path=this_output_path(),
        data=pretraining_data,
        model=model_config,
        trainer=train_config,
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        fn=run_levanter_train_lm,
        config=config,
        resources=train_config.resources,
    )


def _pretraining_data_for_train(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    use_default_validation: bool,
) -> LMMixtureDatasetConfig:
    tokenizer = _get_tokenizer_for_train(tokenized)
    validation_sets = default_validation_sets(tokenizer) if use_default_validation else {}

    if isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
        )
    else:
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)

    return pretraining_data
```

### After (Migrated Code)

```python
# experiments/defaults.py

def train_config_from_tokenized(
    ctx: StepContext,
    tokenized: StepRef,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    use_default_validation: bool = True,
) -> TrainLmOnPodConfig:
    """Build training config from a tokenized dataset step."""
    tokenizer = _get_tokenizer_from_step(tokenized)
    validation_sets = default_validation_sets(tokenizer) if use_default_validation else {}

    data = lm_data_config(
        training_set=tokenized,
        validation_sets=validation_sets,
    )

    return TrainLmOnPodConfig(
        output_path=ctx.output,
        data=data,
        model=model_config,
        trainer=train_config,
    )


def train_config_from_mixture(
    ctx: StepContext,
    mixture: LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    use_default_validation: bool = True,
) -> TrainLmOnPodConfig:
    """Build training config from a pre-built mixture."""
    if use_default_validation:
        validation_sets = default_validation_sets(mixture.tokenizer)
        mixture = add_validation_sets_to_mixture(mixture, validation_sets)

    return TrainLmOnPodConfig(
        output_path=ctx.output,
        data=mixture,
        model=model_config,
        trainer=train_config,
    )


def default_train_step(
    name: str,
    tokenized: StepRef,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    **kwargs,
) -> ExecutorStep:
    """Create a training step from a tokenized dataset."""

    @step(
        name=f"checkpoints/{name}",
        resources=train_config.resources,
    )
    def train(ctx: StepContext):
        tok = ctx.require(tokenized)  # Re-require to register dependency
        return train_config_from_tokenized(ctx, tok, model_config, train_config, **kwargs)

    return train()


def _get_tokenizer_from_step(tokenized: StepRef) -> str:
    """Extract tokenizer from a tokenized dataset step."""
    # StepRef always has a step (or is hardcoded)
    if tokenized._step is None:
        raise ValueError("Cannot infer tokenizer from hardcoded path")

    config = tokenized._step.config
    if isinstance(config, TokenizeConfigBase):
        return config.tokenizer
    raise ValueError(f"Cannot determine tokenizer from {tokenized._step}")
```

---

## Testing Strategy

### Unit Tests for StepRef

```python
def test_stepref_path_operations():
    step = ExecutorStep(name="test", fn=None, config=None)

    ref = StepRef(_step=step)
    assert ref._subpath is None

    ref2 = ref / "data" / "train"
    assert ref2._subpath == "data/train"
    assert ref2._step is step


def test_stepref_resolution():
    step = ExecutorStep(name="test", fn=None, config=None)
    ref = StepRef(_step=step, _subpath="data/train")

    output_paths = {step: "/output/test-abc123"}
    resolved = ref.resolve("/my/output", output_paths)

    assert resolved == "/output/test-abc123/data/train"


def test_context_dependency_tracking():
    step_a = ExecutorStep(name="a", fn=None, config=None)
    step_b = ExecutorStep(name="b", fn=None, config=None)

    ctx = StepContext()
    ref_a = ctx.require(step_a)
    ref_b = ctx.require_nonblocking(step_b)

    assert step_a in ctx._dependencies
    assert step_b in ctx._pseudo_dependencies
    assert ref_b._blocking is False
```

### Integration Tests

```python
def test_step_decorator():
    @step(name="test/step")
    def my_step(ctx: StepContext):
        return {"output": ctx.output}

    step_instance = my_step()
    assert step_instance.name == "test/step"
    assert isinstance(step_instance.config["output"], StepRef)


def test_full_pipeline():
    @step(name="download")
    def download(ctx):
        return DownloadConfig(output_path=ctx.output)

    @step(name="process")
    def process(ctx):
        dl = ctx.require(download_step)
        return ProcessConfig(input=dl / "data", output=ctx.output)

    download_step = download()
    process_step = process()

    # Verify dependency was tracked
    assert download_step in process_step._context._dependencies
```

---

## Appendix: isinstance Locations to Remove

After migration, these checks should be eliminated:

| File | Line | Check |
|------|------|-------|
| `executor.py` | 383-385 | `isinstance(run, ExecutorStep)` / `isinstance(run, InputName)` |
| `executor.py` | 458 | `isinstance(..., OutputName \| InputName \| ExecutorStep)` |
| `executor.py` | 582-587 | `isinstance(obj, ExecutorStep)` / `isinstance(obj, InputName)` |
| `executor.py` | 637-640 | `isinstance(obj, ExecutorStep)` / `isinstance(obj, InputName)` |
| `defaults.py` | 612 | `isinstance(tokenized, InputName \| ExecutorStep)` |
| `defaults.py` | 630-636 | Pattern matching on `ExecutorStep` / `InputName` |
| `tokenize.py` | 121-122 | `isinstance(self.train_paths, str \| InputName)` |
| `tokenize.py` | 172 | `isinstance(item, InputName)` |
| `tokenize.py` | 216 | `any(isinstance(x, InputName \| ExecutorStep) for x in input_paths)` |
| `evals.py` | 104-127 | Multi-branch isinstance cascade |
| `speedrun.py` | 368 | `isinstance(tokenized, InputName \| ExecutorStep)` |
| `executor_utils.py` | 75 | `isinstance(checkpoint_path, InputName)` |
| `hf_upload.py` | 83 | `isinstance(input_path, InputName \| ExecutorStep)` |
| `levanter_checkpoint.py` | 167 | `isinstance(checkpoint_path, InputName)` |
| `scaling_laws.py` | 103, 111 | `isinstance(..., ExecutorStep)` |
| `quality_classifier_experiment_utils.py` | 51 | `isinstance(model_path, ExecutorStep)` |
| `olmoe_1b7b_nemotron_40b.py` | 145 | `isinstance(tokenized, ExecutorStep \| InputName)` |
