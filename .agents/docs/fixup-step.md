# Executor Step Migration Guide

This document describes the new JAX-style tracing design for the Marin executor framework and provides instructions for migrating code from the old pattern.

## Design Summary

The new design separates code into three distinct layers:

### Layer 1: Library Code (`lib/marin/`)

**Purpose**: Pure functions that perform actual work (download, tokenize, train, etc.)

**Rules**:
- Takes concrete types only: `str`, `int`, dataclasses with concrete fields
- Returns concrete values
- **NO** imports from `marin.execution` (no `ExecutorStep`, `StepRef`, `InputName`, `output()`, etc.)
- **NO** default values like `THIS_OUTPUT_PATH` or `this_output_path()`
- If type hints need to reference executor types for compatibility, use `TYPE_CHECKING` guard

**Example - CORRECT**:
```python
# lib/marin/src/marin/download/huggingface/download_hf.py

@dataclass(frozen=True)
class DownloadConfig:
    hf_dataset_id: str
    revision: str
    gcs_output_path: str  # Required, no default!
    wait_for_completion: bool = True

def download_hf(config: DownloadConfig) -> None:
    """Download a HuggingFace dataset to the specified path."""
    # ... implementation using concrete paths ...
```

### Layer 2: Wiring Code (`experiments/steps/`)

**Purpose**: Bridges library code and executor framework. Exports `@deferred` markers and `@step` functions.

**Rules**:
- Import library functions and wrap with `deferred()`
- Create `@step` decorated functions that call deferred functions
- Use `output()` to reference the current step's output path
- Accept `StepRef` parameters for dependencies on other steps
- Return `StepRef` (implicitly, via `@step` decorator)

**Example - CORRECT**:
```python
# experiments/steps/tokenize.py

from marin.execution import StepRef, deferred, output, step
from marin.processing.tokenize.tokenize import TokenizeConfig
from marin.processing.tokenize.tokenize import tokenize as _tokenize

# Mark library function as deferred
tokenize = deferred(_tokenize)

@step(name="{name}")
def tokenize_dataset(
    name: str,
    input_dataset: StepRef,
    tokenizer: str,
) -> StepRef:
    """Tokenize a dataset from another step's output."""
    return tokenize(TokenizeConfig(
        train_paths=[input_dataset],
        cache_path=output(),
        tokenizer=tokenizer,
    ))
```

### Layer 3: User Code (`experiments/*.py`)

**Purpose**: Define pipelines by composing steps. Single top-level `@step` entry point.

**Rules**:
- Import step functions from wiring code (`experiments/steps/`) or inline definitions
- Define a single `@step` entry point that calls sub-steps
- Tracer automatically discovers dependencies - no need to manually list them
- Pass entry point to `executor_main()`

**Example - CORRECT**:
```python
# experiments/speedrun/llama_75m/llama_75m.py

from marin.execution import executor_main, step
from marin.speedrun.speedrun import default_speedrun

@step(name="speedrun/llama_75m/all")
def run_llama_75m():
    """Entry point for Llama 75M training."""
    default_speedrun("llama_75m", speedrun_config)

if __name__ == "__main__":
    executor_main(steps=[run_llama_75m()], description="75M Llama model")
```

---

## Key Concepts

### `@deferred` Decorator

Wraps a library function so it returns a `DeferredCall` during tracing instead of executing:

```python
from marin.execution import deferred
from marin.some_library import do_work as _do_work

do_work = deferred(_do_work)

# During tracing: returns DeferredCall, doesn't execute
# During execution: calls the actual function
```

### `@step` Decorator

Creates an executor step with automatic dependency tracking:

```python
from marin.execution import step, output

@step(name="my_step/{name}")
def my_step(name: str, input_data: StepRef):
    return some_deferred_fn(Config(
        input_path=input_data,      # StepRef - dependency tracked automatically
        output_path=output(),       # This step's output path
    ))
```

### `output()` Function

Returns a `StepRef` to the current step's output path. Must be called inside a `@step` function.

### `StepRef` Type

A reference to a step's output path. Supports:
- Path navigation: `step_ref / "subdir" / "file.txt"`
- Non-blocking deps: `step_ref.nonblocking()` (for checkpoints from running training)
- Override path: `step_ref.with_output_path("custom/path")`

---

## Anti-Patterns to Fix

### Anti-Pattern 1: `THIS_OUTPUT_PATH` / `this_output_path()` in Library Code

**BAD** - Library code using executor constructs:
```python
# lib/marin/src/marin/download/download_hf.py
from marin.execution.executor import THIS_OUTPUT_PATH

@dataclass
class DownloadConfig:
    gcs_output_path: str = THIS_OUTPUT_PATH  # BAD!
```

**GOOD** - Library code with required parameter:
```python
@dataclass
class DownloadConfig:
    gcs_output_path: str  # Required, no default
```

**Files to fix**:
- `lib/marin/src/marin/download/huggingface/download_hf.py`
- `lib/marin/src/marin/download/nemotron_cc/download_nemotron_cc.py`
- `lib/marin/src/marin/processing/classification/decon.py`
- `lib/marin/src/marin/processing/classification/deduplication/pipeline.py`
- `lib/marin/src/marin/transform/conversation/conversation_to_dolma.py`

### Anti-Pattern 2: `ExecutorStep` / `InputName` Imports in Library Code

**BAD** - Library code importing executor types:
```python
# lib/marin/src/marin/export/levanter_checkpoint.py
from marin.execution.executor import ExecutorStep, InputName, this_output_path

@dataclass
class ConvertCheckpointStepConfig:
    checkpoint_path: str | InputName  # BAD - library shouldn't know InputName
    output_path: str = dataclasses.field(default_factory=this_output_path)  # BAD
```

**GOOD** - Library code with concrete types only:
```python
@dataclass
class ConvertCheckpointStepConfig:
    checkpoint_path: str
    output_path: str  # Required
```

**Files to fix**: Any file using `| "StepRef"` or `| "ExecutorStep"` in type hints must import `StepRef` from `marin.execution`.
These should _only_ be experiments or wiring code, _not_ library code.

### Anti-Pattern 4: Explicit Step Lists (`EVAL_DATASET_STEPS`)

**BAD** - Manually listing all steps:
```python
EVAL_DATASET_STEPS: list[ExecutorStep] = [
    gsm8k_convert_dolma(),
    math_convert_dolma(),
    # ... 15 more explicit calls
]

@step(name="all")
def run_all():
    # Explicitly call all raw steps
    gsm8k_raw()
    math_raw()
    # ...
    # Then explicitly call all convert steps
    gsm8k_convert_dolma()
    math_convert_dolma()
    # ...
```

**GOOD** - Let tracer discover dependencies:
```python
@step(name="all")
def run_all():
    # Just call the leaf steps - tracer finds dependencies automatically
    gsm8k_convert_dolma()    # This internally calls gsm8k_raw()
    math_convert_dolma()     # This internally calls math_raw()
    # ...
```

### Anti-Pattern 5: `ExecutorStep` Constructor in User Code

**BAD** - Old explicit step construction:
```python
my_step = ExecutorStep(
    name="my_step",
    fn=some_function,
    config=SomeConfig(
        input_path=other_step,
        output_path=this_output_path(),
    ),
)
```

**GOOD** - New `@step` decorator pattern:
```python
some_function = deferred(_some_function)

@step(name="my_step")
def my_step(input_step: StepRef):
    return some_function(SomeConfig(
        input_path=input_step,
        output_path=output(),
    ))
```

### Anti-Pattern 6: Importing from `marin.execution.executor` Instead of `marin.execution`

**BAD** - Importing from submodule:
```python
from marin.execution.executor import step, deferred, output, versioned
```

**GOOD** - Import from package:
```python
from marin.execution import step, deferred, output, versioned
```

---

## Migration Checklist

For each file being migrated:

### Library Files (`lib/marin/`)

1. [ ] Remove imports of `ExecutorStep`, `InputName`, `StepRef`, `this_output_path`, `THIS_OUTPUT_PATH`
2. [ ] Change any `output_path: str = THIS_OUTPUT_PATH` to `output_path: str` (required)
3. [ ] Reorder dataclass fields so required fields come before optional fields
4. [ ] If type hints need executor types for compatibility, use `TYPE_CHECKING` guard
5. [ ] Add `from __future__ import annotations` if using forward reference unions

### Wiring Files (`experiments/steps/`)

1. [ ] Import library function with underscore prefix: `from marin.lib import fn as _fn`
2. [ ] Wrap with deferred: `fn = deferred(_fn)`
3. [ ] Create `@step` decorated functions that call deferred functions
4. [ ] Use `output()` for current step's output path
5. [ ] Accept `StepRef` for dependencies on other steps

### User Files (`experiments/*.py`)

1. [ ] Import from `marin.execution` not `marin.execution.executor`
2. [ ] Replace `ExecutorStep(...)` constructors with `@step` decorated functions
3. [ ] Replace `this_output_path()` with `output()`
4. [ ] Create single top-level `@step` entry point
5. [ ] Remove explicit step lists - let tracer discover dependencies
6. [ ] Pass entry point to `executor_main(steps=[entry_point()])`

---

## Testing Migration

After migrating a file:

```bash
# Check syntax
uv run python -c "import ast; ast.parse(open('path/to/file.py').read())"

# Run dry-run to verify step tracing
uv run python experiments/path/to/experiment.py --dry_run
```

The dry-run should show all discovered steps and their dependencies without actually executing anything.

---

## Files Requiring Migration

### Completed ✓

**Library Files** (removed `THIS_OUTPUT_PATH`):
- `lib/marin/src/marin/download/nemotron_cc/download_nemotron_cc.py`
- `lib/marin/src/marin/processing/classification/decon.py`
- `lib/marin/src/marin/processing/classification/deduplication/pipeline.py`
- `lib/marin/src/marin/transform/conversation/conversation_to_dolma.py`

**Wiring Files** (added `output()` parameter):
- `experiments/posttrain/instruction_datasets.py`
- `experiments/posttrain/preference_datasets.py`
- `experiments/train_test_overlap/train_test_total.py`
- `experiments/train_test_overlap/train_test_proofpile.py`

**Core Files** (converted to `@step` pattern):
- `experiments/defaults.py` - `default_train` now `@step` decorated

### Pending Migration

**ferries/** - wrap top-level `default_train()` in `@step`:
- `experiments/ferries/ferry_10_31_muonh_feistel.py`
- `experiments/ferries/initial_ferry.py`

**tutorials/** - wrap top-level `default_train()` in `@step`:
- `experiments/tutorials/train_tiny_model_cpu.py`
- `experiments/tutorials/train_tiny_model_gpu.py`
- `experiments/tutorials/train_tiny_model_tpu.py`

**tootsie/** - wrap top-level `default_train()`/`default_tokenize()` in `@step`:
- `experiments/tootsie/exp1063_upload_tootsie.py`
- `experiments/tootsie/exp1246_1b_code_science_base.py`
- `experiments/tootsie/exp1246_upload_datasets.py` (partial)
- `experiments/tootsie/exp1295_32b.py`
- `experiments/tootsie/exp1329_improved_7b.py`
- `experiments/tootsie/exp1380_muon32b.py` ✓
- `experiments/tootsie/exp1384_1b_tuning.py`
- `experiments/tootsie/exp859_baseline_8b.py`
- `experiments/tootsie/exp952_baseline_1b.py`
- `experiments/tootsie/exp1441_8b_continued.py`

**speedrun/** - wrap top-level `default_speedrun()` in `@step`:
- `experiments/speedrun/gemma3/gemma3_1b.py`
- `experiments/speedrun/gemma3/gemma3_4b.py`
- `experiments/speedrun/llama/llama_125m.py`
- `experiments/speedrun/llama/llama_350m.py`
- `experiments/speedrun/llama/llama_760m.py`
- `experiments/speedrun/llama/llama_1b.py`

**dclm/** - wrap top-level `default_train()` in `@step`:
- `experiments/dclm/exp433_dclm_run.py`

**Top-level experiments** - wrap top-level step calls in `@step`:
- `experiments/exp72_baselines.py`
- `experiments/exp1775_nanochat_three_stage.py`
- `experiments/exp1994_32b_sft.py` ✓
- `experiments/exp1858_nanochat_v2_8b.py`
- `experiments/exp1985_dpo.py`
- `experiments/exp2029_nemotron_lm_v3.py`

### Migration Pattern

For each file, the change is:

**Before** (anti-pattern):
```python
my_model = default_train(name="foo", ...)  # Top-level call at import time

if __name__ == "__main__":
    executor_main(steps=[my_model])
```

**After** (correct):
```python
@step(name="experiment/foo/all")
def run_foo():
    """Entry point for foo experiment."""
    default_train(name="foo", ...)

if __name__ == "__main__":
    executor_main(steps=[run_foo()], description="Foo experiment")
```
