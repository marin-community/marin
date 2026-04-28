# Spec: executor inside the training job

Concrete contracts the design implies. Read alongside `design.md`.

## New public API

### `marin.execution.upstream_steps`

Where it lives: `lib/marin/src/marin/execution/dag.py` (new module — keeps `executor.py` from growing further).

```python
from typing import Any
from marin.execution.executor import ExecutorStep

def upstream_steps(obj: Any) -> list[ExecutorStep]:
    """Recursively walk obj and return every ExecutorStep referenced from it.

    Walks dataclasses (via `dataclasses.fields`), dicts (values), lists/tuples/sets
    (elements), and `ExecutorStep` instances themselves. The same step appearing
    multiple times in the object graph is returned exactly once. Order is
    deterministic (depth-first, fields/keys/elements in declaration order).

    Does NOT walk into the returned steps' configs — it returns the steps the
    caller's `obj` references directly. Transitive dependencies are discovered
    by `Executor.run()` itself (which already walks step configs to build
    its dependency graph).

    Args:
        obj: Any object — typically a config dataclass like
            `GrugBaseLaunchConfig`, but accepts any value.

    Returns:
        Deterministically ordered list of unique ExecutorStep instances.
    """
```

The implementation is a port of the private traversal already used in `Executor._build_dependency_graph` (`lib/marin/src/marin/execution/executor.py:418, 462`). Extract that traversal into `dag.py` and have `Executor` call into the public function.

## Entrypoint contract

Pipeline files' `__main__` block:

```python
if __name__ == "__main__":
    training_step.fn(training_step.config)
```

`training_step.fn` is a `RemoteCallable` (`lib/marin/src/marin/execution/remote.py:44`); calling it submits a top-level Iris job via `IrisClient.submit` and blocks until the job terminates (success, failure, or cancellation). Exit code of the entrypoint reflects the inner job's terminal status — same observable behaviour as today's `executor_main(steps=[training_step])`.

No new helper function is introduced for the entrypoint. `executor_main` is not deleted in this PR; it remains in place for callers that haven't migrated.

## Training-launcher contract

Each `@remote`-wrapped training launcher (e.g. `run_grug_base_trial` at `experiments/grug/base/launch.py`, `train_lm` at `lib/marin/src/marin/training/...`) gains exactly one new line:

```python
def run_grug_base_trial(config: GrugBaseLaunchConfig):
    Executor().run(upstream_steps(config))   # new — materialise deps
    train(config)                             # existing body
```

Behaviour:
- `Executor().run(steps)` walks each step's transitive deps, submits any that aren't `STATUS_SUCCESS`, blocks on completion (`step_runner.py:341` distributed lock + heartbeat handles cross-task / cross-process coordination).
- If `upstream_steps(config)` returns `[]`, `Executor().run([])` is a no-op and training proceeds immediately.
- Exceptions propagate; if dependency materialisation fails, training is not attempted.

No new decorator, no new wrapper class — the launcher composes the two existing pieces explicitly.

## Iris contract change

**File**: `lib/iris/src/iris/cluster/worker/task_attempt.py`

**Removed**: line 696, the `env["IRIS_WORKER_REGION"] = region_attr.string_value` injection into child task environments.

**Result**: a parent task in region R submitting a child job with no explicit `regions=[...]` produces a region-unconstrained child. Region pinning becomes opt-in via:
- Explicit `regions=[...]` on `ResourceConfig` at the call site, OR
- The executor's path-based inference (`lib/marin/src/marin/execution/executor.py:561-631`) firing because step output paths land under a region-specific prefix.

**Retained**: `JobInfo.worker_region` (`lib/iris/src/iris/cluster/client/job_info.py:56, 122`) remains populated from the worker's actual region, for callers that explicitly want to know where they're running. Only the *automatic propagation into child constraints* goes away.

## File paths

| Piece | Path |
|---|---|
| `upstream_steps` (new) | `lib/marin/src/marin/execution/dag.py` |
| `Executor` (modified to call into `dag.py`) | `lib/marin/src/marin/execution/executor.py` |
| Iris inheritance removal | `lib/iris/src/iris/cluster/worker/task_attempt.py:696` |
| Reference pipeline migration (proof-of-concept) | `experiments/references/reference_training_pipeline.py` |
| Training launcher wrapper (one line per launcher) | `experiments/grug/base/launch.py`, `lib/marin/src/marin/training/...` |

## Worked example: rewritten `reference_training_pipeline.py`

The full migrated pipeline. The only changes from the current file (`experiments/references/reference_training_pipeline.py`) are: the import of `executor_main` is dropped, and the `__main__` block calls the training step's `@remote` callable directly. Model config, schedule, data construction, mixture, and `training_step` definition are byte-identical to the current file.

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Single-run pretraining → midtraining → SFT pipeline.

Demonstrates that pretrain/midtrain/SFT are all just data mixing phases.
The entire pipeline is one training run with time-varying mixture weights:

  1. Pretrain (steps 0-40k): DCLM baseline
  2. Midtrain (steps 40k-50k): Blend DCLM + Dolmino math
  3. SFT (steps 50k-52k): SmolTalk instruction data
"""

import dataclasses

from levanter.data.text import ChatLmDatasetFormat
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from experiments.defaults import default_tokenize, default_validation_sets
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.dolmino import tokenize_dolmino
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, this_output_path   # <-- executor_main removed
from marin.execution.remote import remote
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# --- Model: 600M Grug ---
model = GrugModelConfig(
    vocab_size=128_256,
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

# --- Schedule ---
PRETRAIN_STEPS = 40_000
MIDTRAIN_STEPS = 10_000
SFT_STEPS = 2_000
TOTAL_STEPS = PRETRAIN_STEPS + MIDTRAIN_STEPS + SFT_STEPS

# --- Data components (unchanged: these still produce ExecutorSteps) ---
pretrain = {"dclm": dclm_components_llama3["dclm_baseline"]}

dolmino = tokenize_dolmino()
midtrain = {"dolmino_math": dolmino["dolmino/math/metamath-owmfilter"]}

smoltalk = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["train"])
sft = {
    "smoltalk": default_tokenize(
        name="smoltalk_marin",
        dataset=smoltalk / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )
}

# --- Time-varying mixture weights (unchanged) ---
data = lm_varying_mixture_data_config(
    components={**pretrain, **midtrain, **sft},
    weights_list=[
        (0, {"dclm": 1.0, "dolmino_math": 0.0, "smoltalk": 0.0}),
        (PRETRAIN_STEPS, {"dclm": 0.7, "dolmino_math": 0.3, "smoltalk": 0.0}),
        (PRETRAIN_STEPS + MIDTRAIN_STEPS, {"dclm": 0.0, "dolmino_math": 0.0, "smoltalk": 1.0}),
    ],
)
data = dataclasses.replace(data, tokenizer=marin_tokenizer)
data = add_validation_sets_to_mixture(data, default_validation_sets(tokenizer=data.tokenizer))

# --- Training step (unchanged: still an ExecutorStep with embedded upstream deps) ---
training_step = ExecutorStep(
    name="reference-pipeline",
    fn=remote(run_grug_base_trial, resources=ResourceConfig.with_tpu("v4-8")),
    config=GrugBaseLaunchConfig(
        model=model,
        data=data,
        output_path=this_output_path(),
        run_id="reference-pipeline",
        resources=ResourceConfig.with_tpu("v4-8"),
        steps=TOTAL_STEPS,
        batch_size=256,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            tags=["reference", "pipeline"],
            group="reference-pipeline",
            name=None,
        ),
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.1,
            warmup=0.05,
            decay=0.2,
        ),
        eval=GrugEvalConfig(
            steps_per_eval=500,
        ),
    ),
)

if __name__ == "__main__":
    # @remote → IrisClient.submit, blocks until the training job terminates.
    # No DAG walk in this entrypoint; the training job materialises its own
    # tokenize deps once it lands on a TPU in a specific region.
    training_step.fn(training_step.config)
```

### And the matching change in `experiments/grug/base/launch.py`

```python
# Before
def run_grug_base_trial(config: GrugBaseLaunchConfig) -> None:
    train(config)

# After
def run_grug_base_trial(config: GrugBaseLaunchConfig) -> None:
    Executor().run(upstream_steps(config))   # materialise tokenize deps
    train(config)
```

That is the entire training-side change. `Executor().run([])` is a no-op; launchers whose configs reference no `ExecutorStep`s gain the line for free with no behaviour change.

## Errors

No new error types. Existing executor errors (`PreviousTaskFailedError` at `lib/marin/src/marin/execution/executor_step_status.py:50`, fsspec errors, Iris job-failure exceptions) propagate up through the training launcher and into the entrypoint as today.

One behavioural change worth calling out: cross-region GCS dependencies inside a single step still raise (`executor.py:244-246`) — that's pre-existing and unchanged. What changes is that the *transitive* graph is no longer pinned to one region by Iris-side inheritance, so a training job in `us-east5` can run with deps that live in `us-east5` (re-tokenized locally) rather than failing because some upstream produced data in `us-central1`.

## Persisted shapes

No new persisted formats. Existing executor status files (`{output_path}/.executor_status`, `executor_step_status.py:42`) and step output directories continue to be the source of truth. The new pattern relies on the existing distributed lock + status protocol — it doesn't add markers, sentinels, or sidecar files.

The only new on-disk consequence is duplication: if the same upstream step runs in `us-east5` and `us-central1`, two copies of its output exist (one per regional `marin-{region}` bucket). Step IDs are identical; only the prefix differs.

## Out of scope

- `executor_main` deletion (deferred to the final cleanup PR after all 89 callers migrate).
- Reservation removal (#4631 — parallel work).
- A non-blocking entrypoint variant (deferred until a real caller wants it).
- A sweep-concurrency helper (open question in `design.md`; deferred to first sweep migration).
- Eval-only / dataset-only pipelines (open question — may need a separate analogous wrapper).
