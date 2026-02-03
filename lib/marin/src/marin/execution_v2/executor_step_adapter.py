# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, fields, is_dataclass, replace
from marin.execution.executor import ExecutorStep
from marin.execution_v2.step import StepCallDeferred, safe_get_func_name

"""
Temporary adapter to convert v2 StepCallDeferred into ExecutorStep for compatibility
with existing execution infrastructure. All of this would gradually go away as we
decouple ExecutorStep.
"""


@dataclass(frozen=True)
class StepConfig:
    # NOTE: this is intentially list not a tuple to match ExecutorStep expectation
    args: list
    kwargs: dict


def deferred_step_to_executor_step(
    deferred_step: StepCallDeferred,
    _cache: dict[int, ExecutorStep] | None = None,
) -> ExecutorStep:
    """Convert a deferred StepCallDeferred into an ExecutorStep

    This is useful to bridge between the v2 Step API and the ExecutorStep API, specifically
    to leverage existing execution infrastructure with the new Step abstraction.

    The _cache parameter ensures the same StepCallDeferred always maps to the
    same ExecutorStep object, which is required for correct dependency resolution
    when multiple downstream steps reference the same upstream step.
    """
    if _cache is None:
        _cache = {}

    # TODO: better fingerprinting
    step_id = id(deferred_step)
    if step_id in _cache:
        return _cache[step_id]

    def recurse(obj):
        if isinstance(obj, StepCallDeferred):
            return deferred_step_to_executor_step(obj, _cache)

        if isinstance(obj, (list, tuple, set)):
            return type(obj)(recurse(item) for item in obj)
        if isinstance(obj, dict):
            return {recurse(k): recurse(v) for k, v in obj.items()}
        if is_dataclass(obj) and not isinstance(obj, type):
            changes = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                new_value = recurse(value)
                if new_value is not value:
                    changes[f.name] = new_value
            if changes:
                return replace(obj, **changes)
            return obj

        return obj

    fn = deferred_step._fn
    result = ExecutorStep(
        name=safe_get_func_name(deferred_step._fn),
        fn=lambda config: fn(*config.args, **config.kwargs),
        config=StepConfig(
            # NOTE: afair ExecutorStep doesn't recurse inside tuple, we make
            # it a list
            args=recurse(list(deferred_step._args)),
            kwargs=recurse(deferred_step._kwargs),
        ),
        description=deferred_step._fn.__doc__,
    )
    _cache[step_id] = result
    return result


def deferred_steps_to_executor_steps(
    *steps: StepCallDeferred,
    _cache: dict[int, ExecutorStep] | None = None,
) -> list[ExecutorStep]:
    """Convert a list of deferred steps, sharing a single cache for consistent identity."""
    if _cache is None:
        _cache = {}
    return [deferred_step_to_executor_step(s, _cache) for s in steps]
