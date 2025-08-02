"""Simple dynamic executor for sequential data processing.

This executor allows building pipelines where dependencies are
registered dynamically during execution rather than up front.
Use :func:`use_step` to mark that a previous step is being read and
:func:`prepare_step` to create a new step whose dependencies are all
currently active ``use_step`` handles.

Example
-------
::

    with DynamicExecutor(prefix="/tmp"):
        with use_step("input") as inp, prepare_step("output") as out:
            process_files(out.path, inp.path)

``output`` will record that it depends on ``input``.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["DynamicExecutor", "use_step", "prepare_step", "DynamicStep"]

_current_executor: Optional["DynamicExecutor"] = None

@dataclass
class DynamicStep:
    name: str
    path: str
    version: Dict
    dependencies: List["DynamicStep"] = field(default_factory=list)

class _UseHandle:
    def __init__(self, executor: "DynamicExecutor", step: DynamicStep):
        self.executor = executor
        self.step = step
        self.active = False
        self._activate()

    def _activate(self):
        if not self.active:
            self.executor._active_inputs.append(self.step)
            self.active = True

    @property
    def path(self) -> str:
        return self.step.path

    def close(self):
        if self.active:
            self.executor._active_inputs.remove(self.step)
            self.active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

class _PrepareHandle:
    def __init__(self, executor: "DynamicExecutor", step: DynamicStep):
        self.executor = executor
        self.step = step
        self.finished = False

    @property
    def path(self) -> str:
        os.makedirs(self.step.path, exist_ok=True)
        return self.step.path

    def finish(self):
        self.finished = True

    def fail(self):
        self.finished = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        else:
            self.fail()

class DynamicExecutor:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.steps: Dict[str, DynamicStep] = {}
        self._active_inputs: List[DynamicStep] = []
        self._prev: Optional["DynamicExecutor"] = None

    def __enter__(self):
        global _current_executor
        self._prev = _current_executor
        _current_executor = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _current_executor
        _current_executor = self._prev

    def _compute_step(self, name: str, deps: List[DynamicStep]) -> DynamicStep:
        config = {f"inputs.[{i}]": f"DEP[{i}]" for i in range(len(deps))}
        version = {
            "name": name,
            "config": config,
            "dependencies": [d.version for d in deps],
        }
        version_str = json.dumps(version, sort_keys=True)
        hashed = hashlib.md5(version_str.encode()).hexdigest()[:6]
        path = os.path.join(self.prefix, f"{name}-{hashed}")
        return DynamicStep(name=name, path=path, version=version, dependencies=deps)

    def use_step(self, name: str) -> _UseHandle:
        step = self.steps.get(name)
        if step is None:
            raise KeyError(f"Unknown step {name}")
        return _UseHandle(self, step)

    def prepare_step(self, name: str) -> _PrepareHandle:
        deps = list(self._active_inputs)
        step = self._compute_step(name, deps)
        self.steps[name] = step
        return _PrepareHandle(self, step)

# Module level helpers -----------------------------------------------------

def use_step(name: str) -> _UseHandle:
    if _current_executor is None:
        raise RuntimeError("use_step() called outside of DynamicExecutor context")
    return _current_executor.use_step(name)

def prepare_step(name: str) -> _PrepareHandle:
    if _current_executor is None:
        raise RuntimeError("prepare_step() called outside of DynamicExecutor context")
    return _current_executor.prepare_step(name)

