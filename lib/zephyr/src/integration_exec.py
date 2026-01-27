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

from functools import wraps
import json
from dataclasses import dataclass
from typing import Annotated, Generic, ParamSpec, TypeVar
from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")


def _safe_get_func_name(fn: Callable[..., T]) -> str:
    return getattr(fn, "__name__", "<unknown>")


@dataclass
class StepCall(Generic[T]):
    """Represents a delayed function call"""

    _fn: Callable[..., T]
    _args: tuple
    _kwargs: dict

    def __hash__(self):
        return hash(
            (
                _safe_get_func_name(self._fn),
                json.dumps(self._args, sort_keys=True),
                json.dumps(self._kwargs, sort_keys=True),
            )
        )

    def __eq__(self, value):
        if not isinstance(value, StepCall):
            # TODO: should just fail?
            return NotImplemented
        # compare hash values
        return hash(self) == hash(value)

    def __repr__(self):
        return f"StepCall({_safe_get_func_name(self._fn)}, args={self._args}, kwargs={self._kwargs})"


@dataclass
class StepFn(Generic[P, T]):
    inner_fn: Callable[P, T]

    def __post_init__(self):
        wraps(self.inner_fn)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> StepCall[T]:
        return StepCall(self.inner_fn, args, kwargs)


def step(fn: Callable[P, T]) -> StepFn[P, T]:
    return StepFn(inner_fn=fn)


# ------------------------------------------------------------------------------------


@dataclass
class Dataset:
    path: str


@dataclass
class Deduped:
    output_path: str
    metadata_path: Annotated[str, "Path to the hash files"]


@step
def download(dataset_42: str) -> Dataset:
    "Download a dataset"
    return Dataset(path=f"gs://raw/{dataset_42}")


@step
def dedup(*datasets: Dataset, mode: str) -> Deduped:
    return Deduped(output_path="gs://deduped/blah/data", metadata_path="gs://deduped/blah/meta")


if __name__ == "__main__":
    # import code; code.interact(local=dict(globals(), **locals()))
    download_foo = download(dataset="dataset_foo")
    download_bar = download(dataset="dataset_bar")
    print(dedup(download_foo, download_bar, mode="exact_paragraph"))
