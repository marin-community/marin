# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from pathlib import Path
from marin.execution.step_runner import StepRunner

from dataclasses import dataclass
from typing import Any

from marin.execution.step import Step, StepCallDeferred, resolve_deferred
import pytest


@dataclass
class Dataset:
    """Generic dataset representation"""

    path: str


@dataclass
class Deduped:
    """Deduplicated dataset representation - with deduplication metadata"""

    output_path: str
    stats: dict[str, Any]
    metadata_path: str


def download(dataset: str) -> Dataset:
    "Download a dataset"
    return Dataset(path=f"gs://raw/{dataset}")


def dedup(*datasets: Dataset, mode: str) -> Deduped:
    return Deduped(output_path="gs://deduped/blah/data", metadata_path="gs://deduped/blah/meta", stats={})


def test_step_defer():
    download_foo = Step(download).defer("foo_dataset")
    with pytest.raises(TypeError, match="is deferred, can't access the attribute 'path'"):
        download_foo.path  # noqa: B018

    deduped = Step(dedup).defer(download_foo, mode="exact_paragraph")

    assert isinstance(download_foo, StepCallDeferred)
    assert isinstance(deduped, StepCallDeferred)


def test_step_wrapping_metadata_is_preserved():
    deferred_download = Step(download)

    assert deferred_download.__name__ == download.__name__  # type: ignore[unresolved-attribute]
    assert deferred_download.__doc__ == download.__doc__


def test_deferred_step_to_executor_step(tmp_path: Path):
    ds = Step(download).defer("my_dataset")
    ds = Step(dedup).defer(ds, mode="exact_paragraph")
    executor_step = resolve_deferred(ds, prefix=tmp_path.as_posix())

    StepRunner().run(
        steps=executor_step,
    )
