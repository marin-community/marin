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

import dataclasses

from draccus.utils import DataclassInstance


def shallow_asdict(obj: DataclassInstance) -> dict:
    """
    Similar to dataclasses.asdict, but doesn't recurse into nested dataclasses.
    """
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def asdict_without_nones(obj: DataclassInstance) -> dict:
    """Convert dataclass to dictionary, omitting None values."""
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return dataclasses.asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
