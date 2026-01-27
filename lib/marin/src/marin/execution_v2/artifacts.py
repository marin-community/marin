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

from typing import Generic, TypeVar
import fsspec
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Artifact(Generic[T]):
    __artifact_file_name = ".artifact"

    @classmethod
    def from_path(cls, artifact_type: type[BaseModel], base_path: str) -> T:
        """Loads an Artifact instance from the specified output base path"""
        with fsspec.open(f"{base_path}/{cls.__artifact_file_name}", "rb") as fd:
            return artifact_type.model_validate_json(fd.read())

    @classmethod
    def to_path(cls, artifact: "Artifact") -> None:
        """Saves an Artifact instance to the specified output base path"""
        with fsspec.open(f"{artifact.output_path}/{cls.__artifact_file_name}", "wb") as fd:
            fd.write(artifact.model_dump_json().encode("utf-8"))
