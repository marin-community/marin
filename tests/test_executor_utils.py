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

from types import SimpleNamespace

import pytest

from marin.execution.executor import StepRef
from marin.utilities.executor_utils import ckpt_path_to_step_name


def make_input_name(name, step_name) -> StepRef:
    return StepRef(_subpath=name, _step=SimpleNamespace(name=step_name))  # type: ignore


def test_ckpt_path_with_valid_string_path():
    path = "checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000"
    assert ckpt_path_to_step_name(path) == "llama-8b-tootsie-phase2-730000"


def test_ckpt_path_with_trailing_slash():
    path = "checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/"
    assert ckpt_path_to_step_name(path) == "llama-8b-tootsie-phase2-730000"


def test_ckpt_path_with_input_name():
    input_name = make_input_name("checkpoints/step-555000", "myrun")
    assert ckpt_path_to_step_name(input_name) == "myrun-555000"


def test_ckpt_path_with_input_name_trailing_slash():
    input_name = make_input_name("checkpoints/step-555000/", "myrun")
    assert ckpt_path_to_step_name(input_name) == "myrun-555000"


def test_ckpt_path_invalid_string_path():
    path = "checkpoints/llama/checkpoints/not-a-step-name"
    with pytest.raises(ValueError, match="Invalid path"):
        ckpt_path_to_step_name(path)


def test_ckpt_path_invalid_input_type():
    with pytest.raises(ValueError, match="Unknown type"):
        ckpt_path_to_step_name(12345)


def test_ckpt_path_with_hf_path():
    path = "meta-llama/Meta-Llama-3.1-8B"
    assert ckpt_path_to_step_name(path) == "Meta-Llama-3.1-8B"
