# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.utilities.executor_utils import ckpt_path_to_step_name


def test_ckpt_path_with_valid_string_path():
    path = "checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000"
    assert ckpt_path_to_step_name(path) == "llama-8b-tootsie-phase2-730000"


def test_ckpt_path_with_trailing_slash():
    path = "checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/"
    assert ckpt_path_to_step_name(path) == "llama-8b-tootsie-phase2-730000"


def test_ckpt_path_invalid_string_path():
    path = "checkpoints/llama/checkpoints/not-a-step-name"
    with pytest.raises(ValueError, match="Invalid path"):
        ckpt_path_to_step_name(path)


def test_ckpt_path_with_hf_path():
    path = "meta-llama/Meta-Llama-3.1-8B"
    assert ckpt_path_to_step_name(path) == "Meta-Llama-3.1-8B"
