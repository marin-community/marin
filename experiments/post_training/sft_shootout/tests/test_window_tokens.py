# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from experiments.post_training.sft_shootout import harbor_glm

SERVED = {harbor_glm.MAX_INPUT_TOKENS_KEY: 73_728, harbor_glm.MAX_OUTPUT_TOKENS_KEY: 8_192}


def window(window_tokens, served=SERVED):
    args = argparse.Namespace(family="swesmith", agent="mini-swe-agent", window_tokens=window_tokens)
    return harbor_glm.policy_agent(args, served)["kwargs"][harbor_glm.MODEL_INFO_KEY][harbor_glm.MAX_INPUT_TOKENS_KEY]


def test_a_window_above_the_student_serve_is_refused():
    with pytest.raises(ValueError, match="max_model_len"):
        window(80_000)
