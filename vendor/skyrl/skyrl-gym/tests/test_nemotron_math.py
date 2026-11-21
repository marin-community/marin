# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrl_gym.envs.nemotron_ultra.math_with_judge import symbolic_math_reward


# SkyRL's lock resolves antlr 4.9.3, under which latex2sympy2_extended 1.11.0 from PyPI raises on every
# percent number, so the library reward rejected a correct decimal and left the answer to the judge.
# vendor/skyrl/pyproject.toml overrides it with a fixed build.
@pytest.mark.parametrize(("generated", "reward"), [(r"\boxed{0.2}", 1.0), (r"\boxed{0.25}", 0.0)])
def test_symbolic_reward_reads_percent_references(generated, reward):
    assert symbolic_math_reward(r"20\%", generated)[0] == reward
