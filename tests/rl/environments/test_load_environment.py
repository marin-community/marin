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

"""Tests for environment loading from EnvConfig."""

from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.environments.mock_env import MockEnv


def test_load_mock_environment():
    """Test loading MockEnv via EnvConfig."""
    config = EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats", "seed": 42})

    env = load_environment_from_spec(config)

    assert isinstance(env, MockEnv)
    assert env.task_type == "cats"
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0
