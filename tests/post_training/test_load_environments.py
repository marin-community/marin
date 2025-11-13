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

import pytest

try:
    from marin.post_training.environments.load_environments import load_environment_from_spec
    from marin.post_training.environments.olym_math_env import OlymMathEnv
except ImportError:
    pytest.skip("Post training dependencies not available in CI.", allow_module_level=True)


def test_load_environment_from_spec():
    env = load_environment_from_spec("olym_math:difficulty=hard", tokenizer=None)
    assert isinstance(env, OlymMathEnv), "Loaded environment should be an instance of OlymMathEnv"
