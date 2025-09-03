#!/usr/bin/env python3
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

from marin.run.ray_run import parse_pip_requirements, tpus_per_node


def test_parse_pip_requirements():
    """
    Test that parse_pip_requirements properly handles extras and pinned packages.
    """
    assert parse_pip_requirements("numpy,scipy,sympy") == ["numpy", "scipy", "sympy"]
    assert parse_pip_requirements("numpy,scipy[extras1,extras2],sympy") == ["numpy", "scipy[extras1,extras2]", "sympy"]
    assert parse_pip_requirements("numpy,scipy==1.8.0,sympy") == ["numpy", "scipy==1.8.0", "sympy"]
    assert parse_pip_requirements("numpy==2.0.0,scipy[extras1,extras2],sympy") == [
        "numpy==2.0.0",
        "scipy[extras1,extras2]",
        "sympy",
    ]
    assert parse_pip_requirements(
        "numpy==2.0.0,datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@tqdm_loggable,scipy"
    ) == [
        "numpy==2.0.0",
        "datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@tqdm_loggable",
        "scipy",
    ]


def test_tpus_per_node():
    assert tpus_per_node("v4-8") == 4
    assert tpus_per_node("v5p-8") == 4
    assert tpus_per_node("v5e-4") == 4
    assert tpus_per_node("v5e-2") == 2
    with pytest.raises(ValueError):
        tpus_per_node("v5e-16")
