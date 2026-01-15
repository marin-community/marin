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

# Test configuration for fluster

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use-docker",
        action="store_true",
        default=False,
        help="Use real Docker containers instead of in-process mocks",
    )


@pytest.fixture(scope="session")
def use_docker(request):
    return request.config.getoption("--use-docker")
