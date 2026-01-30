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
from fray.v2.client import set_current_client
from fray.v2.local import LocalClient
from zephyr.execution import _default_zephyr_context

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


@pytest.fixture(autouse=True)
def reset_backend_context():
    """Reset backend context between tests for isolation."""
    _default_zephyr_context.set(None)
    yield
    _default_zephyr_context.set(None)


@pytest.fixture(autouse=True)
def fray_cluster():
    with set_current_client(LocalClient()):
        yield


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Disable WANDB logging during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
