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

import fsspec
from fsspec import AbstractFileSystem


def test_discover_latest_hf_checkpoints():
    temp_fs: AbstractFileSystem = fsspec.filesystem("memory")

    # Create a temporary directory
    temp_fs.mkdirs("checkpoints/hf/step-1000")
    temp_fs.mkdirs("checkpoints/hf/step-9999")
    temp_fs.touch("checkpoints/hf/step-1000/config.json")
    temp_fs.touch("checkpoints/hf/step-1000/tokenizer_config.json")
    temp_fs.touch("checkpoints/hf/step-9999/config.json")
    temp_fs.touch("checkpoints/hf/step-9999/tokenizer_config.json")

    # Test the function
    from marin.evaluation.utils import discover_hf_checkpoints

    checkpoints = discover_hf_checkpoints("memory:///")

    assert checkpoints == ["memory:///checkpoints/hf/step-1000", "memory:///checkpoints/hf/step-9999"]
