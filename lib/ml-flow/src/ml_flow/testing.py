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

"""Testing utilities for ml-flow pipelines."""

import tempfile
from pathlib import Path

from apache_beam.options.pipeline_options import PipelineOptions


def get_test_pipeline_options() -> PipelineOptions:
    """
    Get pipeline options for local testing with DirectRunner.

    Configured for fast local testing with minimal overhead.

    Returns:
        PipelineOptions configured for DirectRunner

    Example:
        options = get_test_pipeline_options()
        ds = Dataset.from_jsonl_files("test/*.jsonl", options)
    """
    return PipelineOptions(
        runner="DirectRunner",
        direct_num_workers=0,  # Use all available cores
        direct_running_mode="multi_threading",
    )


def create_temp_dir(prefix: str = "ml_flow_test") -> Path:
    """
    Create a temporary directory for testing.

    Directory is automatically cleaned up by OS (in /tmp).

    Args:
        prefix: Prefix for temp directory name

    Returns:
        Path to temporary directory

    Example:
        tmp_dir = create_temp_dir()
        input_path = tmp_dir / "input.jsonl"
    """
    temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    return Path(temp_dir)
