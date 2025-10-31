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

"""Dataflow pipeline options and configuration."""

from dataclasses import dataclass
from typing import Literal

from apache_beam.options.pipeline_options import PipelineOptions


@dataclass
class DataflowOptions:
    """
    Configuration for Dataflow pipelines.

    Provides a high-level, typed interface for common Dataflow settings,
    with sensible defaults for Marin's batch processing workloads.

    Example:
        options = DataflowOptions(
            project="marin-project",
            region="us-central1",
            use_flex_rs=True,  # 40-60% cost savings
        )
        pipeline_opts = options.to_pipeline_options()
    """

    # GCP settings
    project: str
    region: str = "us-central1"
    temp_location: str | None = None
    staging_location: str | None = None

    # Execution mode
    runner: Literal["DirectRunner", "DataflowRunner"] = "DataflowRunner"

    # Auto-scaling
    num_workers: int | None = None  # Initial workers (default: auto)
    max_num_workers: int = 1000
    autoscaling_algorithm: str = "THROUGHPUT_BASED"

    # FlexRS (flexible resource scheduling for batch)
    use_flex_rs: bool = True  # 40-60% cost savings for batch workloads

    # Resources
    machine_type: str = "n1-standard-16"
    disk_size_gb: int = 100

    # Advanced
    save_main_session: bool = True  # Required for cloud functions
    setup_file: str | None = None

    def to_pipeline_options(self) -> PipelineOptions:
        """Convert to Apache Beam PipelineOptions."""
        options_dict = {
            "runner": self.runner,
            "project": self.project,
            "region": self.region,
            "temp_location": self.temp_location or f"gs://{self.project}-temp/dataflow/temp",
            "staging_location": self.staging_location or f"gs://{self.project}-temp/dataflow/staging",
            "max_num_workers": self.max_num_workers,
            "autoscaling_algorithm": self.autoscaling_algorithm,
            "machine_type": self.machine_type,
            "disk_size_gb": self.disk_size_gb,
            "save_main_session": self.save_main_session,
        }

        if self.num_workers is not None:
            options_dict["num_workers"] = self.num_workers

        if self.use_flex_rs:
            options_dict["flexrs_goal"] = "COST_OPTIMIZED"

        if self.setup_file:
            options_dict["setup_file"] = self.setup_file

        return PipelineOptions(**options_dict)
