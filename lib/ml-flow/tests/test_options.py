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

"""Tests for DataflowOptions."""


from ml_flow import DataflowOptions


def test_dataflow_options_defaults():
    """Test default values."""
    opts = DataflowOptions(project="test-project")

    assert opts.project == "test-project"
    assert opts.region == "us-central1"
    assert opts.runner == "DataflowRunner"
    assert opts.use_flex_rs is True
    assert opts.max_num_workers == 1000


def test_dataflow_options_to_pipeline_options():
    """Test conversion to PipelineOptions."""
    opts = DataflowOptions(
        project="test-project",
        region="us-west1",
        use_flex_rs=True,
        max_num_workers=500,
    )

    pipeline_opts = opts.to_pipeline_options()

    # Check key options
    all_opts = pipeline_opts.get_all_options()
    assert all_opts["project"] == "test-project"
    assert all_opts["region"] == "us-west1"
    assert all_opts["runner"] == "DataflowRunner"
    assert all_opts["max_num_workers"] == 500
    assert all_opts["flexrs_goal"] == "COST_OPTIMIZED"


def test_dataflow_options_without_flex_rs():
    """Test FlexRS can be disabled."""
    opts = DataflowOptions(
        project="test-project",
        use_flex_rs=False,
    )

    pipeline_opts = opts.to_pipeline_options()
    all_opts = pipeline_opts.get_all_options()

    # Beam may set flexrs_goal to None by default, so check it's not 'COST_OPTIMIZED'
    assert all_opts.get("flexrs_goal") != "COST_OPTIMIZED"


def test_dataflow_options_direct_runner():
    """Test DirectRunner configuration."""
    opts = DataflowOptions(
        project="test-project",
        runner="DirectRunner",
    )

    pipeline_opts = opts.to_pipeline_options()
    all_opts = pipeline_opts.get_all_options()

    assert all_opts["runner"] == "DirectRunner"


def test_dataflow_options_temp_staging_locations():
    """Test custom temp/staging locations."""
    opts = DataflowOptions(
        project="test-project",
        temp_location="gs://custom-temp/",
        staging_location="gs://custom-staging/",
    )

    pipeline_opts = opts.to_pipeline_options()
    all_opts = pipeline_opts.get_all_options()

    assert all_opts["temp_location"] == "gs://custom-temp/"
    assert all_opts["staging_location"] == "gs://custom-staging/"


def test_dataflow_options_auto_temp_staging():
    """Test automatic temp/staging location generation."""
    opts = DataflowOptions(project="my-project")

    pipeline_opts = opts.to_pipeline_options()
    all_opts = pipeline_opts.get_all_options()

    assert all_opts["temp_location"] == "gs://my-project-temp/dataflow/temp"
    assert all_opts["staging_location"] == "gs://my-project-temp/dataflow/staging"
