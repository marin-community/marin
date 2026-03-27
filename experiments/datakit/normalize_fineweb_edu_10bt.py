# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize FineWeb-Edu 10BT sample to datakit standard Parquet format."""

from iris.marin_fs import marin_temp_bucket
from marin.datakit.normalize import normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

INPUT_PATH = "gs://marin-us-central2/raw/fineweb-edu-87f0914/sample/10BT"

# The download already exists on GCS — create a no-op step pointing at it.
dl = StepSpec(
    name="raw/fineweb-edu",
    fn=lambda output_path: None,
    override_output_path=INPUT_PATH,
)

norm = normalize_step(
    name="fineweb-edu/sample/10BT/normalize",
    download=dl,
    override_output_path=marin_temp_bucket(ttl_days=1, prefix="datakit/fineweb-edu/sample/10BT/normalized"),
)

if __name__ == "__main__":
    StepRunner().run([dl, norm])
