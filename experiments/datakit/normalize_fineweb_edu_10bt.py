# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize FineWeb-Edu 10BT sample to datakit standard Parquet format."""

from iris.marin_fs import marin_temp_bucket
from marin.datakit.canonical.fineweb_edu import download, normalize
from marin.execution.step_runner import StepRunner

dl = download()

norm = normalize(
    subset="sample/10BT",
    override_output_path=marin_temp_bucket(ttl_days=1, prefix="datakit/fineweb-edu/sample/10BT/normalized"),
)

if __name__ == "__main__":
    StepRunner().run([dl, norm])
