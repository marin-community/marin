# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize FineWeb-Edu 10BT sample to datakit standard Parquet format."""

from iris.marin_fs import marin_temp_bucket
from marin.datakit.normalize import normalize_to_parquet

INPUT_PATH = "gs://marin-us-central2/raw/fineweb-edu-87f0914/sample/10BT"


def main():
    output_path = marin_temp_bucket(ttl_days=1, prefix="datakit/fineweb-edu/sample/10BT/normalized")
    normalize_to_parquet(
        input_path=INPUT_PATH,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
