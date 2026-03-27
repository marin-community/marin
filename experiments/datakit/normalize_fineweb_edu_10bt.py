# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize FineWeb-Edu 10BT sample to datakit standard Parquet format."""

from marin.datakit.normalize import normalize_to_parquet

INPUT_PATH = "gs://marin-us-central2/raw/fineweb-edu-87f0914/sample/10BT"
OUTPUT_PATH = "gs://marin-us-central2/datakit/fineweb-edu/sample/10BT/normalized"


def main():
    normalize_to_parquet(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        text_field="text",
        id_field="id",
    )


if __name__ == "__main__":
    main()
