# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off run to validate zephyr counters in fuzzy dedup on the dev cluster."""

from iris.logging import configure_logging
from iris.marin_fs import marin_prefix, marin_temp_bucket
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

if __name__ == "__main__":
    configure_logging()

    input_paths = f"{marin_prefix()}/raw/fineweb-edu-87f0914/sample/10BT"
    output_path = marin_temp_bucket(ttl_days=1, prefix="dedup-counter-test")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=output_path,
        max_parallelism=128,
    )
    print(f"Result: {result}")
