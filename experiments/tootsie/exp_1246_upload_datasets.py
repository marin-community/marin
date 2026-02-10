# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.executor import executor_main
from marin.export.hf_upload import upload_dir_to_hf

medu_exported = upload_dir_to_hf(
    "gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d",
    "marin-community/medu-science-qa",
    "dataset",
)

if __name__ == "__main__":
    executor_main([medu_exported])
