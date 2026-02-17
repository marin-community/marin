# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.export.hf_upload import upload_dir_to_hf

_INPUT_PATH = "gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d"
_REPO_ID = "marin-community/medu-science-qa"

medu_exported = StepSpec(
    name="upload/medu-science-qa",
    hash_attrs={"input_path": _INPUT_PATH, "repo_id": _REPO_ID},
    fn=lambda output_path: upload_dir_to_hf(
        _INPUT_PATH,
        _REPO_ID,
        "dataset",
    ),
)

if __name__ == "__main__":
    StepRunner().run([medu_exported])
