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

from marin.execution.executor import executor_main
from marin.export.hf_upload import upload_dir_to_hf

medu_exported = upload_dir_to_hf(
    "gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d",
    "marin-community/medu-science-qa",
    "dataset",
)

if __name__ == "__main__":
    executor_main([medu_exported])
