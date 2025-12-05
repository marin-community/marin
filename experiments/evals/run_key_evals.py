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

from experiments.evals.evals import default_key_evals
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

# Insert your model path here
# model_path = "gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388"

model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"
key_evals = default_key_evals(
    step=model_path,
    resource_config=ResourceConfig.with_tpu("v6e-8"),
    # model_name="llama-8b-control-00f31b",
    model_name="small_model",
)

if __name__ == "__main__":
    executor_main(steps=key_evals)
