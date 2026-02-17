# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.evals import default_key_evals
from fray.cluster import ResourceConfig
from marin.execution.step_runner import StepRunner

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
    StepRunner().run(key_evals)
