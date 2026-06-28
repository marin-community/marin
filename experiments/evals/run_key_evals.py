# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.evals.evals import default_key_evals

model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"
key_evals = default_key_evals(
    step=model_path,
    resource_config=ResourceConfig.with_tpu("v6e-8"),
    model_name="small_model",
)

if __name__ == "__main__":
    StepRunner().run([lower(x) for x in key_evals])
