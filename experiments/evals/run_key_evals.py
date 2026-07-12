# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import default_key_evals

# A pre-existing checkpoint produced outside this graph: adopt it as a typed handle. Adoption
# resolves consumers to the source and writes only a provenance record — no copy, no recompute.
llama_200m = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)
key_evals = default_key_evals(
    step=llama_200m,
    resource_config=ResourceConfig.with_tpu("v6e-8"),
    model_name="small_model",
)

if __name__ == "__main__":
    StepRunner().run([lower(x) for x in key_evals])
