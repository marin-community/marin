# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the key-evals suite on a checkpoint and compile one report.

An example of the composable eval API: adopt a checkpoint as a typed handle, build one eval step per
``EvalGroup`` in the ``key_evals`` menu, aggregate them into an ``EvalReport``, and run the graph.
"""

from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import eval_report, eval_steps, key_evals

# A pre-existing checkpoint produced outside this graph: adopt it as a typed handle. Adoption
# resolves consumers to the source and writes only a provenance record — no copy, no recompute.
llama_200m = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)

results = eval_steps(llama_200m, key_evals(resources=ResourceConfig.with_tpu("v6e-8")))
report = eval_report(results, name=f"{llama_200m.name}/key")

if __name__ == "__main__":
    run(report)
