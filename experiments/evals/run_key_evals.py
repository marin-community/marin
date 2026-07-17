# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the key-evals suite on a checkpoint and compile one report.

An example of the composable eval API on the deferred-version CLI: adopt a checkpoint as a typed
handle, build one eval step per ``EvalGroup`` in the ``key_evals`` menu, aggregate them into an
``EvalReport``, and let ``--version``/``--run`` drive the build.

    python -m experiments.evals.run_key_evals --version dev            # print the plan
    python -m experiments.evals.run_key_evals --version dev --run      # build it
    python -m experiments.evals.run_key_evals --version dev --run --limit 5   # bounded smoke
"""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

from experiments.evals.evalchemy.serve_and_eval import ServeSpec
from experiments.evals.evals import eval_report, eval_steps, key_evals

# A pre-existing checkpoint produced outside this graph: adopt it as a typed handle. Adoption
# resolves consumers to the source and writes only a provenance record — no copy, no recompute.
# The source is relative, so it resolves against the local bucket (MARIN_PREFIX, set by iris).
llama_200m = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)


def build(limit: int | None):
    results = eval_steps(llama_200m, key_evals(serve=ServeSpec(tpu_type="v6e-8"), max_eval_instances=limit))
    return eval_report(results, name=f"{llama_200m.name}/key")


@click.command()
@click.option("--limit", type=int, default=None, help="Cap examples per task (fast cluster smoke).")
@build_options
def main(limit: int | None):
    return build(limit=limit)


if __name__ == "__main__":
    main()
