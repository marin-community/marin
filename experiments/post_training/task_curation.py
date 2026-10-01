# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise TaskCompendium recipes against the existing GLM bulk relay.

uv run --no-sync --with './lib/taskcompendium[pipeline]' python -m \
    experiments.post_training.task_curation --cluster cw-us-east-02a \
    --relay-job /muchanem/glm53-relay \
    --recipe taskcompendium.pipeline.datasets.svamp \
    --recipe taskcompendium.pipeline.datasets.aime24 \
    --recipe taskcompendium.pipeline.datasets.gpqa \
    --limit 10 --output /tmp/task-curation-pilot
"""

import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from marin.inference.openai_batch import OpenAIBatchClient
from taskcompendium.pipeline.models import DatasetRecipe
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.runner import run_pipeline
from taskcompendium.pipeline.sources import source_rows

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL


class RecipeModule(Protocol):
    recipe: DatasetRecipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument("--base-url", help="Override the resolved address, for example with a local port-forward")
    parser.add_argument(
        "--recipe", action="append", required=True, help="Python module exporting a DatasetRecipe named recipe"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    token = os.environ[GLM_BULK_TOKEN_ENV]
    with connect_controller(cluster_name=args.cluster) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            resolved = client.resolver_for_job(JobName.from_string(args.relay_job)).resolve(GLM_MODEL)
            serving = resolved.first()
            base_url = (args.base_url or serving.url).rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            # The pilot confines cache reuse to this named serving job and local run.
            reviewer = BatchReviewer(
                OpenAIBatchClient(base_url, token), GLM_MODEL, args.relay_job, max_tokens=args.max_tokens
            )
            for module_name in args.recipe:
                recipe = cast(RecipeModule, import_module(module_name)).recipe
                manifest = run_pipeline(
                    recipe,
                    source_rows(recipe.source, args.limit),
                    output_path=args.output / recipe.name,
                    limit=args.limit,
                    reviewer=reviewer,
                )
                print(
                    json.dumps(
                        {
                            "dataset": recipe.name,
                            "input": manifest["input_rows"],
                            "reviewed": manifest["reviewed_rows"],
                            "dispositions": manifest["dispositions"],
                        }
                    )
                )


if __name__ == "__main__":
    main()
