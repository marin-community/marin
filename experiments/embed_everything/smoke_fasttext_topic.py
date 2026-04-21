# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: verify the WebOrganizer topic classifier loads and labels.

Samples ~30 topic docs (n_per_source=2 across Dolma's 15 sources), classifies
with the Dolma-3 fasttext WebOrganizer topic model, and writes a parquet. The
classification parquet is the end state — eval (canonicalization vs the 24
WebOrganizer labels) runs elsewhere.

Submit with::

    iris job run --priority production --region europe-west4 -- \\
        python -m experiments.embed_everything.smoke_fasttext_topic
"""

import logging
import os

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_prefix, marin_temp_bucket  # noqa: E402

from experiments.embed_everything.fasttext_baseline import classify_documents_fasttext_topic  # noqa: E402
from experiments.embed_everything.sample import sample_topic_documents  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

logger = logging.getLogger(__name__)

N_PER_SOURCE = 2  # 15 sources x 2 = 30 docs; enough to stress-test label mapping.
DOLMA_REL_PATH = "raw/dolma/v1.7"

# Separate prefix so smoke outputs don't collide with the main experiment.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="embed-everything-smoke")


sample_topic_smoke = StepSpec(
    name="sample_topic_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_source": N_PER_SOURCE, "seed": 42, "v": 1},
    fn=remote(
        lambda output_path: sample_topic_documents(
            output_path=output_path,
            dolma_base_path=f"{marin_prefix()}/{DOLMA_REL_PATH}",
            n_per_source=N_PER_SOURCE,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)


fasttext_topic_smoke = StepSpec(
    name="fasttext_topic_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_topic_smoke],
    hash_attrs={"model": "allenai/dolma3-fasttext-weborganizer-topic-classifier", "v": 1},
    fn=remote(
        lambda output_path: classify_documents_fasttext_topic(
            output_path=output_path,
            input_path=sample_topic_smoke.output_path,
        ),
        # 4 GB model binary — bump RAM/disk above the coordinator defaults.
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram="8g", disk="20g"),
        pip_dependency_groups=["fasttext"],
    ),
)


SMOKE_STEPS = [sample_topic_smoke, fasttext_topic_smoke]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(SMOKE_STEPS)
