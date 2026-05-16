# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Topic classification across every Datakit source.

Applies AllenAI's Dolma3 fasttext WebOrganizer topic classifier
(``allenai/dolma3-fasttext-weborganizer-topic-classifier``) to every normalized
source in :func:`marin.datakit.sources.all_sources`, producing one
co-partitioned Parquet attributes dataset per source. The output schema is the
datakit ``{id, partition_id, attributes}`` convention — see
:mod:`experiments.datakit.fasttext` for the struct layout.

DAG shape:

    HF: allenai/dolma3-fasttext-weborganizer-topic-classifier@<revision>
        │
        ▼ prepare_fasttext_model_step (datakit/classify/_model/dolma3-weborg-topic)
        │     stages model.bin to GCS once
        │
        ▼ one classify_fasttext_step per Datakit source
              (datakit/classify/topic/<source>)
              workers read the .bin from in-region GCS, scan the source's
              normalized parquet, and emit co-partitioned attributes.

The model prep step is small (~MB-scale download) and runs once. The fan-out
to ~100 sources is what consumes the bulk of the wall-clock time; each source's
classify step is shard-parallel by ``NormalizedData.num_partitions``.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production \\
        -- python experiments/datakit/cluster/weborganizer/all_sources_topic.py
"""

import logging

from fray import ResourceConfig
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_temp_bucket
from rigging.log_setup import configure_logging

from experiments.datakit.fasttext import classify_fasttext_step, prepare_fasttext_model_step

logger = logging.getLogger(__name__)


# Model identity. The revision is pinned to a commit hash so the prep step's
# cache key is tied to bytes — HuggingFace allows silent re-uploads on a
# moving tag like ``main``, which would invisibly change downstream results.
# To bump the model, fetch a new hash via
# ``curl -s https://huggingface.co/api/models/<repo>`` and replace below.
MODEL_HF_REPO = "allenai/dolma3-fasttext-weborganizer-topic-classifier"
MODEL_HF_FILENAME = "model.bin"
MODEL_REVISION = "005a0da7d35651eb6f54553171f146bd62c5cdd2"

# Classifier inference knobs. ``K = -1`` keeps the full label distribution so
# downstream consolidate can apply different topic thresholds without
# re-classifying. ``MAX_TEXT_CHARS = 100 KiB`` caps the predict-call cost on
# pathologically long documents (matches the Dolma3 upstream pipeline). All
# three feed ``hash_attrs`` so changing them invalidates the cache.
K = -1
THRESHOLD = 0.0
MAX_TEXT_CHARS = 100_000

# Resourcing. fasttext is CPU-only and single-threaded per predict call;
# 2 CPU / 8 GB RAM matches the decon mark step and gives the parquet writer
# enough headroom on the larger sources (cp/stackv2_code, finepdfs, nemotron
# medium_high_quality_synthetic).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


def build_classify_steps() -> list[StepSpec]:
    # Model prep: stage the .bin to GCS exactly once. The artifact's path is
    # then a dep on every per-source classify step, so all 100+ workers read
    # the same in-region GCS object.
    model_step = prepare_fasttext_model_step(
        name="datakit/classify/_model/dolma3-weborg-topic",
        hf_repo_id=MODEL_HF_REPO,
        hf_filename=MODEL_HF_FILENAME,
        revision=MODEL_REVISION,
    )

    classify_output_prefix = marin_temp_bucket(ttl_days=7, prefix="rav/classify-topic-all-sources-v0")
    return [model_step] + [
        classify_fasttext_step(
            name=f"datakit/classify/topic/{name}",
            normalized=src.normalized,
            model_step=model_step,
            max_text_chars=MAX_TEXT_CHARS,
            k=K,
            threshold=THRESHOLD,
            worker_resources=WORKER_RESOURCES,
            output_path_prefix=classify_output_prefix,
        )
        for name, src in all_sources().items()
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_classify_steps())
