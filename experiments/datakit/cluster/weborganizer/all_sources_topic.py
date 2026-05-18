# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Topic classification across every Datakit source.

Applies AllenAI's Dolma3 fasttext WebOrganizer topic classifier
(``allenai/dolma3-fasttext-weborganizer-topic-classifier``) to every normalized
source in :func:`marin.datakit.sources.all_sources` (minus the standard
``safety_pt/*`` / ``climblab-ja`` carve-outs -- see ``_EXCLUDE_PREFIXES``),
producing one co-partitioned Parquet attributes dataset per source. The output
schema is the datakit ``{id, partition_id, attributes}`` convention -- see
:mod:`experiments.datakit.fasttext` for the struct layout.

DAG shape::

    HF: allenai/dolma3-fasttext-weborganizer-topic-classifier@<revision>
        │
        ▼ prepare_fasttext_model_step  (_model/dolma3-weborg-topic_<hash>/)
        │     stages model.bin (~4 GiB) to GCS once
        │
        ▼ one classify_fasttext_step per Datakit source
              (topic/<source>_<hash>/)
              workers read the .bin from in-region GCS, scan the source's
              normalized parquet, and emit co-partitioned attributes.

The model prep step downloads ~4 GiB once. The fan-out to ~100 sources is what
consumes the bulk of the wall-clock time; each source's classify step is
shard-parallel by ``NormalizedData.num_partitions``.

Output rooted at ``gs://marin-eu-west4/datakit/weborganizer/`` -- permanent
(no TTL), alongside the v0 clustering artifacts under ``gs://marin-eu-west4/datakit/``.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production --region europe-west4 \\
        --job-name "weborg-topic-all-sources-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.weborganizer.all_sources_topic
"""

import logging

from fray import ResourceConfig
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
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

# Resourcing. fasttext is CPU-only and single-threaded per predict call.
# The Dolma3 weborganizer model.bin is ~4 GiB on disk and ~6-8 GiB resident
# after fasttext.load_model, so 8 GiB workers OOM (observed end-to-end in the
# weborg-topic-smoke-20260517-184843 run). 16 GiB gives enough headroom for
# the model plus the per-shard parquet writer buffer on the larger sources
# (cp/stackv2_code, finepdfs, nemotron medium_high_quality_synthetic).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Permanent output prefix -- sibling of the v0 clustering artifacts under
# ``gs://marin-eu-west4/datakit/``. No TTL, no churn between runs.
_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit/weborganizer"

# Sources excluded from the topic fan-out. Match against the registry name as
# a prefix (``safety_pt/`` skips every ``safety_pt/...`` source). Mirrors the
# standard datakit carve-outs in dedup/all_sources_fuzzy.py and store/all_sources_store.py:
# safety_pt and climblab-ja are deliberately omitted from the downstream
# consolidated store, so classifying them wastes worker time on data nothing
# downstream consumes.
_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "safety_pt/",
    "climblab-ja",
)


def build_classify_steps() -> list[StepSpec]:
    # Model prep: stage the .bin to GCS exactly once. The artifact's path is
    # then a dep on every per-source classify step, so all 100+ workers read
    # the same in-region GCS object.
    model_step = prepare_fasttext_model_step(
        name="_model/dolma3-weborg-topic",
        hf_repo_id=MODEL_HF_REPO,
        hf_filename=MODEL_HF_FILENAME,
        revision=MODEL_REVISION,
        output_path_prefix=_OUTPUT_PREFIX,
    )

    classify_steps: list[StepSpec] = []
    for name, src in all_sources().items():
        if any(name == p or name.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        classify_steps.append(
            classify_fasttext_step(
                name=f"topic/{name}",
                normalized=src.normalized,
                model_step=model_step,
                max_text_chars=MAX_TEXT_CHARS,
                k=K,
                threshold=THRESHOLD,
                worker_resources=WORKER_RESOURCES,
                output_path_prefix=_OUTPUT_PREFIX,
            )
        )
    return [model_step, *classify_steps]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_classify_steps())
