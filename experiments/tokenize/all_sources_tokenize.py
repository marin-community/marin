# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize every Datakit source -- produce co-partitioned attribute parquets.

Mirrors ``experiments/decontamination/all_sources_decon.py`` and
``experiments/datakit/dedup/all_sources_fuzzy.py`` shape: one
:func:`tokenize_attributes_step` per source in
:func:`marin.datakit.sources.all_sources`, hanging off that source's
``normalized`` terminal. Each step emits ``{id, input_ids}`` parquet
co-aligned with the source partitions.

Stage A only -- this script produces the datakit attribute artifact
(``TokenizedAttrData``). Building a Levanter cache from these attributes
is a separate step (``build_levanter_store_step``) and is intentionally
out of scope here.

Tokenizer pinned to ``marin-community/marin-tokenizer`` (Llama 3.1
equivalent, 128K vocab) -- the canonical Marin training tokenizer.

Outputs go to ``<MARIN_PREFIX>/datakit/tokenize/<source>_<hash>/`` under
the worker's region (eu-west4 when run on the marin cluster from
europe-west4). Persistent, content-addressed: re-running with the same
tokenizer + source resolves to the same path and short-circuits.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive \\
        -- python experiments/tokenize/all_sources_tokenize.py
"""

import logging

from fray.types import ResourceConfig
from levanter.tokenizers import TokenizerBackend
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_BACKEND = TokenizerBackend.HF
WORKER_RESOURCES = ResourceConfig(ram="10g", disk="5g")

# Override the tokenize_attributes_step default of 4096. With
# MAX_CONCURRENT_STEPS=20, 4096 caps to up to ~80K concurrent workers
# across the fleet -- the previous run pegged the Iris coordinator with
# worker-registration storms. 1024 keeps the worst-case fleet at ~20K
# while still saturating mid-sized sources (<=1024 partitions get all
# their shards in flight, larger sources see the cap).
MAX_WORKERS_PER_STEP = 1024


def build_tokenize_steps() -> list[StepSpec]:
    return [
        tokenize_attributes_step(
            name=f"datakit/tokenize/{name}",
            train_normalize=src.normalized,
            tokenizer=TOKENIZER,
            tokenizer_backend=TOKENIZER_BACKEND,
            max_workers=MAX_WORKERS_PER_STEP,
            worker_resources=WORKER_RESOURCES,
        )
        for name, src in all_sources().items()
    ]


MAX_CONCURRENT_STEPS = 20


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_tokenize_steps(), max_concurrent=MAX_CONCURRENT_STEPS)
