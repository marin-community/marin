# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: combine 2 tokenize attribute outputs into one Levanter store.

Wires two ``tokenize_attributes_step`` outputs (the 2 smallest sources in
``all_sources()``) into a single :func:`build_levanter_store_step`. The
tokenize steps cache-skip via their already-SUCCESS executor_status from
the all-sources tokenize run, so only the merge step actually executes.

Goal: exercise the Stage A -> Stage B handoff and verify the resulting
Levanter cache loads cleanly.

Submit on iris (eu-west4):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive \\
        -- python experiments/tokenize/smoke_test_levanter_store.py
"""

import logging

from fray.types import ResourceConfig
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step
from marin.processing.tokenize.store_builder import build_levanter_store_step
from rigging.log_setup import configure_logging

from experiments.tokenize.all_sources_tokenize import (
    MAX_WORKERS_PER_STEP,
    TOKENIZER,
    TOKENIZER_BACKEND,
    WORKER_RESOURCES,
)

logger = logging.getLogger(__name__)

# Two tiny single-shard sources from all_sources() -- ~0.02B tokens combined.
# Both already tokenized at gs://marin-eu-west4/datakit/tokenize/<src>_<hash>/
# from the all-sources run, so the tokenize deps short-circuit.
SMOKE_SOURCES = ("cp/peps", "cp/foodista")


def build_smoke_step() -> StepSpec:
    sources = all_sources()
    tokenize_steps = [
        tokenize_attributes_step(
            name=f"datakit/tokenize/{name}",
            train_normalize=sources[name].normalized,
            tokenizer=TOKENIZER,
            tokenizer_backend=TOKENIZER_BACKEND,
            max_workers=MAX_WORKERS_PER_STEP,
            worker_resources=WORKER_RESOURCES,
        )
        for name in SMOKE_SOURCES
    ]
    return build_levanter_store_step(
        name="datakit/levanter-store/smoke",
        tokenize_steps=tokenize_steps,
        # Bump worker memory for the consolidate step; defaults are tight for the
        # exemplar/field-counts derivation.
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
        max_workers=MAX_WORKERS_PER_STEP,
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run([build_smoke_step()])
