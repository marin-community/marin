# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: verify both quality-sampling strategies run end-to-end at N=100.

Builds two tiny sample steps (Nemotron 5-bucket and binary Dolma arxiv+wiki vs
cc_en_tail) at n_per_bucket=100, writes to a temp bucket, and runs via
StepRunner.

Submit with::

    iris job run --priority production -- \
        python -m experiments.embed_everything.smoke_sample
"""

import logging
import os

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_prefix, marin_temp_bucket  # noqa: E402

from experiments.embed_everything.sample import (  # noqa: E402
    sample_quality_documents_binary,
    sample_quality_documents_nemotron,
)
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

logger = logging.getLogger(__name__)

N_PER_BUCKET = 100
NEMOTRON_REL_PATH = "raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
DOLMA_REL_PATH = "raw/dolma/v1.7"

# Separate prefix so smoke outputs don't collide with the main experiment.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="embed-everything-smoke")

sample_quality_nemotron_smoke = StepSpec(
    name="sample_quality_nemotron_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_bucket": N_PER_BUCKET, "seed": 42, "v": 1},
    fn=remote(
        lambda output_path: sample_quality_documents_nemotron(
            output_path=output_path,
            nemotron_base_path=f"{marin_prefix()}/{NEMOTRON_REL_PATH}",
            n_per_bucket=N_PER_BUCKET,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

sample_quality_binary_smoke = StepSpec(
    name="sample_quality_binary_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    hash_attrs={"n_per_bucket": N_PER_BUCKET, "seed": 42, "v": 1},
    fn=remote(
        lambda output_path: sample_quality_documents_binary(
            output_path=output_path,
            dolma_base_path=f"{marin_prefix()}/{DOLMA_REL_PATH}",
            n_per_bucket=N_PER_BUCKET,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)

SMOKE_STEPS = [sample_quality_nemotron_smoke, sample_quality_binary_smoke]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(SMOKE_STEPS)
