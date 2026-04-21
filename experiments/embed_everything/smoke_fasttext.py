# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: verify the Dolma-3 fasttext quality classifier loads and scores.

Reuses the 10-doc trim from smoke_oracle so we don't re-sample. Model is 4 GB
and is fetched from the HuggingFace Hub once per Iris worker; expect the
task duration to be dominated by that download on a cold worker.

Submit with::

    iris job run --priority production --extra fasttext -- \\
        python -m experiments.embed_everything.smoke_fasttext
"""

import logging
import os

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_temp_bucket  # noqa: E402

from experiments.embed_everything.fasttext_baseline import score_documents_fasttext_quality  # noqa: E402
from experiments.embed_everything.smoke_oracle import (  # noqa: E402
    sample_quality_binary_smoke,
    trim_quality_samples_smoke,
)
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

logger = logging.getLogger(__name__)

# Separate prefix so smoke outputs don't collide with the main experiment.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="embed-everything-smoke")


fasttext_quality_smoke = StepSpec(
    name="fasttext_quality_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[trim_quality_samples_smoke],
    hash_attrs={"model": "allenai/dolma3-fasttext-quality-classifier", "v": 1},
    fn=remote(
        lambda output_path: score_documents_fasttext_quality(
            output_path=output_path,
            input_path=trim_quality_samples_smoke.output_path,
        ),
        # 4 GB model + workspace: bump disk and ram above the 1 GB/5 GB default
        # that iris applies to the coordinator job.
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION], ram="8g", disk="20g"),
        pip_dependency_groups=["fasttext"],
    ),
)


# StepRunner only tracks deps in the iterable; include the full chain so
# upstream already-succeeded steps are recognized via on-disk STATUS_SUCCESS.
SMOKE_STEPS = [sample_quality_binary_smoke, trim_quality_samples_smoke, fasttext_quality_smoke]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(SMOKE_STEPS)
