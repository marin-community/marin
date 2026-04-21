# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: verify the oracle labels 10 documents end-to-end via Claude.

Trims the binary-sample smoke output to 10 docs, then runs ``label_quality``
against them. Kept small so the API cost is pennies.

Submit with::

    iris job run --priority production -- \
        python -m experiments.embed_everything.smoke_oracle
"""

import logging
import os
from itertools import islice

# Pin data region before any rigging/marin imports so marin_prefix() picks it up
# when not externally set (e.g., via `iris job run`).
DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.v2 import ResourceConfig  # noqa: E402
from rigging.filesystem import marin_temp_bucket  # noqa: E402
from zephyr.readers import load_parquet  # noqa: E402
from zephyr.writers import write_parquet_file  # noqa: E402

from experiments.embed_everything.oracle import OracleBackend, label_quality  # noqa: E402
from experiments.embed_everything.smoke_sample import sample_quality_binary_smoke  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

logger = logging.getLogger(__name__)

N_DOCS = 10

# Separate prefix so smoke outputs don't collide with the main experiment.
_OUTPUT_PREFIX = marin_temp_bucket(ttl_days=7, prefix="embed-everything-smoke")


def _trim_quality_samples(output_path: str, input_path: str, n: int) -> None:
    """Read the first *n* rows of ``quality_samples.parquet`` and rewrite them."""
    input_file = os.path.join(input_path, "quality_samples.parquet")
    output_file = os.path.join(output_path, "quality_samples.parquet")
    rows = list(islice(load_parquet(input_file), n))
    write_parquet_file(rows, output_file)


trim_quality_samples_smoke = StepSpec(
    name="trim_quality_samples_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[sample_quality_binary_smoke],
    hash_attrs={"n": N_DOCS, "v": 1},
    fn=remote(
        lambda output_path: _trim_quality_samples(
            output_path=output_path,
            input_path=sample_quality_binary_smoke.output_path,
            n=N_DOCS,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
    ),
)


oracle_quality_smoke = StepSpec(
    name="oracle_quality_smoke",
    output_path_prefix=_OUTPUT_PREFIX,
    deps=[trim_quality_samples_smoke],
    hash_attrs={"backend": "claude", "structured_output": True, "v": 1},
    fn=remote(
        lambda output_path: label_quality(
            output_path=output_path,
            input_path=trim_quality_samples_smoke.output_path,
            backend=OracleBackend.CLAUDE,
        ),
        resources=ResourceConfig.with_cpu(regions=[DATA_REGION]),
        pip_dependency_groups=["oracle"],
    ),
)


# StepRunner only tracks steps in the passed iterable; include the upstream
# sample step so its on-disk STATUS_SUCCESS is picked up and it's skipped
# rather than reported as unmet.
SMOKE_STEPS = [sample_quality_binary_smoke, trim_quality_samples_smoke, oracle_quality_smoke]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runner = StepRunner()
    runner.run(SMOKE_STEPS)
