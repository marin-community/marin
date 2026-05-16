# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-embed the two sources that OOM'd in the main production run with bumped RAM.

The default ``ram="16g"`` per Zephyr embed worker is enough for typical CC
text (p95 ~12K chars/doc), but blew up on sources with very long individual
docs:

* ``ghalogs/public``           — GitHub Actions logs (can be MB-sized per doc)
* ``starcoder2/documentation`` — long code documentation

Both crashed with exit-137 (container OOM) at parquet load before encoding
started — a single row group with very long docs exceeded the container limit
when the ``window=4096`` reader buffered it.

This job re-runs ONLY those two sources with ``ram="32g"``. The output path is
the same as the main run (same StepSpec hash), so once this succeeds the
SUCCESS marker is in place and a subsequent re-launch of ``exp_full_clusters``
will cache-hit all 102 embed outputs.

Submit::

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production \\
        --job-name "embed-clusters-fixup-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.embeddings.luxical.exp_fixup_oom
"""

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray import ResourceConfig  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402

from experiments.datakit.embeddings.luxical.pipeline import (  # noqa: E402
    LUXICAL_REPO,
    LUXICAL_WEIGHTS_FILE,
    embed_source,
)

logger = logging.getLogger(__name__)

# Bumped from ram="16g" to ram="32g" so the parquet load can buffer a row group
# with MB-sized docs without tripping the cgroup OOM. Window stays at 4096; if
# 32 GB still isn't enough on some pathological shard we drop window to 1024
# (4x less in-flight memory).
FIXUP_SOURCES = ("ghalogs/public", "starcoder2/documentation")
EMBED_WINDOW = 4096
EMBED_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="32g", regions=[DATA_REGION])
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="4g", regions=[DATA_REGION])
EMBED_MAX_WORKERS_PER_SOURCE = 128

_THREAD_ENV = {
    var: "8"
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
}

_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit"


def _build_steps() -> list[StepSpec]:
    sources = all_sources()
    steps: list[StepSpec] = []
    upstream: list[StepSpec] = []
    for source_name in FIXUP_SOURCES:
        source = sources[source_name]
        normalized = source.normalized
        normalized_path = normalized.output_path
        upstream.extend(source.normalize_steps)
        steps.append(
            StepSpec(
                name=f"embed/luxical/{source_name}",
                output_path_prefix=_OUTPUT_PREFIX,
                deps=[normalized],
                # Same hash_attrs as exp_full_clusters so the output path
                # collides intentionally — replacing the prior FAILED status.
                hash_attrs={
                    "luxical_repo": LUXICAL_REPO,
                    "luxical_weights": LUXICAL_WEIGHTS_FILE,
                    "quant_dtype": "int8",
                    "quant_range": 0.6,
                    "window": EMBED_WINDOW,
                    "v": 2,
                },
                fn=remote(
                    lambda output_path, normalized_path=normalized_path: embed_source(
                        output_path=output_path,
                        normalized_path=normalized_path,
                        window_size=EMBED_WINDOW,
                        worker_resources=EMBED_WORKER_RESOURCES,
                        max_workers=EMBED_MAX_WORKERS_PER_SOURCE,
                    ),
                    resources=COORDINATOR_RESOURCES,
                    env_vars=_THREAD_ENV,
                    pip_dependency_groups=["embed"],
                ),
            )
        )
    return [*upstream, *steps]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps())
