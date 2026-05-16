# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global fuzzy dedup across every Datakit source.

Builds one ``compute_minhash_attrs`` step per source in
:func:`marin.datakit.sources.all_sources` (hanging off that source's
``normalized`` terminal), then feeds every minhash output into a single
``compute_fuzzy_dups_attrs`` terminal. ``StepRunner`` walks the resulting DAG,
materializing the per-source download/normalize chains on demand and
deduping shared family downloads by ``output_path`` — so no pre-staged
normalized data is required.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --config lib/iris/examples/marin.yaml job run --region europe-west4 -- \
        python experiments/datakit/dedup/all_sources_fuzzy.py
"""

import logging

from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from rigging.filesystem import marin_temp_bucket
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


# Sources excluded from the dedup. Match against the registry name as a
# prefix (e.g. ``safety_pt/`` skips every ``safety_pt/...`` source).
#
# These sources landed in the registry *after* an earlier dedup run had
# already built its CC iterations (``it_0`` through ``it_10``) at
# ``gs://marin-eu-west4/tmp/ttl=2d/rav/datakit/dedup_783d0380/`` but OOM'd at
# stage1-Reduce. Holding them out keeps the inputs list identical to that
# earlier run, so this step's deterministic ``hash_id`` lands on
# ``dedup_783d0380`` naturally -- and ``cc_resume=True`` below picks the
# cached CC state up where it left off, skipping the multi-hour stage0
# scatter and the 11 CC iterations.
_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "safety_pt/",
    "climblab-ja",
)


def build_dedup_step() -> StepSpec:
    minhash_steps: list[StepSpec] = []
    for name, src in all_sources().items():
        if any(name == p or name.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        norm = src.normalized
        minhash_steps.append(
            StepSpec(
                name=f"datakit/minhash/{name}",
                deps=[norm],
                fn=lambda op, n=norm: compute_minhash_attrs(
                    source=Artifact.from_path(n, NormalizedData),
                    output_path=op,
                    worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
                ),
            )
        )

    return StepSpec(
        name="datakit/dedup",
        output_path_prefix=marin_temp_bucket(ttl_days=2, prefix="rav"),
        deps=minhash_steps,
        fn=lambda op: compute_fuzzy_dups_attrs(
            inputs=[Artifact.from_path(s, MinHashAttrData) for s in minhash_steps],
            output_path=op,
            max_parallelism=2048,
            cc_resume=True,
            worker_resources=ResourceConfig(cpu=3, ram="32g", disk="5g"),
            coordinator_resources=ResourceConfig(cpu=1, ram="3.5g", preemptible=False),
        ),
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run([build_dedup_step()])
