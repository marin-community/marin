# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit starcoder2-doc ferry: end-to-end pipeline on starcoder2_extras/documentation.

Exercises the long-tail document path that the FineWeb-Edu smoke ferry doesn't:
~60k docs / 1.4B tokens, p50 ~16KB but max ~62MB with 37 docs > 10MB.

The download step is reused verbatim from
``marin.datakit.download.starcoder2_extras`` so its relative
``override_output_path`` resolves under ``MARIN_PREFIX``. We default
``MARIN_PREFIX`` to the region-local stable marin bucket (e.g.
``gs://marin-us-central1``) so the step cache-hits on the staged copy
when one exists in the iris-picked region. Pipeline outputs go to
absolute TTL paths under ``marin_temp_bucket(ttl_days=1)`` to avoid
polluting stable storage with per-run CI data.
"""

import json
import logging
import os

from marin.datakit.download.starcoder2_extras import download_starcoder2_extras_step
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import marin_prefix, marin_temp_bucket, url_to_fs
from rigging.log_setup import configure_logging
from rigging.timing import log_time

logger = logging.getLogger(__name__)

SUBSET = "documentation"


def build_steps(run_id: str) -> list[StepSpec]:
    ttl_base = marin_temp_bucket(ttl_days=1, prefix=f"datakit-starcoder2-doc-smoke/{run_id}")

    download = download_starcoder2_extras_step(SUBSET)

    normalized = normalize_step(
        name="datakit-starcoder2-doc-smoke/normalize",
        download=download,
        text_field="content",
        id_field="id",
        file_extensions=(".parquet",),
        override_output_path=f"{ttl_base}/normalize",
    )

    minhash = StepSpec(
        name="datakit-starcoder2-doc-smoke/minhash",
        deps=[normalized],
        fn=lambda output_path: compute_minhash_attrs(
            source=Artifact.load(normalized, NormalizedData),
            output_path=output_path,
        ),
        override_output_path=f"{ttl_base}/minhash",
    )

    # 60k docs across ~3 shards — modest fan-out for fuzzy_dups CC.
    deduped = StepSpec(
        name="datakit-starcoder2-doc-smoke/fuzzy_dups",
        deps=[minhash],
        hash_attrs={"cc_max_iterations": 3},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(minhash, MinHashAttrData)],
            output_path=output_path,
            max_parallelism=16,
            cc_max_iterations=3,
        ),
        override_output_path=f"{ttl_base}/fuzzy_dups",
    )

    consolidated = StepSpec(
        name="datakit-starcoder2-doc-smoke/consolidate",
        deps=[normalized, deduped],
        fn=lambda output_path: consolidate(
            input_path=Artifact.load(normalized, NormalizedData).main_output_dir,
            output_path=output_path,
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.KEEP_DOC,
                    attribute_path=Artifact.load(deduped, FuzzyDupsAttrData)
                    .sources[Artifact.load(normalized, NormalizedData).main_output_dir]
                    .attr_dir,
                    name="is_cluster_canonical",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
        ),
        override_output_path=f"{ttl_base}/consolidate",
    )

    tokenized = StepSpec(
        name="datakit-starcoder2-doc-smoke/tokenize",
        deps=[consolidated],
        hash_attrs={"tokenizer": "gpt2"},
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[consolidated.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer="gpt2",
            )
        ),
        override_output_path=f"{ttl_base}/tokens",
    )

    return [download, normalized, minhash, deduped, consolidated, tokenized]


def _write_status(status: str, prefix: str) -> None:
    """Write ferry run status to FERRY_STATUS_PATH if set."""
    status_path = os.environ.get("FERRY_STATUS_PATH")
    if not status_path:
        return
    payload = json.dumps({"status": status, "marin_prefix": prefix})
    fs, _ = url_to_fs(status_path)
    with fs.open(status_path, "w") as f:
        f.write(payload)
    logger.info("Wrote ferry status to %s", status_path)


def main() -> None:
    configure_logging()
    # Pin MARIN_PREFIX to the region-local stable marin bucket so the download
    # step's relative override_output_path cache-hits on the staged copy.
    # Pipeline outputs are explicitly routed to TTL paths in build_steps().
    if not os.environ.get("MARIN_PREFIX"):
        os.environ["MARIN_PREFIX"] = marin_prefix()

    prefix = os.environ["MARIN_PREFIX"]
    logger.info("MARIN_PREFIX=%s", prefix)
    run_id = os.environ["SMOKE_RUN_ID"]

    _write_status("running", prefix)
    with log_time("Datakit starcoder2-doc ferry total wall time"):
        StepRunner().run(build_steps(run_id))
    _write_status("succeeded", prefix)


if __name__ == "__main__":
    main()
