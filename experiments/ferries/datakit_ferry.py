# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit smoke ferry: end-to-end download → normalize → dedup → consolidate → tokenize.

Runs against the FineWeb-Edu ``sample/10BT`` subset using the StepSpec DAG runner.
Output paths are placed under ``$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/...``.
"""

import json
import logging
import os

from rigging.filesystem import marin_temp_bucket, url_to_fs
from rigging.log_setup import configure_logging

from fray import ResourceConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import (
    ConsolidateConfig,
    FilterConfig,
    FilterType,
    consolidate,
)
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize

logger = logging.getLogger(__name__)


def build_steps(run_id: str) -> list[StepSpec]:
    base = f"datakit-smoke/{run_id}"

    # Filtered download — restrict to the sample/10BT subset so we don't pull
    # the entire fineweb-edu repo (TBs). Per-run isolated under $base/download.
    downloaded = download_hf_step(
        "datakit-smoke/download",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
        hf_urls_glob=["sample/10BT/*.parquet"],
        zephyr_max_parallelism=14,  # fineweb-edu sample/10BT has 14 parquet shards
        override_output_path=f"{base}/download",
    )

    # Normalize peaked at ~10 GB mem, 17 GB disk on 10BT; bump disk from default 10g.
    normalized = normalize_step(
        name="datakit-smoke/normalize",
        download=downloaded,
        input_path=f"{downloaded.output_path}/sample/10BT",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
        override_output_path=f"{base}/normalize",
    )

    # Dedup peaked at ~5 GB mem (default 32g is 6x over), 22 GB disk (default 5g).
    # max_parallelism=128: 10BT has only 106 normalized shards — 1024 created
    # ~900 empty reduce shards and massive scheduling overhead (~12 min wasted).
    # cc_max_iterations=1: CC converges in iteration 0 for this dataset size;
    # iterations 1-3 produced 0 changes but each still ran a full 3-stage pipeline.
    deduped = StepSpec(
        name="datakit-smoke/dedup_fuzzy_document",
        deps=[normalized],
        hash_attrs={"mode": "fuzzy_document"},
        fn=lambda output_path: dedup_fuzzy_document(
            input_paths=normalized.output_path,
            output_path=output_path,
            max_parallelism=128,
            cc_max_iterations=1,
            worker_resources=ResourceConfig(cpu=5, ram="16g", disk="30g"),
        ),
        override_output_path=f"{base}/dedup",
    )

    consolidated = StepSpec(
        name="datakit-smoke/consolidate",
        deps=[normalized, deduped],
        fn=lambda output_path: consolidate(
            ConsolidateConfig(
                input_path=normalized.output_path,
                output_path=output_path,
                filetype="parquet",
                filters=[
                    FilterConfig(
                        type=FilterType.REMOVE_DOC,
                        attribute_path=f"{deduped.output_path}/data",
                        name="dup_doc",
                        attribute_filetype="parquet",
                        keep_if_missing=True,
                    ),
                ],
                worker_resources=ResourceConfig(cpu=1, ram="8g"),
            )
        ),
        override_output_path=f"{base}/consolidate",
    )

    tokenized = StepSpec(
        name="datakit-smoke/tokenize",
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
        override_output_path=f"{base}/tokens",
    )

    return [downloaded, normalized, deduped, consolidated, tokenized]


def _write_status(status: str, marin_prefix: str) -> None:
    """Write ferry run status to FERRY_STATUS_PATH if set."""
    status_path = os.environ.get("FERRY_STATUS_PATH")
    if not status_path:
        return
    payload = json.dumps({"status": status, "marin_prefix": marin_prefix})
    fs, _ = url_to_fs(status_path)
    with fs.open(status_path, "w") as f:
        f.write(payload)
    logger.info("Wrote ferry status to %s", status_path)


def main() -> None:
    configure_logging()
    if not os.environ.get("MARIN_PREFIX"):
        os.environ["MARIN_PREFIX"] = marin_temp_bucket(ttl_days=1)

    marin_prefix = os.environ["MARIN_PREFIX"]
    logger.info("MARIN_PREFIX defaulted to %s", marin_prefix)
    run_id = os.environ["SMOKE_RUN_ID"]

    _write_status("running", marin_prefix)
    import time

    t0 = time.monotonic()
    StepRunner().run(build_steps(run_id))
    elapsed = time.monotonic() - t0
    logger.info("Datakit ferry total wall time: %.1fs (%.1fm)", elapsed, elapsed / 60)
    _write_status("succeeded", marin_prefix)


if __name__ == "__main__":
    main()
