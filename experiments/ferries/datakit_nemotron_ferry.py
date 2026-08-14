# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit nemotron ferry: weekly full-pipeline run on the Nemotron-CC high split.

Pipeline: verify raw dump → normalize → minhash → fuzzy_dups → full-text
verification → consolidate → tokenize. The first step confirms the ``quality=high``
subtree of the Nemotron-CC dump is already staged at ``NEMOTRON_RAW_PATH`` and
refuses to initiate a Common Crawl download.

Pipeline outputs land under a region-local one-day temp prefix.
"""

import json
import logging
import os

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)
from marin.processing.classification.deduplication.fuzzy_dups import (
    FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    DEFAULT_PIPELINE_SHARDS_PER_WORKER,
    REFERENCE_LARGE_CLUSTER_PARAMS,
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyVerificationStoreConfig,
    VerifiedFuzzyDupsAttrData,
    verify_fuzzy_dups,
)
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import (
    StoragePath,
    check_path_in_region,
    marin_temp_bucket,
    prefix_join,
    region_from_metadata,
    url_to_fs,
)
from rigging.log_setup import configure_logging
from rigging.timing import log_time

logger = logging.getLogger(__name__)

# Canonical, region-pinned location of the staged Nemotron-CC raw dump. The
# dump was populated by a one-off download into marin-eu-west4; the ferry only
# reads from it and will fail-fast if it isn't there.
NEMOTRON_RAW_PATH = "gs://marin-eu-west4/raw/nemotro-cc-eeb783"
NEMOTRON_DATA_SUBDIR = "contrib/Nemotron/Nemotron-CC/data-jsonl"
NEMOTRON_QUALITY_DIR = "quality=high"
FUZZY_VERIFICATION_STORE_CONFIG = FuzzyVerificationStoreConfig(
    recovery_timeout=1_800,
    ready_timeout=1_800,
    lookup_batch_size=128,
    shards_per_worker=1,
)


def _verify_nemotron_quality_present(output_path: str) -> None:
    """Confirm the quality split is staged at ``output_path``; never downloads.

    Invoked by StepRunner only on a cache miss. Raises with a clear message so
    that an accidental cache eviction can never trigger a multi-TB Common Crawl
    re-download.
    """
    quality_dir = f"{output_path}/{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}"
    fs, _ = url_to_fs(quality_dir)
    if not fs.exists(quality_dir):
        raise RuntimeError(
            f"Nemotron-CC {NEMOTRON_QUALITY_DIR} not found at {quality_dir}. "
            "The nemotron ferry refuses to download Common Crawl — stage the raw dump externally first."
        )
    sample = fs.glob(f"{quality_dir}/**/*.jsonl.*", maxdepth=4)
    if not sample:
        raise RuntimeError(f"Nemotron-CC {NEMOTRON_QUALITY_DIR} at {quality_dir} contains no .jsonl.* files.")
    logger.info("Nemotron-CC %s confirmed at %s (e.g. %s)", NEMOTRON_QUALITY_DIR, quality_dir, sample[0])


def build_steps(base: str) -> list[StepSpec]:
    base_path = StoragePath(base)

    # Verify-only raw step. Uses an absolute override so it points at the
    # pre-staged dump regardless of MARIN_PREFIX.
    download = StepSpec(
        name="datakit-nemotron-smoke/download",
        fn=_verify_nemotron_quality_present,
        override_output_path=NEMOTRON_RAW_PATH,
    )

    # Sizes mirror validate_normalize_phase1.py, which ran successfully on
    # nemotron_v1 in eu-west4. 512 workers across all fan-out stages.
    # The yaml sets FERRY_TEST_MAX_FILES=1000 to cap the input shard count
    # (quality=high has ~2,755 shards / ~960 GB; 1000 keeps the run inside
    # the GH 6h cap). Read at execution time by `_discover_files`.
    normalized = normalize_step(
        name="datakit-nemotron-smoke/normalize",
        download=download,
        text_field="text",
        id_field="id",
        relative_input_path=f"{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=512,
        override_output_path=str(base_path / "normalize"),
    )  # ~1,380 output shards

    minhash = StepSpec(
        name="datakit-nemotron-smoke/minhash",
        deps=[normalized],
        fn=lambda output_path: compute_minhash_attrs(
            source=read_artifact(normalized.output_path, NormalizedData),
            output_path=output_path,
            worker_resources=(resources := ResourceConfig(cpu=16, ram="64g", disk="32g")),
            map_task_resources=resources.scale(1 / 16),
            reduce_task_resources=resources.scale(3 / 16),
        ),
        override_output_path=str(base_path / "minhash"),
    )  # ~1,380 output shards

    candidates = StepSpec(
        name="datakit-nemotron-smoke/fuzzy_dups",
        deps=[minhash],
        hash_attrs={"artifact_version": FUZZY_DUPS_ATTR_DATA_VERSION, "cc_max_iterations": 3},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(minhash.output_path, MinHashAttrData)],
            output_path=output_path,
            cc_max_iterations=3,
            worker_resources=(resources := ResourceConfig(cpu=16, ram="160g", disk="32g")),
            map_task_resources=resources.scale(1 / 16),
            reduce_task_resources=resources.scale(3 / 16),
        ),
        override_output_path=str(base_path / "fuzzy_dups"),
    )  # ~1,380 output shards

    verification_params = FuzzyVerificationParams()
    verified = StepSpec(
        name="datakit-nemotron-smoke/verify_fuzzy_dups",
        deps=[normalized, minhash, candidates],
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
            "local_representatives": REFERENCE_LOCAL_REPRESENTATIVE_PARAMS.model_dump(mode="json"),
            "large_clusters": REFERENCE_LARGE_CLUSTER_PARAMS.model_dump(mode="json"),
            "pipeline_shards_per_worker": DEFAULT_PIPELINE_SHARDS_PER_WORKER,
        },
        fn=lambda output_path: verify_fuzzy_dups(
            normalized_sources={"source": read_artifact(normalized.output_path, NormalizedData)},
            minhash_sources={"source": read_artifact(minhash.output_path, MinHashAttrData)},
            candidates=read_artifact(candidates.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            verification_params=verification_params,
            local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            store_config=FUZZY_VERIFICATION_STORE_CONFIG,
            worker_resources=(resources := ResourceConfig(cpu=16, ram="160g", disk="32g")),
            map_task_resources=resources.scale(1 / 16),
            reduce_task_resources=resources.scale(3 / 16),
        ),
        override_output_path=prefix_join(base, "verify_fuzzy_dups"),
    )

    consolidated = StepSpec(
        name="datakit-nemotron-smoke/consolidate",
        deps=[normalized, verified],
        fn=lambda output_path: consolidate(
            input_path=read_artifact(normalized.output_path, NormalizedData).main_output_dir,
            output_path=output_path,
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.REMOVE_DOC,
                    attribute_path=read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData).attr_dir_for_source(
                        read_artifact(normalized.output_path, NormalizedData).main_output_dir
                    ),
                    name="dup_doc",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
            worker_resources=(resources := ResourceConfig(cpu=16, ram="32g", disk="16g")),
            map_task_resources=resources.scale(1 / 16),
        ),
        override_output_path=str(base_path / "consolidate"),
    )  # ~1,380 output shards

    tokenized = StepSpec(
        name="datakit-nemotron-smoke/tokenize",
        deps=[consolidated],
        hash_attrs={"tokenizer": "gpt2"},
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[consolidated.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer="gpt2",
                worker_resources=(resources := ResourceConfig(cpu=16, ram="80g", disk="16g")),
                map_task_resources=resources.scale(1 / 16),
            )
        ),
        override_output_path=str(base_path / "tokens"),
    )  # ~1,380 output shards

    return [download, normalized, minhash, candidates, verified, consolidated, tokenized]


def _write_status(status: str, marin_prefix: str) -> None:
    """Write ferry run status to FERRY_STATUS_PATH if set."""
    status_path = os.environ.get("FERRY_STATUS_PATH")
    if not status_path:
        return
    payload = json.dumps({"status": status, "marin_prefix": marin_prefix})
    StoragePath(status_path).write_text(payload)
    logger.info("Wrote ferry status to %s", status_path)


def main() -> None:
    configure_logging()
    run_id = os.environ["SMOKE_RUN_ID"]
    output_prefix = marin_temp_bucket(ttl_days=1, prefix=f"datakit-nemotron-smoke/{run_id}")
    logger.info("Output prefix: %s", output_prefix)

    # Guard against accidental cross-region reads of the multi-TB raw dump.
    region = region_from_metadata()
    if region:
        check_path_in_region("nemotron_raw", NEMOTRON_RAW_PATH, region)

    _write_status("running", output_prefix)
    with log_time("Datakit nemotron ferry total wall time"):
        StepRunner().run(build_steps(output_prefix))
    _write_status("succeeded", output_prefix)


if __name__ == "__main__":
    main()
