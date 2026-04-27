# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit nemotron ferry: weekly full-pipeline run on the Nemotron-CC medium split.

Pipeline: verify raw dump → normalize → minhash → fuzzy_dups → consolidate →
tokenize. The first step is verification-only: it confirms the ``quality=medium``
subtree of the Nemotron-CC dump is already staged at ``NEMOTRON_RAW_PATH`` and
refuses to initiate a Common Crawl download.

Pipeline outputs land under ``$MARIN_PREFIX/datakit-nemotron-smoke/$SMOKE_RUN_ID/...``;
``MARIN_PREFIX`` defaults to a region-local temp bucket with 1-day TTL.
"""

import json
import logging
import os

from rigging.filesystem import (
    check_path_in_region,
    marin_temp_bucket,
    region_from_metadata,
    url_to_fs,
)
from rigging.log_setup import configure_logging
from rigging.timing import log_time

from fray import ResourceConfig
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

logger = logging.getLogger(__name__)

# Canonical, region-pinned location of the staged Nemotron-CC raw dump. The
# dump was populated by a one-off download into marin-eu-west4; the ferry only
# reads from it and will fail-fast if it isn't there. Matches the path used in
# ``experiments/dedup/poc_nemotron.py``.
NEMOTRON_RAW_PATH = "gs://marin-eu-west4/raw/nemotro-cc-eeb783"
NEMOTRON_DATA_SUBDIR = "contrib/Nemotron/Nemotron-CC/data-jsonl"
NEMOTRON_MEDIUM_DIR = "quality=medium"


def _verify_nemotron_medium_present(output_path: str) -> None:
    """Confirm the medium split is staged at ``output_path``; never downloads.

    Invoked by StepRunner only on a cache miss. Raises with a clear message so
    that an accidental cache eviction can never trigger a multi-TB Common Crawl
    re-download.
    """
    medium_dir = f"{output_path}/{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_MEDIUM_DIR}"
    fs, _ = url_to_fs(medium_dir)
    if not fs.exists(medium_dir):
        raise RuntimeError(
            f"Nemotron-CC medium split not found at {medium_dir}. "
            "The nemotron ferry refuses to download Common Crawl — stage the raw dump externally first."
        )
    sample = fs.glob(f"{medium_dir}/**/*.jsonl.*", maxdepth=4)
    if not sample:
        raise RuntimeError(f"Nemotron-CC medium split at {medium_dir} contains no .jsonl.* files.")
    logger.info("Nemotron-CC medium split confirmed at %s (e.g. %s)", medium_dir, sample[0])


def build_steps(run_id: str) -> list[StepSpec]:
    base = f"datakit-nemotron-smoke/{run_id}"

    # Verify-only raw step. Uses an absolute override so it points at the
    # pre-staged dump regardless of MARIN_PREFIX.
    download = StepSpec(
        name="datakit-nemotron-smoke/download",
        fn=_verify_nemotron_medium_present,
        override_output_path=NEMOTRON_RAW_PATH,
    )

    # Sizes mirror validate_normalize_phase1.py, which ran successfully on
    # nemotron_v1 in eu-west4. 512 workers across all fan-out stages.
    normalized = normalize_step(
        name="datakit-nemotron-smoke/normalize",
        download=download,
        text_field="text",
        id_field="id",
        relative_input_path=f"{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_MEDIUM_DIR}",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=512,
        override_output_path=f"{base}/normalize",
    )

    minhash = StepSpec(
        name="datakit-nemotron-smoke/minhash",
        deps=[normalized],
        fn=lambda output_path: compute_minhash_attrs(
            source=Artifact.load(normalized, NormalizedData),
            output_path=output_path,
            worker_resources=ResourceConfig(cpu=5, ram="16g", disk="5g"),
            max_workers=512,
        ),
        override_output_path=f"{base}/minhash",
    )

    deduped = StepSpec(
        name="datakit-nemotron-smoke/fuzzy_dups",
        deps=[minhash],
        hash_attrs={"cc_max_iterations": 3},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(minhash, MinHashAttrData)],
            output_path=output_path,
            max_parallelism=512,
            cc_max_iterations=3,
            worker_resources=ResourceConfig(cpu=1, ram="16g", disk="5g"),
        ),
        override_output_path=f"{base}/fuzzy_dups",
    )

    consolidated = StepSpec(
        name="datakit-nemotron-smoke/consolidate",
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
            worker_resources=ResourceConfig(cpu=1, ram="16g", disk="5g"),
            max_workers=512,
        ),
        override_output_path=f"{base}/consolidate",
    )

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
                max_workers=512,
                worker_resources=ResourceConfig(ram="16g", disk="5g"),
            )
        ),
        override_output_path=f"{base}/tokens",
    )

    return [download, normalized, minhash, deduped, consolidated, tokenized]


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

    # Guard against accidental cross-region reads of the multi-TB raw dump.
    region = region_from_metadata()
    if region:
        check_path_in_region("nemotron_raw", NEMOTRON_RAW_PATH, region)

    _write_status("running", marin_prefix)
    with log_time("Datakit nemotron ferry total wall time"):
        StepRunner().run(build_steps(run_id))
    _write_status("succeeded", marin_prefix)


if __name__ == "__main__":
    main()
