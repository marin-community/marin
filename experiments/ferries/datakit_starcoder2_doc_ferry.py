# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit starcoder2-doc ferry: end-to-end pipeline on starcoder2_extras/documentation.

Exercises the long-tail document path that the FineWeb-Edu smoke ferry doesn't:
~60k docs / 1.4B tokens, p50 ~16KB but max ~62MB with 37 docs > 10MB. The
download step pins the canonical staged dump so reruns reuse the cached copy
instead of re-downloading from HuggingFace. If the staged dump has been
evicted, ``download_hf_step`` falls back to a fresh HF download into the same
absolute path.

Pipeline outputs land under ``$MARIN_PREFIX/datakit-starcoder2-doc-smoke/$SMOKE_RUN_ID/...``;
``MARIN_PREFIX`` defaults to a region-local temp bucket with 1-day TTL.
"""

import json
import logging
import os

from fray import ResourceConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.starcoder2_extras import HF_DATASET_ID, HF_REVISION
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
from rigging.filesystem import (
    check_path_in_region,
    marin_temp_bucket,
    region_from_metadata,
    url_to_fs,
)
from rigging.log_setup import configure_logging
from rigging.timing import log_time

logger = logging.getLogger(__name__)

SUBSET = "documentation"

# Canonical, region-pinned location of the staged starcoder2-extras documentation
# subset. Points the download step at an absolute path so reruns hit the cache
# instead of pulling 1.85 GB from HuggingFace. Path encodes ``HF_REVISION`` so
# bumping the pin in starcoder2_extras.py forces a fresh download on next run.
STARCODER2_DOC_RAW_PATH = f"gs://marin-us-central1/raw/starcoder2_extras-{HF_REVISION}/{SUBSET}"


def build_steps(run_id: str) -> list[StepSpec]:
    base = f"datakit-starcoder2-doc-smoke/{run_id}"

    # Cache-hit on the staged dump when present; falls through to HF download
    # only on cache eviction or revision bump.
    download = download_hf_step(
        f"datakit-starcoder2-doc-smoke/download/{SUBSET}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{SUBSET}/*.parquet"],
        override_output_path=STARCODER2_DOC_RAW_PATH,
    )

    # Bumped RAM (32g) to absorb the heavy tail — 37 docs are 10-62 MB and
    # bloat several-fold once decoded into pyarrow.
    normalized = normalize_step(
        name="datakit-starcoder2-doc-smoke/normalize",
        download=download,
        text_field="content",
        id_field="id",
        file_extensions=(".parquet",),
        worker_resources=ResourceConfig(cpu=2, ram="32g", disk="20g"),
        override_output_path=f"{base}/normalize",
    )

    minhash = StepSpec(
        name="datakit-starcoder2-doc-smoke/minhash",
        deps=[normalized],
        fn=lambda output_path: compute_minhash_attrs(
            source=Artifact.load(normalized, NormalizedData),
            output_path=output_path,
            worker_resources=ResourceConfig(cpu=5, ram="32g", disk="20g"),
        ),
        override_output_path=f"{base}/minhash",
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
            worker_resources=ResourceConfig(cpu=1, ram="16g", disk="30g"),
        ),
        override_output_path=f"{base}/fuzzy_dups",
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
            worker_resources=ResourceConfig(cpu=1, ram="16g"),
        ),
        override_output_path=f"{base}/consolidate",
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

    # Fail-fast if Iris picked a region that doesn't host the staged dump:
    # cross-region reads of the parquet shards would dominate the run.
    region = region_from_metadata()
    if region:
        check_path_in_region("starcoder2_doc_raw", STARCODER2_DOC_RAW_PATH, region)

    _write_status("running", marin_prefix)
    with log_time("Datakit starcoder2-doc ferry total wall time"):
        StepRunner().run(build_steps(run_id))
    _write_status("succeeded", marin_prefix)


if __name__ == "__main__":
    main()
