# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit nemotron ferry: weekly full-pipeline run on the Nemotron-CC medium split.

Pipeline: verify raw dump → normalize → minhash → fuzzy_dups → consolidate →
tokenize. The first step is verification-only: it confirms the ``quality=medium``
subtree of the Nemotron-CC dump is already staged at ``NEMOTRON_RAW_PATH`` and
refuses to initiate a Common Crawl download.

Optionally, when ``--sft-general-path <gs://...>`` is given, runs a second
pipeline (verify → normalize → minhash → fuzzy_dups → consolidate → tokenize)
on the Nemotron-Pretraining-SFT-v1 / Nemotron-SFT-General split. The path is
caller-supplied (no hardcoded region), verified at runtime, and never
downloaded; the ferry fails fast if the expected parquet shards aren't there.
The SFT-General chain runs after the medium chain so each stage's
scheduled-baseline timing is recorded under a stable step name.

Pipeline outputs land under ``$MARIN_PREFIX/datakit-nemotron-smoke/$SMOKE_RUN_ID/...``;
``MARIN_PREFIX`` defaults to a region-local temp bucket with 1-day TTL.
"""

import argparse
import json
import logging
import os

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
from rigging.filesystem import (
    check_path_in_region,
    marin_temp_bucket,
    region_from_metadata,
    url_to_fs,
)
from rigging.log_setup import configure_logging
from rigging.timing import log_time

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


# Subdirectory under the staged Nemotron-Pretraining-SFT-v1 dump. The full
# bucket path is caller-supplied via ``--sft-general-path`` so the ferry isn't
# pinned to a single region.
SFT_GENERAL_SUBDIR = "Nemotron-SFT-General"


def _verify_sft_general_present(output_path: str) -> None:
    """Confirm Nemotron-SFT-General is staged at ``output_path``; never downloads.

    Mirrors ``_verify_nemotron_medium_present``: a cache eviction must never
    trigger an HF download — the gate runs on already-staged data only.
    """
    sft_dir = f"{output_path}/{SFT_GENERAL_SUBDIR}"
    fs, _ = url_to_fs(sft_dir)
    if not fs.exists(sft_dir):
        raise RuntimeError(
            f"Nemotron-SFT-General not found at {sft_dir}. "
            "The nemotron ferry refuses to download from HuggingFace — stage the raw dump externally first."
        )
    sample = fs.glob(f"{sft_dir}/**/*.parquet", maxdepth=4)
    if not sample:
        raise RuntimeError(f"Nemotron-SFT-General at {sft_dir} contains no .parquet files.")
    logger.info("Nemotron-SFT-General confirmed at %s (e.g. %s)", sft_dir, sample[0])


def build_steps(
    run_id: str,
    *,
    file_stride: int = 1,
    sft_general_path: str | None = None,
) -> list[StepSpec]:
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
        file_stride=file_stride,
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

    steps: list[StepSpec] = [download, normalized, minhash, deduped, consolidated, tokenized]

    # Optional: parallel chain on Nemotron-SFT-General (parquet). Wired only
    # when the caller passes ``--sft-general-path``. Step names are prefixed
    # with ``sft-general/`` so the medium chain's existing names remain
    # unchanged (preserves Iris cache and the metric-collector regex).
    if sft_general_path is not None:
        sft_download = StepSpec(
            name="datakit-nemotron-smoke/sft-general/download",
            fn=_verify_sft_general_present,
            override_output_path=sft_general_path,
        )

        # SFT-General has skewed shards with row groups up to ~4.7 GB
        # (see lib/marin/src/marin/datakit/download/nemotron_v2.py:140-143).
        # Bump normalize workers to 64 GB RAM / 10 GB disk to match the
        # convention used by the datakit download flow; later stages stay
        # at the default 16 GB because the data is already smaller per-shard
        # after normalize.
        sft_normalized = normalize_step(
            name="datakit-nemotron-smoke/sft-general/normalize",
            download=sft_download,
            text_field="text",
            id_field="id",
            relative_input_path=SFT_GENERAL_SUBDIR,
            worker_resources=ResourceConfig(cpu=2, ram="64g", disk="10g"),
            max_workers=512,
            override_output_path=f"{base}/sft-general/normalize",
        )

        sft_minhash = StepSpec(
            name="datakit-nemotron-smoke/sft-general/minhash",
            deps=[sft_normalized],
            fn=lambda output_path: compute_minhash_attrs(
                source=Artifact.load(sft_normalized, NormalizedData),
                output_path=output_path,
                worker_resources=ResourceConfig(cpu=5, ram="16g", disk="5g"),
                max_workers=512,
            ),
            override_output_path=f"{base}/sft-general/minhash",
        )

        sft_deduped = StepSpec(
            name="datakit-nemotron-smoke/sft-general/fuzzy_dups",
            deps=[sft_minhash],
            hash_attrs={"cc_max_iterations": 3},
            fn=lambda output_path: compute_fuzzy_dups_attrs(
                inputs=[Artifact.load(sft_minhash, MinHashAttrData)],
                output_path=output_path,
                max_parallelism=512,
                cc_max_iterations=3,
                worker_resources=ResourceConfig(cpu=1, ram="16g", disk="5g"),
            ),
            override_output_path=f"{base}/sft-general/fuzzy_dups",
        )

        sft_consolidated = StepSpec(
            name="datakit-nemotron-smoke/sft-general/consolidate",
            deps=[sft_normalized, sft_deduped],
            fn=lambda output_path: consolidate(
                input_path=Artifact.load(sft_normalized, NormalizedData).main_output_dir,
                output_path=output_path,
                filetype="parquet",
                filters=[
                    FilterConfig(
                        type=FilterType.KEEP_DOC,
                        attribute_path=Artifact.load(sft_deduped, FuzzyDupsAttrData)
                        .sources[Artifact.load(sft_normalized, NormalizedData).main_output_dir]
                        .attr_dir,
                        name="is_cluster_canonical",
                        attribute_filetype="parquet",
                        keep_if_missing=True,
                    ),
                ],
                worker_resources=ResourceConfig(cpu=1, ram="16g", disk="5g"),
                max_workers=512,
            ),
            override_output_path=f"{base}/sft-general/consolidate",
        )

        sft_tokenized = StepSpec(
            name="datakit-nemotron-smoke/sft-general/tokenize",
            deps=[sft_consolidated],
            hash_attrs={"tokenizer": "gpt2"},
            fn=lambda output_path: tokenize(
                TokenizeConfig(
                    train_paths=[sft_consolidated.output_path],
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer="gpt2",
                    max_workers=512,
                    worker_resources=ResourceConfig(ram="16g", disk="5g"),
                )
            ),
            override_output_path=f"{base}/sft-general/tokens",
        )

        steps += [sft_download, sft_normalized, sft_minhash, sft_deduped, sft_consolidated, sft_tokenized]

    return steps


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
    parser = argparse.ArgumentParser(description="Datakit nemotron ferry.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Take every Nth input file from nemotron-medium (default 1 = full medium). "
            "Used by zephyr-perf Gate 2 to sub-slice without copying data; "
            "calibrated start: 5 (~1/5 of medium, ~2-3h wall)."
        ),
    )
    parser.add_argument(
        "--sft-general-path",
        default=None,
        help=(
            "Optional gs:// path to a staged Nemotron-Pretraining-SFT-v1 dump "
            "(must contain a `Nemotron-SFT-General/` subdirectory of .parquet "
            "shards). When set, the ferry runs a second pipeline on this "
            "dataset after the medium chain. The path is caller-supplied so "
            "the ferry isn't pinned to any single region; verified at runtime "
            "and never downloaded."
        ),
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error(f"--stride must be >= 1, got {args.stride}")

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
        if args.sft_general_path:
            check_path_in_region("nemotron_sft_general", args.sft_general_path, region)

    _write_status("running", marin_prefix)
    with log_time("Datakit nemotron ferry total wall time"):
        StepRunner().run(
            build_steps(
                run_id,
                file_stride=args.stride,
                sft_general_path=args.sft_general_path,
            )
        )
    _write_status("succeeded", marin_prefix)


if __name__ == "__main__":
    main()
