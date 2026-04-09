# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit smoke ferry: end-to-end download → normalize → dedup → consolidate → tokenize.

Runs against the FineWeb-Edu ``sample/10BT`` subset using the StepSpec DAG runner.
Output paths are placed under ``$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/...``.
"""

import os

from rigging.log_setup import configure_logging

from marin.datakit.canonical.fineweb_edu import download as fineweb_edu_download
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


def build_steps(run_id: str) -> list[StepSpec]:
    base = f"datakit-smoke/{run_id}"

    # Reuse the canonical download step. Its output path is shared across runs
    # (raw/fineweb-edu-<rev> under MARIN_PREFIX); only the per-run derived
    # outputs land under datakit-smoke/<run_id>/.
    downloaded = fineweb_edu_download()

    normalized = normalize_step(
        name="datakit-smoke/normalize",
        download=downloaded,
        input_path=f"{downloaded.output_path}/sample/10BT",
        override_output_path=f"{base}/normalize",
    )

    deduped = StepSpec(
        name="datakit-smoke/dedup_fuzzy_document",
        deps=[normalized],
        hash_attrs={"mode": "fuzzy_document"},
        fn=lambda output_path: dedup_fuzzy_document(
            input_paths=normalized.output_path,
            output_path=output_path,
            max_parallelism=1024,
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


def main() -> None:
    configure_logging()
    run_id = os.environ["SMOKE_RUN_ID"]
    StepRunner().run(build_steps(run_id))


if __name__ == "__main__":
    main()
