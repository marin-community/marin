# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate full nemotron v1 pipeline: normalize → minhash → fuzzy_dups → consolidate → tokenize.

Run via Iris:

  iris job run --region europe-west4 --no-wait \\
    -e MARIN_PREFIX gs://marin-eu-west4 \\
    -e VALIDATE_DATASET nemotron_v1 -- \\
    python scripts/validate_normalize_phase1.py

Entrypoint uses default resources (auto non-preemptible via iris heuristic).
Raw data is read from production buckets. All output goes to
gs://marin-tmp-eu-west4/ttl=1d/<run_id>/... with auto-cleanup.
"""

import os
import time

from fray.v2 import ResourceConfig

from marin.datakit.download.nemotron_v1 import NEMOTRON_V1_SPLITS, download_nemotron_v1_step
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

# Output bucket — auto-cleaned after 1 day
TMP_BUCKET = "gs://marin-tmp-eu-west4/ttl=1d"


# ---------------------------------------------------------------------------
# nemotron_v1: normalize → minhash → fuzzy_dups → consolidate → tokenize
# ---------------------------------------------------------------------------
def nemotron_v1_steps(run_id: str) -> list[StepSpec]:
    base = f"{TMP_BUCKET}/{run_id}"

    download = download_nemotron_v1_step()
    data_root = f"{download.output_path}/contrib/Nemotron/Nemotron-CC/data-jsonl"

    normalized = normalize_step(
        name="normalized/nemotron_v1/medium",
        download=download,
        text_field="text",
        id_field="id",
        file_extensions=(".jsonl.gz",),
        input_path=f"{data_root}/{NEMOTRON_V1_SPLITS['medium']}",
        max_workers=512,
        override_output_path=f"{base}/normalize",
    )

    minhash = StepSpec(
        name="minhash/nemotron_v1",
        deps=[normalized],
        hash_attrs={"mode": "minhash_attrs"},
        fn=lambda output_path: compute_minhash_attrs(
            source=Artifact.load(normalized, NormalizedData),
            output_path=output_path,
            worker_resources=ResourceConfig(cpu=5, ram="16g", disk="10g"),
            max_workers=512,
        ),
        override_output_path=f"{base}/minhash",
    )

    fuzzy_dups = StepSpec(
        name="fuzzy_dups/nemotron_v1",
        deps=[minhash],
        hash_attrs={"mode": "fuzzy_dups", "cc_max_iterations": 3},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(minhash, MinHashAttrData)],
            output_path=output_path,
            cc_max_iterations=3,
            max_parallelism=512,
            worker_resources=ResourceConfig(cpu=1, ram="32g", disk="10g"),
        ),
        override_output_path=f"{base}/fuzzy_dups",
    )

    consolidated = StepSpec(
        name="consolidated/nemotron_v1",
        deps=[normalized, fuzzy_dups],
        fn=lambda output_path: consolidate(
            input_path=Artifact.load(normalized, NormalizedData).main_output_dir,
            output_path=output_path,
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.REMOVE_DOC,
                    attribute_path=Artifact.load(fuzzy_dups, FuzzyDupsAttrData)
                    .sources[Artifact.load(normalized, NormalizedData).main_output_dir]
                    .attr_dir,
                    name="dup_doc",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
            worker_resources=ResourceConfig(cpu=1, ram="16g"),
            max_workers=512,
        ),
        override_output_path=f"{base}/consolidate",
    )

    tokenized = StepSpec(
        name="tokenized/nemotron_v1",
        deps=[consolidated],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[consolidated.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer="gpt2",
                max_workers=512,
            )
        ),
        override_output_path=f"{base}/tokenize",
    )

    return [download, normalized, minhash, fuzzy_dups, consolidated, tokenized]


DATASETS = {
    "nemotron_v1": nemotron_v1_steps,
}

if __name__ == "__main__":
    dataset = os.environ.get("VALIDATE_DATASET")
    if not dataset:
        raise SystemExit(f"Set VALIDATE_DATASET to one of: {', '.join(DATASETS)}")
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}")

    run_id = os.environ.get("RUN_ID") or f"norm-val-{dataset}-{int(time.time())}"
    steps = DATASETS[dataset](run_id)
    StepRunner().run(steps)
