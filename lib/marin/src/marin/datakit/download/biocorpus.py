# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize timodonnell's sequence-first biology corpus."""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_BUCKET_ID = "buckets/timodonnell/biocorpus"
BUCKET_SNAPSHOT = "2026-07-30"
SOURCE_XET_FINGERPRINT = "1c3ff878f17cd8398919a414f1b3646c1d010b99aaaf0a6fb82274edc939ea39"


def biocorpus_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the pinned bucket download and normalization chain for biocorpus."""
    download = download_hf_step(
        "raw/biocorpus",
        hf_dataset_id=HF_BUCKET_ID,
        revision=BUCKET_SNAPSHOT,
        hf_urls_glob=["data/*.jsonl.zst"],
        hf_repo_type_prefix="",
        expected_source_xet_fingerprint=SOURCE_XET_FINGERPRINT,
    )
    normalize = normalize_step(
        name="normalized/biocorpus",
        download=download,
        relative_input_path="data",
        file_extensions=(".jsonl.zst",),
    )
    return (download, normalize)
