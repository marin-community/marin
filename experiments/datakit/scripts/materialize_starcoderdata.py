# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize StarCoderData (download and normalize) in the job's region.

StarCoder's training corpus: source files in 86 programming languages plus GitHub issues, Jupyter notebooks
and Git commits, 0.31 TB of Parquet in one directory per subset, with the document text in ``content``. The
subsets carry different metadata columns, so normalization keeps only the text and its identifiers. The
dataset is gated on Hugging Face: the downloading token must be allowed to read it.

The source is not in the shared registry (``all_sources``), so mixtures built from it are unchanged. Pin the
region when submitting, for example ``iris job run --region us-east5 ...``; the Iris worker exports the
region's ``MARIN_PREFIX``.
"""

import argparse

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

HF_DATASET_ID = "bigcode/starcoderdata"
HF_REVISION = "9fc30b578cedaec69e47302df72cf00feed7c8c4"


def normalize_steps() -> tuple[StepSpec, ...]:
    """The ``(download, normalize)`` chain for StarCoderData."""
    download = download_hf_step(
        "raw/starcoderdata",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["*/*.parquet"],
    )
    return download, normalize_step(
        name="normalized/starcoderdata",
        download=download,
        text_field="content",
        file_extensions=(".parquet",),
        bare=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the steps that would run without running them.")
    args = parser.parse_args()
    configure_logging()
    StepRunner().run([normalize_steps()[-1]], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
