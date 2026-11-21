# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the nanoswe Lightning trajectories: pretraining text and structured chat, train and validation.

The source is not in the shared registries (``all_sources``, ``all_sft_sources``), so mixtures built from
them are unchanged. Pin the staging region when submitting, for example ``iris job run --region us-east5 ...``;
the Iris worker exports the region's ``MARIN_PREFIX``.
"""

import argparse

from marin.datakit.chat_normalize import normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.nanoswe_lightning import (
    SOURCE_CHAT_SCHEMA,
    SOURCE_ID,
    SOURCE_NAME,
    SPLIT_GLOBS,
    TRANSFORM_VERSION,
    Split,
    transform,
    transform_chat,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

HF_REVISION = "64d22ccf2cf5c7693e19f329e5d56d2ea34e1cad"


def download_step() -> StepSpec:
    return download_hf_step(
        "raw/nanoswe-trajs-261002",
        hf_dataset_id=SOURCE_ID,
        revision=HF_REVISION,
        hf_urls_glob=[SPLIT_GLOBS[Split.TRAIN], SPLIT_GLOBS[Split.VALIDATION]],
    )


def _suffix(split: Split) -> str:
    return "" if split == Split.TRAIN else f"-{split}"


def text_steps(split: Split) -> tuple[StepSpec, ...]:
    """The ``(download, transform, normalize)`` pretraining-text chain for one split."""
    dl = download_step()
    processed = StepSpec(
        name=f"processed/{SOURCE_NAME}{_suffix(split)}",
        deps=[dl],
        fn=lambda output_path: transform(dl.output_path, output_path, split),
        hash_attrs={"version": TRANSFORM_VERSION, "split": str(split)},
    )
    return dl, processed, normalize_step(name=f"normalized/{SOURCE_NAME}{_suffix(split)}", download=processed)


def chat_steps(split: Split) -> tuple[StepSpec, ...]:
    """The ``(download, transform, chat normalize)`` structured-chat chain for one split."""
    dl = download_step()
    processed = StepSpec(
        name=f"processed-chat/{SOURCE_NAME}{_suffix(split)}",
        deps=[dl],
        fn=lambda output_path: transform_chat(dl.output_path, output_path, split),
        hash_attrs={"version": TRANSFORM_VERSION, "split": str(split)},
    )
    return (
        dl,
        processed,
        normalize_chat_step(
            name=f"normalized-chat/{SOURCE_NAME}{_suffix(split)}",
            download=processed,
            output_schema=SOURCE_CHAT_SCHEMA,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the steps that would run without running them.")
    args = parser.parse_args()
    configure_logging()
    terminals = [chain(split)[-1] for split in Split for chain in (text_steps, chat_steps)]
    StepRunner().run(terminals, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
