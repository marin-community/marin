# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit download stage — fetch a HuggingFace dataset to persistent storage."""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.step_spec import StepSpec


def download_step(
    name: str,
    *,
    hf_dataset_id: str,
    revision: str,
    hf_urls_glob: list[str] | None = None,
    zephyr_max_parallelism: int = 8,
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that downloads a HuggingFace dataset.

    The raw download is preserved as-is in its original format and directory structure.

    Args:
        name: Step name (e.g. "fineweb/download").
        hf_dataset_id: HuggingFace dataset identifier (e.g. "HuggingFaceFW/fineweb").
        revision: Commit hash from the HF dataset repo.
        hf_urls_glob: Glob patterns to select specific files. Empty means all files.
        zephyr_max_parallelism: Maximum download parallelism.
        deps: Optional upstream dependencies.
        output_path_prefix: Override the default output path prefix.
        override_output_path: Override the computed output path entirely.

    Returns:
        A StepSpec whose output_path contains the raw downloaded files.
    """
    resolved_glob = hf_urls_glob or []

    def _run(output_path: str) -> None:
        download_hf(
            DownloadConfig(
                hf_dataset_id=hf_dataset_id,
                revision=revision,
                hf_urls_glob=resolved_glob,
                gcs_output_path=output_path,
                zephyr_max_parallelism=zephyr_max_parallelism,
            )
        )

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={
            "hf_dataset_id": hf_dataset_id,
            "revision": revision,
            "hf_urls_glob": resolved_glob,
        },
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
