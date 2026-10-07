# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute Hugging Face file transfers on an owned or shared Zephyr pool."""

from contextlib import nullcontext

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.download.huggingface import (
    HF_DATASET_REPO_TYPE_PREFIX,
    DownloadConfig,
    finish_download,
    plan_download,
    stream_file_to_fsspec,
)
from marin.execution.step_spec import StepSpec

DEFAULT_MAX_WORKERS = 8


def download_hf(
    config: DownloadConfig,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    resources: ResourceConfig | None = None,
    context: ZephyrContext | None = None,
) -> None:
    """Execute pinned transfers without taking ownership of a supplied context."""
    plan = plan_download(config)
    pipeline = (
        Dataset.from_list(list(plan.tasks)).map(stream_file_to_fsspec).write_jsonl(plan.metrics_path, skip_existing=True)
    )
    manager = (
        nullcontext(context)
        if context is not None
        else ZephyrContext(name="download-hf", max_workers=max_workers, resources=resources)
    )
    with manager as active_context:
        active_context.execute(pipeline)
    finish_download(plan)


def download_hf_step(
    name: str,
    *,
    hf_dataset_id: str,
    revision: str,
    hf_urls_glob: list[str] | None = None,
    append_sha_to_path: bool = False,
    zephyr_max_parallelism: int = DEFAULT_MAX_WORKERS,
    deps: list[StepSpec] | None = None,
    override_output_path: str | None = None,
    worker_resources: ResourceConfig | None = None,
    hf_repo_type_prefix: str = HF_DATASET_REPO_TYPE_PREFIX,
    expected_source_xet_fingerprint: str | None = None,
) -> StepSpec:
    """Create a StepSpec that downloads a HuggingFace dataset.

    The raw download is preserved as-is in its original format and directory structure.

    Args:
        name: Step name (e.g. "raw/fineweb").
        hf_dataset_id: HuggingFace dataset identifier (e.g. "HuggingFaceFW/fineweb").
        revision: Commit hash from the HF dataset repo.
        hf_urls_glob: Glob patterns to select specific files. Empty means all files.
        append_sha_to_path: If True, write outputs under ``output_path/<revision>``.
        zephyr_max_parallelism: Maximum download parallelism.
        deps: Optional upstream dependencies.
        override_output_path: Override the computed output path entirely.
        hf_repo_type_prefix: Hugging Face source namespace. Use an empty string
            when ``hf_dataset_id`` is a ``buckets/...`` path.
        expected_source_xet_fingerprint: Expected fingerprint of the selected
            files' relative paths, sizes, and Xet hashes.

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
                append_sha_to_path=append_sha_to_path,
                hf_repo_type_prefix=hf_repo_type_prefix,
                expected_source_xet_fingerprint=expected_source_xet_fingerprint,
            ),
            max_workers=zephyr_max_parallelism,
            resources=worker_resources,
        )

    hash_attrs = {
        "hf_dataset_id": hf_dataset_id,
        "revision": revision,
        "hf_urls_glob": resolved_glob,
        "append_sha_to_path": append_sha_to_path,
    }
    if hf_repo_type_prefix != HF_DATASET_REPO_TYPE_PREFIX:
        hash_attrs["hf_repo_type_prefix"] = hf_repo_type_prefix
    if expected_source_xet_fingerprint is not None:
        hash_attrs["expected_source_xet_fingerprint"] = expected_source_xet_fingerprint

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs=hash_attrs,
        override_output_path=override_output_path,
    )
