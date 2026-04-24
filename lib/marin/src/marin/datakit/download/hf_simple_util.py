# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic ``(download, normalize)`` chain for plain-HuggingFace sources.

For the large set of Datakit sources whose pipeline is just "download the HF
parquet, run normalize, done" — no custom filter, no multi-subset family
sharing. Per-family modules exist for sources that need more (see
``nemotron_v2.py``, ``common_corpus.py``, ``common_pile.py``, etc.).

Usage::

    from marin.datakit.download.hf_simple import hf_normalize_steps

    coderforge_steps = hf_normalize_steps(
        marin_name="coderforge",
        hf_dataset_id="togethercomputer/CoderForge-Preview",
        revision="060fca9",
        staged_path="raw/coderforge-preview_ad26b119",
    )
    # coderforge_steps is a (download, normalize) tuple; normalize is the
    # canonical artifact consumers sample/tokenize off of.
"""

from __future__ import annotations

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step as _normalize_step
from marin.execution.step_spec import StepSpec


def hf_normalize_steps(
    *,
    marin_name: str,
    hf_dataset_id: str,
    revision: str,
    staged_path: str | None = None,
    hf_urls_glob: tuple[str, ...] | None = None,
    data_subdir: str = "",
    id_field: str = "id",
    text_field: str = "text",
    file_extensions: tuple[str, ...] = (".parquet",),
) -> tuple[StepSpec, StepSpec]:
    """Return ``(download, normalize)`` for a generic HF-backed source.

    Args:
        marin_name: Stable short name used for the ``normalize`` step's output
            path (``normalized/<marin_name>``). Matches the ``DatakitSource.name``
            of the source this chain materializes.
        hf_dataset_id: HuggingFace repo id, e.g. ``"togethercomputer/CoderForge-Preview"``.
        revision: Pinned HF commit SHA.
        staged_path: Optional relative path under ``MARIN_PREFIX`` for a
            pre-staged dump; passed to ``download_hf_step`` as
            ``override_output_path``.
        hf_urls_glob: Optional file-path glob restriction on the download.
        data_subdir: Subdirectory within the downloaded tree that holds the
            actual data files (common for family dumps with per-subset subdirs).
        id_field, text_field, file_extensions: Schema hints forwarded to
            ``normalize_step``. Defaults match the Dolma convention.
    """
    base = hf_dataset_id.replace("/", "__")
    # Same-repo-different-staging cases (e.g. bigcode/StarCoder2-Extras subsets)
    # need distinct StepSpec names; append the last segment of staged_path when
    # it would otherwise collide.
    download_name = f"raw/{base}"
    if staged_path:
        tail = staged_path.rstrip("/").rsplit("/", 1)[-1]
        if tail and tail != base:
            download_name = f"{download_name}__{tail}"

    download = download_hf_step(
        download_name,
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        hf_urls_glob=list(hf_urls_glob) if hf_urls_glob else None,
        override_output_path=staged_path,
    )

    input_path = f"{download.output_path}/{data_subdir}" if data_subdir else download.output_path
    normalize = _normalize_step(
        name=f"data/normalized/{marin_name}",
        download=download,
        text_field=text_field,
        id_field=id_field,
        input_path=input_path,
        file_extensions=file_extensions,
    )
    return (download, normalize)
