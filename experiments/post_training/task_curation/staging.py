# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage recipe inputs without importing artifact graph construction on workers."""

from dataclasses import dataclass

import requests
from marin.datakit.download.huggingface import (
    DownloadConfig,
    download_hf,
    finish_download,
    plan_download,
    stream_file_to_fsspec,
)
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.inputs import HubDownload, UrlDownload
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset


@dataclass(frozen=True)
class DownloadInputs:
    downloads: tuple[HubDownload | UrlDownload, ...]
    output_path: str


def download_inputs(config: DownloadInputs, *, context: ZephyrContext | None = None) -> None:
    for declaration in config.downloads:
        if isinstance(declaration, HubDownload):
            destination = StoragePath(config.output_path)
            if declaration.subdirectory:
                destination = destination / declaration.subdirectory
            download = DownloadConfig(
                hf_dataset_id=declaration.dataset,
                revision=declaration.revision,
                hf_urls_glob=list(declaration.patterns),
                gcs_output_path=str(destination),
                wait_for_completion=True,
            )
            if context is None:
                download_hf(download)
                continue
            plan = plan_download(download)
            context.execute(
                Dataset.from_list(list(plan.tasks))
                .map(stream_file_to_fsspec)
                .write_jsonl(plan.metrics_path, skip_existing=True)
            )
            finish_download(plan)
        else:
            with requests.get(declaration.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                destination = StoragePath(config.output_path) / declaration.filename
                with destination.open("wb", auto_mkdir=True) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
