# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind complete pinned source files to reusable curation readers."""

import hashlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import requests
from fray.types import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import hf_download, raw_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.datasets.nemotron_ultra_catalog import NEMOTRON_SOURCES
from taskcompendium.pipeline.models import DatasetRecipe
from taskcompendium.pipeline.sources import (
    PLACEHOLDER_FILES,
    PLACEHOLDER_PINS,
    SourceFiles,
    SourceFormat,
    UltraComponentSelector,
    resolve_ultra_placeholder,
    select_eurus_code,
    unpack_task_binary,
)

from experiments.post_training.task_curation_source_bindings import SOURCE_DEFINITIONS

DIRECT_NAMES = frozenset(
    {
        "aime_1983_2024",
        "apps",
        "asdiv",
        "dapo_math",
        "eurus2_code",
        "gretel_text_to_sql",
        "gsm8k",
        "math500",
        "numina_math",
        "openscience",
        "rlvr_math",
        "verifiable_code",
    }
)
MATH_NAMES = frozenset({"hardmath", "hendrycks_math", "deepscaler"})
SWE_GYM_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SWE_GYM_FILE = "data/train-00000-of-00001.parquet"
DOWNLOAD_VERSION = "2026.10.02.1"


@dataclass(frozen=True)
class ExternalDownload:
    url: str
    output_path: str
    filename: str


@dataclass(frozen=True)
class HubFile:
    dataset: str
    revision: str
    path: str
    subdirectory: str


@dataclass(frozen=True)
class UltraDownload:
    dataset: str
    revision: str
    blend: str
    output_path: str
    auxiliary_files: tuple[HubFile, ...]


def _download_external(config: ExternalDownload) -> None:

    with requests.get(config.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        destination = StoragePath(config.output_path) / config.filename
        with destination.open("wb", auto_mkdir=True) as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)


def _download_ultra(config: UltraDownload) -> None:
    download_hf(
        DownloadConfig(
            hf_dataset_id=config.dataset,
            revision=config.revision,
            hf_urls_glob=[f"{config.blend}.jsonl"],
            gcs_output_path=config.output_path,
            wait_for_completion=True,
        )
    )
    for file in config.auxiliary_files:
        download_hf(
            DownloadConfig(
                hf_dataset_id=file.dataset,
                revision=file.revision,
                hf_urls_glob=[file.path],
                gcs_output_path=str(StoragePath(config.output_path) / file.subdirectory),
                wait_for_completion=True,
            )
        )


def _ultra_auxiliary_files(family: str) -> tuple[HubFile, ...]:
    if family == "swe-repo":
        return (HubFile("SWE-Gym/SWE-Gym", SWE_GYM_REVISION, SWE_GYM_FILE, "swe-gym-membership"),)
    if family == "math-answer":
        files = []
        for upstream, pin in PLACEHOLDER_PINS.items():
            subdirectory, path = PLACEHOLDER_FILES[upstream].split("/", 1)
            files.append(HubFile(upstream, pin, path, subdirectory))
        return tuple(files)
    return ()


def source_files(name: str, recipe: DatasetRecipe) -> SourceFiles:
    """Resolve the source owner's file declaration for a pinned recipe."""
    if name in SOURCE_DEFINITIONS:
        return SOURCE_DEFINITIONS[name].files
    if name in NEMOTRON_SOURCES:
        component = NEMOTRON_SOURCES[name]
        return SourceFiles(
            patterns=(f"{component.blend}.jsonl",),
            format=SourceFormat.JSONL,
            selector=UltraComponentSelector(component.selector, component.component, component.family),
            decoder=resolve_ultra_placeholder if component.family == "math-answer" else None,
        )
    if name in DIRECT_NAMES:
        module = import_module(f"taskcompendium.pipeline.datasets.{name}")
        if name == "asdiv":
            return SourceFiles(patterns=(Path(module.SOURCE_FILE).name,), format=SourceFormat.XML)
        if name == "eurus2_code":
            return SourceFiles((module.SOURCE_FILE,), SourceFormat(module.SOURCE_FORMAT), selector=select_eurus_code)
        return SourceFiles(patterns=(module.SOURCE_FILE,), format=SourceFormat(module.SOURCE_FORMAT))
    if name in MATH_NAMES:
        return import_module(f"taskcompendium.pipeline.datasets.{name}").SOURCE_FILES
    if name == "nemotron_if":
        return SourceFiles(("RL/instruction_following/instruction_following.jsonl",), SourceFormat.JSONL)
    if name == "rlvr_ifeval":
        return SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET)
    if name == "reasoning_gym_generated":
        return SourceFiles(("generator.tar.gz",), SourceFormat.GENERATED)
    if name == "nemo_actions":
        return SourceFiles(("train.jsonl",), SourceFormat.JSONL)
    if recipe.source.dataset == "open-thoughts/TaskTrove":
        return SourceFiles((f"{recipe.source.config}/tasks.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary)
    raise ValueError(f"No pinned source files declared for {name}")


def source_download(
    name: str, resources: ResourceConfig, recipe: DatasetRecipe
) -> tuple[ArtifactStep[Artifact], SourceFiles]:
    """Build a complete download shared by runs with different audit limits."""
    source = recipe.source
    files = source_files(name, recipe)
    is_ultra = name in NEMOTRON_SOURCES
    auxiliary_files = _ultra_auxiliary_files(NEMOTRON_SOURCES[name].family) if is_ultra else ()
    selection = hashlib.sha256(
        repr(
            (
                source.dataset,
                source.revision,
                files.patterns,
                auxiliary_files,
            )
        ).encode()
    ).hexdigest()[:16]
    download_name = f"task-curation/download/{source.dataset}/{selection}"
    if name == "reasoning_gym_generated":
        url = f"https://api.github.com/repos/{source.dataset}/tarball/{source.revision}"

        def config(ctx: StepContext) -> ExternalDownload:
            return ExternalDownload(url, ctx.output_path, "generator.tar.gz")

        download = raw_download(
            download_name,
            fn=_download_external,
            build_config=config,
            version=DOWNLOAD_VERSION,
            resources=resources,
        )
        return download, files
    if is_ultra:
        component = NEMOTRON_SOURCES[name]

        def config(ctx: StepContext) -> UltraDownload:
            return UltraDownload(source.dataset, source.revision, component.blend, ctx.output_path, auxiliary_files)

        download = raw_download(
            download_name,
            fn=_download_ultra,
            build_config=config,
            version=DOWNLOAD_VERSION,
            resources=resources,
        )
        return download, files
    if name == "asdiv":
        module = import_module("taskcompendium.pipeline.datasets.asdiv")

        def config(ctx: StepContext) -> ExternalDownload:
            return ExternalDownload(module.SOURCE_FILE, ctx.output_path, Path(module.SOURCE_FILE).name)

        download = raw_download(
            download_name,
            fn=_download_external,
            build_config=config,
            version=DOWNLOAD_VERSION,
            resources=resources,
        )
        return download, files
    download = hf_download(
        download_name,
        hf_id=source.dataset,
        revision=source.revision,
        version=DOWNLOAD_VERSION,
        urls_glob=files.patterns,
        resources=resources,
    )
    return download, files
