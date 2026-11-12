# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable artifact types shared by Atlas lexical entrypoints."""

from marin.datakit.decon import AllSourceDropSets, DeconAttributes
from marin.execution.artifact import Artifact


class SourceTextPopulation(Artifact):
    source_id: str
    dataset_id: str
    dataset_revision: str
    source_version: str
    converter_revision: str
    text_renderer_version: str
    task_count: int
    text_sha256: str
    task_specs_sha256: str
    main_output_dir: str
    specs_output_dir: str
    converter_rejections: dict[str, int]
    admission_rejections: dict[str, int]


class LexicalDropSets(Artifact):
    attributes: AllSourceDropSets


class SourceLexicalMarks(Artifact):
    source_id: str
    population_path: str
    attributes: DeconAttributes


class LexicalMeasurements(Artifact):
    population_set: str
    source_ids: list[str]
    measurements: dict[str, dict]
