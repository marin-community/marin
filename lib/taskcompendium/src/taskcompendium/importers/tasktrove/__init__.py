# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers and converters for TaskTrove Clean archives."""

import tomllib
from collections.abc import Callable, Mapping
from types import MappingProxyType

from taskcompendium.importers.tasktrove.convert import TASK_MANIFEST
from taskcompendium.importers.tasktrove.ifeval import IFEVAL_MODE
from taskcompendium.importers.tasktrove.ifeval import import_task as import_ifeval
from taskcompendium.importers.tasktrove.mcqa import import_task as import_mcqa
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.importers.tasktrove.structured_outputs import (
    CSV_MODE,
    XML_MODE,
    import_csv_task,
    import_xml_task,
)
from taskcompendium.models import TaskSpec

_TASKTROVE_IMPORTERS: Mapping[tuple[str, str], Callable[[TaskArchive], TaskSpec]] = MappingProxyType(
    {
        ("nemotron_mcqa", "mcq"): import_mcqa,
        ("nemotron_ifeval", IFEVAL_MODE): import_ifeval,
        ("nemotron_structured_outputs", XML_MODE): import_xml_task,
        ("nemotron_structured_outputs", CSV_MODE): import_csv_task,
    }
)


def import_task(archive: TaskArchive) -> TaskSpec:
    """Dispatch a bounded TaskTrove archive to its supported source converter."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())["metadata"]
        key = (metadata["converter"], metadata["mode"])
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, TypeError) as error:
        raise ValueError("TaskTrove archive does not identify a supported converter") from error
    importer = _TASKTROVE_IMPORTERS.get(key)
    if importer is None:
        raise ValueError(f"Unsupported TaskTrove converter: {key!r}")
    return importer(archive)
