# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore component provenance only after a complete pinned KTO/DPO content join."""

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.datasets import preference_tasks
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import HubDownload, RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow, ReviewRubric, TaskPolicy
from taskcompendium.runtime.resources import inline_resource

KTO_REVISION = "4470f033f33364e7d064c9f920c3df54d0cce767"
PARENT_REVISION = "f8869fc91bde5c71a104667292addcbbfd15985d"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
PARENT_FILE = "component-parent/data/train-00000-of-00001.parquet"
COMPONENTS = (
    "argilla/distilabel-capybara-dpo-7k-binarized",
    "argilla/distilabel-intel-orca-dpo-pairs",
    "argilla/ultrafeedback-binarized-preferences-cleaned",
)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_PARENT_ROWS = 10000
MAX_KTO_ROWS = 20000
PARQUET_BATCH_ROWS = 32


class KtoComponentError(ValueError):
    """A failed join prevents admission of every component selector."""

    def __init__(self, kind: ImportFailureKind, reason: str):
        self.kind = kind
        self.reason = reason
        super().__init__(f"{kind.value}: {reason}")


def _rows(path: StoragePath, max_rows: int) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        if stream.tell() > MAX_FILE_BYTES:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_input_exceeds_read_budget")
        stream.seek(0)
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows > max_rows:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_input_exceeds_row_budget")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, use_threads=False):
            yield from batch.to_pylist()


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_message_schema_unavailable")
    for message in value:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
        ):
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_message_schema_unavailable")
    return value


def _key(prompt: list[dict[str, str]], completion: list[dict[str, str]], label: bool) -> str:
    value = json.dumps([label, prompt, completion], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if len(value.encode()) > MAX_MESSAGE_BYTES:
        raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_messages_exceed_join_budget")
    return value


def _kto_key(row: dict[str, Any]) -> str:
    if not isinstance(row.get("label"), bool):
        raise KtoComponentError(ImportFailureKind.SOURCE_DEFECT, "malformed_kto_boolean_label")
    prompt = _messages(row.get("prompt"))
    completion = _messages(row.get("completion"))
    return _key(prompt, completion, row["label"])


@dataclass(frozen=True)
class KtoComponentRows:
    parent_revision: str
    parent_file: str
    parent_path: str | None = None

    def __call__(self, path: StoragePath) -> Iterator[dict[str, Any]]:
        # The selected file is data/train-*. Parent is staged at the source root.
        parent = StoragePath(self.parent_path) if self.parent_path is not None else path.parent.parent / self.parent_file
        with TemporaryDirectory() as directory, sqlite3.connect(directory + "/join.sqlite") as connection:
            connection.execute("PRAGMA cache_size = -4096")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                "CREATE TABLE matches (key TEXT PRIMARY KEY, component TEXT NOT NULL, "
                "expected INTEGER NOT NULL, observed INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute("CREATE TABLE provenance (key TEXT NOT NULL, parent_row INTEGER NOT NULL)")
            connection.execute("CREATE INDEX provenance_key ON provenance (key)")
            for index, row in enumerate(_rows(parent, MAX_PARENT_ROWS)):
                component = row.get("dataset")
                if component not in COMPONENTS:
                    raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "unknown_parent_component")
                chosen = _messages(row.get("chosen"))
                rejected = _messages(row.get("rejected"))
                if (
                    chosen[-1]["role"] != "assistant"
                    or rejected[-1]["role"] != "assistant"
                    or chosen[:-1] != rejected[:-1]
                ):
                    raise KtoComponentError(ImportFailureKind.SOURCE_DEFECT, "parent_preference_context_conflict")
                for messages, label in ((chosen, True), (rejected, False)):
                    key = _key(messages[:-1], messages[-1:], label)
                    existing = connection.execute("SELECT component FROM matches WHERE key = ?", (key,)).fetchone()
                    if existing is not None and existing[0] != component:
                        raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "ambiguous_component_content")
                    connection.execute(
                        "INSERT INTO matches (key, component, expected) VALUES (?, ?, 1) "
                        "ON CONFLICT(key) DO UPDATE SET expected = expected + 1",
                        (key, component),
                    )
                    connection.execute("INSERT INTO provenance VALUES (?, ?)", (key, index))
            for row in _rows(path, MAX_KTO_ROWS):
                key = _kto_key(row)
                if connection.execute("UPDATE matches SET observed = observed + 1 WHERE key = ?", (key,)).rowcount != 1:
                    raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "unmatched_kto_content_or_label")
            if connection.execute("SELECT 1 FROM matches WHERE expected != observed LIMIT 1").fetchone() is not None:
                raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "kto_parent_multiplicity_mismatch")
            if connection.execute("SELECT 1 FROM matches LIMIT 1").fetchone() is None:
                raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "empty_component_population")
            # No row is visible to selectors until whole-split validation succeeds.
            for row in _rows(path, MAX_KTO_ROWS):
                key = _kto_key(row)
                component = connection.execute("SELECT component FROM matches WHERE key = ?", (key,)).fetchone()[0]
                parent_rows = [
                    result[0]
                    for result in connection.execute(
                        "SELECT parent_row FROM provenance WHERE key = ? ORDER BY parent_row", (key,)
                    )
                ]
                yield {
                    **row,
                    "kto_component_provenance": {
                        "component": component,
                        "parent_dataset": "argilla/dpo-mix-7k",
                        "parent_revision": self.parent_revision,
                        "parent_file": self.parent_file,
                        "parent_rows": parent_rows,
                    },
                }


@dataclass(frozen=True)
class KtoComponentSelector:
    component: str

    def __call__(self, row: dict[str, Any], _root: StoragePath) -> bool:
        return row["kto_component_provenance"]["component"] == self.component


def component_inputs(*, component: str, kto_revision: str, parent_revision: str) -> RecipeInputs:
    """Declare both pinned files for existing shared acquisition and selection."""
    if component not in COMPONENTS:
        raise ValueError(f"Unknown KTO component: {component}")
    files = SourceFiles(
        (TRAIN_FILE,),
        SourceFormat.PARQUET,
        reader=KtoComponentRows(parent_revision, PARENT_FILE),
        selector=KtoComponentSelector(component),
    )
    return RecipeInputs(
        files,
        (
            HubDownload("trl-lib/kto-mix-14k", kto_revision, (TRAIN_FILE,)),
            HubDownload("argilla/dpo-mix-7k", parent_revision, (TRAIN_FILE,), subdirectory="component-parent"),
        ),
    )


def normalize_component_binary(row: RawRow) -> TaskSpec | ImportRejection:
    """Retain recovered acquisition evidence privately without changing KTO scoring."""
    task = preference_tasks.normalize_binary(row)
    if isinstance(task, ImportRejection):
        return task
    evidence = json.dumps(row.data["kto_component_provenance"], sort_keys=True, ensure_ascii=False).encode()
    resource = inline_resource("acquisition/kto_component_provenance.json", evidence)
    resources = task.resources.model_copy(update={"verifier": (*task.resources.verifier, resource)})
    return task.model_copy(update={"resources": resources})


COMPONENT_RUBRIC = ReviewRubric(
    id="kto-component-answerability",
    version="1",
    criteria=(
        "Read the complete public prompt messages; the labeled candidate completion remains private.",
        "The boolean label is an unpaired preference observation; do not invent a chosen/rejected counterpart.",
        "Assess public task coherence separately from candidate quality or the source preference label.",
        "Component membership is recovered from a fully validated exact-content join to the pinned parent "
        "mixture; use the private acquisition evidence, not guessed text or ordinal slices.",
        "Missing inputs and contradictions are task defects; an unbound reward model alone is not.",
    ),
)


def component_policy() -> TaskPolicy:
    return replace(preference_tasks.binary_policy(COMPONENT_RUBRIC), normalize=normalize_component_binary)
