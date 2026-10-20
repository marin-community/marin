# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import native structure-closure rows as token-preserving current TaskSpecs."""

import json

from verifyit.closure_scoring import ClosureFormat
from verifyit.json_objects import unique_object
from verifyit.spec import StructureClosureSpec

from taskcompendium.convert.answers import unsupported
from taskcompendium.grader import verifyit_package
from taskcompendium.models import Source, TaskResource, TaskSpec, TokenizedInput
from taskcompendium.pipeline.models import ImportRejection, NormalizationChange, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes


def structure_closure_task(
    row: RawRow,
    *,
    tokenizer_model: str,
    tokenizer_revision: str,
    terminal_markers: tuple[str, ...],
) -> NormalizedTask | ImportRejection:
    """Convert a lowered native row without re-tokenizing or exposing its grader.

    The caller supplies the pinned tokenizer and source-row identity. Native
    identity changes remain audit metadata; the raw row retains the original
    serialized contract. Neither enters the semantic resources used to deduplicate
    aliases. Original verifier resources remain private and unchanged.
    """
    document = json.loads(row.data["lowered_task_spec"], object_pairs_hook=unique_object)
    native = document["task"]
    if native["source"]["dataset"] != "structure-closure":
        return unsupported("native_task_family", "Only structure-closure tasks use this converter")
    if native["grader"] != {"kind": "session"} or document["session"]["task_session"] != "structure_closure":
        return unsupported("native_session", "Expected the native structure_closure session grader")
    if document["runtime"] != {"task_machine": None, "verifier_machine": None}:
        return unsupported("native_machine", "Structure closure requires a machine-free source contract")
    if document["session"]["max_turns"] != 1:
        return unsupported("native_turns", "Structure closure requires exactly one model turn")
    configs = [resource for resource in native["resources"]["verifier"] if resource["path"] == "config.json"]
    if len(configs) != 1:
        return unsupported("native_config", "Expected exactly one native config.json verifier resource")
    config = json.loads(resource_bytes(TaskResource.model_validate(configs[0])), object_pairs_hook=unique_object)
    extras = config["extras"]
    grader = verifyit_package(
        StructureClosureSpec(
            prefix=extras["prefix"],
            expected=extras["expected"],
            format=ClosureFormat(extras["format"]),
            terminal_markers=terminal_markers,
        )
    ).grader
    task = TaskSpec.model_validate(
        {
            **{key: value for key, value in native.items() if key != "schema_version"},
            "id": row.id,
            "source": row.source,
            "context": TokenizedInput(
                events=native["context"]["events"],
                token_ids=extras["prompt_token_ids"],
                tokenizer_model=tokenizer_model,
                tokenizer_revision=tokenizer_revision,
            ),
            "grader": grader,
        }
    )
    original_source = Source.model_validate(native["source"])
    changes = []
    if native["id"] != row.id:
        changes.append(
            NormalizationChange(
                field="id",
                original=native["id"],
                replacement=row.id,
                reason="Use the curation source-row identity; the native identity remains in the raw-row audit.",
            )
        )
    if original_source != row.source:
        changes.append(
            NormalizationChange(
                field="source",
                original=original_source.model_dump_json(),
                replacement=row.source.model_dump_json(),
                reason="Pin the ingested release and source row; native provenance remains in the raw-row audit.",
            )
        )
    return NormalizedTask(task, tuple(changes))
