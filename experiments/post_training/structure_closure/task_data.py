# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish generated closure cases as assistant-prefix TaskSpecs."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from transformers import PreTrainedTokenizerFast
from verifyit.modes.grade_structure_closure import validate_structure_closure
from verifyit.spec import StructureClosureSpec
from verifyit.tool_call_markup import TOOL_CLOSE, TOOL_OPEN

from experiments.post_training.structure_closure.benchmark import (
    CLOSE_REQUIRED,
    TASK_ID_PREFIX,
    ClosureCase,
    Family,
    Split,
    closure_format,
)
from experiments.post_training.structure_closure.token_data import (
    ToolPrompt,
    native_tool_case,
    native_tool_text,
    write_result_tool,
)

IMPORTER_REVISION = "1"


def case_task(case: ClosureCase, source: Source, *, tools: tuple[FunctionDefinition, ...] = ()) -> TaskSpec:
    """Keep task content in messages and the completion contract in the private grader."""
    spec = StructureClosureSpec(
        prefix=case.prefix,
        expected=case.expected,
        format=closure_format(case.family),
    )
    validate_structure_closure(spec)
    return TaskSpec(
        id=TASK_ID_PREFIX + case.case_id,
        source=source,
        context=ConversationInput(
            events=(
                TextMessage(role="user", content=case.instruction),
                TextMessage(role="assistant", content=case.prefix),
            )
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        final_tools=tools,
        grader=verifyit_package(spec).grader,
        tags=(f"family:{case.family}", f"control:{case.control}", f"split:{case.split}"),
    )


def paired_task_probes(case: ClosureCase, source: Source, tokenizer: PreTrainedTokenizerFast) -> list[TaskSpec]:
    """Build the four declaration/serialization arms without requiring a live tool."""
    if case.family != Family.TOOL or case.control != CLOSE_REQUIRED:
        raise ValueError("Paired probes require a close-required tool case")
    body = json.loads(case.expected.removeprefix(TOOL_OPEN).removesuffix(TOOL_CLOSE))
    native_prefix = native_tool_text(body, tokenizer, instruction=case.instruction, tools=None).removesuffix(TOOL_CLOSE)
    definition = FunctionDefinition.model_validate(write_result_tool()["function"])
    tasks = []
    for declaration, tools in (("undeclared", ()), ("declared", (definition,))):
        for serialization, prefix in (("compact", case.prefix), ("native", native_prefix)):
            variant = f"{declaration}-{serialization}"
            updated = replace(
                case,
                case_id=hashlib.sha256(f"{case.case_id}/{variant}".encode()).hexdigest(),
                prefix=prefix,
                expected=prefix + TOOL_CLOSE,
            )
            task = case_task(updated, source, tools=tools)
            tasks.append(task.model_copy(update={"tags": (*task.tags, f"probe:{variant}")}))
    return tasks


def prepare_tasks(cases_directory: Path, output: Path, *, tool_prompt: ToolPrompt, initializer: Path | None) -> dict:
    """Validate every source row before writing either immutable task split."""
    if output.exists():
        raise FileExistsError(output)
    tokenizer = None
    if tool_prompt == ToolPrompt.NATIVE_DECLARED:
        if initializer is None:
            raise ValueError("Native serialization requires an explicit local tokenizer")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(initializer, local_files_only=True)
    manifest_path = cases_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payloads = {}
    failures = []
    counts = Counter()
    parent_ids: dict[Split, set[str]] = {}
    identities: set[str] = set()
    for split in Split:
        path = cases_directory / f"{split}.jsonl"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["hashes"][split]:
            raise ValueError(f"Source hash mismatch: {split}")
        tasks = []
        parent_ids[split] = set()
        for number, line in enumerate(payload.splitlines(), 1):
            try:
                raw = json.loads(line)
                case = ClosureCase(**(raw | {"family": Family(raw["family"])}))
                if case.split != split or case.case_id in identities:
                    raise ValueError("Wrong split or duplicate case identity")
                identities.add(case.case_id)
                tools = ()
                if tokenizer is not None and case.family in (Family.TOOL, Family.REPEATED_OPENER):
                    case = native_tool_case(case, tokenizer)
                    tools = (FunctionDefinition.model_validate(write_result_tool()["function"]),)
                task = case_task(
                    case,
                    Source(
                        dataset="structure-closure",
                        revision=digest,
                        row=f"{split}.jsonl:{number}",
                        importer_revision=IMPORTER_REVISION,
                    ),
                    tools=tools,
                )
                tasks.append(task.model_dump_json() + "\n")
                counts[f"{split}/{case.family}/{case.control}"] += 1
                parent_ids[split].add(case.task_id)
            except (KeyError, TypeError, ValueError) as error:
                failures.append({"split": split, "line": number, "error": str(error)})
        payloads[split] = "".join(tasks).encode()
    if parent_ids[Split.TRAIN] & parent_ids[Split.VALIDATION]:
        failures.append({"error": "Train and validation contain the same parent task"})
    output.mkdir(parents=True)
    (output / "conversion-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise ValueError(f"{len(failures)} task conversion failures; see {output / 'conversion-failures.json'}")
    hashes = {}
    for split, payload in payloads.items():
        (output / f"{split}.jsonl").write_bytes(payload)
        hashes[split] = hashlib.sha256(payload).hexdigest()
    report = {
        "status": "tasks-validated",
        "hashes": hashes,
        "counts": dict(counts),
        "tool_prompt": tool_prompt,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-prompt", type=ToolPrompt, choices=list(ToolPrompt), required=True)
    parser.add_argument("--initializer", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_tasks(args.cases, args.output, tool_prompt=args.tool_prompt, initializer=args.initializer)))
