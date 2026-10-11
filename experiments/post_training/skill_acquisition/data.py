# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build immutable tool-call replay diagnostics from training-approved BFCL traces."""

import argparse
import gzip
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rolloutengine.tool_protocol import THINK_CLOSE, THINK_OPEN, TOOL_OPEN, ToolCallStatus, tool_call_regions
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ReplayCase:
    task_source_id: str
    source_file: str
    source_record: str
    source_phase: str
    model_source_identity: str
    token_mask_sha256: str
    prompt_token_ids: tuple[int, ...]
    prefix_sha256: str
    source_status: str


def diagnostic_task_split(task_ids: list[str], validation_tasks: int, salt: str) -> tuple[list[str], list[str]]:
    """Reserve task identities before examining trace failures."""
    if len(set(task_ids)) != len(task_ids) or not 0 < validation_tasks < len(task_ids):
        raise ValueError("Require unique task IDs and a nonempty train/validation split")
    ordered = sorted(task_ids, key=lambda task: (hashlib.sha256(f"{salt}:{task}".encode()).hexdigest(), task))
    return sorted(ordered[validation_tasks:]), sorted(ordered[:validation_tasks])


def replay_prefixes(ids: list[int], mask: list[int], tokenizer: Tokenizer) -> list[tuple[tuple[int, ...], str]]:
    """Retain unchanged token prefixes at outside-thinking repeated openers.

    Only the first malformed call per assistant span is considered. The prompt
    ends at its initial opener, before the rejected second opener. Earlier
    observations and assistant turns remain byte-for-byte token-identical.
    """
    if len(ids) != len(mask) or any(value not in (0, 1) for value in mask):
        raise ValueError("Input tokens and binary assistant masks must align")
    open_id = tokenizer.token_to_id(TOOL_OPEN)
    think_open = tokenizer.token_to_id(THINK_OPEN)
    think_close = tokenizer.token_to_id(THINK_CLOSE)
    header_id = tokenizer.token_to_id("<|start_header_id|>")
    if None in (open_id, think_open, think_close, header_id):
        raise ValueError("Tokenizer lacks required Snowball protocol tokens")
    result = []
    position = 0
    thinking = False
    while position < len(ids):
        token = ids[position]
        if token == header_id:
            thinking = False
        elif token == think_open:
            thinking = True
        elif token == think_close:
            thinking = False
        if not mask[position]:
            position += 1
            continue
        start = position
        while position < len(ids) and mask[position]:
            position += 1
        span = ids[start:position]
        text = tokenizer.decode(span, skip_special_tokens=False)
        regions = tool_call_regions(text, initial_thinking=thinking)
        for region in regions:
            if region.status != ToolCallStatus.REPEATED_OPENER:
                continue
            open_positions = [index for index, value in enumerate(span) if value == open_id]
            if len(open_positions) != text.count(TOOL_OPEN):
                raise ValueError("Decoded opening delimiters do not align with special token IDs")
            index = open_positions[text[: region.start].count(TOOL_OPEN)]
            if tokenizer.decode(span[: index + 1], skip_special_tokens=False) != text[: region.payload_start]:
                raise ValueError("Tool-call boundary does not preserve the decoded token prefix")
            result.append((tuple(ids[: start + index + 1]), region.status.value))
        # The next masked prompt includes its own chat header. Do not infer the
        # next assistant's thinking state from arbitrary strings in this span.
        thinking = False
    return result


def build_diagnostic(
    archives: list[Path],
    partition_file: Path,
    tokenizer_file: Path,
    output: Path,
    *,
    version: str,
    validation_tasks: int,
    max_prefix_tokens: int,
) -> dict:
    """Audit task identities, select replay cases, and publish a new local version."""
    if output.exists():
        raise FileExistsError(output)
    partition_bytes = partition_file.read_bytes()
    partition = json.loads(partition_bytes)
    approved = {task["source_id"]: task["digest"] for task in partition["complement"]}
    task_names = {task["name"]: task["source_id"] for task in partition["complement"]}
    parity = {task["source_id"] for task in partition["parity"]}
    if parity & approved.keys():
        raise ValueError("Training-approved tasks overlap BFCL-Parity")
    train_ids, validation_ids = diagnostic_task_split(list(approved), validation_tasks, version)
    validation_set = set(validation_ids)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    cases: dict[str, ReplayCase] = {}
    failures = []
    sources = []
    selected_rows = 0
    oversized_prefixes = 0
    for path in sorted(archives):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append({"path": str(path), "sha256": digest})
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json.gz"):
                    continue
                row = json.loads(gzip.decompress(archive.read(member)))
                task_name = row["trajectory"]["instance_id"]
                task_id = task_names.get(task_name)
                if task_id is None:
                    failures.append(
                        {
                            "file": path.name,
                            "record": row["record_id"],
                            "task": task_name,
                            "error": "task is not in the training-approved complement",
                        }
                    )
                    continue
                if task_id not in validation_set:
                    continue
                selected_rows += 1
                prompt = row["prompt"]["token_ids"]
                ids = prompt + row["response"]["token_ids"]
                mask = [0] * len(prompt) + row["response"]["loss_mask"]
                try:
                    prefixes = replay_prefixes(ids, mask, tokenizer)
                except ValueError as error:
                    failures.append(
                        {"file": path.name, "record": row["record_id"], "task": task_id, "error": str(error)}
                    )
                    continue
                for prefix, status in prefixes:
                    if len(prefix) > max_prefix_tokens:
                        oversized_prefixes += 1
                        continue
                    identity = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
                    token_mask_hash = hashlib.sha256(json.dumps([ids, mask], separators=(",", ":")).encode()).hexdigest()
                    candidate = ReplayCase(
                        task_id,
                        path.name,
                        row["record_id"],
                        row["phase"],
                        row["provenance"]["model_source_identity"],
                        token_mask_hash,
                        prefix,
                        identity,
                        status,
                    )
                    if task_id not in cases or candidate.prefix_sha256 < cases[task_id].prefix_sha256:
                        cases[task_id] = candidate
    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures}, indent=2))
    if not sources or not cases:
        raise ValueError("No diagnostic cases found in the audited source files")
    manifest = {
        "version": version,
        "source_partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "source_dataset_commit": partition["dataset_commit"],
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_format": "native trajectory retention archives",
        "source_identity_limit": (
            "task names audited against pinned partition; source records do not contain task-content hashes"
        ),
        "max_prefix_tokens": max_prefix_tokens,
        "train_task_ids": train_ids,
        "diagnostic_reserved_task_ids": validation_ids,
        "parity_task_ids": sorted(parity),
        "selected_rows": selected_rows,
        "oversized_prefixes": oversized_prefixes,
        "diagnostic_cases": len(cases),
        "sources": sources,
    }
    output.mkdir(parents=True)
    case_file = output / "cases.jsonl"
    with case_file.open("x") as stream:
        for task_id in sorted(cases):
            stream.write(json.dumps(asdict(cases[task_id]), separators=(",", ":")) + "\n")
    manifest["cases_sha256"] = hashlib.sha256(case_file.read_bytes()).hexdigest()
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--validation-tasks", type=int, required=True)
    parser.add_argument("--max-prefix-tokens", type=int, required=True)
    args = parser.parse_args()
    manifest = build_diagnostic(
        args.archive,
        args.partition,
        args.tokenizer,
        args.output,
        version=args.version,
        validation_tasks=args.validation_tasks,
        max_prefix_tokens=args.max_prefix_tokens,
    )
    print(json.dumps({key: manifest[key] for key in ("version", "diagnostic_cases", "selected_rows", "cases_sha256")}))


if __name__ == "__main__":
    main()
