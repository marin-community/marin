# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the LR-sweep replay slice from DataKit's nemotron_sft_v3 sources, rendered for Snowball v0.

This is a seeded sampled subset read directly from the pinned Hugging Face files that DataKit registers, not the
materialized DataKit chain (which does not exist in marin-us-east-02a). Each row passes through DataKit's own
``row_to_chat_doc`` (same parsing, filters and validation), is converted to inference messages with DataKit's
renderer helper, and is rendered with v0's chat template. Loss is on assistant turns only.

Partition quotas of supervised tokens are proportional to NEMOTRON_SFT_V3_TOKEN_COUNTS_B. JSONL rows are drawn by
seeded random byte offsets (seek, skip to the next newline, take that line); parquet rows by a seeded random row group
and row. Every kept row records its partition, file, position and the sha256 of its raw source bytes.
"""

import argparse
import collections
import hashlib
import json
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from marin.datakit.chat_render import _inference_messages
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.download.nemotron_sft_v3 import NEMOTRON_SFT_V3_REPOSITORIES, _parse_messages, row_to_chat_doc
from marin.datakit.sft_sources import NEMOTRON_SFT_V3_TOKEN_COUNTS_B
from openai_harmony import Message
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sweep_data.prepare_agentic import (
    BATCH,
    PAD_TOKEN,
    STUDENT_MODEL,
    STUDENT_REVISION,
    load_tokenizer,
    output_hashes,
    upload_checked,
    verify_rows,
    write_rows,
)
from experiments.post_training.sweep_data.select import CONTEXT

SEED = 20261016
REPLAY_FRACTION_OF_B = 0.25
# Its prompts are withheld in the source and restored from seed datasets by a separate DataKit step,
# so single rows cannot be sampled from the raw file.
EXCLUDED = {"nemotron_sft_v3/instruction_following_chat_v3/chat": "prompts restored by a separate DataKit step"}
MAX_ATTEMPTS_PER_ROW = 50
LINE_CHUNK = 1 << 20
# A line longer than this cannot fit 32,768 tokens. Some source files hold one huge malformed line; capping
# both the skipped span and the taken line keeps such a line from attracting most draws or being read in full.
MAX_LINE_BYTES = 4 << 20


def quotas_for(target: int) -> dict[str, int]:
    """Supervised-token quota per partition: `target` split in proportion to NEMOTRON_SFT_V3_TOKEN_COUNTS_B, min 1."""
    weights = {name: size for name, size in NEMOTRON_SFT_V3_TOKEN_COUNTS_B.items() if name not in EXCLUDED}
    total = sum(weights.values())
    return {name: max(1, round(target * size / total)) for name, size in sorted(weights.items())}


def partition_files(fs: HfFileSystem, name: str) -> tuple[str, str, list[tuple[str, int]]]:
    _, family, partition = name.split("/")
    repository, revision, partitions = NEMOTRON_SFT_V3_REPOSITORIES[family]
    pattern = f"datasets/{repository}@{revision}/{partitions[partition]}"
    files = sorted((path, info["size"]) for path, info in fs.glob(pattern, detail=True).items())
    if not files:
        raise ValueError(f"No files for {name}: {pattern}")
    return family, partition, files


def bounded_line(stream, limit: int) -> bytes:
    """Read from the current position through the next newline, at most `limit` bytes; seek back past the newline."""
    start = stream.tell()
    data = b""
    while len(data) < limit:
        chunk = stream.read(min(LINE_CHUNK, limit - len(data)))
        if not chunk:
            break
        end = chunk.find(b"\n")
        if end >= 0:
            data += chunk[: end + 1]
            break
        data += chunk
    stream.seek(start + len(data))
    return data


def read_line(fs: HfFileSystem, path: str, offset: int, size: int) -> tuple[int, bytes] | str:
    """Return (start, bytes) of the first full line that starts at or after `offset`, or a rejection reason."""
    with fs.open(path, "rb", block_size=LINE_CHUNK) as stream:
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n" and not bounded_line(stream, MAX_LINE_BYTES).endswith(b"\n"):
                return "landed_in_oversized_line"
        start = stream.tell()
        if start >= size:
            return "past_last_line"
        line = bounded_line(stream, MAX_LINE_BYTES)
    if not line.endswith(b"\n") and start + len(line) < size:
        return "oversized_line"
    return start, line


class PartitionSampler:
    def __init__(self, fs: HfFileSystem, name: str, rng: random.Random):
        self.fs, self.name, self.rng = fs, name, rng
        self.family, self.partition, self.files = partition_files(fs, name)
        self.parquet = self.files[0][0].endswith(".parquet")
        self.weights = [size for _, size in self.files]
        if self.parquet:
            self.metadata = {path: pq.ParquetFile(fs.open(path, "rb")).metadata for path, _ in self.files}
            self.weights = [self.metadata[path].num_rows for path, _ in self.files]
        self.decoder = msgspec.json.Decoder()

    def draw(self) -> tuple[dict, dict] | str:
        """Draw one raw row; return (row, position) or a rejection reason."""
        path, size = self.rng.choices(self.files, weights=self.weights)[0]
        if self.parquet:
            metadata = self.metadata[path]
            group = self.rng.choices(
                range(metadata.num_row_groups),
                weights=[metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)],
            )[0]
            index = self.rng.randrange(metadata.row_group(group).num_rows)
            table = pq.ParquetFile(self.fs.open(path, "rb")).read_row_group(group)
            row = table.slice(index, 1).to_pylist()[0]
            raw = json.dumps(row, sort_keys=True, default=str).encode()
            return row, {"file": path, "row_group": group, "row": index, "sha256": hashlib.sha256(raw).hexdigest()}
        found = read_line(self.fs, path, self.rng.randrange(size), size)
        if isinstance(found, str):
            return found
        start, line = found
        row = self.decoder.decode(line)
        return row, {"file": path, "byte_offset": start, "sha256": hashlib.sha256(line).hexdigest()}


def to_rendered(tokenizer, sampler: PartitionSampler, row: dict) -> tuple[list[int], list[int]] | str:
    """Run DataKit's transform and render with DataKit's Marin template; return a rejection reason on failure.

    v0's template is this template with one default changed (no enable_thinking means "/think"); rendering
    with DataKit's keeps rows that set no reasoning mode (e.g. reasoning_off) without a reasoning header.
    """
    documents = row_to_chat_doc(row, family=sampler.family, partition_name=sampler.partition)
    if not documents:
        return "datakit_filtered"
    document = documents[0]
    trained_turns = source_assistant_flags(row, document["source_train_turns"])
    if isinstance(trained_turns, str):
        return trained_turns
    messages = list(_inference_messages([Message.from_dict(message) for message in document["messages"]]))
    for message in messages:
        if message["role"] == "assistant" and not message["tool_calls"]:
            del message["tool_calls"]
    kwargs = json.loads(document["chat_template_kwargs"]) if document["chat_template_kwargs"] else {}
    tools = kwargs.pop("tools", None)
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=MARIN_CHAT_TEMPLATE,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    if trained_turns is not None:
        spans = [index for index in range(len(mask)) if mask[index] and (index == 0 or not mask[index - 1])]
        if len(spans) != len(trained_turns):
            return "train_turns_alignment_failed"
        for start, trained in zip(spans, trained_turns, strict=True):
            if not trained:
                end = start
                while end < len(mask) and mask[end]:
                    mask[end] = 0
                    end += 1
    if not any(mask):
        return "no_assistant_tokens"
    if len(ids) > CONTEXT:
        return "over_context"
    return ids, mask


def source_assistant_flags(row: dict, train_turns: list[bool] | None) -> list[bool] | None | str:
    """Map the source's per-message train_turns to one flag per rendered assistant turn.

    Consecutive source assistant messages render as one turn, so they must agree. Returns None when the source
    marks nothing, or a rejection reason.
    """
    if train_turns is None:
        return None
    roles = [message["role"] for message in _parse_messages(row["messages"])]
    flags: list[bool] = []
    previous = None
    for role, trained in zip(roles, train_turns, strict=True):
        if role == "assistant":
            if previous == "assistant":
                if flags[-1] != trained:
                    return "train_turns_split_within_a_turn"
            else:
                flags.append(trained)
        previous = role
    return flags


def sample_partition(tokenizer, fs: HfFileSystem, name: str, quota: int) -> tuple[list[dict], dict]:
    rng = random.Random(f"{SEED}:{name}")
    sampler = PartitionSampler(fs, name, rng)
    kept, seen, rejected, normalized = [], set(), collections.Counter(), collections.Counter()
    supervised = 0
    attempts = 0
    while supervised < quota:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_ROW * (len(kept) + 1):
            raise ValueError(f"{name}: too many rejected draws ({dict(rejected)})")
        try:
            drawn = sampler.draw()
        except (msgspec.DecodeError, UnicodeError, ValueError):
            rejected["undecodable_line"] += 1
            continue
        if isinstance(drawn, str):
            rejected[drawn] += 1
            continue
        row, position = drawn
        key = (position["file"], position.get("byte_offset"), position.get("row_group"), position.get("row"))
        if key in seen:
            rejected["duplicate_draw"] += 1
            continue
        seen.add(key)
        if row.get("tools") == "":
            # DataKit's _tool_definitions json-decodes "" and quarantines the row; an empty string means no tools.
            row = row | {"tools": None}
            normalized["empty_tools_string_as_none"] += 1
        result = to_rendered(tokenizer, sampler, row)
        if isinstance(result, str):
            rejected[result] += 1
            continue
        ids, mask = result
        record = {"family": "replay", "task": name, "trajectory": f"{name}#{len(kept)}", "ids": ids, "mask": mask}
        record["position"] = position | {"partition": name}
        kept.append(record)
        supervised += sum(mask[1:])
    return kept, {
        "quota_supervised_tokens": quota,
        "supervised_tokens": supervised,
        "context_tokens": sum(len(record["ids"]) for record in kept),
        "rows": len(kept),
        "draws": attempts,
        "rejected": dict(rejected),
        "normalized": dict(normalized),
        "files": [{"path": path, "bytes": size} for path, size in sampler.files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentic", required=True, help="prepared agentic prefix; B is its supervised_tokens")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    agentic = json.loads((StoragePath(args.agentic) / "manifest.json").read_text())
    if agentic["student_revision"] != STUDENT_REVISION:
        raise ValueError("Agentic data was prepared for another student")
    target = round(REPLAY_FRACTION_OF_B * agentic["supervised_tokens"])
    quotas = quotas_for(target)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "tokenizer").mkdir()
        tokenizer = load_tokenizer(directory / "tokenizer", agentic["tokenizer_files_sha256"])
        fs = HfFileSystem()

        def run(name: str) -> tuple[str, list[dict], dict]:
            records, report = sample_partition(tokenizer, fs, name, quotas[name])
            print(
                "REPLAY_PARTITION " + json.dumps({"name": name} | {k: v for k, v in report.items() if k != "files"}),
                flush=True,
            )
            return name, records, report

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(run, sorted(quotas)))
        records = [record for _, partition_records, _ in results for record in partition_records]
        random.Random(SEED).shuffle(records)
        out = directory / "out"
        out.mkdir()
        rows, padding = write_rows(out / "replay-prebuilt.jsonl.gz", records)
        supervised = verify_rows(out / "replay-prebuilt.jsonl.gz", records, padding)
        (out / "replay-rows.jsonl").write_text(
            "".join(
                json.dumps(
                    record["position"]
                    | {"supervised_tokens": sum(record["mask"][1:]), "context_tokens": len(record["ids"])}
                )
                + "\n"
                for record in records
            )
        )
        manifest = {
            "task": 202,
            "component": "replay",
            "source": (
                "DataKit nemotron_sft_v3 (marin.datakit.sft_sources), "
                "pinned HF revisions from NEMOTRON_SFT_V3_REPOSITORIES"
            ),
            "sampled_subset": (
                "a seeded sampled subset read from the pinned HF files through DataKit's row_to_chat_doc; "
                "not the materialized DataKit chain"
            ),
            "excluded_partitions": EXCLUDED,
            "seed": SEED,
            "student_model": STUDENT_MODEL,
            "student_revision": STUDENT_REVISION,
            "pad_token_id": PAD_TOKEN,
            "max_sequence_tokens": CONTEXT,
            "batch_size": BATCH,
            "agentic_manifest_sha256": (
                hashlib.sha256((StoragePath(args.agentic) / "manifest.json").read_bytes()).hexdigest()
            ),
            "agentic_supervised_tokens_B": agentic["supervised_tokens"],
            "target_supervised_tokens": target,
            "rows": rows,
            "padding_rows": padding,
            "supervised_tokens": supervised,
            "context_tokens": sum(len(record["ids"]) for record in records),
            "achieved_ratio_to_B": supervised / agentic["supervised_tokens"],
            "partition_quotas": quotas,
            "partitions": {name: report for name, _, report in results},
            "rendering": (
                "DataKit row_to_chat_doc -> chat_render._inference_messages -> DataKit's MARIN_CHAT_TEMPLATE with v0's "
                "tokenizer; reasoning kept (analysis channel rendered as <|start_think|>...<|end_think|>); per-row "
                "chat_template_kwargs. v0's chat_template.jinja differs from it only in defaulting a missing "
                "enable_thinking to '/think'; DataKit's template omits the reasoning header there, as DataKit renders "
                'it. A source tools value of "" is passed to DataKit as absent (DataKit at this commit would '
                'quarantine the row on json.loads(""); this affects science_v2 among others). '
                f"Over-context rows and draws landing in lines over {MAX_LINE_BYTES} bytes are skipped. A row's "
                "draw probability is proportional to the byte length of the line before it (capped), not exactly "
                "uniform per row"
            ),
            "loss_mask": (
                "assistant turns only (template generation blocks), minus assistant turns the source's train_turns "
                "marks untrained; loss_weights[i] scores input_ids[i + 1]"
            ),
            "output_hashes": output_hashes(out),
        }
        upload_checked(out, output, manifest)
        print(
            "REPLAY_PREPARED " + json.dumps({k: v for k, v in manifest.items() if k not in ("partitions",)}), flush=True
        )


if __name__ == "__main__":
    main()
