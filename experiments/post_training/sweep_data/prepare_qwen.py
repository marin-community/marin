# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-render the task #202 sweep set for a Qwen3 model (task #230): same source rows, same order, Qwen's template.

Every training and validation row of the frozen #202 artifact is rebuilt from its source and rendered with
qwen_render (the model's chat template with a think block on every assistant turn; loss on what the model emits after
its generation prompt). Selection, held-out ids, seeds and row order are the #202 artifact's; nothing is re-sampled.
A row that no longer fits 32,768 tokens in the Qwen template is dropped (never truncated) and counted per family; the
remaining rows keep their #202 order, and the file is padded with zero-loss rows to a multiple of 16.

Sources:
- TMax and Kimi SWE-smith rows: the pinned HF datasets #202 read, keyed by trajectory name.
- Replay rows: re-read from their recorded HF file and position, checked against the recorded sha256 of the raw
  bytes, and passed through DataKit's row_to_chat_doc as #202 did. Harmony analysis becomes reasoning_content,
  commentary and final text become content, function calls become tool_calls; a developer message is rendered as a
  system message (Qwen's template has no developer role). Assistant turns the source's train_turns marks untrained
  stay masked.

Writes the #202 file set: train-prebuilt.jsonl.gz, validation-{tmax,swesmith}-prebuilt.jsonl.gz, train-index.jsonl,
token-audit.jsonl, row-probes.jsonl, heldout-ids.json, replay-rows.jsonl and manifest.json (with a v0 vs Qwen diff).

    python -m experiments.post_training.sweep_data.prepare_qwen --model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
        --output s3://marin-us-east-02a/marin/teacher-traces/mopd2/task230-qwen3-30b-sweep-prepared-<date>
"""

import argparse
import collections
import glob
import gzip
import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

import msgspec
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from marin.datakit.chat_normalize import ChatChannel, message_text
from marin.datakit.download.nemotron_sft_v3 import row_to_chat_doc
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.sweep_data.prepare_agentic import BATCH, KIMI, TMAX
from experiments.post_training.sweep_data.prepare_replay import read_line, source_assistant_flags
from experiments.post_training.sweep_data.qwen_render import (
    QWEN_MODELS,
    TOKENIZER_FILES,
    Rendered,
    every_turn_thinks,
    mask_turns,
    render,
    served_messages,
)
from experiments.post_training.sweep_data.render import kimi_messages, tmax_messages, training_row
from experiments.post_training.sweep_data.select import CONTEXT

SOURCE = "s3://marin-us-east-02a/marin/teacher-traces/mopd2/task202-sweep-prepared-20261016d"
SOURCE_MANIFEST_SHA256 = "2fbe5fe17710b98e127158350303cc0c8f8887a1d905843cf96bb31f55119bed"
SOURCE_FILES = ("train-index.jsonl", "token-audit.jsonl", "heldout-ids.json", "replay-rows.jsonl")
FAMILIES = ("tmax", "swesmith")


def read_source() -> tuple[dict, dict[str, bytes]]:
    root = StoragePath(SOURCE)
    manifest_bytes = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("The #202 manifest changed")
    manifest = json.loads(manifest_bytes)
    files = {}
    for name in SOURCE_FILES:
        files[name] = (root / name).read_bytes()
        expected = manifest["output_hashes"].get(name)
        if expected is not None and hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f"#202 {name} changed")
    return manifest, files


def agentic_sources() -> dict[str, tuple[str, dict]]:
    """trajectory name -> (family, source row), from the pinned datasets #202 read."""
    tmax_dir = snapshot_download(TMAX[0], repo_type="dataset", revision=TMAX[1], allow_patterns=[f"{TMAX[2]}/*"])
    kimi_dir = snapshot_download(KIMI[0], repo_type="dataset", revision=KIMI[1], allow_patterns=["data/*"])
    sources = {}
    for row in pq.read_table(glob.glob(f"{tmax_dir}/{TMAX[2]}/*.parquet")[0]).to_pylist():
        sources[row["metadata"]["trial_name"]] = ("tmax", row)
    kimi = pa.concat_tables([pq.read_table(path) for path in sorted(glob.glob(f"{kimi_dir}/data/*.parquet"))])
    for row in kimi.to_pylist():
        sources[row["trial_name"]] = ("swesmith", row)
    return sources


def agentic_messages(family: str, row: dict) -> tuple[list[dict], list[dict] | None]:
    """The #202 conversion of a source row (render.tmax_messages / kimi_messages)."""
    return tmax_messages(row) if family == "tmax" else (kimi_messages(row), None)


def harmony_to_messages(messages: list[Message]) -> tuple[list[dict], collections.Counter]:
    """Harmony messages as chat messages for the Qwen template; counts notable conversions."""
    notes = collections.Counter()
    converted = []
    for role, group in groupby(messages, key=lambda message: message.author.role):
        if role == Role.ASSISTANT:
            reasoning, content, calls, last = [], [], [], None
            for message in group:
                text = message_text(message)
                if message.recipient is not None:
                    if message.channel != ChatChannel.COMMENTARY or not message.recipient.startswith("functions."):
                        raise ValueError("Tool calls require commentary addressed to functions.<name>")
                    arguments = json.loads(text)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool-call arguments must be a JSON object")
                    name = message.recipient.removeprefix("functions.")
                    calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
                elif message.channel == ChatChannel.ANALYSIS:
                    if last == "content":
                        notes["reasoning_after_content_moved_first"] += 1
                    reasoning.append(text)
                    last = "reasoning"
                elif message.channel in (ChatChannel.COMMENTARY, ChatChannel.FINAL):
                    content.append(text)
                    last = "content"
                else:
                    raise ValueError(f"Unsupported assistant channel: {message.channel!r}")
            turn = {"role": "assistant", "content": "".join(content)}
            if reasoning:
                turn["reasoning_content"] = "\n\n".join(reasoning)
                notes["turns_with_several_reasoning_parts"] += len(reasoning) > 1
            if calls:
                turn["tool_calls"] = calls
            converted.append(turn)
            continue
        for message in group:
            text = message_text(message)
            if role == Role.TOOL:
                converted.append({"role": "tool", "content": text})
            elif role == Role.DEVELOPER:
                notes["developer_as_system"] += 1
                converted.append({"role": "system", "content": text})
            else:
                converted.append({"role": role.value, "content": text})
    return converted, notes


def replay_source(fs: HfFileSystem, entry: dict) -> tuple[list[dict], list[dict] | None, list[bool] | None, dict]:
    """Re-read one #202 replay row from its recorded position and turn it into chat messages."""
    if "byte_offset" in entry:
        found = read_line(fs, entry["file"], entry["byte_offset"], fs.info(entry["file"])["size"])
        if isinstance(found, str) or found[0] != entry["byte_offset"]:
            raise ValueError(f"Replay row moved: {entry['file']}@{entry['byte_offset']}")
        raw = found[1]
        row = msgspec.json.decode(raw)
    else:
        table = pq.ParquetFile(fs.open(entry["file"], "rb")).read_row_group(entry["row_group"])
        row = table.slice(entry["row_in_group"], 1).to_pylist()[0]
        raw = json.dumps(row, sort_keys=True, default=str).encode()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Replay row bytes changed: {entry['file']}")
    if row.get("tools") == "":
        # As in #202's sampler: DataKit json-decodes "" and quarantines the row; an empty string means no tools.
        row = row | {"tools": None}
    _, family, partition = entry["partition"].split("/")
    documents = row_to_chat_doc(row, family=family, partition_name=partition)
    if not documents:
        raise ValueError(f"DataKit now filters a #202 replay row: {entry['file']}")
    document = documents[0]
    trained = source_assistant_flags(row, document["source_train_turns"])
    if isinstance(trained, str):
        raise ValueError(f"Replay row train_turns: {trained}")
    messages, notes = harmony_to_messages([Message.from_dict(message) for message in document["messages"]])
    kwargs = json.loads(document["chat_template_kwargs"]) if document["chat_template_kwargs"] else {}
    tools = kwargs.pop("tools", None)
    if kwargs.get("enable_thinking") is False:
        notes["source_reasoning_off"] += 1
    return messages, tools, trained, dict(notes)


def record(rendered: Rendered, entry: dict) -> dict:
    return {"entry": entry, "ids": rendered.ids, "mask": rendered.mask}


def write_rows(path: Path, records: list[dict], pad: int) -> int:
    padding = -len(records) % BATCH
    empty = json.dumps({"input_ids": [pad] * CONTEXT, "loss_weights": [0] * CONTEXT}, separators=(",", ":"))
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for item in records:
            stream.write(json.dumps(training_row(item["ids"], item["mask"], CONTEXT, pad), separators=(",", ":")))
            stream.write("\n")
        for _ in range(padding):
            stream.write(empty + "\n")
    return padding


def verify_rows(path: Path, records: list[dict], padding: int, pad: int) -> int:
    """Re-read a written file; every row must equal its rendering; padding rows must be empty."""
    supervised = 0
    with gzip.open(path, "rt") as stream:
        lines = stream.read().splitlines()
    if len(lines) != len(records) + padding or len(lines) % BATCH:
        raise ValueError(f"{path.name}: row count")
    for number, line in enumerate(lines):
        row = json.loads(line)
        if number >= len(records):
            if any(row["loss_weights"]) or set(row["input_ids"]) != {pad}:
                raise ValueError(f"{path.name}: padding row {number} is not empty")
            continue
        expected = training_row(records[number]["ids"], records[number]["mask"], CONTEXT, pad)
        if row != expected:
            raise ValueError(f"{path.name}: row {number} differs from its rendering")
        supervised += sum(row["loss_weights"])
    return supervised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(QWEN_MODELS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="render only the first N rows of each split (local smoke test)")
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    revision = QWEN_MODELS[args.model]
    v0, source_files = read_source()
    index = [json.loads(line) for line in source_files["train-index.jsonl"].decode().splitlines()]
    audits = [json.loads(line) for line in source_files["token-audit.jsonl"].decode().splitlines()]
    validation = {family: [a for a in audits if a["split"] == f"validation-{family}"] for family in FAMILIES}
    if args.limit:
        index = index[: args.limit]
        validation = {family: rows[: args.limit] for family, rows in validation.items()}

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        tokenizer_hashes = {}
        for name in TOKENIZER_FILES:
            path = hf_hub_download(args.model, name, revision=revision, local_dir=directory / "tokenizer")
            tokenizer_hashes[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        native = tokenizer.chat_template
        template = every_turn_thinks(native)
        pad = tokenizer.pad_token_id
        sources = agentic_sources()
        fs = HfFileSystem()
        replay_entries = [entry for entry in index if entry["family"] == "replay"]
        with ThreadPoolExecutor(16) as pool:
            replay = dict(
                zip(
                    [entry["row"] for entry in replay_entries],
                    pool.map(lambda entry: replay_source(fs, entry), replay_entries),
                    strict=True,
                )
            )

        notes = collections.Counter()
        dropped = collections.Counter()

        def build(entry: dict) -> dict | None:
            if entry["family"] == "replay":
                messages, tools, trained, row_notes = replay[entry["row"]]
                notes.update(row_notes)
                rendered = render(tokenizer, template, messages, tools)
                if trained is not None:
                    rendered = mask_turns(rendered, trained)
            else:
                family, row = sources[entry["trajectory"]]
                if family != entry["family"]:
                    raise ValueError(f"Family mismatch for {entry['trajectory']}")
                messages, tools = agentic_messages(family, row)
                rendered = render(tokenizer, template, messages, tools)
                single_user = sum(message["role"] == "user" for message in messages) == 1
                if single_user and all(
                    (message.get("reasoning_content") or "").strip()
                    for message in messages
                    if message["role"] == "assistant"
                ):
                    native_text = tokenizer.apply_chat_template(served_messages(messages), tools=tools, tokenize=False)
                    if native_text != rendered.text:
                        raise ValueError(f"Option B differs from the native template on {entry['trajectory']}")
                    notes[f"{family}_single_user_full_reasoning_rows_byte_identical_to_native"] += 1
                notes[f"{family}_rows_more_than_one_user_message"] += not single_user
                notes[f"{family}_empty_reasoning_turns"] += sum(
                    1 for m in messages if m["role"] == "assistant" and not (m.get("reasoning_content") or "").strip()
                )
            if not rendered.supervised:
                raise ValueError(f"No trained tokens in {entry}")
            if len(rendered.ids) > CONTEXT:
                dropped[(entry.get("split", "train"), entry["family"])] += 1
                return None
            return record(rendered, entry)

        train = [item for item in map(build, index) if item is not None]
        heldout = {
            family: [build(entry | {"split": f"validation-{family}"}) for entry in validation[family]]
            for family in FAMILIES
        }
        heldout = {family: [item for item in rows if item is not None] for family, rows in heldout.items()}

        out = directory / "out"
        out.mkdir()
        counts = {}
        files = {"train-prebuilt.jsonl.gz": train} | {
            f"validation-{family}-prebuilt.jsonl.gz": heldout[family] for family in FAMILIES
        }
        for name, records in files.items():
            padding = write_rows(out / name, records, pad)
            supervised = verify_rows(out / name, records, padding, pad)
            if supervised != sum(sum(item["mask"][1:]) for item in records):
                raise ValueError(f"Supervised count changed in {name}")
            counts[name] = {"rows": len(records), "padding_rows": padding, "supervised_tokens": supervised}

        def tokens(item: dict) -> dict:
            return {"supervised_tokens": sum(item["mask"][1:]), "context_tokens": len(item["ids"])}

        new_index = []
        for row, item in enumerate(train):
            entry = {
                key: value for key, value in item["entry"].items() if key not in ("supervised_tokens", "context_tokens")
            }
            new_index.append(entry | {"row": row, "v0_row": item["entry"]["row"]} | tokens(item))
        (out / "train-index.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in new_index))
        audit_rows = [
            {key: value for key, value in item["entry"].items() if key not in ("supervised_tokens", "context_tokens")}
            | tokens(item)
            for split in [train, *heldout.values()]
            for item in split
            if "trajectory" in item["entry"]
        ]
        (out / "token-audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit_rows))
        probes = []
        for item in [*train[:2], *(heldout[family][0] for family in FAMILIES if heldout[family])]:
            spans, previous = [], None
            for token, weight in zip(item["ids"], item["mask"], strict=True):
                if weight != previous:
                    spans.append({"trained": bool(weight), "text": ""})
                    previous = weight
                spans[-1]["text"] += tokenizer.decode([token])
            probes.append(item["entry"] | tokens(item) | {"spans": spans})
        (out / "row-probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in probes))
        (out / "heldout-ids.json").write_bytes(source_files["heldout-ids.json"])
        (out / "replay-rows.jsonl").write_bytes(source_files["replay-rows.jsonl"])

        family_rows = collections.Counter(entry["family"] for entry in new_index)
        family_supervised = collections.Counter()
        family_context = collections.Counter()
        for entry in new_index:
            family_supervised[entry["family"]] += entry["supervised_tokens"]
            family_context[entry["family"]] += entry["context_tokens"]
        supervised = sum(family_supervised.values())
        train_counts = counts["train-prebuilt.jsonl.gz"]
        v0_counts = {
            "rows": v0["family_rows"],
            "supervised_tokens": v0["family_supervised_tokens"],
            "context_tokens": v0["family_context_tokens"],
            "validation_rows": v0["validation_rows"],
            "validation_supervised_tokens": v0["validation_supervised_tokens"],
        }
        manifest = {
            "task": 230,
            "description": (
                f"#202 LR-sweep set re-rendered for {args.model}: same source rows, held-out ids, seeds and order"
            ),
            "model": args.model,
            "model_revision": revision,
            "tokenizer_files_sha256": tokenizer_hashes,
            "pad_token_id": pad,
            "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "max_sequence_tokens": CONTEXT,
            "batch_size": BATCH,
            "training_rows": train_counts["rows"],
            "padding_rows": train_counts["padding_rows"],
            "steps_per_epoch": (train_counts["rows"] + train_counts["padding_rows"]) // BATCH,
            "supervised_tokens": train_counts["supervised_tokens"],
            "context_tokens": sum(family_context.values()),
            "family_rows": dict(family_rows),
            "family_supervised_tokens": dict(family_supervised),
            "family_context_tokens": dict(family_context),
            "family_supervised_share": {family: count / supervised for family, count in family_supervised.items()},
            "replay_to_agentic_ratio": (
                family_supervised["replay"] / (family_supervised["tmax"] + family_supervised["swesmith"])
            ),
            "validation_files": {family: f"validation-{family}-prebuilt.jsonl.gz" for family in FAMILIES},
            "validation_rows": {f: counts[f"validation-{f}-prebuilt.jsonl.gz"]["rows"] for f in FAMILIES},
            "validation_padding_rows": {
                f: counts[f"validation-{f}-prebuilt.jsonl.gz"]["padding_rows"] for f in FAMILIES
            },
            "validation_supervised_tokens": {
                f: counts[f"validation-{f}-prebuilt.jsonl.gz"]["supervised_tokens"] for f in FAMILIES
            },
            "validation_scope": v0["validation_scope"],
            "source": {
                "uri": SOURCE,
                "manifest_sha256": SOURCE_MANIFEST_SHA256,
                "rule": "every #202 row rebuilt from its source; no re-sampling; #202 order kept",
            },
            "v0_counts": v0_counts,
            "dropped_over_context": {f"{split}/{family}": n for (split, family), n in sorted(dropped.items())},
            "conversion_notes": dict(notes),
            "loss_mask": (
                "each assistant turn's output after the generation prompt '<|im_start|>assistant\\n<think>\\n': "
                "reasoning, '\\n</think>\\n\\n', content, tool calls and '<|im_end|>'; everything else is context; "
                "replay turns the source's train_turns marks untrained are context. "
                "loss_weights[i] scores the prediction of input_ids[i + 1]."
            ),
            "rendering": {
                "template": (
                    f"{args.model}@{revision} tokenizer_config chat_template with both think-block conditions made "
                    "unconditional (qwen_render.every_turn_thinks): every assistant turn carries a think block, "
                    "empty when the source has no reasoning"
                ),
                "served_as": (
                    "vLLM 0.25.1 with apply_chat_template (serve config sets no parser): tool-call arguments parsed to "
                    "objects before rendering, as vLLM's _postprocess_messages does"
                ),
                "tokenization": (
                    "context and trained segments tokenized separately (no special tokens added), as the prompt and "
                    "the model's output are at inference"
                ),
                "history_mismatch": (
                    "the native template drops reasoning from turns before the last real user message, so served "
                    "prompts lack earlier reasoning that these rows keep in context: all Kimi rows (terminus-2 user "
                    "turns) and TMax rows with more than one user message; the v0 rows carry the same mismatch"
                ),
                "tmax": "reasoning_content -> <think>; tool_calls -> <tool_call> JSON; tools in the system block",
                "swesmith": "inline <think> moved to reasoning_content as in #202; terminus-2 JSON content verbatim",
                "replay": (
                    "DataKit row_to_chat_doc as in #202; Harmony analysis -> reasoning_content (parts joined by a "
                    "blank line), commentary and final -> content, function calls -> tool_calls, developer -> system"
                ),
            },
            "test_exposed": False,
            "output_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.mkdirs()
        for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
            (output / path.name).write_bytes(path.read_bytes())
            if (
                path.name != "manifest.json"
                and hashlib.sha256((output / path.name).read_bytes()).hexdigest() != manifest["output_hashes"][path.name]
            ):
                raise ValueError(f"Upload hash mismatch: {path.name}")
        print("QWEN_PREPARED " + json.dumps({k: v for k, v in manifest.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    main()
