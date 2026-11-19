# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind pinned raw OpenCode questions to frozen cache rows and repair only their wrappers."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.chat_normalize import normalize_chat_record
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.nemotron_sft_v3 import row_to_chat_doc
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_filters import BOS, EOS, MAX_TOKENS, MESSAGE, unquote_opencode_prompt
from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned, tokenizer_for
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json

INDEX_SCHEMA = pa.schema(
    [
        ("native_sha256", pa.string()),
        ("stable_row_id", pa.string()),
        ("normalized_id", pa.string()),
        ("question", pa.large_string()),
    ]
)
BRIDGE_SCHEMA = pa.schema(
    [
        ("row", pa.int64()),
        ("native_sha256", pa.string()),
        ("stable_row_ids", pa.list_(pa.string())),
        ("question", pa.large_string()),
    ]
)

# The raw-row adapter chain the index renders. If these files change between the
# index build and the frozen caches, every bridge row silently becomes unjoined
# and the repair no-ops, so every recipe must pin their bytes.
ADAPTER_PIN_FILES = (
    "lib/marin/src/marin/datakit/chat_normalize.py",
    "lib/marin/src/marin/datakit/chat_render.py",
    "lib/marin/src/marin/datakit/download/nemotron_sft_v3.py",
)


def require_adapter_pins(recipe: dict) -> None:
    missing = [name for name in ADAPTER_PIN_FILES if name not in recipe["implementation"]]
    if missing:
        raise ValueError("Opencode recipe implementation pin omits the adapter chain: " + ", ".join(missing))


def canonical_native(text: str) -> str:
    return text.strip().removeprefix(BOS).removesuffix(EOS).strip()


def native_identity(text: str) -> str:
    return hashlib.sha256(canonical_native(text).encode()).hexdigest()


def parquet_rows(uri: str) -> Iterator[dict]:
    with StoragePath(uri).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def write_parquet(uri: str, rows: list[dict], schema: pa.Schema) -> dict:
    with StoragePath(uri).open("wb") as out:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out, compression="zstd")
    return {"uri": uri, "sha256": hashlib.sha256(StoragePath(uri).read_bytes()).hexdigest(), "rows": len(rows)}


def index_part(job: dict) -> dict:
    configure_storage()
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    rows = []
    counts = Counter()
    fixtures = []
    for record in parquet_rows(job["input"]["uri"]):
        raw = json.loads(record["raw_json"])
        counts["raw_rows"] += 1
        converted = row_to_chat_doc(raw, family="opencode_v1", partition_name=job["partition"])
        if len(converted) != 1:
            counts["adapter_rejected"] += 1
            continue
        try:
            doc = normalize_chat_record(converted[0], "messages", "id")
        except ValueError as error:
            counts["normalization_rejected/" + type(error).__name__] += 1
            continue
        text = render_chat_record(doc)["text"]
        if tokenizer.normalizer is not None:
            text = tokenizer.normalizer.normalize_str(text)
        question = raw.get("question")
        if not isinstance(question, str):
            question = None
            counts["missing_question"] += 1
        stable = str(raw.get("uuid") or raw.get("id") or doc["id"])
        rows.append(
            {
                "native_sha256": native_identity(text),
                "stable_row_id": stable,
                "normalized_id": doc["id"],
                "question": question,
            }
        )
        counts["indexed_rows"] += 1
        if question is not None and len(question) < 10000:
            for match in MESSAGE.finditer(text):
                content = match[2]
                if match[1] != "user" or len(content) > 24000:
                    continue
                changed = unquote_opencode_prompt(content, question) != content
                category = "changed" if changed else "quoted_subagent" if content.startswith('"') else "other"
                category += "_" + str(content.count("<note>"))
                if len(fixtures) < 12 and category not in {x["category"] for x in fixtures}:
                    fixtures.append(
                        {"stable_row_id": stable, "question": question, "content": content, "category": category}
                    )
    pin = write_parquet(job["output"], rows, INDEX_SCHEMA)
    return {"partition": job["partition"], "pin": pin, "counts": dict(counts), "fixtures": fixtures}


@cache
def question_lookup(uri: str, sha: str) -> dict[str, tuple[str | None, list[str], str]]:
    groups = defaultdict(list)
    for row in pq.read_table(pa.BufferReader(read_pinned(uri, sha))).to_pylist():
        groups[row["native_sha256"]].append(row)
    result = {}
    for key, rows in groups.items():
        questions = {r["question"] for r in rows}
        ids = sorted({r["stable_row_id"] for r in rows})
        if len(questions) == 1 and None not in questions:
            result[key] = (next(iter(questions)), ids, "joined")
        else:
            result[key] = (None, ids, "missing_raw_question" if questions == {None} else "ambiguous_raw_question")
    return result


def bridge_part(job: dict) -> dict:
    configure_storage()
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    lookup = question_lookup(job["index"]["uri"], job["index"]["sha256"])
    store = JaggedArrayStore.open(prefix_join(job["input_path"], "input_ids"), mode="r", dtype=np.int32)
    if len(store) != job["expected_rows"]:
        raise ValueError("Frozen original cache row count changed")
    rows = []
    counts = Counter()
    digest = hashlib.sha256()
    for start in range(job["row_start"], job["row_stop"], 32):
        indices = list(range(start, min(start + 32, job["row_stop"])))
        for i, tokens in zip(indices, store.get_batch_sync(indices), strict=True):
            digest.update(tokens.astype("<i4", copy=False).tobytes())
            key = native_identity(tokenizer.decode(tokens.tolist(), skip_special_tokens=False))
            question, ids, reason = lookup.get(key, (None, [], "native_fingerprint_not_found"))
            counts["joined" if question is not None else "unjoined"] += 1
            counts["reason/" + reason] += 1
            rows.append({"row": i, "native_sha256": key, "stable_row_ids": ids, "question": question})
    if len(rows) != job["row_stop"] - job["row_start"]:
        raise ValueError("Bridge omits cache rows")
    return {
        "input_path": job["input_path"],
        "source": job["source"],
        "row_start": job["row_start"],
        "row_stop": job["row_stop"],
        "pin": write_parquet(job["output"], rows, BRIDGE_SCHEMA),
        "counts": dict(counts),
        "input_ids_sha256": digest.hexdigest(),
    }


def retained_rows(uri: str, input_key: str, output_key: str) -> dict[int, int]:
    result = {}
    with StoragePath(uri).open("rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["stage"] == "retained":
                index = int(row[output_key])
                if index in result:
                    raise ValueError("Duplicate retained row index")
                result[index] = int(row[input_key])
    if set(result) != set(range(len(result))):
        raise ValueError("Non-contiguous retained row mapping")
    return result


def repair_text(text: str, question: str) -> tuple[str, int]:
    pieces = []
    cursor = 0
    changed = 0
    for match in MESSAGE.finditer(text):
        if match[1] != "user":
            continue
        before = match[2]
        after = unquote_opencode_prompt(before, question)
        if before == after:
            continue
        pieces.extend((text[cursor : match.start(2)], after))
        cursor = match.end(2)
        changed += 1
    if not changed:
        return text, 0
    pieces.append(text[cursor:])
    return "".join(pieces), changed


def repair_part(job: dict) -> dict:
    configure_storage()
    source = job["source"]
    base = job["base_shard"]
    root = StoragePath(job["output_prefix"])
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["complete"] and receipt["job"] == job:
            return {
                "source": source,
                "base": base,
                "pin": {"uri": str(receipt_path), "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
            }
        raise ValueError("Existing question repair receipt differs")
    gate = pinned_json({"uri": base["receipt_uri"], "sha256": base["receipt_sha256"]})
    scripted = gate["job"]["base_shard"]
    original = pinned_json({"uri": scripted["receipt_uri"], "sha256": scripted["receipt_sha256"]})["job"]
    script_map = retained_rows(scripted["decisions_uri"], "row", "output_row")
    gate_map = retained_rows(base["decisions_uri"], "base_row", "output_row")
    if len(gate_map) != base["rows"]:
        raise ValueError("Gate lineage omits candidate rows")
    bridge = {}
    for item in job["bridges"]:
        if item["input_path"] != original["input_path"]:
            raise ValueError("Question bridge source mismatch")
        content = read_pinned(item["pin"]["uri"], item["pin"]["sha256"])
        for row in pq.read_table(pa.BufferReader(content)).to_pylist():
            if row["row"] in bridge:
                raise ValueError("Overlapping question bridges")
            bridge[row["row"]] = row
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    prefix = tuple(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False).ids)
    start_id = tokenizer.token_to_id("<|start_header_id|>")
    tokens = JaggedArrayStore.open(prefix_join(base["cache_uri"], "input_ids"), mode="r", dtype=np.int32)
    masks = JaggedArrayStore.open(prefix_join(base["cache_uri"], "loss_weights"), mode="r", dtype=np.float32)
    if len(tokens) != base["rows"] or len(masks) != base["rows"]:
        raise ValueError("Candidate cache row count changed")
    counts = Counter()
    lengths = []
    supervised = 0
    digest = hashlib.sha256()

    def records():
        nonlocal supervised
        with (root / "decisions.jsonl").open("wt") as audit:
            for start in range(0, base["rows"], 128):
                indices = list(range(start, min(start + 128, base["rows"])))
                for i, ids, weights in zip(
                    indices, tokens.get_batch_sync(indices), masks.get_batch_sync(indices), strict=True
                ):
                    raw_row = script_map[gate_map[i]]
                    if raw_row not in bridge:
                        raise ValueError("Missing original-row bridge accounting")
                    entry = bridge[raw_row]
                    question = entry["question"]
                    changed = 0
                    if question is not None and job["repair_enabled"]:
                        text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                        after, changed = repair_text(text, question)
                        if changed:
                            new = np.asarray(tokenizer.encode(after, add_special_tokens=False).ids, dtype=np.int32)
                            if len(new) > MAX_TOKENS:
                                raise ValueError("Question repair exceeds context limit")
                            new_weights = prediction_weights(
                                new,
                                loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
                                message_start_id=start_id,
                                assistant_prefix=prefix,
                                trained_assistant_turns=None,
                            )
                            if not np.array_equal(ids[1:][weights[:-1] > 0], new[1:][new_weights[:-1] > 0]):
                                raise ValueError("Question-only repair changed assistant targets")
                            ids, weights = new, new_weights
                    status = "unjoined" if question is None else "changed" if changed else "unchanged"
                    counts[status] += 1
                    counts["changed_user_messages"] += changed
                    lengths.append(len(ids))
                    supervised += int(weights.sum())
                    digest.update(ids.astype("<i4", copy=False).tobytes())
                    digest.update(weights.astype("<f4", copy=False).tobytes())
                    audit.write(
                        json.dumps(
                            {
                                "base_row": i,
                                "original_row": raw_row,
                                "stable_row_ids": entry["stable_row_ids"],
                                "status": status,
                                "changed_messages": changed,
                            }
                        )
                        + "\n"
                    )
                    yield {"input_ids": ids, "loss_weights": weights}

    output = str(root / "cache")
    result = write_levanter_cache(
        records(),
        output,
        metadata={"source": source, "repair": "exact_raw_question"},
        batch_size=128,
        max_pending_batches=2,
    )
    if result["count"] != base["rows"] or supervised != base["supervised_prediction_positions"]:
        raise ValueError("Question repair changed rows or assistant supervision")
    stats = {
        "input_documents": base["rows"],
        "output_documents": base["rows"],
        "input_tokens": base["tokens"],
        "output_tokens": sum(lengths),
        "dropped_documents": 0,
        "dropped_tokens": 0,
        "changed_documents": counts["changed"],
        "input_supervised_prediction_positions": supervised,
        "output_supervised_prediction_positions": supervised,
    }
    receipt = {
        "job": job,
        "complete": True,
        "output_path": output,
        "output_rows": len(lengths),
        "output_tokens": sum(lengths),
        "lengths": lengths,
        "supervised_prediction_positions": supervised,
        "changed_documents": counts["changed"],
        "counts": dict(counts),
        "output_tokens_and_weights_sha256": digest.hexdigest(),
        "stats": {"opencode_user_quotes": stats},
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True))
    return {
        "source": source,
        "base": base,
        "pin": {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    for relative, expected in recipe["implementation"].items():
        if hashlib.sha256((Path(__file__).parents[2] / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Raw-question implementation differs from its recipe pin")
    require_adapter_pins(recipe)
    mode = recipe["mode"]
    root = StoragePath(recipe["output_prefix"])
    function = {"index": index_part, "bridge": bridge_part, "repair": repair_part}[mode]
    with ZephyrContext(
        name="opencode-question-" + mode,
        max_workers=recipe["workers"],
        resources=ResourceConfig(cpu=2, ram="16g"),
        chunk_storage_prefix=str(root / "zephyr"),
        max_execution_retries=2,
    ) as context:
        results = context.execute(Dataset.from_list(recipe["jobs"]).map(function)).results
    (root / "results.json").write_text(json.dumps(results, sort_keys=True))
    if mode == "index":
        grouped = defaultdict(list)
        for result in results:
            grouped[result["partition"]].append(result)
        indices = {}
        fixtures = []
        counts = Counter()
        for partition, parts in grouped.items():
            records = []
            for part in parts:
                records.extend(parquet_rows(part["pin"]["uri"]))
                counts.update(part["counts"])
                fixtures.extend(part["fixtures"])
            indices[partition] = write_parquet(str(root / f"{partition}.parquet"), records, INDEX_SCHEMA)
        (root / "fixtures.json").write_text(json.dumps(fixtures, sort_keys=True))
        expected = sum(job["input"]["rows"] for job in recipe["jobs"])
        if counts["raw_rows"] != expected:
            raise ValueError("Raw index omits staged source rows")
        manifest = {"complete": True, "indices": indices, "counts": dict(counts), "recipe": recipe}
    elif mode == "bridge":
        counts = Counter()
        for result in results:
            counts.update(result["counts"])
        expected = sum(job["row_stop"] - job["row_start"] for job in recipe["jobs"])
        if counts["joined"] + counts["unjoined"] != expected:
            raise ValueError("Question bridge omits original rows")
        if "min_joined_fraction" not in recipe:
            raise ValueError("Bridge recipe must declare min_joined_fraction")
        minimum = math.ceil(recipe["min_joined_fraction"] * expected)
        if counts["joined"] < minimum:
            raise ValueError("Question bridge joined fewer original rows than the recipe requires")
        ranges = defaultdict(list)
        for part in results:
            ranges[part["input_path"]].append((part["row_start"], part["row_stop"]))
        for path, spans in ranges.items():
            cursor = 0
            for start, stop in sorted(spans):
                if start != cursor:
                    raise ValueError("Question bridge ranges overlap or omit original rows")
                cursor = stop
            if cursor != next(job["expected_rows"] for job in recipe["jobs"] if job["input_path"] == path):
                raise ValueError("Question bridge does not cover the complete cache")
        manifest = {"complete": True, "parts": results, "counts": dict(counts), "recipe": recipe}
    else:
        sources = {s["source"]: s for s in recipe["sources"]}
        counts = Counter()
        groups = defaultdict(list)
        for result in results:
            r = pinned_json(result["pin"])
            counts.update(r["counts"])
            groups[result["source"]].append(
                {
                    "cache_uri": r["output_path"],
                    "rows": r["output_rows"],
                    "tokens": r["output_tokens"],
                    "supervised_prediction_positions": r["supervised_prediction_positions"],
                    "receipt_uri": result["pin"]["uri"],
                    "receipt_sha256": result["pin"]["sha256"],
                    "original_input_path": result["base"]["cache_uri"],
                    "decisions_uri": str(StoragePath(result["pin"]["uri"]).parent / "decisions.jsonl"),
                }
            )
        if set(groups) != set(sources):
            raise ValueError("Repair result omits selected OpenCode sources")
        output = []
        transforms = []
        for name, shards in sorted(groups.items()):
            old = sources[name]
            if sum(s["rows"] for s in shards) != old["counts"]["output_documents"]:
                raise ValueError("Question repair changed source population")
            totals = dict(old["counts"])
            totals.update(
                output_documents=sum(s["rows"] for s in shards),
                output_tokens=sum(s["tokens"] for s in shards),
                supervised_prediction_positions=sum(s["supervised_prediction_positions"] for s in shards),
            )
            output.append({**old, "counts": totals, "shards": shards})
            for result in results:
                if result["source"] == name:
                    r = pinned_json(result["pin"])
                    transforms.append(
                        {
                            "source": name,
                            "component": old["component"],
                            "loss_mode": old["loss_mode"],
                            "transform": "opencode_user_quotes",
                            **r["stats"]["opencode_user_quotes"],
                        }
                    )
        manifest = {
            "complete": True,
            "sources": output,
            "transforms": transforms,
            "counts": dict(counts),
            "tokenizer_uri": recipe["jobs"][0]["tokenizer_uri"],
            "tokenizer_sha256": recipe["jobs"][0]["tokenizer_sha256"],
        }
    if mode == "repair":
        (root / "cohort.json").write_text(json.dumps(manifest, sort_keys=True))
        base = pinned_json(recipe["base_inventory"])
        updated = {source["source"]: source for source in manifest["sources"]}
        original = {source["source"]: source for source in base["sources"]}
        for name, source in updated.items():
            before = original[name]
            for field in ("loss_mode", "teacher_only", "source_revision", "loss_weights_key", "loss_weights_indexing"):
                if source[field] != before[field]:
                    raise ValueError("Question repair changed source provenance or supervision")
            if source["counts"]["output_documents"] != before["counts"]["output_documents"]:
                raise ValueError("Question repair changed source row population")
        manifest = {
            **base,
            "sources": [updated.get(source["source"], source) for source in base["sources"]],
            "transforms": base["transforms"] + manifest["transforms"],
            "opencode_raw_question_repair": {
                "recipe": {"uri": args.recipe, "sha256": args.recipe_sha256},
                "counts": manifest["counts"],
            },
        }
    (root / "inventory.json").write_text(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
