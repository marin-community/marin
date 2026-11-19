# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build teacher-masked relay shards from the frozen source, with row receipts."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from levanter.store.cache import write_levanter_cache
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.download.calibforge_relay import (
    EXCLUSION_DATASET_ID,
    EXCLUSION_LIST_SIZE,
    EXCLUSION_REVISION,
    HF_DATASET_ID,
    HF_REVISION,
    SOURCE_NAME,
)
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers.utils.chat_template_utils import render_jinja_template
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_filters import MAX_TOKENS, Document, Stage, apply_stage
from experiments.sft_new_mix.exp3_runtime import (
    build_config,
    configure_storage,
    count_stage,
    implementation_identity,
    tokenizer_for,
)
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.relay_teacher import render_teacher_transcript

# The files whose bytes determine relay teacher shards; the shared runtime module
# is included so helper drift changes the pin.
RELAY_IMPLEMENTATION_FILES = (
    "exp3_runtime.py",
    "materialize_relay.py",
    "relay_teacher.py",
    "loss_modes.py",
    "exp3_filters.py",
)


def materialize_shard(job: dict) -> dict:
    configure_storage()
    if job["implementation_sha256"] != implementation_identity(RELAY_IMPLEMENTATION_FILES):
        raise ValueError("Relay builder implementation changed")
    root = StoragePath(job["output_prefix"]) / f"shard-{job['shard']:05d}"
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != job:
            raise ValueError("Existing relay shard has a different build identity")
        return receipt
    if (root / "cache").exists():
        raise ValueError("Unreceipted relay cache preserved; choose a new output version")
    filename = f"data/terminus2_relay_sft/train-{job['shard']:05d}-of-00008.parquet"
    raw_path = hf_hub_download(HF_DATASET_ID, filename, revision=HF_REVISION, repo_type="dataset")
    exclusion_path = hf_hub_download(
        EXCLUSION_DATASET_ID, "excluded_tasks.jsonl", revision=EXCLUSION_REVISION, repo_type="dataset"
    )
    with open(raw_path, "rb") as stream:
        raw_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with open(exclusion_path, "rb") as stream:
        exclusion_bytes = stream.read()
    exclusions = frozenset(json.loads(line)["task_id"] for line in exclusion_bytes.splitlines())
    if len(exclusions) != EXCLUSION_LIST_SIZE:
        raise ValueError("Frozen relay exclusion-list population changed")
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    _, config = build_config(job["config_uri"], job["config_sha256"])
    prefix = tuple(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False).ids)
    message_start = tokenizer.token_to_id("<|start_header_id|>")
    stats = defaultdict(Counter)
    diagnostics = Counter()
    lengths = []
    supervised = teacher_turns = student_turns = raw_rows = raw_tokens = 0
    digest = hashlib.sha256()

    def records() -> Iterator[dict[str, np.ndarray]]:
        nonlocal supervised, teacher_turns, student_turns, raw_rows, raw_tokens
        with (root / "decisions.jsonl").open("wt") as audit:
            for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    raw_rows += 1
                    rendered, _ = render_jinja_template(
                        conversations=[[{"role": turn["role"], "content": turn["content"]} for turn in row["messages"]]],
                        chat_template=MARIN_CHAT_TEMPLATE,
                        bos_token="",
                        add_generation_prompt=False,
                    )
                    input_tokens = len(tokenizer.encode(rendered[0], add_special_tokens=False).ids)
                    raw_tokens += input_tokens
                    if row["n_tokens"] is None:
                        diagnostics["source_reported_token_count_missing"] += 1
                    reason = ""
                    if row["task"].startswith("calibforge__") and row["task"].split("__", 2)[2] in exclusions:
                        reason = "source_benchmark_exclusion"
                    elif row["passed"] is not True:
                        reason = "missing_or_failed_outcome"
                    if reason:
                        count_stage(stats, "relay_admission", input_tokens, 0, reason)
                        audit.write(json.dumps({"source_id": row["sid"], "reason": reason}) + "\n")
                        continue
                    try:
                        transcript = render_teacher_transcript(row["messages"])
                    except ValueError as error:
                        reason = "invalid_teacher_mapping:" + str(error).split(": ", 1)[0][:100]
                        count_stage(stats, "relay_admission", input_tokens, 0, reason)
                        audit.write(json.dumps({"source_id": row["sid"], "reason": reason}) + "\n")
                        continue
                    if sum(transcript.trained_assistant_turns) != row["teacher_turns"]:
                        diagnostics["source_teacher_summary_mismatch"] += 1
                    if (
                        len(transcript.trained_assistant_turns) - sum(transcript.trained_assistant_turns)
                        != row["student_turns"]
                    ):
                        diagnostics["source_student_summary_mismatch"] += 1
                    document = Document(SOURCE_NAME, row["sid"], transcript.text, True)
                    token_ids = np.asarray(tokenizer.encode(document.text, add_special_tokens=False).ids, dtype=np.int32)
                    count_stage(stats, "relay_admission", input_tokens, len(token_ids), changed=True)
                    for stage in Stage:
                        if stage not in config.enabled:
                            continue
                        before = len(token_ids)
                        decision = apply_stage(
                            document,
                            stage,
                            config,
                            lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
                        )
                        if decision.document.text != document.text:
                            token_ids = np.asarray(
                                tokenizer.encode(decision.document.text, add_special_tokens=False).ids, dtype=np.int32
                            )
                        count_stage(
                            stats,
                            stage,
                            before,
                            len(token_ids) if decision.keep else 0,
                            decision.reason,
                            changed=decision.document.text != document.text,
                        )
                        document = decision.document
                        if not decision.keep:
                            audit.write(json.dumps({"source_id": row["sid"], "reason": decision.reason}) + "\n")
                            break
                    else:
                        if len(token_ids) > MAX_TOKENS:
                            raise ValueError("Relay exceeded the bounded context after edits")
                        weights = prediction_weights(
                            token_ids,
                            loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
                            message_start_id=message_start,
                            assistant_prefix=prefix,
                            trained_assistant_turns=transcript.trained_assistant_turns,
                        )
                        if not weights.any():
                            raise ValueError("Admitted relay has zero teacher supervision")
                        teacher_turns += sum(transcript.trained_assistant_turns)
                        student_turns += len(transcript.trained_assistant_turns) - sum(
                            transcript.trained_assistant_turns
                        )
                        supervised += int(weights.sum())
                        audit.write(
                            json.dumps(
                                {
                                    "source_id": row["sid"],
                                    "output_row": len(lengths),
                                    "trained_assistant_turns": transcript.trained_assistant_turns,
                                }
                            )
                            + "\n"
                        )
                        lengths.append(len(token_ids))
                        digest.update(token_ids.astype("<i4").tobytes())
                        digest.update(weights.astype("<f4").tobytes())
                        yield {"input_ids": token_ids, "loss_weights": weights}

    destination = str(root / "cache")
    result = write_levanter_cache(
        records(),
        destination,
        metadata={
            "source": SOURCE_NAME,
            "source_revision": HF_REVISION,
            "loss_mode": "chat_assistant_only",
            "teacher_only": True,
            "loss_weights_indexing": "prediction",
        },
        batch_size=8,
        max_pending_batches=2,
    )
    if result["count"] != len(lengths):
        raise ValueError("Relay writer population differs from receipt")
    receipt = {
        "job": job,
        "complete": True,
        "source": SOURCE_NAME,
        "source_revision": HF_REVISION,
        "raw_path": filename,
        "raw_file_sha256": raw_sha256,
        "exclusion_revision": EXCLUSION_REVISION,
        "exclusion_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
        "input_documents": raw_rows,
        "input_tokens": raw_tokens,
        "input_token_measurement": "complete original message list rendered with pinned Marin template and tokenizer",
        "output_path": destination,
        "output_rows": len(lengths),
        "output_tokens": sum(lengths),
        "supervised_prediction_positions": supervised,
        "teacher_turns": teacher_turns,
        "student_turns": student_turns,
        "loss_mode": "chat_assistant_only",
        "teacher_only": True,
        "output_tokens_and_weights_sha256": digest.hexdigest(),
        "stats": stats,
        "diagnostics": diagnostics,
        "lengths": lengths,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return {key: value for key, value in receipt.items() if key != "lengths"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    plan = json.loads(StoragePath(args.plan).read_text())
    base = {key: plan["jobs"][0][key] for key in ("tokenizer_uri", "tokenizer_sha256", "config_uri", "config_sha256")}
    jobs = [
        {
            **base,
            "shard": shard,
            "output_prefix": args.output_prefix,
            "implementation_sha256": implementation_identity(RELAY_IMPLEMENTATION_FILES),
        }
        for shard in range(8)
    ]
    with ZephyrContext(
        name="exp3-relay-teacher-cache",
        max_workers=args.workers,
        resources=ResourceConfig(cpu=2, ram="8g", disk="20g"),
        chunk_storage_prefix=prefix_join(args.output_prefix, "zephyr"),
        max_execution_retries=2,
    ) as context:
        context.execute(Dataset.from_list(jobs).map(materialize_shard))
    receipts = [
        json.loads((StoragePath(args.output_prefix) / f"shard-{shard:05d}/receipt.json").read_text())
        for shard in range(8)
    ]
    (StoragePath(args.output_prefix) / "manifest.json").write_text(
        json.dumps({"source": SOURCE_NAME, "shards": receipts})
    )


if __name__ == "__main__":
    main()
