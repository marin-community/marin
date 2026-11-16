# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build immutable candidate shards and their per-transform retention receipts."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.datakit.cluster.quality.fast_transformer.calibrate import load_calibration
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme
from experiments.sft_new_mix.exp3_filters import (
    MAX_TOKENS,
    Decision,
    Document,
    Stage,
    apply_stage,
    needs_pass,
    source_exclusion,
)
from experiments.sft_new_mix.exp3_runtime import (
    BUILD_IMPLEMENTATION_FILES,
    build_config,
    configure_storage,
    count_stage,
    implementation_identity,
    tokenizer_for,
)
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.score_replay import MODEL_DIR, model_digests

logger = logging.getLogger(__name__)
BATCH_ROWS = 512


@dataclass(frozen=True)
class ShardJob:
    source: str
    input_path: str
    expected_rows: int
    expected_tokens: int
    row_start: int
    row_stop: int
    loss_mode: LossMode
    source_revision: str
    passed: bool
    pass_receipt_sha256: str
    output_prefix: str
    config_uri: str
    config_sha256: str
    implementation_sha256: str
    tokenizer_uri: str
    tokenizer_sha256: str


@cache
def replay_scorer():
    return load_pooled_scorer(MODEL_DIR), load_calibration(MODEL_DIR)


def verified_replay_scorer(config_value: dict):
    """Refuse to score with junkgate weights that moved after the cutoff was calibrated."""
    if config_value.get("junkgate_model_dir") != MODEL_DIR:
        raise ValueError("Junkgate model directory changed after cutoff calibration")
    if config_value.get("junkgate_model_sha256") != model_digests(MODEL_DIR):
        raise ValueError("Junkgate model weights changed after cutoff calibration")
    return replay_scorer()


def input_batches(
    store: JaggedArrayStore, lengths: np.ndarray, row_start: int
) -> Iterator[list[tuple[int, np.ndarray]]]:
    indices = np.flatnonzero(lengths <= MAX_TOKENS)
    for start in range(0, len(indices), BATCH_ROWS):
        batch_indices = indices[start : start + BATCH_ROWS] + row_start
        arrays = store.get_batch_sync(batch_indices.tolist())
        yield list(zip(batch_indices.tolist(), arrays, strict=True))


def process_shard(job_dict: dict) -> dict:
    configure_storage()
    return build_shard(job_dict)


def build_shard(job_dict: dict) -> dict:
    """Write one complete new shard; never publish a catalog from partial outputs."""
    job = ShardJob(**{**job_dict, "loss_mode": LossMode(job_dict["loss_mode"])})
    if implementation_identity(BUILD_IMPLEMENTATION_FILES) != job.implementation_sha256:
        raise ValueError("Candidate implementation changed")
    config_value, config = build_config(job.config_uri, job.config_sha256)
    identity = hashlib.sha256(json.dumps(job_dict, sort_keys=True).encode()).hexdigest()
    source_identity = hashlib.sha256(job.source.encode()).hexdigest()
    root = StoragePath(job.output_prefix) / "components" / source_identity / "shards" / identity
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != job_dict or not receipt["complete"]:
            raise ValueError("Shard receipt identity mismatch")
        return receipt
    if (root / "cache").exists():
        # A preempted worker can leave the cache in place before its receipt
        # lands; shard output is deterministic from the job identity, so the
        # retry deletes it and rebuilds identical bytes.
        logger.warning("Deleting unreceipted cache from a preempted worker: %s", root / "cache")
        (root / "cache").rmtree()
    if job.passed and not job.pass_receipt_sha256:
        raise ValueError("A pass assertion requires a pinned source receipt")
    stats: dict[str, Counter] = defaultdict(Counter)
    stage = Stage.SOURCE_EXCLUSIONS
    reason = source_exclusion(job.source) if Stage.SOURCE_EXCLUSIONS in config.enabled else ""
    if not reason and Stage.OUTCOME in config.enabled and needs_pass(job.source) and not job.passed:
        reason = "missing_or_failed_outcome"
        stage = Stage.OUTCOME
    if reason:
        stats[stage].update(
            input_documents=job.expected_rows,
            input_tokens=job.expected_tokens,
            dropped_documents=job.expected_rows,
            dropped_tokens=job.expected_tokens,
        )
        stats[stage]["drop_reason:" + reason] = job.expected_rows
        receipt = {
            "job": job_dict,
            "complete": True,
            "output_rows": 0,
            "output_tokens": 0,
            "stats": stats,
            "loss_mode": job.loss_mode,
        }
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        return receipt
    tokenizer = tokenizer_for(job.tokenizer_uri, job.tokenizer_sha256)
    assistant_prefix = tuple(
        tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False).ids
    )
    message_start = tokenizer.token_to_id("<|start_header_id|>")
    store = JaggedArrayStore.open(prefix_join(job.input_path, "input_ids"), mode="r", dtype=np.int32)
    if not 0 <= job.row_start < job.row_stop <= len(store) or job.expected_rows != job.row_stop - job.row_start:
        raise ValueError(f"Input row range changed for {job.input_path}")
    ends = np.asarray(store.offsets[job.row_start + 1 : job.row_stop + 1].read().result(), dtype=np.int64)
    begin = int(store.offsets[job.row_start].read().result()) if job.row_start else 0
    lengths = np.diff(np.concatenate(([begin], ends)))
    if int(lengths.sum()) != job.expected_tokens:
        raise ValueError(f"Input token count changed for {job.input_path}")
    over = lengths > MAX_TOKENS
    stats[Stage.CONTEXT_LIMIT].update(
        input_documents=len(lengths),
        input_tokens=int(lengths.sum()),
        output_documents=int((~over).sum()),
        output_tokens=int(lengths[~over].sum()),
        dropped_documents=int(over.sum()),
        dropped_tokens=int(lengths[over].sum()),
    )
    stats[Stage.CONTEXT_LIMIT]["drop_reason:over_context_limit"] = int(over.sum())
    # The context gate is mandatory for bounded IO; turning it off requires a
    # new compatible context recipe rather than silently reading/truncating rows.
    if Stage.CONTEXT_LIMIT not in config.enabled and over.any():
        raise ValueError("Disabled context gate with over-limit documents is not training-ready")
    output_lengths: list[int] = []
    supervised = 0
    rejected = Counter()
    input_digest, output_digest = hashlib.sha256(), hashlib.sha256()

    def records() -> Iterator[dict[str, np.ndarray]]:
        nonlocal supervised
        audit_path = root / "decisions.jsonl"
        with audit_path.open("wt") as audit:
            for index in np.flatnonzero(over):
                audit.write(
                    json.dumps(
                        {
                            "row": int(index) + job.row_start,
                            "stage": Stage.CONTEXT_LIMIT,
                            "reason": "over_context_limit",
                            "tokens": int(lengths[index]),
                        }
                    )
                    + "\n"
                )
            for batch in input_batches(store, lengths, job.row_start):
                pending = []
                for index, original in batch:
                    input_digest.update(np.asarray(original, dtype="<i4").tobytes())
                    text = tokenizer.decode(original.tolist(), skip_special_tokens=False)
                    document = Document(job.source, f"{identity}:{index}", text, job.passed)
                    token_ids = original

                    def token_count(value: str) -> int:
                        return len(tokenizer.encode(value, add_special_tokens=False).ids)

                    for stage in Stage:
                        if stage not in config.enabled or stage == Stage.CONTEXT_LIMIT:
                            continue
                        if job.loss_mode == LossMode.FULL_TOKEN and stage not in (
                            Stage.SOURCE_EXCLUSIONS,
                            Stage.OUTCOME,
                            Stage.REPLAY_HEURISTICS,
                            Stage.REPLAY_LINE_LENGTH,
                            Stage.REPLAY_EMAILS,
                        ):
                            continue
                        before = len(token_ids)
                        try:
                            decision = apply_stage(document, stage, config, token_count)
                        except (ValueError, json.JSONDecodeError) as error:
                            # This is a retained parsing disposition, not a swallowed IO failure.
                            decision = Decision(document, "invalid_native_structure:" + type(error).__name__)
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
                            rejected[decision.reason] += 1
                            audit.write(
                                json.dumps({"row": index, "stage": stage, "reason": decision.reason, "tokens": before})
                                + "\n"
                            )
                            break
                    else:
                        # Edits can lengthen a document, so check again explicitly.
                        if len(token_ids) > MAX_TOKENS:
                            count_stage(stats, "post_edit_context_limit", len(token_ids), 0, "over_context_limit")
                            audit.write(
                                json.dumps(
                                    {"row": index, "stage": "post_edit_context_limit", "reason": "over_context_limit"}
                                )
                                + "\n"
                            )
                            continue
                        pending.append((index, document, np.asarray(token_ids, dtype=np.int32)))
                scores = None
                if (
                    job.loss_mode == LossMode.FULL_TOKEN
                    and job.source.startswith("replay/")
                    and config_value["junkgate_enabled"]
                    and pending
                ):
                    scorer, calibration = replay_scorer()
                    raw = score_bme(scorer, [ids for _, _, ids in pending])
                    scores = calibration.apply(raw, types=None)
                for position, (index, _document, token_ids) in enumerate(pending):
                    if scores is not None:
                        score = float(scores[position])
                        drop = score <= config_value["junkgate_cutoff"]
                        count_stage(
                            stats,
                            "junkgate2",
                            len(token_ids),
                            0 if drop else len(token_ids),
                            "junkgate_cutoff" if drop else "",
                        )
                        audit.write(
                            json.dumps({"row": index, "stage": "junkgate2", "score": score, "dropped": drop}) + "\n"
                        )
                        if drop:
                            continue
                    weights = prediction_weights(
                        token_ids,
                        loss_mode=job.loss_mode,
                        message_start_id=message_start,
                        assistant_prefix=assistant_prefix,
                        trained_assistant_turns=None,
                    )
                    if not weights.any():
                        count_stage(stats, "loss_mask", len(token_ids), 0, "zero_supervised_tokens")
                        continue
                    supervised += int(weights.sum())
                    output_lengths.append(len(token_ids))
                    audit.write(
                        json.dumps({"row": index, "stage": "retained", "output_row": len(output_lengths) - 1}) + "\n"
                    )
                    output_digest.update(token_ids.astype("<i4", copy=False).tobytes())
                    output_digest.update(weights.astype("<f4", copy=False).tobytes())
                    yield {"input_ids": token_ids, "loss_weights": weights}

    output_path = str(root / "cache")
    if job.loss_mode == LossMode.FULL_TOKEN and job.source.startswith("replay/") and config_value["junkgate_enabled"]:
        # Verify before the writer consumes the first batch that would score.
        verified_replay_scorer(config_value)
    result = write_levanter_cache(
        records(),
        output_path,
        metadata={
            "candidate": "2026.11.15",
            "source": job.source,
            "source_revision": job.source_revision,
            "loss_mode": job.loss_mode,
            "loss_weights_indexing": "prediction",
            "config_uri": job.config_uri,
            "tokenizer_sha256": job.tokenizer_sha256,
        },
        batch_size=BATCH_ROWS,
        max_pending_batches=2,
    )
    if result["count"] != len(output_lengths):
        raise ValueError("Writer count differs from retained documents")
    if output_lengths:
        ledger = CacheLedger.load(output_path)
        if not ledger.is_finished or ledger.total_num_rows != len(output_lengths):
            raise ValueError("Candidate shard did not complete")
    receipt = {
        "job": job_dict,
        "complete": True,
        "output_path": output_path,
        "output_rows": len(output_lengths),
        "output_tokens": sum(output_lengths),
        "supervised_prediction_positions": supervised,
        "input_eligible_tokens_sha256": input_digest.hexdigest(),
        "output_tokens_and_weights_sha256": output_digest.hexdigest(),
        "stats": stats,
        "loss_mode": job.loss_mode,
        "lengths": output_lengths,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return {key: value for key, value in receipt.items() if key != "lengths"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-execution-retries", type=int, default=2)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("levanter.store.cache").setLevel(logging.WARNING)
    configure_storage()
    plan = json.loads(StoragePath(args.plan).read_text())
    jobs = plan["jobs"]
    if args.local:
        for job in jobs:
            print(json.dumps(process_shard(job)), flush=True)
        return
    context = ZephyrContext(
        max_workers=args.workers,
        resources=ResourceConfig(cpu=2, ram="16g"),
        name="exp3-data-candidate",
        chunk_storage_prefix=prefix_join(plan["output_prefix"], "zephyr"),
        max_execution_retries=args.max_execution_retries,
    )
    pipeline = Dataset.from_list(jobs).map(process_shard)
    result = context.execute(pipeline, verbose=True)
    (StoragePath(plan["output_prefix"]) / "execution.json").write_text(
        json.dumps({"results": result.results, "counters": result.counters}, default=str)
    )


if __name__ == "__main__":
    main()
