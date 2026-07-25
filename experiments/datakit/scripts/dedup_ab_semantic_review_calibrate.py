# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the production chunked semantic reviewer against manual labels."""

import argparse
import asyncio
from typing import Any

from marin.inference.iris import remote_inference
from openai import AsyncOpenAI
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.scripts.dedup_ab_semantic_judge import (
    CANONICAL_CHUNKS_PER_MEMBER,
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    MODEL_ID,
    _inference_config,
    load_calibration_cases,
)
from experiments.datakit.scripts.dedup_ab_semantic_review import review_cases


class ChunkCalibrationData(BaseModel):
    """Exact chunk-review calibration results."""

    version: str = "v1"
    model: str
    machine_labels_path: str
    manual_labels_path: str
    chunk_chars: int
    overlap_chars: int
    canonical_chunks_per_member: int
    pairs: int
    correct_pairs: int
    resolved_pairs: int
    member_chunks: int
    request_attempts: int
    passed: bool
    results: list[dict[str, Any]]


def calibration_result(
    *,
    model: str,
    machine_labels_path: str,
    manual_labels_path: str,
    chunk_chars: int,
    overlap_chars: int,
    canonical_chunks_per_member: int,
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> ChunkCalibrationData:
    """Bind forced-chunk outcomes to their exact expected labels."""
    if len(cases) != len(outcomes):
        raise AssertionError(f"Chunk calibration outcome count differs: {len(outcomes)}/{len(cases)}")

    results = []
    for case, outcome in zip(cases, outcomes, strict=True):
        identity_mismatches = [
            field
            for field in (
                "review_key",
                "variant",
                "member_id",
                "canonical_id",
                "raw_sha256",
                "canonical_raw_sha256",
                "pair_path",
                "pair_row_index",
            )
            if outcome[field] != case[field]
        ]
        if identity_mismatches:
            raise AssertionError(
                f"Chunk calibration outcome identity differs for {case['review_key']}: " f"{identity_mismatches}"
            )
        if outcome["review_mode"] != "chunked":
            raise AssertionError(f"Calibration outcome did not use chunked review: {case['review_key']}")
        correct = outcome["status"] == "resolved" and outcome["label"] == case["expected_label"]
        results.append(
            {
                "review_key": case["review_key"],
                "variant": case["variant"],
                "member_id": case["member_id"],
                "canonical_id": case["canonical_id"],
                "expected_label": case["expected_label"],
                "expected_basis": case["expected_basis"],
                "correct": correct,
                "outcome": outcome,
            }
        )

    correct_pairs = sum(bool(result["correct"]) for result in results)
    resolved_pairs = sum(outcome["status"] == "resolved" for outcome in outcomes)
    return ChunkCalibrationData(
        model=model,
        machine_labels_path=machine_labels_path,
        manual_labels_path=manual_labels_path,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        canonical_chunks_per_member=canonical_chunks_per_member,
        pairs=len(cases),
        correct_pairs=correct_pairs,
        resolved_pairs=resolved_pairs,
        member_chunks=sum(int(outcome["member_chunks"]) for outcome in outcomes),
        request_attempts=sum(int(outcome["request_attempts"]) for outcome in outcomes),
        passed=correct_pairs == len(cases) and resolved_pairs == len(cases),
        results=results,
    )


def run_chunk_calibration(
    *,
    machine_labels_path: str,
    manual_labels_path: str,
    model: str,
    output: str,
    chunk_chars: int,
    overlap_chars: int,
    canonical_chunks_per_member: int,
) -> ChunkCalibrationData:
    """Serve the production reviewer and require exact manual-label agreement."""
    cases = load_calibration_cases(
        machine_labels_path=machine_labels_path,
        manual_labels_path=manual_labels_path,
    )
    served_model, engine, iris, broker = _inference_config(model)
    with remote_inference(served_model, engine, iris, broker=broker) as session:
        client = AsyncOpenAI(
            base_url=session.model.endpoint.base_url,
            api_key=session.model.endpoint.api_key or "local",
        )

        async def judge_and_close() -> list[dict[str, Any]]:
            try:
                return await review_cases(
                    client,
                    model=session.model.endpoint.model,
                    cases=cases,
                    force_mode="chunked",
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                    canonical_chunks_per_member=canonical_chunks_per_member,
                )
            finally:
                await client.close()

        outcomes = asyncio.run(judge_and_close())

    result = calibration_result(
        model=model,
        machine_labels_path=machine_labels_path,
        manual_labels_path=manual_labels_path,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        canonical_chunks_per_member=canonical_chunks_per_member,
        cases=cases,
        outcomes=outcomes,
    )
    StoragePath(output).write_text(result.model_dump_json(indent=2))
    if not result.passed:
        raise AssertionError(
            f"Chunk semantic calibration failed: correct={result.correct_pairs}/{result.pairs}, "
            f"resolved={result.resolved_pairs}/{result.pairs}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-labels", required=True)
    parser.add_argument("--manual-labels", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=CHUNK_OVERLAP_CHARS)
    parser.add_argument(
        "--canonical-chunks-per-member",
        type=int,
        default=CANONICAL_CHUNKS_PER_MEMBER,
    )
    args = parser.parse_args()
    configure_logging()

    result = run_chunk_calibration(
        machine_labels_path=args.machine_labels,
        manual_labels_path=args.manual_labels,
        model=args.model,
        output=args.output,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        canonical_chunks_per_member=args.canonical_chunks_per_member,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
