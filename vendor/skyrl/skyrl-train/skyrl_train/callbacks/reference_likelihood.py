"""Fixed-reference likelihood diagnostics on the synchronized rollout policy."""

import asyncio
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import pyarrow.parquet as pq

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.callbacks.base import TrainerCallback, TrainerState
from skyrl_train.callbacks.types import CallbackErrorBehavior
from skyrl_train.io import io


@dataclass(frozen=True)
class ReferenceExample:
    id: str
    domain: str
    token_ids: tuple[int, ...]
    reference_start: int

    def __post_init__(self) -> None:
        if not self.id or not self.domain:
            raise ValueError("Reference examples need an ID and domain")
        if not 0 < self.reference_start < len(self.token_ids):
            raise ValueError(f"Reference {self.id} needs a nonempty prompt and reference")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in self.token_ids):
            raise ValueError(f"Reference {self.id} has invalid token IDs")


@dataclass(frozen=True)
class ReferenceScore:
    id: str
    domain: str
    token_ids: tuple[int, ...]
    reference_start: int
    token_logprobs: tuple[float, ...]
    nll: float


def score_reference(example: ReferenceExample, logprobs: Sequence[Mapping[int, float] | None]) -> ReferenceScore:
    """Extract chosen-token scores at reference positions, retaining exact alignment."""
    if len(logprobs) != len(example.token_ids):
        raise ValueError(f"Reference {example.id}: prompt logprobs do not align with token IDs")
    chosen = []
    for token, probabilities in zip(
        example.token_ids[example.reference_start :], logprobs[example.reference_start :], strict=True
    ):
        if probabilities is None or token not in probabilities:
            raise ValueError(f"Reference {example.id}: missing chosen-token logprob")
        value = float(probabilities[token])
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"Reference {example.id}: invalid chosen-token logprob")
        chosen.append(value)
    return ReferenceScore(
        example.id,
        example.domain,
        example.token_ids,
        example.reference_start,
        tuple(chosen),
        -math.fsum(chosen) / len(chosen),
    )


def reference_metrics(scores: Sequence[ReferenceScore]) -> dict[str, float]:
    """Report both equal-example and token-weighted NLL by domain."""
    grouped = defaultdict(list)
    for score in scores:
        grouped[score.domain].append(score)
    metrics = {}
    for domain, rows in grouped.items():
        prefix = f"eval/reference/{domain}"
        tokens = sum(len(row.token_logprobs) for row in rows)
        metrics[f"{prefix}/example_mean_nll"] = math.fsum(row.nll for row in rows) / len(rows)
        metrics[f"{prefix}/token_mean_nll"] = -math.fsum(value for row in rows for value in row.token_logprobs) / tokens
        metrics[f"{prefix}/examples"] = float(len(rows))
        metrics[f"{prefix}/tokens"] = float(tokens)
    return metrics


def load_references(paths: Sequence[str]) -> tuple[ReferenceExample, ...]:
    rows = []
    for path in paths:
        with io.open_file(path, "rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                rows.append(
                    ReferenceExample(
                        id=row["id"],
                        domain=row["validation_domain"],
                        token_ids=tuple(row["reference_token_ids"]),
                        reference_start=row["reference_start"],
                    )
                )
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Reference likelihood requires nonempty, uniquely identified examples")
    return tuple(rows)


class ReferenceLikelihoodCallback(TrainerCallback):
    """Score explicit Parquet references at each existing evaluation event.

    Inputs contain id, validation_domain, reference_token_ids (prompt + answer),
    and reference_start.
    Output JSONL retains token IDs and answer logprobs for each training step.
    """

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, *, reference_paths: list[str], output_path: str, batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("Reference batch_size must be positive")
        self.reference_paths = tuple(reference_paths)
        self.output_path = output_path
        self.batch_size = batch_size
        self.rows: tuple[ReferenceExample, ...] | None = None

    async def collect_evaluation(self, state: TrainerState, *, trainer, pool) -> dict[str, float]:
        if self.rows is None:
            self.rows = await asyncio.to_thread(load_references, self.reference_paths)
        scores = []
        for offset in range(0, len(self.rows), self.batch_size):
            batch = self.rows[offset : offset + self.batch_size]
            output = await trainer.inference_engine_client.generate(
                {
                    "prompts": None,
                    "prompt_token_ids": [list(row.token_ids) for row in batch],
                    "session_ids": None,
                    "sampling_params": {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 0},
                }
            )
            probabilities = output["prompt_logprobs"]
            if probabilities is None or len(probabilities) != len(batch):
                raise ValueError("Reference likelihood did not receive every prompt's scores")
            scores.extend(score_reference(row, probs) for row, probs in zip(batch, probabilities, strict=True))
        payload = "".join(
            json.dumps({"step": state.global_step, **asdict(score)}, allow_nan=False) + "\n" for score in scores
        )
        path = join_resource_path(self.output_path, f"step-{state.global_step}.jsonl")
        await asyncio.to_thread(io.makedirs, self.output_path, exist_ok=True)
        await asyncio.to_thread(io.write_bytes_atomic, path, payload.encode())
        return reference_metrics(scores)
