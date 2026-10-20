"""An online check that the trainer computes the rollout engines' log-probabilities bit for bit.

After a qualifying weight sync each live engine generates one fixed short response (``ExactnessCheck.start_generation``,
collected by ``finish_generation``). The trainer's next policy forward, while the policy still holds the weights the
engines received, scores those responses (``ExactnessCheck.batch``), and ``ExactnessCheck.compare`` counts the response
tokens whose float32 log-probability differs from the engine's, in total and per engine data-parallel rank. The check
runs under every numerics mode; native and batch-invariant runs report their real mismatch.
"""

import asyncio
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.weight_sync.base import WeightSyncReason

# The fixed prompt and response length of every check: a few decode steps on each engine.
CHECK_TEXT = "Every token's log-probability is the same in the trainer as in the inference engine that sampled it."
CHECK_RESPONSE_TOKENS = 16
CHECK_SEED = 0
METRIC_PREFIX = "exactness_check"


@dataclass(frozen=True)
class EngineSample:
    """One engine's response to the check prompt, its log-probabilities, the engine's data-parallel rank and, when the
    engine returned them, its expert routes aligned with the response tokens."""

    engine_dp_rank: int
    response_token_ids: list[int]
    response_logprobs: list[float]
    routed_experts: np.ndarray | None


class ExactnessCheck:
    """Engine samples taken after a weight sync, and their comparison with the trainer's scores."""

    def __init__(self, every_weight_syncs: int, sampling_params: DictConfig, tokenizer: PreTrainedTokenizerBase):
        self.every_weight_syncs = every_weight_syncs
        self.tokenizer = tokenizer
        self.prompt_token_ids = tokenizer.encode(CHECK_TEXT, add_special_tokens=False)
        self.sampling_params = {
            **get_vllm_sampling_params(sampling_params),
            "max_tokens": CHECK_RESPONSE_TOKENS,
            "logprobs": 0,
            "seed": CHECK_SEED,
            "stop": None,
        }
        self.pending: list[EngineSample] | None = None
        self.generate_failures = 0
        self._generation: asyncio.Task[list[EngineSample]] | None = None
        self._training_syncs = 0
        self._generate_seconds = 0.0

    def count_sync(self, reason: WeightSyncReason) -> bool:
        """Count a weight sync made for ``reason``; True when the check follows it: the first sync, a checkpoint-restore
        sync and every ``every_weight_syncs``-th training-step sync."""
        if reason in (WeightSyncReason.INITIAL, WeightSyncReason.CHECKPOINT_RESTORE):
            return True
        if reason != WeightSyncReason.TRAINING_STEP:
            return False
        self._training_syncs += 1
        return self._training_syncs % self.every_weight_syncs == 0

    def start_generation(self, client: InferenceEngineClient) -> None:
        """Start every live engine's response to the check prompt; ``finish_generation`` collects them."""
        if self._generation is not None:
            self._generation.cancel()
        self._generation = asyncio.create_task(self._generate(client))

    async def finish_generation(self) -> dict[str, float]:
        """Wait for the started generation and keep its samples for ``batch``. A generation during which an engine
        died is skipped and counted in ``exactness_check/generate_failures``."""
        if self._generation is None:
            return {}
        generation, self._generation = self._generation, None
        try:
            self.pending = await generation
        except ray.exceptions.RayActorError as error:
            self.generate_failures += 1
            logger.warning("Exactness check skipped: an engine died while generating the check responses: {}", error)
            return {f"{METRIC_PREFIX}/generate_failures": float(self.generate_failures)}
        return {}

    async def _generate(self, client: InferenceEngineClient) -> list[EngineSample]:
        started = time.monotonic()
        outputs = await client.generate_on_live_engines(self.prompt_token_ids, self.sampling_params)
        self._generate_seconds = time.monotonic() - started
        return [
            EngineSample(
                engine_dp_rank=rank,
                response_token_ids=list(output["response_ids"][0]),
                response_logprobs=list(output["response_logprobs"][0]),
                routed_experts=(output.get("routed_experts") or [None])[0],
            )
            for rank, output in outputs
        ]

    def batch(self, dp_size: int, num_experts: int | None) -> TrainingInputBatch:
        """The pending responses as a policy forward batch, repeated to a multiple of ``dp_size`` rows, with their
        expert routes when the engines returned them."""
        samples = self.pending
        rows = [samples[index % len(samples)] for index in range(-(-len(samples) // dp_size) * dp_size)]
        responses = [row.response_token_ids for row in rows]
        sequences, attention_mask, action_mask, *_ = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            [self.prompt_token_ids] * len(rows),
            responses,
            rewards=[[0.0] * len(response) for response in responses],
            loss_masks=[[1] * len(response) for response in responses],
        )
        routes = tuple(row.routed_experts for row in rows)
        batch = TrainingInputBatch(
            {"sequences": sequences, "attention_mask": attention_mask},
            routed_expert_rows=(
                None if routes[0] is None else RoutedExpertRows(routes, action_mask.shape[1], num_experts)
            ),
        )
        batch.metadata = {"response_length": action_mask.shape[1]}
        return batch

    def compare(self, logprobs: torch.Tensor, score_seconds: float) -> dict[str, float]:
        """The check's metrics from the trainer's ``[rows, response_length]`` scores of ``batch()``, in total and per
        engine data-parallel rank. A token matches when its score has the engine's float32 bits."""
        samples, self.pending = self.pending, None
        tokens: Counter[int] = Counter()
        mismatched: Counter[int] = Counter()
        largest = 0.0
        for row, scores in enumerate(logprobs.float().cpu()):
            sample = samples[row % len(samples)]
            engine = np.asarray(sample.response_logprobs, dtype=np.float32)
            trainer = scores[: engine.size].numpy()
            tokens[sample.engine_dp_rank] += engine.size
            mismatched[sample.engine_dp_rank] += int((trainer.view(np.int32) != engine.view(np.int32)).sum())
            largest = max(largest, float(np.abs(trainer.astype(np.float64) - engine.astype(np.float64)).max(initial=0)))
        metrics = {
            f"{METRIC_PREFIX}/engines": float(len(samples)),
            f"{METRIC_PREFIX}/tokens": float(tokens.total()),
            f"{METRIC_PREFIX}/mismatched_tokens": float(mismatched.total()),
            f"{METRIC_PREFIX}/log_ratio_abs_max": largest,
            f"{METRIC_PREFIX}/generate_seconds": self._generate_seconds,
            f"{METRIC_PREFIX}/score_seconds": score_seconds,
        }
        for rank, count in tokens.items():
            metrics[f"{METRIC_PREFIX}/dp_rank_{rank}/tokens"] = float(count)
            metrics[f"{METRIC_PREFIX}/dp_rank_{rank}/mismatched_tokens"] = float(mismatched[rank])
        if mismatched.total():
            logger.info(
                "Exactness check: {} of {} log-probabilities the trainer computed for {} engines' check responses "
                "differ from the engines' (log_ratio_abs_max {:.3g})",
                mismatched.total(),
                tokens.total(),
                len(samples),
                largest,
            )
        return metrics
