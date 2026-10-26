"""An online count of the response tokens whose log-probability differs between the trainer and the rollout engines."""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.numerics import CheckFailure
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import ROLLOUT_ENGINE_DP_RANKS_KEY
from skyrl_train.weight_sync.base import WeightSyncReason

# The fixed prompt and response length of every check: a few decode steps on each engine.
CHECK_TEXT = "Every token's log-probability is the same in the trainer as in the inference engine that sampled it."
CHECK_RESPONSE_TOKENS = 16
CHECK_SEED = 0
METRIC_PREFIX = "exactness_check"


@dataclass(frozen=True)
class EngineSample:
    """One engine's response to the check prompt."""

    engine_dp_rank: int
    response_token_ids: list[int]
    response_logprobs: list[float]
    routed_experts: np.ndarray | None


class ExactnessCheck:
    """Engine responses to a fixed prompt after a weight sync, and their comparison with the trainer's scores."""

    def __init__(
        self,
        every_weight_syncs: int,
        on_failure: CheckFailure,
        sampling_params: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.every_weight_syncs = every_weight_syncs
        self.on_failure = on_failure
        self.tokenizer = tokenizer
        self.prompt_token_ids = tokenizer.encode(CHECK_TEXT, add_special_tokens=False)
        self.sampling_params = {
            **get_vllm_sampling_params(sampling_params),
            "max_tokens": CHECK_RESPONSE_TOKENS,
            "logprobs": 0,
            "seed": CHECK_SEED,
            "stop": None,
        }
        self._samples: list[EngineSample] | None = None
        self._generate_failures = 0
        self._training_syncs = 0
        self._generate_seconds = 0.0

    async def after_weight_sync(self, reason: WeightSyncReason, client: InferenceEngineClient) -> dict[str, float]:
        """After the first sync, the re-sync after a checkpoint save and every ``every_weight_syncs``-th training-step
        sync, take every live engine's response to the check prompt for ``score``. Under ``on_failure=log``, a
        generation during which an engine died is skipped and counted in ``exactness_check/generate_failures``."""
        if not self._checks_after(reason):
            return {}
        self._samples = None
        started = time.monotonic()
        try:
            outputs = await client.generate_on_live_engines(self.prompt_token_ids, self.sampling_params)
        except ray.exceptions.RayActorError as error:
            if self.on_failure is CheckFailure.STOP:
                raise
            self._generate_failures += 1
            logger.warning("Exactness check skipped: an engine died while generating the check responses: {}", error)
            return {f"{METRIC_PREFIX}/generate_failures": float(self._generate_failures)}
        self._generate_seconds = time.monotonic() - started
        self._samples = [
            EngineSample(
                engine_dp_rank=rank,
                response_token_ids=list(output["response_ids"][0]),
                response_logprobs=list(output["response_logprobs"][0]),
                routed_experts=(output.get("routed_experts") or [None])[0],
            )
            for rank, output in outputs
        ]
        return {}

    def _checks_after(self, reason: WeightSyncReason) -> bool:
        if reason in (WeightSyncReason.INITIAL, WeightSyncReason.AFTER_CHECKPOINT_SAVE):
            return True
        if reason != WeightSyncReason.TRAINING_STEP:
            return False
        self._training_syncs += 1
        return self._training_syncs % self.every_weight_syncs == 0

    def score(
        self, forward: Callable[[TrainingInputBatch], torch.Tensor], dp_size: int, num_experts: int | None
    ) -> dict[str, float]:
        """Score the engines' responses with ``forward``, the policy's ``[rows, response_length]`` log-probabilities of
        a batch, and count the response tokens whose float32 bits differ from the engine's, in total and per engine
        data-parallel rank; under ``on_failure=stop`` a mismatch raises. Returns no metrics when no response waits to
        be scored."""
        if self._samples is None:
            return {}
        samples, self._samples = self._samples, None
        started = time.monotonic()
        logprobs = forward(self._batch(samples, dp_size, num_experts))
        score_seconds = time.monotonic() - started
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
        if self.on_failure is CheckFailure.STOP and mismatched.total():
            raise RuntimeError(
                f"Exactness check: {mismatched.total()} of {tokens.total()} log-probabilities the trainer computed for "
                f"{len(samples)} engines' check responses differ from the engines' (log_ratio_abs_max {largest:.3g})"
            )
        return metrics

    def _batch(self, samples: list[EngineSample], dp_size: int, num_experts: int | None) -> TrainingInputBatch:
        """The responses as a policy forward batch, repeated to a multiple of ``dp_size`` rows, with each row's response
        tokens as its loss mask and its engine data-parallel rank."""
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
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": action_mask,
                ROLLOUT_ENGINE_DP_RANKS_KEY: torch.tensor([row.engine_dp_rank for row in rows], dtype=torch.long),
            },
            routed_expert_rows=(
                None if routes[0] is None else RoutedExpertRows(routes, action_mask.shape[1], num_experts)
            ),
        )
        batch.metadata = {"response_length": action_mask.shape[1]}
        return batch
