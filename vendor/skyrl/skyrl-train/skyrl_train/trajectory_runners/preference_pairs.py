"""Static trajectory runner that emits dataset-supplied preference pairs.

DPO reads chosen and rejected completions from the training parquet instead of
generating rollouts. Each prompt is leased with ``n_samples_per_prompt=2``; repetition
0 becomes the chosen row (+1 role) and repetition 1 the rejected row (-1 role), so the
trainer's group machinery delivers adjacent rows that the DPO loss pairs in order.
"""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from skyrl_train.dataset.dataset import prompt_token_ids
from skyrl_train.dataset.preference_pairs import PreferencePairFormat, completion_text, tokenized_pair
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import (
    TrajectoryBatch,
    PreferenceRequestBatch,
)


class PreferencePairTrajectoryRunner:
    """Emit tokenized dataset completions without touching any inference engine."""

    def __init__(
        self,
        tokenizer,
        *,
        data_format: PreferencePairFormat,
        max_generate_length: int,
        max_input_length: int,
        generator_config: Mapping[str, Any],
    ):
        self.tokenizer = tokenizer
        self.data_format = data_format
        self.max_generate_length = max_generate_length
        self.max_input_length = max_input_length
        self.generator_config = generator_config
        self.trajectory_sink: RetentionSink | None = None

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        batch = await self.run(task.request)
        return await write_trajectory_batch(task, writer, batch)

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        logger.warning("preference-pair evaluation replays the fixed dataset completions without generation")

    async def stop_eval_session(self) -> None:
        pass

    def _tokenize_pair(self, prompt, chosen: str, rejected: str) -> tuple[list[int], list[list[int]]]:
        """Tokenize one pair, failing loudly when either side exceeds the rollout budget."""
        prompt_ids = prompt_token_ids(self.tokenizer, prompt)
        completions = []
        for field, text in (("chosen", chosen), ("rejected", rejected)):
            completion_ids = [int(token) for token in self.tokenizer(text, add_special_tokens=False)["input_ids"]]
            if not completion_ids:
                raise ValueError(f"preference-pair {field} completion tokenized to zero tokens")
            if len(completion_ids) > self.max_generate_length:
                raise ValueError(
                    f"preference-pair {field} completion needs {len(completion_ids)} tokens but "
                    f"generator.sampling_params.max_generate_length is {self.max_generate_length}"
                )
            if len(prompt_ids) + len(completion_ids) > self.max_input_length + self.max_generate_length:
                raise ValueError(
                    f"preference-pair {field} completion exceeds the "
                    f"{self.max_input_length + self.max_generate_length}-token sequence budget; "
                    "filter the pair at dataset load instead"
                )
            completions.append(completion_ids)
        return prompt_ids, completions

    def _validate_request(self, input_batch: PreferenceRequestBatch) -> None:
        prompts = input_batch.prompts
        preferences = input_batch.preferences
        if len(prompts) % 2 or len(preferences) != len(prompts):
            raise ValueError(
                f"preference pairs arrive as chosen/rejected row pairs; got {len(prompts)} rows "
                f"with {len(preferences)} preference records"
            )
        trajectory_ids = input_batch.trajectory_ids
        if trajectory_ids is None:
            return
        for pair_index in range(len(prompts) // 2):
            chosen_id, rejected_id = trajectory_ids[2 * pair_index : 2 * pair_index + 2]
            if chosen_id.instance_id != rejected_id.instance_id or (
                chosen_id.repetition_id,
                rejected_id.repetition_id,
            ) != (0, 1):
                raise ValueError("preference pairs require trajectory IDs ordered as repetition 0 then 1")

    async def run(self, input_batch: PreferenceRequestBatch) -> TrajectoryBatch:
        self._validate_request(input_batch)
        prompts = input_batch.prompts
        preferences = input_batch.preferences

        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        pair_roles: list[int] = []
        for pair_index in range(len(prompts) // 2):
            prompt = prompts[2 * pair_index]
            preference = preferences[2 * pair_index]
            if prompts[2 * pair_index + 1] != prompt or preferences[2 * pair_index + 1] != preference:
                raise ValueError("Chosen and rejected rows must carry the same prompt and preference record")
            if self.data_format is PreferencePairFormat.TOKENIZED:
                pair = tokenized_pair(preference, max_sequence_length=self.max_input_length + self.max_generate_length)
                if pair.prompt_ids != prompt:
                    raise ValueError("Tokenized pair's initial prompt differs from the dataset prompt")
                prompt_ids, completions = pair.prompt_ids, pair.response_ids
                masks = pair.loss_masks
            else:
                prompt_ids, completions = self._tokenize_pair(
                    prompt,
                    completion_text(preference.get("chosen"), "chosen"),
                    completion_text(preference.get("rejected"), "rejected"),
                )
                masks = ([1] * len(completions[0]), [1] * len(completions[1]))
            prompt_token_ids.extend((prompt_ids, prompt_ids))
            response_ids.extend(completions)
            rewards.extend((1.0, 0.0))
            loss_masks.extend(masks)
            pair_roles.extend((1, -1))

        batch: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "data_sources": None,
            "rewards": rewards,
            "unshaped_rewards": None,
            "unshaped_reward_available": None,
            "reward_shaping_components": None,
            "reward_shaping_loop_spans": None,
            "loop_advantages": None,
            "reward_shaping_versions": None,
            "verification_results": [None] * len(response_ids),
            "evidence_messages": [None] * len(response_ids),
            "verifier_tests": None,
            "loss_masks": loss_masks,
            "stop_reasons": ["preference_pair"] * len(response_ids),
            "exception_types": None,
            "error_treatments": None,
            "server_errors": None,
            "rollout_metrics": {},
            "rollout_logprobs": None,
            "student_topk_indices": None,
            "behavior_topk_logprobs": None,
            "rollout_routed_experts": None,
            "teacher_evidence": None,
            "distillation": None,
            "token_level_shaping": None,
            "response_span_tags": None,
            "teacher_route_keys": None,
            "is_last_step": None,
            "exclude_from_baseline": None,
            "pair_roles": pair_roles,
        }
        return await finalize_trajectory_batch(input_batch, batch, self.generator_config, self.trajectory_sink)
