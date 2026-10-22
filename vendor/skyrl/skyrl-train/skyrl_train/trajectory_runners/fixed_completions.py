"""Static trajectory runner that emits dataset-supplied (prompt, completion) rows.

Fixed-data SFT (M1 and expert building) reads expert completions from the training
parquet instead of generating rollouts. Each row is one training sequence: the loss
trains the completion tokens through the configured policy loss, and batch geometry
comes from the same config fields as rollout training so the update budget matches.
Rows carry a task tag (``data_source``) for collision mass and an optional
``teacher_route`` for joint-run routing.
"""

from loguru import logger

from marinskyrl.runtime_options import FixedCompletionFormat
from skyrl_train.dataset.dataset import prompt_token_ids
from skyrl_train.dataset.preference_pairs import completion_text
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import (
    TrajectoryBatch,
    FixedCompletionRequestBatch,
)


class FixedCompletionsTrajectoryRunner:
    """Emit tokenized dataset completions without touching any inference engine."""

    def __init__(
        self,
        tokenizer,
        *,
        data_format: FixedCompletionFormat,
        max_generate_length: int,
        max_input_length: int,
        generator_config,
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
        logger.warning("fixed-completion evaluation replays the fixed dataset completions without generation")

    async def stop_eval_session(self) -> None:
        pass

    def _text_row(self, prompt, completion) -> tuple[list[int], list[int], list[int]]:
        """Tokenize one text row, failing loudly when it exceeds the rollout budget."""
        prompt_ids = prompt_token_ids(self.tokenizer, prompt)
        completion_ids = [int(token) for token in self.tokenizer(completion, add_special_tokens=False)["input_ids"]]
        if not completion_ids:
            raise ValueError("fixed completion tokenized to zero tokens")
        if len(completion_ids) > self.max_generate_length:
            raise ValueError(
                f"fixed completion needs {len(completion_ids)} tokens but "
                f"generator.sampling_params.max_generate_length is {self.max_generate_length}"
            )
        if len(prompt_ids) + len(completion_ids) > self.max_input_length + self.max_generate_length:
            raise ValueError(
                f"fixed completion exceeds the {self.max_input_length + self.max_generate_length}-token "
                "sequence budget; filter the row at dataset load instead"
            )
        return prompt_ids, completion_ids, [1] * len(completion_ids)

    def _tokenized_row(self, prompt, completion_ids, mask) -> tuple[list[int], list[int], list[int]]:
        """Validate one tokenized row against the rollout budget it would train under."""
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("tokenized fixed completions require the dataset's prompt token IDs")
        if len(completion_ids) != len(mask) or not any(mask):
            raise ValueError("tokenized fixed completions require an aligned mask with supervised tokens")
        if len(completion_ids) > self.max_generate_length:
            raise ValueError(
                f"fixed completion response span needs {len(completion_ids)} tokens but "
                f"generator.sampling_params.max_generate_length is {self.max_generate_length}"
            )
        if len(prompt) + len(completion_ids) > self.max_input_length + self.max_generate_length:
            raise ValueError(
                f"fixed completion exceeds the {self.max_input_length + self.max_generate_length}-token sequence budget"
            )
        return list(prompt), list(completion_ids), list(mask)

    def _validate_request(self, input_batch: FixedCompletionRequestBatch) -> None:
        prompts = input_batch.prompts
        completions = input_batch.completions
        metadata = input_batch.metadata
        if not (len(prompts) == len(completions) == len(metadata)):
            raise ValueError(
                f"fixed completions require aligned rows; got {len(prompts)} prompts, "
                f"{len(completions)} completions, and {len(metadata)} metadata records"
            )
        masks = input_batch.completion_loss_masks
        if masks is not None and len(masks) != len(completions):
            raise ValueError("fixed-completion loss masks must align with completions")
        if masks is None and self.data_format is FixedCompletionFormat.TOKENIZED:
            raise ValueError("tokenized fixed completions require per-row loss masks")
        for prompt, completion in zip(prompts, completions):
            if self.data_format is FixedCompletionFormat.TEXT:
                if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                    raise ValueError("text fixed completions require message-list prompts")
                if not isinstance(completion, str) and not (
                    isinstance(completion, list) and completion and isinstance(completion[-1], dict)
                ):
                    raise ValueError("text fixed completions require text or message-list completions")
            elif not (isinstance(prompt, list) and isinstance(completion, list)):
                raise ValueError("tokenized fixed completions require token-ID prompts and completions")

    async def run(self, input_batch: FixedCompletionRequestBatch) -> TrajectoryBatch:
        self._validate_request(input_batch)
        masks = input_batch.completion_loss_masks

        prompt_token_ids_rows: list[list[int]] = []
        response_ids: list[list[int]] = []
        loss_masks: list[list[int]] = []
        for index, (prompt, completion) in enumerate(zip(input_batch.prompts, input_batch.completions)):
            mask = None if masks is None else masks[index]
            if self.data_format is FixedCompletionFormat.TEXT:
                row = self._text_row(prompt, completion_text(completion, "completion"))
            else:
                row = self._tokenized_row(prompt, completion, list(mask or ()))
            prompt_token_ids_rows.append(row[0])
            response_ids.append(row[1])
            loss_masks.append(row[2])

        batch: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids_rows,
            "response_ids": response_ids,
            "data_sources": [row.data_source for row in input_batch.metadata],
            "rewards": [1.0] * len(response_ids),
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
            "stop_reasons": ["fixed_completion"] * len(response_ids),
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
        }
        return await finalize_trajectory_batch(input_batch, batch, self.generator_config, self.trajectory_sink)
