"""Static trajectory runner that emits dataset-supplied (prompt, completion) rows.

Fixed-data SFT reads expert completions from the training parquet instead of
generating rollouts. Each row is one training sequence: the loss trains the
completion tokens through the configured policy loss, and batch geometry comes
from the same config fields as rollout training so the update budget matches.
Rows carry a task tag (``data_source``) that selects their response cap and
attributes collision mass, and an optional ``teacher_route`` for joint-run routing.
"""

from loguru import logger

from marinskyrl.runtime_options import FixedCompletionFormat
from skyrl_train.config.training_budget import ResponseCaps
from skyrl_train.dataset.dataset import completion_token_ids, prompt_token_ids
from skyrl_train.dataset.fixed_completions import FixedCompletionTokens
from skyrl_train.dataset.preference_pairs import completion_text
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import (
    TrajectoryBatch,
    FixedCompletionRequestBatch,
    static_trajectory_batch,
)


class FixedCompletionsTrajectoryRunner:
    """Emit tokenized dataset completions without touching any inference engine."""

    def __init__(
        self,
        tokenizer,
        *,
        data_format: FixedCompletionFormat,
        response_caps: ResponseCaps,
        max_input_length: int,
        generator_config,
    ):
        self.tokenizer = tokenizer
        self.data_format = data_format
        self.response_caps = response_caps
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

    def _row(self, prompt, completion, mask, data_source: str) -> FixedCompletionTokens:
        """Budget-check one row against its task's response cap and the sequence budget it trains under."""
        cap = self.response_caps.for_source(data_source)
        if self.data_format is FixedCompletionFormat.TEXT:
            prompt_ids = prompt_token_ids(self.tokenizer, prompt)
            response_ids = completion_token_ids(
                self.tokenizer,
                completion_text(completion, "completion"),
                field=f"fixed ({data_source})",
                max_generate_length=cap,
            )
            loss_mask = [1] * len(response_ids)
        else:
            prompt_ids, response_ids, loss_mask = list(prompt), list(completion), list(mask or ())
            if not prompt_ids or len(response_ids) != len(loss_mask) or not any(loss_mask):
                raise ValueError(
                    "tokenized fixed completions require prompt IDs and an aligned mask with supervised tokens"
                )
            if len(response_ids) > cap:
                raise ValueError(
                    f"fixed completion ({data_source}) response span needs {len(response_ids)} tokens over "
                    f"its {cap}-token response cap"
                )
        if len(prompt_ids) + len(response_ids) > self.max_input_length + cap:
            raise ValueError(
                f"fixed completion ({data_source}) exceeds its {self.max_input_length + cap}-token "
                "sequence budget; reject the row at dataset load instead"
            )
        return FixedCompletionTokens(prompt_ids, response_ids, loss_mask)

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

    async def run(self, input_batch: FixedCompletionRequestBatch) -> TrajectoryBatch:
        self._validate_request(input_batch)
        masks = input_batch.completion_loss_masks

        rows = [
            self._row(prompt, completion, None if masks is None else masks[index], metadata.data_source)
            for index, (prompt, completion, metadata) in enumerate(
                zip(input_batch.prompts, input_batch.completions, input_batch.metadata)
            )
        ]
        batch = static_trajectory_batch(
            prompt_token_ids=[row.prompt_ids for row in rows],
            response_ids=[row.response_ids for row in rows],
            loss_masks=[row.loss_mask for row in rows],
            data_sources=[source.data_source for source in input_batch.metadata],
            rewards=[1.0] * len(rows),
            stop_reason="fixed_completion",
        )
        return await finalize_trajectory_batch(input_batch, batch, self.generator_config, self.trajectory_sink)
