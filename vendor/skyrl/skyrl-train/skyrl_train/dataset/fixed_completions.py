"""Pre-generated (prompt, completion) rows for fixed-data SFT.

Fixed-data training (M1, expert building) reads expert completions from the training
parquet instead of generating rollouts. Text rows carry a message-list prompt and
completion text; tokenized rows carry a complete conversation with assistant masks so
tool observations and the expert's exact tokenization survive. Rows may carry a
``data_source`` task tag and a ``teacher_route`` for joint-run routing.
"""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from loguru import logger
from transformers import PreTrainedTokenizerBase

from marinskyrl.runtime_options import FIXED_COMPLETION_ENV_CLASS, FixedCompletionFormat
from skyrl_train.dataset.dataset import PromptDataset, read_prompt_files
from skyrl_train.dataset.preference_pairs import completion_text
from skyrl_train.task_inputs import FixedCompletionInput, TrajectoryMetadata


def tokenized_completion(row: Mapping[str, Any], *, max_sequence_length: int) -> tuple[list[int], list[int], list[int]]:
    """Split one complete tokenized conversation at its first supervised token.

    Returns prompt IDs, response IDs, and the response-aligned supervision mask. The
    response span keeps every token after the prompt — including unsupervised tool
    observations — so a tokenized row replays the conversation exactly.
    """
    ids = row["input_ids"]
    mask = row["assistant_masks"]
    if not ids or len(ids) != len(mask):
        raise ValueError("tokenized completion IDs and assistant masks must be nonempty and aligned")
    if any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in ids):
        raise ValueError("tokenized completion IDs must be nonnegative integers")
    if any(value not in (0, 1) for value in mask) or not any(mask):
        raise ValueError("tokenized completion requires a binary mask with supervised assistant tokens")
    if len(ids) > max_sequence_length:
        raise ValueError(f"tokenized completion exceeds the {max_sequence_length}-token sequence budget")
    first = list(mask).index(1)
    if first == 0:
        raise ValueError("tokenized completions require a nonempty unsupervised prompt prefix")
    return (
        [int(token) for token in ids[:first]],
        [int(token) for token in ids[first:]],
        [int(value) for value in mask[first:]],
    )


class FixedCompletionsDataset(PromptDataset):
    """Load text completions or validate complete tokenized conversations with assistant masks.

    Text rows are tokenized and length-filtered. Tokenized rows preserve all tokens,
    tool observations and supervision masks; invalid rows raise instead of being
    dropped. Per-row environment overrides are removed in both formats.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data_format: FixedCompletionFormat,
        max_completion_length: int,
        **kwargs,
    ):
        self.data_format = data_format
        self.max_completion_length = max_completion_length
        super().__init__(tokenizer=tokenizer, **kwargs)

    def collate_fn(self, rows) -> list[dict]:
        tokenized = self.data_format is FixedCompletionFormat.TOKENIZED
        inputs = []
        for prompt, _, extra, uid in rows:
            mask = extra.get("completion_assistant_masks") if tokenized else None
            inputs.append(
                {
                    "uid": uid,
                    "input": FixedCompletionInput(
                        prompt=prompt,
                        completion=extra["completion_input_ids"] if tokenized else extra["completion"],
                        metadata=TrajectoryMetadata(
                            uid=uid,
                            metric_class=FIXED_COMPLETION_ENV_CLASS,
                            data_source=extra.get("data_source"),
                            teacher_route=extra.get("teacher_route"),
                        ),
                        completion_loss_mask=tuple(mask) if mask is not None else None,
                    ),
                }
            )
        return inputs

    def _read_files_and_tokenize(self):
        if self.data_format is FixedCompletionFormat.TOKENIZED:
            self.dataframe = read_prompt_files(self.datasets)
            if "env_class" in self.dataframe.column_names:
                self.dataframe = self.dataframe.remove_columns("env_class")
            budget = self.max_prompt_length + self.max_completion_length

            def split(row):
                prompt_ids, response_ids, mask = tokenized_completion(row, max_sequence_length=budget)
                if len(response_ids) > self.max_completion_length:
                    raise ValueError(
                        f"tokenized completion response span needs {len(response_ids)} tokens but the "
                        f"{self.max_completion_length}-token response cap is exhausted; filter the row at dataset load instead"
                    )
                return {
                    self.prompt_key: prompt_ids,
                    "completion_input_ids": response_ids,
                    "completion_assistant_masks": mask,
                }

            self.dataframe = self.dataframe.map(
                split, num_proc=self.num_workers, desc="Validating tokenized completions"
            )
            self.dataframe = self.dataframe.remove_columns(("input_ids", "assistant_masks"))
            logger.info(
                "Loaded {} tokenized fixed completions without retokenization or filtering", len(self.dataframe)
            )
            return
        super()._read_files_and_tokenize()
        if "completion" not in self.dataframe.column_names:
            raise ValueError("fixed-completion text data requires a 'completion' column")
        if "env_class" in self.dataframe.column_names:
            self.dataframe = self.dataframe.remove_columns("env_class")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")

        tokenizer = self.tokenizer
        budget = self.max_completion_length

        def normalize(row):
            return {"completion": completion_text(row["completion"], "completion")}

        def fits(row) -> bool:
            return len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"]) <= budget

        self.dataframe = self.dataframe.map(normalize, num_proc=self.num_workers, desc="Normalizing completions")
        before = len(self.dataframe)
        self.dataframe = self.dataframe.filter(fits, num_proc=self.num_workers, desc="Filtering fixed completions")
        logger.info(
            "Fixed-completion dataset kept {} of {} rows within the {}-token response cap",
            len(self.dataframe),
            before,
            budget,
        )
