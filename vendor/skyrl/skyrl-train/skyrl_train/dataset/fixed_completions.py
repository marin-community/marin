"""Pre-generated (prompt, completion) rows for fixed-data SFT.

Fixed-data SFT reads expert completions from the training parquet instead of
generating rollouts. Text rows carry a message-list prompt and completion text;
tokenized rows carry a complete conversation with assistant masks so tool
observations and the expert's exact tokenization survive. Rows may carry a
``data_source`` task tag (required) and an optional ``teacher_route`` for joint-run routing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from loguru import logger
from transformers import PreTrainedTokenizerBase

from marinskyrl.runtime_options import FIXED_COMPLETION_ENV_CLASS, FixedCompletionFormat
from skyrl_train.config.training_budget import ResponseCaps
from skyrl_train.dataset.dataset import ROUTE_COLUMN, PromptDataset, prompt_token_ids, read_prompt_files
from skyrl_train.dataset.preference_pairs import completion_text
from skyrl_train.task_inputs import FixedCompletionInput, TrajectoryMetadata


@dataclass(frozen=True)
class FixedCompletionTokens:
    """One row split at its first supervised token."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]


def tokenized_completion(row: Mapping[str, Any], *, max_sequence_length: int) -> FixedCompletionTokens:
    """Split one complete tokenized conversation at its first supervised token.

    The response span keeps every token after the prompt — including unsupervised
    tool observations — so a tokenized row replays the conversation exactly.
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
    return FixedCompletionTokens(
        prompt_ids=[int(token) for token in ids[:first]],
        response_ids=[int(token) for token in ids[first:]],
        loss_mask=[int(value) for value in mask[first:]],
    )


class FixedCompletionsDataset(PromptDataset):
    """Load text completions or validate complete tokenized conversations with assistant masks.

    Every row must fit its task's response cap and carry a nonempty ``data_source``; a row that
    does not raises instead of being dropped, so the rows trained are exactly the rows in the files.
    Text completions must end with the tokenizer's EOS token. Per-row environment overrides are
    removed in both formats.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data_format: FixedCompletionFormat,
        response_caps: ResponseCaps,
        **kwargs,
    ):
        self.data_format = data_format
        self.response_caps = response_caps
        super().__init__(tokenizer=tokenizer, **kwargs)

    def teacher_routes(self) -> set[str]:
        """The distinct ``teacher_route`` values; empty when rows carry no route."""
        if ROUTE_COLUMN not in self.dataframe.column_names:
            return set()
        return {str(route) for route in self.dataframe.unique(ROUTE_COLUMN)}

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
                            data_source=extra["data_source"],
                            teacher_route=extra.get(ROUTE_COLUMN),
                        ),
                        completion_loss_mask=tuple(mask) if mask is not None else None,
                    ),
                }
            )
        return inputs

    def _read_files_and_tokenize(self):
        self.dataframe = read_prompt_files(self.datasets)
        if "env_class" in self.dataframe.column_names:
            self.dataframe = self.dataframe.remove_columns("env_class")
        tokenized = self.data_format is FixedCompletionFormat.TOKENIZED
        required = ("input_ids", "assistant_masks") if tokenized else (self.prompt_key, "completion")
        missing = [column for column in (*required, "data_source") if column not in self.dataframe.column_names]
        if missing:
            raise ValueError(f"fixed-completion {self.data_format} data requires columns {missing}")
        if tokenized:
            self._split_tokenized_rows()
        else:
            self._validate_text_rows()
        logger.info("Loaded {} {} fixed completions without filtering", len(self.dataframe), self.data_format)

    def _split_tokenized_rows(self) -> None:
        caps = self.response_caps
        max_prompt_length = self.max_prompt_length

        def split(row, index):
            data_source = _data_source(row, index)
            cap = caps.for_source(data_source)
            tokens = tokenized_completion(row, max_sequence_length=max_prompt_length + cap)
            if len(tokens.response_ids) > cap:
                raise ValueError(
                    f"tokenized fixed-completion row {index} ({data_source}) has a {len(tokens.response_ids)}-token "
                    f"response span over its {cap}-token response cap"
                )
            return {
                self.prompt_key: tokens.prompt_ids,
                "completion_input_ids": tokens.response_ids,
                "completion_assistant_masks": tokens.loss_mask,
            }

        self.dataframe = self.dataframe.map(
            split, with_indices=True, num_proc=self.num_workers, desc="Validating tokenized completions"
        )
        self.dataframe = self.dataframe.remove_columns(("input_ids", "assistant_masks"))

    def _validate_text_rows(self) -> None:
        tokenizer = self.tokenizer
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("fixed-completion text rows require a tokenizer with an EOS token")
        caps = self.response_caps
        max_prompt_length = self.max_prompt_length
        prompt_key = self.prompt_key

        def validate(row, index):
            data_source = _data_source(row, index)
            prompt_length = len(prompt_token_ids(tokenizer, row[prompt_key]))
            if prompt_length > max_prompt_length:
                raise ValueError(
                    f"fixed-completion row {index} ({data_source}) has a {prompt_length}-token prompt over "
                    f"trainer.max_prompt_length={max_prompt_length}"
                )
            completion = completion_text(row["completion"], "completion")
            ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            cap = caps.for_source(data_source)
            if not ids or len(ids) > cap:
                raise ValueError(
                    f"fixed-completion row {index} ({data_source}) has a {len(ids)}-token completion outside "
                    f"its [1, {cap}]-token response cap"
                )
            if ids[-1] != eos_token_id:
                raise ValueError(
                    f"fixed-completion row {index} ({data_source}) does not end with the tokenizer EOS token "
                    f"{tokenizer.eos_token!r}; a completion that never ends trains the model not to stop"
                )
            return {"completion": completion}

        self.dataframe = self.dataframe.map(
            validate, with_indices=True, num_proc=self.num_workers, desc="Validating text completions"
        )


def _data_source(row: Mapping[str, Any], index: int) -> str:
    data_source = row["data_source"]
    if not isinstance(data_source, str) or not data_source:
        raise ValueError(f"fixed-completion row {index} requires a nonempty data_source task tag, got {data_source!r}")
    return data_source
