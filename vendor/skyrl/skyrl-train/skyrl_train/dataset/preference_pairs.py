"""Text completions or full assistant-masked conversations for preference training."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from typing import Any

from loguru import logger
from transformers import PreTrainedTokenizerBase
from marinskyrl.runtime_options import PREFERENCE_PAIR_ENV_CLASS

from skyrl_train.dataset.dataset import PromptDataset, read_prompt_files
from skyrl_train.task_inputs import PreferenceInput, TrajectoryMetadata


class PreferencePairFormat(StrEnum):
    TEXT = "text"
    TOKENIZED = "tokenized"


@dataclass(frozen=True)
class PreferencePairTokens:
    prompt_ids: list[int]
    response_ids: tuple[list[int], list[int]]
    loss_masks: tuple[list[int], list[int]]


def tokenized_pair(row: Mapping[str, Any], *, max_sequence_length: int) -> PreferencePairTokens:
    """Split complete branches at their shared, initially unmasked token prefix."""
    tokens = []
    masks = []
    first_assistant = []
    for role in ("chosen", "rejected"):
        ids = row[f"{role}_input_ids"]
        mask = row[f"{role}_assistant_masks"]
        if not ids or len(ids) != len(mask):
            raise ValueError(f"tokenized {role} IDs and assistant masks must be nonempty and aligned")
        if any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in ids):
            raise ValueError(f"tokenized {role} IDs must be nonnegative integers")
        if any(value not in (0, 1) for value in mask) or not any(mask):
            raise ValueError(f"tokenized {role} requires a binary mask with supervised assistant tokens")
        if len(ids) > max_sequence_length:
            raise ValueError(f"tokenized {role} exceeds the {max_sequence_length}-token sequence budget")
        tokens.append([int(token) for token in ids])
        masks.append([int(value) for value in mask])
        first_assistant.append(mask.index(1))
    prefix = 0
    for position in range(min(first_assistant)):
        if tokens[0][position] != tokens[1][position]:
            break
        prefix += 1
    if prefix == 0:
        raise ValueError("tokenized preferences require an identical nonempty unmasked initial prompt")
    return PreferencePairTokens(
        tokens[0][:prefix],
        (tokens[0][prefix:], tokens[1][prefix:]),
        (masks[0][prefix:], masks[1][prefix:]),
    )


def completion_text(value: Any, field: str) -> str:
    """Normalize a completion column to text (the assistant turn's content for message lists)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if sum(isinstance(message, dict) and message.get("role") == "assistant" for message in value) > 1:
            raise ValueError("Multi-turn preference completions require the tokenized pair format and assistant masks")
        last = value[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    raise ValueError(f"preference-pair {field} completions must be text or message lists with assistant content")


class PreferencePairDataset(PromptDataset):
    """Load text pairs or validate complete tokenized conversations with assistant masks.

    Text pairs are tokenized and length-filtered. Tokenized pairs preserve all tokens,
    tool observations and supervision masks; invalid rows raise instead of being dropped.
    Per-row environment overrides are removed in both formats.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data_format: PreferencePairFormat,
        max_completion_length: int,
        **kwargs,
    ):
        self.data_format = data_format
        self.max_completion_length = max_completion_length
        super().__init__(tokenizer=tokenizer, **kwargs)

    def collate_fn(self, rows) -> list[dict]:
        inputs = []
        for prompt, _, preference, uid in rows:
            extra_info = preference.get("extra_info") or {}
            inputs.append(
                {
                    "uid": uid,
                    "input": PreferenceInput(
                        prompt=prompt,
                        preference=preference,
                        metadata=TrajectoryMetadata(
                            uid=uid,
                            metric_class=PREFERENCE_PAIR_ENV_CLASS,
                            data_source=extra_info.get("data_source") or preference.get("data_source"),
                            teacher_route=preference.get("teacher_route"),
                        ),
                    ),
                }
            )
        return inputs

    def _read_files_and_tokenize(self):
        if self.data_format is PreferencePairFormat.TOKENIZED:
            self.dataframe = read_prompt_files(self.datasets)
            if "env_class" in self.dataframe.column_names:
                self.dataframe = self.dataframe.remove_columns("env_class")
            budget = self.max_prompt_length + self.max_completion_length

            def validate(row):
                pair = tokenized_pair(row, max_sequence_length=budget)
                return {self.prompt_key: pair.prompt_ids}

            self.dataframe = self.dataframe.map(validate, num_proc=self.num_workers, desc="Validating tokenized pairs")
            logger.info("Loaded {} tokenized preference pairs without retokenization or filtering", len(self.dataframe))
            return
        super()._read_files_and_tokenize()
        for column in ("chosen", "rejected"):
            if column not in self.dataframe.column_names:
                raise ValueError(f"preference-pair training data requires a {column!r} column")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        if "env_class" in self.dataframe.column_names:
            self.dataframe = self.dataframe.remove_columns("env_class")

        tokenizer = self.tokenizer
        budget = self.max_completion_length

        def normalize(row):
            return {
                "chosen": completion_text(row["chosen"], "chosen"),
                "rejected": completion_text(row["rejected"], "rejected"),
            }

        def fits(row) -> bool:
            for column in ("chosen", "rejected"):
                if len(tokenizer(row[column], add_special_tokens=False)["input_ids"]) > budget:
                    return False
            return True

        self.dataframe = self.dataframe.map(normalize, num_proc=self.num_workers, desc="Normalizing completions")
        before = len(self.dataframe)
        self.dataframe = self.dataframe.filter(fits, num_proc=self.num_workers, desc="Filtering preference pairs")
        logger.info(
            "Preference-pair dataset kept {} of {} rows within the {}-token completion budget",
            len(self.dataframe),
            before,
            budget,
        )
