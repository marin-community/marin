from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
"""OpenAI MRCR benchmark using the corrected, revision-pinned dataset."""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import tiktoken
from datasets import load_dataset

from eval.limits import (
    ContextWindowExceededError,
    encoded_token_count,
    message_content_token_count,
    require_context_length,
)
from eval.task import BaseBenchmark

DATASET_NAME = "openai/mrcr"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
DATA_FILES = (
    "2needle/2needle_0.parquet",
    "2needle/2needle_1.parquet",
    "4needle/4needle_0.parquet",
    "4needle/4needle_1.parquet",
    "8needle/8needle_0.parquet",
    "8needle/8needle_1.parquet",
)
MRCR_BIN_UPPER_BOUNDS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576)
MRCR_NEEDLE_COUNTS = (2, 4, 8)
_OFFICIAL_TOKENIZER = tiktoken.get_encoding("o200k_base")
PreparedPrompt = Union[List[Dict[str, str]], str]
SelectedExample = tuple[Dict[str, Any], PreparedPrompt]


def mrcr_bin(total_tokens: int) -> Optional[int]:
    """Return the published MRCR bin upper bound for a token count."""
    lower = 4_096
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower <= total_tokens <= upper:
            return upper
        lower = upper + 1
    return None


def parse_prompt(prompt: str) -> List[Dict[str, str]]:
    """Parse and validate the dataset's JSON-encoded chat messages."""
    messages = json.loads(prompt)
    if not isinstance(messages, list):
        raise ValueError(f"MRCR prompt must be a message list, got {type(messages).__name__}")
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("MRCR prompt messages must contain string role and content fields")
        if message["role"] == "assistant":
            message.setdefault("reasoning_content", "")
    return messages


def official_token_count(example: Dict[str, Any]) -> int:
    """Count prompt content and answer with OpenAI MRCR's o200k tokenizer."""
    messages = parse_prompt(str(example["prompt"]))
    return message_content_token_count(_OFFICIAL_TOKENIZER.encode, messages) + encoded_token_count(
        _OFFICIAL_TOKENIZER.encode, str(example["answer"])
    )




def _interleave_cells(
    cells: Dict[tuple[int, int], List[SelectedExample]],
    desired_cells: List[tuple[int, int]],
    limit: Optional[int],
) -> List[SelectedExample]:
    ordered: List[SelectedExample] = []
    rounds = max((len(examples) for examples in cells.values()), default=0)
    for position in range(rounds):
        for cell in desired_cells:
            if position < len(cells[cell]):
                ordered.append(cells[cell][position])
                if limit is not None and len(ordered) == limit:
                    return ordered
    return ordered


class MRCRBenchmark(BaseBenchmark):
    """OpenAI's multi-round co-reference resolution long-context benchmark."""

    def __init__(
        self,
        n_needles: Optional[List[int]] = None,
        debug: bool = False,
        seed: Optional[List[int]] = None,
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if system_instruction is not None:
            raise ValueError("MRCR does not accept a system instruction because it changes the canonical prompt")
        super().__init__(logger=logger)
        self.n_needles = tuple(n_needles) if n_needles is not None else MRCR_NEEDLE_COUNTS
        unknown_needles = set(self.n_needles) - set(MRCR_NEEDLE_COUNTS)
        if unknown_needles:
            raise ValueError(f"unsupported MRCR needle counts: {sorted(unknown_needles)}")
        self.debug = debug
        self.seed = list(seed) if seed is not None else [0, 1234, 1234, 1234]
        self.max_new_tokens = max_tokens
        self.cache_dir = cache_dir if cache_dir is not None else os.environ.get("HF_HUB_CACHE")

    def _load_rows(self, n_needles: Optional[int] = None) -> Iterable[Dict[str, Any]]:
        data_files = [path for path in DATA_FILES if n_needles is None or path.startswith(f"{n_needles}needle/")]
        dataset = load_dataset(
            DATASET_NAME,
            revision=DATASET_REVISION,
            data_files=data_files,
            streaming=True,
            cache_dir=self.cache_dir,
        )["train"]
        return iter(dataset)

    def _collect_cells(
        self,
        desired_cells: List[tuple[int, int]],
        limit: Optional[int] = None,
    ) -> Dict[tuple[int, int], List[SelectedExample]]:
        cells: Dict[tuple[int, int], List[SelectedExample]] = defaultdict(list)
        desired_cell_set = set(desired_cells)
        # A sparse cell can exhaust before the other cells, so any cell may supply the whole cap.
        per_cell_limit = int(limit) if limit is not None and limit >= 1 else None
        # Read each needle partition separately so capped runs stop downloading completed partitions.
        partitions = self.n_needles if per_cell_limit is not None else (None,)
        for partition in partitions:
            partition_cells = [cell for cell in desired_cells if partition is None or cell[1] == partition]
            for source in self._load_rows(partition):
                needles = int(source["n_needles"])
                if needles not in self.n_needles:
                    continue
                upper = mrcr_bin(official_token_count(source))
                if upper is None:
                    continue
                cell = (upper, needles)
                if cell not in desired_cell_set:
                    continue
                if per_cell_limit is not None and len(cells[cell]) >= per_cell_limit:
                    continue

                example = dict(source)
                messages = parse_prompt(str(example["prompt"]))
                example["mrcr_bin_upper"] = upper
                cells[cell].append((example, messages))
                if per_cell_limit is not None and all(len(cells[key]) >= per_cell_limit for key in partition_cells):
                    break
        return cells

    def _selected_examples(self, options) -> List[SelectedExample]:
        context_length = require_context_length(options.context_length, task_name="MRCR")
        eligible_bins = tuple(upper for upper in MRCR_BIN_UPPER_BOUNDS if upper <= context_length)
        desired_cells = [(upper, needles) for upper in eligible_bins for needles in self.n_needles]
        if not desired_cells:
            raise ContextWindowExceededError(
                context_length=context_length,
                required_tokens=MRCR_BIN_UPPER_BOUNDS[0],
            )

        limit = options.limit
        if self.debug:
            limit = min(limit, 2) if limit is not None else 2
        cells = self._collect_cells(desired_cells, limit)
        missing_cells = [cell for cell in desired_cells if not cells[cell]]
        if missing_cells:
            raise ValueError(f"pinned MRCR dataset has no examples for requested cells: {missing_cells}")
        ordered = _interleave_cells(cells, desired_cells, limit)

        self.logger.info(
            "Selected %d MRCR examples across %d cells",
            len(ordered),
            len({(example["mrcr_bin_upper"], int(example["n_needles"])) for example, _ in ordered}),
        )
        return ordered




    SOURCE = SourceRef(DATASET_NAME, split="train", revision=DATASET_REVISION)
    METRICS = ("mrcr_accuracy",)
    PRIMARY_METRIC = "mrcr_accuracy"
    REDUCER = ReducerSpec(trial_metric=None, accuracy="mrcr_accuracy")

    def iter_specs(self, options):
        rows = (row for row, _ in self._selected_examples(options))
        return text_specs(
            rows, options, prompt=lambda row: parse_prompt(str(row["prompt"])), target=lambda row: row["answer"],
            grader="nonce_similarity", reference=lambda row: {"nonce": row["random_string_to_prepend"]},
            generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_new_tokens},
            tags=lambda row: {"cell": f"{row['mrcr_bin_upper']}_{row['n_needles']}needle"},
        )
