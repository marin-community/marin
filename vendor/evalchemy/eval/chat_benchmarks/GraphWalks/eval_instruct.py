"""Revision-pinned GraphWalks with model-specific context selection."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from types import MappingProxyType

import pyarrow.parquet as pq
from eval.contracts.benchmark_metadata import MetricKind
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.execution.generation import canonical_generation_kwargs
from eval.graders.graphwalks import FINAL_ANSWER_PREFIX
from eval.limits import require_context_length
from eval.sources.common import SourceOptions, text_specs
from eval.task import BaseBenchmark
from rigging.filesystem.factory import open_url

GRAPHWALKS_DATASET = "openai/graphwalks"
GRAPHWALKS_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
GRAPHWALKS_FILES = ("graphwalks_128k_and_shorter.parquet", "graphwalks_256k_to_1mil.parquet")
_PARQUET_BATCH_SIZE = 32
_CONTEXT_MARGIN = 64
_PREFIX_TOKEN_MARGIN = 256
_PREFIX_CHARS_PER_TOKEN = 8
_MIN_OUTPUT_TOKENS = 4096
_REASONING_RESERVE = 4096
_OUTPUT_BUDGET_MULTIPLIER = 2
_MIN_OUTPUT_BUDGET = _OUTPUT_BUDGET_MULTIPLIER * _MIN_OUTPUT_TOKENS


class GraphWalksBenchmark(BaseBenchmark):
    """Score the context-eligible subset while retaining full benchmark coverage."""

    SOURCE = SourceRef(GRAPHWALKS_DATASET, split="train", revision=GRAPHWALKS_REVISION)
    METRICS = ("f1", "precision", "recall", "exact_match")
    PRIMARY_METRIC = "f1"
    REDUCER = ReducerSpec(trial_metric=None)
    METRIC_NAME_OVERRIDES = MappingProxyType({"exact_match": "exact_match"})
    METRIC_KIND_OVERRIDES = MappingProxyType({"exact_match": MetricKind.BINARY})

    def __init__(self, max_tokens=131072, logger=None, system_instruction=None, cache_dir=None):
        if system_instruction is not None:
            raise ValueError("GraphWalks uses the canonical dataset prompt without a system instruction")
        super().__init__(logger=logger)
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.selection = None

    def benchmark_size(self):
        return None if self.selection is None else self.selection["n_benchmark"]

    def describe(self, task_name=None):
        metadata = super().describe(task_name)
        return metadata if self.selection is None else replace(metadata, n_attempted=self.selection["n_attempted"])

    def evaluation_metadata(self):
        return {"selection": self.selection}

    def iter_specs(self, options: SourceOptions):
        context = require_context_length(options.context_length, task_name=options.task)
        if options.system_instruction is not None:
            raise ValueError("GraphWalks uses the canonical dataset prompt without a system instruction")
        if options.prompt_token_count is None:
            raise ValueError("GraphWalks requires the served model's tokenizer")
        generation = canonical_generation_kwargs(options.generation)
        cap = generation.pop("max_gen_toks", None) or options.max_tokens or self.max_tokens
        if context <= _MIN_OUTPUT_BUDGET + _CONTEXT_MARGIN or cap < _MIN_OUTPUT_BUDGET:
            raise ValueError(f"GraphWalks requires room for the prompt and at least {_MIN_OUTPUT_BUDGET} output tokens")
        with ExitStack() as stack:
            files = []
            for filename in GRAPHWALKS_FILES:
                path = fetch_hf(
                    GRAPHWALKS_DATASET,
                    filename,
                    revision=GRAPHWALKS_REVISION,
                    cache_prefix=self.dataset_cache or self.cache_dir,
                )
                files.append(pq.ParquetFile(stack.enter_context(open_url(path, "rb"))))
            count = sum(file.metadata.num_rows for file in files)
            rows = (
                row
                for file in files
                for batch in file.iter_batches(batch_size=_PARQUET_BATCH_SIZE)
                for row in batch.to_pylist()
            )
            selected = self._selected_rows(rows, count, options, context, cap)
        # The run cap selects eligible examples; each request retains its full gold-dependent budget.
        selected_options = replace(options, limit=None, max_tokens=None, generation=generation)
        for spec in text_specs(
            selected,
            selected_options,
            prompt=lambda row: row["prompt"],
            target=lambda row: row["answer_nodes"],
            grader="graphwalks",
            generation_kwargs=lambda row: {
                "max_gen_toks": row["max_output_tokens"],
                "temperature": 0.0,
            },
        ):
            request = replace(spec.requests[0], output_token_cap=spec.doc["max_output_tokens"])
            yield replace(spec, requests=(request,))

    def _selected_rows(self, rows, count, options, context, cap):
        limit = options.limit
        if limit is not None:
            limit = int(limit * count) if limit < 1 else int(limit)
        selected = []
        context_skips, output_skips = Counter(), Counter()
        capped = 0
        for index, row in enumerate(rows):
            if limit is not None and len(selected) >= limit:
                capped = count - index
                break
            answer = FINAL_ANSWER_PREFIX + " [" + ", ".join(row["answer_nodes"]) + "]"
            answer_tokens = options.prompt_token_count(answer)
            budget = _OUTPUT_BUDGET_MULTIPLIER * max(_MIN_OUTPUT_TOKENS, answer_tokens + _REASONING_RESERVE)
            if budget > cap:
                output_skips[row["problem_type"]] += 1
                continue
            prompt_budget = context - budget - _CONTEXT_MARGIN
            if prompt_budget <= 0:
                context_skips[row["problem_type"]] += 1
                continue
            messages = [{"role": "user", "content": row["prompt"]}]
            if row["prompt_chars"] > prompt_budget * _PREFIX_CHARS_PER_TOKEN:
                prefix = [{"role": "user", "content": row["prompt"][: prompt_budget * _PREFIX_CHARS_PER_TOKEN]}]
                if options.prompt_token_count(prefix) > prompt_budget + _PREFIX_TOKEN_MARGIN:
                    context_skips[row["problem_type"]] += 1
                    continue
            tokens = options.prompt_token_count(messages)
            if tokens > prompt_budget:
                context_skips[row["problem_type"]] += 1
                continue
            selected.append({**row, "source_index": index, "prompt_tokens": tokens, "max_output_tokens": budget})
        self.selection = {
            "dataset": GRAPHWALKS_DATASET,
            "revision": GRAPHWALKS_REVISION,
            "n_benchmark": count,
            "n_attempted": len(selected),
            "skipped_context_by_type": dict(context_skips),
            "skipped_output_cap_by_type": dict(output_skips),
            "not_inspected_after_limit": capped,
            "max_model_len": context,
            "max_output_tokens": cap,
            "chat_template_kwargs": dict(options.chat_template),
            "output_budget_policy": "2 * max(4096, tokenized_gold_list_length + 4096)",
        }
        if not selected:
            raise ValueError("No GraphWalks examples fit the served context and output cap")
        return selected
