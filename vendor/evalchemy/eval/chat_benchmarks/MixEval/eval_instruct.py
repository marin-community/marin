from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from .mix_eval.prompts.evaluation_prompts import construct_prompt_freeform, construct_prompt_multichoice
from typing import Any, Optional
from dataclasses import replace
import logging
import os
import json

import fsspec

from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.data_sources import fetch_hf
from eval.task import BaseBenchmark

HF_REVISION = "66a05ec3821a2b5089ed1e32a5d65ab6ecdbf906"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_eval", "data")


class MixEvalBenchmark(BaseBenchmark):
    """
    MixEval benchmark for evaluating language model responses on various tasks.
    """

    REQUIRES_JUDGE = True
    SOURCE = SourceRef("MixEval/MixEval", "MixEval", HF_REVISION)
    METRICS = ("overall",)
    PRIMARY_METRIC = "overall"
    REDUCER = ReducerSpec(name="rounded_split_mean", trial_metric=None)

    def __init__(
        self,
        benchmark: str = "mixeval",
        version: str = "2024-06-01",
        data_path: str | None = None,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MixEval benchmark.

        Args:
            benchmark: Benchmark name
            version: Benchmark version
            data_path: Path to evaluation data
            annotator_model: Model selected for the judge requests
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        if annotator_model == AUTO_ANNOTATOR_MODEL:
            annotator_model = "gpt-3.5-turbo-0125"
        self.judge_model = annotator_model
        self.benchmark = benchmark
        self.version = version
        self.data_path = data_path
        self.debug = debug
        self.dataset_split = "MixEval_Hard" if benchmark == "mixeval_hard" else "MixEval"
        if version != "2024-06-01":
            self.dataset_name = "MixEval"
            self.dataset_split = "test"
            self.dataset_revision = version

    def load_split(self, filename: str) -> list[dict[str, Any]]:
        """Load one split from the pinned June source or the selected local snapshot."""
        folder = "mixeval-hard" if self.benchmark == "mixeval_hard" else "mixeval"
        if self.version == "2024-06-01" and self.data_path is None:
            path = fetch_hf(
                self.SOURCE.dataset, f"data/mixeval-jsonl/{folder}/{filename}l",
                revision=HF_REVISION, cache_prefix=self.dataset_cache,
            )
            with fsspec.open(path, "rt", encoding="utf-8") as data:
                rows = [json.loads(line) for line in data]
            for row in rows:
                if row["context"] == "None":
                    row["context"] = None
                if filename == "free-form.json":
                    del row["options"]
        else:
            path = os.path.join(self.data_path or DEFAULT_DATA_PATH, f"mixeval-{self.version}", folder, filename)
            with fsspec.open(path, "rt", encoding="utf-8") as data:
                rows = [{**row, "id": identity} for identity, row in json.load(data).items()]
        return rows[:2] if self.debug else rows

    def validate_prepared_data(self) -> None:
        for filename, formatter in [("free-form.json", construct_prompt_freeform),
                                    ("multiple-choice.json", construct_prompt_multichoice)]:
            rows = self.load_split(filename)
            expected = 2 if self.debug else (500 if self.benchmark == "mixeval_hard" else 2000)
            if self.data_path is None and len(rows) != expected:
                raise ValueError(f"MixEval expected {expected} {filename} rows, found {len(rows)}")
            if not rows or any(not formatter(row).strip() or not row["target"] or not row["id"] for row in rows):
                raise ValueError("MixEval requires nonempty prompts, targets, and IDs")

    def iter_specs(self, options):
        options = replace(options, source=SourceRef(self.dataset_name, self.dataset_split, self.dataset_revision))
        for split, filename, formatter in [("freeform", "free-form.json", construct_prompt_freeform),
                                           ("multichoice", "multiple-choice.json", construct_prompt_multichoice)]:
            rows = self.load_split(filename)
            yield from text_specs(
                rows, options, namespace=split, prompt=formatter, target=lambda row: row["target"],
                grader="mixeval", reference=lambda row: {"kind": split},
                generation_kwargs={"temperature": 1.0, "do_sample": True},
                seed_repeats=False,
                tags=lambda row: {
                    "split": split,
                    "benchmark_split": f"{row['benchmark_name']}|{split}",
                },
            )
