"""Original GenomeQA questions and externally loaded, revision-pinned prompt."""

import ast
import hashlib
from collections import Counter
from dataclasses import replace
from string import ascii_uppercase

import pyarrow.parquet as pq
from huggingface_hub.constants import HF_HUB_CACHE
from rigging.filesystem.factory import open_url

from eval.contracts.benchmark_metadata import MetricKind
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf, fetch_url
from eval.execution.generation import canonical_generation_kwargs
from eval.limits import ensure_context_window, require_context_length
from eval.sources.common import SourceOptions, text_specs
from eval.task import BaseBenchmark

DATASET = "HuggingFaceBio/genomeqa"
REVISION = "e655e415fa30681dddbaf4acc182bcbd8ac41943"
PROMPT_REVISION = "7ccc230e734fc11a365fd1963b593b5fd11e834d"
PROMPT_URL = f"https://raw.githubusercontent.com/ai4nucleome/GenomeQA/{PROMPT_REVISION}/run.py"
SUBTASKS = (
    "1_enhancer_vs_promoter",
    "2_splice_sites",
    "3_species_cls",
    "4_histone_marks",
    "5_tfbs",
    "6_motif",
)
FORMATS = ("binary", "mcq")
CELL_COUNTS = {f"{task}/{format}": 100 if task == "6_motif" else 500 for task in SUBTASKS for format in FORMATS}
FILE_HASHES = {
    "binary": "0ef4c3d33541980e3fded9a5b23a952296ce3c51cb0415b42c0299a61c21938a",
    "mcq": "c1cbb51449eb8a5241890337c88dc1bc0a58974b8eb4cb8bc49f310b12faeb75",
}
MAX_TOKENS = 16384
GENERATION = {"temperature": 0.7, "top_p": 0.95, "do_sample": True}


def load_system_prompt(cache_prefix: str) -> str:
    """Read only the pinned literal; never execute the upstream Python module."""
    path = fetch_url(PROMPT_URL, version=PROMPT_REVISION, cache_prefix=cache_prefix)
    with open_url(path, "r") as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SIMPLE_SYSTEM_PROMPT" for target in node.targets
        ):
            prompt = ast.literal_eval(node.value)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("GenomeQA system prompt must be a nonempty string literal")
            return prompt
    raise ValueError("Pinned GenomeQA runner has no SIMPLE_SYSTEM_PROMPT literal")


def render_question(row: dict) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(ascii_uppercase, row["options"], strict=False))
    return f"Question:\n{row['question']}\n\nOptions:\n{options}\n\nPlease select the correct option."


class GenomeQABenchmark(BaseBenchmark):
    SOURCE = SourceRef(DATASET, "train", REVISION)
    METRICS = (
        "genomeqa_macro",
        "accuracy",
        *(f"{group}/genomeqa_macro" for group in (*SUBTASKS, *FORMATS)),
        *(f"{cell}/accuracy" for cell in CELL_COUNTS),
    )
    PRIMARY_METRIC = "genomeqa_macro"
    METRIC_KIND_OVERRIDES = {"genomeqa_macro": MetricKind.CONTINUOUS}
    REDUCER = ReducerSpec(
        groups={tag: "{group}/{metric}" for tag in ("cell", "subtask", "format")},
        trial_metric=None,
    )
    CELLS = CELL_COUNTS

    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.prompt_sha256 = None

    def benchmark_size(self):
        return sum(self.CELLS.values())

    def iter_specs(self, options: SourceOptions):
        if options.limit is not None:
            raise ValueError("GenomeQA requires a complete population; use GenomeQAMotifMCQ for the 100-item check")
        if options.num_samples != 1:
            raise ValueError("GenomeQA requires exactly one response per question")
        if options.system_instruction:
            raise ValueError("GenomeQA uses the pinned upstream system prompt")
        generation = canonical_generation_kwargs(options.generation)
        for key, expected in GENERATION.items():
            if generation.get(key, expected) != expected:
                raise ValueError(f"GenomeQA requires {key}={expected}; conflicting generation override")
        if options.max_tokens not in (None, MAX_TOKENS) or generation.get("max_gen_toks", MAX_TOKENS) != MAX_TOKENS:
            raise ValueError("GenomeQA requires a shared 16384-token generation cap")
        context = require_context_length(options.context_length, task_name=options.task)
        if options.prompt_token_count is None:
            raise ValueError("GenomeQA requires the served tokenizer for full-prompt context preflight")
        prompt = load_system_prompt(self.dataset_cache or HF_HUB_CACHE)
        self.prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        configured = replace(options, system_instruction=prompt, max_tokens=MAX_TOKENS)
        prepared = []
        for spec in text_specs(
            self.load_questions(),
            configured,
            prompt=render_question,
            target=lambda row: ascii_uppercase[row["answer_index"]],
            grader="genomeqa",
            source_id=lambda row: f"{row['format']}/{row['subtask']}/{row['id']}",
            generation_kwargs=GENERATION,
            seed_repeats=False,
            reference=lambda row: {
                "options": ascii_uppercase[: len(row["options"])],
                "cell_weight": max(CELL_COUNTS.values()) / CELL_COUNTS[f"{row['subtask']}/{row['format']}"],
            },
            tags=lambda row: {
                "cell": f"{row['subtask']}/{row['format']}",
                "subtask": row["subtask"],
                "format": row["format"],
            },
        ):
            ensure_context_window(
                context_length=context,
                prompt_tokens=options.prompt_token_count(spec.requests[0].prompt),
                output_tokens=MAX_TOKENS,
            )
            prepared.append(replace(spec, requests=(replace(spec.requests[0], output_token_cap=MAX_TOKENS),)))
        # Validate the entire selected population before allowing any model request.
        yield from prepared

    def load_questions(self):
        counts = Counter()
        seen = set()
        for format in FORMATS:
            if not any(cell.endswith(f"/{format}") for cell in self.CELLS):
                continue
            path = fetch_hf(
                DATASET, f"{format}/train-00000-of-00001.parquet", revision=REVISION, cache_prefix=self.dataset_cache
            )
            with open_url(path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != FILE_HASHES[format]:
                    raise ValueError(f"GenomeQA {format} checksum mismatch")
                stream.seek(0)
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        cell = f"{row['subtask']}/{format}"
                        if cell not in CELL_COUNTS:
                            raise ValueError(f"Unknown GenomeQA cell: {cell}")
                        if cell not in self.CELLS:
                            continue
                        key = (format, row["subtask"], row["id"])
                        if key in seen:
                            raise ValueError(f"Duplicate GenomeQA identity: {key}")
                        seen.add(key)
                        choices = row["options"]
                        answer = row["answer_index"]
                        if len(choices) != (2 if format == "binary" else 4) or not 0 <= answer < len(choices):
                            raise ValueError(f"Invalid GenomeQA options or answer: {key}")
                        if choices[answer] != row["answer_text"]:
                            raise ValueError(f"Inconsistent GenomeQA answer: {key}")
                        counts[cell] += 1
                        yield {**row, "format": format}
        if counts != self.CELLS:
            raise ValueError(f"GenomeQA population differs from pinned counts: {counts}")

    def evaluation_metadata(self):
        return {
            "genomeqa_protocol": {
                "dataset_revision": REVISION,
                "prompt_revision": PROMPT_REVISION,
                "prompt_sha256": self.prompt_sha256,
                "dataset_license": "unspecified",
                "prompt_source": PROMPT_URL,
            }
        }
