"""Measure prompts directly from source specs without executing an evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from eval.contracts.prompt_length import REFERENCE_TOKENIZER, BenchmarkPromptLength
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions

PLACEHOLDER_CREDENTIAL_VARIABLES = ("OPENAI_API_KEY", "JUDGE_API_KEY")
PLACEHOLDER_CREDENTIAL = "prompt-corpus-render"




def load_reference_tokenizer(name: str = REFERENCE_TOKENIZER) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name)




def render_prompt_corpus(task_name: str, tokenizer: PreTrainedTokenizerBase) -> BenchmarkPromptLength:
    """Measure every distinct prompt ``task_name`` renders, with no model call."""
    # Local to break the cycle: eval.task reads the stored lengths from
    # eval.contracts.prompt_length, which this module imports.
    from eval.task import TaskManager

    manager = TaskManager(task_list=[task_name])
    failure = manager.load_failures.get(task_name)
    if failure is not None:
        raise failure
    benchmark = manager.get_benchmark(task_name)
    options = SourceOptions(task_name, SourceRef(task_name), encode=tokenizer.encode)
    prompts = [request.prompt for spec in benchmark.iter_specs(options) for request in spec.requests]
    return measure_prompts(task_name, prompts, tokenizer)


def measure_prompts(
    task_name: str, prompts: Sequence[Any], tokenizer: PreTrainedTokenizerBase
) -> BenchmarkPromptLength:
    """Reduce captured prompts to the stored measurements."""
    if not prompts:
        raise ValueError(f"{task_name} rendered no prompts")
    rendered = sorted({_prompt_text(prompt, tokenizer) for prompt in prompts})
    digest = hashlib.sha256(json.dumps(rendered, ensure_ascii=False).encode("utf-8")).hexdigest()
    longest = max(rendered, key=len)
    return BenchmarkPromptLength(
        distinct_prompt_count=len(rendered),
        longest_prompt_chars=len(longest),
        longest_prompt_tokens=max(len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered),
        sha256=digest,
    )


def _prompt_text(prompt: Any, tokenizer: PreTrainedTokenizerBase) -> str:
    """Return one captured request as the text the endpoint will receive."""
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "prompt"):
        return str(prompt.prompt)
    if isinstance(prompt, Sequence) and all(isinstance(message, Mapping) for message in prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"unsupported rendered prompt: {type(prompt).__name__}")
