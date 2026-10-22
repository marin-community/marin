from types import SimpleNamespace
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs
from typing import List, Optional
import logging
from datasets import load_dataset

from .archive_data.utils import load_data, construct_trainable_data
from .data.utils import construct_prompt
from eval.task import BaseBenchmark


class RepoBenchmark(BaseBenchmark):
    """
    RepoBench benchmark implementation evaluating code completion capabilities
    across different programming languages and contexts.
    """

    def __init__(
        self,
        languages: List[str] = ["python", "java"],
        subsets: List[str] = ["cross_file_first", "cross_file_random", "in_file"],
        max_tokens: int = 128,
        debug: bool = False,
        legacy_mode: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize RepoBench benchmark.

        Args:
            languages: List of programming languages to evaluate
            subsets: List of dataset subsets to use
            max_tokens: Maximum number of tokens for generation
            debug: If true, run on debug mode using 2 samples
            legacy_mode: Whether to use legacy (v0) evaluation
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.languages = languages
        self.subsets = subsets
        self.max_tokens = max_tokens
        self.debug = debug
        self.legacy_mode = legacy_mode





    def iter_specs(self, options):
        encode = options.tokenizer_encode or options.encode
        tokenizer = SimpleNamespace(encode=encode) if encode is not None else None
        for language in self.languages:
            for subset in self.subsets:
                if self.legacy_mode:
                    rows = construct_trainable_data(load_data(split="test", task="completion", language=language, length="2k", settings=subset), language=language)
                    use_fim = "star" in (options.model_name or "")
                    prompt = lambda row: "<fim_prefix>" + row["data"] + "<fim_suffix><fim_middle>" if use_fim else row["data"]
                    target = lambda row: row["label"]
                else:
                    rows = load_dataset(f"tianyang/repobench_{language}_v1.1", verification_mode="no_checks")[subset]
                    prompt = lambda row: construct_prompt(row, tokenizer=tokenizer, max_token_nums=self.max_tokens, language=language)
                    target = lambda row: row["next_line"]
                if self.debug:
                    rows = list(rows)[:2]
                yield from text_specs(
                    rows, options, namespace=f"{language}:{subset}", prompt=prompt, target=target, prompt_format="completion",
                    grader="code_line", reference=lambda row: {"language": language},
                    generation_kwargs={"temperature": 0.2, "top_p": 0.95, "do_sample": True, "max_gen_toks": self.max_tokens},
                    tags=lambda row: {"subset": f"{language}_{subset}"},
                )
    SOURCE = SourceRef("tianyang/repobench_v1.1")
    METRICS = ("EM", "ES")
    PRIMARY_METRIC = "EM"
    REDUCER = ReducerSpec(trial_metric=None, groups={"subset": "{group}_{metric}"})
