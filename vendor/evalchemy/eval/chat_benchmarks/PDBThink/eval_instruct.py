"""Original PDBThink prompts, repeated completions, and exact context budgeting."""

from dataclasses import replace
from typing import TypedDict

from eval.chat_benchmarks.PDBThink.data import PDBTHINK_REVISION, load_bundle, manifest
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.execution.generation import canonical_generation_kwargs
from eval.limits import ContextWindowExceededError, require_context_length, safe_generation_cap
from eval.sources.common import SourceOptions, text_specs
from eval.task import BaseBenchmark


class RenderBudget(TypedDict):
    render_id: str
    input_tokens: int
    output_tokens: int


class PDBThinkBenchmark(BaseBenchmark):
    SOURCE = SourceRef("Open-Athena/pdbthink", "candidates_v1", PDBTHINK_REVISION)
    METRICS = ("macro_score", "micro_score")
    PRIMARY_METRIC = "macro_score"
    REDUCER = ReducerSpec(name="pdbthink", trial_metric=None)

    def __init__(self, *, logger=None, system_instruction=None):
        super().__init__(logger=logger, system_instruction=system_instruction)
        if system_instruction is not None:
            raise ValueError("PDBThink preserves its versioned system prompt")
        self.n_repeat = 3
        self._bundle: dict | None = None
        self._budgets: list[RenderBudget] = []

    def benchmark_size(self) -> int:
        return manifest()["n_renders"]

    def validate_prepared_data(self) -> None:
        if self.dataset_cache is None:
            raise ValueError("PDBThink requires --dataset_cache containing the prepared evaluation bundle")
        self._bundle = load_bundle(self.dataset_cache)

    def evaluation_metadata(self) -> dict:
        return {"pdbthink": {**manifest(), "budgets": self._budgets}}

    def iter_specs(self, options: SourceOptions):
        if self._bundle is None:
            self.validate_prepared_data()
        assert self._bundle is not None
        context = require_context_length(options.context_length, task_name=options.task)
        if options.prompt_token_count is None:
            raise ValueError("PDBThink requires the served model's tokenizer")
        if options.system_instruction is not None:
            raise ValueError("PDBThink preserves its versioned system prompt")
        generation = canonical_generation_kwargs(options.generation)
        requested_cap = generation.pop("max_gen_toks", options.max_tokens)
        if generation.pop("tools", None) or generation.get("tool_choice", "none") != "none":
            raise ValueError("PDBThink requires no tools and tool_choice=none")
        selected_options = replace(options, max_tokens=None, generation=generation)
        specs = text_specs(
            self._bundle["renders"],
            selected_options,
            source_id=lambda row: row["render_id"],
            prompt=lambda row: row["messages"],
            target=lambda row: row["gold_answer"],
            grader="pdbthink",
            generation_kwargs={"do_sample": True, "temperature": 0.7, "tool_choice": "none"},
            repeats=self.n_repeat,
            seed_repeats=False,
            reference=lambda row: {"answer_schema": row["answer_schema"], "parameters": row["parameters"]},
        )
        # Check the complete selected cohort before the first model request. A prompt
        # that cannot fit must not silently change coordinates or benchmark coverage.
        prepared = []
        budgets: dict[str, RenderBudget] = {}
        for spec in specs:
            render_id = spec.source_id
            if render_id not in budgets:
                tokens = options.prompt_token_count(spec.requests[0].prompt)
                try:
                    cap = safe_generation_cap(
                        context_length=context,
                        prompt_tokens=tokens,
                        requested_max_tokens=context if requested_cap is None else int(requested_cap),
                    )
                except ContextWindowExceededError as error:
                    error.add_note(f"PDBThink render {render_id}")
                    raise
                budgets[render_id] = {"render_id": render_id, "input_tokens": tokens, "output_tokens": cap}
            budget = budgets[render_id]
            request = replace(
                spec.requests[0],
                generation={**spec.requests[0].generation, "max_gen_toks": budget["output_tokens"]},
                output_token_cap=budget["output_tokens"],
            )
            metadata = {
                **spec.doc["report_metadata"],
                "model_id": options.model_name,
                "model_provider": "evalchemy",
                "model_revision": None,
                "reasoning_effort": generation.get("reasoning_effort"),
            }
            prepared.append(replace(spec, requests=(request,), doc={**spec.doc, "report_metadata": metadata}))
        self._budgets = list(budgets.values())
        yield from prepared
