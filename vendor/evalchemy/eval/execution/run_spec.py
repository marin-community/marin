"""Resolve run-level intent once before sources construct item specifications."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field, fields
from typing import Any

from eval.passk import parse_pass_at_k
from eval.limits import parse_key_value_args
from eval.robust_api import parse_generation_overrides
from eval.execution.generation import canonical_generation_kwargs


@dataclass(frozen=True)
class RunSpec:
    limit: int | float | None = None
    max_length: int | None = None
    max_tokens: int | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int | None, ...] = (0, 1234, 1234, 1234)
    num_fewshot: int | None = None
    system_instruction: str | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    num_samples: int = 1
    pass_at_k: tuple[int, ...] = (1, 8, 32, 128)
    predict_only: bool = False
    confirm_run_unsafe_code: bool = False
    judge_model: str | None = None
    model_name: str | None = None

    @classmethod
    def from_args(cls, args: Namespace) -> RunSpec:
        values = {
            item.name: getattr(args, item.name, item.default)
            for item in fields(cls) if item.name not in {"generation", "seeds", "pass_at_k"}
        }
        generation = canonical_generation_kwargs(parse_generation_overrides(getattr(args, "gen_kwargs", None)))
        model_args = parse_key_value_args(getattr(args, "model_args", ""))
        values["model_name"] = model_args.get("pretrained", model_args.get("model", values["model_name"]))
        if values["limit"] is not None and values["limit"] >= 1:
            values["limit"] = int(values["limit"])
        annotator = getattr(args, "annotator_model", None)
        values["judge_model"] = None if annotator == "auto" else annotator
        if values["max_tokens"] is not None:
            generation["max_gen_toks"] = values["max_tokens"]
        return cls(
            **values,
            generation=generation,
            seeds=tuple(getattr(args, "seed", cls.seeds)),
            pass_at_k=tuple(parse_pass_at_k(getattr(args, "pass_at_k", None))),
        )
