# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interpret the protocol choices encoded by historical Evalchemy launch labels."""

from dataclasses import dataclass
from typing import Literal

from evalrecords.records import EvalRunRecord, ResolvedEvaluationSelection


@dataclass(frozen=True)
class HistoricalProtocol:
    benchmark: str
    protocol: str
    num_fewshot: int | None
    prompt_format: Literal["completion", "chat"] | None
    model_prompt_format: bool
    partitions: tuple[str, ...] = ()


# This is the historical launch contract. New benchmark declarations belong to curation.
HISTORICAL_PROTOCOLS = {
    "aime24-smoke": HistoricalProtocol("aime24", "evalchemy", None, "chat", False),
    "aime24": HistoricalProtocol("aime24", "evalchemy", None, "chat", False),
    "arc-challenge": HistoricalProtocol("arc-challenge", "harness", 25, None, False),
    "arc-easy": HistoricalProtocol("arc-easy", "harness", 0, None, False),
    "boolq": HistoricalProtocol("boolq", "harness", 0, None, False),
    "cruxeval": HistoricalProtocol("cruxeval", "evalchemy", None, "chat", False),
    "drop": HistoricalProtocol("drop", "harness", 3, None, True),
    "financebench": HistoricalProtocol("financebench", "evalchemy", None, "chat", False),
    "gpqa-diamond": HistoricalProtocol("gpqa-diamond", "evalchemy", None, "chat", False),
    "graphwalks": HistoricalProtocol("graphwalks", "evalchemy", None, "chat", False),
    "gsm8k-0shot": HistoricalProtocol("gsm8k", "harness", 0, None, True),
    "gsm8k-smoke": HistoricalProtocol("gsm8k", "harness", 5, None, True),
    "gsm8k": HistoricalProtocol("gsm8k", "harness", 5, None, True),
    "hellaswag": HistoricalProtocol("hellaswag", "harness", 10, None, False),
    "humaneval": HistoricalProtocol("humaneval", "evalchemy", 0, "completion", False),
    "humanevalplus": HistoricalProtocol("humanevalplus", "evalchemy", None, "chat", False),
    "ifbench": HistoricalProtocol("ifbench", "evalchemy", None, "chat", False),
    "ifeval": HistoricalProtocol("ifeval", "evalchemy", None, None, True),
    "lambada": HistoricalProtocol("lambada", "harness", None, None, False),
    "math500": HistoricalProtocol("math500", "evalchemy", None, "chat", False),
    "mbppplus": HistoricalProtocol("mbppplus", "evalchemy", None, "chat", False),
    "mmlu-pro": HistoricalProtocol("mmlu-pro", "evalchemy", None, "chat", False),
    "mmlu-smoke": HistoricalProtocol("mmlu", "harness", 0, None, False),
    "mmlu": HistoricalProtocol("mmlu", "harness", 5, None, False),
    "mrcr": HistoricalProtocol("mrcr", "evalchemy", None, "chat", False),
    "nq-open": HistoricalProtocol("nq-open", "harness", 5, None, True),
    "olympiadbench-deterministic": HistoricalProtocol("olympiadbench", "deterministic", None, "chat", False),
    "olympiadbench": HistoricalProtocol("olympiadbench", "evalchemy", None, "chat", False),
    "openbookqa": HistoricalProtocol("openbookqa", "harness", 0, None, False),
    "piqa": HistoricalProtocol("piqa", "harness", 0, None, False),
    "triviaqa": HistoricalProtocol("triviaqa", "harness", 5, None, True),
    "truthfulqa": HistoricalProtocol("truthfulqa", "harness", 0, None, False),
    "winogrande": HistoricalProtocol("winogrande", "harness", 5, None, False),
    "arc-challenge-10shot": HistoricalProtocol("arc-challenge", "harness", 10, None, False),
    "arc-easy-10shot": HistoricalProtocol("arc-easy", "harness", 10, None, False),
    "boolq-10shot": HistoricalProtocol("boolq", "harness", 10, None, False),
    "piqa-10shot": HistoricalProtocol("piqa", "harness", 10, None, False),
    "winogrande-0shot": HistoricalProtocol("winogrande", "harness", 0, None, False),
    "hellaswag-0shot": HistoricalProtocol("hellaswag", "harness", 0, None, False),
    "truthfulqa-6shot": HistoricalProtocol("truthfulqa", "harness", 6, None, False),
    "mmlu-0shot": HistoricalProtocol("mmlu", "harness", 0, None, False),
    "commonsense-qa-10shot": HistoricalProtocol("commonsense-qa", "likelihood", 10, None, False),
    "copa": HistoricalProtocol("copa", "likelihood", 0, None, False),
    "wsc273": HistoricalProtocol("wsc273", "likelihood", 0, None, False),
    "agieval-lsat-ar-3shot": HistoricalProtocol("agieval-lsat-ar", "likelihood", 3, None, False),
    "mmlu-pro-5shot": HistoricalProtocol("mmlu-pro", "likelihood", 5, None, False),
    "bbh-3shot": HistoricalProtocol("bbh", "likelihood", 3, None, False),
    "gpqa-0shot": HistoricalProtocol("gpqa", "likelihood", 0, None, False),
    "gsm8k-cot-8shot": HistoricalProtocol("gsm8k", "cot", 8, "completion", False),
    "bbh-cot-3shot": HistoricalProtocol("bbh", "cot", 3, "completion", False),
    "minerva-math-4shot": HistoricalProtocol("hendrycks-math", "cot", 4, "completion", False),
    "nq-open-0shot": HistoricalProtocol("nq-open", "harness", 0, "completion", False),
    "triviaqa-0shot": HistoricalProtocol("triviaqa", "harness", 0, "completion", False),
    "drop-0shot": HistoricalProtocol("drop", "harness", 0, "completion", False),
    "humaneval-10shot": HistoricalProtocol("humaneval", "evalchemy", 10, "completion", False),
    "nupa": HistoricalProtocol("nupa5k-loose", "evalchemy", None, "chat", False),
    "mrcr-32k": HistoricalProtocol("mrcr", "32k", None, "chat", False),
    "mrcr-65k": HistoricalProtocol("mrcr", "65k", None, "chat", False),
}


def historical_selection(record: EvalRunRecord) -> ResolvedEvaluationSelection | None:
    legacy = HISTORICAL_PROTOCOLS.get(record.evaluation.name)
    if legacy is None or record.evaluation.harbor is not None:
        return None
    shots = {task.num_fewshot for task in record.evaluation.tasks if task.num_fewshot is not None}
    form = legacy.prompt_format
    if legacy.model_prompt_format:
        if record.evaluation.evalchemy is not None:
            form = "chat" if record.evaluation.evalchemy.apply_chat_template else "completion"
        elif record.model.config is not None:
            form = "chat" if record.model.config.apply_chat_template else "completion"
    return ResolvedEvaluationSelection(
        benchmark=legacy.benchmark,
        protocol=legacy.protocol,
        num_fewshot=next(iter(shots)) if len(shots) == 1 else legacy.num_fewshot,
        prompt_format=form,
        partitions=legacy.partitions,
    )


def comparison_column(selection: ResolvedEvaluationSelection) -> str:
    settings = [selection.protocol] if selection.protocol is not None else []
    if selection.num_fewshot is not None:
        settings.append(f"{selection.num_fewshot}-shot")
    if selection.prompt_format is not None:
        settings.append(selection.prompt_format)
    if selection.partitions:
        settings.append("partitions=" + "+".join(selection.partitions))
    return selection.benchmark + (" [" + "; ".join(settings) + "]" if settings else "")


def compatible_selection(historical: ResolvedEvaluationSelection, current: ResolvedEvaluationSelection) -> bool:
    return (
        historical.benchmark == current.benchmark
        and historical.partitions == current.partitions
        and all(
            before is None or after is None or before == after
            for before, after in (
                (historical.protocol, current.protocol),
                (historical.num_fewshot, current.num_fewshot),
                (historical.prompt_format, current.prompt_format),
            )
        )
    )
