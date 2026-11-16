"""Source-specific transforms for supported RLVR datasets."""

from __future__ import annotations

import ast
import copy
import itertools
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import reasoning_gym
import requests
from skyrl_gym.envs.aime.utils import last_boxed_only_string, remove_boxed
from skyrl_gym.envs.mcq.utils import extract_mcq_answer
from skyrl_gym.envs.nemotron_ultra import (
    NEMOTRON_ULTRA_MOPD_AGENTS,
    NEMOTRON_ULTRA_RLVR1_AGENTS,
    NEMOTRON_ULTRA_RLVR2_AGENTS,
)
from skyrl_gym.envs.text_to_sql import scoring as text_to_sql_scoring

from infra.rl_data.contracts import VerifierDataContract

RLVR_MATH_DATASET = "allenai/RLVR-MATH"
RLVR_IFEVAL_DATASET = "allenai/RLVR-IFeval"
DAPO_MATH_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
AIME24_DATASET = "HuggingFaceH4/aime_2024"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEEPSCALER_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
GSM8K_DATASET = "openai/gsm8k"
HENDRYCKS_MATH_DATASET = "EleutherAI/hendrycks_math"
AIME_1983_2024_DATASET = "di-zhang-fdu/AIME_1983_2024"
ASDIV_DATASET = "chaochun/nlu-asdiv-dataset"
SVAMP_DATASET = "ChilleD/SVAMP"
NUMINA_MATH_DATASET = "AI-MO/NuminaMath-CoT"
HARDMATH_DATASET = "pafitis/HARDMath_processed_training"
VERIFIABLE_CODE_DATASET = "open-r1/verifiable-coding-problems-python"
APPS_DATASET = "codeparrot/apps"
GPQA_DATASET = "Idavidrein/gpqa"
OPENSCIENCE_DATASET = "nvidia/OpenScience"
OPENSCIENCE_SUBSETS = ("OS-Q2.5-32B-10", "OS-Q2.5-32B-4", "OS-Q2.5-72B-10", "OS-Q3-235B-4")
KTO_MIX_DATASET = "trl-lib/kto-mix-14k"
HH_RLHF_DATASET = "Anthropic/hh-rlhf"
EURUS2_DATASET = "PRIME-RL/Eurus-2-RL-Data"
NEMOTRON_DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"
REASONING_GYM_DATASET = "open-thought/reasoning-gym"
GRETEL_TEXT_TO_SQL_DATASET = "gretelai/synthetic_text_to_sql"
NEMOTRON_ULTRA_RL_DATASET = "nvidia/Nemotron-RL-Ultra-Training-Blends"
NEMOTRON_ULTRA_REVISION = "79f8eda15ea12e1adf7bb14dcb338a29d391b80e"
NEMOTRON_ULTRA_DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
NEMOTRON_ULTRA_SKYWORK_REVISION = "1cdedc52e0e2db85fdf252f9be682e63a5a38c33"
TEST_ONLY_SOURCE_LABELS = {"aime24": "AIME24", "math500": "MATH-500"}
TEST_ONLY_SOURCE_NAMES = frozenset(TEST_ONLY_SOURCE_LABELS)

_DAPO_LEADING_MARKERS = ("The last line of your response", "Solve the following math problem")
_DAPO_TRAILING_MARKERS = ("Remember to put your answer", "The last line of your response")
_EURUS2_CODE_ABILITY = "code"
_HENDRYCKS_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
_ASDIV_XML_URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{revision}/dataset/ASDiv.xml"
_PLAIN_NUMERIC_ANSWER = re.compile(r"^-?\d+(?:\.\d+)?(?:/\d+)?$|^-?\d+:-?\d+$")


PreparedRow = dict[str, Any]
PrepareRow = Callable[[Mapping[str, Any], int, VerifierDataContract], PreparedRow]


@dataclass(frozen=True)
class Source:
    """A downloadable RLVR source and its transform to the SkyRL row schema."""

    name: str
    dataset_id: str
    env_id: str
    split: str
    streaming: bool
    verification: str
    prepare_row: PrepareRow
    load_rows: Callable[[Source, str, Mapping[str, Any]], Iterable[Mapping[str, Any]]] | None = None
    deduplicate_by_prompt: bool = True


NEMOTRON_ULTRA_SWE_AGENT = "swe_pivot_single_step_tool_use_with_argument_comparison_agent"
# Row metadata key naming the TaskTrove proxy task bound to a Harbor SWE row.
TASKTROVE_PROXY_PATH_KEY = "tasktrove_proxy_path"
_NEMOTRON_PLACEHOLDER_KEY = "_hf_question_placeholder"
_NEMOTRON_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
_NEMOTRON_DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def _message_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("NeMo Gym message content must be a string, null, or a list of content items.")
    text: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") not in {"input_text", "output_text"}:
            raise ValueError(f"Unsupported NeMo Gym message content item: {item!r}.")
        value = item.get("text")
        if not isinstance(value, str):
            raise TypeError("NeMo Gym text content items must contain a string text field.")
        text.append(value)
    return "".join(text)


def _nemotron_ultra_messages(raw_input: Any) -> list[dict[str, Any]]:
    """Translate the released Responses-API history to vLLM chat messages.

    Reasoning items carry no user-visible content in the released blend. They are
    deliberately omitted, while function calls and their outputs retain their IDs
    and exact JSON argument strings.
    """
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if not isinstance(raw_input, list) or not raw_input:
        raise TypeError("responses_create_params.input must be a non-empty string or list.")

    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_calls)})
            pending_calls.clear()

    for item in raw_input:
        if not isinstance(item, Mapping):
            raise TypeError("NeMo Gym input items must be mappings.")
        item_type = item.get("type", "message")
        if item_type == "reasoning":
            continue
        if item_type == "function_call":
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            if not all(isinstance(value, str) for value in (name, call_id, arguments)):
                raise TypeError("NeMo Gym function calls require string name, call_id, and arguments fields.")
            pending_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue

        flush_calls()
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise TypeError("NeMo Gym function-call outputs require a string call_id.")
            output = item.get("output", "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                }
            )
            continue
        if item_type != "message":
            raise ValueError(f"Unsupported NeMo Gym input item type {item_type!r}.")
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Unsupported NeMo Gym message role {role!r}.")
        messages.append({"role": role, "content": _message_text(item.get("content"))})
    flush_calls()
    if not messages:
        raise ValueError("NeMo Gym input did not contain any chat-visible messages.")
    return messages


def _freeze_instruction_references(example: Mapping[str, Any], index: int, seed: int) -> Mapping[str, Any]:
    """Resolve hidden instruction references before serializing a task for generation."""
    if type(seed) is not int:
        raise ValueError("Instruction reference seed must be an integer")
    from skyrl_gym.envs.nemotron_ultra.instruction_references import freeze_instruction_references

    record = freeze_instruction_references(dict(example), f"{seed}:{example.get('uuid', index)}:{index}")
    record["instruction_reference_seed"] = seed
    return record


def _prepare_nemotron_ultra(
    example: Mapping[str, Any],
    index: int,
    contract: VerifierDataContract,
    *,
    agents: frozenset[str],
    blend: str,
    instruction_reference_seed: int | None = None,
    math_reference_kind: str | None = None,
) -> PreparedRow:
    del contract
    request = example.get("responses_create_params")
    agent_ref = example.get("agent_ref")
    if not isinstance(request, Mapping):
        raise TypeError("Nemotron Ultra row responses_create_params must be a mapping.")
    if not isinstance(agent_ref, Mapping):
        raise TypeError("Nemotron Ultra row agent_ref must be a mapping.")
    agent = agent_ref.get("name")
    if not isinstance(agent, str) or agent not in agents:
        raise ValueError(f"Nemotron Ultra row has unsupported agent_ref.name {agent!r}.")

    route = "terminal_bench" if agent == NEMOTRON_ULTRA_SWE_AGENT else "task_session"
    metadata = example.get("metadata")
    instance_id = metadata.get("instance_id") if isinstance(metadata, Mapping) else None
    if route == "terminal_bench" and not isinstance(instance_id, str):
        raise ValueError("Nemotron Ultra SWE pivot row is missing metadata.instance_id.")
    # The stored schema name is historical: snapshot-backed SWE rows use the
    # exact TaskTrove archive path as their Harbor task identifier.
    terminal_bench_task_id = (
        metadata.get(TASKTROVE_PROXY_PATH_KEY, instance_id) if isinstance(metadata, Mapping) else None
    )
    if _NEMOTRON_PLACEHOLDER_KEY in example:
        raise ValueError("Nemotron Ultra math placeholder was not restored before row preparation.")

    if instruction_reference_seed is not None and agent == "instruction_following_simple_agent":
        example = _freeze_instruction_references(example, index, instruction_reference_seed)

    if agent in {"math_with_judge_simple_agent", "ns_tools_simple_agent"}:
        from skyrl_gym.envs.nemotron_ultra.math_references import prepare_math_reference, reference_kind

        if math_reference_kind is not None:
            example = prepare_math_reference(example, math_reference_kind)
        else:
            reference_kind(example)

    return {
        "data_source": NEMOTRON_ULTRA_RL_DATASET,
        "prompt": _nemotron_ultra_messages(request.get("input")),
        "env_class": "nemotron_ultra",
        "reward_model": {"ground_truth": agent},
        "extra_info": {
            "split": "train",
            "index": index,
            "nemotron_ultra": {
                "uuid": str(example.get("uuid", index)),
                "blend": blend,
                "agent": agent,
                "route": route,
                "terminal_bench_instance_id": terminal_bench_task_id,
                "request_json": json.dumps(
                    {key: value for key, value in request.items() if key != "input"},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "record_json": json.dumps(
                    {
                        key: value
                        for key, value in example.items()
                        if key not in {"responses_create_params", "agent_ref"}
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        },
    }


def _last_user_content(messages: list[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message["content"])
    raise ValueError("Expected a user message.")


def _strip_rlvr_math_fewshot(text: str) -> str:
    marker = "Question:"
    index = text.rfind(marker)
    return (text[index + len(marker) :] if index >= 0 else text).strip()


def _strip_dapo_boilerplate(text: str) -> str:
    body = text
    if any(marker in body[:200] for marker in _DAPO_LEADING_MARKERS):
        divider = body.find("\n\n")
        if divider >= 0:
            body = body[divider + 2 :]

    trailing_positions = [body.find(marker) for marker in _DAPO_TRAILING_MARKERS]
    trailing_positions = [position for position in trailing_positions if position >= 0]
    if trailing_positions:
        body = body[: min(trailing_positions)]
    return body.strip()


def _prepared_verifier_row(problem: str, normalized: str, source: Source, index: int) -> PreparedRow:
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": problem}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _schema_row(
    problem: str, ground_truth: Any, source: Source, index: int, contract: VerifierDataContract
) -> PreparedRow:
    if not contract.prompt_instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(ground_truth)
    return _prepared_verifier_row(problem + contract.prompt_instruction, normalized, source, index)


def _math_row(
    problem: str, ground_truth: Any, source: Source, index: int, contract: VerifierDataContract
) -> PreparedRow:
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    if ground_truth is None:
        raise ValueError(f"{source.name} row is missing ground_truth.")

    normalized = contract.normalize_ground_truth(ground_truth)
    verified = contract.validate_example(normalized, f"Answer: \\boxed{{{normalized}}}", "")
    return _prepared_verifier_row(problem + instruction, verified, source, index)


def _prepare_rlvr_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = rlvr_math_source()
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise TypeError("RLVR-MATH row messages must be a list.")
    return _math_row(
        _strip_rlvr_math_fewshot(_last_user_content(messages)), example.get("ground_truth"), source, index, contract
    )


def _prepare_dapo_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = dapo_math_source()
    messages = example.get("prompt")
    reward_model = example.get("reward_model")
    if not isinstance(messages, list) or not isinstance(reward_model, Mapping):
        raise TypeError("DAPO-Math row prompt messages and reward_model must be present.")
    return _math_row(
        _strip_dapo_boilerplate(_last_user_content(messages)),
        reward_model.get("ground_truth"),
        source,
        index,
        contract,
    )


def _prepare_problem_answer_math(
    example: Mapping[str, Any],
    index: int,
    contract: VerifierDataContract,
    source: Source,
    label: str,
) -> PreparedRow:
    problem = example.get("problem")
    if not isinstance(problem, str):
        raise TypeError(f"{label} row problem must be a string.")
    return _math_row(problem, example.get("answer"), source, index, contract)


def _prepare_math500(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    return _prepare_problem_answer_math(example, index, contract, math500_source(), "MATH-500")


def _prepare_aime24(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    return _prepare_problem_answer_math(example, index, contract, aime24_source(), "AIME24")


def _prepare_rlvr_ifeval(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise TypeError("RLVR-IFeval row messages must be a list.")
    normalized = contract.normalize_ground_truth(example.get("ground_truth"))
    return {
        "data_source": RLVR_IFEVAL_DATASET,
        "prompt": [{"role": "user", "content": _last_user_content(messages)}],
        "env_class": "ifeval",
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index, "constraint_type": example.get("constraint_type")},
    }


def rlvr_math_source() -> Source:
    return Source("rlvr_math", RLVR_MATH_DATASET, "aime", "train", False, "two_sided", _prepare_rlvr_math)


def dapo_math_source() -> Source:
    return Source("dapo_math", DAPO_MATH_DATASET, "aime", "train", True, "two_sided", _prepare_dapo_math)


def math500_source() -> Source:
    return Source("math500", MATH500_DATASET, "aime", "test", False, "two_sided", _prepare_math500)


def aime24_source() -> Source:
    return Source("aime24", AIME24_DATASET, "aime", "train", False, "two_sided", _prepare_aime24)


def rlvr_ifeval_source() -> Source:
    return Source("rlvr_ifeval", RLVR_IFEVAL_DATASET, "ifeval", "train", False, "schema_only", _prepare_rlvr_ifeval)


# ---------------------------------------------------------------------------
# Math: DeepScaleR and GSM8K
# ---------------------------------------------------------------------------


def _prepare_deepscaler(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    return _prepare_problem_answer_math(example, index, contract, deepscaler_source(), "DeepScaleR")


def _gsm8k_extract_answer(answer_text: str) -> str:
    marker = "####"
    idx = answer_text.rfind(marker)
    if idx < 0:
        raise ValueError("GSM8K answer field missing '####' delimiter.")
    return answer_text[idx + len(marker) :].strip().replace(",", "").replace("$", "")


def _prepare_gsm8k(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = gsm8k_source()
    question = example.get("question")
    answer_text = example.get("answer")
    if not isinstance(question, str):
        raise TypeError("GSM8K row question must be a string.")
    if not isinstance(answer_text, str):
        raise TypeError("GSM8K row answer must be a string.")
    ground_truth = _gsm8k_extract_answer(answer_text)
    return _schema_row(question, ground_truth, source, index, contract)


# ---------------------------------------------------------------------------
# Math: curriculum-ladder sources
# ---------------------------------------------------------------------------


def _boxed_answer(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError("solution is missing a boxed answer.")
    answer = remove_boxed(boxed).strip()
    if not answer:
        raise ValueError("solution has an empty boxed answer.")
    return answer


def _plain_numeric_answer(answer: Any) -> str:
    normalized = str(answer).split("(", 1)[0].strip().replace(",", "")
    if not _PLAIN_NUMERIC_ANSWER.fullmatch(normalized):
        raise ValueError("answer is not a plain number, fraction, or ratio.")
    if normalized.endswith(".0"):
        return normalized[: -len(".0")]
    return normalized


def _prepare_hendrycks_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = hendrycks_math_source()
    problem = example.get("problem")
    solution = example.get("solution")
    level = example.get("level")
    subject = example.get("subject")
    if not all(isinstance(value, str) and value for value in (problem, solution, level, subject)):
        raise TypeError("Hendrycks MATH rows require problem, solution, level, and subject strings.")
    row = _math_row(problem, _boxed_answer(solution), source, index, contract)
    row["extra_info"].update({"level": level, "subject": subject})
    return row


def _prepare_aime_1983_2024(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = aime_1983_2024_source()
    question = example.get("Question")
    answer = example.get("Answer")
    if not isinstance(question, str) or not question:
        raise TypeError("AIME 1983-2024 rows require a Question string.")
    if answer is None or not str(answer).strip() or str(answer).strip().lower() == "none":
        raise ValueError("AIME 1983-2024 rows require an answer.")
    return _math_row(question, str(answer).strip(), source, index, contract)


def _prepare_numeric_word_problem(
    example: Mapping[str, Any],
    index: int,
    contract: VerifierDataContract,
    source: Source,
    label: str,
) -> PreparedRow:
    body = example.get("Body")
    question = example.get("Question")
    if not all(isinstance(value, str) and value for value in (body, question)):
        raise TypeError(f"{label} rows require Body and Question strings.")
    return _math_row(
        f"{body.strip()} {question.strip()}",
        _plain_numeric_answer(example.get("Answer")),
        source,
        index,
        contract,
    )


def _prepare_asdiv(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    grade = example.get("Grade")
    if not isinstance(grade, str) or not grade:
        raise TypeError("ASDiv rows require a Grade string.")
    row = _prepare_numeric_word_problem(example, index, contract, asdiv_source(), "ASDiv")
    row["extra_info"]["grade"] = grade
    return row


def _prepare_svamp(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    return _prepare_numeric_word_problem(example, index, contract, svamp_source(), "SVAMP")


def _prepare_numina_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = numina_math_source()
    problem = example.get("problem")
    solution = example.get("solution")
    source_tag = example.get("source")
    if not all(isinstance(value, str) and value for value in (problem, solution, source_tag)):
        raise TypeError("NuminaMath rows require problem, solution, and source strings.")
    row = _math_row(problem, _boxed_answer(solution), source, index, contract)
    row["extra_info"]["source"] = source_tag
    return row


def _prepare_hardmath(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = hardmath_source()
    question = example.get("question")
    ground_truth = example.get("ground_truths")
    if not isinstance(question, str) or not question:
        raise TypeError("HARDMath rows require a question string.")
    if not isinstance(ground_truth, str):
        raise TypeError("HARDMath rows require a ground_truths string.")
    answer = _boxed_answer(ground_truth)
    if "[" in answer:
        raise ValueError("HARDMath list-valued answers are not supported.")
    if "\\approx" in answer:
        answer = answer.rsplit("\\approx", 1)[1].strip()
    elif "=" in answer:
        answer = answer.rsplit("=", 1)[1].strip()
    if not answer:
        raise ValueError("HARDMath row has an empty answer.")
    return _math_row(question, answer, source, index, contract)


# ---------------------------------------------------------------------------
# Code: verifiable-coding-problems-python and APPS
# ---------------------------------------------------------------------------


def _prepare_verifiable_code(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = verifiable_code_source()
    problem = example.get("problem_statement")
    verification_info = example.get("verification_info")
    if not isinstance(problem, str):
        raise TypeError("verifiable-coding-problems row problem_statement must be a string.")
    if verification_info is None:
        raise ValueError("verifiable-coding-problems row is missing verification_info.")
    if isinstance(verification_info, str):
        try:
            verification_info = ast.literal_eval(verification_info)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("verification_info string could not be parsed.") from exc
    return _schema_row(problem, verification_info, source, index, contract)


def _prepare_apps(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = apps_source()
    problem = example.get("question")
    input_output = example.get("input_output")
    if not isinstance(problem, str):
        raise TypeError("APPS row question must be a string.")
    if input_output is None:
        raise ValueError("APPS row is missing input_output.")
    if isinstance(input_output, str):
        try:
            input_output = json.loads(input_output)
        except json.JSONDecodeError as exc:
            raise ValueError("APPS input_output string is not valid JSON.") from exc
    return _schema_row(problem, input_output, source, index, contract)


def _prepare_eurus2_code(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = eurus2_code_source()
    if example.get("ability") != _EURUS2_CODE_ABILITY:
        raise ValueError("Eurus-2 row is not in the code subset.")
    prompt = example.get("prompt")
    reward_model = example.get("reward_model")
    if not isinstance(prompt, list) or not isinstance(reward_model, Mapping):
        raise TypeError("Eurus-2 code row requires prompt and reward_model fields.")
    return _schema_row(_last_user_content(prompt), reward_model.get("ground_truth"), source, index, contract)


_NEMOTRON_CONSTRAINTS: dict[str, tuple[str, dict[str, str]]] = {
    "keywords:existence": ("verify_keywords", {"keywords": "keyword_list"}),
    "keywords:frequency": (
        "verify_keyword_frequency_relation",
        {"keywords": "keyword_list", "frequency": "N", "relation": "quantifier"},
    ),
    "keywords:forbidden_words": ("validate_forbidden_words", {"forbidden_words": "forbidden_words"}),
    "keywords:letter_frequency": ("verify_letter_frequency", {"letter": "letter", "let_frequency": "N"}),
    "language:response_language": ("validate_response_language", {"language": "language"}),
    "length_constraints:number_paragraphs": ("verify_paragraph_count", {"num_paragraphs": "N"}),
    "length_constraints:number_words": (
        "validate_word_constraint",
        {"num_words": "N", "relation": "quantifier"},
    ),
    "length_constraints:number_sentences": (
        "verify_sentence_constraint",
        {"num_sentences": "N", "relation": "quantifier"},
    ),
    "length_constraints:nth_paragraph_first_word": (
        "validate_paragraphs",
        {"num_paragraphs": "N", "first_word": "first_word", "nth_paragraph": "i"},
    ),
    "detectable_content:postscript": ("verify_postscript", {"postscript_marker": "postscript_marker"}),
    "detectable_content:number_placeholders": ("validate_placeholders", {"num_placeholders": "N"}),
    "detectable_format:number_bullet_lists": ("verify_bullet_points", {"num_bullets": "N"}),
    "detectable_format:title": ("validate_title", {}),
    "detectable_format:constrained_response": ("validate_choice", {"options": "options"}),
    "detectable_format:number_highlighted_sections": (
        "validate_highlighted_sections",
        {"num_highlights": "N"},
    ),
    "detectable_format:multiple_sections": (
        "validate_sections",
        {"num_sections": "N", "section_spliter": "section_splitter"},
    ),
    "detectable_format:json_format": ("validate_json_format", {}),
    "combination:repeat_prompt": ("validate_repeat_prompt", {"original_prompt": "original_prompt"}),
    "combination:two_responses": ("validate_two_responses", {}),
    "change_case:capital_word_frequency": (
        "validate_frequency_capital_words",
        {"frequency": "N", "relation": "quantifier"},
    ),
    "change_case:english_capital": ("validate_uppercase", {}),
    "change_case:english_lowercase": ("validate_lowercase", {}),
    "startend:end_checker": ("validate_end", {"end_phrase": "end_phrase"}),
    "punctuation:no_comma": ("validate_no_commas", {}),
    "detectable_format:quotation": ("validate_quotation", {}),
}


def _nemotron_constraint(instruction_id: str, arguments: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    try:
        function_name, argument_names = _NEMOTRON_CONSTRAINTS[instruction_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported Nemotron instruction id: {instruction_id!r}.") from exc
    constraint: dict[str, Any] = {"func_name": function_name}
    for source_name, target_name in argument_names.items():
        value = prompt if source_name == "original_prompt" else arguments.get(source_name)
        if instruction_id == "keywords:frequency" and source_name == "keywords" and value is None:
            keyword = arguments.get("keyword")
            value = [keyword] if isinstance(keyword, str) else re.findall(r'"([^"\n]+)"', prompt.split("\n\n", 1)[0])
        if value is None:
            raise ValueError(f"Nemotron instruction {instruction_id!r} requires {source_name!r}.")
        constraint[target_name] = value
    return constraint


def _prepare_nemotron_if(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = nemotron_if_source()
    messages = example.get("input")
    arguments = example.get("args")
    if not isinstance(messages, list) or not isinstance(arguments, Mapping):
        raise TypeError("Nemotron IF row requires input messages and args.")
    prompt = _last_user_content(messages)
    instruction_ids = arguments.get("instruction_id_list")
    instruction_kwargs = arguments.get("instruction_kwargs")
    if not isinstance(instruction_ids, list) or not isinstance(instruction_kwargs, list):
        raise TypeError("Nemotron IF args require instruction_id_list and instruction_kwargs lists.")
    if not instruction_ids or len(instruction_ids) != len(instruction_kwargs):
        raise ValueError("Nemotron IF instruction ids and kwargs must be non-empty and aligned.")
    constraints = [
        _nemotron_constraint(str(instruction_id), kwargs, prompt)
        for instruction_id, kwargs in zip(instruction_ids, instruction_kwargs)
        if isinstance(kwargs, Mapping)
    ]
    if len(constraints) != len(instruction_ids):
        raise TypeError("Nemotron IF instruction kwargs must be mappings.")
    normalized = contract.normalize_ground_truth(constraints)
    return _prepared_verifier_row(prompt, normalized, source, index)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Reasoning Gym metadata value {type(value).__name__} is not JSON serializable.")


def _prepare_reasoning_gym(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = reasoning_gym_source()
    question = example.get("question")
    metadata = example.get("metadata")
    answer = example.get("answer")
    if not isinstance(question, str) or not isinstance(metadata, Mapping) or not isinstance(answer, str):
        raise TypeError("Reasoning Gym rows require string question/answer and mapping metadata.")
    task = metadata.get("source_dataset")
    if not isinstance(task, str) or not task:
        raise ValueError("Reasoning Gym metadata requires source_dataset.")
    entry = json.loads(json.dumps(dict(example), default=_json_default))
    ground_truth = {"task": task, "entry": entry}
    normalized = contract.validate_example(ground_truth, answer, "definitely wrong")
    return _prepared_verifier_row(question, normalized, source, index)


# ---------------------------------------------------------------------------
# Science MCQ: GPQA and OpenScience
# ---------------------------------------------------------------------------

import random as _random


def _build_mcq_prompt(question: str, options: list[str], correct_idx: int, seed: int) -> tuple[str, str]:
    """Shuffle options deterministically and return (prompt, correct_letter)."""
    rng = _random.Random(seed)
    order = list(range(len(options)))
    rng.shuffle(order)
    labels = "ABCDEFGHIJKLMNOP"[: len(options)]
    lines = [f"{labels[pos]}) {options[order[pos]]}" for pos in range(len(options))]
    prompt = question.strip() + "\n" + "\n".join(lines)
    correct_letter = labels[order.index(correct_idx)]
    return prompt, correct_letter


def _prepare_gpqa(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = gpqa_source()
    question = example.get("Question")
    correct = example.get("Correct Answer")
    if not isinstance(question, str) or not isinstance(correct, str):
        raise TypeError("GPQA row must have string Question and Correct Answer.")
    options = [correct]
    for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"):
        val = example.get(key)
        if isinstance(val, str):
            options.append(val)
    if len(options) < 2:
        raise ValueError("GPQA row needs at least one incorrect answer.")
    prompt_text, correct_letter = _build_mcq_prompt(question, options, 0, index)
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(correct_letter)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _mcq_options(prompt: str) -> dict[str, str]:
    """Map each ``X: text`` option label to its text, or raise on conflicts.

    OpenScience prompts list one option per line. A label repeated with
    different text makes that option ambiguous, so the row is rejected.
    """
    options: dict[str, str] = {}
    for match in re.finditer(r"^([A-Z]): ?(.+)$", prompt, re.MULTILINE):
        label, text = match.group(1), match.group(2).strip()
        previous = options.setdefault(label, text)
        if previous != text:
            raise ValueError(f"OpenScience input has conflicting options for label {label}.")
    if not options:
        raise ValueError("OpenScience input has no 'X: option' lines.")
    return options


def _prepare_openscience(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = openscience_source()
    prompt_text = example.get("input")
    output = example.get("output")
    if not isinstance(prompt_text, str):
        raise TypeError("OpenScience row input must be a string.")
    if not isinstance(output, str):
        raise TypeError("OpenScience row output must be a string.")
    match = extract_mcq_answer(output)
    if match is None:
        raise ValueError("OpenScience output missing \\boxed{X} answer letter.")
    ground_truth = match
    if ground_truth not in _mcq_options(prompt_text):
        raise ValueError(f"OpenScience answer letter {ground_truth} is not among the offered options.")
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    wrong_letter = chr((ord(ground_truth) - ord("A") + 1) % 26 + ord("A"))
    normalized = contract.validate_example(
        ground_truth, f"\\boxed{{{ground_truth}}}", f"\\boxed{{{wrong_letter}}}"
    )
    row = {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }
    subset = example.get("subset")
    if isinstance(subset, str) and subset:
        row["extra_info"]["subset"] = subset
    return row


# ---------------------------------------------------------------------------
# Preference: KTO-mix-14k and HH-RLHF
# ---------------------------------------------------------------------------


def _last_assistant_content(messages: list[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message["content"])
    raise ValueError("Expected an assistant message.")


def _prepare_kto_mix(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = kto_mix_source()
    prompt = example.get("prompt")
    completion = example.get("completion")
    label = example.get("label")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise TypeError("KTO row prompt and completion must be lists.")
    if str(label).lower() != "true":
        raise ValueError("KTO row with label=False has no preferred completion; skipping.")
    chosen_text = _last_assistant_content(completion)
    prompt_text = _last_user_content(prompt)
    normalized = contract.normalize_ground_truth(chosen_text)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _parse_hhrlhf(text: str) -> tuple[str, str]:
    """Extract (prompt, response) from an HH-RLHF dialogue string."""
    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"
    h_idx = text.find(human_marker)
    a_idx = text.find(assistant_marker)
    if h_idx < 0 or a_idx < 0:
        raise ValueError("HH-RLHF text missing Human/Assistant markers.")
    prompt = text[h_idx + len(human_marker) : a_idx].strip()
    response = text[a_idx + len(assistant_marker) :].strip()
    return prompt, response


def _prepare_hh_rlhf(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = hh_rlhf_source()
    chosen_raw = example.get("chosen")
    rejected_raw = example.get("rejected")
    if not isinstance(chosen_raw, str) or not isinstance(rejected_raw, str):
        raise TypeError("HH-RLHF chosen and rejected must be strings.")
    prompt_text, chosen_response = _parse_hhrlhf(chosen_raw)
    _, rejected_response = _parse_hhrlhf(rejected_raw)
    if chosen_response == rejected_response:
        raise ValueError("HH-RLHF chosen and rejected responses are identical.")
    normalized = contract.normalize_ground_truth(chosen_response)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index, "rejected": rejected_response},
    }


# ---------------------------------------------------------------------------
# Gretel synthetic text-to-SQL (result-set-equivalence verifier)
# ---------------------------------------------------------------------------


def _ensure_semicolon(stmt: str) -> str:
    stmt = stmt.strip()
    return stmt if stmt.endswith(";") else stmt + ";"


def _gretel_reference_select(reference_raw: str) -> str:
    """The single deterministic SELECT a gretel row's ``sql`` field must be, or raise to skip the row."""
    stmts = [s for s in text_to_sql_scoring.split_statements(reference_raw) if s.strip()]
    if len(stmts) != 1 or text_to_sql_scoring.classify_statement(stmts[0]) != "select":
        raise ValueError("gretel text-to-SQL reference is not a single SELECT.")
    if text_to_sql_scoring.is_nondeterministic(stmts[0]):
        raise ValueError("gretel text-to-SQL reference depends on the clock or RNG.")
    return stmts[0]


def _split_gretel_context(context: str) -> tuple[list[str], list[str], list[str]]:
    """Partition ``sql_context`` into (CREATE TABLE, INSERT, table-name) lists, or raise to skip the row."""
    create_stmts: list[str] = []
    insert_stmts: list[str] = []
    table_names: list[str] = []
    for stmt in text_to_sql_scoring.split_statements(context):
        kind = text_to_sql_scoring.classify_statement(stmt)
        if kind == "create_table":
            if text_to_sql_scoring.create_table_is_schema_qualified(stmt):
                raise ValueError("gretel text-to-SQL context has a schema-qualified CREATE TABLE.")
            create_stmts.append(stmt)
            name = text_to_sql_scoring.create_table_name(stmt)
            if name:
                table_names.append(name)
        elif kind == "insert":
            insert_stmts.append(stmt)
        elif kind in ("other_ddl", "dml", "unknown"):
            raise ValueError(f"gretel text-to-SQL context has an unsupported statement: {kind}.")
    if not create_stmts:
        raise ValueError("gretel text-to-SQL context has no CREATE TABLE statements.")
    if not insert_stmts:
        raise ValueError("gretel text-to-SQL context has no INSERT statements.")
    return create_stmts, insert_stmts, table_names


def _prepare_gretel_text_to_sql(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    """Static transform only. Whether the reference query actually executes is checked once, by the
    contract's two-sided ``validate_example`` preflight — not in bulk here."""
    source = gretel_text_to_sql_source()
    question = example.get("sql_prompt")
    context = example.get("sql_context")
    reference_raw = example.get("sql")
    if not isinstance(question, str) or not question.strip():
        raise TypeError("gretel text-to-SQL row sql_prompt must be a non-empty string.")
    if not isinstance(context, str) or not context.strip():
        raise TypeError("gretel text-to-SQL row sql_context must be a non-empty string.")
    if not isinstance(reference_raw, str) or not reference_raw.strip():
        raise TypeError("gretel text-to-SQL row sql must be a non-empty string.")

    reference_sql = _gretel_reference_select(reference_raw)
    create_stmts, insert_stmts, table_names = _split_gretel_context(context)

    ground_truth = {
        "schema_sql": "\n".join(_ensure_semicolon(s) for s in create_stmts),
        "insert_sql": "\n".join(_ensure_semicolon(s) for s in insert_stmts),
        "reference_sql": reference_sql,
        "order_significant": text_to_sql_scoring.has_top_level_order_by(reference_sql),
        "table_names": sorted(table_names),
    }
    problem = f"{question.strip()}\n\nDatabase schema (SQLite):\n{ground_truth['schema_sql']}"
    verified = contract.validate_example(ground_truth, f"<solution>{reference_sql}</solution>", "SELECT 1")
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": problem + contract.prompt_instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": verified},
        "extra_info": {
            "split": "train",
            "index": index,
            "id": example.get("id"),
            "domain": example.get("domain"),
            "sql_complexity": example.get("sql_complexity"),
            "sql_task_type": example.get("sql_task_type"),
            "order_significant": ground_truth["order_significant"],
        },
    }


# ---------------------------------------------------------------------------
# Source factories
# ---------------------------------------------------------------------------


def deepscaler_source() -> Source:
    return Source("deepscaler", DEEPSCALER_DATASET, "aime", "train", False, "two_sided", _prepare_deepscaler)


def gsm8k_source() -> Source:
    return Source("gsm8k", GSM8K_DATASET, "gsm8k", "train", False, "two_sided", _prepare_gsm8k)


def hendrycks_math_source() -> Source:
    return Source(
        "hendrycks_math",
        HENDRYCKS_MATH_DATASET,
        "aime",
        "train",
        False,
        "two_sided",
        _prepare_hendrycks_math,
        _load_hendrycks_math_rows,
    )


def aime_1983_2024_source() -> Source:
    return Source(
        "aime_1983_2024",
        AIME_1983_2024_DATASET,
        "aime",
        "train",
        False,
        "two_sided",
        _prepare_aime_1983_2024,
    )


def asdiv_source() -> Source:
    return Source("asdiv", ASDIV_DATASET, "aime", "train", False, "two_sided", _prepare_asdiv, _load_asdiv_rows)


def svamp_source() -> Source:
    return Source("svamp", SVAMP_DATASET, "aime", "train", False, "two_sided", _prepare_svamp)


def numina_math_source() -> Source:
    return Source("numina_math", NUMINA_MATH_DATASET, "aime", "train", False, "two_sided", _prepare_numina_math)


def hardmath_source() -> Source:
    return Source("hardmath", HARDMATH_DATASET, "aime", "train", False, "two_sided", _prepare_hardmath)


def verifiable_code_source() -> Source:
    return Source(
        "verifiable_code", VERIFIABLE_CODE_DATASET, "lcb", "train", False, "schema_only", _prepare_verifiable_code
    )


def apps_source() -> Source:
    return Source("apps", APPS_DATASET, "lcb", "train", False, "schema_only", _prepare_apps)


def eurus2_code_source() -> Source:
    return Source(
        "eurus2_code",
        EURUS2_DATASET,
        "lcb",
        "train",
        True,
        "schema_only",
        _prepare_eurus2_code,
        _load_eurus2_rows,
    )


def nemotron_if_source() -> Source:
    return Source(
        "nemotron_if",
        NEMOTRON_DATASET,
        "ifeval",
        "instruction_following",
        True,
        "schema_only",
        _prepare_nemotron_if,
        _load_nemotron_rows,
    )


def reasoning_gym_source() -> Source:
    return Source(
        "reasoning_gym",
        REASONING_GYM_DATASET,
        "reasoning_gym",
        "generated",
        False,
        "two_sided",
        _prepare_reasoning_gym,
        _load_reasoning_gym_rows,
    )


def gpqa_source() -> Source:
    return Source("gpqa", GPQA_DATASET, "mcq", "gpqa_diamond", False, "two_sided", _prepare_gpqa)


def openscience_source() -> Source:
    return Source(
        "openscience",
        OPENSCIENCE_DATASET,
        "mcq",
        "train",
        True,
        "two_sided",
        _prepare_openscience,
        _load_openscience_rows,
    )


def kto_mix_source() -> Source:
    return Source("kto_mix", KTO_MIX_DATASET, "preference", "train", False, "schema_only", _prepare_kto_mix)


def hh_rlhf_source() -> Source:
    return Source("hh_rlhf", HH_RLHF_DATASET, "preference", "train", False, "two_sided", _prepare_hh_rlhf)


def gretel_text_to_sql_source() -> Source:
    return Source(
        "gretel_text_to_sql",
        GRETEL_TEXT_TO_SQL_DATASET,
        "text_to_sql",
        "train",
        False,
        "two_sided",
        _prepare_gretel_text_to_sql,
    )


def _nemotron_ultra_source(*, name: str, agents: frozenset[str], blend: str, instruction_reference_seed: int | None = None, math_reference_kind: str | None = None) -> Source:
    return Source(
        name,
        NEMOTRON_ULTRA_RL_DATASET,
        "nemotron_ultra",
        "train",
        True,
        "row_selected",
        lambda example, index, contract: _prepare_nemotron_ultra(example, index, contract, agents=agents, blend=blend, instruction_reference_seed=instruction_reference_seed, math_reference_kind=math_reference_kind),
        _load_nemotron_ultra_rows,
        deduplicate_by_prompt=False,
    )


def nemotron_ultra_rlvr1_source(*, instruction_reference_seed: int | None = None, math_reference_kind: str | None = None) -> Source:
    return _nemotron_ultra_source(name="nemotron_ultra_rlvr1", agents=NEMOTRON_ULTRA_RLVR1_AGENTS, blend="rlvr1", instruction_reference_seed=instruction_reference_seed, math_reference_kind=math_reference_kind)


def nemotron_ultra_rlvr2_source(*, instruction_reference_seed: int | None = None, math_reference_kind: str | None = None) -> Source:
    return _nemotron_ultra_source(name="nemotron_ultra_rlvr2", agents=NEMOTRON_ULTRA_RLVR2_AGENTS, blend="rlvr2", instruction_reference_seed=instruction_reference_seed, math_reference_kind=math_reference_kind)


def nemotron_ultra_mopd_source(*, instruction_reference_seed: int | None = None, math_reference_kind: str | None = None) -> Source:
    return _nemotron_ultra_source(name="nemotron_ultra_mopd", agents=NEMOTRON_ULTRA_MOPD_AGENTS, blend="mopd", instruction_reference_seed=instruction_reference_seed, math_reference_kind=math_reference_kind)


def generate_reasoning_gym_rows(*, tasks: tuple[str, ...], rows_per_task: int, seed: int, start_index: int = 0):
    """Generate a deterministic, index-disjoint slice for each Reasoning Gym task."""
    if not tasks:
        raise ValueError("Reasoning Gym requires at least one task.")
    if rows_per_task <= 0:
        raise ValueError("Reasoning Gym rows_per_task must be positive.")
    if start_index < 0:
        raise ValueError("Reasoning Gym start_index cannot be negative.")
    for task_index, task in enumerate(tasks):
        dataset = reasoning_gym.create_dataset(
            task,
            size=start_index + rows_per_task,
            seed=seed + task_index,
        )
        for index in range(start_index, start_index + rows_per_task):
            yield dataset[index]


def _validate_source_parameters(source_name: str, parameters: Mapping[str, Any], supported: set[str]) -> None:
    unknown = set(parameters) - supported
    if unknown:
        raise ValueError(f"{source_name} parameters contain unsupported fields: {sorted(unknown)}.")


def _load_reasoning_gym_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    installed_version = version("reasoning-gym")
    if revision != installed_version:
        raise ValueError(
            f"Reasoning Gym revision must match the installed package version {installed_version!r}, got {revision!r}."
        )
    _validate_source_parameters("Reasoning Gym", parameters, {"tasks", "rows_per_task", "seed", "start_index"})
    tasks = parameters.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) and task for task in tasks):
        raise TypeError("Reasoning Gym parameters.tasks must be a non-empty list of task names.")
    rows_per_task = parameters.get("rows_per_task", 100)
    seed = parameters.get("seed", 42)
    start_index = parameters.get("start_index", 0)
    if type(rows_per_task) is not int or type(seed) is not int or type(start_index) is not int:
        raise TypeError("Reasoning Gym rows_per_task, seed, and start_index must be integers.")
    return generate_reasoning_gym_rows(
        tasks=tuple(tasks),
        rows_per_task=rows_per_task,
        seed=seed,
        start_index=start_index,
    )


def _load_hendrycks_math_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    _validate_source_parameters(source.name, parameters, {"subjects", "skip"})
    subjects = parameters.get("subjects", list(_HENDRYCKS_MATH_SUBJECTS))
    if not isinstance(subjects, list) or not subjects or not all(isinstance(subject, str) for subject in subjects):
        raise TypeError("Hendrycks MATH parameters.subjects must be a non-empty list of strings.")
    unknown = set(subjects) - set(_HENDRYCKS_MATH_SUBJECTS)
    if unknown:
        raise ValueError(f"Unknown Hendrycks MATH subjects: {sorted(unknown)}.")

    def rows():
        for subject in subjects:
            for example in _load_hugging_face_dataset(source, revision, subject):
                yield {**example, "subject": subject}

    return _skip_source_rows(source, rows(), {"skip": parameters.get("skip", 0)})


def _load_asdiv_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    response = requests.get(_ASDIV_XML_URL.format(revision=revision), timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = (
        {
            "Body": problem.findtext("Body", ""),
            "Question": problem.findtext("Question", ""),
            "Answer": problem.findtext("Answer", ""),
            "Grade": problem.get("Grade", ""),
        }
        for problem in root.iter("Problem")
    )
    return _skip_source_rows(source, rows, parameters)


def _load_hugging_face_dataset(source: Source, revision: str, config: str | None = None):
    return datasets.load_dataset(
        source.dataset_id,
        config,
        split=source.split,
        revision=revision,
        streaming=source.streaming,
    )


def _skip_source_rows(source: Source, rows, parameters: Mapping[str, Any]):
    _validate_source_parameters(source.name, parameters, {"skip"})
    skip = parameters.get("skip", 0)
    if type(skip) is not int or skip < 0:
        raise ValueError(f"{source.name} parameters.skip must be a non-negative integer.")
    return itertools.islice(rows, skip, None)


def _load_hugging_face_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    return _skip_source_rows(source, _load_hugging_face_dataset(source, revision), parameters)


def _load_eurus2_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    rows = _load_hugging_face_dataset(source, revision)
    code_rows = rows.filter(lambda example: example["ability"] == _EURUS2_CODE_ABILITY)
    return _skip_source_rows(source, code_rows, parameters)


def _load_nemotron_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    return _skip_source_rows(source, _load_hugging_face_dataset(source, revision, "RL"), parameters)


def _load_openscience_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    _validate_source_parameters(source.name, parameters, {"subset", "skip"})
    subset = parameters.get("subset")
    if subset not in OPENSCIENCE_SUBSETS:
        raise ValueError(
            f"{source.name} parameters.subset must select one of {list(OPENSCIENCE_SUBSETS)}; got {subset!r}."
        )
    rows = _skip_source_rows(
        source, _load_hugging_face_dataset(source, revision, subset), {"skip": parameters.get("skip", 0)}
    )
    return ({**example, "subset": subset} for example in rows)


def _iter_jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    """Read heterogeneous JSONL records without imposing an inferred Arrow schema."""
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error.msg}") from error
            if not isinstance(row, Mapping):
                raise TypeError(f"Expected a JSON object in {path} at line {line_number}.")
            yield row


def _load_nemotron_ultra_rows(source: Source, revision: str, parameters: Mapping[str, Any]):
    from huggingface_hub import hf_hub_download

    # Local import breaks the source/sidechannel module cycle while keeping the
    # heavyweight TaskTrove archive scan out of ordinary source imports.
    from infra.rl_data.nemotron_ultra_swe import bind_tasktrove_swe_proxies, load_tasktrove_swe_proxy_index

    filename = source.name.removeprefix("nemotron_ultra_") + ".jsonl"
    local_path = hf_hub_download(
        repo_id=source.dataset_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
    )
    rows = _skip_source_rows(source, _iter_jsonl_rows(Path(local_path)), parameters)
    rows = bind_tasktrove_swe_proxies(rows, load_tasktrove_swe_proxy_index())

    placeholder_sources = load_nemotron_ultra_placeholder_sources()
    return (restore_nemotron_ultra_placeholder(row, placeholder_sources) for row in rows)


def load_nemotron_ultra_placeholder_sources() -> dict[tuple[str, str], datasets.Dataset]:
    """Load the pinned datasets that NVIDIA's math placeholder rows point into."""
    return {
        (DAPO_MATH_DATASET, "train"): datasets.load_dataset(
            DAPO_MATH_DATASET,
            split="train",
            revision=NEMOTRON_ULTRA_DAPO_REVISION,
        ),
        ("Skywork/Skywork-OR1-RL-Data", "math"): datasets.load_dataset(
            "Skywork/Skywork-OR1-RL-Data",
            split="math",
            revision=NEMOTRON_ULTRA_SKYWORK_REVISION,
        ),
    }


def _unwrap_nemotron_answer(raw: Any) -> str:
    if not isinstance(raw, str):
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)
    stripped = raw.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(value, list) and value:
            return str(value[0])
        return str(value)
    return stripped


def restore_nemotron_ultra_placeholder(
    row: Mapping[str, Any],
    sources: Mapping[tuple[str, str], Any],
) -> Mapping[str, Any]:
    placeholder = row.get(_NEMOTRON_PLACEHOLDER_KEY)
    if not isinstance(placeholder, Mapping):
        return row
    dataset_id = str(placeholder["dataset"])
    split = str(placeholder["split"])
    source = sources[(dataset_id, split)][int(placeholder["row"])]
    bare = str(source["prompt"][0]["content"])
    if dataset_id == DAPO_MATH_DATASET:
        if _NEMOTRON_DAPO_PREFIX in bare:
            bare = bare.split(_NEMOTRON_DAPO_PREFIX, 1)[1]
        if _NEMOTRON_DAPO_SUFFIX in bare:
            bare = bare.rsplit(_NEMOTRON_DAPO_SUFFIX, 1)[0]
    bare = bare.strip()
    if placeholder.get("mode") == "canonical":
        question = str(placeholder.get("lead", "")) + bare + str(placeholder.get("trail", ""))
    else:
        question = str(placeholder.get("prefix", "")) + bare + str(placeholder.get("suffix", ""))
    answer = _unwrap_nemotron_answer((source.get("reward_model") or {}).get("ground_truth"))

    restored = copy.deepcopy(dict(row))
    restored.pop(_NEMOTRON_PLACEHOLDER_KEY)
    restored["question"] = question
    restored["expected_answer"] = answer
    request_input = restored.get("responses_create_params", {}).get("input")
    if isinstance(request_input, list) and request_input and isinstance(request_input[0], dict):
        request_input[0]["content"] = question
    for matched_source in restored.get("matched_sources") or []:
        if isinstance(matched_source, dict) and "expected_answer" in matched_source:
            matched_source["expected_answer"] = answer
    return restored


def load_source_rows(source: Source, revision: str, parameters: Mapping[str, Any] | None = None):
    """Load source rows only when the CLI is invoked, keeping core tests offline."""
    loader = source.load_rows or _load_hugging_face_rows
    return loader(source, revision, parameters or {})


SOURCES = {
    source.name: source
    for source in (
        rlvr_math_source(),
        dapo_math_source(),
        aime24_source(),
        math500_source(),
        rlvr_ifeval_source(),
        deepscaler_source(),
        gsm8k_source(),
        hendrycks_math_source(),
        aime_1983_2024_source(),
        asdiv_source(),
        svamp_source(),
        numina_math_source(),
        hardmath_source(),
        verifiable_code_source(),
        apps_source(),
        eurus2_code_source(),
        nemotron_if_source(),
        reasoning_gym_source(),
        gpqa_source(),
        openscience_source(),
        kto_mix_source(),
        hh_rlhf_source(),
        gretel_text_to_sql_source(),
        nemotron_ultra_rlvr1_source(),
        nemotron_ultra_rlvr2_source(),
        nemotron_ultra_mopd_source(),
    )
}


def source_by_name(name: str) -> Source:
    """Return a named supported source or explain the supported names."""
    try:
        return SOURCES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown RLVR source {name!r}; choose from {sorted(SOURCES)}.") from exc
