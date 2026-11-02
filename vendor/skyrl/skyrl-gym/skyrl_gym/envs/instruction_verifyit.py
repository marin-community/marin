"""Pinned source instruction predicates executed by IFEval under one deadline."""

from __future__ import annotations

import inspect
import json
import math
import random
from typing import Any

from verifyit.execution.worker import call_bounded


class ReferenceError(ValueError):
    """A supplied instruction descriptor cannot define a valid task."""


def _finite(node: Any) -> bool:
    if isinstance(node, float):
        return math.isfinite(node)
    if isinstance(node, dict):
        return all(_finite(value) for value in node.values())
    if isinstance(node, list):
        return all(_finite(value) for value in node)
    return True


def _validated_random_state(value):
    """Validate the trusted subprocess's JSON state before mutating host RNG."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Malformed random state")
    version, words, gaussian = value
    if version != 3 or type(version) is not int or not isinstance(words, (list, tuple)) or len(words) != 625:
        raise ValueError("Malformed random state")
    if (
        any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words[:-1])
        or type(words[-1]) is not int
        or not 0 <= words[-1] <= 624
    ):
        raise ValueError("Malformed random state")
    if gaussian is not None and (type(gaussian) not in (int, float) or not math.isfinite(gaussian)):
        raise ValueError("Malformed Gaussian state")
    state = (version, tuple(words), gaussian)
    random.Random().setstate(state)
    return state


def _check(text: str, params: dict, predicate) -> tuple[bool, str]:
    try:
        if not isinstance(text, str):
            raise RuntimeError("Instruction candidate frame must contain text")
        passed = predicate(text, **params)
        if type(passed) is not bool:
            raise RuntimeError("Instruction checker returned a nonboolean result")
        return passed, json.dumps({"passed": passed, "error": None})
    except Exception as error:
        return False, json.dumps({"passed": False, "error": f"{type(error).__name__}: {str(error)[:200]}"})


def _json_format(text: str, fenced: bool = False) -> bool:
    from verifyit.grade import Status
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate

    if fenced:
        text = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    try:
        value = json.loads(text)
    except ValueError:
        return False
    verdict = grade_json_schema_candidate({}, value)
    if verdict.status is not Status.SCORED:
        raise RuntimeError("JSON format verifier did not produce a scored result")
    return verdict.reward == 1.0


def _references(kind: str, data: Any, *, runtime: bool = False):
    if not _finite(data):
        raise ReferenceError("Nonfinite instruction reference")
    references = []
    if kind == "standalone":
        from skyrl_gym.envs.ifeval import utils

        decoded = json.loads(data) if isinstance(data, str) else data
        specs = decoded if isinstance(decoded, list) else [decoded]
        if not specs or not all(isinstance(spec, dict) for spec in specs):
            raise ReferenceError("Instructions must be a nonempty list of objects")
        for spec in specs:
            normalized = utils._normalize_constraint(spec)
            name = normalized.pop("func_name")
            if not _finite(normalized):
                raise ReferenceError("Nonfinite instruction parameters")
            for key, value in normalized.items():
                if key in {"N", "i"} and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0 or (key == "i" and value == 0)
                ):
                    raise ReferenceError("Invalid instruction count or index")
                if key in {"keyword_list", "forbidden_words", "options"} and (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item for item in value)
                ):
                    raise ReferenceError("Empty or malformed instruction word policy")
                if key not in {
                    "N",
                    "i",
                    "keyword_list",
                    "forbidden_words",
                    "options",
                } and (not isinstance(value, str) or not value):
                    raise ReferenceError("Invalid instruction text parameter")
            if name in {"validate_placeholders", "validate_highlighted_sections"} and normalized["N"] == 0:
                raise ReferenceError("Vacuous minimum instruction count")
            if normalized.get("N") == 0 and normalized.get("quantifier") == "at least":
                raise ReferenceError("Vacuous minimum instruction count")
            predicate = _json_format if name == "validate_json_format" else utils.IF_FUNCTIONS_MAP[name]
            inspect.signature(predicate).bind("", **normalized)
            references.append(("marin_skyrl:rlvr:" + name, normalized, predicate))
        return references, "fraction", decoded
    if kind != "nemotron":
        raise ReferenceError("Unknown instruction profile")
    from verifiable_instructions import instructions_registry
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        _ensure_nltk_data,
        build_instruction,
    )
    from importlib.metadata import distribution

    provenance = json.loads(distribution("verifiable-instructions").read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != "f46a5ac87b1400a4f8973039844b6be9b56e3faf":
        raise RuntimeError("Instruction registry does not match the pinned source dependency")

    if not isinstance(data, dict):
        raise ReferenceError("Instruction record must be an object")
    if (
        "instruction_reference_seed" in data
        and data.get("instruction_reference_revision") != "f46a5ac87b1400a4f8973039844b6be9b56e3faf"
    ):
        raise ReferenceError("Frozen instruction references do not match the pinned registry")
    ids = data.get("instruction_id_list")
    kwargs = data.get("kwargs")
    if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
        raise ReferenceError("Instruction IDs and kwargs must be nonempty aligned lists")
    mode = data.get("grading_mode", "binary")
    if mode not in {"binary", "fraction"}:
        raise ReferenceError("Invalid instruction grading mode")
    # Validate every reference before any candidate is graded.
    for identity, arguments in zip(ids, kwargs):
        if not isinstance(identity, str) or identity not in instructions_registry.INSTRUCTION_DICT:
            raise ReferenceError("Unknown instruction ID")
        if arguments is not None and not isinstance(arguments, dict):
            raise ReferenceError("Instruction kwargs must be objects")
        arguments = {key: value for key, value in (arguments or {}).items() if value is not None}
        runtime_instruction = None
        if runtime:
            try:
                runtime_instruction = build_instruction(identity, arguments)
                arguments = runtime_instruction.get_instruction_args() or {}
                if not isinstance(arguments, dict) or not _finite(arguments):
                    raise ValueError("Invalid resolved instruction parameters")
            except Exception as error:
                raise ReferenceError("Invalid instruction kwargs") from error
        for key, value in arguments.items():
            if key.startswith("num_") or key in {
                "N",
                "m",
                "n",
                "n_end",
                "n_sent",
                "n_start",
                "n_words",
                "small_n",
                "nth_paragraph",
                "nth_word",
                "capital_frequency",
                "frequency",
                "let_frequency",
            }:
                signed_span_index = identity == "new:copy_span_idx" and key in {
                    "n_start",
                    "n_end",
                }
                if isinstance(value, bool) or not isinstance(value, int) or (value < 0 and not signed_span_index):
                    raise ReferenceError("Invalid instruction count")
            if key in {"keywords", "forbidden_words", "options"} and (
                not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value)
            ):
                raise ReferenceError("Empty instruction word policy")
        instruction = instructions_registry.INSTRUCTION_DICT[identity](identity)
        parameters = inspect.signature(instruction.build_description).parameters
        if identity == "keywords:exclude_word_harder":
            keyword = arguments.get("keyword")
            if not isinstance(keyword, str) or not keyword.strip():
                raise ReferenceError("Exclusion keyword must be frozen before grading")
        for key, parameter in parameters.items():
            if identity == "keywords:exclude_word_harder" and key == "instruction":
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if not runtime and parameter.default is None and key not in arguments:
                raise ReferenceError("Missing deterministic instruction parameter")
        if identity in {
            "detectable_content:number_placeholders",
            "detectable_format:number_highlighted_sections",
        }:
            count_key = "num_placeholders" if "num_placeholders" in arguments else "num_highlights"
            if arguments.get(count_key) == 0:
                raise ReferenceError("Vacuous minimum instruction count")
        minimum_policies = {
            "keywords:frequency": ("frequency", "relation"),
            "keywords:letter_frequency": ("let_frequency", "let_relation"),
            "length_constraints:number_sentences": ("num_sentences", "relation"),
            "length_constraints:number_words": ("num_words", "relation"),
            "change_case:capital_word_frequency": (
                "capital_frequency",
                "capital_relation",
            ),
            "keywords:word_count_different_numbers": ("frequency", "relation"),
            "letters:letter_counting2": ("let_frequency", "let_relation"),
            "detectable_format:multiple_sections": ("num_sections", None),
        }
        if identity in minimum_policies:
            count_key, relation_key = minimum_policies[identity]
            if arguments.get(count_key) == 0 and (relation_key is None or arguments.get(relation_key) == "at least"):
                raise ReferenceError("Vacuous minimum instruction count")
        state = random.getstate()
        try:
            instruction = runtime_instruction if runtime else build_instruction(identity, arguments)
        except Exception as error:
            raise ReferenceError("Invalid instruction kwargs") from error
        if not runtime and random.getstate() != state:
            raise ReferenceError("Instruction construction randomized the reference")
        if (
            identity == "new:copy_span_idx"
            and not instruction._prompt_to_repeat[instruction._n_start : instruction._n_end].strip()
        ):
            raise ReferenceError("Empty instruction copy span")
        predicate = instruction.check_following
        if identity == "language:response_language":
            language = instruction._language

            def predicate(text, language=language):
                import langdetect

                return langdetect.detect(text) == language

        elif identity == "change_case:english_capital":

            def predicate(text):
                import langdetect

                return text.isupper() and langdetect.detect(text) == "en"

        elif identity == "change_case:english_lowercase":

            def predicate(text):
                import langdetect

                return text.islower() and langdetect.detect(text) == "en"

        if identity == "detectable_format:json_format":

            def predicate(text):
                return _json_format(text, fenced=True)

        references.append(("marin_skyrl:nvidia:" + identity, {}, predicate))
    _ensure_nltk_data()
    return references, mode, data


def _evaluate(kind: str, text: str, data: Any, *, runtime: bool = False) -> dict:
    from verifyit.grade import InvalidTask
    from verifyit.modes.grade_ifeval import grade_ifeval_candidate
    from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec

    try:
        references, mode, decoded = _references(kind, data, runtime=runtime)
    except (ReferenceError, ValueError, TypeError, KeyError):
        return {
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {
                "error_type": "schema_error",
                "error_message": "Invalid instruction configuration",
            },
        }
    constraints = []
    registry = {}
    for index, (name, params, predicate) in enumerate(references):
        if name == "marin_skyrl:rlvr:verify_keywords":
            name, params = (
                "keywords:existence",
                {
                    "keywords": params["keyword_list"],
                    "word_boundary": False,
                },
            )
        elif name == "marin_skyrl:rlvr:validate_forbidden_words":
            name, params = (
                "keywords:forbidden_words",
                {
                    **params,
                    "word_boundary": False,
                },
            )
        elif name == "marin_skyrl:rlvr:validate_no_commas":
            name = "punctuation:no_comma"
        elif name in {
            "marin_skyrl:nvidia:punctuation:no_comma",
            "marin_skyrl:nvidia:punctuation:punctuation_dot",
            "marin_skyrl:nvidia:punctuation:punctuation_exclamation",
        }:
            name = name.removeprefix("marin_skyrl:nvidia:")
        else:
            name += ":" + str(index)

            def check(candidate, arguments, predicate=predicate):
                return _check(candidate, arguments, predicate)

            registry[name] = check
        constraints.append(Constraint(name, params))
    try:
        verdict = grade_ifeval_candidate(
            IfevalSpec(tuple(constraints), empty_output=EmptyOutputPolicy.GRADE),
            text,
            registry=registry,
        )
    except InvalidTask:
        return {
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"error_type": "schema_error"},
        }
    results = []
    errors = []
    for result in verdict.detail["constraints"]:
        results.append(result["passed"])
        errors.append(json.loads(result["detail"])["error"] if result["name"] in registry else None)
    if any(errors):
        return {
            "status": "infra_error",
            "reward": 0.0,
            "detail": {
                "error_type": "verification_error",
                "instruction_errors": errors,
                "error_message": "Instruction checker failed",
            },
        }
    reward = float(all(results)) if mode == "binary" else sum(results) / len(results)
    if kind == "standalone":
        feedback = {
            "score": reward,
            "acc": all(results),
            "func_name": (decoded.get("func_name") if isinstance(decoded, dict) else None),
            "constraints_satisfied": sum(results),
            "constraints_total": len(results),
        }
    else:
        feedback = {
            "follow_all_instructions": all(results),
            "follow_instruction_list": results,
            "instruction_errors": errors,
            "grading_mode": mode,
            "num_passed": sum(results),
            "num_total": len(results),
        }
    return {
        "status": "scored",
        "reward": reward,
        "detail": {"source_feedback": feedback},
    }


def _execute(kind: str, text: str, data: Any, timeout: float) -> tuple[float, dict]:
    try:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 300
            or not math.isfinite(timeout)
        ):
            return 0.0, {
                "error_type": "schema_error",
                "error_message": "Invalid instruction deadline",
            }
        verdict = call_bounded(
            _evaluate_with_state,
            kind,
            text,
            data,
            random.getstate() if kind == "nemotron" else None,
            timeout=timeout,
        )
        if verdict["status"] != "scored":
            category = "schema_error" if verdict["status"] == "invalid_task" else "verification_error"
            return 0.0, {
                "error_type": category,
                "error_message": "Instruction verifier failed or exceeded deadline",
            }
        feedback = verdict["detail"]["source_feedback"]
        if not isinstance(feedback, dict):
            raise ValueError("Malformed instruction feedback")
        if kind == "nemotron":
            results = feedback["follow_instruction_list"]
            errors = feedback["instruction_errors"]
            mode = feedback["grading_mode"]
            if (
                not isinstance(results, list)
                or not results
                or len(results) != len(data["instruction_id_list"])
                or any(type(value) is not bool for value in results)
                or errors != [None] * len(results)
                or type(feedback["follow_all_instructions"]) is not bool
                or feedback["follow_all_instructions"] != all(results)
                or type(feedback["num_passed"]) is not int
                or feedback["num_passed"] != sum(results)
                or type(feedback["num_total"]) is not int
                or feedback["num_total"] != len(results)
                or mode != data.get("grading_mode", "binary")
                or mode not in {"binary", "fraction"}
                or verdict["reward"] != (float(all(results)) if mode == "binary" else sum(results) / len(results))
            ):
                raise ValueError("Malformed instruction feedback")
            next_state = _validated_random_state(verdict["detail"]["random_state"])
            random.setstate(next_state)
        return verdict["reward"], feedback
    except Exception:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "Instruction verification failed",
        }


def grade_standalone_instructions(text: str, ground_truth: Any, timeout: float = 30.0) -> dict:
    reward, feedback = _execute("standalone", text, ground_truth, timeout)
    if "error_type" in feedback:
        return {
            "score": 0.0,
            "acc": False,
            "constraints_satisfied": 0,
            "constraints_total": 0,
            **feedback,
        }
    return feedback


def grade_nemotron_instructions(text: str, record: dict, timeout: float = 30.0) -> tuple[float, dict]:
    return _execute("nemotron", text, record, timeout)


def _evaluate_with_state(kind: str, text: str, data: Any, state) -> dict:
    runtime = kind == "nemotron"
    if runtime:
        random.setstate(_validated_random_state(state))
    verdict = _evaluate(kind, text, data, runtime=runtime)
    if runtime and verdict["status"] == "scored":
        verdict["detail"]["random_state"] = random.getstate()
    return verdict
