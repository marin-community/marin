"""Bounded client compositions for source abstention, rubric and policy judges."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

from verifyit.grade import InvalidTask, Status, run
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec, JudgeSpec, ScriptSpec, render_spec

from skyrl_gym.envs.nemotron_ultra.answer_extraction import (
    final_answer_text,
    last_boxed_answer,
)
from skyrl_gym.envs.nemotron_ultra.jailbreak import _configuration
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import (
    _MULTICHALLENGE_PROMPT,
    _MULTICHALLENGE_SYSTEM,
    _normalize_answer,
)
from importlib.resources import files


def _text(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise InvalidTask(f"Invalid {name}")
    return value


def _literal(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def _evaluate(data: dict[str, Any], root: Path) -> dict[str, Any]:
    record, kind, judge = data["record"], data["kind"], data["judge"]
    if not isinstance(record, dict) or not isinstance(judge, dict):
        raise InvalidTask("Missing judge task configuration")
    os.environ["VERIFYIT_JUDGE_BASE_URL"] = judge["base_url"]
    os.environ["VERIFYIT_JUDGE_MODEL"] = judge["model"]
    os.environ["VERIFYIT_JUDGE_API_KEY"] = os.environ.get(
        judge.get("api_key_env") or "", judge.get("api_key", "dummy_key")
    )
    candidate_file, spec_file = root / "answer.txt", root / "verifier.toml"

    def decide(
        candidate: str,
        reference: str,
        question: str,
        template: str,
        labels: dict[str, float],
        *,
        system: str = "",
        scan: str = "literal",
    ):
        candidate_file.write_text(candidate)
        spec_file.write_text(
            render_spec(
                JudgeSpec(
                    rubric="labels",
                    references=(reference,),
                    question=question,
                    prompt_template=template,
                    system_prompt=system,
                    label_scores=labels,
                    strip_reasoning_blocks=True,
                    label_scan=scan,
                    output=str(candidate_file),
                    request_timeout=judge["timeout_seconds"],
                    max_completion_tokens=8192,
                    reasoning_effort=judge.get("reasoning_effort") or "",
                )
            )
        )
        result = run(spec_file, root)
        if result.status is not Status.SCORED:
            raise (
                InvalidTask("Judge task invalid")
                if result.status is Status.INVALID_TASK
                else RuntimeError("Judge verification failed")
            )
        return result.reward, result.detail["verdict"], result.detail["completion"]

    if kind == "abstention":
        question = _text(record.get("question"), "question")
        expected = _text(record.get("answer"), "answer")
        response = final_answer_text(_text(data["text"], "response", nonempty=False))
        extracted = last_boxed_answer(response) or response
        exact = grade_exact_candidate(
            ExactSpec(
                expected=("idk",),
                ignore_case=False,
                ignore_whitespace=False,
                strip_outer_whitespace=False,
            ),
            _normalize_answer(extracted),
        )
        if exact.status is not Status.SCORED:
            raise RuntimeError("Abstention normalization failed")
        # The source's NOT_ATTEMPTED label includes an omitted final answer.
        if not extracted.strip() or exact.reward == 1.0:
            reward, feedback = (
                0.5,
                {
                    "verdict": "abstain",
                    "extracted_answer": extracted,
                },
            )
        else:
            template = (
                files(__package__)
                .joinpath("abstention_prompt.txt")
                .read_text()
                .replace("{target}", "{reference}")
                .replace("{predicted_answer}", "{candidate}")
            )
            reward, label, output = decide(
                extracted,
                expected,
                question,
                template,
                {"A": 1.0, "B": 0.0, "C": 0.5},
                scan="lines",
            )
            feedback = {
                "verdict": {"A": "correct", "B": "incorrect", "C": "abstain"}[label],
                "extracted_answer": extracted,
                "judge_output": output,
                "omniscience_index": {"A": 1.0, "B": -1.0, "C": 0.0}[label],
            }
    elif kind == "multichallenge":
        rubric = record.get("rubric") or record.get("metadata", {}).get("rubric") or []
        if not isinstance(rubric, list) or not rubric:
            raise InvalidTask("Rubric must contain meaningful criteria")
        context = _text(record.get("context", ""), "context", nonempty=False)
        prepared = []
        for item in rubric:
            if not isinstance(item, dict):
                raise InvalidTask("Malformed rubric criterion")
            question = _text(item.get("question"), "criterion")
            expected = _text(item.get("pass_criteria", "YES"), "pass criteria")
            prepared.append((question, expected))
        response = final_answer_text(_text(data["text"], "response", nonempty=False))
        template = (
            _MULTICHALLENGE_PROMPT.replace("{response}", "{candidate}")
            .replace("{pass_criteria}", "{reference}")
            .replace("{context}", _literal(context))
        )
        evaluations = []
        for question, expected in prepared:
            correct_label = "[[NO]]" if expected.upper() == "NO" else "[[YES]]"
            labels = {label: float(label == correct_label) for label in ("[[YES]]", "[[NO]]")}
            score, label, output = decide(
                response,
                expected,
                question,
                template,
                labels,
                system=_MULTICHALLENGE_SYSTEM,
            )
            evaluations.append(
                {
                    "question": question,
                    "verdict": label[2:-2],
                    "score": score,
                    "judge_output": output,
                }
            )
        reward = sum(item["score"] for item in evaluations) / len(evaluations)
        feedback = {
            "rubric_evaluations": evaluations,
            "num_passed": sum(item["score"] >= 0.99 for item in evaluations),
            "num_total": len(evaluations),
        }
    elif kind == "jailbreak":
        config = _configuration()
        policy = record.get("response_policy_mapped")
        policy_map = config["policy_verifier_map"]
        if policy not in policy_map:
            raise InvalidTask("Unsupported response policy")
        question = _text(record.get("adversarial_prompt"), "adversarial prompt")
        response = _text(data["text"], "response", nonempty=False)
        names = policy_map[policy]["verifiers"]
        combination = policy_map[policy].get("reward_combination", "product")
        if not names or combination not in {"product", "average", "first"}:
            raise InvalidTask("Malformed policy composition")
        rewards, labels, outputs = {}, {}, {}
        for name in names:
            verifier = config[name]
            positive, negative = (
                verifier["labels"]["positive"],
                verifier["labels"]["negative"],
            )
            reward_table = {
                positive: verifier["rewards"].get("positive", 1.0),
                negative: verifier["rewards"].get("negative", 0.0),
            }
            prompt = verifier["prompt_template"]
            if prompt.count("{model_response}") != 1:
                raise InvalidTask("Malformed policy prompt")
            prefix, suffix = prompt.split("{model_response}")
            prefix = prefix.format(adversarial_prompt=question)
            suffix = suffix.replace("{adversarial_prompt}", "{question}")
            template = "{reference}{candidate}" + suffix
            score, label, output = decide(response, prefix, question, template, reward_table)
            rewards[name], labels[name], outputs[name] = score, label, output
        reward = (
            math.prod(rewards.values())
            if combination == "product"
            else (sum(rewards.values()) / len(rewards) if combination == "average" else next(iter(rewards.values())))
        )
        feedback = {
            "response_policy": policy,
            "verifier_rewards": rewards,
            "verifier_labels": labels,
            "judge_outputs": outputs,
        }
    else:
        raise InvalidTask("Unknown judge profile")
    return {
        "schema_version": 1,
        "status": "scored",
        "reward": reward,
        "detail": {"source_feedback": feedback},
    }


def grade_judge_profile_verifyit(
    text: str,
    record: dict[str, Any],
    judge,
    *,
    kind: str,
    timeout_seconds: float = 120.0,
) -> tuple[float, dict[str, Any]]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= 3600
        or not math.isfinite(timeout_seconds)
    ):
        return 0.0, {
            "error_type": "schema_error",
            "error_message": "Invalid judge deadline",
        }
    with tempfile.TemporaryDirectory(prefix="skyrl-judge-profile-") as directory:
        root = Path(directory)
        payload = root / "input.json"
        try:
            payload.write_text(
                json.dumps(
                    {
                        "text": text,
                        "record": record,
                        "kind": kind,
                        "judge": dataclasses.asdict(judge),
                    },
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError):
            return 0.0, {
                "error_type": "schema_error",
                "error_message": "Invalid judge task",
            }
        checker = root / "check.sh"
        checker.write_text(
            "set -eu\nexec "
            + shlex.quote(sys.executable)
            + " -m skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit --check "
            + shlex.quote(str(payload))
            + "\n"
        )
        spec_path = root / "outer.toml"
        spec_path.write_text(
            render_spec(
                ScriptSpec(
                    path=checker.name,
                    timeout=float(timeout_seconds),
                    verdict_file="judge-profile-verdict.json",
                )
            )
        )
        result = run(spec_path, root)
        if result.status is not Status.SCORED:
            return 0.0, {
                "error_type": ("schema_error" if result.status is Status.INVALID_TASK else "verification_error"),
                "error_message": "Judge verification failed",
                "verifyit_verdict": dataclasses.asdict(result),
            }
        return result.reward, result.detail["source_feedback"]


def _main() -> None:
    root = Path(sys.argv[2]).parent
    data = json.loads(Path(sys.argv[2]).read_text())
    inner = root / "inner"
    inner.mkdir()
    calls, active = [], {}

    def observe(frame, event, arg):
        if frame.f_code.co_name not in {
            "grade",
            "grade_exact_candidate",
        } or not frame.f_code.co_filename.endswith(
            ("/verifyit/modes/grade_judge.py", "/verifyit/modes/grade_exact.py")
        ):
            return
        if event == "call":
            item = {
                "path": frame.f_code.co_filename,
                "spec": dataclasses.asdict(frame.f_locals["spec"]),
                "candidate": (
                    frame.f_locals.get("candidate")
                    if frame.f_code.co_name == "grade_exact_candidate"
                    else Path(frame.f_locals["spec"].output).read_text()
                ),
            }
            calls.append(item)
            active[id(frame)] = item
        elif event == "return" and id(frame) in active:
            active.pop(id(frame))["verdict"] = dataclasses.asdict(arg) if arg else None

    sys.setprofile(observe)
    try:
        verdict = _evaluate(data, inner)
    except (InvalidTask, KeyError, TypeError, ValueError) as error:
        verdict = {
            "schema_version": 1,
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"reason": type(error).__name__},
        }
    except Exception as error:
        verdict = {
            "schema_version": 1,
            "status": "infra_error",
            "reward": 0.0,
            "detail": {"reason": type(error).__name__},
        }
    finally:
        sys.setprofile(None)
    verdict["detail"]["primitive_calls"] = calls
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "judge-profile-verdict.json").write_text(
        json.dumps(verdict, allow_nan=False)
    )


if __name__ == "__main__":
    _main()
