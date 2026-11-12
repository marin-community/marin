"""Math and symmetric final-label judging through existing verifyit modes."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Any

from verifyit.grade import Status, run
from verifyit.spec import JudgeSpec, MathProfile, MathSpec, ScriptSpec, render_spec

from skyrl_gym.envs.nemotron_ultra.answer_extraction import (
    final_answer_text,
    last_boxed_answer,
)
from skyrl_gym.envs.nemotron_ultra.math_references import reference_kind
from skyrl_gym.envs.nemotron_ultra.math_with_judge import (
    _JUDGE_PROMPT,
    _JUDGE_SYSTEM,
    _strip_delimiters,
)


def _balanced_reference_delimiters(reference: str) -> bool:
    depth = 0
    for index, character in enumerate(reference):
        if index and reference[index - 1] == "\\":
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth:
        return False
    sizes = re.findall(r"\\(?:big|Big|bigg|Bigg)([lr])(?![A-Za-z])", reference)
    depth = 0
    for side in sizes:
        depth += 1 if side == "l" else -1
        if depth < 0:
            return False
    return depth == 0


def _evaluate(data: dict[str, Any], root: Path) -> dict[str, Any]:
    record = data["record"]
    text = final_answer_text(data["text"])
    try:
        kind = reference_kind(record)
    except (ValueError, UnicodeError) as error:
        return {
            "schema_version": 1,
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"reason": str(error)},
        }
    if (
        not isinstance(record.get("expected_answer"), str)
        or not record["expected_answer"].strip()
        or not isinstance(record.get("question"), str)
        or (kind == "symbolic" and not _balanced_reference_delimiters(record["expected_answer"]))
    ):
        return {
            "schema_version": 1,
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"reason": "Malformed math reference"},
        }
    from math_verify import parse
    from math_verify.parser import LatexExtractionConfig

    reference_values = []
    if kind != "semantic":
        try:
            reference_values = parse(
                r"\boxed{" + _strip_delimiters(record["expected_answer"]) + "}",
                extraction_config=[LatexExtractionConfig()],
                raise_on_error=True,
            )
        except Exception:
            # Normal syntax rejection is an empty/string parse. Raised failures
            # are not distinguishable from parser infrastructure errors here.
            return {
                "schema_version": 1,
                "status": "infra_error",
                "reward": 0.0,
                "detail": {"reason": "Mathematical reference parser failed"},
            }
    symbolic_reference = any(not isinstance(value, str) for value in reference_values)
    # The pinned source is a semantic-reference judge with a symbolic shortcut.
    # A parser miss in the legacy route remains subject to the actual symmetric
    # judge, never a positive reward derived from admission itself.
    prose_reference = kind != "symbolic" and not symbolic_reference
    typographic_reference = False
    if not symbolic_reference and not prose_reference:
        # These sizing commands change rendering, not the trusted mathematical value.
        normalized_reference = re.sub(
            r"\\(?:big|Big|bigg|Bigg)(?:l|r)?(?![A-Za-z])",
            "",
            _strip_delimiters(record["expected_answer"]),
        )
        if normalized_reference != _strip_delimiters(record["expected_answer"]):
            normalized_values = parse(
                r"\boxed{" + normalized_reference + "}",
                extraction_config=[LatexExtractionConfig()],
                raise_on_error=True,
            )
            typographic_reference = any(not isinstance(value, str) for value in normalized_values)
    if not symbolic_reference and not prose_reference and not typographic_reference:
        return {
            "schema_version": 1,
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"reason": "Unparsed mathematical reference"},
        }
    if not text:
        return {
            "schema_version": 1,
            "status": "scored",
            "reward": 0.0,
            "detail": {
                "source_feedback": {
                    "result": "missing_final_answer",
                    "extracted_answer": None,
                }
            },
        }
    candidate_path = root / "answer.txt"
    spec_path = root / "verifier.toml"
    boxed = last_boxed_answer(text)
    pure = re.fullmatch(r"[\\\w\s{}()+*/^.,=+\-]+", text) and not re.search(r"\b[A-Za-z]{3,}\b", text)
    reward = 0.0
    extracted = None
    if (prose_reference or typographic_reference) and (boxed is not None or pure):
        # Source feedback is metadata: parsing does not determine correctness.
        from math_verify import parse
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        candidate = r"\boxed{" + boxed + "}" if boxed is not None else text
        predictions = parse(
            candidate,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
            raise_on_error=True,
        )
        extracted = str(predictions[0]) if predictions else None
    if symbolic_reference and (boxed is not None or pure):
        candidate = r"\boxed{" + boxed + "}" if boxed is not None else text
        candidate_path.write_text(candidate)
        integration = bool(re.search(r"indefinite|antiderivative|primitive", record["question"], re.I))
        spec_path.write_text(
            render_spec(
                MathSpec(
                    expected=_strip_delimiters(record["expected_answer"]),
                    output=str(candidate_path),
                    profile=MathProfile.RAW,
                    allow_additive_constant=integration,
                )
            )
        )
        result = run(spec_path, root)
        if result.status is not Status.SCORED:
            return {
                "schema_version": 1,
                "status": result.status.value,
                "reward": 0.0,
                "detail": {
                    "reason": "Math verifier failed",
                    "math_verdict": dataclasses.asdict(result),
                },
            }
        reward = result.reward
        extracted = result.detail.get("parsed_candidate")
    diagnostics = {"library_reward": reward, "extracted_answer": extracted}
    if reward > 0.5:
        return {
            "schema_version": 1,
            "status": "scored",
            "reward": reward,
            "detail": {"source_feedback": diagnostics},
        }
    judge = data.get("judge")
    if not judge:
        return {
            "schema_version": 1,
            "status": "infra_error",
            "reward": 0.0,
            "detail": {"reason": "General judge is required"},
        }
    os.environ["VERIFYIT_JUDGE_BASE_URL"] = judge["base_url"]
    os.environ["VERIFYIT_JUDGE_MODEL"] = judge["model"]
    os.environ["VERIFYIT_JUDGE_API_KEY"] = os.environ.get(
        judge.get("api_key_env") or "", judge.get("api_key", "dummy_key")
    )
    diagnostics["judge_outputs"] = []
    template = _JUDGE_PROMPT.replace("{first}", "{reference}").replace("{second}", "{candidate}")
    for reference, candidate in (
        (record["expected_answer"], text),
        (text, record["expected_answer"]),
    ):
        candidate_path.write_text(candidate)
        spec_path.write_text(
            render_spec(
                JudgeSpec(
                    rubric="labels",
                    references=(reference,),
                    question=record["question"],
                    system_prompt=_JUDGE_SYSTEM,
                    prompt_template=template,
                    label_scores={"[[A=B]]": 1.0, "[[A!=B]]": 0.0},
                    strip_reasoning_blocks=True,
                    output=str(candidate_path),
                    request_timeout=judge["timeout_seconds"],
                    max_completion_tokens=8192,
                    incomplete_retry_tokens=16384,
                    reasoning_effort=judge.get("reasoning_effort") or "",
                )
            )
        )
        result = run(spec_path, root)
        if result.status is not Status.SCORED:
            return {
                "schema_version": 1,
                "status": result.status.value,
                "reward": 0.0,
                "detail": {
                    "reason": "Judge verifier failed",
                    "judge_verdict": dataclasses.asdict(result),
                },
            }
        diagnostics["judge_outputs"].append(result.detail["completion"])
        if result.reward == 0.0:
            return {
                "schema_version": 1,
                "status": "scored",
                "reward": 0.0,
                "detail": {"source_feedback": diagnostics},
            }
    return {
        "schema_version": 1,
        "status": "scored",
        "reward": 1.0,
        "detail": {"source_feedback": diagnostics},
    }


def grade_math_verifyit(
    text: str, record: dict[str, Any], *, judge, timeout_seconds: float = 60.0
) -> tuple[float, dict[str, Any]]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= 3600
        or not math.isfinite(timeout_seconds)
    ):
        return 0.0, {
            "error_type": "schema_error",
            "error_message": "Invalid total verifier deadline",
        }
    with tempfile.TemporaryDirectory(prefix="skyrl-math-judge-") as directory:
        root = Path(directory)
        payload = root / "input.json"
        try:
            payload.write_text(
                json.dumps(
                    {
                        "text": text,
                        "record": record,
                        "judge": dataclasses.asdict(judge) if judge else None,
                    },
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError):
            return 0.0, {
                "error_type": "schema_error",
                "error_message": "Invalid math verifier reference/configuration",
            }
        checker = root / "check.sh"
        checker.write_text(
            "set -eu\nexec "
            + shlex.quote(sys.executable)
            + " "
            + "-m skyrl_gym.envs.nemotron_ultra.math_judge_verifyit"
            + " --check "
            + shlex.quote(str(payload))
            + "\n"
        )
        spec_path = root / "outer.toml"
        spec_path.write_text(
            render_spec(
                ScriptSpec(
                    path=checker.name,
                    timeout=float(timeout_seconds),
                    verdict_file="math-judge-verdict.json",
                )
            )
        )
        result = run(spec_path, root)
        if result.status is not Status.SCORED:
            return 0.0, {
                "error_type": ("schema_error" if result.status is Status.INVALID_TASK else "verification_error"),
                "error_message": "Math/judge verification failed",
                "verifyit_verdict": dataclasses.asdict(result),
            }
        return result.reward, result.detail["source_feedback"]


def _main() -> None:
    data = json.loads(Path(sys.argv[2]).read_text())
    calls = []
    active = {}

    def observe(frame, event, arg):
        if frame.f_code.co_name != "grade" or not frame.f_code.co_filename.endswith(
            ("/verifyit/modes/grade_math.py", "/verifyit/modes/grade_judge.py")
        ):
            return
        if event == "call":
            item = {
                "path": frame.f_code.co_filename,
                "spec": dataclasses.asdict(frame.f_locals["spec"]),
                "candidate": Path(frame.f_locals["spec"].output).read_text(),
            }
            calls.append(item)
            active[id(frame)] = item
        elif event == "return" and id(frame) in active:
            active.pop(id(frame))["verdict"] = dataclasses.asdict(arg) if arg else None

    sys.setprofile(observe)
    directory = Path(sys.argv[2]).parent / "inner"
    directory.mkdir()
    try:
        verdict = _evaluate(data, directory)
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
    destination = Path(os.environ["VERIFYIT_LOGS_DIR"]) / "math-judge-verdict.json"
    destination.write_text(json.dumps(verdict, allow_nan=False))


if __name__ == "__main__":
    _main()
