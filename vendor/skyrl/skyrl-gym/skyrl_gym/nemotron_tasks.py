"""Direct task session for the released Nemotron Ultra source rows."""

import json
import re
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import reasoning_gym
import requests
from rolloutengine.contracts import LENGTH_STOP_REASON, ModelTurn, SessionStart, Transition
from shellbox.machine import ExitReason, Machine
from skyrl_gym.source_task import session_parameters
from taskcompendium.grading_result import GradeResult, Outcome
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start
from verifyit.adapters.skyrl import grade_grid_candidate

from skyrl_gym.code_execution import execute_code
from skyrl_gym.envs.lcb.livecodebench import (
    DEFAULT_LIMITS,
    VerifierLimits,
    extract_code_from_model,
    normalize_lcb_ground_truth,
)
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer
from skyrl_gym.envs.nemotron_ultra import GENRM_AGENTS, NEMOTRON_ULTRA_MOPD_AGENTS
from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar
from skyrl_gym.envs.nemotron_ultra.calendar_verifyit import grade_calendar_verifyit
from skyrl_gym.envs.nemotron_ultra.code_gen import DEFAULT_PER_TEST_TIMEOUT_SECONDS, has_reasoning_format_violation
from skyrl_gym.envs.nemotron_ultra.format_verification import grade_format
from skyrl_gym.envs.nemotron_ultra.format_verifyit import grade_format_verifyit
from skyrl_gym.envs.nemotron_ultra.instruction_following import grade_instruction_following
from skyrl_gym.envs.nemotron_ultra.indirect_prompt_injection import create_session, execute_tool_calls, grade_session
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit import grade_judge_profile_verifyit
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention, grade_multichallenge
from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import grade_math_verifyit
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.nvarc import extract_python, valid_grid, grade_transductive_arc
from skyrl_gym.envs.nemotron_ultra.lean_feedback import build_correction_prompt, format_error_feedback
from skyrl_gym.envs.nemotron_ultra.lean_proof import build_lean4_proof, determine_proof_status
from skyrl_gym.envs.nemotron_ultra.rdkit_chemistry import grade_rdkit_chemistry
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.structured_outputs_verifyit import grade_structured_output_verifyit
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.envs.nemotron_ultra.tool_comparison_verifyit import grade_expected_action_verifyit
from skyrl_gym.envs.reasoning_gym.scoring import extract_answer
from skyrl_gym.envs.instruction_verifyit import grade_nemotron_instructions
from skyrl_gym.envs.verifyit_clients import grade_reasoning_entry
from skyrl_gym.python_execution import PythonKernel
from skyrl_gym.lean_execution import compile_lean
from skyrl_gym.task_records import fold_grades, grade_result, skipped_grade
from skyrl_gym.task_sessions import BlockingOperations
from skyrl_gym.verification import VERIFIER_RUNTIME_ERROR, VerificationResult

NS_TOOLS_AGENT = "ns_tools_simple_agent"
IPI_AGENT = "indirect_prompt_injection_simple_agent"
LEAN_AGENT = "math_formal_lean_refinement_agent"
PYTHON_TOOL_TURN_LIMIT = 50
LEAN_TURN_LIMIT = 3
TOOL_COMPARISON_AGENTS = {
    "single_step_tool_use_with_argument_comparison_agent",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
}
FORMAT_AGENTS = {"citation_format_simple_agent", "freeform_formatting_simple_agent"}
STRUCTURED_OUTPUT_AGENTS = {"structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"}
JAILBREAK_AGENTS = {
    "jailbreak_engagement_with_disclaimer",
    "jailbreak_hard_refusal_no_redirection",
    "jailbreak_hard_refusal_with_helplines",
    "jailbreak_refusal_with_explanation",
}


class GradingMode(StrEnum):
    VERIFY = "verify"
    SKIP = "skip"


@dataclass(frozen=True)
class LeanAttempt:
    reward: float
    diagnostics: dict[str, Any]
    correction: str | None


class NemotronTaskSession:
    """Keep row-specific task state and execute tools through the supplied Shellbox machine."""

    def __init__(self, lowered: LoweredTaskSpec, machine: Machine | None, *, executor: Executor | None = None):
        task = lowered.task
        parameters = session_parameters(task)
        self.task = task
        self.machine = machine
        self.config = parameters.config
        self.judge_timeout = self.config.get("verifyit_judge_total_timeout_seconds", 120.0)
        self.blocking = BlockingOperations(executor)
        self.verifyit_enabled = bool(self.config.get("verifyit_enabled", False))
        self.grading = GradingMode(self.config.get("grading", GradingMode.VERIFY))
        ultra = parameters.extras["extra_info"]["nemotron_ultra"]
        if ultra["route"] != "task_session":
            raise ValueError("Terminal source rows require a portable Harbor task")
        self.agent = ultra["agent"]
        if self.agent not in NEMOTRON_ULTRA_MOPD_AGENTS:
            raise ValueError(f"Unsupported Nemotron source agent: {self.agent!r}")
        self.record = json.loads(ultra["record_json"])
        self.request = json.loads(ultra["request_json"])
        self.ipi_session = create_session(self.record) if self.agent == IPI_AGENT else None
        source_turn_limit = (
            PYTHON_TOOL_TURN_LIMIT
            if self.agent in {NS_TOOLS_AGENT, IPI_AGENT}
            else LEAN_TURN_LIMIT
            if self.agent == LEAN_AGENT
            else 1
        )
        self.max_turns = min(source_turn_limit, lowered.session.max_turns)
        self.turns = 0
        self.grades: list[GradeResult] = []
        judges = self.config.get("judges", {})
        self.general_judge = OpenAIJudge(**judges["general"]) if judges.get("general") is not None else None
        self.safety_judge = OpenAIJudge(**judges["safety"]) if judges.get("safety") is not None else None
        self.python = None
        if self.agent == NS_TOOLS_AGENT:
            assert machine is not None
            self.python = PythonKernel(machine)

    async def prepare(self) -> SessionStart:
        if self.python is not None:
            await self.python.start()
        options = {key: value for key, value in self.request.items() if key != "input"}
        start = session_start(self.task, shell_tool=None, provider_tools=())
        return SessionStart(start.messages, {**start.options, **options})

    async def advance(self, turn: ModelTurn) -> Transition:
        self.turns += 1
        action = final_answer_text(turn.text)
        message = {**turn.message, "content": action}
        diagnostics = {"agent": self.agent}
        if self.agent == NS_TOOLS_AGENT and turn.stop_reason == LENGTH_STOP_REASON and message.get("tool_calls"):
            grade = GradeResult(Outcome.UNAVAILABLE, None, "The model stopped within a tool call")
            self.grades.append(grade)
            return Transition(done=True, reward=0.0, grade=grade)
        try:
            transition = await self._advance(action, message, diagnostics, turn.stop_reason)
        except (requests.RequestException, RuntimeError, ValueError, OSError) as error:
            details = {
                "agent": self.agent,
                "error_type": VERIFIER_RUNTIME_ERROR,
                "error_category": "infrastructure",
                "cause_error_type": type(error).__name__,
                "error_message": str(error),
                "grading_action": action,
            }
            transition = Transition(
                done=True,
                reward=0.0,
                grade=GradeResult(Outcome.INFRA_ERROR, None, "Verifier failed", diagnostics=details),
                metrics=details,
            )
        assert transition.grade is not None
        if transition.reset_conversation is not None:
            self.grades.clear()
        else:
            self.grades.append(transition.grade)
        return transition

    async def _advance(self, action, message, diagnostics, stop_reason):
        if self.agent == NS_TOOLS_AGENT:
            observations = await self._python_calls(message)
            assert self.python is not None
            if self.python.failure is not None:
                diagnostics["candidate_failure"] = self.python.failure.reason.value
                return Transition(
                    done=True,
                    reward=0.0,
                    grade=GradeResult(Outcome.GRADED, 0.0, passed=False, diagnostics=diagnostics),
                    observations=tuple(observations or ()),
                    metrics=diagnostics,
                )
            if observations is not None and self.turns < self.max_turns:
                return Transition(
                    done=False,
                    observations=tuple(observations),
                    reward=0.0,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "Tool execution is not a terminal verdict"),
                    metrics={**diagnostics, "num_tool_calls": len(observations)},
                )
            diagnostics["max_steps_exhausted"] = observations is not None
        if self.agent == IPI_AGENT:
            assert self.ipi_session is not None
            observations = execute_tool_calls(self.ipi_session, message)
            if observations is not None and self.turns < self.max_turns and stop_reason != LENGTH_STOP_REASON:
                return Transition(
                    done=False,
                    observations=tuple(observations),
                    reward=0.0,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "Tool execution is not a terminal verdict"),
                    metrics={**diagnostics, "num_tool_calls": len(observations)},
                )
        if self.grading == GradingMode.SKIP and self.agent != LEAN_AGENT:
            diagnostics["graded"] = 0.0
            return Transition(
                done=True,
                reward=0.0,
                grade=skipped_grade("Grading is skipped", diagnostics=diagnostics),
                metrics=diagnostics,
            )
        if self.agent == IPI_AGENT:
            assert self.ipi_session is not None
            reward, details = await self.blocking.run(
                grade_session,
                self.ipi_session,
                thinking_incomplete=stop_reason == LENGTH_STOP_REASON,
                timeout=self.judge_timeout,
            )
            diagnostics.update(details)
        elif self.agent == LEAN_AGENT:
            attempt = await self._lean_attempt(action)
            reward = attempt.reward
            diagnostics.update(attempt.diagnostics)
            if attempt.correction is not None and self.turns < self.max_turns:
                return Transition(
                    done=False,
                    reward=0.0,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "A correction will replace the failed Lean attempt"),
                    metrics=diagnostics,
                    reset_conversation=({"role": "user", "content": attempt.correction},),
                )
        elif self.agent == "code_gen_simple_agent":
            reward, details = await self._code_grade(action, message)
            diagnostics.update(details)
        elif self.agent == "nvarc_inductive_simple_agent":
            reward, details = await self._arc_grade(action)
            diagnostics.update(details)
        else:
            reward, details = await self.blocking.run(self._answer_grade, action, message)
            diagnostics.update(details)
        if diagnostics.get("error_type") in {"schema_error", "verification_error"} or any(
            diagnostics.get("instruction_errors", [])
        ):
            return Transition(
                done=True,
                reward=0.0,
                grade=GradeResult(Outcome.INFRA_ERROR, None, "Verifier configuration failed", diagnostics=diagnostics),
                metrics=diagnostics,
            )
        diagnostics["graded"] = 1.0
        verification = VerificationResult.verified(
            reward, passed=reward >= 1.0, diagnostics={**diagnostics, "grading_action": action}
        )
        return Transition(done=True, reward=reward, grade=grade_result(verification), metrics=diagnostics)

    async def _python_calls(self, message):
        calls = message.get("tool_calls") or []
        if not calls:
            return None
        assert self.python is not None
        observations = []
        for call in calls:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments", ""))
            except (json.JSONDecodeError, TypeError) as error:
                output = json.dumps({"error": f"Invalid tool call arguments: {error!r}"})
            else:
                if function.get("name") != "stateful_python_code_exec":
                    output = json.dumps({"error": f"Unknown tool: {function.get('name')}"})
                elif not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
                    output = json.dumps({"error": "stateful_python_code_exec requires a string code argument"})
                else:
                    result = await self.python.execute(arguments["code"], timeout=10)
                    output = (result.stdout + result.stderr).decode().removesuffix("\n")
            observations.append({"role": "tool", "tool_call_id": str(call.get("id")), "content": output})
        return observations

    def _answer_grade(self, action, message):
        if self.agent in GENRM_AGENTS:
            return float(self.config.get("genrm", {}).get("default_score", 3.0)), {"cohort_reward_pending": True}
        if self.agent in {NS_TOOLS_AGENT, "math_with_judge_simple_agent"}:
            if self.verifyit_enabled:
                return grade_math_verifyit(
                    action,
                    self.record,
                    judge=self.general_judge,
                    timeout_seconds=self.config.get("verifyit_math_total_timeout_seconds", 60.0),
                )
            return grade_math(action, self.record, judge=self.general_judge)
        if self.agent in TOOL_COMPARISON_AGENTS:
            comparator = grade_expected_action_verifyit if self.verifyit_enabled else grade_expected_action
            reward, category = comparator(self.record["expected_action"], message)
            return reward, {"category": category.value}
        if self.agent == "calendar_simple_agent":
            scorer = grade_calendar_verifyit if self.verifyit_enabled else grade_calendar
            reward, reason = scorer(action, self.record["exp_cal_state"])
            return reward, {"reason": reason}
        if self.agent in FORMAT_AGENTS:
            scorer = grade_format_verifyit if self.verifyit_enabled else grade_format
            return scorer(action, self.record["verifier"])
        if self.agent == "mcqa_simple_agent":
            return grade_mcqa(action, self.record, verifyit_enabled=self.verifyit_enabled)
        if self.agent in STRUCTURED_OUTPUT_AGENTS:
            scorer = grade_structured_output_verifyit if self.verifyit_enabled else grade_structured_output
            return scorer(action, self.record, message)
        if self.agent == "rdkit_chemistry_agent":
            return grade_rdkit_chemistry(action, self.record)
        if self.agent == "nvarc_transductive_simple_agent":
            return grade_transductive_arc(action, self.record)
        if self.agent == "instruction_following_simple_agent":
            scorer = grade_nemotron_instructions if self.verifyit_enabled else grade_instruction_following
            return scorer(action, self.record)
        if self.agent in {"abstention_simple_agent", "multichallenge_simple_agent", *JAILBREAK_AGENTS}:
            kind = (
                "abstention"
                if self.agent == "abstention_simple_agent"
                else "multichallenge"
                if self.agent == "multichallenge_simple_agent"
                else "jailbreak"
            )
            judge = self.safety_judge if kind == "jailbreak" else self.general_judge
            if judge is None:
                raise RuntimeError(f"Nemotron verifier {self.agent!r} requires a judge")
            if self.verifyit_enabled:
                return grade_judge_profile_verifyit(
                    action,
                    self.record,
                    judge,
                    kind=kind,
                    timeout_seconds=self.judge_timeout,
                )
            scorers = {
                "abstention": grade_abstention,
                "multichallenge": grade_multichallenge,
                "jailbreak": grade_jailbreak,
            }
            return scorers[kind](action, self.record, judge)
        if self.agent == "reasoning_gym_simple_agent":
            task_name = self.record["metadata"]["source_dataset"]
            entry = {
                "question": self.record["question"],
                "answer": self.record.get("answer"),
                "metadata": self.record["metadata"],
            }
            matches = list(re.finditer(r"<answer>(.*?)</answer>", action, re.DOTALL))
            answer = matches[-1].group(1).strip() if matches else last_boxed_answer(action) or extract_answer(action)
            reward = (
                grade_reasoning_entry(task_name, entry, answer)
                if self.verifyit_enabled
                else float(reasoning_gym.get_score_answer_fn(task_name)(answer=answer, entry=entry))
            )
            return reward, {"task_name": task_name, "extracted_answer": answer}
        raise NotImplementedError(f"Nemotron verifier {self.agent!r} has no task implementation")

    async def _code_grade(self, action, message):
        assert self.machine is not None
        code = extract_code_from_model(action)
        tests = json.loads(normalize_lcb_ground_truth(self.record["verifier_metadata"]["unit_tests"]))
        config = self.config.get("code_verifier", {})
        reward, execution = await execute_code(
            self.machine,
            tests,
            code or "",
            timeout=config.get("per_test_timeout_seconds", DEFAULT_PER_TEST_TIMEOUT_SECONDS),
            limits=VerifierLimits(
                max_memory_bytes=config.get("max_memory_bytes", DEFAULT_LIMITS.max_memory_bytes),
                total_timeout_seconds=config.get("total_timeout_seconds", DEFAULT_LIMITS.total_timeout_seconds),
            ),
        )
        violation = has_reasoning_format_violation(action, message)
        results = execution["test_results"]
        return 0.0 if violation else reward, {
            "extracted_model_code": code,
            "test_results": results,
            "executed_tests": len(results),
            "total_tests": len(tests),
            "execution_output": execution,
            "result": "pass" if reward == 1.0 else "failed_tests",
            "reasoning_format_violation_rate": float(violation),
            "difficulty": self.record.get("verifier_metadata", {}).get("difficulty"),
        }

    async def _lean_attempt(self, action: str) -> LeanAttempt:
        if not action.strip():
            error = "Empty generation received. Please provide a valid Lean 4 proof."
            return LeanAttempt(
                0.0,
                {
                    "proof_status": "empty_generation",
                    "predicted_proof": "",
                    "error_feedback": error,
                },
                build_correction_prompt(proof_attempt="(empty)", error_message=error),
            )
        assert self.machine is not None
        proof = build_lean4_proof(action, self.record)
        output = await compile_lean(
            self.machine,
            proof,
            project=self.task.environment_requirements.working_directory,
            timeout=float(self.config.get("lean_timeout", 30.0)),
        )
        status = determine_proof_status(output)
        if status == "output_truncated":
            raise RuntimeError(f"Lean verification unavailable: {output}")
        details = {
            "proof_status": status,
            "predicted_proof": proof,
            "compiler_output": {
                **asdict(output),
                "stdout": output.stdout.decode(errors="replace"),
                "stderr": output.stderr.decode(errors="replace"),
            },
        }
        if status == "completed":
            return LeanAttempt(1.0, details, None)
        feedback = format_error_feedback(output, proof)
        details["error_feedback"] = feedback
        return LeanAttempt(0.0, details, build_correction_prompt(proof_attempt=action, error_message=feedback))

    async def _arc_grade(self, action):
        assert self.machine is not None
        code = extract_python(action)
        predicted = None
        execution = None
        if code is not None:
            kernel = PythonKernel(self.machine)
            try:
                await kernel.start()
                program = (
                    "import json\n"
                    + code
                    + "\n"
                    + f"_arc_result = transform({self.record['test_input']!r})\n"
                    + "print(json.dumps(_arc_result.tolist() if isinstance(_arc_result, __import__('numpy').ndarray) else _arc_result))"
                )
                output = await kernel.execute(program, timeout=30)
                execution = {
                    "stdout": output.stdout.decode(),
                    "stderr": output.stderr.decode(),
                    "exit_code": output.exit_code,
                }
                if output.reason == ExitReason.EXITED and output.exit_code == 0 and not output.stdout_truncated:
                    try:
                        value = json.loads(output.stdout.decode().strip().rsplit("\n", 1)[-1])
                    except (ValueError, IndexError):
                        value = None
                    predicted = value if valid_grid(value) else None
            finally:
                await kernel.close()
        correct = grade_grid_candidate(self.record["expected_output"], predicted).reward == 1.0
        return float(correct), {
            "agent_mode": "inductive",
            "extraction_successful": predicted is not None,
            "exact_match": correct,
            "predicted_output": predicted,
            "execution_output": execution,
        }

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return fold_grades(self.grades)

    async def close(self) -> None:
        try:
            await self.blocking.close()
        finally:
            if self.python is not None:
                await self.python.close()
