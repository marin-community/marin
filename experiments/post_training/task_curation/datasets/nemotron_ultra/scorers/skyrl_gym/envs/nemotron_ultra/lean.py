# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lean4 verification and NVIDIA's fresh-context proof-refinement loop."""

from __future__ import annotations

from typing import Any

from skyrl_gym.envs.nemotron_ultra.lean_feedback import build_correction_prompt, format_error_feedback
from skyrl_gym.envs.nemotron_ultra.lean_proof_utils import (
    ProofBuildConfig,
    build_lean4_proof,
    determine_proof_status,
)
from skyrl_gym.envs.nemotron_ultra.sandbox import MAX_VERIFIER_OUTPUT_CHARACTERS, SandboxClient


def verify_lean_attempt(
    generation: str,
    record: dict[str, Any],
    *,
    sandbox: SandboxClient,
    timeout_seconds: float = 30.0,
    verifyit_enabled: bool = False,
) -> tuple[float, dict[str, Any], str | None]:
    if not generation.strip():
        error = "Empty generation received. Please provide a valid Lean 4 proof."
        return (
            0.0,
            {"proof_status": "empty_generation", "predicted_proof": "", "error_feedback": error},
            (build_correction_prompt(proof_attempt="(empty)", error_message=error)),
        )

    predicted_proof = build_lean4_proof(
        generation,
        {"header": record["header"], "formal_statement": record["formal_statement"]},
        ProofBuildConfig(extract_code_mode="last", restate_formal_statement=True, strip_theorem_from_proof=True),
    )
    if verifyit_enabled:
        from skyrl_gym.envs.nemotron_ultra.lean_verifyit import compile_lean_verifyit

        status, compiler_output = compile_lean_verifyit(predicted_proof, sandbox, timeout_seconds)
    else:
        compiler_output = sandbox.execute(
            predicted_proof,
            language="lean4",
            timeout_seconds=timeout_seconds,
            max_output_characters=MAX_VERIFIER_OUTPUT_CHARACTERS,
        )
        status = determine_proof_status(compiler_output)
    if status in {"error", "unknown", "output_truncated"}:
        raise RuntimeError(f"Lean verification unavailable: {compiler_output}")
    details = {
        "proof_status": status,
        "predicted_proof": predicted_proof,
        "compiler_output": compiler_output,
    }
    if status == "completed":
        return 1.0, details, None
    feedback = format_error_feedback(compiler_output, predicted_proof)
    details["error_feedback"] = feedback
    return 0.0, details, build_correction_prompt(proof_attempt=generation, error_message=feedback)
