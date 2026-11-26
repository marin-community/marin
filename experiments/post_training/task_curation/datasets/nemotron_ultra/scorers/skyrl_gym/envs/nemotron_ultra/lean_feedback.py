# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lean compiler diagnostics and correction prompt from NVIDIA NeMo Gym."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)


def parse_error(log_string: str) -> List[Dict[str, Any]]:
    """Parse Lean4 compiler error messages from log output.

    Args:
        log_string: The compiler output containing error messages

    Returns:
        List of error dictionaries with position and error data
    """
    error_pattern = re.compile(
        r"(/lean4/my_project/.*?:\d+:\d+: error:.*?)(?=\n/lean4/my_project|\Z)",
        re.DOTALL,
    )
    errors = error_pattern.findall(log_string)
    pattern = re.compile(r":(\d+):(\d+):")
    error_list = []
    for error in errors:
        match = pattern.search(error)
        if match:
            error_list.append(
                {
                    "pos": {"line": int(match.group(1)), "column": int(match.group(2))},
                    "endPos": None,
                    "data": error.split("error:")[1].strip() if "error:" in error else error,
                }
            )

    return error_list


def get_error_str(code: str, errors: List[Dict[str, Any]], error_thres: int = 8) -> str:
    """Format compiler errors with code context for display.

    Args:
        code: The Lean code that was compiled
        errors: List of parsed error dictionaries
        error_thres: Maximum number of errors to include (default 8)

    Returns:
        Formatted error string with code context
    """
    if not errors:
        return ""

    err_str = ""
    code_lines = code.split("\n")

    for i, error in enumerate(errors[:error_thres]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]
        if start_line >= len(code_lines) or start_line < 0:
            LOG.warning(
                "Error line %d out of bounds (code has %d lines). Error: %s",
                start_line,
                len(code_lines),
                error,
            )
            continue

        if error["endPos"] is None:
            end_line = start_line
            end_col = len(code_lines[start_line]) if start_line < len(code_lines) else 0
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        # Show context lines before error
        error_code = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                error_code += f"{code_lines[start_line + ii]}\n"

        # Show error line(s) with <error> markers
        if start_line < len(code_lines):
            if start_line != end_line:
                error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    if j < len(code_lines):
                        error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    error_code += "... --[Truncated]-- ...\n"
                if end_line < len(code_lines):
                    error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
            else:
                error_code += (
                    code_lines[start_line][:start_col]
                    + "<error>"
                    + code_lines[start_line][start_col:end_col]
                    + "</error>"
                    + code_lines[start_line][end_col:]
                    + "\n"
                )

        # Show one line after error
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"

    if len(errors) > error_thres:
        err_str += f"\n... [Omitted {len(errors) - error_thres} more errors] ...\n"

    return err_str


def format_error_feedback(compiler_output: Dict[str, Any], predicted_proof: str) -> str:
    """Format compiler errors into feedback for self-correction.

    Args:
        compiler_output: The compiler output dictionary
        predicted_proof: The proof code that was compiled

    Returns:
        Formatted error message string
    """
    process_status = compiler_output.get("process_status", "unknown")
    stdout = compiler_output.get("stdout", "")
    stderr = compiler_output.get("stderr", "")

    if process_status == "timeout":
        return "The compilation timed out. Please simplify your proof or use more efficient tactics."

    # Parse structured errors from stderr
    errors = parse_error(stderr)
    if errors:
        return get_error_str(predicted_proof, errors)

    # Fallback to raw output if no structured errors found
    combined = stdout + "\n" + stderr
    if combined.strip():
        # Truncate if too long
        if len(combined) > 2000:
            combined = combined[:2000] + "\n... [Truncated]"
        return f"Compilation output:\n{combined}"

    return "The proof failed but no specific error message was captured."


# Nemotron single-turn refinement prompt template
REFINEMENT_PROMPT_TEMPLATE = """Here is a proof attempt for the following theorem in Lean4.

{proof_attempt}

The proof is not correct. Following is the compilation error message:

{error_message}

Your task is to fix this proof. Before producing the Lean 4 code to formally prove the given theorem, do a detailed analysis of the error message. Your final answer must be a single, complete Lean 4 markdown code block containing the completed theorem. Do NOT include any text or explanation before or after the code block. Begin with ```lean4 and end with ```."""


def build_correction_prompt(
    proof_attempt: str,
    error_message: str,
    refinement_template: Optional[str] = None,
) -> str:
    """Build a Nemotron-style single-turn correction prompt.

    Args:
        proof_attempt: The previous proof attempt that failed
        error_message: The formatted error feedback
        refinement_template: Optional custom template (uses default if None)

    Returns:
        The formatted correction prompt
    """
    template = refinement_template or REFINEMENT_PROMPT_TEMPLATE
    return template.format(
        proof_attempt=proof_attempt,
        error_message=error_message,
    )
