# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Lean4 proof processing and evaluation.

Ported from NeMo-Skills:
https://github.com/NVIDIA-NeMo/NeMo-Skills/blob/main/nemo_skills/code_execution/proof_utils.py
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProofBuildConfig:
    """Configuration for building proofs from generations."""

    final_answer_key: Optional[str] = None
    extract_code_mode: str = "last"  # "first" or "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


def extract_code_block(text: str, languages: Optional[list] = None, extract_code_mode: str = "last") -> str:
    """Extract code from markdown code blocks.

    Args:
        text: Text containing markdown code blocks
        languages: List of language tags to try (e.g., ["lean4", "lean", ""])
        extract_code_mode: "first" or "last" - which code block to extract

    Returns:
        Extracted code or empty string if no code block found
    """
    if languages is None:
        languages = [""]
    for language in languages:
        matches = re.findall(rf"```{language}\s*\n?(.*?)\n?```", text, re.DOTALL)
        if matches:
            idx = 0 if extract_code_mode == "first" else -1
            return matches[idx].strip()
    return ""


def clean_formal_generation(
    generation: str,
    final_answer_key: Optional[str] = None,
    extract_code_mode: str = "last",
) -> str:
    """Clean LLM generation to extract Lean code.

    Args:
        generation: Raw LLM generation
        final_answer_key: Optional key to extract part after
        extract_code_mode: "first" or "last" code block

    Returns:
        Cleaned Lean code
    """
    # Extract part after final_answer_key if present and configured
    if final_answer_key and final_answer_key in generation:
        generation = generation.split(final_answer_key, 1)[1].strip()

    languages = ["lean4", "lean", ""]
    extracted_code = extract_code_block(generation, languages, extract_code_mode=extract_code_mode)
    if extracted_code:
        return extracted_code

    # If no explicit code block, remove any surrounding triple backticks
    return re.sub(r"^\s*```(?:lean4|lean)?\s*|\s*```[\s]*$", "", generation).strip()


def extract_proof_only(lean_code: str) -> str:
    """Extract only the proof part from a Lean theorem/example.

    This function removes the theorem/example header and returns just the proof body.

    Args:
        lean_code: The full Lean code including theorem statement

    Returns:
        The proof part only (everything after ':=')
    """
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r"^\s*(theorem|example)\b")
    header_start_idx = None

    # 1. Find where the theorem starts
    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    # 2. Find where ':=' occurs, starting from the header
    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    # 3. Extract the line after ':='
    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    # 4. Collect proof lines
    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    # 5. Remove leading 'by' (with or without indentation)
    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]  # Strip 'by '

    return "\n".join(proof_lines).rstrip()


def build_lean4_proof(generation: str, data_point: Dict[str, Any], config: ProofBuildConfig) -> str:
    """Build a complete Lean4 proof from generation and data point.

    Args:
        generation: The raw generation from the model
        data_point: Dictionary containing header, formal_statement, etc.
        config: Configuration for proof building

    Returns:
        Complete Lean4 proof ready for execution
    """
    # Clean the generation and extract the formal proof
    cleaned_generation = clean_formal_generation(
        generation, final_answer_key=config.final_answer_key, extract_code_mode=config.extract_code_mode
    )

    # Combine header + formal_statement + proof
    header = data_point.get("header", "")
    formal_statement = data_point.get("formal_statement", "") if config.restate_formal_statement else ""

    if config.strip_theorem_from_proof:
        proof_part = extract_proof_only(cleaned_generation)
    else:
        proof_part = cleaned_generation

    return header + formal_statement + proof_part


def determine_proof_status(compiler_output: Dict[str, Any]) -> str:
    """Determine proof status from compiler output.

    Args:
        compiler_output: Dictionary containing process_status, stdout, stderr

    Returns:
        Status string: "completed", "timeout", "has_sorry", or other process status
    """
    process_status = compiler_output.get("process_status", "unknown")

    if process_status == "timeout":
        return "timeout"
    elif process_status != "completed":
        return process_status

    if compiler_output.get("output_truncated") or compiler_output.get("truncated"):
        return "output_truncated"
    if compiler_output.get("returncode", compiler_output.get("exit_code", 0)) != 0:
        return "failed"

    # Check stdout and stderr for proof status indicators
    stdout = compiler_output.get("stdout", "").lower()
    stderr = compiler_output.get("stderr", "").lower()
    combined = stdout + "\n" + stderr

    # Check for sorry (incomplete proof)
    if re.search(r"\bsorry\b", combined) is not None:
        return "has_sorry"

    if re.search(r"\berror\b", combined):
        return "failed"
    return "completed"
