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

"""Lean proof assembly and compiler verdicts."""

import re
from textwrap import indent
from typing import Any

from shellbox.machine import ExitReason, Result


def _proof_body(code: str) -> str:
    lines = code.strip().splitlines()
    header = next((index for index, line in enumerate(lines) if re.match(r"\s*(theorem|lemma|example)\b", line)), None)
    if header is None:
        return code.strip()
    for index in range(header, len(lines)):
        if ":=" in lines[index]:
            return "\n".join([lines[index].split(":=", 1)[1].strip(), *lines[index + 1 :]]).strip()
    return code.strip()


def build_lean4_proof(generation: str, record: dict[str, Any]) -> str:
    """Keep the source theorem and replace its proof with the final generated code block."""
    code = generation
    for language in ("lean4", "lean", ""):
        blocks = re.findall(rf"```{language}\s*\n?(.*?)\n?```", generation, re.DOTALL)
        if blocks:
            code = blocks[-1].strip()
            break
    proof = _proof_body(code)
    statement, assignment, source_proof = record["formal_statement"].partition(":=")
    if not assignment:
        raise ValueError("The formal statement has no proof assignment")
    if re.match(r"by\b", source_proof.lstrip()) and proof == code.strip() and not re.match(r"by\b", proof):
        proof = "by\n" + indent(proof, "  ")
    return record["header"] + statement.rstrip() + " := " + proof


def determine_proof_status(output: Result) -> str:
    """Return the compiler verdict without credit for incomplete or truncated proofs."""
    if output.reason == ExitReason.TIMED_OUT:
        return "timeout"
    if output.stdout_truncated or output.stderr_truncated:
        return "output_truncated"
    if output.exit_code != 0:
        return "failed"
    diagnostics = (output.stdout + b"\n" + output.stderr).decode(errors="replace").lower()
    if re.search(r"\bsorry\b", diagnostics):
        return "has_sorry"
    if re.search(r"\berror\b", diagnostics):
        return "failed"
    return "completed"
