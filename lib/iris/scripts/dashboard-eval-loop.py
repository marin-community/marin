#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#   "anthropic>=0.40.0",
#   "claude-agent-sdk>=0.1.0",
# ]
# ///
"""
Dashboard evaluation and improvement loop using Claude Agent SDK.

This script runs an iterative evaluation loop:
1. Analysis agent evaluates dashboard screenshots and produces a report
2. If status is "NOT_OK", fixer agent implements improvements
3. Loop continues until status is "OK" or max iterations reached

Usage:
    uvx dashboard-eval-loop.py --screenshot-dir logs/dashboard-eval-20260128
"""

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from claude_agent_sdk import query, ClaudeAgentOptions, TextBlock


@dataclass
class EvalResult:
    """Result from the analysis agent."""

    status: Literal["OK", "NOT_OK"]
    summary: str
    report_path: Path


def generate_screenshots(output_dir: Path) -> None:
    """Generate dashboard screenshots using screenshot-dashboard.py."""
    print(f"\n{'=' * 80}")
    print("Generating dashboard screenshots...")
    print(f"{'=' * 80}")

    script_path = Path(__file__).parent / "screenshot-dashboard.py"
    cmd = ["uv", "run", str(script_path), "--output-dir", str(output_dir)]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Warning: Screenshot generation failed with code {result.returncode}")

    print(f"{'=' * 80}")


async def run_analysis_agent(
    screenshot_dir: Path,
    analysis_instructions: str,
    iteration: int,
) -> EvalResult:
    """Run the analysis agent to evaluate dashboard screenshots.

    Returns:
        EvalResult with status ("OK" or "NOT_OK"), summary, and report path
    """
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration}: ANALYSIS AGENT")
    print(f"{'=' * 80}")

    # Get all screenshots
    screenshots = sorted(screenshot_dir.glob("*.png"))
    if not screenshots:
        raise ValueError(f"No screenshots found in {screenshot_dir}")

    print(f"Found {len(screenshots)} screenshots: {[s.name for s in screenshots]}")

    # Build the prompt - the agent can read files itself
    screenshot_list = "\n".join(f"- {s}" for s in screenshots)
    prompt = f"""You are evaluating the Iris dashboard against debugging requirements.

Screenshot directory: {screenshot_dir.absolute()}
Screenshots to analyze:
{screenshot_list}

Your task:
1. Read and analyze each screenshot thoroughly using the Read tool
2. Produce a comprehensive evaluation report following the instructions below
3. At the END of your response, provide a status line:
   STATUS: OK (if all criteria met)
   STATUS: NOT_OK (if critical gaps exist)

Instructions:
{analysis_instructions}
"""

    # Add system prompt to the user prompt
    full_prompt = (
        "You are a senior software engineer evaluating a dashboard UI. "
        "Be thorough, critical, and specific in your analysis.\n\n"
        f"{prompt}"
    )

    # Run the query with full permissions
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        cwd=str(screenshot_dir.parent),
        permission_mode="bypassPermissions",
    )

    # Collect all messages and stream output
    response_text = ""
    async for message in query(prompt=full_prompt, options=options):
        # Extract text from message content blocks
        if hasattr(message, "content"):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                    response_text += block.text + "\n"

    # Save the full report
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = screenshot_dir / f"eval-report-{timestamp}.md"
    report_path.write_text(response_text)
    print(f"\nSaved report to: {report_path}")

    # Extract status
    status = "NOT_OK"  # default
    summary_lines = []
    for line in response_text.split("\n"):
        if line.strip().startswith("STATUS:"):
            status_text = line.split("STATUS:", 1)[1].strip()
            if "OK" in status_text.upper() and "NOT" not in status_text.upper():
                status = "OK"
            else:
                status = "NOT_OK"
        # Collect summary from "Overall Assessment" or "Final Assessment" section
        if "overall assessment" in line.lower() or "grade:" in line.lower():
            summary_lines.append(line)

    summary = "\n".join(summary_lines[:5]) if summary_lines else response_text[:300]

    print(f"\n{'=' * 80}")
    print("ANALYSIS SUMMARY:")
    print(f"Status: {status}")
    print(f"Summary:\n{summary}")
    print(f"{'=' * 80}")

    return EvalResult(status=status, summary=summary, report_path=report_path)


async def run_fixer_agent(
    eval_result: EvalResult,
    context_files: dict[str, Path],
    iteration: int,
) -> str:
    """Run the fixer agent to implement improvements.

    Args:
        eval_result: Result from analysis agent
        context_files: Dict of context file descriptions to paths
        iteration: Current iteration number

    Returns:
        Summary of changes made
    """
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration}: FIXER AGENT")
    print(f"{'=' * 80}")

    # Build context - reference files for the agent to read
    context_parts = [
        "You are implementing improvements to the Iris dashboard based on an evaluation report.",
        "",
        f"EVALUATION REPORT: {eval_result.report_path.absolute()}",
        "Read this file to understand what needs to be fixed.",
        "",
        "CONTEXT FILES (read these for guidelines and context):",
    ]

    for description, path in context_files.items():
        if path.exists():
            context_parts.append(f"- {description}: {path.absolute()}")

    context_parts.extend(
        [
            "",
            "Your task:",
            "1. Read the evaluation report to understand the issues",
            "2. Read the context files for coding guidelines and architecture",
            "3. Implement the highest priority fixes from the evaluation report",
            "4. Focus on P0 and P1 issues first",
            "5. Make targeted changes - don't refactor unnecessarily",
            "6. Update or create tests if needed",
            "7. Follow the project's coding guidelines from AGENTS.md",
            "",
            "At the END of your response, provide:",
            "CHANGES SUMMARY: <brief description of what you fixed>",
        ]
    )

    # Add system prompt to context
    full_prompt = (
        "You are a senior software engineer implementing dashboard improvements. "
        "Follow the project's coding guidelines and make focused, high-quality changes. "
        "Use the Read tool to read the evaluation report and context files.\n\n"
    ) + "\n".join(context_parts)

    # Run the query with full permissions
    base_dir = Path(__file__).parent.parent
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        cwd=str(base_dir),
        permission_mode="bypassPermissions",
    )

    # Collect all messages and stream output
    response_text = ""
    async for message in query(prompt=full_prompt, options=options):
        # Extract text from message content blocks
        if hasattr(message, "content"):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                    response_text += block.text + "\n"

    # Extract changes summary
    summary = "Changes made (see full output for details)"
    for line in response_text.split("\n"):
        if "CHANGES SUMMARY:" in line:
            summary = line.split("CHANGES SUMMARY:", 1)[1].strip()
            break

    print(f"\n{'=' * 80}")
    print("FIXER SUMMARY:")
    print(f"{summary}")
    print(f"{'=' * 80}")

    return summary


async def main_async():
    """Run the evaluation loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard evaluation and improvement loop")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        required=True,
        help="Directory containing dashboard screenshots",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of evaluation/fix iterations",
    )
    parser.add_argument(
        "--analysis-instructions",
        type=Path,
        default=Path("docs/dashboard-analyze.md"),
        help="Path to analysis agent instructions",
    )
    args = parser.parse_args()

    # Create screenshot directory if it doesn't exist
    args.screenshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Screenshot directory: {args.screenshot_dir}")

    # Generate initial screenshots if none exist
    if not any(args.screenshot_dir.glob("*.png")):
        print(f"No screenshots found in {args.screenshot_dir}")
        generate_screenshots(args.screenshot_dir)

    if not args.analysis_instructions.exists():
        print(f"Error: Analysis instructions not found: {args.analysis_instructions}")
        sys.exit(1)

    # Load analysis instructions
    analysis_instructions = args.analysis_instructions.read_text()

    # Context files for the fixer agent
    base_dir = Path(__file__).parent.parent
    context_files = {
        "Agent Guidelines": base_dir / "AGENTS.md",
        "Iris README": base_dir / "README.md",
        "Analysis Instructions": args.analysis_instructions,
        "Known Issues": base_dir / "docs" / "dashboard-be-better.md",
    }

    # Main loop
    print(f"\n{'=' * 80}")
    print("DASHBOARD EVALUATION LOOP")
    print(f"Screenshot directory: {args.screenshot_dir}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'=' * 80}")

    for iteration in range(1, args.max_iterations + 1):
        # Step 1: Run analysis agent
        eval_result = await run_analysis_agent(
            screenshot_dir=args.screenshot_dir,
            analysis_instructions=analysis_instructions,
            iteration=iteration,
        )

        # Check status
        if eval_result.status == "OK":
            print(f"\n{'=' * 80}")
            print("✅ EVALUATION PASSED!")
            print(f"Dashboard meets all criteria after {iteration} iteration(s)")
            print(f"{'=' * 80}")
            break

        # Step 2: Run fixer agent
        if iteration < args.max_iterations:
            await run_fixer_agent(
                eval_result=eval_result,
                context_files=context_files,
                iteration=iteration,
            )

            # After fixes, regenerate screenshots
            generate_screenshots(args.screenshot_dir)
        else:
            print(f"\n{'=' * 80}")
            print(f"❌ EVALUATION FAILED after {args.max_iterations} iterations")
            print(f"Last status: {eval_result.status}")
            print(f"See report: {eval_result.report_path}")
            print(f"{'=' * 80}")

    print("\nEvaluation loop complete.")


def main():
    """Entry point that runs the async main loop."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
