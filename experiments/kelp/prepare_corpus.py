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

"""Prepare a large Python function corpus for Kelp tree diffusion training.

Combines multiple sources to maximize diversity:
1. Local Python repos: extract functions from source trees
2. MBPP: curated coding problems from Google Research (~374 programs)
3. HumanEval: OpenAI's coding benchmark (~164 programs)
4. codeparrot/github-code: streaming Python from GitHub (~5,000+ functions)

Eval task decontamination: programs matching EVAL_TASKS in evaluate.py
are automatically excluded from the training corpus to prevent train/test leakage.

Output format: one program per block, separated by '# ---' sentinel lines,
compatible with train.py --corpus-file flag.

Usage:
    uv run python experiments/kelp/prepare_corpus.py --output experiments/kelp/corpus.txt
    uv run python experiments/kelp/prepare_corpus.py --output corpus.txt --source-dirs /path/to/repos
    uv run python experiments/kelp/prepare_corpus.py --output corpus.txt --no-github  # offline mode
"""

import argparse
import ast
import logging
import random
import sys
import textwrap
from pathlib import Path

from experiments.kelp.corpus import CORPUS_SEPARATOR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules", "checkpoints", ".eggs", "build", "dist"}

# Function signatures of EVAL_TASKS in evaluate.py. Any training program that
# contains one of these as a substring is excluded to prevent train/test leakage.
# This catches both direct matches (the function itself) and indirect leakage
# (test fixtures / helpers that embed eval task code as string literals).
EVAL_SIGNATURES = [
    "def add(a, b):",
    "def sub(a, b):",
    "def mul(a, b):",
    "def neg(x):",
    "def abs_val(x):",
    "def max_val(a, b):",
    "def min_val(a, b):",
    "def clamp(x, lo, hi):",
    "def double(x):",
    "def square(x):",
]


def extract_functions_from_file(source: str, max_length: int) -> list[str]:
    """Extract individual function definitions from a Python source file.

    Returns dedented function source strings that are parseable and under max_length.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        func_src = ast.get_source_segment(source, node)
        if func_src is None:
            continue
        # Dedent in case the function is nested inside a class.
        func_src = textwrap.dedent(func_src)
        if len(func_src) > max_length:
            continue
        if len(func_src) < 20:
            continue
        # Verify it parses standalone.
        try:
            ast.parse(func_src)
        except SyntaxError:
            continue
        functions.append(func_src)
    return functions


def extract_local_functions(source_dir: Path, max_length: int) -> list[str]:
    """Extract Python functions from a local directory tree."""
    logger.info(f"Extracting functions from {source_dir}...")
    functions = []
    file_count = 0

    for py_file in source_dir.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in py_file.parts):
            continue
        file_count += 1
        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        functions.extend(extract_functions_from_file(source, max_length))

    logger.info(f"  Scanned {file_count} files, extracted {len(functions)} functions")
    return functions


def load_mbpp_programs() -> list[str]:
    """Load programs from Google's MBPP benchmark."""
    try:
        from datasets import load_dataset

        programs = []
        for split in ["train", "validation", "test", "prompt"]:
            try:
                ds = load_dataset("google-research-datasets/mbpp", "full", split=split, trust_remote_code=True)
                for item in ds:
                    code = item.get("code", "")
                    if code:
                        programs.append(code)
            except Exception:
                continue
        logger.info(f"  MBPP: loaded {len(programs)} programs")
        return programs
    except ImportError:
        logger.warning("  MBPP: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  MBPP: failed to load: {e}")
        return []


def load_humaneval_programs() -> list[str]:
    """Load programs from OpenAI's HumanEval benchmark."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        programs = []
        for item in ds:
            prompt = item.get("prompt", "")
            solution = item.get("canonical_solution", "")
            full = prompt + solution
            if full.strip():
                programs.append(full)
        logger.info(f"  HumanEval: loaded {len(programs)} programs")
        return programs
    except ImportError:
        logger.warning("  HumanEval: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  HumanEval: failed to load: {e}")
        return []


def stream_github_code(max_functions: int, max_length: int) -> list[str]:
    """Stream Python functions from codeparrot/github-code.

    Iterates over Python files from GitHub, extracting individual functions
    until we reach the target count.
    """
    try:
        from datasets import load_dataset

        logger.info(f"  Streaming github-code (target: {max_functions} functions)...")
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
            trust_remote_code=True,
        )

        functions = []
        files_scanned = 0
        for item in ds:
            files_scanned += 1
            code = item.get("code", "")
            funcs = extract_functions_from_file(code, max_length)
            functions.extend(funcs)
            if files_scanned % 1000 == 0:
                logger.info(f"    Scanned {files_scanned} files, {len(functions)} functions so far...")
            if len(functions) >= max_functions:
                functions = functions[:max_functions]
                break

        logger.info(f"  github-code: extracted {len(functions)} functions from {files_scanned} files")
        return functions
    except ImportError:
        logger.warning("  github-code: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  github-code: failed to stream: {e}")
        return []


def deduplicate_and_filter(programs: list[str], max_length: int) -> list[str]:
    """Remove duplicates, blocklisted eval programs, and filter.

    Keeps only programs that:
    - Parse as valid Python
    - Are under max_length characters
    - Are not trivially short (<20 chars)
    - Are unique (exact match dedup)
    - Do NOT match any EVAL_BLOCKLIST entry (train/test decontamination)
    """
    seen = set()
    filtered = []
    blocked = 0

    for prog in programs:
        prog = prog.rstrip()
        if not prog.endswith("\n"):
            prog = prog + "\n"

        if len(prog) > max_length or len(prog) < 20:
            continue

        # Normalize whitespace for dedup but keep original.
        key = prog.strip()
        if key in seen:
            continue
        seen.add(key)

        # Decontamination: exclude programs containing eval task signatures.
        # Catches both direct matches and indirect leakage (e.g., test fixtures
        # that embed eval programs as string literals).
        if any(sig in prog for sig in EVAL_SIGNATURES):
            blocked += 1
            continue

        try:
            ast.parse(prog)
        except SyntaxError:
            continue

        filtered.append(prog)

    if blocked:
        logger.info(f"  Decontamination: blocked {blocked} programs matching eval tasks")

    return filtered


def write_corpus(programs: list[str], output_path: Path) -> None:
    """Write programs to a corpus file, separated by '# ---' sentinel lines.

    This separator allows programs to contain internal blank lines,
    unlike the previous blank-line-separated format.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, prog in enumerate(programs):
            if i > 0:
                f.write(CORPUS_SEPARATOR + "\n")
            f.write(prog.rstrip("\n"))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Python function corpus for Kelp training")
    parser.add_argument("--output", type=str, required=True, help="Output corpus file path")
    parser.add_argument(
        "--source-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Directories to scan for Python files (default: auto-detect marin repo root)",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum function length in characters")
    parser.add_argument("--max-github", type=int, default=5000, help="Max functions to stream from github-code")
    parser.add_argument("--no-github", action="store_true", help="Skip github-code streaming (offline mode)")
    parser.add_argument("--no-hf", action="store_true", help="Skip all HuggingFace datasets (fully offline)")
    parser.add_argument(
        "--holdout-mbpp",
        action="store_true",
        default=True,
        help="Exclude all MBPP programs from training corpus (default: True, since evaluate_mbpp.py uses all splits)",
    )
    parser.add_argument(
        "--include-mbpp",
        dest="holdout_mbpp",
        action="store_false",
        help="Include MBPP programs in training (WARNING: contaminates evaluate_mbpp.py eval)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Resolve source directories.
    if args.source_dirs:
        source_dirs = [Path(d) for d in args.source_dirs]
    else:
        source_dirs = [Path(__file__).resolve().parents[2]]
    logger.info(f"Source directories: {[str(d) for d in source_dirs]}")

    all_programs: list[str] = []

    # Source 1: Local Python functions from source directories.
    for source_dir in source_dirs:
        local_funcs = extract_local_functions(source_dir, args.max_length)
        all_programs.extend(local_funcs)

    if not args.no_hf:
        # Source 2: MBPP (skip if held out for eval).
        if not args.holdout_mbpp:
            mbpp = load_mbpp_programs()
            all_programs.extend(mbpp)
        else:
            logger.info("  MBPP: held out for evaluation (--holdout-mbpp)")

        # Source 3: HumanEval.
        humaneval = load_humaneval_programs()
        all_programs.extend(humaneval)

        # Source 4: codeparrot/github-code.
        if not args.no_github:
            github = stream_github_code(args.max_github, args.max_length)
            all_programs.extend(github)

    # Deduplicate and filter.
    logger.info(f"Total raw programs: {len(all_programs)}")
    filtered = deduplicate_and_filter(all_programs, args.max_length)
    logger.info(f"After dedup/filter: {len(filtered)} programs")

    # Shuffle for training diversity.
    rng.shuffle(filtered)

    # Write output.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_corpus(filtered, output_path)

    # Report stats.
    total_chars = sum(len(p) for p in filtered)
    avg_len = total_chars / max(len(filtered), 1)
    logger.info(f"Wrote {len(filtered)} programs to {output_path}")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Average length: {avg_len:.0f} chars")
    logger.info(f"File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
