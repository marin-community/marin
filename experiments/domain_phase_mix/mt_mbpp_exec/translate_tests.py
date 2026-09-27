# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Translate MBPP's Python asserts into test code for each MT-MBPP language with DeepSeek.

For every (language, task) the model sees MBPP's task text, Python reference and asserts, and the o4-mini reference
solution in the target language with the signature the prompt discloses. It returns JSON with three fields:
``imports`` (lines placed above the solution), ``main`` (test code placed after the solution that exits non-zero on
any failed case) and ``stub`` (the reference solution with the tested function returning a fixed wrong value, used to
reject vacuous tests). Validation in the sandbox keeps a test only when the reference passes it and the stub fails.
Python needs no model: its test is MBPP's asserts with the tested name mapped to the reference's function.

Results append to a JSONL file as they arrive, so an interrupted run resumes where it stopped. Progress (done/total and
a projected finish) goes to ``--log``.

usage: (set -a; source ~/.zshrc.secrets; set +a; uv run --offline --no-sync python -m \
    experiments.domain_phase_mix.mt_mbpp_exec.translate_tests --signatures SIGNATURES.jsonl.gz \
    --sources SOURCES.json.gz --output OUT.jsonl --log LOG)
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MAX_TOKENS = 16384
REPAIR_MAX_TOKENS = 65536
FIRST_PASS_EFFORT = "high"
REPAIR_EFFORT = "max"
WORKERS = 96
ATTEMPTS = 6

CONVENTIONS = {
    "bash": "`main` is shell code run after the solution. Call the function exactly as in the usage line (quote each "
    "argument; pass list items as separate arguments where the usage line ends in `...`), capture its standard output "
    "with $(...), and compare it to the expected text exactly as the reference solution prints it; on any mismatch run "
    "`exit 1`. If the usage line ends in `< stdin`, feed the input on standard input the way the reference reads it.",
    "c": "`imports` holds the #include lines the test needs. `main` defines `int main(void)` that returns 1 on the first "  # noqa: E501
    "failed case and 0 otherwise. Follow the signature's memory conventions (output lengths, returned buffers).",
    "cpp": "`imports` holds the #include lines the test needs. `main` defines `int main()` that returns 1 on the first "
    "failed case and 0 otherwise.",
    "csharp": "`imports` holds the using directives the test needs. `main` defines `public static class TestMain` with "
    "`public static int Main()` returning 1 on the first failed case and 0 otherwise. Call the tested method through "
    "the literal placeholder `{CLASS}` for the class that declares it (`{CLASS}.Method(...)` for a static method, "
    "`new {CLASS}().Method(...)` for an instance method); the grader substitutes the class name.",
    "go": "The file is `package main` (the solution supplies it). `imports` lists the import paths the test needs, one "
    'per line, without the `import` keyword or quotes. `main` defines `func main()` that calls `panic("test failed")` '
    "on the first failed case.",
    "haskell": "`imports` holds the import lines the test needs. `main` defines `main :: IO ()` and `main = do ...` that "  # noqa: E501
    'calls `error "test failed"` on the first failed case.',
    "java": "`imports` holds the import lines the test needs. `main` defines `public class Main` with "
    "`public static void main(String[] args)` that calls `System.exit(1)` on the first failed case. Call the tested "
    "method through the literal placeholder `{CLASS}` for the class that declares it (`{CLASS}.method(...)` for a static "  # noqa: E501
    "method, `new {CLASS}().method(...)` for an instance method); the grader substitutes the class name.",
    "javascript": "`main` is top-level code that runs after the solution and calls `process.exit(1)` on the first failed "  # noqa: E501
    "case; compare arrays and objects structurally (for example with JSON.stringify).",
    "matlab": "The file runs as a GNU Octave script. `main` is script code after the solution that calls `exit(1)` on "
    "the first failed case (use isequal, or a tolerance for floating-point results).",
    "php": "`main` is PHP code (no opening or closing tag) run after the solution that calls `exit(1);` on the first "
    "failed case.",
    "r": "`main` is R code run after the solution that calls `quit(status = 1)` on the first failed case (use identical "
    "or isTRUE(all.equal(...)) as appropriate).",
    "ruby": "`main` is Ruby code run after the solution that calls `exit 1` on the first failed case.",
    "rust": "`imports` holds the `use` lines the test needs. `main` defines `fn main()` that panics (assert!/assert_eq!) "  # noqa: E501
    "on the first failed case.",
    "scala": "Scala 3. `imports` holds the import lines the test needs. `main` defines `@main def testMain(): Unit` "
    "that calls `sys.exit(1)` on the first failed case.",
    "swift": "`imports` holds the import lines the test needs (for example `import Foundation` for exit). `main` is "
    "top-level code run after the solution that calls `exit(1)` on the first failed case.",
    "typescript": "`main` is top-level code that runs after the solution and calls `process.exit(1)` on the first "
    "failed case; compare arrays and objects structurally (for example with JSON.stringify).",
}

SYSTEM = (
    "You write unit tests for code-generation benchmarks. You translate the test cases of a Python task into a test "
    "program in another language, calling the function exactly as its signature declares. You reply with one JSON "
    "object and nothing else."
)

PROMPT = """Language: {language}

Task description: {text}

Python reference solution:
```python
{python_code}
```

Python test cases (the function under test is `{mbpp_function}`):
```python
{asserts}
```

Reference solution in {language} (the function under test is declared by the signature below):
```{language}
{code}
```

Signature shown to the model under test: `{signature}`

Write test code in {language} with exactly the same test cases as the Python asserts: the same inputs and the same
expected outputs, converted to the {language} types that the signature declares. Floating-point results compare within
a relative tolerance of 1e-6. The test program is assembled as `imports`, then the solution, then `main`, so `main`
must call the function by the name and argument order of the signature and must not redefine it. It must not print
anything on success. {conventions}

Also write `stub`: a complete copy of the {language} reference solution in which the function under test keeps its
signature but returns one fixed, type-correct value that fails at least one of the test cases (for example 0, an
empty string, an empty list, or false); keep any helper definitions so that it compiles.

Reply with a JSON object with the string fields "imports", "main" and "stub"."""


def python_test(record: dict, mbpp: dict) -> dict:
    """MBPP's own asserts, with the tested name bound to the reference's function when the names differ."""
    asserts = "\n".join(mbpp["test_list"])
    if mbpp["test_setup_code"]:
        asserts = mbpp["test_setup_code"] + "\n" + asserts
    function = record.get("function") or record["mbpp_function"]
    alias = "" if function == record["mbpp_function"] else f"{record['mbpp_function']} = {function}\n"
    return {"imports": "", "main": alias + asserts, "stub": None}


REPAIR = """

Your previous answer was:
{previous}

In the sandbox, {failure}
Fix the test so that the reference solution passes it and the stub fails it. Do not change the reference solution;
change `imports`, `main` or `stub`. Keep the Python test cases' inputs and expected outputs: if the reference solution
itself returns results that disagree with the Python expected outputs, do not adapt the test to it; reply with
"reference_wrong": true and empty strings for the other fields. Otherwise reply with a corrected JSON object with the
string fields "imports", "main" and "stub"."""


def failure_text(validation: dict) -> str:
    if not validation["reference_passed"]:
        run = validation["reference"]
        return (
            f"the reference solution failed the assembled test in the {run['phase']} phase (exit code "
            f"{run['exit_code']}, timeout {run['timeout']}). Standard error ended with:\n{run['stderr_tail'][-1500:]}\n"
            f"Standard output ended with:\n{run.get('stdout_tail', '')[-500:]}"
        )
    return "the stub passed the test, so the test does not detect a wrong answer."


def request_body(
    record: dict, mbpp: dict, repair: tuple[dict, dict] | None = None, max_tokens: int = MAX_TOKENS
) -> dict:
    asserts = "\n".join(mbpp["test_list"])
    if mbpp["test_setup_code"]:
        asserts = mbpp["test_setup_code"] + "\n" + asserts
    prompt = PROMPT.format(
        language=record["language"],
        text=mbpp["text"],
        python_code=mbpp["code"].replace("\r\n", "\n").strip(),
        mbpp_function=record["mbpp_function"],
        asserts=asserts.replace("\r\n", "\n"),
        code=record["code"],
        signature=record["signature"],
        conventions=CONVENTIONS[record["language"]],
    )
    effort = FIRST_PASS_EFFORT
    if repair is not None:
        previous, validation = repair
        prompt += REPAIR.format(previous=json.dumps(previous, indent=1), failure=failure_text(validation))
        effort, max_tokens = REPAIR_EFFORT, REPAIR_MAX_TOKENS
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "reasoning_effort": effort},
        "response_format": {"type": "json_object"},
    }


def call(body: dict, key: str) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not parsed.get("reference_wrong") and not all(
                isinstance(parsed.get(k), str) for k in ("imports", "main", "stub")
            ):
                raise ValueError("missing fields")
            return {
                "response": parsed,
                "usage": result.get("usage"),
                "finish_reason": result["choices"][0]["finish_reason"],
            }
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            if attempt == ATTEMPTS - 1:
                return {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            time.sleep(min(60, 2**attempt) + random.random())
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--per-language", type=int, default=0, help="translate only this many items per language (pilot)"
    )
    parser.add_argument("--repair", type=Path, help="validation JSONL: retranslate invalid tests at max effort")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="first-pass output cap (reasoning included)")
    args = parser.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"]
    sources = json.loads(gzip.decompress(args.sources.read_bytes()))
    mbpp = {r["task_id"]: r for r in sources["mbpp"]["test"]}
    records = [json.loads(line) for line in gzip.decompress(args.signatures.read_bytes()).decode().splitlines()]
    records = [r for r in records if r["split"] == "test"]
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["language"], row["task_id"]))
    pending = [r for r in records if (r["language"], r["task_id"]) not in done]
    repairs = {}
    if args.repair:
        latest = {}
        for row in map(json.loads, args.repair.read_text().splitlines()):
            latest[(row["language"], row["task_id"])] = row
        tests = {}
        for row in map(json.loads, args.output.read_text().splitlines()):
            if "error" not in row and row.get("model"):
                tests[(row["language"], row["task_id"])] = row["response"]
        repairs = {k: (tests[k], v) for k, v in latest.items() if not v["valid"] and k in tests}
        pending = [r for r in records if (r["language"], r["task_id"]) in repairs]
    lock = threading.Lock()

    def write(row: dict) -> None:
        with lock, args.output.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    for r in [r for r in pending if r["language"] == "python" and not args.repair]:
        write({"language": "python", "task_id": r["task_id"], "model": None, **python_test(r, mbpp[r["task_id"]])})
    pending = [r for r in pending if r["language"] != "python"]
    if args.per_language:
        pending = [
            r
            for i, r in enumerate(pending)
            if sum(p["language"] == r["language"] for p in pending[:i]) < args.per_language
        ]
    start, total, finished, failed = time.time(), len(pending), 0, 0
    with args.log.open("a") as log:
        log.write(f"{time.strftime('%H:%M:%S')} start {total} translations with {MODEL}, {WORKERS} workers\n")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for r in pending:
            body = request_body(r, mbpp[r["task_id"]], repairs.get((r["language"], r["task_id"])), args.max_tokens)
            prompt_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            futures[pool.submit(call, body, key)] = (r, prompt_sha)
        for future in as_completed(futures):
            r, prompt_sha = futures[future]
            result = future.result()
            write(
                {
                    "language": r["language"],
                    "task_id": r["task_id"],
                    "model": MODEL,
                    "prompt_sha256": prompt_sha,
                    "effort": REPAIR_EFFORT if args.repair else FIRST_PASS_EFFORT,
                    "repair": bool(args.repair),
                    **result,
                }
            )
            finished += 1
            failed += "error" in result
            if finished % 50 == 0 or finished == total:
                rate = finished / (time.time() - start)
                with args.log.open("a") as log:
                    log.write(
                        f"{time.strftime('%H:%M:%S')} {finished}/{total} failed={failed} "
                        f"ETA {(total - finished) / rate / 60:.0f} min\n"
                    )


if __name__ == "__main__":
    main()
