# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade retained Table-9 math/code generations without repeating TPU inference.

Run with uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1
python -m experiments.domain_phase_mix.grade_table9_accuracy --plan PLAN.json.
Python code runs in disposable, credential-free Docker containers, never on the host.

MBPP's native prompts describe each task in words and never name the function its asserts call, so the grader
binds that name to the function the completion defines before the asserts run (``mbpp_binding``).
"""

import argparse
import ast
import builtins
import gzip
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    process_results,
    remove_boxed,
)
from marin.evaluation.olmo_base_eval.accuracy import GENERATION_BACKFILL_TASKS, validate_task_samples

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference

PYTHON_IMAGE = "python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
PRIMARY_METRICS = {
    task: "math_verify" if task.startswith("minerva_math_") else "pass@1" for task in GENERATION_BACKFILL_TASKS
}
BUILTIN_NAMES = frozenset(dir(builtins))


def grade_math(sample: dict) -> dict[str, float]:
    solution = sample["metadata"]["solution_text"]
    answer = normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))
    metrics = process_results({"solution": solution, "answer": answer}, [sample["generation"]])
    return {key: float(value) for key, value in metrics.items()}


def tested_names(test: str) -> list[str]:
    """Functions the asserts call that are neither builtins nor defined or imported by the test itself."""
    tree = ast.parse(test)
    local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local |= {
        (a.asname or a.name).split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom) for a in n.names
    }
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in BUILTIN_NAMES and name not in local and name not in names:
                names.append(name)
    return names


def call_shapes(test: str, name: str) -> set[tuple[int, frozenset[str]]]:
    return {
        (len(n.args), frozenset(k.arg for k in n.keywords if k.arg))
        for n in ast.walk(ast.parse(test))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    }


def accepts(fn: ast.FunctionDef | ast.AsyncFunctionDef, positional: int, keywords: frozenset[str]) -> bool:
    a = fn.args
    params = [p.arg for p in (*a.posonlyargs, *a.args)]
    if positional > len(params) and a.vararg is None:
        return False
    if not keywords <= set(params[positional:]) | {p.arg for p in a.kwonlyargs} and a.kwarg is None:
        return False
    required = params[: len(params) - len(a.defaults)]
    if any(p not in keywords for p in required[positional:]):
        return False
    return all(d is not None or p.arg in keywords for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True))


def module_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        else:
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return names


def mbpp_binding(code: str, test: str) -> dict:
    """Choose the completion's function for the name MBPP's asserts call.

    Among top-level functions that accept every call in the asserts, prefer those no other top-level function calls
    (the entry points) and take the last defined. A completion that already defines the tested name, does not parse,
    or has no function that accepts the calls is left unbound, as is a test that calls more than one such name.
    """
    names = tested_names(test)
    if len(names) != 1:
        return {"category": "multi_name_test", "tested": names}
    (name,) = names
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return {"category": "syntax_error", "tested": name}
    if name in module_names(tree):
        return {"category": "named", "tested": name}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    shapes = call_shapes(test, name)
    compatible = [f for f in functions if all(accepts(f, p, k) for p, k in shapes)]
    if not compatible:
        return {"category": "no_compatible_function", "tested": name}
    called = {
        n.func.id
        for f in functions
        for n in ast.walk(f)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != f.name
    }
    entry = [f for f in compatible if f.name not in called] or compatible
    return {"category": "bound", "tested": name, "chosen": entry[-1].name}


def python_program(sample: dict, text: str) -> tuple[str, dict | None]:
    # Completion prompts already open the code fence (MBPP) or function body (HumanEval).
    code = sample["metadata"].get("answer_prefix", "") + text.split("```")[0]
    test = sample["metadata"]["test"]
    if sample["task"] != "mbpp":
        return code + "\n" + test + "\n", None
    binding = mbpp_binding(code, test)
    alias = f"{binding['tested']} = {binding['chosen']}\n" if binding["category"] == "bound" else ""
    return code + "\n" + alias + test + "\n", binding


def sandbox_python(program: str, *, timeout: int = 10) -> dict:
    """Execute one program with no mounts, credentials, network, or host privileges."""
    name = "table9-grade-" + uuid.uuid4().hex
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--log-driver",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "-i",
        PYTHON_IMAGE,
        "python",
        "-I",
        "-",
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=30)
        # Generated programs can flood either stream; retain their source and exit status, not unbounded logs.
        timed_out = False
        try:
            subprocess.run(
                ["docker", "start", "-ai", name],
                input=program.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        state = json.loads(
            subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", name], timeout=30)
        )
        if state["Error"] or state["StartedAt"].startswith("0001-"):
            raise RuntimeError(f"Sandbox did not start: {state}")
        if timed_out:
            return {"passed": False, "exit_code": None, "timeout": True}
        if state["Status"] != "exited":
            raise RuntimeError(f"Sandbox did not finish: {state}")
        return {"passed": state["ExitCode"] == 0, "exit_code": state["ExitCode"], "timeout": False}
    finally:
        # The exact UUID names only this invocation's sandbox, including a timed-out one.
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def grade_sample(sample: dict) -> dict:
    if sample["task"].startswith("minerva_math_"):
        return sample | {"metrics": grade_math(sample)}
    program, binding = python_program(sample, sample["generation"])
    execution = sandbox_python(program)
    return sample | {"metrics": {"pass@1": float(execution["passed"])}, "execution": execution, "binding": binding}


def grader_identity() -> dict:
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "math_verify": importlib.metadata.version("math-verify"),
        "antlr4": importlib.metadata.version("antlr4-python3-runtime"),
        "lm_eval_revision": inference.existing.lm_eval_revision(),
        "math_scorer_sha256": hashlib.sha256(Path(process_results.__code__.co_filename).read_bytes()).hexdigest(),
        "python_image": PYTHON_IMAGE,
    }


def graded_result(plan: dict, row: dict, task: str, limit: int) -> dict | None:
    """Read and verify existing scores without executing generated programs."""
    marker = inference.completed_task(plan, row, task, limit)
    if marker is None:
        return None
    root = inference.result_root(plan, row, task, limit)
    grader = grader_identity()
    grade_root = root + "/grading/" + inference.digest(grader)
    fs, _ = fsspec.core.url_to_fs(grade_root)
    if not fs.exists(grade_root + "/SUCCESS.json"):
        return None
    saved = inference.existing.read_json(grade_root + "/SUCCESS.json")
    payload = inference.existing.read_bytes(grade_root + "/samples.json.gz")
    if (
        saved["input_artifact"] != marker["artifact"]
        or saved["grader"] != grader
        or saved["row"] != row
        or saved["task"] != task
        or saved["limit"] != limit
    ):
        raise ValueError("Completed grading provenance differs")
    if saved["artifact"] != {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}:
        raise ValueError("Completed grading artifact differs")
    samples = json.loads(gzip.decompress(payload))
    count = marker["count"]
    keys = samples[0]["metrics"].keys()
    metrics = {key: validate_task_samples(task, samples, list(range(count)), key) for key in keys}
    return {"task": task, "metrics": metrics, "count": count, "grading_root": grade_root, "grader": grader}


def grade_task(plan: dict, row: dict, task: str, limit: int) -> dict | None:
    result = graded_result(plan, row, task, limit)
    if result is not None:
        return result
    marker = inference.completed_task(plan, row, task, limit)
    if marker is None:
        return None
    root = inference.result_root(plan, row, task, limit)
    grader = grader_identity()
    grade_root = root + "/grading/" + inference.digest(grader)
    samples = json.loads(gzip.decompress(inference.existing.read_bytes(root + "/samples.json.gz")))
    if task.startswith("minerva_math_"):
        samples = [grade_sample(s) for s in samples]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            samples = list(pool.map(grade_sample, samples))
    for key in samples[0]["metrics"]:
        validate_task_samples(task, samples, list(range(marker["count"])), key)
    data = gzip.compress(json.dumps(samples, sort_keys=True, allow_nan=False).encode(), mtime=0)
    artifact = inference.write_verified(grade_root + "/samples.json.gz", data)
    saved = {
        "input_artifact": marker["artifact"],
        "grader": grader,
        "artifact": artifact,
        "row": row,
        "task": task,
        "limit": limit,
    }
    inference.write_verified(grade_root + "/SUCCESS.json", inference.existing.canonical_json(saved))
    return graded_result(plan, row, task, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--canary-documents", type=int, default=0)
    parser.add_argument("--reference-audit", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.reference_audit:
        for task in GENERATION_BACKFILL_TASKS:
            samples = inference.task_requests(plan, task)
            if args.canary_documents:
                samples = samples[: args.canary_documents]
            samples = [s | {"generation": s["reference"]} for s in samples]
            if task.startswith("minerva_math_"):
                results = [grade_sample(s) for s in samples]
            else:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(grade_sample, samples))
            failures = [s for s in results if not s["metrics"][PRIMARY_METRICS[task]]]
            print(
                json.dumps(
                    {
                        "task": task,
                        "count": len(samples),
                        "failures": len(failures),
                        "failed_ids": [s["doc_id"] for s in failures],
                    }
                ),
                flush=True,
            )
            report = {
                "task": task,
                "grader": grader_identity(),
                "primary_metric": PRIMARY_METRICS[task],
                "count": len(samples),
                "failures": failures,
            }
            uri = plan["output_root"] + "/reference-audit/" + inference.digest(grader_identity())
            inference.write_verified(uri + f"/{task}-n{args.canary_documents}.json", json.dumps(report).encode())
        return
    for row in plan["rows"]:
        for task in GENERATION_BACKFILL_TASKS:
            result = grade_task(plan, row, task, args.canary_documents)
            print(json.dumps({"checkpoint": row["name"], "task": task, "result": result}), flush=True)


if __name__ == "__main__":
    main()
