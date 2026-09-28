# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regrade the retained Python MBPP generations with a name-agnostic harness.

The native MBPP prompt describes the task in words and never names the function the MBPP asserts call, so the
frozen grading fails every generation that picks another name. This regrade binds the tested name to the function
the generation defines (``tested_name = chosen``, inserted between the generation and the asserts) and re-executes
only those programs in the grader's sandbox. Programs that already define the tested name are byte-identical to the
frozen ones and keep their frozen result; a sample of them is re-executed to check that the sandbox reproduces it.

Binding rule: among the generation's top-level functions that accept every call shape in the asserts, prefer those
no other top-level function calls (the entry points), and take the last one defined. Ties are re-run with the first
as well, to report sensitivity to the tie-break.

Controls use MBPP's reference solutions (google-research-datasets/mbpp, full, test): each is run as released, then
with its tested function renamed so the harness has to bind it (positive control), and a shuffled set binds another
problem's renamed reference to the asserts (negative control).

usage (from the repo root; Docker must be running):
  uv run --offline --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 \
    python .agents/projects/mariner_table9_accuracy_20260914/mbpp_regrade/regrade_mbpp_name_agnostic.py [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gzip
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np

from experiments.domain_phase_mix.grade_table9_accuracy import sandbox_python

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "olmix_t9_1e21_east1/accuracy_vs_bpb/coverage_merged.json"
MIXTURES = {
    "proportional_1e21-2f1a48": "Proportional",
    "unimax8_1e21-d685cd": "UniMax-8",
    "olmixq_t9_kl0p005_cap04_1e21-3f95f2": "Olmix",
    "lwspu_t9_snc_cap08_1e21-e8e9d7": "MARINER",
}
MBPP_ROWS = "https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/mbpp&config=full&split=test"
BUILTIN_NAMES = frozenset(dir(builtins))
RENAMED = "solution_fn"
WORKERS = 6
REPRODUCE_SAMPLE = 40
NEGATIVE_SAMPLE = 200
BOOTSTRAP = 10_000
SEED = 0


def tested_names(test: str) -> list[str]:
    """Functions the asserts call that are neither builtins nor defined or imported by the test itself."""
    tree = ast.parse(test)
    local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local |= {
        (a.asname or a.name).split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in n.names
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
    free = set(params[positional:]) | {p.arg for p in a.kwonlyargs}
    if not keywords <= free and a.kwarg is None:
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


def binding(code: str, test: str) -> dict:
    """Classify one generation and choose the function the asserts should call."""
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
    if not functions:
        return {"category": "no_function", "tested": name}
    shapes = call_shapes(test, name)
    compatible = [f for f in functions if all(accepts(f, p, k) for p, k in shapes)]
    if not compatible:
        return {"category": "no_compatible_function", "tested": name, "defined": [f.name for f in functions]}
    called = {
        n.func.id
        for f in functions
        for n in ast.walk(f)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != f.name
    }
    entry = [f for f in compatible if f.name not in called] or compatible
    return {
        "category": "bound" if len(entry) == 1 else "bound_tiebreak",
        "tested": name,
        "chosen": entry[-1].name,
        "alternative": entry[0].name if len(entry) > 1 else None,
        "defined": [f.name for f in functions],
    }


def program(code: str, test: str, alias: str | None = None, tested: str | None = None) -> str:
    bind = f"{tested} = {alias}\n" if alias else ""
    return code + "\n" + bind + test + "\n"


def run_all(jobs: list[tuple[str, str]], log: Path, label: str) -> dict[str, dict]:
    """Execute (key, program) pairs in the sandbox, logging progress and a projected finish."""
    results, start = {}, time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, log.open("a") as out:
        for i, (key, result) in enumerate(zip((k for k, _ in jobs), pool.map(sandbox_python, (p for _, p in jobs)), strict=True), 1):
            results[key] = result
            if i % 50 == 0 or i == len(jobs):
                rate = i / (time.time() - start)
                out.write(f"{time.strftime('%H:%M:%S')} {label} {i}/{len(jobs)} ETA {(len(jobs) - i) / rate / 60:.1f} min\n")
                out.flush()
    return results


def load_generations() -> dict[str, list[dict]]:
    coverage = json.loads(COVERAGE.read_text())
    samples = {}
    for report in coverage:
        name = report["checkpoint"]["name"]
        root = report["tasks"]["mbpp"]["grading_root"]
        with fsspec.open(root + "/samples.json.gz", "rb") as handle:
            rows = json.loads(gzip.decompress(handle.read()))
        assert [r["doc_id"] for r in rows] == list(range(500)), name
        assert abs(sum(r["metrics"]["pass@1"] for r in rows) / 500 - report["tasks"]["mbpp"]["metrics"]["pass@1"]) < 1e-12
        samples[name] = rows
    assert set(samples) == set(MIXTURES)
    return samples


def load_references(expected_tests: dict[int, str]) -> dict[int, str]:
    rows = []
    for offset in range(0, 500, 100):
        with urllib.request.urlopen(f"{MBPP_ROWS}&offset={offset}&length=100", timeout=60) as response:
            rows += [r["row"] for r in json.load(response)["rows"]]
    references = {}
    for row in rows:
        task = int(row["task_id"])
        # OLMo-Eval prepends MBPP's setup code (task 367 builds its trees there) to the asserts.
        asserts = "\n".join(row["test_list"])
        expected = row["test_setup_code"] + "\n" + asserts if row["test_setup_code"] else asserts
        assert expected.strip() == expected_tests[task].strip(), task
        references[task] = row["code"]
    assert len(references) == 500
    return references


class Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self.old:
            node.name = self.new
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self.old:
            node.id = self.new
        return node


def renamed_reference(code: str, name: str) -> str:
    return ast.unparse(Rename(name, RENAMED).visit(ast.parse(code)))


def bootstrap(passed: dict[str, np.ndarray]) -> dict:
    """Paired bootstrap over the 500 problems: pass@1 per mixture and every pairwise difference."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 500, size=(BOOTSTRAP, 500))
    draws = {m: v[idx].mean(axis=1) for m, v in passed.items()}
    out = {"pass@1": {}, "differences": {}}
    for m, v in passed.items():
        lo, hi = np.percentile(draws[m], [2.5, 97.5])
        out["pass@1"][m] = {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "se": float(draws[m].std())}
    names = list(passed)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = draws[b] - draws[a]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["differences"][f"{b} - {a}"] = {
                "mean": float(passed[b].mean() - passed[a].mean()),
                "ci95": [float(lo), float(hi)],
                "p_leq_0": float((d <= 0).mean()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="classify and build programs without executing")
    args = parser.parse_args()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    log = out_dir / "regrade.log"
    samples = load_generations()
    tests = {int(r["metadata"]["id"]): r["metadata"]["test"] for r in next(iter(samples.values()))}
    for rows in samples.values():
        assert {int(r["metadata"]["id"]): r["metadata"]["test"] for r in rows} == tests

    records, jobs = {}, []
    for mixture, rows in samples.items():
        for r in rows:
            code = r["metadata"].get("answer_prefix", "") + r["generation"].split("```")[0]
            b = binding(code, r["metadata"]["test"])
            frozen = r["metrics"]["pass@1"] == 1.0
            if frozen and b["category"] != "named":
                b["category"] = "frozen_pass_other"
            records[(mixture, r["doc_id"])] = {"mixture": mixture, "doc_id": r["doc_id"], "task_id": int(r["metadata"]["id"]), "frozen": frozen, **b}
            if b["category"] in ("bound", "bound_tiebreak"):
                jobs.append((f"{mixture}|{r['doc_id']}|chosen", program(code, r["metadata"]["test"], b["chosen"], b["tested"])))
                if b["alternative"]:
                    jobs.append((f"{mixture}|{r['doc_id']}|alternative", program(code, r["metadata"]["test"], b["alternative"], b["tested"])))
    rng = random.Random(SEED)
    named = [k for k, v in records.items() if v["category"] == "named"]
    for mixture, doc in rng.sample(named, min(REPRODUCE_SAMPLE, len(named))):
        r = samples[mixture][doc]
        code = r["metadata"].get("answer_prefix", "") + r["generation"].split("```")[0]
        jobs.append((f"{mixture}|{doc}|reproduce", program(code, r["metadata"]["test"])))

    references = load_references(tests)
    single = {t for t in references if len(tested_names(tests[t])) == 1}
    control = {}
    for task, code in references.items():
        if task not in single:
            continue
        (name,) = tested_names(tests[task])
        renamed = renamed_reference(code, name)
        b = binding(renamed, tests[task])
        control[task] = {"tested": name, "renamed_binding": b}
        jobs.append((f"reference|{task}|as_released", program(code, tests[task])))
        if b["category"] in ("bound", "bound_tiebreak"):
            jobs.append((f"reference|{task}|renamed", program(renamed, tests[task], b["chosen"], name)))
    tasks = sorted(single)
    negatives = []
    for task in rng.sample(tasks, NEGATIVE_SAMPLE):
        (name,) = tested_names(tests[task])
        shapes = call_shapes(tests[task], name)
        donors = [t for t in tasks if t != task]
        rng.shuffle(donors)
        for donor in donors:
            (donor_name,) = tested_names(tests[donor])
            renamed = renamed_reference(references[donor], donor_name)
            b = binding(renamed, tests[task])
            if b["category"] == "bound" and b["chosen"] == RENAMED and shapes:
                negatives.append({"task": task, "donor": donor})
                jobs.append((f"negative|{task}|{donor}", program(renamed, tests[task], RENAMED, name)))
                break

    counts = {}
    for v in records.values():
        counts.setdefault(MIXTURES[v["mixture"]], {}).setdefault(v["category"], 0)
        counts[MIXTURES[v["mixture"]]][v["category"]] += 1
    print(json.dumps(counts, indent=1))
    print("reference renamed-binding categories:", {c: sum(1 for x in control.values() if x["renamed_binding"]["category"] == c) for c in {x["renamed_binding"]["category"] for x in control.values()}})
    print(f"{len(jobs)} sandbox programs; negatives {len(negatives)}")
    if args.dry_run:
        return

    log.write_text(f"{time.strftime('%H:%M:%S')} start {len(jobs)} programs, {WORKERS} workers\n")
    results = run_all(jobs, log, "sandbox")

    per_sample = []
    passed = {"frozen": {}, "regraded": {}, "regraded_first_tie": {}}
    for mixture in MIXTURES:
        frozen, regraded, first_tie = [], [], []
        for doc in range(500):
            v = records[(mixture, doc)]
            chosen = results.get(f"{mixture}|{doc}|chosen")
            alternative = results.get(f"{mixture}|{doc}|alternative")
            new = v["frozen"] if chosen is None else chosen["passed"]
            alt = new if alternative is None else alternative["passed"]
            v |= {"regraded": new, "regraded_first_tie": alt}
            frozen.append(v["frozen"])
            regraded.append(new)
            first_tie.append(alt)
            per_sample.append(v)
        label = MIXTURES[mixture]
        passed["frozen"][label] = np.array(frozen, dtype=float)
        passed["regraded"][label] = np.array(regraded, dtype=float)
        passed["regraded_first_tie"][label] = np.array(first_tie, dtype=float)
    reproduce = [(k, r["passed"], records[(k.split("|")[0], int(k.split("|")[1]))]["frozen"]) for k, r in results.items() if k.endswith("|reproduce")]
    as_released = {int(k.split("|")[1]): r["passed"] for k, r in results.items() if k.startswith("reference|") and k.endswith("|as_released")}
    renamed_ok = {int(k.split("|")[1]): r["passed"] for k, r in results.items() if k.startswith("reference|") and k.endswith("|renamed")}
    negative_pass = [r["passed"] for k, r in results.items() if k.startswith("negative|")]
    summary = {
        "categories": counts,
        "bootstrap": {k: bootstrap(v) for k, v in passed.items()},
        "controls": {
            "reproduce_named": {"n": len(reproduce), "matches_frozen": sum(a == b for _, a, b in reproduce)},
            "reference_as_released": {"n": len(as_released), "passed": sum(as_released.values())},
            "reference_renamed": {
                "bound": len(renamed_ok),
                "passed": sum(renamed_ok.values()),
                "passed_where_released_passes": sum(renamed_ok.get(t, False) for t, ok in as_released.items() if ok),
                "released_passes": sum(as_released.values()),
            },
            "negative_shuffled": {"n": len(negative_pass), "passed": sum(negative_pass)},
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with gzip.open(out_dir / "samples.jsonl.gz", "wt") as handle:
        for v in per_sample:
            handle.write(json.dumps(v, sort_keys=True) + "\n")
    with log.open("a") as out:
        out.write(f"{time.strftime('%H:%M:%S')} DONE\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
