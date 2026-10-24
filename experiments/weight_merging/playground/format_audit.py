# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rescore fixed completions with deterministic extraction, without regenerating."""

import argparse
import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.weight_merging.playground.evaluation import python_code, read_rows, score

FENCES = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ROOT_CODE = re.compile(
    r"(?:async\s+def\s|def\s|class\s|import\s|from\s|@|#|assert\s|if\s|elif\s|else:|for\s|while\s|"
    r"try:|except\b|finally:|with\s|[A-Za-z_]\w*(?:\[.*?\])?\s*(?:=|\+=|-=))"
)
ALLOWED = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Assert,
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.With,
)


def tolerant_code(text: str) -> tuple[str | None, str]:
    """Prefer last closed Python fence, else last complete contiguous draft.

    Draft selection uses syntax only. No expected answer or execution result is
    available to this function. A salvaged draft is not a completed model answer.
    """
    fenced = FENCES.findall(text)
    if fenced:
        return fenced[-1].strip(), "last_closed_fence"
    lines = text.replace("<think>", "").replace("</think>", "").splitlines()
    selected = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line[0].isspace() or not ROOT_CODE.match(line):
            index += 1
            continue
        start = index
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and not line[0].isspace() and not ROOT_CODE.match(line):
                break
            index += 1
        for end in range(index, start, -1):
            candidate = "\n".join(lines[start:end]).strip()
            try:
                tree = ast.parse(candidate)
            except SyntaxError:
                continue
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body
            ) and all(isinstance(node, ALLOWED) for node in tree.body):
                selected = candidate
                break
    return (selected, "unfenced_ast_draft") if selected else (None, "no_code")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = {item["id"]: item for item in json.loads((args.run / "panel.json").read_text())["items"]}
    rows = [
        row
        for row in read_rows(args.run / "outputs.jsonl")
        if row["candidate"] == args.candidate and row["dataset"] == "mbpp" and row["mode"] == "greedy"
    ]

    def rescore(row):
        code, policy = tolerant_code(row["completion"])
        if code is None:
            correct = False
        elif code == python_code(row["completion"]).strip():
            correct = row["correct"]
        else:
            correct = score(items[row["item_id"]], code)["correct"]
        return {
            "item_id": row["item_id"],
            "strict_correct": row["correct"],
            "tolerant_correct": correct,
            "extraction_policy": policy,
            "extracted_code": code,
            "finish_reason": row["finish_reason"],
            "has_thinking": "<think>" in row["completion"],
            "closed_thinking": "</think>" in row["completion"],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        audited = list(pool.map(rescore, rows))
    summary = {
        "candidate": args.candidate,
        "n": len(audited),
        "strict_correct": sum(row["strict_correct"] for row in audited),
        "tolerant_correct": sum(row["tolerant_correct"] for row in audited),
        "recovered": sum(row["tolerant_correct"] and not row["strict_correct"] for row in audited),
        "lost": sum(row["strict_correct"] and not row["tolerant_correct"] for row in audited),
        "recovered_truncated_drafts": sum(
            row["tolerant_correct"] and not row["strict_correct"] and row["finish_reason"] == "length" for row in audited
        ),
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": audited}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
