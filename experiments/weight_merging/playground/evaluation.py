# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-panel prompts and scores, with generated Python executed in Docker."""

import json
import re
import subprocess
import uuid
from decimal import Decimal, InvalidOperation

DOCKER_IMAGE = "python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f"
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
GSM_SMOKE_IDS = (0, 3, 7, 11, 19, 29, 43, 59, 79, 101, 127, 151)


def prompt(item: dict) -> str:
    if item["dataset"] == "gsm8k":
        return item["prompt"] + "\nSolve step by step. End with '#### <number>'."
    tests = "\n".join(item["tests"])
    return f"Write Python code for this task. Return only a Python code block.\n{item['prompt']}\nTests:\n{tests}"


def numeric_answer(text: str) -> Decimal | None:
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    elif "\\boxed{" in text:
        text = text.rsplit("\\boxed{", 1)[-1].split("}", 1)[0]
    numbers = NUMBER.findall(text)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


def python_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else text.strip()


def score(item: dict, completion: str) -> dict:
    """Return correctness and failure detail without executing Python on the host."""
    if item["dataset"] == "gsm8k":
        answer = numeric_answer(completion)
        return {
            "correct": answer == numeric_answer(item["answer"]),
            "prediction": str(answer) if answer is not None else None,
        }
    setup = item["setup"]
    if isinstance(setup, list):
        setup = "\n".join(setup)
    code = "\n".join([setup, python_code(completion), *item["tests"]])
    name = f"merge-eval-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=128m",
        "--cpus=0.5",
        "--user=65534:65534",
        "--interactive",
        DOCKER_IMAGE,
        "python",
        "-I",
        "-",
    ]
    container = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )
    try:
        container.communicate(code, timeout=10)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "--force", name], check=True, capture_output=True, timeout=15)
        container.communicate()
        return {"correct": False, "error": "timeout"}
    if container.returncode == 125:
        raise RuntimeError("Docker evaluator failed to start")
    return {"correct": container.returncode == 0, "error": f"exit-{container.returncode}"}


def paired_summary(rows: list[dict], baseline: str) -> list[dict]:
    """Compare only matching (item ID, decoding mode, seed) observations."""
    reference = {(r["item_id"], r["mode"], r["seed"]): r for r in rows if r["candidate"] == baseline}
    candidates = sorted({r["candidate"] for r in rows})
    summaries = []
    for candidate in candidates:
        for dataset in ("gsm8k", "mbpp"):
            for mode in ("greedy", "sample"):
                selected = [r for r in rows if (r["candidate"], r["dataset"], r["mode"]) == (candidate, dataset, mode)]
                wins = losses = paired = 0
                for row in selected:
                    key = (row["item_id"], mode, row["seed"])
                    if key not in reference:
                        continue
                    paired += 1
                    wins += row["correct"] and not reference[key]["correct"]
                    losses += reference[key]["correct"] and not row["correct"]
                summaries.append(
                    {
                        "candidate": candidate,
                        "dataset": dataset,
                        "mode": mode,
                        "n": len(selected),
                        "correct": sum(r["correct"] for r in selected),
                        "paired_n": paired,
                        "wins": wins,
                        "losses": losses,
                    }
                )
    return summaries


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
