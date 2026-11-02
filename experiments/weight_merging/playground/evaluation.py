# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-panel prompts and scores, with generated Python executed in Docker."""

import json
import math
import random
import re
import subprocess
import time
import uuid
from decimal import Decimal, InvalidOperation

DOCKER_IMAGE = "python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f"
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
GSM_SMOKE_IDS = (0, 3, 7, 11, 19, 29, 43, 59, 79, 101, 127, 151)


def remove_timed_out_container(name: str) -> None:
    """Allow an auto-removal race only after the named container disappears."""
    try:
        subprocess.run(["docker", "rm", "--force", name], check=True, capture_output=True, text=True, timeout=15)
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip()
        raced = "No such container" in message or (
            f"removal of container {name}" in message and "already in progress" in message
        )
        if not raced:
            raise RuntimeError(f"Docker timeout cleanup failed: {message}") from error
        for _ in range(20):
            remaining = subprocess.run(
                ["docker", "ps", "--all", "--filter", f"name={name}", "--format", "{{.Names}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if name not in {line.strip().lstrip("/") for line in remaining.stdout.splitlines()}:
                return
            time.sleep(0.1)
        raise RuntimeError(f"Docker timeout cleanup did not remove {name}: {message}") from error


def validate_vocabulary(expected: dict[str, int], actual: dict[str, int], padding_token: str | None) -> None:
    """Require aligned shared rows; allow one newly named padding row only."""
    if any(actual.get(token) != index for token, index in expected.items()):
        raise ValueError("Shared vocabulary token IDs differ")
    extra = set(actual) - set(expected)
    if extra and (extra != {padding_token} or actual[padding_token] <= max(expected.values())):
        raise ValueError("Unexpected extra vocabulary tokens")


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
        remove_timed_out_container(name)
        container.communicate()
        return {"correct": False, "error": "timeout"}
    if container.returncode == 125:
        raise RuntimeError("Docker evaluator failed to start")
    return {"correct": container.returncode == 0, "error": f"exit-{container.returncode}"}


def unclosed_thinking(text: str) -> bool:
    """Count literal thinking tags, allowing a closed section before a new one."""
    depth = 0
    for token in re.findall(r"</?think>", text):
        depth = max(0, depth - 1) if token == "</think>" else depth + 1
    return depth > 0


def sign_test_pvalue(wins: int, losses: int) -> float:
    """Exact two-sided binomial test on discordant independent item pairs."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (1 << n))


def grade_completion(item: dict, completion: str, *, defer_code: bool = False) -> dict:
    """Leave execution pending on GPU pods; an isolated CPU scorer fills it in."""
    if defer_code and item["dataset"] == "mbpp":
        return {"correct": None, "pending_execution": True}
    return score(item, completion)


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
                        "sign_test_p_two_sided": sign_test_pvalue(wins, losses) if mode == "greedy" else None,
                        "unclosed_think": sum(unclosed_thinking(r.get("completion", "")) for r in selected),
                        "opens_think": sum("<think>" in r.get("completion", "") for r in selected),
                        "truncated": sum(r.get("finish_reason") == "length" for r in selected),
                    }
                )
    return summaries


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def item_bootstrap(differences: list[list[int]], *, resamples: int = 5000, seed: int = 290) -> dict:
    """Bootstrap items after averaging their paired decode-level score differences."""
    means = [sum(values) / len(values) for values in differences]
    effect = sum(means) / len(means)
    if len(set(means)) == 1:
        interval = [effect, effect]
    else:
        rng = random.Random(seed)
        samples = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(resamples))
        interval = [samples[int(0.025 * (resamples - 1))], samples[int(0.975 * (resamples - 1))]]
    return {
        "paired_items": len(means),
        "mean_item_gain": effect,
        "item_bootstrap_ci95": interval,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }
