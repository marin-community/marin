# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prespecified wall-time cost proxies from the published policy runtime table."""

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = {
    "math500": "MATH-500",
    "humanevalplus": "HumanEval+",
    "mbppplus": "MBPP+",
    "olympiadbench": "OlympiadBench (30-question subset)",
    "gsm8k-0shot": "GSM8K",
    "truthfulqa": "TruthfulQA",
    "triviaqa": "TriviaQA",
    "aime24": "AIME24",
    "mmlu-pro": "MMLU-Pro",
    "gpqa-diamond": "GPQA Diamond",
    "cruxeval": "CRUXEval (output)",
    "ifbench": "IFBench",
    "mrcr": "MRCR",
    "NUPA": "NUPA stratified (Loose)",
    "swebench-verified": "SWE-bench Verified",
    "ot-tblite-recovery": "OpenThoughts-TBLite 2.0",
    "tb2-recovery": "Terminal-Bench 2.1",
    "ds-1000-local": "DS-1000 mini-200",
    "bfclparity-pi": "BFCL-Parity @ Pi",
    "bixbench-pi": "BixBench CLI corrections @ Pi",
    "tau3-pi": "τ³-bench @ Pi",
}
DOMAINS = {
    "reasoning": {"math500", "aime24", "olympiadbench", "gsm8k-0shot", "gpqa-diamond"},
    "coding": {"humanevalplus", "mbppplus", "cruxeval"},
    "instruction": {"ifbench", "NUPA"},
    "long_context": {"mrcr"},
    "interactive": {
        "swebench-verified",
        "ot-tblite-recovery",
        "tb2-recovery",
        "ds-1000-local",
        "bfclparity-pi",
        "bixbench-pi",
        "tau3-pi",
    },
}
COVERAGE_DOMAINS = tuple(DOMAINS)


def policy_costs(tasks):
    path = Path(__file__).with_name("policy_10197.md")
    text = path.read_text()
    means = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        fields = [value.strip() for value in line.strip("|").split("|")]
        if len(fields) != 9 or fields[0] not in LABELS.values():
            continue
        parts = re.findall(r"(\d+)\s*([hms])", fields[7])
        if not parts:
            raise ValueError(f"Cannot parse policy runtime: {line}")
        means[fields[0]] = sum(int(value) * {"h": 3600, "m": 60, "s": 1}[unit] for value, unit in parts)
    counts = Counter(task["benchmark"] for task in tasks)
    by_benchmark = {benchmark: means[label] / counts[benchmark] for benchmark, label in LABELS.items()}
    costs = np.array([by_benchmark.get(task["benchmark"], np.nan) for task in tasks])
    domains = np.array(
        [
            next((name for name, benchmarks in DOMAINS.items() if task["benchmark"] in benchmarks), "general")
            for task in tasks
        ]
    )
    description = {
        "kind": "Published mean evaluator wall seconds per unique task; selector proxy, not measured subset GPU-hours",
        "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds_per_task": by_benchmark,
        "coverage": (
            "In each block of 21 selections, reserve one task for each of the five "
            "capability domains; micro >=10 and mini >=40 per domain."
        ),
    }
    return costs, domains, description


def covered_candidates(candidates, pool, domains, step):
    if domains is None or step % 21 >= len(COVERAGE_DOMAINS):
        return candidates
    required = COVERAGE_DOMAINS[step % 21]
    selected = candidates[domains[pool[candidates]] == required]
    already_selected = int(np.sum(domains[pool] == required)) - len(selected)
    if already_selected >= step // 21 + 1:
        return candidates
    if not len(selected):
        raise ValueError(f"Insufficient tasks for coverage domain {required}")
    return selected
