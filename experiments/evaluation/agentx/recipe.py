# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client inputs shared by CPU preparation and isolated AgentX replay."""

import os
from pathlib import Path

SERIES_PATH = Path(__file__).with_name("series.json")
REQUIREMENTS_PATH = Path(__file__).with_name("requirements-harness.txt")
PREPARATION_VERSION = 1


def harness_command(series: dict) -> list[str]:
    return [
        "uvx",
        "--python",
        "3.12",
        "--from",
        f"agentx-harness=={series['harness_version']}",
        "--with-requirements",
        str(REQUIREMENTS_PATH),
    ]


def profile_arguments(series: dict, concurrency: int, url: str, output: Path) -> list[str]:
    """Let the pinned agentx preset own warmup, replay and session-start defaults."""
    return [
        "profile",
        "--scenario",
        "agentx",
        "--url",
        url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        series["model"],
        "--tokenizer",
        series["model"],
        "--tokenizer-revision",
        series["model_revision"],
        "--public-dataset",
        series["dataset_loader"],
        "--max-context-length",
        str(series["max_context"]),
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(series["duration"]),
        "--failed-request-threshold",
        str(series["max_profiling_error_ratio"]),
        "--random-seed",
        str(series["random_seed"]),
        "--agentic-warmup-grace-period",
        str(series["warmup_grace_period"]),
        "--trace-idle-gap-cap-seconds",
        str(series["trace_idle_gap_cap_seconds"]),
        "--server-metrics",
        f"{url}/metrics",
        "--output-artifact-dir",
        str(output),
        "--ui",
        "none",
    ]


def benchmark_command(series: dict, concurrency: int, url: str, output: Path) -> list[str]:
    return [*harness_command(series), "agentx", *profile_arguments(series, concurrency, url, output)]


def workload_environment(bundle: Path, series: dict) -> dict[str, str]:
    # A comparison recipe must not inherit hidden AIPerf overrides or a shared
    # HF cache path. CLI inputs and the pinned preset own the workload settings.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AIPERF_")
        and key not in {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"}
    }
    return environment | {
        "HF_HOME": str(bundle / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "AIPERF_VERSION": f"agentx-{series['harness_version']}",
        "AIPERF_DATASET_MMAP_CACHE_ENABLED": "true",
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(bundle / "native"),
    }
