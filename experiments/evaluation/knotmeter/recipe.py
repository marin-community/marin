# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build isolated upstream commands without importing the Marin workspace."""

import os
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

DIRECTORY = Path(__file__).parent


SERIES_PATH = DIRECTORY / "recipe.yaml"


REQUIREMENTS_PATH = DIRECTORY / "requirements-harness.txt"


PREPARATION_VERSION = 1
SERVER_PORT = 38197
ATTEMPT_RECEIPT_NAME = "receipt.json"


class PointManifest(TypedDict):
    profile: str
    concurrency: int
    series: dict
    recipe_fingerprint: str
    vllm_release: dict
    workload: dict
    source: dict
    placement: dict
    diagnostic: bool


class TrialPlan(TypedDict):
    schema_version: int
    trial_id: str
    trial_uri: str
    campaign_uri: str
    controller: str
    cluster: str
    marin_prefix: str
    series: dict
    vllm_release: dict
    placement: dict
    job_id: str
    gpu_job_id: str
    source: dict
    research_logbook: str | None
    manifest: NotRequired[PointManifest]


def unmeasured_result(plan: TrialPlan, state: str, *, cost_complete: bool = False) -> dict:
    """Keep plans without collected measurements visible in campaign reports."""
    return {
        "trial_id": plan["trial_id"],
        "trial_uri": plan["trial_uri"],
        "manifest": (
            plan.get("manifest")
            or {
                "series": plan["series"],
                "diagnostic": bool(plan["series"].get("diagnostic")),
                "vllm_release": plan["vllm_release"],
                "workload": {"recipe": None},
            }
        ),
        "job_state": state,
        "job_succeeded": False,
        "attempts": [],
        "measurement": None,
        "gpu_seconds": 0.0,
        "cost_complete": cost_complete,
        "cost_basis": (
            "GPU child was not submitted or the controller confirmed it absent"
            if cost_complete
            else "Attempts have not been collected; GPU allocation cost remains unknown"
        ),
    }


def load_series(path: Path = SERIES_PATH) -> dict:
    return yaml.safe_load(path.read_text())


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


def preparation_command(series: dict, series_path: Path, bundle: Path) -> list[str]:
    return [
        *harness_command(series),
        "python",
        "-m",
        "experiments.evaluation.knotmeter.prepare",
        "--series",
        str(series_path),
        "--output",
        str(bundle),
    ]


def processor_package(series: dict) -> str:
    return (
        "infx @ git+https://github.com/SemiAnalysisAI/InferenceX.git@"
        + series["processor_source"]
        + "#subdirectory=inferencex-e2e"
    )


def profile_arguments(
    series: dict, concurrency: int, url: str, output: Path, *, metrics_url: str | None = None
) -> list[str]:
    dataset = (
        ["--input-file", series["dataset"], "--custom-dataset-type", "weka_trace"]
        if series["dataset_loader"] == "weka_trace"
        else ["--public-dataset", series["dataset_loader"]]
    )
    args = [
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
        series["tokenizer"],
        "--tokenizer-revision",
        series["tokenizer_revision"],
        *dataset,
        "--max-context-length",
        str(series["max_context"]),
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(series["duration"]),
        "--warmup-requests-per-lane",
        str(series["warmup_requests_per_lane"]),
        "--failed-request-threshold",
        str(series["max_profiling_error_ratio"]),
        "--random-seed",
        str(series["random_seed"]),
        "--agentic-warmup-grace-period",
        str(series["warmup_grace_period"]),
        "--trace-idle-gap-cap-seconds",
        str(series["trace_idle_gap_cap_seconds"]),
        "--server-metrics",
        metrics_url or f"{url}/metrics",
        "--output-artifact-dir",
        str(output),
        "--ui",
        "none",
    ]
    return [*args, "--unsafe-override"] if series["unsafe_override"] else args


def benchmark_command(
    series: dict, concurrency: int, url: str, output: Path, *, metrics_url: str | None = None
) -> list[str]:
    return [
        *harness_command(series),
        "agentx",
        *profile_arguments(series, concurrency, url, output, metrics_url=metrics_url),
    ]


def workload_environment(bundle: Path, series: dict) -> dict[str, str]:
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
        "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID": "true",
    }
