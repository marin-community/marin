from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
TELEMETRY_PREFIX = "PARITY_TELEMETRY "
BACKEND_LABELS = {"levanter-gpu": "Levanter", "vllm-gpu": "vLLM"}
COLORS = {"levanter-gpu": "#4C78A8", "vllm-gpu": "#F58518"}
TASK_SECONDS = {"levanter-gpu": 8 * 60 + 19.11, "vllm-gpu": 4 * 60 + 15.38}
TOLERANCE = 0.075


def read_telemetry(path: Path) -> list[dict[str, object]]:
    records = []
    decoder = json.JSONDecoder()
    for line in path.read_text().splitlines():
        position = 0
        while (marker := line.find(TELEMETRY_PREFIX, position)) >= 0:
            payload_start = marker + len(TELEMETRY_PREFIX)
            record, consumed = decoder.raw_decode(line[payload_start:])
            records.append(record)
            position = payload_start + consumed
    return records


def complete_gpu_cycles(gpu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backend, backend_rows in gpu.groupby("backend", sort=False):
        cycle = -1
        current = []
        for record in backend_rows.to_dict("records"):
            if record["gpu_index"] == 0:
                if len(current) == 8 and {row["gpu_index"] for row in current} == set(range(8)):
                    rows.append(aggregate_gpu_cycle(backend, cycle, current))
                cycle += 1
                current = []
            current.append(record)
        if len(current) == 8 and {row["gpu_index"] for row in current} == set(range(8)):
            rows.append(aggregate_gpu_cycle(backend, cycle, current))
    return pd.DataFrame(rows)


def aggregate_gpu_cycle(backend: str, cycle: int, rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    return {
        "backend": backend,
        "cycle": cycle,
        "elapsed_seconds": frame["elapsed_seconds"].median(),
        "emission_span_seconds": frame["elapsed_seconds"].max() - frame["elapsed_seconds"].min(),
        "phase": frame["phase"].mode().iat[0],
        "memory_used_mib_mean": frame["memory_used_mib"].mean(),
        "memory_used_mib_max": frame["memory_used_mib"].max(),
        "gpu_utilization_percent_mean": frame["gpu_utilization_percent"].mean(),
        "gpu_utilization_percent_max": frame["gpu_utilization_percent"].max(),
        "power_watts_mean": frame["power_watts"].mean(),
        "power_watts_max": frame["power_watts"].max(),
        "temperature_c_max": frame["temperature_c"].max(),
    }


def save_runtime_breakdown(events: pd.DataFrame, pytest_seconds: dict[str, float]) -> dict[str, dict[str, float]]:
    phase = events[events.event == "phase_complete"].set_index(["backend", "phase"])["duration_seconds"]
    remote = events[events.event == "remote_complete"].set_index("backend")["total_seconds"]
    levanter_batches = events[events.event == "levanter_batch"]["batch_seconds"].sum()
    vllm_scoring = events[events.event == "vllm_wave"]["wave_seconds"].sum() + events[
        events.event == "vllm_sentinel_wave"
    ]["wave_seconds"].sum()

    breakdown = {
        "levanter-gpu": {
            "Client / scheduler": pytest_seconds["levanter-gpu"] - TASK_SECONDS["levanter-gpu"],
            "Iris environment / finalization": TASK_SECONDS["levanter-gpu"] - remote["levanter-gpu"],
            "Imports": phase["levanter-gpu", "imports"],
            "Fixture + model config": phase["levanter-gpu", "fixture"] + phase["levanter-gpu", "model_config"],
            "Model load / server startup": phase["levanter-gpu", "model_load"],
            "Prompt scoring": levanter_batches,
            "Shutdown + remainder": remote["levanter-gpu"]
            - phase["levanter-gpu", "imports"]
            - phase["levanter-gpu", "fixture"]
            - phase["levanter-gpu", "model_config"]
            - phase["levanter-gpu", "model_load"]
            - levanter_batches,
        },
        "vllm-gpu": {
            "Client / scheduler": pytest_seconds["vllm-gpu"] - TASK_SECONDS["vllm-gpu"],
            "Iris environment / finalization": TASK_SECONDS["vllm-gpu"] - remote["vllm-gpu"],
            "Imports": phase["vllm-gpu", "imports"],
            "Fixture + model config": phase["vllm-gpu", "fixture"],
            "Model load / server startup": phase["vllm-gpu", "server_startup"],
            "Prompt scoring": vllm_scoring,
            "Shutdown + remainder": remote["vllm-gpu"]
            - phase["vllm-gpu", "imports"]
            - phase["vllm-gpu", "fixture"]
            - phase["vllm-gpu", "server_startup"]
            - vllm_scoring,
        },
    }

    component_colors = {
        "Client / scheduler": "#BAB0AC",
        "Iris environment / finalization": "#9D755D",
        "Imports": "#72B7B2",
        "Fixture + model config": "#54A24B",
        "Model load / server startup": "#E45756",
        "Prompt scoring": "#4C78A8",
        "Shutdown + remainder": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(2)
    bottom = np.zeros(2)
    for component in component_colors:
        values = np.array([breakdown[backend][component] for backend in BACKEND_LABELS])
        ax.bar(x, values, bottom=bottom, label=component, color=component_colors[component], width=0.62)
        bottom += values
    for index, value in enumerate(bottom):
        ax.text(index, value + 8, f"{value:.1f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, BACKEND_LABELS.values())
    ax.set_ylabel("End-to-end pytest time (seconds)")
    ax.set_title("Cold standing-cluster runtime breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(bottom) * 1.12)
    fig.tight_layout()
    fig.savefig(ROOT / "runtime-breakdown.png", dpi=180)
    plt.close(fig)
    return breakdown


def save_batch_runtime(events: pd.DataFrame) -> None:
    levanter = events[events.event == "levanter_batch"].sort_values("batch_index")
    vllm = events[events.event == "vllm_wave"].sort_values("wave")
    labels = [f"{int(bucket):,}\n{letter}" for bucket, letter in zip(levanter.bucket_max_tokens, "ABCDEFGH")]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - width / 2, levanter.batch_seconds, width, label="Levanter", color=COLORS["levanter-gpu"])
    ax.bar(x + width / 2, vllm.wave_seconds, width, label="vLLM", color=COLORS["vllm-gpu"])
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Padded bucket max tokens and batch/wave (8 prompts)")
    ax.set_ylabel("Scoring time (seconds, log scale)")
    ax.set_title("Prompt scoring runtime by production-shaped batch")
    ax.legend()
    ax.text(
        0.01,
        0.98,
        "Levanter's first occurrence of a shape includes JIT compilation.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "batch-runtime.png", dpi=180)
    plt.close(fig)


def save_accuracy(parity: pd.DataFrame) -> None:
    representative = parity[(parity.backend == "levanter-gpu") | (parity.request_kind == "representative")]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for backend, rows in representative.groupby("backend"):
        ax.scatter(
            rows.prompt_tokens,
            rows.max_probability_error,
            alpha=0.78,
            s=34,
            color=COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
        worst = rows.loc[rows.max_probability_error.idxmax()]
        ax.annotate(
            str(worst.case_id),
            (worst.prompt_tokens, worst.max_probability_error),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(TOLERANCE, color="#E45756", linestyle="--", linewidth=1.5, label="0.075 gate")
    ax.set_xscale("log")
    ax.set_xlabel("Prompt tokens (log scale)")
    ax.set_ylabel("Maximum top-25 probability error")
    ax.set_title("Representative-case error versus prompt length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "accuracy-vs-prompt-length.png", dpi=180)
    plt.close(fig)


def save_vllm_latency(parity: pd.DataFrame) -> None:
    rows = parity[(parity.backend == "vllm-gpu") & (parity.request_kind == "representative")]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    scatter = ax.scatter(
        rows.prompt_tokens,
        rows.request_seconds,
        c=rows.bucket_max_tokens,
        cmap="viridis",
        alpha=0.8,
        s=38,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens (log scale)")
    ax.set_ylabel("HTTP request time (seconds, log scale)")
    ax.set_title("vLLM rank-pinned request latency")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Bucket max tokens")
    fig.tight_layout()
    fig.savefig(ROOT / "vllm-request-latency.png", dpi=180)
    plt.close(fig)


def save_sentinel(parity: pd.DataFrame) -> None:
    sentinel = parity[(parity.backend == "vllm-gpu") & (parity.request_kind == "sentinel")].sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(sentinel["rank"], sentinel.max_probability_error, color=COLORS["vllm-gpu"])
    axes[0].axhline(TOLERANCE, color="#E45756", linestyle="--", label="0.075 gate")
    axes[0].set_xlabel("Data-parallel rank")
    axes[0].set_ylabel("Maximum probability error")
    axes[0].set_title("Sentinel numerical spread")
    axes[0].legend()
    axes[1].bar(sentinel["rank"], sentinel.request_seconds, color="#72B7B2")
    axes[1].set_xlabel("Data-parallel rank")
    axes[1].set_ylabel("Request time (seconds)")
    axes[1].set_title("Sentinel latency spread")
    fig.suptitle("15,025-token knowledge-longbench-02 replayed on all eight vLLM ranks")
    fig.tight_layout()
    fig.savefig(ROOT / "vllm-sentinel-by-rank.png", dpi=180)
    plt.close(fig)


def save_gpu_resources(cycles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    metrics = [
        ("memory_used_mib_max", "Max HBM used (MiB)"),
        ("gpu_utilization_percent_mean", "Mean GPU utilization (%)"),
        ("power_watts_mean", "Mean power per GPU (W)"),
    ]
    for column, backend in enumerate(BACKEND_LABELS):
        rows = cycles[cycles.backend == backend].sort_values("elapsed_seconds")
        for row, (metric, label) in enumerate(metrics):
            axes[row, column].plot(rows.elapsed_seconds, rows[metric], color=COLORS[backend], linewidth=1.4)
            axes[row, column].set_ylabel(label)
            if row == 0:
                axes[row, column].set_title(BACKEND_LABELS[backend])
            if row == 2:
                axes[row, column].set_xlabel("Remote elapsed time (seconds)")
    fig.suptitle("Two-second GPU telemetry across all eight H100s")
    fig.tight_layout()
    fig.savefig(ROOT / "gpu-resource-timeline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    records = read_telemetry(ROOT / "levanter-job.log") + read_telemetry(ROOT / "vllm-job.log")
    events = pd.DataFrame(records)
    with (ROOT / "telemetry.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    gpu = events[events.event == "gpu_sample"].copy()
    gpu.to_csv(ROOT / "gpu-samples.csv", index=False)
    cycles = complete_gpu_cycles(gpu)
    cycles.to_csv(ROOT / "gpu-cycles.csv", index=False)

    levanter_parity = events[events.event == "parity"].copy()
    levanter_parity["request_kind"] = "representative"
    vllm_parity = events[events.event == "vllm_request"].copy()
    parity = pd.concat([levanter_parity, vllm_parity], ignore_index=True, sort=False)
    parity.to_csv(ROOT / "parity-observations.csv", index=False)

    runtime_events = events[
        events.event.isin(
            ["phase_complete", "levanter_batch", "vllm_wave", "vllm_sentinel_wave", "remote_complete"]
        )
    ].copy()
    runtime_events.to_json(ROOT / "runtime-events.jsonl", orient="records", lines=True)

    junit = ET.parse(ROOT / "junit.xml")
    pytest_seconds = {}
    for testcase in junit.findall(".//testcase"):
        backend = "levanter-gpu" if "levanter-gpu" in testcase.attrib["name"] else "vllm-gpu"
        pytest_seconds[backend] = float(testcase.attrib["time"])

    breakdown = save_runtime_breakdown(events, pytest_seconds)
    save_batch_runtime(events)
    save_accuracy(parity)
    save_vllm_latency(parity)
    save_sentinel(parity)
    save_gpu_resources(cycles)

    representative = parity[(parity.backend == "levanter-gpu") | (parity.request_kind == "representative")]
    sentinel = parity[(parity.backend == "vllm-gpu") & (parity.request_kind == "sentinel")]
    batches = events[events.event == "levanter_batch"]
    vllm_requests = parity[(parity.backend == "vllm-gpu") & (parity.request_kind == "representative")]
    event_counts = {
        backend: dict(Counter(events[events.backend == backend].event)) for backend in BACKEND_LABELS
    }
    summary = {
        "event_counts": event_counts,
        "pytest_total_seconds": float(junit.getroot().find("testsuite").attrib["time"]),
        "pytest_test_seconds": pytest_seconds,
        "task_seconds": TASK_SECONDS,
        "remote_seconds": {
            backend: float(
                events[(events.backend == backend) & (events.event == "remote_complete")].iloc[0].total_seconds
            )
            for backend in BACKEND_LABELS
        },
        "runtime_breakdown_seconds": breakdown,
        "prompt_count": 64,
        "total_real_prompt_tokens": int(batches.real_tokens.sum()),
        "levanter_total_padded_tokens": int(batches.padded_tokens.sum()),
        "levanter_scoring_seconds": float(batches.batch_seconds.sum()),
        "vllm_scoring_seconds_including_sentinel": float(
            events[events.event == "vllm_wave"].wave_seconds.sum()
            + events[events.event == "vllm_sentinel_wave"].wave_seconds.sum()
        ),
        "accuracy": {
            backend: {
                "count": int(len(rows)),
                "max_probability_error": float(rows.max_probability_error.max()),
                "max_error_case": str(rows.loc[rows.max_probability_error.idxmax()].case_id),
                "p50_probability_error": float(rows.max_probability_error.quantile(0.5)),
                "p95_probability_error": float(rows.max_probability_error.quantile(0.95)),
                "winner_changes": int((rows.golden_probability_gap_to_greedy > 0).sum()),
            }
            for backend, rows in representative.groupby("backend")
        },
        "sentinel": {
            "count": int(len(sentinel)),
            "max_probability_error": float(sentinel.max_probability_error.max()),
            "min_probability_error": float(sentinel.max_probability_error.min()),
            "mean_probability_error": float(sentinel.max_probability_error.mean()),
            "max_error_rank": int(sentinel.loc[sentinel.max_probability_error.idxmax()]["rank"]),
            "request_seconds_min": float(sentinel.request_seconds.min()),
            "request_seconds_max": float(sentinel.request_seconds.max()),
        },
        "vllm_representative_request_seconds": {
            "min": float(vllm_requests.request_seconds.min()),
            "p50": float(vllm_requests.request_seconds.quantile(0.5)),
            "p95": float(vllm_requests.request_seconds.quantile(0.95)),
            "max": float(vllm_requests.request_seconds.max()),
        },
        "gpu": {
            backend: {
                "raw_samples": int(len(gpu[gpu.backend == backend])),
                "complete_cycles": int(len(rows)),
                "peak_hbm_mib": float(gpu[gpu.backend == backend].memory_used_mib.max()),
                "peak_gpu_utilization_percent": float(gpu[gpu.backend == backend].gpu_utilization_percent.max()),
                "peak_power_watts": float(gpu[gpu.backend == backend].power_watts.max()),
                "peak_temperature_c": float(gpu[gpu.backend == backend].temperature_c.max()),
                "telemetry_emission_span_p50_seconds": float(rows.emission_span_seconds.quantile(0.5)),
                "telemetry_emission_span_max_seconds": float(rows.emission_span_seconds.max()),
            }
            for backend, rows in cycles.groupby("backend")
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
