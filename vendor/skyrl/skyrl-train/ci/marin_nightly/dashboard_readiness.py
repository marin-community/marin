"""Report whether the run just finished can be read on the RL runs dashboard.

A run can train perfectly and still be invisible there, and nothing in the training log says so.
This is a report, not a gate: it exits zero whatever it finds, and prints one line the workflow
renders into the job summary.

    NIGHTLY_TELEMETRY_READINESS {"run_id": ..., "signals": [...], "missing": [...]}
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass

from connectrpc.errors import ConnectError
from finelog.client.log_client import LogClient
from finelog.deploy.config import load_finelog_config

# Rows land asynchronously, so the first query after a run routinely sees nothing.
POLL_ATTEMPTS = 6
POLL_INTERVAL = 20.0

MARKER = "NIGHTLY_TELEMETRY_READINESS"


@dataclass(frozen=True)
class Signal:
    """One telemetry signal, and the panel that goes blank without it.

    The engine's metrics and the trainer's share a service, told apart only by `metric_source`.
    """

    key: str
    panel: str
    name: str
    metric_source: str = ""
    work_kind: str = ""

    def matches(self, row: dict[str, object]) -> bool:
        return (
            row["name"] == self.name
            and row["metric_source"] == self.metric_source
            and (not self.work_kind or row["work_kind"] == self.work_kind)
        )


# The panels whose rows carry this run id. Rollout buffer and staleness belong to the asynchronous
# trainer; GPU utilization reads the node agent, which stamps no run id; the engine latency panel
# selects histogram families by suffix, so it has no single signal to check.
REQUIRED_SIGNALS = (
    Signal("policy_step", "Optimizer step and policy lag", "policy_step"),
    Signal("rollouts", "Rollouts, samples and tokens completed", "work_completed", work_kind="rollout"),
    Signal("samples", "Rollouts, samples and tokens completed", "work_completed", work_kind="sample"),
    Signal("generated_tokens", "Rollouts, samples and tokens completed", "work_completed", work_kind="generated_token"),
    Signal("phase_timing", "Rollout wait vs train step (critical path)", "phase_duration_seconds"),
    Signal("engine_throughput", "Engine token throughput", "generation_tokens_total", metric_source="vllm"),
    Signal("engine_queue", "Engine queue and KV-cache", "num_requests_running", metric_source="vllm"),
    Signal("engine_cache", "Prefix-cache hit rate", "prefix_cache_queries_total", metric_source="vllm"),
    Signal("engine_outcomes", "Why engine requests finished", "request_success_total", metric_source="vllm"),
    Signal("ray_object_store", "Ray object store occupancy", "ray_object_store_used_memory", metric_source="ray"),
    Signal("ray_spill", "Ray spill manager bytes by state", "ray_spill_manager_objects_bytes", metric_source="ray"),
)

_QUERY = """
SELECT name,
       COALESCE(json_get(attributes_json, 'metric_source'), '') AS metric_source,
       COALESCE(json_get(attributes_json, 'work_kind'), '') AS work_kind,
       COUNT(*) AS records
FROM "telemetry_v1.marinskyrl"
WHERE run_id = '{run_id}'
GROUP BY 1, 2, 3
"""


@dataclass(frozen=True)
class SignalStatus:
    """How many records one signal contributed, and which panel reads it."""

    key: str
    panel: str
    records: int


@dataclass(frozen=True)
class Readiness:
    """What a run published, per signal the dashboard reads."""

    run_id: str
    signals: tuple[SignalStatus, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(signal.key for signal in self.signals if signal.records == 0)

    def as_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "signals": [
                {"key": signal.key, "panel": signal.panel, "records": signal.records} for signal in self.signals
            ],
            "missing": list(self.missing),
        }


def readiness(run_id: str, rows: list[dict[str, object]]) -> Readiness:
    """Return the per-signal record counts for one run's rows."""
    return Readiness(
        run_id=run_id,
        signals=tuple(
            SignalStatus(
                key=signal.key,
                panel=signal.panel,
                records=sum(int(row["records"]) for row in rows if signal.matches(row)),
            )
            for signal in REQUIRED_SIGNALS
        ),
    )


def report_line(readiness_or_error: "Readiness | dict[str, object]") -> str:
    """Render the single line the workflow parses out of the streamed job log.

    One line, because it is recovered by scanning a log whose every line carries an Iris prefix.
    """
    payload = readiness_or_error.as_payload() if isinstance(readiness_or_error, Readiness) else readiness_or_error
    return f"{MARKER} {json.dumps(payload)}"


def _rows_for(client: LogClient, run_id: str) -> list[dict[str, object]]:
    if "'" in run_id:
        raise ValueError(f"run id cannot contain an apostrophe: {run_id!r}")
    table = client.query(_QUERY.format(run_id=run_id), max_rows=10_000)
    return table.to_pylist()


def collect(
    client: LogClient, run_id: str, *, attempts: int = POLL_ATTEMPTS, interval: float = POLL_INTERVAL
) -> Readiness:
    """Poll until every required signal has arrived, or the attempts run out."""
    result = readiness(run_id, [])
    for attempt in range(attempts):
        if attempt:
            time.sleep(interval)
        result = readiness(run_id, _rows_for(client, run_id))
        if not result.missing:
            return result
    return result


def _hand_query(run_id: str) -> str:
    return (
        "cd <marin checkout> && uv run --no-sync finelog query marin --format table <<'SQL'\n"
        f"{_QUERY.format(run_id=run_id).strip()}\nSQL"
    )


def main() -> int:
    try:
        from finelog.deploy.connect import open_client
    except ImportError as error:
        raise RuntimeError("Dashboard reporting requires Marin's Finelog environment") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="the run identity telemetry rows join on")
    args = parser.parse_args()
    try:
        with open_client(load_finelog_config("marin"), "marin") as client:
            result = collect(client, args.run_id)
    except (ConnectError, ConnectionError, TimeoutError, OSError, ValueError) as error:
        # An unreachable log server says nothing about the run. Print the query instead.
        print(report_line({"run_id": args.run_id, "error": str(error)}))
        print(f"::: telemetry readiness: could not reach the log server ({error})")
        print(_hand_query(args.run_id))
        return 0

    print(report_line(result.as_payload()))
    print(f"::: telemetry readiness for {result.run_id}")
    for signal in result.signals:
        state = "ok" if signal.records else "MISSING"
        print(f":::   {state:>7}  {signal.records:>7} records  {signal.panel}")
    if result.missing:
        print(f"::: {len(result.missing)} panel(s) will be empty; the run itself is unaffected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
