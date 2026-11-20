# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise TTL publication and restore through an external preparation CLI fixture.

Native AgentX format compatibility is checked in its isolated pinned environment.
These tests cover the storage and process boundary without adding that dependency.
"""

import json
import os
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from marin.profiling.trace_summary import sha256_for_path
from tokenizers import Tokenizer, models, pre_tokenizers

from experiments.evaluation.knotmeter.artifacts import pack_server_metrics, restore_server_metrics
from experiments.evaluation.knotmeter.cli import main as run_cli
from experiments.evaluation.knotmeter.collect import process_point, validate_window
from experiments.evaluation.knotmeter.config import Stage, load_profile, resolved_series
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import benchmark_command, workload_environment
from experiments.evaluation.knotmeter.workload import prepare_workload


def test_workload_restores_without_preparation_and_rebuilds_expired_or_changed_inputs(tmp_path, monkeypatch):
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    producer = bin_directory / "uvx"
    fixture = Path(__file__).with_name("fixtures") / "prepare.py"
    producer.write_text(f"#!{sys.executable}\n" + fixture.read_text())
    producer.chmod(0o755)
    events = tmp_path / "events"
    monkeypatch.setenv("PATH", f"{bin_directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGENTX_FIXTURE_EVENTS", str(events))
    series = resolved_series(load_profile(Path("snowball-h100"), []), Stage.CONFIRM)
    series_path = tmp_path / "series.yaml"
    series_path.write_text(json.dumps(series))
    store = str(tmp_path / "store")

    first = prepare_workload(tmp_path / "first", series_path=series_path, cache_root=store)
    pointer = Path(first["archive_uri"]).parent / "workload.json"
    for path in (Path(first["archive_uri"]), pointer):
        os.utime(path, (1, 1))
    restored = prepare_workload(tmp_path / "restored", series_path=series_path, cache_root=store)
    assert restored["cache_status"] == "hit"
    assert restored["archive_sha256"] == first["archive_sha256"]
    assert (tmp_path / "restored/bundle/native/dataset.dat").read_bytes() == b"42" * 100
    assert events.read_text().splitlines().count("build") == 1
    assert Path(first["archive_uri"]).stat().st_mtime > 1
    assert pointer.stat().st_mtime > 1

    # Serving and measurement updates reuse derived data.
    series.update({"duration": 900, "gpu_memory_utilization": 0.9})
    series_path.write_text(json.dumps(series))
    measurement = prepare_workload(tmp_path / "measurement", series_path=series_path, cache_root=store)
    assert measurement["archive_uri"] == first["archive_uri"]
    assert measurement["cache_status"] == "hit"

    # TTL can expire data before its pointer. Only this fixture's exact blob is removed.
    Path(first["archive_uri"]).unlink()
    rebuilt = prepare_workload(tmp_path / "rebuilt", series_path=series_path, cache_root=store)
    assert rebuilt["cache_status"] == "built"
    assert rebuilt["key"] == first["key"]
    assert (tmp_path / "rebuilt/bundle/native/dataset.dat").read_bytes() == b"42" * 100
    assert events.read_text().splitlines().count("build") == 2

    series["random_seed"] = 43
    series_path.write_text(json.dumps(series))
    changed = prepare_workload(tmp_path / "changed", series_path=series_path, cache_root=store)
    assert changed["key"] != first["key"]
    assert (tmp_path / "changed/bundle/native/dataset.dat").read_bytes() == b"43" * 100
    assert Path(rebuilt["archive_uri"]).exists()


@dataclass(frozen=True)
class ReplayServer:
    url: str
    sessions: list[tuple[str | None, str | None]]
    scrapes: list[str]


@pytest.fixture
def replay_server():
    sessions = []
    scrapes = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                scrapes.append(self.path)
                body = b"# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 1\n"
            else:
                body = b'{"data": [{"id": "tiny"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sessions.append((self.headers.get("X-Session-ID"), self.headers.get("X-Correlation-ID")))
            count = request["max_completion_tokens"]
            chunks = [
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": " word"}, "finish_reason": None}],
                }
                for _ in range(count)
            ]
            chunks.append(
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 64, "completion_tokens": count, "total_tokens": 64 + count},
                }
            )
            body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield ReplayServer(f"http://127.0.0.1:{server.server_port}", sessions, scrapes)
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_pinned_tools_prepare_restore_replay_and_normalize_physical_gpus(tmp_path, replay_server, monkeypatch):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0} | {f"word{i}": i + 1 for i in range(512)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", "model_max_length": 2048})
    )
    corpus = tmp_path / "trace.json"
    corpus.write_text(
        json.dumps(
            {
                "id": "local-session",
                "models": ["tiny"],
                "block_size": 16,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": i * 0.01,
                        "type": "n",
                        "model": "tiny",
                        "in": 64 + 16 * i,
                        "out": 4,
                        "hash_ids": list(range(4 + i)),
                        "api_time": 0.01,
                    }
                    for i in range(20)
                ],
            }
        )
    )
    profile = load_profile(Path("snowball-h100"), []).model_dump()
    profile["model"] |= {"repo": "tiny", "tokenizer": str(tokenizer_dir), "tokenizer_revision": "local"}
    profile["serving"]["max_model_len"] = 2048
    profile["checks"] = []
    preset = tmp_path / "profile.json"
    preset.write_text(json.dumps(profile))
    plan_path = tmp_path / "plan.json"
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", lambda self, repo, *, revision: SimpleNamespace(sha=revision)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "knotmeter",
            "plan",
            "--profile",
            str(preset),
            "--campaign",
            "compat",
            "--stage",
            Stage.SCREEN.value,
            "--session-concurrency",
            "17",
            "--concurrency",
            "2",
            "--duration",
            "3",
            "--warmup-requests-per-lane",
            "2",
            "--unsafe-override",
            "--output",
            str(plan_path),
        ],
    )
    run_cli()
    plan = json.loads(plan_path.read_text())
    series = plan["series"] | {
        "dataset": str(corpus),
        "dataset_loader": "weka_trace",
        "sessions": 1,
        "dataset_sha256": sha256_for_path(corpus),
        "tokenizer": str(tokenizer_dir),
        "tokenizer_revision": "local",
        "max_context": 2048,
    }
    series_path = tmp_path / "local.yaml"
    series_path.write_text(json.dumps(series))
    first = prepare_workload(tmp_path / "first", series_path=series_path, cache_root=str(tmp_path / "cache"))
    restored = prepare_workload(tmp_path / "restored", series_path=series_path, cache_root=str(tmp_path / "cache"))
    assert first["cache_status"] == "built"
    assert restored["cache_status"] == "hit"
    assert restored["archive_sha256"] == first["archive_sha256"]
    assert restored["lineage"]["turns"] >= 20
    point = tmp_path / "point"
    point.mkdir()
    url = replay_server.url
    # Short replay proves compatibility, and deliberately cannot qualify as a comparison window.
    command = benchmark_command(series, series["concurrency"], url, point / "aiperf_artifacts")
    run_logged(
        command,
        workload_environment(tmp_path / "restored/bundle", series),
        point / "agentx.log",
        timeout=120,
        sample=point / "client-load.jsonl",
    )
    export = json.loads((point / "aiperf_artifacts/profile_export_aiperf.json").read_text())
    phase = export["input_config"]["phases"][0]
    assert (phase["concurrency"], phase["duration"], phase["warmup_requests_per_lane"]) == (2, 3, 2)
    assert plan["series"]["measurement_requested"]["target_concurrency"] == 17
    assert json.loads((point / "client-load.jsonl").read_text().splitlines()[0])["available_cpus"] > 0
    requests = [json.loads(row) for row in (point / "aiperf_artifacts/profile_export.jsonl").read_text().splitlines()]
    assert any(row["metadata"]["benchmark_phase"] == "profiling" and not row.get("error") for row in requests)
    assert len(replay_server.sessions) > 2 and all(
        session and session == correlation for session, correlation in replay_server.sessions
    )
    assert len({session for session, _ in replay_server.sessions}) < len(replay_server.sessions)
    assert replay_server.scrapes
    metrics = point / "aiperf_artifacts/server_metrics_export.json"
    digest = sha256_for_path(metrics)
    pack_server_metrics(point)
    restore_server_metrics(point)
    assert sha256_for_path(metrics) == digest
    validity = validate_window(
        point,
        {"series": series | {"duration": 3}, "diagnostic": True},
        {"worker_metrics_url": url + "/metrics"},
    )
    assert validity["profiling_successes"] > 0
    assert not validity["coverage_complete"] and not validity["protocol_valid"]
    (point / "attempt.json").write_text("{}")
    (point / "hardware.txt").write_text("CPU fixture; physical GPU counts are processor inputs only.\n")
    outputs = []
    for profile, count in (("h100", 8), ("gb200", 16)):
        artifacts = tmp_path / profile
        process_point(
            point,
            {
                "profile": profile,
                "series": (
                    series
                    | {
                        "profile": (
                            load_profile(Path("snowball-h100" if count == 8 else "hero-gb200"), []).model_dump(
                                mode="json"
                            )
                        )
                    }
                ),
                "concurrency": 1,
                "vllm_release": {"version": "fixture", "source_commit": "0" * 40},
                "recipe_fingerprint": "fixture",
                "source": {"kind": "fixture"},
            },
            artifacts,
        )
        result = json.loads((artifacts / "agg.json").read_text())
        assert result["num_gpus"] == count
        rate = result["request_metrics"]["throughput"]
        assert rate["per_gpu"]["output_tput_tps"] > 0
        assert rate["per_gpu"]["output_tput_tps"] * count == pytest.approx(
            rate["output"]["tokens_per_second"], abs=0.001
        )
        outputs.append(rate["per_gpu"]["output_tput_tps"])
    assert outputs[0] == pytest.approx(2 * outputs[1], abs=0.001)
