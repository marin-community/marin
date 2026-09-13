# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify stage-2 native requests on regional CPUs without submitting training."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import fsspec
from marin.rl.skyrl import _launcher_command, sanitize_job_name


def read_bounded(uri: str, maximum: int) -> bytes:
    filesystem, path = fsspec.core.url_to_fs(uri, config_kwargs={"s3": {"addressing_style": "virtual"}})
    before = filesystem.info(path)
    assert 0 < int(before["size"]) <= maximum, uri
    payload = filesystem.cat(path)
    assert len(payload) == int(before["size"]) and filesystem.info(path) == before, uri
    return payload


def persist_new(uri: str, payload: bytes) -> None:
    filesystem, path = fsspec.core.url_to_fs(uri, config_kwargs={"s3": {"addressing_style": "virtual"}})
    assert not filesystem.exists(path), uri
    with filesystem.open(path, "wb") as stream:
        stream.write(payload)
    assert read_bounded(uri, len(payload)) == payload


def check_process(result: subprocess.CompletedProcess, uri: str, arm: str, phase: str) -> None:
    if result.returncode == 0:
        return
    stderr = result.stderr
    for key, value in os.environ.items():
        if value and any(term in key.upper() for term in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY")):
            stderr = stderr.replace(value, "<redacted>")
    failure = {
        "arm": arm,
        "phase": phase,
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_first_16k": stderr[:16384],
        "stderr_last_4k": stderr[-4096:],
    }
    persist_new(uri, json.dumps(failure, sort_keys=True).encode())
    raise RuntimeError((arm, phase, "bounded initiating error retained", uri))


def prepare(packets: list[dict], output_prefix: str) -> None:
    """Run installed native composition/serialization and retain exact receipts."""
    assert output_prefix.startswith("s3://marin-us-east-02a/")
    assert len(packets) == 5 and len({packet["arm"] for packet in packets}) == 5
    reports = []
    for packet in packets:
        request = packet["request"]
        canonical = {key: value for key, value in request.items() if key != "attempt_id"}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert digest == packet["request_hash"] and request["attempt_id"] == digest[:12]
        assert packet["msr"] == request["runtime"]["commit"]
        assert packet["gpu_released"] is False and packet["native_K16_gate"] is None
        uri = output_prefix + "/" + packet["arm"] + ".json"
        filesystem, path = fsspec.core.url_to_fs(uri)
        assert not filesystem.exists(path), uri
        for input_uri, expected in packet["input_objects"].items():
            payload = read_bounded(input_uri, 1048576)
            assert len(payload) == expected["bytes"] and hashlib.sha256(payload).hexdigest() == expected["sha256"]
        envelope = {
            "schema_version": 2,
            "request": request,
            "execution": {
                **packet["execution"],
                "job_name": sanitize_job_name(request["run_id"] + "-" + request["attempt_id"]),
            },
        }
        requirement = "marinskyrl @ git+https://github.com/marin-community/MarinSkyRL.git@" + packet["msr"]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            envelope_path = temporary / "envelope.json"
            envelope_path.write_text(json.dumps(envelope))
            packet_path = temporary / "packet.json"
            packet_path.write_text(json.dumps(packet))
            command = _launcher_command(requirement, str(envelope_path))
            prefix = command[: command.index("marinskyrl")]
            assert requirement in prefix and prefix[:2] == ["uv", "run"]
            result = subprocess.run([*command, "--dry-run"], capture_output=True, text=True)
            check_process(result, output_prefix + "/" + packet["arm"] + "-failure.json", packet["arm"], "native_dry_run")
            checks = {
                "native_dry_run_output_sha256": hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest()
            }
            for helper, name in [
                ("e61_stage2_native_preview", "compose"),
                ("e61_stage2_native_serialization", "serialization"),
            ]:
                target = temporary / (name + ".json")
                result = subprocess.run(
                    [
                        *prefix,
                        "python",
                        "-m",
                        "cloud.iris." + helper,
                        "--packet",
                        str(packet_path),
                        "--output",
                        str(target),
                    ],
                    capture_output=True,
                    text=True,
                )
                check_process(result, output_prefix + "/" + packet["arm"] + "-failure.json", packet["arm"], name)
                payload = target.read_bytes()
                assert len(payload) <= 1048576
                checks[name] = json.loads(payload)
                assert checks[name]["request_hash"] == packet["request_hash"]
            report = {
                "status": "E61_STAGE2_REGIONAL_NATIVE_CPU_PASS",
                "request_hash": packet["request_hash"],
                "msr": packet["msr"],
                "arm": packet["arm"],
                "checks": checks,
                "input_objects": packet["input_objects"],
                "gpu_released": False,
                "native_K16_gate": None,
            }
            payload = json.dumps(report, sort_keys=True).encode()
            assert len(payload) <= 1048576
            persist_new(uri, payload)
            reports.append({"uri": uri, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            print("E61_STAGE2_REGIONAL_NATIVE_CPU_PASS", packet["arm"], flush=True)
    persist_new(
        output_prefix + "/receipt.json", json.dumps({"reports": reports, "gpu_released": False}, sort_keys=True).encode()
    )


def validate_destination_revision(previous: dict, packet: dict) -> None:
    """Accept only the two explicit owned destinations on an already composed request."""
    old, new = previous["request"], packet["request"]
    attempts = new["output"]["attempts_root"]
    additions = [
        "++generator.trajectory_retention.output_path=" + json.dumps(attempts + "/trajectories"),
        "++terminal_bench_config.trials_dir=" + json.dumps(attempts + "/trace_jobs"),
    ]
    assert new["overrides"] == old["overrides"] + additions
    assert {k: v for k, v in old.items() if k not in ("overrides", "attempt_id")} == {
        k: v for k, v in new.items() if k not in ("overrides", "attempt_id")
    }
    canonical = {key: value for key, value in new.items() if key != "attempt_id"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert packet["request_hash"] == digest and new["attempt_id"] == digest[:12]
    assert packet["input_objects"] == previous["input_objects"]
    assert packet["msr"] == previous["msr"] == new["runtime"]["commit"]


def qualify_owned_destinations(revisions: list[dict], output_prefix: str) -> None:
    """Recheck native output destinations without rerunning input or endpoint tests."""
    assert output_prefix.startswith("s3://marin-us-east-02a/")
    assert len(revisions) == 5 and len({r["packet"]["arm"] for r in revisions}) == 5
    summaries = []
    for revision in revisions:
        packet, previous = revision["packet"], revision["previous"]
        validate_destination_revision(previous, packet)
        proof = revision["previous_receipt"]
        prior_bytes = read_bounded(proof["uri"], 1048576)
        assert hashlib.sha256(prior_bytes).hexdigest() == proof["sha256"] and len(prior_bytes) == proof["bytes"]
        prior = json.loads(prior_bytes)
        assert prior["request_hash"] == previous["request_hash"] and prior["msr"] == packet["msr"]
        assert prior["status"] == "E61_STAGE2_REGIONAL_NATIVE_CPU_PASS"
        uri = output_prefix + "/" + packet["arm"] + ".json"
        requirement = "marinskyrl @ git+https://github.com/marin-community/MarinSkyRL.git@" + packet["msr"]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            packet_path, target = temporary / "packet.json", temporary / "compose.json"
            packet_path.write_text(json.dumps(packet))
            command = _launcher_command(requirement, str(packet_path))
            prefix = command[: command.index("marinskyrl")]
            assert requirement in prefix
            result = subprocess.run(
                [
                    *prefix,
                    "python",
                    "-m",
                    "cloud.iris.e61_stage2_native_preview",
                    "--packet",
                    str(packet_path),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            check_process(result, uri.replace(".json", "-failure.json"), packet["arm"], "owned_destinations")
            composed = json.loads(target.read_bytes())
            assert composed["request_hash"] == packet["request_hash"] and composed["runtime"] == packet["msr"]
            config = composed["composed"]
            attempts = packet["request"]["output"]["attempts_root"]
            assert config["generator"]["trajectory_retention"]["output_path"] == attempts + "/trajectories"
            assert config["terminal_bench_config"]["trials_dir"] == attempts + "/trace_jobs"
            report = {
                "status": "E61_STAGE2_OWNED_DESTINATIONS_PASS",
                "arm": packet["arm"],
                "request_hash": packet["request_hash"],
                "msr": packet["msr"],
                "previous_receipt": proof,
                "compose": composed,
                "gpu_released": False,
            }
            payload = json.dumps(report, sort_keys=True).encode()
            assert len(payload) <= 1048576
            persist_new(uri, payload)
            summaries.append({"uri": uri, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            print("E61_STAGE2_OWNED_DESTINATIONS_PASS", packet["arm"], flush=True)
    persist_new(output_prefix + "/receipt.json", json.dumps({"reports": summaries}, sort_keys=True).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    prepare(json.loads(args.packets.read_bytes()), args.output_prefix)
