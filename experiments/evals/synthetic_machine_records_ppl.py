# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed synthetic machine-record PPL validation slices.

These slices are supervised target-only probes for machine-generated records:
logs, events, stack traces, config files, and manifests. The prompt is plain
base-model continuation context; only the held-out target continuation should
contribute to loss.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
SYNTHETIC_MACHINE_RECORDS_ISSUE = 5618
SYNTHETIC_MACHINE_RECORDS_HF_DATASET_ID = "marin-community/synth-machine-records-ppl"
SYNTHETIC_MACHINE_RECORDS_SOURCE = "generated_synthetic_machine_records_ppl_v1"
SYNTHETIC_MACHINE_RECORDS_HF_REVISION = "5efd713c2781e0a1a82dd850acd7d8de14f81748"
SYNTHETIC_MACHINE_RECORDS_SEED = 20260511
EXAMPLES_PER_CONFIG = 1000


class MachineRecordFamily(StrEnum):
    SERVICE_LOGS = "service_logs"
    TRACE_ERRORS = "trace_errors"
    CI_LOGS = "ci_logs"
    STRUCTURED_LOGS = "structured_logs"
    CONFIG_MANIFESTS = "config_manifests"


@dataclass(frozen=True)
class MachineRecordPplSlice:
    family: MachineRecordFamily
    task_name: str
    hf_config_name: str
    subset: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_machine_records_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_machine_records_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{SYNTHETIC_MACHINE_RECORDS_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"subset:{self.subset}",
            f"seed:{SYNTHETIC_MACHINE_RECORDS_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{SYNTHETIC_MACHINE_RECORDS_SOURCE}",
            f"hf_revision:{SYNTHETIC_MACHINE_RECORDS_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(
                id=hf_dataset_id,
                name=self.hf_config_name,
                revision=SYNTHETIC_MACHINE_RECORDS_HF_REVISION,
            ),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


MACHINE_RECORD_PPL_SLICES: tuple[MachineRecordPplSlice, ...] = (
    MachineRecordPplSlice(
        family=MachineRecordFamily.SERVICE_LOGS,
        task_name="zeek_conn_http_dns",
        hf_config_name="service_logs_zeek_conn_http_dns",
        subset="zeek",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.SERVICE_LOGS,
        task_name="nginx_access_error",
        hf_config_name="service_logs_nginx_access_error",
        subset="nginx",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.SERVICE_LOGS,
        task_name="k8s_system_events",
        hf_config_name="service_logs_k8s_system_events",
        subset="k8s",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.SERVICE_LOGS,
        task_name="systemd_journal",
        hf_config_name="service_logs_systemd_journal",
        subset="system",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.TRACE_ERRORS,
        task_name="python_tracebacks",
        hf_config_name="trace_errors_python_tracebacks",
        subset="python_traceback",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.TRACE_ERRORS,
        task_name="compiler_errors",
        hf_config_name="trace_errors_compiler_errors",
        subset="compiler",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CI_LOGS,
        task_name="github_actions_jobs",
        hf_config_name="ci_logs_github_actions_jobs",
        subset="github_actions",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.STRUCTURED_LOGS,
        task_name="json_application_logs",
        hf_config_name="structured_logs_json_application_logs",
        subset="json_logs",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="package_json",
        hf_config_name="config_manifests_package_json",
        subset="package_json",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="pyproject_toml",
        hf_config_name="config_manifests_pyproject_toml",
        subset="pyproject",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="github_actions_yaml",
        hf_config_name="config_manifests_github_actions_yaml",
        subset="github_actions_yaml",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="dockerfile",
        hf_config_name="config_manifests_dockerfile",
        subset="dockerfile",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="terraform_hcl",
        hf_config_name="config_manifests_terraform_hcl",
        subset="terraform",
    ),
    MachineRecordPplSlice(
        family=MachineRecordFamily.CONFIG_MANIFESTS,
        task_name="kubernetes_yaml",
        hf_config_name="config_manifests_kubernetes_yaml",
        subset="kubernetes_yaml",
    ),
)


def synthetic_machine_records_raw_validation_sets(
    *,
    hf_dataset_id: str = SYNTHETIC_MACHINE_RECORDS_HF_DATASET_ID,
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in MACHINE_RECORD_PPL_SLICES
    }


Record = dict[str, Any]
RecordGenerator = Callable[[MachineRecordPplSlice, int, random.Random], tuple[str, str, dict[str, Any]]]


def _base_prompt(subset: str, task: str, context: str) -> str:
    return f"Machine record subset: {subset}\nTask: continue the {task} record exactly in the same format.\n\n{context}"


def _choice(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def _service_log_record(
    slice_: MachineRecordPplSlice, index: int, rng: random.Random
) -> tuple[str, str, dict[str, Any]]:
    minute = 10 + index
    host = f"node-{rng.randint(1, 9):02d}"
    pod = f"api-{rng.randrange(1000, 9999)}"
    if slice_.subset == "zeek":
        uid = f"C{rng.randrange(10**8, 10**9)}"
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            "#separator \\x09\n#set_separator\t,\n#fields\tts\tuid\tid.orig_h\tid.resp_h\tproto\tservice\tduration\n"
            f"171543{minute}.120000\t{uid}\t10.4.1.{rng.randint(2, 240)}\t172.16.8.{rng.randint(2, 240)}\t",
        )
        target = f"tcp\thttp\t0.{rng.randint(100000, 999999)}\n"
    elif slice_.subset == "nginx":
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            f'10.12.4.{rng.randint(2, 240)} - - [11/May/2026:14:{minute}:02 +0000] "GET /v1/',
        )
        target = f'healthz HTTP/1.1" 200 {rng.randint(42, 512)} "-" "kube-probe/1.30"\n'
    elif slice_.subset == "k8s":
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            f"{minute}m Normal Pulled pod/{pod} Successfully pulled image ",
        )
        target = f'"registry.local/{pod}:sha-{rng.randrange(16**8):08x}" in {rng.randint(1, 9)}.{rng.randint(10, 99)}s\n'
    else:
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            f"May 11 14:{minute}:17 {host} systemd[1]: ",
        )
        target = f"Started {_choice(rng, ('container runtime', 'session cleanup', 'log rotation'))} service.\n"
    return input_text, target, {"host": host, "minute": minute}


def _trace_error_record(
    slice_: MachineRecordPplSlice, index: int, rng: random.Random
) -> tuple[str, str, dict[str, Any]]:
    module = _choice(rng, ("loader", "scheduler", "serializer", "worker"))
    if slice_.subset == "python_traceback":
        line = rng.randint(40, 240)
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            "Traceback (most recent call last):\n"
            f'  File "/srv/app/{module}.py", line {line}, in run\n'
            '    result = handler(payload["items"])\n'
            f'  File "/srv/app/{module}.py", line {line + 18}, in handler\n'
            '    return items[0]["',
        )
        missing_key = _choice(rng, ("status", "checksum", "tenant_id"))
        target = f"{missing_key}\"]\nKeyError: '{missing_key}'\n"
        return input_text, target, {"module": module, "line": line}

    file_name = f"src/{module}.rs"
    line = rng.randint(8, 160)
    input_text = _base_prompt(
        slice_.subset,
        slice_.task_name,
        f"error[E0308]: mismatched types\n  --> {file_name}:{line}:",
    )
    returned = _choice(rng, ("None", "0", "payload"))
    target = (
        f"{rng.randint(5, 80)}\n"
        "   |\n"
        f"{line} |     return {returned};\n"
        "   |            ^^^^^ expected `Result<Job, Error>`, found `Option<_>`\n"
    )
    return input_text, target, {"file": file_name, "line": line}


def _ci_log_record(slice_: MachineRecordPplSlice, index: int, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    step = _choice(rng, ("Install dependencies", "Run unit tests", "Build image", "Upload coverage"))
    input_text = _base_prompt(
        slice_.subset,
        slice_.task_name,
        f"2026-05-11T14:{20 + index:02d}:00.0000000Z ##[group]{step}\n"
        "2026-05-11T14:20:01.0000000Z shell: /usr/bin/bash -e {0}\n"
        "2026-05-11T14:20:02.0000000Z ",
    )
    target = f"##[command]{_choice(rng, ('uv run pytest tests', 'docker build -t app:${{ github.sha }} .', 'npm ci'))}\n"
    return input_text, target, {"ci_step": step}


def _json_log_record(slice_: MachineRecordPplSlice, index: int, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    request_id = f"req-{rng.randrange(16**10):010x}"
    input_text = _base_prompt(
        slice_.subset,
        slice_.task_name,
        '{"ts":"2026-05-11T14:32:',
    )
    target_obj = {
        "level": _choice(rng, ("INFO", "WARN", "ERROR")),
        "service": _choice(rng, ("api", "billing", "worker")),
        "request_id": request_id,
        "latency_ms": rng.randint(12, 900),
        "message": "request completed",
    }
    target = f'{10 + index:02d}Z",' + json.dumps(target_obj, sort_keys=True)[1:] + "\n"
    return input_text, target, {"request_id": request_id}


def _manifest_record(slice_: MachineRecordPplSlice, index: int, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    name = f"svc-{rng.randint(10, 99)}"
    if slice_.subset == "package_json":
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            '{\n  "name": "@example/service",\n  "version": "0.',
        )
        target = (
            f'{index}.0",\n'
            '  "scripts": {"test": "vitest run", "build": "tsc -p tsconfig.json"},\n'
            '  "dependencies": {"zod": "^3.23.8"}\n'
            "}\n"
        )
    elif slice_.subset == "pyproject":
        input_text = _base_prompt(
            slice_.subset,
            slice_.task_name,
            '[project]\nname = "machine-records"\nversion = "0.',
        )
        target = (
            f'{index}.0"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = ["pydantic>=2", "rich>=13"]\n\n'
            "[tool.pytest.ini_options]\n"
            'addopts = "-q"\n'
        )
    elif slice_.subset == "github_actions_yaml":
        input_text = _base_prompt(
            slice_.subset, slice_.task_name, "name: ci\non:\n  pull_request:\n\njobs:\n  test:\n    runs-on: "
        )
        target = (
            "ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: astral-sh/setup-uv@v5\n"
            "      - run: uv run pytest\n"
        )
    elif slice_.subset == "dockerfile":
        input_text = _base_prompt(
            slice_.subset, slice_.task_name, "FROM python:3.12-slim\nWORKDIR /app\nCOPY pyproject.toml uv.lock ./\nRUN "
        )
        target = 'pip install uv && uv sync --frozen --no-dev\nCOPY . .\nCMD ["python", "-m", "service.main"]\n'
    elif slice_.subset == "terraform":
        input_text = _base_prompt(slice_.subset, slice_.task_name, 'resource "aws_s3_bucket" "logs" {\n  bucket = "')
        target = f'{name}-logs"\n  force_destroy = false\n\n  tags = {{\n    Environment = "staging"\n  }}\n}}\n'
    else:
        input_text = _base_prompt(
            slice_.subset, slice_.task_name, "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: "
        )
        target = f"{name}\nspec:\n  replicas: {rng.randint(2, 5)}\n  selector:\n    matchLabels:\n      app: {name}\n"
    return input_text, target, {"name": name}


_GENERATORS_BY_FAMILY: dict[MachineRecordFamily, RecordGenerator] = {
    MachineRecordFamily.SERVICE_LOGS: _service_log_record,
    MachineRecordFamily.TRACE_ERRORS: _trace_error_record,
    MachineRecordFamily.CI_LOGS: _ci_log_record,
    MachineRecordFamily.STRUCTURED_LOGS: _json_log_record,
    MachineRecordFamily.CONFIG_MANIFESTS: _manifest_record,
}


def machine_record_example(
    slice_: MachineRecordPplSlice, index: int, *, seed: int = SYNTHETIC_MACHINE_RECORDS_SEED
) -> Record:
    rng = random.Random(seed + index * 1009 + sum(ord(ch) for ch in slice_.registry_key))
    input_text, target, metadata = _GENERATORS_BY_FAMILY[slice_.family](slice_, index, rng)
    return {
        "input": input_text,
        "target": target,
        "id": f"{slice_.hf_config_name}-{index:06d}",
        "subset": slice_.subset,
        "task": slice_.task_name,
        "seed": seed,
        "metadata": {
            "family": slice_.family.value,
            "hf_config_name": slice_.hf_config_name,
            **metadata,
        },
    }


def iter_machine_record_examples(
    *,
    examples_per_config: int,
    slices: Iterable[MachineRecordPplSlice] = MACHINE_RECORD_PPL_SLICES,
    seed: int = SYNTHETIC_MACHINE_RECORDS_SEED,
) -> Iterable[Record]:
    for slice_ in slices:
        for index in range(examples_per_config):
            yield machine_record_example(slice_, index, seed=seed)


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local synthetic machine-record PPL JSONL sample.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-config", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SYNTHETIC_MACHINE_RECORDS_SEED)
    args = parser.parse_args()

    write_jsonl(
        args.output,
        iter_machine_record_examples(examples_per_config=args.examples_per_config, seed=args.seed),
    )


if __name__ == "__main__":
    main()
