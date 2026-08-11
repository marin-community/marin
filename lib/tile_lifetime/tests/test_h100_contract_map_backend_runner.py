# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import zlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from jax._src import compilation_cache as pinned_jax_compilation_cache
from ml_dtypes import bfloat16

import tile_lifetime.contract_map_backend as contract_map_backend
import tile_lifetime.contract_map_backend_resources as resources
import tile_lifetime.cuda_toolchain as cuda_toolchain
from tile_lifetime.contract_map_backend_resources import ContractMapCompilePlan, PtxasKernelResources
from tile_lifetime.h100_contract_map_benchmark import (
    ArchitectureStatus,
    BackendVariant,
    MeasurementBoundary,
    NumericalFloorError,
    default_h100_contract_map_benchmark_plan,
    validate_backend_numerical_evidence,
)

runner = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner")


def test_output_numerical_evidence_counts_nonfinite_positions_once_and_keeps_finite_metrics() -> None:
    nonfinite = np.asarray([1.0, np.inf], dtype=bfloat16)
    finite = np.asarray([1.0, 2.0], dtype=bfloat16)
    repeats = ((nonfinite,), (finite,), (finite,))

    evidence = runner._output_numerical_evidence(0, repeats, np.asarray([1.0, 2.0], dtype=np.float64))

    assert evidence["nonfinite_values"] == 1
    assert evidence["maximum_absolute_error"] == 0.0
    assert evidence["mean_absolute_error"] == 0.0
    assert evidence["maximum_ulp_distance"] == 0


def test_output_numerical_evidence_rejects_shape_mismatch_before_broadcasting() -> None:
    observed = np.zeros(2, dtype=bfloat16)
    repeats = ((observed,), (observed,), (observed,))

    with pytest.raises(ValueError, match="identical shapes"):
        runner._output_numerical_evidence(0, repeats, np.zeros((1, 2), dtype=np.float64))


@pytest.mark.parametrize("bad_repeat", (1, 2))
@pytest.mark.parametrize("bad_value", (np.nan, np.inf))
def test_output_numerical_evidence_counts_nonfinite_values_in_every_repeat(bad_repeat: int, bad_value: float) -> None:
    finite = np.asarray([1.0, 2.0], dtype=bfloat16)
    nonfinite = np.asarray([1.0, bad_value], dtype=bfloat16)
    repeats = [finite.copy() for _ in range(3)]
    repeats[bad_repeat] = nonfinite

    evidence = runner._output_numerical_evidence(
        0, tuple((repeat,) for repeat in repeats), np.asarray([1.0, 2.0], dtype=np.float64)
    )

    assert evidence["nonfinite_values"] == 1
    outputs = {role: evidence for role in ("forward", "dx", "dw0", "dw1")}
    case = default_h100_contract_map_benchmark_plan().cases[0]
    with pytest.raises(ValueError, match="metric=nonfinite_values"):
        validate_backend_numerical_evidence(
            BackendVariant.ORDINARY_XLA,
            outputs,
            case_id=case.case_id,
            measurement_boundary=MeasurementBoundary.LOGICAL_TRAINING_STEP,
        )


@pytest.mark.parametrize("bad_repeat", (1, 2))
def test_output_numerical_evidence_rejects_non_bfloat16_later_repeat(bad_repeat: int) -> None:
    repeats = [np.asarray([1.0, 2.0], dtype=bfloat16) for _ in range(3)]
    repeats[bad_repeat] = np.asarray([1.0, 2.0], dtype=np.float32)

    with pytest.raises(TypeError, match=rf"repeat {bad_repeat} must have BF16 dtype"):
        runner._output_numerical_evidence(
            0, tuple((repeat,) for repeat in repeats), np.asarray([1.0, 2.0], dtype=np.float64)
        )


@pytest.mark.parametrize(
    ("value", "hexadecimal", "sign", "exponent", "classification"),
    (
        (0.0, "0x0000", "positive", -126, "zero"),
        (-0.0, "0x8000", "negative", -126, "zero"),
        (float(np.asarray([1], dtype=np.uint16).view(bfloat16)[0]), "0x0001", "positive", -126, "subnormal"),
        (-1.0, "0xbf80", "negative", 0, "normal"),
        (np.inf, "0x7f80", "positive", None, "infinity"),
        (np.nan, "0x7fc0", "positive", None, "nan"),
    ),
)
def test_bfloat16_scalar_diagnostic_reports_canonical_bits_and_class(
    value: float,
    hexadecimal: str,
    sign: str,
    exponent: int | None,
    classification: str,
) -> None:
    diagnostic = runner._bfloat16_scalar_diagnostic(value)

    assert diagnostic.hexadecimal == hexadecimal
    assert diagnostic.sign == sign
    assert diagnostic.exponent == exponent
    assert diagnostic.classification == classification


def test_worst_pair_diagnostic_reports_near_zero_sign_crossing_without_arrays() -> None:
    actual = np.asarray([1.0, -0.0003604888916015625, 2.0], dtype=bfloat16)
    reference = np.asarray([1.0, 0.0004482269287109375, 2.0], dtype=bfloat16)

    diagnostic = runner._worst_pair_diagnostic(actual, reference, absolute_threshold=0.0078125)

    assert diagnostic is not None
    assert diagnostic.index == (1,)
    assert diagnostic.actual.hexadecimal == "0xb9bd"
    assert diagnostic.actual.value == -0.0003604888916015625
    assert diagnostic.actual.sign == "negative"
    assert diagnostic.actual.exponent == -12
    assert diagnostic.actual.classification == "normal"
    assert diagnostic.reference.hexadecimal == "0x39eb"
    assert diagnostic.reference.value == 0.0004482269287109375
    assert diagnostic.reference.sign == "positive"
    assert diagnostic.reference.exponent == -12
    assert diagnostic.reference.classification == "normal"
    assert diagnostic.absolute_error == 0.0008087158203125
    assert diagnostic.ulp_distance == 29608
    assert diagnostic.finite_values == 3
    assert diagnostic.exact_mismatches == 1
    assert diagnostic.one_ulp_mismatches == 1
    assert diagnostic.absolute_threshold == 0.0078125
    assert diagnostic.absolute_mismatches == 0


def test_worst_pair_failure_diagnostic_rejects_oversized_serialization() -> None:
    actual = np.asarray([1.0], dtype=bfloat16)
    diagnostic = runner._worst_pair_diagnostic(actual, actual, absolute_threshold=0.0078125)
    assert diagnostic is not None
    oversized = replace(diagnostic, index=tuple(range(1_000)))

    error = runner._with_worst_pair_diagnostic(
        NumericalFloorError("base", output_name="dx"),
        oversized,
    )

    assert str(error) == "numerical worst-pair diagnostic exceeded the closed 2048-character bound"
    assert error.output_name == "dx"


_NCU_SASS_SEPARATOR = "------------------ " + "-" * 60 + " ------ ------ ------ ------"
_NCU_SASS_COLUMN_WIDTHS = (18, 60, 6, 6, 6, 6)
_NCU_KERNEL_A = "ordinary_xla_kernel_nam_00"
_NCU_KERNEL_B = "ordinary_xla_kernel_nam_01"


def _ncu_sass_kernel_row(name: str) -> str:
    return f"Kernel Name        {name}{' ' * 62}"


def _ncu_sass_export(*sections: tuple[str, tuple[str, ...]]) -> str:
    lines = [_NCU_SASS_SEPARATOR]
    for name, instructions in sections:
        lines.extend(
            (
                _ncu_sass_kernel_row(name),
                _NCU_SASS_SEPARATOR,
                "Address Source",
                _NCU_SASS_SEPARATOR,
            )
        )
        lines.extend(instructions)
    return "\n".join(lines) + "\n"


def _ncu_sass_fixed_column_row(columns: tuple[str, ...], gaps: tuple[bytes, ...] = (b" ",) * 5) -> str:
    assert len(columns) == len(_NCU_SASS_COLUMN_WIDTHS)
    assert len(gaps) == len(_NCU_SASS_COLUMN_WIDTHS) - 1
    chunks: list[bytes] = []
    for index, (column, width) in enumerate(zip(columns, _NCU_SASS_COLUMN_WIDTHS, strict=True)):
        value = column.encode("utf-8")
        assert len(value) <= width
        chunks.append(value.ljust(width, b" "))
        if index < len(gaps):
            assert len(gaps[index]) == 1
            chunks.append(gaps[index])
    row = b"".join(chunks)
    assert len(row) == len(_NCU_SASS_SEPARATOR.encode("ascii"))
    return row.decode("utf-8")


def _ncu_sass_diagnostic(error: ValueError) -> dict[str, object]:
    message = str(error)
    prefix = "unrecognized Nsight Compute SASS export record: "
    assert message.startswith(prefix)
    assert len(message.encode("utf-8")) <= 2048
    diagnostic = json.loads(message.removeprefix(prefix))
    assert set(diagnostic) == {"line_number", "line_sha256", "line_structure", "line_utf8_bytes"}
    structure = diagnostic["line_structure"]
    assert isinstance(structure, dict)
    expected_structure_fields = {
        "ascii_classes",
        "delimiters",
        "leading_spaces",
        "non_ascii_codepoints",
        "public_patterns",
        "public_vocabulary",
        "spaces",
        "tabs",
        "token_count",
        "token_max_utf8_bytes",
        "trailing_spaces",
    }
    if diagnostic["line_utf8_bytes"] == len(_NCU_SASS_SEPARATOR.encode("ascii")):
        expected_structure_fields.add("fixed_columns")
    assert set(structure) == expected_structure_fields
    return diagnostic


def _ncu_sass_structure(error: ValueError) -> dict[str, object]:
    structure = _ncu_sass_diagnostic(error)["line_structure"]
    assert isinstance(structure, dict)
    return structure


def test_runner_preflight_records_exact_clean_source_tools_and_h100(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)

    def run(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        if arguments[1:3] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(arguments, 0, config.source_sha + "\n", "")
        if arguments[1:3] == ("status", "--porcelain"):
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if "--query-gpu=name,compute_cap" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "NVIDIA H100 80GB HBM3, 9.0\n", "")
        return subprocess.CompletedProcess(arguments, 0, f"{Path(arguments[0]).name} exact-version\n", "")

    evidence = runner.require_clean_h100_preflight(config, run=run)

    assert evidence.source_sha == config.source_sha
    assert evidence.gpu_name == "NVIDIA H100 80GB HBM3"
    assert evidence.compute_capability == "9.0"
    assert evidence.architecture == "sm_90a"
    assert tuple(tool.name for tool in evidence.tools) == (
        "git",
        "nvidia_smi",
        "nvcc",
        "ptxas",
        "cuobjdump",
        "ncu",
        "nsys",
    )
    assert all(len(tool.sha256) == 64 and tool.version_output.endswith("exact-version") for tool in evidence.tools)


def test_runner_preflight_rejects_existing_artifact_directory_before_commands(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    config.artifact_directory.mkdir()

    def run(*args, **kwargs):
        raise AssertionError("preflight must reject before invoking external tools")

    with pytest.raises(ValueError, match="must not already exist"):
        runner.require_clean_h100_preflight(config, run=run)


def test_runner_preflight_rejects_tracked_or_untracked_source_changes(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)

    def run(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        if arguments[1:3] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(arguments, 0, config.source_sha + "\n", "")
        if arguments[1:3] == ("status", "--porcelain"):
            assert "--untracked-files=all" in arguments
            return subprocess.CompletedProcess(arguments, 0, "?? injected.py\n", "")
        raise AssertionError(f"preflight continued after dirty source: {arguments}")

    with pytest.raises(ValueError, match="no modifications or untracked files"):
        runner.require_clean_h100_preflight(config, run=run)


def test_runner_capsule_preflight_records_manifest_identity_without_git_status(tmp_path: Path) -> None:
    config = _capsule_runner_config(tmp_path)

    def run(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        assert "rev-parse" not in arguments and "status" not in arguments
        if "--query-gpu=name,compute_cap" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "NVIDIA H100 80GB HBM3, 9.0\n", "")
        return subprocess.CompletedProcess(arguments, 0, f"{Path(arguments[0]).name} exact-version\n", "")

    evidence = runner.require_clean_h100_preflight(config, run=run)

    assert evidence.source_tree == "2" * 40
    assert evidence.source_capsule_manifest_sha256 == config.source_capsule_manifest_sha256


def test_runner_capsule_module_audit_rejects_local_import_outside_manifest(tmp_path: Path) -> None:
    config = _capsule_runner_config(tmp_path)
    installed = tmp_path / "installed" / "tile_lifetime" / "omitted.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("VALUE = 1\n")
    module = ModuleType("tile_lifetime.omitted")
    module.__file__ = str(installed)
    with pytest.raises(ValueError, match="loaded outside"):
        runner.audit_imported_local_modules(config, {module.__name__: module})


def test_runner_capsule_module_audit_accepts_exact_manifested_local_import(tmp_path: Path) -> None:
    config = _capsule_runner_config(tmp_path)
    module = ModuleType("tile_lifetime.local")
    module.__file__ = str(config.source_root / "lib/tile_lifetime/local.py")
    runner.audit_imported_local_modules(config, {module.__name__: module})


def test_runner_ncu_parser_requires_every_closed_launch_metric(tmp_path: Path) -> None:
    output = tmp_path / "ncu.csv"
    _write_ncu_csv(output)

    records = runner.parse_ncu_metrics(output)

    assert records == (
        runner.NcuKernelMetrics(
            name="KernelA",
            block_size=(256, 1, 1),
            registers_per_thread=48,
            static_shared_memory_bytes=128,
            dynamic_shared_memory_bytes=0,
            active_blocks_per_sm=2,
            limiting_occupancy_resource="registers,warps",
            achieved_occupancy=0.625,
        ),
    )
    fieldnames, rows = _read_ncu_csv(output)
    fieldnames.remove("launch__registers_per_thread")
    for row in rows:
        del row["launch__registers_per_thread"]
    _write_ncu_rows(output, fieldnames, rows)
    with pytest.raises(ValueError, match="omits required columns"):
        runner.parse_ncu_metrics(output)


def test_runner_ncu_parser_accepts_exact_units_row_only_in_first_position(tmp_path: Path) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    output = tmp_path / "ncu.csv"

    _write_ncu_rows(output, fieldnames, rows[1:])
    with pytest.raises(ValueError, match="first data row must be the exact units row"):
        runner.parse_ncu_metrics(output)

    _write_ncu_rows(output, fieldnames, (rows[1], rows[0]))
    with pytest.raises(ValueError, match="first data row must be the exact units row"):
        runner.parse_ncu_metrics(output)

    _write_ncu_rows(output, fieldnames, (rows[0], rows[0], rows[1]))
    with pytest.raises(ValueError, match="units row may appear only once"):
        runner.parse_ncu_metrics(output)


@pytest.mark.parametrize(
    "identity",
    (
        "ID",
        "Process ID",
        "Process Name",
        "Host Name",
        "Kernel Name",
        "Context",
        "Stream",
        "Block Size",
        "Grid Size",
        "Device",
        "CC",
    ),
)
def test_runner_ncu_parser_requires_every_units_row_identity_to_be_empty(tmp_path: Path, identity: str) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    rows[0][identity] = "unexpected-identity"
    output = tmp_path / "ncu.csv"
    _write_ncu_rows(output, fieldnames, rows)

    with pytest.raises(ValueError, match="first data row must be the exact units row"):
        runner.parse_ncu_metrics(output)


@pytest.mark.parametrize(
    ("metric", "unit"),
    (
        ("launch__block_size", "thread"),
        ("launch__registers_per_thread", "register"),
        ("launch__shared_mem_per_block_static", "byte"),
        ("launch__shared_mem_per_block_dynamic", "byte"),
        ("launch__occupancy_limit_blocks", "blocks"),
        ("launch__occupancy_limit_registers", "blocks"),
        ("launch__occupancy_limit_shared_mem", "blocks"),
        ("launch__occupancy_limit_warps", "blocks"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "percent"),
    ),
)
def test_runner_ncu_parser_rejects_wrong_or_extra_required_units(tmp_path: Path, metric: str, unit: str) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    rows[0][metric] = unit
    output = tmp_path / "ncu.csv"
    _write_ncu_rows(output, fieldnames, rows)

    with pytest.raises(ValueError, match="first data row must be the exact units row"):
        runner.parse_ncu_metrics(output)


def test_runner_ncu_parser_rejects_blank_or_units_as_kernel_data(tmp_path: Path) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    output = tmp_path / "ncu.csv"

    _write_ncu_rows(output, fieldnames, (rows[0], dict.fromkeys(fieldnames, "")))
    with pytest.raises(ValueError, match="units row may appear only once"):
        runner.parse_ncu_metrics(output)

    units_as_data = dict(rows[0])
    units_as_data["ID"] = "1"
    units_as_data["Kernel Name"] = "KernelA"
    _write_ncu_rows(output, fieldnames, (rows[0], units_as_data))
    with pytest.raises(ValueError, match="units row may appear only once"):
        runner.parse_ncu_metrics(output)


@pytest.mark.parametrize("missing_identity", ("ID", "Kernel Name"))
def test_runner_ncu_parser_does_not_echo_profiler_rows(tmp_path: Path, missing_identity: str) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    fieldnames.append("private_profiler_field")
    private = "/private/profiler/path/" + "secret" * 10_000
    rows[0]["private_profiler_field"] = "opaque-unit"
    rows[1]["private_profiler_field"] = private
    rows[1][missing_identity] = ""
    output = tmp_path / "ncu.csv"
    _write_ncu_rows(output, fieldnames, rows)

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_metrics(output)
    assert str(failure.value) == f"Nsight Compute row omits required field {missing_identity!r}"
    assert private not in str(failure.value)


def test_runner_ncu_parser_bounds_and_decodes_input_before_csv_parsing(tmp_path: Path) -> None:
    output = tmp_path / "ncu.csv"

    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    padding_fields = tuple(f"opaque_padding_{index}" for index in range(9))
    fieldnames.extend(padding_fields)
    for field in padding_fields:
        rows[0][field] = ""
        rows[1][field] = ""
    _write_ncu_rows(output, fieldnames, rows)
    padding = (1 << 20) - output.stat().st_size
    assert padding > 0
    quotient, remainder = divmod(padding, len(padding_fields))
    for index, field in enumerate(padding_fields):
        rows[1][field] = "x" * (quotient + (index < remainder))
    _write_ncu_rows(output, fieldnames, rows)
    assert output.stat().st_size == 1 << 20
    assert runner.parse_ncu_metrics(output)[0].name == "KernelA"

    output.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        runner.parse_ncu_metrics(output)

    output.write_bytes(_NCU_RAW_FIXTURE.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="must not contain NUL"):
        runner.parse_ncu_metrics(output)

    with output.open("wb") as stream:
        stream.seek((1 << 20) + 1)
        stream.write(b"x")
    with pytest.raises(ValueError, match="reviewed byte bound"):
        runner.parse_ncu_metrics(output)


@pytest.mark.parametrize(
    "variant",
    (
        "valid",
        "missing-top-level",
        "private-line1",
        "missing-identity-close",
        "unrecognized-record",
        "fixed-column-record",
    ),
)
def test_runner_ncu_profile_parses_real_public_exports_at_process_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    private_line1 = "private-line1-/workspace/secret"
    environment_token = "environment-private-token"
    monkeypatch.setenv("NCU_PRIVATE_TEST_TOKEN", environment_token)
    config = _runner_config(tmp_path)
    cache_source, cache_contract = _worker_cache_fixture(tmp_path, ("ordinary_xla",))
    generated_manifest = tmp_path / "generated.json"
    generated_manifest.write_text("[]\n")

    def run_ncu(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        arguments = tuple(str(value) for value in command)
        assert arguments[0] == str(config.tools.ncu)
        assert "--page=raw" in arguments
        raw_metrics = _NCU_RAW_FIXTURE.read_text().replace(",KernelA,", f",{_NCU_KERNEL_A},")
        Path(arguments[arguments.index("--log-file") + 1]).write_text(raw_metrics)
        Path(arguments[arguments.index("--export") + 1]).write_bytes(b"exact-ncu-report")
        output = Path(arguments[arguments.index("--json-output") + 1])
        output.write_text(json.dumps({"persistent_cache": {}, "final_hlo": "exact-final-hlo"}))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def export_sass(command: Any) -> subprocess.CompletedProcess[str]:
        arguments = tuple(str(value) for value in command)
        source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
        if variant == "missing-top-level":
            source = source.split("\n", maxsplit=1)[1]
        elif variant == "private-line1":
            source = private_line1 + "\n" + source
        elif variant == "missing-identity-close":
            source = source.replace(
                _ncu_sass_kernel_row(_NCU_KERNEL_A) + "\n" + _NCU_SASS_SEPARATOR + "\n",
                _ncu_sass_kernel_row(_NCU_KERNEL_A) + "\n",
                1,
            )
        elif variant == "unrecognized-record":
            section = _ncu_sass_kernel_row(_NCU_KERNEL_A)
            source = source.replace(section, "public-unknown-record\n" + section)
        elif variant == "fixed-column-record":
            source = source.replace(
                "Address Source",
                _ncu_sass_fixed_column_row(("Address", "private", "Source", "", "", "")),
            )
        Path(arguments[arguments.index("--log-file") + 1]).write_text(source)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run_ncu)
    monkeypatch.setattr(runner, "_run_retained", export_sass)

    profile = tmp_path / "profile"
    arguments = (
        config,
        "contract_map_case",
        "ordinary_xla",
        generated_manifest,
        profile,
        cache_source,
        cache_contract,
    )
    if variant == "missing-top-level":
        with pytest.raises(ValueError) as failure:
            runner._run_ncu_profile(*arguments)
        diagnostic = _ncu_sass_diagnostic(failure.value)
        assert diagnostic["line_number"] == 1
        structure = diagnostic["line_structure"]
        assert isinstance(structure, dict)
        assert structure["public_patterns"]["section"] is True
        assert "fixed_columns" in structure
        assert _NCU_KERNEL_A not in str(failure.value)
        return
    if variant == "private-line1":
        with pytest.raises(ValueError) as failure:
            runner._run_ncu_profile(*arguments)
        diagnostic = _ncu_sass_diagnostic(failure.value)
        assert diagnostic["line_number"] == 1
        assert diagnostic["line_sha256"] == hashlib.sha256(private_line1.encode()).hexdigest()
        assert private_line1 not in str(failure.value)
        assert _NCU_SASS_SEPARATOR not in str(failure.value)
        assert environment_token not in str(failure.value)
        return
    if variant == "missing-identity-close":
        with pytest.raises(ValueError, match=r"identity table omits.*line 3"):
            runner._run_ncu_profile(*arguments)
        return
    if variant == "unrecognized-record":
        with pytest.raises(ValueError) as failure:
            runner._run_ncu_profile(*arguments)
        assert _ncu_sass_diagnostic(failure.value) == {
            "line_number": 2,
            "line_sha256": hashlib.sha256(b"public-unknown-record").hexdigest(),
            "line_structure": {
                "ascii_classes": {
                    "control": 0,
                    "digit": 0,
                    "lowercase": 19,
                    "punctuation": 2,
                    "uppercase": 0,
                    "whitespace": 0,
                },
                "delimiters": {"colon": 0, "comma": 0, "hyphen": 2, "pipe": 0},
                "leading_spaces": 0,
                "non_ascii_codepoints": 0,
                "public_patterns": {
                    "header": False,
                    "instruction": False,
                    "section": False,
                    "separator": False,
                    "status": False,
                },
                "public_vocabulary": {
                    "Address": False,
                    "Function": False,
                    "Kernel": False,
                    "Name": False,
                    "Section": False,
                    "Source": False,
                },
                "spaces": 0,
                "tabs": 0,
                "token_count": 1,
                "token_max_utf8_bytes": 21,
                "trailing_spaces": 0,
            },
            "line_utf8_bytes": 21,
        }
        assert "public-unknown-record" not in str(failure.value)
        return
    if variant == "fixed-column-record":
        with pytest.raises(ValueError) as failure:
            runner._run_ncu_profile(*arguments)
        diagnostic = _ncu_sass_diagnostic(failure.value)
        assert diagnostic["line_number"] == 4
        structure = diagnostic["line_structure"]
        assert isinstance(structure, dict)
        fixed_columns = structure["fixed_columns"]
        assert isinstance(fixed_columns, dict)
        assert fixed_columns["column_widths"] == [18, 60, 6, 6, 6, 6]
        assert fixed_columns["gap_single_ascii_space"] == [True] * 5
        return

    evidence = runner._run_ncu_profile(*arguments)

    assert tuple(metric.name for metric in evidence.metrics) == (_NCU_KERNEL_A,)
    assert evidence.metrics[0].registers_per_thread == 48
    assert evidence.final_hlo == "exact-final-hlo"
    assert evidence.report_sha256 == hashlib.sha256(b"exact-ncu-report").hexdigest()


def test_runner_ncu_sass_parser_requires_valid_instruction_rows_for_exact_kernel_sections() -> None:
    source = _ncu_sass_export(
        (_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2", "0000000000000010 EXIT")),
        (_NCU_KERNEL_B, ("0000000000000020 FFMA R3, R4, R5, R6",)),
    )
    records = runner.parse_ncu_sass(source, (_NCU_KERNEL_A, _NCU_KERNEL_B))

    assert tuple(record.name for record in records) == (_NCU_KERNEL_A, _NCU_KERNEL_B)
    assert tuple(instruction.mnemonic for instruction in records[0].instructions) == ("MOV", "EXIT")
    assert records[1].instructions == (runner.NcuSassInstruction(address=0x20, mnemonic="FFMA"),)


def test_runner_ncu_sass_parser_accepts_exact_colonless_padded_kernel_identity() -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))

    records = runner.parse_ncu_sass(source, (f"void {_NCU_KERNEL_A}()",))

    assert records[0].name == _NCU_KERNEL_A
    section = source.splitlines()[1]
    assert len(section.encode("utf-8")) == 107
    assert section == "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 62


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "missing-second",
        "duplicate",
        "moved",
        "wrong-literal",
        "blank-before-close",
        "separator-before-kernel",
    ),
)
def test_runner_ncu_sass_parser_requires_immediate_exact_identity_table_close(mutation: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    row = _ncu_sass_kernel_row(_NCU_KERNEL_A)
    expected_names = (_NCU_KERNEL_A,)
    if mutation == "missing-second":
        source = _ncu_sass_export(
            (_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)),
            (_NCU_KERNEL_B, ("0000000000000010 EXIT",)),
        )
        row = _ncu_sass_kernel_row(_NCU_KERNEL_B)
        expected_names = (_NCU_KERNEL_A, _NCU_KERNEL_B)
    identity_table = row + "\n" + _NCU_SASS_SEPARATOR + "\nAddress Source"
    if mutation in {"missing", "missing-second"}:
        source = source.replace(identity_table, row + "\nAddress Source", 1)
    elif mutation == "duplicate":
        source = source.replace(
            identity_table,
            row + "\n" + _NCU_SASS_SEPARATOR + "\n" + _NCU_SASS_SEPARATOR + "\nAddress Source",
            1,
        )
    elif mutation == "moved":
        source = source.replace(identity_table, row + "\nAddress Source\n" + _NCU_SASS_SEPARATOR, 1)
    elif mutation == "wrong-literal":
        source = source.replace(identity_table, row + "\n" + _NCU_SASS_SEPARATOR + " \nAddress Source", 1)
    elif mutation == "blank-before-close":
        source = source.replace(identity_table, row + "\n\n" + _NCU_SASS_SEPARATOR + "\nAddress Source", 1)
    elif mutation == "separator-before-kernel":
        source = source.replace(
            _NCU_SASS_SEPARATOR + "\n" + row,
            _NCU_SASS_SEPARATOR + "\n" + _NCU_SASS_SEPARATOR + "\n" + row,
            1,
        )
    else:
        raise AssertionError(f"unhandled mutation {mutation!r}")

    with pytest.raises(ValueError, match=r"identity table omits|misplaced"):
        runner.parse_ncu_sass(source, expected_names)


@pytest.mark.parametrize(
    "section",
    (
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 61,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 63,
        " " + "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 62,
        "Kernel  Name" + " " * 7 + _NCU_KERNEL_A + " " * 62,
        "Kernel Name" + " " * 9 + _NCU_KERNEL_A + " " * 61,
        "Kernel Name" + " " * 7 + "\t" + _NCU_KERNEL_A + " " * 62,
        "Kernel Name" + " " * 7 + "\x1b" + _NCU_KERNEL_A + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 61 + "\t",
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A + " " * 61 + "\x1b",
        "Name Kernel" + " " * 8 + _NCU_KERNEL_A + " " * 62,
        "KernelName" + " " * 9 + _NCU_KERNEL_A + " " * 62,
        "Kernel Name:" + " " * 7 + _NCU_KERNEL_A + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A.replace("_", "-", 1) + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A.replace("_", ",", 1) + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A.replace("_", "|", 1) + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A.replace("_", ".", 1) + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A.replace("_", "$", 1) + " " * 62,
        "Kernel Name" + " " * 8 + "0" + _NCU_KERNEL_A[1:] + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A[:-1] + "é" + " " * 62,
        "Kernel Name" + " " * 7 + "extra " + _NCU_KERNEL_A + " " * 57,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A[:-1] + " " * 62,
        "Kernel Name" + " " * 8 + _NCU_KERNEL_A + "x" + " " * 62,
    ),
    ids=(
        "106-bytes",
        "108-bytes",
        "leading-space",
        "first-gap-distribution",
        "trailing-padding-distribution",
        "internal-tab",
        "internal-control",
        "trailing-tab",
        "trailing-control",
        "word-order",
        "word-lookalike",
        "colon-form",
        "hyphen-name-character",
        "comma-name-character",
        "pipe-name-character",
        "dot-name-character",
        "dollar-name-character",
        "invalid-name-prefix",
        "unicode-name-continuation",
        "extra-token",
        "name-too-short",
        "name-too-long",
    ),
)
def test_runner_ncu_sass_parser_rejects_kernel_identity_row_mutations(section: str) -> None:
    source = "\n".join(
        (
            _NCU_SASS_SEPARATOR,
            section,
            _NCU_SASS_SEPARATOR,
            "Address Source",
            _NCU_SASS_SEPARATOR,
            "0000000000000000 MOV R1, R2",
            "",
        )
    )

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    diagnostic = _ncu_sass_diagnostic(failure.value)
    assert diagnostic["line_number"] == 2


@pytest.mark.parametrize(
    ("record", "public_pattern"),
    (
        ("", None),
        ("==PROF== Connected to process 123 (/public/tool)", "status"),
        (_ncu_sass_kernel_row(_NCU_KERNEL_A), "section"),
        ("Address Source", "header"),
        ("0000000000000000 MOV R1, R2", "instruction"),
        (_NCU_SASS_SEPARATOR[:-1] + "x", None),
        ("unicode-" + "\N{SNOWMAN}" * 3, None),
        ("\tcontrol\x1brecord", None),
    ),
    ids=("blank", "status", "identity", "header", "instruction", "separator-lookalike", "unicode", "control"),
)
def test_runner_ncu_sass_line1_mismatch_uses_closed_structural_diagnostic(
    record: str, public_pattern: str | None
) -> None:
    source = record + "\n" + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    diagnostic = _ncu_sass_diagnostic(failure.value)
    assert diagnostic["line_number"] == 1
    assert diagnostic["line_utf8_bytes"] == len(record.encode("utf-8"))
    assert diagnostic["line_sha256"] == hashlib.sha256(record.encode("utf-8")).hexdigest()
    structure = diagnostic["line_structure"]
    assert isinstance(structure, dict)
    patterns = structure["public_patterns"]
    assert isinstance(patterns, dict)
    assert {name for name, matched in patterns.items() if matched} == ({public_pattern} if public_pattern else set())
    assert ("fixed_columns" in structure) is (len(record.encode("utf-8")) == 107)
    if record:
        assert record not in str(failure.value)


@pytest.mark.parametrize("size", (1024, 1025))
def test_runner_ncu_sass_line1_mismatch_retains_line_and_diagnostic_bounds(size: int) -> None:
    private_record = "x" * size
    source = private_record + "\n" + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    if size == 1024:
        diagnostic = _ncu_sass_diagnostic(failure.value)
        assert diagnostic["line_number"] == 1
        assert diagnostic["line_utf8_bytes"] == 1024
    else:
        assert str(failure.value) == "Nsight Compute SASS export line 1 exceeds its reviewed bound"
    assert len(str(failure.value).encode("utf-8")) <= 2048
    assert private_record not in str(failure.value)


def test_runner_ncu_sass_line1_mismatch_diagnostic_fails_closed_at_serialized_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_NCU_SASS_DIAGNOSTIC_PREFIX", "x" * 2048)
    private_record = "private-line-1"

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(private_record + "\n" + _NCU_SASS_SEPARATOR, (_NCU_KERNEL_A,))

    assert str(failure.value) == "unrecognized Nsight Compute SASS export record; diagnostic exceeds reviewed bound"
    assert len(str(failure.value).encode("utf-8")) <= 2048
    assert private_record not in str(failure.value)


@pytest.mark.parametrize(
    "record",
    (
        "x" * 512,
        "x" * 513,
        ('"\\' * 256),
        "private-token-/workspace/secret",
        "\tcontrol\x1brecord",
        "unicode-" + "\N{SNOWMAN}" * 300,
    ),
    ids=("512-bytes", "513-bytes", "escape-expansion", "private-token", "control", "unicode"),
)
def test_runner_ncu_sass_unrecognized_record_diagnostic_is_metadata_only(record: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, (record,)))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    record_bytes = record.encode("utf-8")
    diagnostic = _ncu_sass_diagnostic(failure.value)
    structure = diagnostic.pop("line_structure")
    assert diagnostic == {
        "line_number": 6,
        "line_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "line_utf8_bytes": len(record_bytes),
    }
    assert isinstance(structure, dict)
    assert record not in str(failure.value)


@pytest.mark.parametrize(
    ("record", "expected"),
    (
        ("x" * 512, {"token_max_utf8_bytes": 512, "lowercase": 512}),
        ("x" * 513, {"token_max_utf8_bytes": 513, "lowercase": 513}),
        ('"\\' * 256, {"token_max_utf8_bytes": 512, "punctuation": 512}),
        ("\tcontrol\x1brecord", {"tabs": 1, "control": 1, "whitespace": 1}),
        ("unicode-" + "\N{SNOWMAN}" * 300, {"token_max_utf8_bytes": 908, "non_ascii_codepoints": 300}),
    ),
)
def test_runner_ncu_sass_line_structure_counts_boundary_and_escape_classes(
    record: str, expected: dict[str, int]
) -> None:
    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    structure = _ncu_sass_structure(failure.value)
    ascii_classes = structure["ascii_classes"]
    assert isinstance(ascii_classes, dict)
    for field, value in expected.items():
        if field in ascii_classes:
            assert ascii_classes[field] == value
        else:
            assert structure[field] == value


def test_runner_ncu_sass_line_structure_is_closed_aggregate_metadata() -> None:
    record = "  Address Source:\tKernel,Name|Section-Function  "

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    diagnostic = _ncu_sass_diagnostic(failure.value)
    assert diagnostic["line_structure"] == {
        "ascii_classes": {
            "control": 0,
            "digit": 0,
            "lowercase": 32,
            "punctuation": 4,
            "uppercase": 6,
            "whitespace": 6,
        },
        "delimiters": {"colon": 1, "comma": 1, "hyphen": 1, "pipe": 1},
        "leading_spaces": 2,
        "non_ascii_codepoints": 0,
        "public_patterns": {
            "header": False,
            "instruction": False,
            "section": False,
            "separator": False,
            "status": False,
        },
        "public_vocabulary": {
            "Address": True,
            "Function": True,
            "Kernel": True,
            "Name": True,
            "Section": True,
            "Source": True,
        },
        "spaces": 5,
        "tabs": 1,
        "token_count": 3,
        "token_max_utf8_bytes": 28,
        "trailing_spaces": 2,
    }
    assert record not in str(failure.value)


def test_runner_ncu_sass_fixed_columns_report_closed_per_column_aggregates() -> None:
    record = _ncu_sass_fixed_column_row(("Address", "private alpha", "Source", "A1_b", "\tab", "\x1bxy"))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    diagnostic = _ncu_sass_diagnostic(failure.value)
    assert len(str(failure.value).encode("utf-8")) <= 2048
    assert diagnostic["line_utf8_bytes"] == 107
    structure = diagnostic["line_structure"]
    assert isinstance(structure, dict)
    assert structure["fixed_columns"] == {
        "ascii_class_fields": ["control", "digit", "lowercase", "punctuation", "uppercase", "whitespace"],
        "column_fields": [
            "index",
            "trimmed_utf8_bytes",
            "ascii_class_counts",
            "non_ascii_bytes",
            "token_count",
            "Address",
            "Source",
        ],
        "column_widths": [18, 60, 6, 6, 6, 6],
        "columns": [
            [0, 7, [0, 0, 6, 0, 1, 0], 0, 1, True, False],
            [1, 13, [0, 0, 12, 0, 0, 1], 0, 2, False, False],
            [2, 6, [0, 0, 5, 0, 1, 0], 0, 1, False, True],
            [3, 4, [0, 1, 1, 1, 1, 0], 0, 1, False, False],
            [4, 3, [0, 0, 2, 0, 0, 1], 0, 1, False, False],
            [5, 3, [1, 0, 2, 0, 0, 0], 0, 1, False, False],
        ],
        "gap_single_ascii_space": [True, True, True, True, True],
    }
    assert record not in str(failure.value)


@pytest.mark.parametrize("gap_index", range(5))
@pytest.mark.parametrize("replacement", (b"\t", b"x"), ids=("tab", "nonspace"))
def test_runner_ncu_sass_fixed_columns_report_each_invalid_gap(gap_index: int, replacement: bytes) -> None:
    gaps = [b" "] * 5
    gaps[gap_index] = replacement
    record = _ncu_sass_fixed_column_row(("Address", "private", "Source", "", "", ""), tuple(gaps))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    fixed_columns = _ncu_sass_structure(failure.value)["fixed_columns"]
    assert isinstance(fixed_columns, dict)
    expected = [True] * 5
    expected[gap_index] = False
    assert fixed_columns["gap_single_ascii_space"] == expected


def test_runner_ncu_sass_fixed_columns_public_words_reject_attached_lookalikes() -> None:
    record = _ncu_sass_fixed_column_row(("_Address", "\N{LATIN SMALL LETTER E WITH ACUTE}Source", "", "", "", ""))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    fixed_columns = _ncu_sass_structure(failure.value)["fixed_columns"]
    assert isinstance(fixed_columns, dict)
    assert fixed_columns["columns"][0][-2:] == [False, False]
    assert fixed_columns["columns"][1][-2:] == [False, False]
    assert fixed_columns["columns"][1][1] == 8
    assert fixed_columns["columns"][1][3] == 2


def test_runner_ncu_sass_fixed_columns_trim_ascii_space_on_both_sides() -> None:
    record = _ncu_sass_fixed_column_row(("  alpha  ", "", "", "", "", ""))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    fixed_columns = _ncu_sass_structure(failure.value)["fixed_columns"]
    assert isinstance(fixed_columns, dict)
    assert fixed_columns["columns"][0][1:5] == [5, [0, 0, 5, 0, 0, 0], 0, 1]


def test_runner_ncu_sass_fixed_columns_use_the_reviewed_byte_boundaries() -> None:
    record = _ncu_sass_fixed_column_row(("A" * 18, "b" * 60, "C" * 6, "1" * 6, "_" * 6, "\x7f" * 6))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    fixed_columns = _ncu_sass_structure(failure.value)["fixed_columns"]
    assert isinstance(fixed_columns, dict)
    assert [column[1] for column in fixed_columns["columns"]] == [18, 60, 6, 6, 6, 6]
    assert [column[2] for column in fixed_columns["columns"]] == [
        [0, 0, 0, 0, 18, 0],
        [0, 0, 60, 0, 0, 0],
        [0, 0, 0, 0, 6, 0],
        [0, 6, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0],
        [6, 0, 0, 0, 0, 0],
    ]


def test_runner_ncu_sass_fixed_columns_do_not_expose_private_values_or_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_values = ("secret", "hidden")
    adjacent = "adjacent-private-value"
    environment_value = "environment-private-value"
    monkeypatch.setenv("NCU_PRIVATE_TEST_TOKEN", environment_value)
    diagnostics = []
    messages = []
    for first, second in (private_values, tuple(reversed(private_values))):
        record = _ncu_sass_fixed_column_row((first, second, "Source", "", "", ""))
        source = _ncu_sass_export((_NCU_KERNEL_A, (record, adjacent)))
        with pytest.raises(ValueError) as failure:
            runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))
        diagnostic = _ncu_sass_diagnostic(failure.value)
        diagnostics.append(diagnostic)
        messages.append(str(failure.value))

    structures = [diagnostic["line_structure"] for diagnostic in diagnostics]
    assert all(isinstance(structure, dict) for structure in structures)
    assert structures[0]["fixed_columns"] == structures[1]["fixed_columns"]
    assert diagnostics[0]["line_sha256"] != diagnostics[1]["line_sha256"]
    for message in messages:
        assert all(private not in message for private in private_values)
        assert adjacent not in message
        assert environment_value not in message
        assert "column_sha" not in message


@pytest.mark.parametrize("width", (106, 108))
def test_runner_ncu_sass_fixed_columns_are_only_reported_for_exact_width(width: int) -> None:
    record = "x" * width

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    assert "fixed_columns" not in _ncu_sass_structure(failure.value)


@pytest.mark.parametrize(
    ("record", "pattern"),
    (
        ("==PROF== Connected to process 123 (/private/tool)", "status"),
        ("0000000000000000 MOV R1, R2", "instruction"),
    ),
)
def test_runner_ncu_sass_line_structure_reports_only_closed_public_patterns(record: str, pattern: str) -> None:
    source = (
        _NCU_SASS_SEPARATOR
        + "\n"
        + record
        + "\n"
        + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).split("\n", maxsplit=1)[1]
    )

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    patterns = _ncu_sass_structure(failure.value)["public_patterns"]
    assert isinstance(patterns, dict)
    assert patterns == {
        "header": False,
        "instruction": pattern == "instruction",
        "section": False,
        "separator": False,
        "status": pattern == "status",
    }
    assert record not in str(failure.value)


@pytest.mark.parametrize(
    "record",
    (
        "==PROF== Connected to process secret",
        "==PROF== Future status",
        "PROF Connected to process 123",
    ),
)
def test_runner_ncu_sass_line_structure_rejects_status_pattern_lookalikes(record: str) -> None:
    source = (
        _NCU_SASS_SEPARATOR
        + "\n"
        + record
        + "\n"
        + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).split("\n", maxsplit=1)[1]
    )

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    patterns = _ncu_sass_structure(failure.value)["public_patterns"]
    assert isinstance(patterns, dict)
    assert patterns["status"] is False
    assert record not in str(failure.value)


@pytest.mark.parametrize(
    "record",
    (
        "Kernel_Private Name1 Address2 source Sectioned functional",
        "_Kernel 0Name \N{LATIN SMALL LETTER E WITH ACUTE}Address _Source 1Section 2Function",
    ),
)
def test_runner_ncu_sass_line_structure_public_words_require_exact_standalone_tokens(record: str) -> None:

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))

    vocabulary = _ncu_sass_structure(failure.value)["public_vocabulary"]
    assert isinstance(vocabulary, dict)
    assert vocabulary == {
        "Address": False,
        "Function": False,
        "Kernel": False,
        "Name": False,
        "Section": False,
        "Source": False,
    }


def test_runner_ncu_sass_line_structure_does_not_encode_token_order_or_values() -> None:
    diagnostics = []
    for record in ("private,token", "token,private"):
        with pytest.raises(ValueError) as failure:
            runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (record,))), (_NCU_KERNEL_A,))
        assert record not in str(failure.value)
        diagnostics.append(_ncu_sass_diagnostic(failure.value))

    assert diagnostics[0]["line_structure"] == diagnostics[1]["line_structure"]
    assert diagnostics[0]["line_sha256"] != diagnostics[1]["line_sha256"]


def test_runner_ncu_sass_unrecognized_record_diagnostic_does_not_leak_adjacent_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_record = "private-token-/workspace/secret"
    adjacent_record = "adjacent-private-token"
    environment_token = "environment-private-token"
    monkeypatch.setenv("NCU_PRIVATE_TEST_TOKEN", environment_token)
    source = _ncu_sass_export((_NCU_KERNEL_A, (private_record, adjacent_record)))

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    message = str(failure.value)
    assert _ncu_sass_diagnostic(failure.value)["line_number"] == 6
    assert private_record not in message
    assert adjacent_record not in message
    assert environment_token not in message


def test_runner_ncu_sass_nul_rejection_remains_metadata_only() -> None:
    private_record = "private\x00token"

    with pytest.raises(ValueError, match="reviewed text bound") as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (private_record,))), (_NCU_KERNEL_A,))

    assert private_record not in str(failure.value)


def test_runner_ncu_sass_unrecognized_record_diagnostic_fails_closed_at_serialized_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_NCU_SASS_DIAGNOSTIC_PREFIX", "x" * 2048)
    private_record = "private-token"

    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(_ncu_sass_export((_NCU_KERNEL_A, (private_record,))), (_NCU_KERNEL_A,))

    assert str(failure.value) == "unrecognized Nsight Compute SASS export record; diagnostic exceeds reviewed bound"
    assert len(str(failure.value).encode("utf-8")) <= 2048
    assert private_record not in str(failure.value)


@pytest.mark.parametrize(
    "replacement",
    (
        "",
        "----------------- ------------------------------------------------------------ ------ ------ ------ ------",
        "------------------ ----------------------------------------------------------- ------ ------ ------ ------",
        "------------------ ------------------------------------------------------------ ------ ------ ------",
        "------------------ " + "-" * 60 + " ------ ------ ------ ------ ------",
        "------------------ dashed-text ------------------------------------------------ ------ ------ ------ ------",
    ),
)
def test_runner_ncu_sass_parser_rejects_missing_or_mutated_table_separator(replacement: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    section_separator = "Address Source\n" + _NCU_SASS_SEPARATOR

    with pytest.raises(ValueError):
        runner.parse_ncu_sass(
            source.replace(section_separator, "Address Source\n" + replacement),
            (_NCU_KERNEL_A,),
        )


@pytest.mark.parametrize(
    "source",
    (
        _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).split("\n", maxsplit=1)[1],
        "\n" + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))),
        "not-the-reviewed-separator\n" + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))),
        _NCU_SASS_SEPARATOR
        + " \n"
        + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).split("\n", maxsplit=1)[1],
    ),
    ids=("missing", "moved-to-line-2", "arbitrary-line-1", "padded-line-1"),
)
def test_runner_ncu_sass_parser_requires_exact_top_level_separator_at_line_one(source: str) -> None:
    with pytest.raises(ValueError) as failure:
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))

    assert _ncu_sass_diagnostic(failure.value)["line_number"] == 1


def test_runner_ncu_sass_parser_rejects_duplicate_top_level_separator() -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    source = _NCU_SASS_SEPARATOR + "\n" + source

    with pytest.raises(ValueError, match=r"misplaced.*line 2"):
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))


@pytest.mark.parametrize(
    "header",
    ("", " Address Source", "Address Source ", "Address  Source", "Address\tSource", "address Source"),
)
def test_runner_ncu_sass_parser_rejects_missing_or_nonliteral_header(header: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    with pytest.raises(ValueError):
        runner.parse_ncu_sass(source.replace("Address Source", header), (_NCU_KERNEL_A,))


@pytest.mark.parametrize(
    "replacement",
    (
        "Address Source\nAddress Source",
        _NCU_SASS_SEPARATOR + "\nAddress Source",
    ),
)
def test_runner_ncu_sass_parser_rejects_duplicate_or_misplaced_header(replacement: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    with pytest.raises(ValueError, match=r"misplaced|unrecognized"):
        runner.parse_ncu_sass(source.replace("Address Source", replacement), (_NCU_KERNEL_A,))


def test_runner_ncu_sass_parser_rejects_header_before_kernel_section() -> None:
    section = _ncu_sass_kernel_row(_NCU_KERNEL_A)
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).replace(
        section + "\n" + _NCU_SASS_SEPARATOR + "\nAddress Source",
        "Address Source\n" + section + "\n" + _NCU_SASS_SEPARATOR,
    )
    with pytest.raises(ValueError, match="misplaced"):
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))


@pytest.mark.parametrize("separator", (" " + _NCU_SASS_SEPARATOR, _NCU_SASS_SEPARATOR + " "))
def test_runner_ncu_sass_parser_rejects_separator_outer_whitespace(separator: str) -> None:
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    section_separator = "Address Source\n" + _NCU_SASS_SEPARATOR
    with pytest.raises(ValueError):
        runner.parse_ncu_sass(
            source.replace(section_separator, "Address Source\n" + separator),
            (_NCU_KERNEL_A,),
        )


@pytest.mark.parametrize(
    "source",
    (
        _NCU_SASS_SEPARATOR + "\n" + _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))),
        _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).replace(
            "0000000000000000 MOV R1, R2",
            _NCU_SASS_SEPARATOR + "\n0000000000000000 MOV R1, R2",
        ),
        _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))) + _NCU_SASS_SEPARATOR + "\n",
    ),
)
def test_runner_ncu_sass_parser_rejects_misplaced_or_duplicate_table_separator(source: str) -> None:
    with pytest.raises(ValueError, match=r"misplaced|unrecognized"):
        runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))


def test_runner_ncu_sass_file_boundary_is_bounded_and_nonleaking(tmp_path: Path) -> None:
    path = tmp_path / "ncu-sass.txt"
    private = "/private/profiler/" + "secret" * 10_000
    path.write_text(_ncu_sass_export((_NCU_KERNEL_A, (private,))))

    with pytest.raises(ValueError) as failure:
        runner._parse_ncu_sass_file(path, (_NCU_KERNEL_A,))
    assert str(failure.value) == "Nsight Compute SASS export line 6 exceeds its reviewed bound"
    assert private not in str(failure.value)

    with path.open("wb") as stream:
        stream.seek((1 << 20) + 1)
        stream.write(b"x")
    with pytest.raises(ValueError, match="bounded regular file"):
        runner._parse_ncu_sass_file(path, (_NCU_KERNEL_A,))


def test_runner_ncu_sass_accepts_exact_file_and_line_bounds(tmp_path: Path) -> None:
    path = tmp_path / "ncu-sass.txt"
    source = _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)))
    source = source.replace(
        _NCU_SASS_SEPARATOR + "\n",
        _NCU_SASS_SEPARATOR + "\n" + " " * 1024 + "\n",
        1,
    )
    assert runner.parse_ncu_sass(source, (_NCU_KERNEL_A,))[0].name == _NCU_KERNEL_A

    remaining = (1 << 20) - len(source.encode())
    padding: list[str] = []
    while remaining:
        width = min(1024, max(0, remaining - 1))
        padding.append(" " * width + "\n")
        remaining -= width + 1
    path.write_text(source + "".join(padding))
    assert path.stat().st_size == 1 << 20
    assert runner._parse_ncu_sass_file(path, (_NCU_KERNEL_A,))[0].name == _NCU_KERNEL_A

    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        runner._parse_ncu_sass_file(path, (_NCU_KERNEL_A,))

    path.write_bytes(_ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))).encode() + b"\x00")
    with pytest.raises(ValueError, match="reviewed text bound"):
        runner._parse_ncu_sass_file(path, (_NCU_KERNEL_A,))

    with pytest.raises(ValueError, match="line 2 exceeds"):
        runner.parse_ncu_sass(
            source.replace(
                _NCU_SASS_SEPARATOR + "\n",
                _NCU_SASS_SEPARATOR + "\n" + " " * 1025 + "\n",
                1,
            ),
            (_NCU_KERNEL_A,),
        )


def test_runner_ncu_sass_spills_are_read_only_from_validated_instruction_rows(tmp_path: Path) -> None:
    sass_path = tmp_path / "ordinary.sass"
    sass_path.write_text(
        _ncu_sass_export(
            (_NCU_KERNEL_A, ("0000000000000000 LDL.64 R2, [R4]", "0000000000000010 STL [R4], R2")),
        )
    )
    profile = runner.NcuProfileEvidence(
        metrics=(_ncu_metric(_NCU_KERNEL_A),),
        report_path="profile.ncu-rep",
        report_sha256="1" * 64,
        sass_source_path=str(sass_path),
        sass_source_sha256="2" * 64,
        final_hlo="hlo",
    )

    parsed = runner.parse_ncu_sass(sass_path.read_text(), (_NCU_KERNEL_A,))
    assert tuple(instruction.mnemonic for instruction in parsed[0].instructions) == ("LDL.64", "STL")
    with pytest.raises(RuntimeError, match="local-memory spills"):
        runner.ordinary_kernel_records(
            profile,
            {
                "ptx_path": "ordinary.ptx",
                "ptx_sha256": "3" * 64,
                "cubin": {
                    "availability": "unavailable",
                    "unavailable_reason": "public_xla_dump_omits_cubin",
                },
            },
        )


@pytest.mark.parametrize(
    ("source", "expected_names", "message"),
    [
        (
            _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",))),
            (_NCU_KERNEL_A, _NCU_KERNEL_B),
            "coverage differs",
        ),
        (
            _ncu_sass_export(
                (_NCU_KERNEL_A, ("0000000000000000 MOV R1, R2",)),
                (_NCU_KERNEL_A, ("0000000000000010 EXIT",)),
            ),
            (_NCU_KERNEL_A,),
            "repeats a kernel section",
        ),
        (
            _ncu_sass_export((_NCU_KERNEL_A[:-1] + "9", ("0000000000000000 MOV R1, R2",))),
            (_NCU_KERNEL_A,),
            "coverage differs",
        ),
        (
            _ncu_sass_export((_NCU_KERNEL_A, ("not-an-address MOV R1, R2",))),
            (_NCU_KERNEL_A,),
            "unrecognized.*line",
        ),
        (
            _ncu_sass_export((_NCU_KERNEL_A, ("0000000000000000 NOTREAL R1, R2",))),
            (_NCU_KERNEL_A,),
            "unrecognized SASS instruction mnemonic",
        ),
        (
            "==WARNING== Profiler output was truncated\n",
            (_NCU_KERNEL_A,),
            "warning, error, or unavailable source",
        ),
        (
            "==ERROR== Profiler report cannot be imported\n",
            (_NCU_KERNEL_A,),
            "warning, error, or unavailable source",
        ),
        (
            "SASS is unavailable for KernelA\n",
            (_NCU_KERNEL_A,),
            "warning, error, or unavailable source",
        ),
        (
            "ordinary plain text with no SASS structure\n",
            (_NCU_KERNEL_A,),
            "unrecognized Nsight Compute SASS export record",
        ),
    ],
    ids=(
        "missing",
        "duplicate",
        "lookalike",
        "bad-address",
        "bad-mnemonic",
        "warning",
        "error",
        "source-unavailable",
        "plain-text",
    ),
)
def test_runner_ncu_sass_parser_rejects_unverifiable_exports(
    source: str, expected_names: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        runner.parse_ncu_sass(source, expected_names)


def _write_nsys_trace_database(database_path: Path, *, include_memcpy_table: bool) -> None:
    with sqlite3.connect(database_path) as database:
        database.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(start INTEGER, end INTEGER, eventType INTEGER, text TEXT, globalTid INTEGER, "
            "endGlobalTid INTEGER, domainId INTEGER, eventId INTEGER PRIMARY KEY)"
        )
        database.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        database.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, demangledName INTEGER, deviceId INTEGER, correlationId INTEGER, "
            "graphNodeId INTEGER)"
        )
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, correlationId INTEGER)"
        )
        if include_memcpy_table:
            database.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
                "(start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER)"
            )
        database.execute(
            "INSERT INTO NVTX_EVENTS VALUES (100, 400, 59, 'contract_map.steady.0.ordinary_xla', 1234, 1234, 0, 1)"
        )
        database.execute("INSERT INTO StringIds VALUES (7, 'fusion_kernel')")
        database.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100')")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (150, 250, 7, 0, 11, NULL)")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (120, 140, 0, 1234, 11)")


def test_runner_nsys_parser_attributes_kernel_and_copy_activity_to_exact_ranges(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    with sqlite3.connect(database_path) as database:
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?)",
            (
                (260, 270, 32, 1),
                (271, 280, 64, 2),
                (281, 290, 128, 8),
            ),
        )

    records = runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))

    assert records == (
        runner.TraceRange(
            name="contract_map.steady.0.ordinary_xla",
            ordered_kernel_names=("fusion_kernel",),
            kernel_duration_ns=100,
            device_to_device_count=1,
            device_to_device_bytes=128,
            host_to_device_count=1,
            host_to_device_bytes=32,
            device_to_host_count=1,
            device_to_host_bytes=64,
            unexpected_copy_count=0,
        ),
    )


def test_runner_nsys_parser_accepts_lazy_omission_only_with_complete_cuda_trace_provenance(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=False)

    records = runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))

    assert records == (
        runner.TraceRange(
            name="contract_map.steady.0.ordinary_xla",
            ordered_kernel_names=("fusion_kernel",),
            kernel_duration_ns=100,
            device_to_device_count=0,
            device_to_device_bytes=0,
            host_to_device_count=0,
            host_to_device_bytes=0,
            device_to_host_count=0,
            device_to_host_bytes=0,
            unexpected_copy_count=0,
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("trace-disabled", "omits required trace tables"),
        ("empty-kernels", "without CUDA kernels"),
        ("missing-gpu-metadata", "TARGET_INFO_GPU identity"),
        ("wrong-gpu-schema", "TARGET_INFO_GPU identity"),
        ("empty-gpu-metadata", "contains no CUDA device identity"),
        ("unknown-kernel-device", "unknown CUDA device ids"),
        ("dangling-kernel-name", "invalid activity record"),
        ("ambiguous-kernel-name", "do not resolve exactly once"),
        ("wrong-schedule", "exact steady-state schedule"),
    ),
)
def test_runner_nsys_parser_rejects_missing_memcpy_without_complete_cuda_trace_provenance(
    tmp_path: Path, mutation: str, message: str
) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=False)
    expected_ranges = ("contract_map.steady.0.ordinary_xla",)
    with sqlite3.connect(database_path) as database:
        if mutation == "trace-disabled":
            database.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
        elif mutation == "empty-kernels":
            database.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        elif mutation == "missing-gpu-metadata":
            database.execute("DROP TABLE TARGET_INFO_GPU")
        elif mutation == "wrong-gpu-schema":
            database.execute("DROP TABLE TARGET_INFO_GPU")
            database.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, label TEXT)")
        elif mutation == "empty-gpu-metadata":
            database.execute("DELETE FROM TARGET_INFO_GPU")
        elif mutation == "unknown-kernel-device":
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId = 1")
        elif mutation == "dangling-kernel-name":
            database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (160, 240, 999, 0, 12, NULL)")
            database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125, 135, 0, 1234, 12)")
        elif mutation == "ambiguous-kernel-name":
            database.execute("INSERT INTO StringIds VALUES (7, 'ambiguous_kernel')")
        elif mutation == "wrong-schedule":
            expected_ranges = ("contract_map.steady.1.ordinary_xla",)
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ValueError, match=message):
        runner.parse_nsys_sqlite(database_path, expected_ranges)


def test_runner_nsys_parser_rejects_corrupt_or_wrong_activity_schema(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.sqlite"
    corrupt_path.write_bytes(b"not sqlite")
    with pytest.raises(ValueError, match="SQLite export is unreadable"):
        runner.parse_nsys_sqlite(corrupt_path, ("contract_map.steady.0.ordinary_xla",))

    wrong_schema_path = tmp_path / "wrong.sqlite"
    _write_nsys_trace_database(wrong_schema_path, include_memcpy_table=True)
    with sqlite3.connect(wrong_schema_path) as database:
        database.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY")
        database.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, bytes INTEGER)")
    with pytest.raises(ValueError, match="omits start, end, bytes, or copyKind"):
        runner.parse_nsys_sqlite(wrong_schema_path, ("contract_map.steady.0.ordinary_xla",))


def _nsys_failure_diagnostic(error: pytest.ExceptionInfo[ValueError]) -> dict[str, Any]:
    marker = " diagnostic="
    message = str(error.value)
    assert marker in message
    return json.loads(message.split(marker, maxsplit=1)[1])


def test_runner_nsys_no_kernel_diagnostic_is_bounded_and_classifies_exact_intervals(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    report_path = tmp_path / "trace.nsys-rep"
    report_path.write_bytes(b"synthetic report")
    _write_nsys_trace_database(database_path, include_memcpy_table=False)
    with sqlite3.connect(database_path) as database:
        database.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        database.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 7, 0, ?, NULL)",
            (
                (50, 100, 20),
                (90, 110, 21),
                (390, 410, 22),
                (400, 425, 23),
            ),
        )
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, 1234, ?)",
            (
                (40, 45, 20),
                (80, 85, 21),
                (380, 385, 22),
                (410, 415, 23),
            ),
        )
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "(start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER, graphId INTEGER, "
            "graphExecId INTEGER)"
        )
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (150, 250, 0, 30, 1, 2)")
        database.execute("CREATE TABLE UNREVIEWED_PRIVATE_TRACE_NAME (raw_value TEXT)")
        database.execute("INSERT INTO UNREVIEWED_PRIVATE_TRACE_NAME VALUES ('unbounded raw trace value')")

    with pytest.raises(ValueError, match="contains no associated CUDA kernels") as error:
        runner.parse_nsys_sqlite(
            database_path,
            ("contract_map.steady.0.ordinary_xla",),
            report_path=report_path,
        )

    assert len(str(error.value)) <= runner._MAX_NSYS_NO_KERNEL_DIAGNOSTIC_CHARS
    diagnostic = _nsys_failure_diagnostic(error)
    assert diagnostic["range"] == {
        "domain_id": 0,
        "duration_ns": 300,
        "end_global_tid": 1234,
        "end_ns": 400,
        "event_type": 59,
        "global_tid": 1234,
        "name": "contract_map.steady.0.ordinary_xla",
        "start_ns": 100,
    }
    assert diagnostic["kernels"]["interval_counts"] == {
        "after": 1,
        "before": 1,
        "contained": 0,
        "overlap": 2,
    }
    assert diagnostic["kernels"]["nearest_previous_end_offset_ns"] == 0
    assert diagnostic["kernels"]["nearest_next_start_offset_ns"] == 0
    assert diagnostic["kernels"]["row_count"] == 4
    assert diagnostic["kernels"]["unique_name_count"] == 1
    assert diagnostic["graph_trace"]["interval_counts"]["contained"] == 1
    assert diagnostic["graph_trace"]["row_count"] == 1
    assert diagnostic["runtime_correlation"]["associated_kernel_count"] == 0
    assert diagnostic["runtime_correlation"]["resolved_kernel_count"] == 4
    assert diagnostic["runtime_correlation"]["same_thread_contained_count"] == 1
    assert diagnostic["sqlite"]["sha256"] == hashlib.sha256(database_path.read_bytes()).hexdigest()
    assert diagnostic["report"]["sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    with sqlite3.connect(database_path) as database:
        table_names = tuple(
            sorted(str(row[0]) for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        )
    expected_table_hash = hashlib.sha256(
        json.dumps(table_names, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert diagnostic["database"]["table_count"] == len(table_names)
    assert diagnostic["database"]["table_names_sha256"] == expected_table_hash
    kernel_columns = tuple(sorted(("correlationId", "demangledName", "deviceId", "end", "graphNodeId", "start")))
    assert diagnostic["relevant_tables"]["CUPTI_ACTIVITY_KIND_KERNEL"] == {
        "column_count": len(kernel_columns),
        "columns_sha256": (
            hashlib.sha256(json.dumps(kernel_columns, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "present": True,
        "row_count": 4,
    }
    assert diagnostic["profile_args"] == [
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
    ]
    assert diagnostic["export_args"] == ["--type=sqlite", "--lazy=true"]
    assert "fusion_kernel" not in str(error.value)
    assert "UNREVIEWED_PRIVATE_TRACE_NAME" not in str(error.value)
    assert "unbounded raw trace value" not in str(error.value)


def test_runner_nsys_no_kernel_diagnostic_rejects_empty_retained_report(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    report_path = tmp_path / "trace.nsys-rep"
    report_path.touch()
    _write_nsys_trace_database(database_path, include_memcpy_table=False)
    with sqlite3.connect(database_path) as database:
        database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start = 500, end = 600")
        database.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start = 450, end = 475")

    with pytest.raises(ValueError, match="retained Nsight Systems artifact is empty"):
        runner.parse_nsys_sqlite(
            database_path,
            ("contract_map.steady.0.ordinary_xla",),
            report_path=report_path,
        )


def test_runner_nsys_no_kernel_diagnostic_fails_closed_when_reviewed_bound_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=False)
    with sqlite3.connect(database_path) as database:
        database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start = 500, end = 600")
        database.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start = 450, end = 475")
    monkeypatch.setattr(runner, "_MAX_NSYS_NO_KERNEL_DIAGNOSTIC_CHARS", 128)

    with pytest.raises(AssertionError, match="diagnostic exceeds its reviewed bound"):
        runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))


def test_runner_nsys_parser_rejects_overlapping_kernel_with_associated_contained_runtime(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    with sqlite3.connect(database_path) as database:
        database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start = 90, end = 110")

    with pytest.raises(ValueError, match="contains no associated CUDA kernels"):
        runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing-runtime-table", "omits required trace tables"),
        ("dangling-correlation", "does not resolve exactly once"),
        ("ambiguous-correlation", "does not resolve exactly once"),
        ("wrong-runtime-thread", "contains no associated CUDA kernels"),
        ("runtime-outside-range", "contains no associated CUDA kernels"),
    ),
)
def test_runner_nsys_parser_rejects_unassociated_kernel_activity(tmp_path: Path, mutation: str, message: str) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    with sqlite3.connect(database_path) as database:
        if mutation == "missing-runtime-table":
            database.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
        elif mutation == "dangling-correlation":
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId = 999")
        elif mutation == "ambiguous-correlation":
            database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125, 135, 0, 1234, 11)")
        elif mutation == "wrong-runtime-thread":
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid = 5678")
        elif mutation == "runtime-outside-range":
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start = 10, end = 20")
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ValueError, match=message):
        runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong-event-type", "push/pop range"),
        ("wrong-domain", "default NVTX domain"),
        ("cross-thread-range", "same thread"),
        ("overlapping-ranges", "strict source order"),
    ),
)
def test_runner_nsys_parser_rejects_invalid_range_clock_or_order(tmp_path: Path, mutation: str, message: str) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    expected = ("contract_map.steady.0.ordinary_xla",)
    with sqlite3.connect(database_path) as database:
        if mutation == "wrong-event-type":
            database.execute("UPDATE NVTX_EVENTS SET eventType = 60")
        elif mutation == "wrong-domain":
            database.execute("UPDATE NVTX_EVENTS SET domainId = 1")
        elif mutation == "cross-thread-range":
            database.execute("UPDATE NVTX_EVENTS SET endGlobalTid = 5678")
        elif mutation == "overlapping-ranges":
            database.execute(
                "INSERT INTO NVTX_EVENTS VALUES "
                "(350, 500, 59, 'contract_map.steady.1.ordinary_xla', 1234, 1234, 0, 2)"
            )
            expected = (
                "contract_map.steady.0.ordinary_xla",
                "contract_map.steady.1.ordinary_xla",
            )
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ValueError, match=message):
        runner.parse_nsys_sqlite(database_path, expected)


def test_runner_nsys_parser_rejects_equal_start_kernel_order_as_ambiguous(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    with sqlite3.connect(database_path) as database:
        database.execute("INSERT INTO StringIds VALUES (8, 'second_kernel')")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (150, 240, 8, 0, 12, 5)")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125, 145, 0, 1234, 12)")

    with pytest.raises(ValueError, match="equal start timestamps"):
        runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))


@pytest.mark.parametrize(
    "values",
    (
        (260, 250, 64, 8),
        (260, 280, -1, 8),
        (260, 280, 64, -1),
        (b"260", 280, 64, 8),
    ),
)
def test_runner_nsys_parser_rejects_malformed_memcpy_records(tmp_path: Path, values: tuple[object, ...]) -> None:
    database_path = tmp_path / "trace.sqlite"
    _write_nsys_trace_database(database_path, include_memcpy_table=True)
    with sqlite3.connect(database_path) as database:
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?)", values)

    with pytest.raises(ValueError, match="invalid activity record"):
        runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))


def test_runner_profiles_cuda_range_with_supported_nsys_end_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    generated_manifest = tmp_path / "generated.json"
    generated_manifest.write_text("{}\n")
    cache, cache_contract = _worker_cache_fixture(tmp_path, (BackendVariant.ORDINARY_XLA.value,))

    monkeypatch.setattr(runner, "_worker_base_command", lambda *args, **kwargs: ("worker", "--case"))

    def reject_after_inspecting_command(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        assert arguments[:2] == (str(config.tools.nsys), "profile")
        profile_policy = arguments[arguments.index("--force-overwrite=true") + 1 : arguments.index("--output")]
        assert profile_policy == (
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
        )
        assert [value for value in arguments if value.startswith("--capture-range-end")] == ["--capture-range-end=stop"]
        assert not any(value.startswith("--stop-on-range-end") for value in arguments)
        assert arguments[-2:] == ("worker", "--case")
        return subprocess.CompletedProcess(arguments, 2, "", "synthetic nsys refusal")

    monkeypatch.setattr(runner.subprocess, "run", reject_after_inspecting_command)

    with pytest.raises(RuntimeError, match="synthetic nsys refusal"):
        runner._run_profiled_case(config, "case-id", generated_manifest, case_directory, cache, cache_contract)


def test_runner_profile_boundary_explicitly_binds_lazy_export_to_cuda_trace_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    generated_manifest = tmp_path / "generated.json"
    generated_manifest.write_text("{}\n")
    cache, cache_contract = _worker_cache_fixture(tmp_path, (BackendVariant.ORDINARY_XLA.value,))
    result_path = case_directory / "case_result.json"
    monkeypatch.setattr(runner, "_worker_base_command", lambda *args, **kwargs: ("worker", "--case"))

    def successful_profile(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        assert arguments[:2] == (str(config.tools.nsys), "profile")
        assert [value for value in arguments if value.startswith("--trace=")] == ["--trace=cuda,nvtx"]
        assert [value for value in arguments if value.startswith("--cuda-graph-trace=")] == ["--cuda-graph-trace=node"]
        assert arguments[-2:] == ("worker", "--case")
        result_path.write_text(
            json.dumps(
                {
                    "raw_samples": [
                        {
                            "sample_index": 0,
                            "backend_order": [BackendVariant.ORDINARY_XLA.value],
                        }
                    ]
                }
            )
        )
        (case_directory / "steady_trace.nsys-rep").write_bytes(b"synthetic retained report")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def successful_lazy_export(command):
        arguments = tuple(str(value) for value in command)
        assert arguments[:2] == (str(config.tools.nsys), "export")
        assert [value for value in arguments if value.startswith("--lazy=")] == ["--lazy=true"]
        sqlite_path = Path(arguments[arguments.index("--output") + 1])
        _write_nsys_trace_database(sqlite_path, include_memcpy_table=False)

    monkeypatch.setattr(runner.subprocess, "run", successful_profile)
    monkeypatch.setattr(runner, "_run_retained", successful_lazy_export)

    result, records = runner._run_profiled_case(
        config, "case-id", generated_manifest, case_directory, cache, cache_contract
    )

    assert result["raw_samples"][0]["sample_index"] == 0
    assert records[0].ordered_kernel_names == ("fusion_kernel",)
    assert records[0].device_to_device_count == 0
    assert records[0].host_to_device_count == 0
    assert records[0].device_to_host_count == 0


def test_runner_profile_failure_binds_bounded_diagnostic_to_retained_report_and_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    generated_manifest = tmp_path / "generated.json"
    generated_manifest.write_text("{}\n")
    cache, cache_contract = _worker_cache_fixture(tmp_path, (BackendVariant.ORDINARY_XLA.value,))
    result_path = case_directory / "case_result.json"
    report_path = case_directory / "steady_trace.nsys-rep"
    sqlite_path = case_directory / "steady_trace.sqlite"
    monkeypatch.setattr(runner, "_worker_base_command", lambda *args, **kwargs: ("worker", "--case"))

    def successful_profile(command, **kwargs):
        result_path.write_text(
            json.dumps(
                {
                    "raw_samples": [
                        {
                            "sample_index": 0,
                            "backend_order": [BackendVariant.ORDINARY_XLA.value],
                        }
                    ]
                }
            )
        )
        report_path.write_bytes(b"retained profile report")
        return subprocess.CompletedProcess(command, 0, "", "")

    def export_without_contained_kernel(command):
        _write_nsys_trace_database(sqlite_path, include_memcpy_table=False)
        with sqlite3.connect(sqlite_path) as database:
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start = 500, end = 600")
            database.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start = 450, end = 475")

    monkeypatch.setattr(runner.subprocess, "run", successful_profile)
    monkeypatch.setattr(runner, "_run_retained", export_without_contained_kernel)

    with pytest.raises(ValueError, match="contains no associated CUDA kernels") as error:
        runner._run_profiled_case(config, "case-id", generated_manifest, case_directory, cache, cache_contract)

    diagnostic = _nsys_failure_diagnostic(error)
    assert diagnostic["report"] == {
        "bytes": report_path.stat().st_size,
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    assert diagnostic["sqlite"] == {
        "bytes": sqlite_path.stat().st_size,
        "sha256": hashlib.sha256(sqlite_path.read_bytes()).hexdigest(),
    }


def test_runner_numerical_failure_reports_logical_training_step_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    case = default_h100_contract_map_benchmark_plan().cases[0]
    value = np.zeros((1,), dtype=bfloat16)
    outputs = (value, value, value, value)
    source_candidate = SimpleNamespace(program=object())
    context = runner._WorkerCaseContext(
        case=case,
        inputs=(value, value, value, value),
        candidates={BackendVariant.SHUTTLE_SOURCE_ORDERED.value: source_candidate},
        artifacts={},
        libraries=(),
    )
    executables = {backend.value: lambda *args, outputs=outputs: outputs for backend in BackendVariant}
    monkeypatch.setattr(runner, "_real_algebra_reference", lambda *args: outputs)
    monkeypatch.setattr(
        contract_map_backend,
        "execute_contract_map_source_ordered_forward",
        lambda *args: SimpleNamespace(output=value),
    )
    monkeypatch.setattr(
        contract_map_backend,
        "execute_contract_map_source_ordered_reverse",
        lambda *args: SimpleNamespace(
            input_adjoint=value,
            first_weight_adjoint=value,
            second_weight_adjoint=value,
        ),
    )

    def numerical_output(index: int, repeats: Any, reference: Any) -> dict[str, Any]:
        maximum_absolute_error = 0.031251 if index == 0 else 0.0
        return {
            "maximum_absolute_error": maximum_absolute_error,
            "mean_absolute_error": 0.0,
            "maximum_ulp_distance": 0,
            "mean_ulp_distance": 0.0,
            "nonfinite_values": 0,
            "repeat_hashes": ["a" * 64, "a" * 64, "a" * 64],
            "pairwise_drift": [
                {
                    "left_repeat_index": left,
                    "right_repeat_index": right,
                    "maximum_absolute_error": 0.0,
                    "mean_absolute_error": 0.0,
                    "maximum_ulp_distance": 0,
                    "mean_ulp_distance": 0.0,
                }
                for left, right in ((0, 1), (0, 2), (1, 2))
            ],
        }

    monkeypatch.setattr(runner, "_output_numerical_evidence", numerical_output)
    jax = SimpleNamespace(block_until_ready=lambda values: None)

    with pytest.raises(ValueError) as error:
        runner._numerical_evidence(context, executables, jax=jax)

    diagnostic = str(error.value)
    assert f"case={case.case_id}" in diagnostic
    assert "backend=ordinary_xla" in diagnostic
    assert "boundary=logical_training_step" in diagnostic
    assert "output=forward" in diagnostic
    assert "metric=maximum_absolute_error" in diagnostic


def test_runner_source_ordered_failure_reports_bounded_worst_pair_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = default_h100_contract_map_benchmark_plan().cases[0]
    zero = np.zeros((1, 3), dtype=bfloat16)
    reference_dx = np.asarray([[1.0, 0.0004482269287109375, 2.0]], dtype=bfloat16)
    actual_dx = np.asarray([[1.0, -0.0003604888916015625, 2.0]], dtype=bfloat16)
    reference = (zero, reference_dx, zero, zero)
    context = runner._WorkerCaseContext(
        case=case,
        inputs=(zero, zero, zero, zero),
        candidates={BackendVariant.SHUTTLE_SOURCE_ORDERED.value: SimpleNamespace(program=object())},
        artifacts={},
        libraries=(),
    )
    executables = {
        BackendVariant.ORDINARY_XLA.value: lambda *args: reference,
        BackendVariant.SHUTTLE_SOURCE_ORDERED.value: lambda *args: (zero, actual_dx, zero, zero),
        BackendVariant.SHUTTLE_FAST.value: lambda *args: reference,
    }
    monkeypatch.setattr(runner, "_real_algebra_reference", lambda *args: reference)
    monkeypatch.setattr(
        contract_map_backend,
        "execute_contract_map_source_ordered_forward",
        lambda *args: SimpleNamespace(output=zero),
    )
    monkeypatch.setattr(
        contract_map_backend,
        "execute_contract_map_source_ordered_reverse",
        lambda *args: SimpleNamespace(
            input_adjoint=reference_dx,
            first_weight_adjoint=zero,
            second_weight_adjoint=zero,
        ),
    )

    with pytest.raises(ValueError) as error:
        runner._numerical_evidence(context, executables, jax=SimpleNamespace(block_until_ready=lambda values: None))

    diagnostic = str(error.value)
    assert len(diagnostic) <= 2048
    assert "backend=shuttle_source_ordered" in diagnostic
    assert "boundary=logical_training_step" in diagnostic
    assert "output=dx" in diagnostic
    assert "worst_index=0,1" in diagnostic
    assert "worst_actual_hex=0xb9bd" in diagnostic
    assert "worst_actual=-0.0003604888916015625" in diagnostic
    assert "worst_actual_sign=negative" in diagnostic
    assert "worst_actual_exponent=-12" in diagnostic
    assert "worst_actual_class=normal" in diagnostic
    assert "worst_reference_hex=0x39eb" in diagnostic
    assert "worst_reference=0.0004482269287109375" in diagnostic
    assert "worst_reference_sign=positive" in diagnostic
    assert "worst_reference_exponent=-12" in diagnostic
    assert "worst_reference_class=normal" in diagnostic
    assert "worst_absolute_error=0.0008087158203125" in diagnostic
    assert "worst_ulp_distance=29608" in diagnostic
    assert "finite_values=3" in diagnostic
    assert "exact_mismatches=1" in diagnostic
    assert "one_ulp_mismatches=1" in diagnostic
    assert "absolute_threshold=0.0078125" in diagnostic
    assert "absolute_mismatches=0" in diagnostic
    assert "array" not in diagnostic


def test_runner_merges_device_and_logical_timings_only_for_exact_copy_free_schedule() -> None:
    plan = default_h100_contract_map_benchmark_plan()
    worker_rows = []
    traces = []
    for row in plan.timing.steady_state_schedule:
        worker_rows.append(
            {
                "sample_index": row.sample_index,
                "backend_order": [backend.value for backend in row.backend_order],
                "logical_training_step_ns": {backend.value: 10_000 + row.sample_index for backend in BackendVariant},
            }
        )
        for backend in row.backend_order:
            traces.append(
                runner.TraceRange(
                    name=f"contract_map.steady.{row.sample_index}.{backend.value}",
                    ordered_kernel_names=("first", "second") * plan.timing.iterations_per_sample,
                    kernel_duration_ns=2_000 * plan.timing.iterations_per_sample,
                    device_to_device_count=0,
                    device_to_device_bytes=0,
                    host_to_device_count=0,
                    host_to_device_bytes=0,
                    device_to_host_count=0,
                    device_to_host_bytes=0,
                    unexpected_copy_count=0,
                )
            )

    raw, summary = runner.merge_trace_timing(plan, {"raw_samples": worker_rows}, tuple(traces))

    assert len(raw) == 24
    assert tuple(raw[0]["measurements_ns"]) == tuple(backend.value for backend in BackendVariant)
    assert raw[0]["measurements_ns"][BackendVariant.ORDINARY_XLA.value] == {
        "kernel_only": 2_000,
        "logical_training_step": 10_000,
    }
    assert summary[BackendVariant.SHUTTLE_FAST.value]["launch_count"] == 2
    assert summary[BackendVariant.SHUTTLE_FAST.value]["copies"] == {
        "device_to_device_count": 0,
        "device_to_device_bytes": 0,
        "host_to_device_count": 0,
        "host_to_device_bytes": 0,
        "device_to_host_count": 0,
        "device_to_host_bytes": 0,
        "unexpected_copy_count": 0,
    }
    for copy_fields in (
        {"device_to_device_count": 1, "device_to_device_bytes": 64},
        {"host_to_device_count": 1, "host_to_device_bytes": 64},
        {"device_to_host_count": 1, "device_to_host_bytes": 64},
    ):
        copied = replace(traces[0], **copy_fields)
        with pytest.raises(ValueError, match="unexpected copies") as error:
            runner.merge_trace_timing(plan, {"raw_samples": worker_rows}, (copied, *traces[1:]))
        for field, value in copy_fields.items():
            assert f"'{field}': {value}" in str(error.value)


def test_runner_retains_public_ordinary_xla_ptx_with_typed_absent_cubin(tmp_path: Path) -> None:
    dump_directory = tmp_path / "compile_dump"
    dump_directory.mkdir()
    (dump_directory / "module.ptx").write_text(".version 8.0\n")

    retained = runner.retain_ordinary_xla_cuda_artifacts(
        dump_directory=dump_directory,
        retained_directory=tmp_path / "retained",
    )

    assert Path(retained["ptx_path"]).read_text() == ".version 8.0\n"
    assert retained["cubin"] == {
        "availability": "unavailable",
        "unavailable_reason": "public_xla_dump_omits_cubin",
    }


def test_runner_rejects_missing_public_ordinary_xla_ptx(tmp_path: Path) -> None:
    dump_directory = tmp_path / "compile_dump"
    dump_directory.mkdir()

    with pytest.raises(RuntimeError, match="exactly one ordinary-XLA PTX"):
        runner.retain_ordinary_xla_cuda_artifacts(
            dump_directory=dump_directory,
            retained_directory=tmp_path / "retained",
        )


def test_worker_environment_replaces_inherited_cache_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/inherited/cache")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_CHECK_CONTENTS", "true")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY", "true")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_MAX_SIZE", "123")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "all")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "99")

    uncached = runner._worker_environment(tmp_path / "uncached_dump")
    isolated = runner._worker_environment(tmp_path / "cached_dump", tmp_path / "isolated_cache")

    assert "JAX_COMPILATION_CACHE_DIR" not in uncached
    assert "JAX_COMPILATION_CACHE_CHECK_CONTENTS" not in uncached
    assert "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY" not in uncached
    assert "JAX_COMPILATION_CACHE_MAX_SIZE" not in uncached
    assert "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in uncached
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in uncached
    assert isolated["JAX_COMPILATION_CACHE_DIR"] == str((tmp_path / "isolated_cache").resolve())
    assert isolated["JAX_COMPILATION_CACHE_CHECK_CONTENTS"] == "false"
    assert isolated["JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY"] == "false"
    assert isolated["JAX_COMPILATION_CACHE_MAX_SIZE"] == "-1"
    assert isolated["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == "none"
    assert isolated["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"


def test_compile_timing_stops_at_worker_compile_completion_before_delayed_publication(tmp_path: Path) -> None:
    output = tmp_path / "compile.json"
    clock = [100]

    def run(command, **kwargs):
        output.write_text(
            json.dumps(
                {
                    "compile_done_monotonic_ns": 400,
                    "persistent_cache_identity": "cache",
                    "final_hlo": "hlo",
                }
            )
        )
        clock[0] = 1_000
        return subprocess.CompletedProcess(command, 0, "", "")

    record = runner.run_timed_compile_worker_command(
        ("worker",),
        environment={},
        json_output=output,
        run=run,
        now=lambda: clock[0],
    )

    assert record["compile_ns"] == 300
    assert record["postcompile_ns"] == 600


def _compressed_cache_entry(executable: bytes, *, compile_time: int = 7) -> bytes:
    return zlib.compress(compile_time.to_bytes(4, "big") + executable)


def test_pinned_jax_0_10_1_cache_layout_is_four_byte_big_endian_time_prefix() -> None:
    executable = b"serialized executable"

    combined = pinned_jax_compilation_cache.combine_executable_and_time(executable, 0x01020304)

    assert combined == b"\x01\x02\x03\x04serialized executable"
    assert pinned_jax_compilation_cache.extract_executable_and_time(combined) == (executable, 0x01020304)


class _FakeMonitoring:
    def __init__(self) -> None:
        self.listener = None

    def register_event_listener(self, listener) -> None:
        self.listener = listener

    def unregister_event_listener(self, listener) -> None:
        assert self.listener is listener
        self.listener = None


def test_compile_cache_listener_is_scoped_across_compile_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monitoring = _FakeMonitoring()

    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(runner, "_compiled_backend", fail)

    with pytest.raises(RuntimeError, match="compile failed"):
        runner._compile_with_cache_events(
            SimpleNamespace(),
            "ordinary_xla",
            jax=SimpleNamespace(monitoring=monitoring),
        )
    assert monitoring.listener is None


def test_compile_worker_records_target_cache_key_executable_root_events_and_final_hlo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    key = f"jit_step-{'a' * 64}"
    target_entry = _compressed_cache_entry(b"serialized executable", compile_time=17)
    (cache_directory / f"{key}-cache").write_bytes(target_entry)
    (cache_directory / f"jit_helper-{'b' * 64}-cache").write_bytes(_compressed_cache_entry(b"helper"))
    compiled = SimpleNamespace(
        __call__=lambda *args: ("output",),
        as_text=lambda: "authoritative final HLO",
    )

    class FakeCompiled:
        def __call__(self, *args):
            return compiled.__call__(*args)

        def as_text(self):
            return compiled.as_text()

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache_directory))
    compression_checks = []
    monkeypatch.setattr(runner, "_require_pinned_cache_compression_runtime", lambda: compression_checks.append(True))
    monitoring = _FakeMonitoring()

    def compile_backend(*args, **kwargs):
        assert monitoring.listener is not None
        monitoring.listener(runner._CACHE_EVENT_NAMES[0])
        monitoring.listener(runner._CACHE_EVENT_NAMES[2])
        return FakeCompiled(), 123

    monkeypatch.setattr(runner, "_compiled_backend", compile_backend)
    context = SimpleNamespace(inputs=("x", "w0", "w1", "do"))
    args = SimpleNamespace(case_id="case", backend="ordinary_xla", cache_kind="compile")

    record = runner._run_compile_worker(
        args,
        context,
        jax=SimpleNamespace(block_until_ready=lambda value: None, monitoring=monitoring),
    )

    root_identity = runner._persistent_cache_snapshot(runner._persistent_cache_files(cache_directory)).root_identity
    executable_identity = hashlib.sha256(b"serialized executable").hexdigest()
    contract_identity = hashlib.sha256(key.encode() + bytes.fromhex(executable_identity)).hexdigest()
    assert record["persistent_cache_identity"] == contract_identity
    assert record["persistent_cache_key"] == key
    assert record["persistent_cache_serialized_executable_sha256"] == executable_identity
    assert record["persistent_cache_compile_time"] == 17
    assert record["persistent_cache_entry_sha256"] == hashlib.sha256(target_entry).hexdigest()
    assert record["persistent_cache_root_identity"] == root_identity
    assert record["persistent_cache_compression"] == "zlib"
    assert record["persistent_cache_events"] == {
        runner._CACHE_EVENT_NAMES[0]: 1,
        runner._CACHE_EVENT_NAMES[1]: 0,
        runner._CACHE_EVENT_NAMES[2]: 1,
    }
    assert record["persistent_cache_file_count"] == 2
    assert record["persistent_cache_total_bytes"] == len(target_entry) + len(_compressed_cache_entry(b"helper"))
    assert record["final_hlo"] == "authoritative final HLO"
    assert compression_checks == [True]


def test_cached_worker_requires_scoped_public_hit_and_unchanged_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = BackendVariant.ORDINARY_XLA.value
    cache, contract_path = _worker_cache_fixture(tmp_path, (backend,))
    contract = json.loads(contract_path.read_text())
    monitoring = _FakeMonitoring()

    class Compiled:
        def as_text(self):
            return f"hlo-{backend}"

    def cached_compile(*args, **kwargs):
        assert monitoring.listener is not None
        monitoring.listener(runner._CACHE_EVENT_NAMES[0])
        monitoring.listener(runner._CACHE_EVENT_NAMES[1])
        return Compiled(), 123

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))
    monkeypatch.setattr(runner, "_compiled_backend", cached_compile)
    compiled, evidence = runner._cache_hit_executable(
        SimpleNamespace(),
        SimpleNamespace(),
        backend,
        contract,
        jax=SimpleNamespace(monitoring=monitoring),
    )
    assert compiled.as_text() == f"hlo-{backend}"
    assert evidence["persistent_cache_events"] == dict(zip(runner._CACHE_EVENT_NAMES, (1, 1, 0), strict=True))

    def fallback_compile(*args, **kwargs):
        assert monitoring.listener is not None
        monitoring.listener(runner._CACHE_EVENT_NAMES[0])
        monitoring.listener(runner._CACHE_EVENT_NAMES[2])
        return Compiled(), 123

    monkeypatch.setattr(runner, "_compiled_backend", fallback_compile)
    with pytest.raises(ValueError, match="did not prove one public persistent-cache hit"):
        runner._cache_hit_executable(
            SimpleNamespace(),
            SimpleNamespace(),
            backend,
            contract,
            jax=SimpleNamespace(monitoring=monitoring),
        )


def test_cached_worker_rejects_context_or_compile_mutation_of_cloned_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = BackendVariant.ORDINARY_XLA.value
    cache, contract_path = _worker_cache_fixture(tmp_path, (backend,))
    contract = json.loads(contract_path.read_text())
    monitoring = _FakeMonitoring()

    def mutate_cache(*args, **kwargs):
        (cache / f"jit_helper-{'e' * 64}-cache").write_bytes(_compressed_cache_entry(b"mutation"))
        assert monitoring.listener is not None
        monitoring.listener(runner._CACHE_EVENT_NAMES[0])
        monitoring.listener(runner._CACHE_EVENT_NAMES[1])
        return SimpleNamespace(as_text=lambda: f"hlo-{backend}"), 123

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))
    monkeypatch.setattr(runner, "_compiled_backend", mutate_cache)
    with pytest.raises(ValueError, match="differs from its sealed source"):
        runner._cache_hit_executable(
            SimpleNamespace(),
            SimpleNamespace(),
            backend,
            contract,
            jax=SimpleNamespace(monitoring=monitoring),
        )


def test_persistent_cache_target_rejects_metadata_nested_and_ambiguous_entries(tmp_path: Path) -> None:
    key = f"jit_step-{'a' * 64}"
    (tmp_path / f"{key}-cache").write_bytes(_compressed_cache_entry(b"executable"))

    atime = tmp_path / f"{key}-atime"
    atime.write_bytes(b"timestamp")
    with pytest.raises(RuntimeError, match="unexpected entry") as raised:
        runner._persistent_cache_target(tmp_path)
    assert atime.name not in str(raised.value)
    atime.unlink()

    nested = tmp_path / "xla_gpu_per_fusion_autotune_cache_dir"
    nested.mkdir()
    with pytest.raises(RuntimeError, match="only flat regular cache entries"):
        runner._persistent_cache_target(tmp_path)
    nested.rmdir()

    (tmp_path / f"jit_step-{'b' * 64}-cache").write_bytes(_compressed_cache_entry(b"other"))
    with pytest.raises(RuntimeError, match="exactly one jit_step"):
        runner._persistent_cache_target(tmp_path)


def test_persistent_cache_target_strips_only_pinned_compile_time_prefix(tmp_path: Path) -> None:
    executable = b"same serialized executable"
    identities = []
    for index, compile_time in enumerate((1, 257)):
        root = tmp_path / str(index)
        root.mkdir()
        key = f"jit_step-{'a' * 64}"
        compressed = _compressed_cache_entry(executable, compile_time=compile_time)
        (root / f"{key}-cache").write_bytes(compressed)
        target, _ = runner._persistent_cache_target(root)
        identities.append(target)

    assert identities[0].compile_time == 1
    assert identities[1].compile_time == 257
    assert identities[0].compressed_entry_sha256 != identities[1].compressed_entry_sha256
    assert identities[0].serialized_executable_sha256 == identities[1].serialized_executable_sha256
    assert identities[0].serialized_executable_sha256 == hashlib.sha256(executable).hexdigest()


def test_persistent_cache_target_rejects_wrong_compression_and_oversized_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = f"jit_step-{'a' * 64}"
    path = tmp_path / f"{key}-cache"
    path.write_bytes(b"not zlib")
    with pytest.raises(RuntimeError, match="pinned zlib"):
        runner._persistent_cache_target(tmp_path)

    path.write_bytes(_compressed_cache_entry(b"executable"))
    monkeypatch.setattr(runner, "_MAX_PERSISTENT_CACHE_ENTRY_BYTES", path.stat().st_size - 1)
    with pytest.raises(RuntimeError, match="invalid bounded size"):
        runner._persistent_cache_target(tmp_path)


def test_persistent_cache_target_rejects_trailing_stream_high_expansion_and_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = f"jit_step-{'a' * 64}"
    path = tmp_path / f"{key}-cache"
    path.write_bytes(_compressed_cache_entry(b"executable") + b"trailing")
    with pytest.raises(RuntimeError, match="one complete bounded zlib stream"):
        runner._persistent_cache_target(tmp_path)

    path.write_bytes(_compressed_cache_entry(b"x" * 1_024))
    monkeypatch.setattr(runner, "_MAX_SERIALIZED_EXECUTABLE_BYTES", 32)
    with pytest.raises(RuntimeError, match="one complete bounded zlib stream"):
        runner._persistent_cache_target(tmp_path)

    path.unlink()
    os.mkfifo(path)
    with pytest.raises(RuntimeError, match="only flat regular cache entries"):
        runner._persistent_cache_target(tmp_path)


def test_persistent_cache_target_bounds_all_entries_and_root_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _compressed_cache_entry(b"executable")
    helper = _compressed_cache_entry(b"helper")
    (tmp_path / f"jit_step-{'a' * 64}-cache").write_bytes(target)
    (tmp_path / f"jit_helper-{'b' * 64}-cache").write_bytes(helper)

    monkeypatch.setattr(runner, "_MAX_PERSISTENT_CACHE_FILES", 1)
    with pytest.raises(RuntimeError, match="file-count bound"):
        runner._persistent_cache_target(tmp_path)
    monkeypatch.setattr(runner, "_MAX_PERSISTENT_CACHE_FILES", 2)
    monkeypatch.setattr(runner, "_MAX_PERSISTENT_CACHE_ROOT_BYTES", len(target) + len(helper) - 1)
    with pytest.raises(RuntimeError, match="total-byte bound"):
        runner._persistent_cache_target(tmp_path)


def test_canonical_cache_snapshot_keeps_one_full_base_and_overlays_only_distinct_targets(tmp_path: Path) -> None:
    sources = []
    records = {}
    for index, backend in enumerate(("ordinary_xla", "shuttle_source_ordered", "shuttle_fast")):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        key = f"jit_step-{index + 1:x}" + "a" * 63
        executable = f"executable-{backend}".encode()
        entry = _compressed_cache_entry(executable, compile_time=index + 1)
        (source / f"{key}-cache").write_bytes(entry)
        (source / f"jit_helper-{'f' * 64}-cache").write_bytes(_compressed_cache_entry(f"helper-{index}".encode()))
        target, files = runner._persistent_cache_target(source)
        source_snapshot = runner._persistent_cache_snapshot(files)
        snapshot = tmp_path / f"canonical-{index}"
        runner._seal_canonical_target_snapshot(
            source,
            snapshot,
            {
                "persistent_cache_entry_sha256": target.compressed_entry_sha256,
                "persistent_cache_file_count": source_snapshot.file_count,
                "persistent_cache_key": key,
                "persistent_cache_root_identity": source_snapshot.root_identity,
                "persistent_cache_serialized_executable_sha256": target.serialized_executable_sha256,
                "persistent_cache_total_bytes": source_snapshot.total_bytes,
            },
        )
        assert len(runner._persistent_cache_files(snapshot)) == len(files)
        sources.append(snapshot)
        records[backend] = {
            "persistent_cache_entry_sha256": target.compressed_entry_sha256,
            "persistent_cache_key": key,
            "persistent_cache_serialized_executable_sha256": target.serialized_executable_sha256,
            "final_hlo": f"hlo-{backend}",
        }

    merged = tmp_path / "merged"
    runner._merge_canonical_target_snapshots(sources, merged)
    files = dict(runner._persistent_cache_files(merged))
    assert files[f"jit_helper-{'f' * 64}-cache"] == _compressed_cache_entry(b"helper-0")
    assert {name for name in files if name.startswith("jit_step-")} == {
        f"{record['persistent_cache_key']}-cache" for record in records.values()
    }
    contract = runner._write_worker_cache_contract(tmp_path / "contract.json", merged, records)
    assert json.loads(contract.read_text())["snapshot"]["file_count"] == 4

    clone = tmp_path / "clone"
    runner._clone_cache_snapshot(merged, clone)
    first_target = next(path for path in clone.iterdir() if path.name.startswith("jit_step-"))
    first_target.write_bytes(b"changed clone")
    assert dict(runner._persistent_cache_files(merged))[first_target.name] != b"changed clone"


def test_worker_cache_contract_rejects_duplicate_backend_target_key(tmp_path: Path) -> None:
    cache, contract = _worker_cache_fixture(tmp_path, ("ordinary_xla",))
    payload = json.loads(contract.read_text())
    payload["backends"]["shuttle_fast"] = dict(payload["backends"]["ordinary_xla"])
    records = {
        backend: {
            "persistent_cache_entry_sha256": record["compressed_entry_sha256"],
            "persistent_cache_key": record["cache_key"],
            "persistent_cache_serialized_executable_sha256": record["serialized_executable_sha256"],
            "final_hlo": "hlo-ordinary_xla",
        }
        for backend, record in payload["backends"].items()
    }
    with pytest.raises(ValueError, match="one distinct target"):
        runner._write_worker_cache_contract(tmp_path / "duplicate.json", cache, records)


def test_canonical_snapshot_rejects_helper_mutation_after_compile_worker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    key = f"jit_step-{'a' * 64}"
    (source / f"{key}-cache").write_bytes(_compressed_cache_entry(b"executable"))
    helper = source / f"jit_helper-{'b' * 64}-cache"
    helper.write_bytes(_compressed_cache_entry(b"helper"))
    target, files = runner._persistent_cache_target(source)
    snapshot = runner._persistent_cache_snapshot(files)
    record = {
        "persistent_cache_entry_sha256": target.compressed_entry_sha256,
        "persistent_cache_file_count": snapshot.file_count,
        "persistent_cache_key": key,
        "persistent_cache_root_identity": snapshot.root_identity,
        "persistent_cache_serialized_executable_sha256": target.serialized_executable_sha256,
        "persistent_cache_total_bytes": snapshot.total_bytes,
    }
    helper.write_bytes(_compressed_cache_entry(b"changed helper"))
    with pytest.raises(ValueError, match="root changed after its compile worker"):
        runner._seal_canonical_target_snapshot(source, tmp_path / "canonical", record)


def test_cache_compression_contract_rejects_optional_zstandard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda distribution: "0.25.0")

    with pytest.raises(RuntimeError, match="requires zstandard to be absent"):
        runner._require_pinned_cache_compression_runtime()


def test_cache_compression_contract_accepts_pinned_python_without_zstandard(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(distribution: str) -> str:
        raise runner.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(runner.importlib.metadata, "version", absent)

    runner._require_pinned_cache_compression_runtime()


def test_pinned_jax_target_key_and_public_events_across_fresh_and_hit_processes(tmp_path: Path) -> None:
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "zstandard.py").write_text("raise ImportError('disabled by pinned-runtime test')\n")
    script = "\n".join(
        (
            "import json, sys",
            "import jax",
            "import jax.numpy as jnp",
            "from lib.tile_lifetime.benchmarks import h100_contract_map_backend_runner as runner",
            "counts = {name: 0 for name in runner._CACHE_EVENT_NAMES}",
            "def listener(event, **metadata):",
            "    if event in counts: counts[event] += 1",
            "def step(value):",
            "    return jnp.tanh(value @ value)",
            "value = jnp.ones((8, 8), dtype=jnp.float32)",
            "jax.block_until_ready(value)",
            "jax.monitoring.register_event_listener(listener)",
            "try:",
            "    compiled = jax.jit(step).lower(value).compile()",
            "finally:",
            "    jax.monitoring.unregister_event_listener(listener)",
            "jax.block_until_ready(compiled(value))",
            "target, entries = runner._persistent_cache_target(runner.Path(sys.argv[1]))",
            "print(json.dumps({'events': counts, 'key': target.cache_key, "
            "'executable': target.serialized_executable_sha256, 'entry_count': len(entries)}, sort_keys=True))",
        )
    )
    base_environment = {
        **os.environ,
        **runner._CACHE_ENVIRONMENT,
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    def run(cache: Path) -> dict[str, object]:
        cache.mkdir(exist_ok=True)
        completed = subprocess.run(
            (sys.executable, "-c", script, str(cache)),
            check=True,
            capture_output=True,
            text=True,
            env={**base_environment, "JAX_COMPILATION_CACHE_DIR": str(cache)},
        )
        return json.loads(completed.stdout)

    cold_records = [run(tmp_path / f"fresh-{index}") for index in range(3)]
    hit_record = run(tmp_path / "fresh-0")

    assert all(
        record["events"] == dict(zip(runner._CACHE_EVENT_NAMES, (1, 0, 1), strict=True)) for record in cold_records
    )
    assert hit_record["events"] == dict(zip(runner._CACHE_EVENT_NAMES, (1, 1, 0), strict=True))
    assert len({record["key"] for record in cold_records}) == 1
    assert len({record["executable"] for record in cold_records}) == 1
    assert cold_records[0]["key"] == hit_record["key"]
    assert cold_records[0]["executable"] == hit_record["executable"]
    assert cold_records[0]["entry_count"] == hit_record["entry_count"]


@pytest.mark.parametrize("compile_done", [99, 1_001, True])
def test_compile_timing_rejects_worker_timestamp_outside_spawn_and_exit(tmp_path: Path, compile_done: object) -> None:
    output = tmp_path / "compile.json"
    clock = [100]

    def run(command, **kwargs):
        output.write_text(json.dumps({"compile_done_monotonic_ns": compile_done}))
        clock[0] = 1_000
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="between coordinator spawn and exit"):
        runner.run_timed_compile_worker_command(
            ("worker",),
            environment={},
            json_output=output,
            run=run,
            now=lambda: clock[0],
        )


def _cache_root_record(
    executable_identity: str,
    *,
    cache_key_digest: str = "a" * 64,
    phase: str = "compile",
    final_hlo: str = "stable final HLO",
    file_count: int = 2,
    total_bytes: int = 4096,
    root_identity: str = "c" * 64,
    compile_time: int = 7,
) -> dict[str, object]:
    cache_key = f"jit_step-{cache_key_digest}"
    identity = hashlib.sha256(cache_key.encode() + bytes.fromhex(executable_identity)).hexdigest()
    return {
        "persistent_cache_compression": "zlib",
        "persistent_cache_compile_time": compile_time,
        "persistent_cache_entry_sha256": "b" * 64,
        "persistent_cache_events": {
            runner._CACHE_EVENT_NAMES[0]: 1,
            runner._CACHE_EVENT_NAMES[1]: int(phase != "compile"),
            runner._CACHE_EVENT_NAMES[2]: int(phase == "compile"),
        },
        "persistent_cache_identity": identity,
        "persistent_cache_file_count": file_count,
        "persistent_cache_key": cache_key,
        "persistent_cache_root_identity": root_identity,
        "persistent_cache_serialized_executable_sha256": executable_identity,
        "persistent_cache_total_bytes": total_bytes,
        "final_hlo": final_hlo,
    }


def _materialized_cache_record(
    environment: Mapping[str, str],
    executable: bytes,
    *,
    phase: str,
    final_hlo: str = "stable final HLO",
) -> dict[str, object]:
    cache = Path(environment["JAX_COMPILATION_CACHE_DIR"])
    cache.mkdir(parents=True, exist_ok=True)
    key = f"jit_step-{'a' * 64}"
    target_path = cache / f"{key}-cache"
    if not target_path.exists():
        target_path.write_bytes(_compressed_cache_entry(executable))
    target, files = runner._persistent_cache_target(cache, key)
    snapshot = runner._persistent_cache_snapshot(files)
    return {
        **_cache_root_record(
            target.serialized_executable_sha256,
            phase=phase,
            final_hlo=final_hlo,
            file_count=snapshot.file_count,
            total_bytes=snapshot.total_bytes,
            root_identity=snapshot.root_identity,
        ),
        "persistent_cache_compile_time": target.compile_time,
        "persistent_cache_entry_sha256": target.compressed_entry_sha256,
    }


def _canonical_cache_record(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    cache = tmp_path / "canonical-source"
    record = _materialized_cache_record(
        {"JAX_COMPILATION_CACHE_DIR": str(cache)},
        b"canonical",
        phase="compile",
        final_hlo="private HLO canonical",
    )
    return cache, record


def test_cache_protocol_accepts_fresh_executable_nondeterminism_but_rejects_cached_mismatch() -> None:
    executable_identity = "a" * 64
    compile_records = tuple(_cache_root_record(executable_identity, phase="compile") for _ in range(3))
    cold_records = tuple(_cache_root_record(executable_identity, phase="cold") for _ in range(3))
    hit_records = tuple(_cache_root_record(executable_identity, phase="hit") for _ in range(3))
    identity = str(compile_records[0]["persistent_cache_identity"])

    assert (
        runner.validated_cache_protocol_identity(
            compile_records,
            cold_records,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )
        == identity
    )
    mismatched = (*compile_records[:2], _cache_root_record("d" * 64, phase="compile"))
    assert (
        runner.validated_cache_protocol_identity(
            mismatched,
            cold_records,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )
        == identity
    )
    mismatched_cold = (*cold_records[:2], _cache_root_record("d" * 64, phase="cold"))
    with pytest.raises(ValueError, match="canonical executable"):
        runner.validated_cache_protocol_identity(
            mismatched,
            mismatched_cold,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )


def test_cache_protocol_uses_unmeasured_canonical_record_not_any_fresh_compile() -> None:
    canonical = _cache_root_record("f" * 64, phase="compile", root_identity="e" * 64)
    compile_records = tuple(
        _cache_root_record(f"{index + 1:x}" * 64, phase="compile", root_identity=f"{index + 1:x}" * 64)
        for index in range(3)
    )

    def cloned(phase: str) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                **_cache_root_record("f" * 64, phase=phase, root_identity="e" * 64),
                "persistent_cache_entry_sha256": canonical["persistent_cache_entry_sha256"],
            }
            for _ in range(3)
        )

    expected = hashlib.sha256(str(canonical["persistent_cache_key"]).encode() + bytes.fromhex("f" * 64)).hexdigest()
    assert (
        runner.validated_cache_protocol_identity(
            compile_records,
            cloned("cold"),
            cloned("hit"),
            canonical_record=canonical,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )
        == expected
    )


def test_cache_protocol_accepts_fresh_root_differences_but_requires_exact_cloned_roots() -> None:
    executable = "a" * 64

    def records(phase: str) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                **_cache_root_record(
                    executable,
                    phase=phase,
                    compile_time=index + 1,
                    root_identity=f"{index + 1:x}" * 64,
                    total_bytes=4_000 + index,
                ),
                "persistent_cache_entry_sha256": f"{index + 4:x}" * 64,
            }
            for index in range(3)
        )

    compile_records = records("compile")
    canonical_entry = compile_records[0]["persistent_cache_entry_sha256"]
    canonical_root = compile_records[0]["persistent_cache_root_identity"]

    def cloned(phase: str) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                **_cache_root_record(executable, phase=phase, root_identity=str(canonical_root)),
                "persistent_cache_entry_sha256": canonical_entry,
            }
            for _ in range(3)
        )

    cold_records = cloned("cold")
    hit_records = cloned("hit")

    assert (
        runner.validated_cache_protocol_identity(
            compile_records,
            cold_records,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )
        == compile_records[0]["persistent_cache_identity"]
    )


def test_cache_protocol_rejects_fresh_target_key_difference_but_records_executable_difference() -> None:
    compile_records = [_cache_root_record("a" * 64, phase="compile") for _ in range(3)]
    cold_records = tuple(_cache_root_record("a" * 64, phase="cold") for _ in range(3))
    hit_records = tuple(_cache_root_record("a" * 64, phase="hit") for _ in range(3))

    compile_records[2] = _cache_root_record("a" * 64, cache_key_digest="d" * 64, phase="compile")
    with pytest.raises(ValueError, match="one target key"):
        runner.validated_cache_protocol_identity(
            compile_records,
            cold_records,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )
    compile_records[2] = _cache_root_record("d" * 64, phase="compile")
    runner.validated_cache_protocol_identity(
        compile_records,
        cold_records,
        hit_records,
        case_id="contract_map_case",
        backend="ordinary_xla",
        required_processes=3,
    )


@pytest.mark.parametrize(
    ("phase", "events"),
    (
        ("compile", (0, 0, 1)),
        ("compile", (1, 1, 0)),
        ("cold", (1, 0, 0)),
        ("hit", (1, 0, 1)),
        ("hit", (1, 1, 1)),
    ),
)
def test_cache_protocol_requires_exact_public_hit_miss_events(phase: str, events: tuple[int, int, int]) -> None:
    groups = {
        "compile": [_cache_root_record("a" * 64, phase="compile") for _ in range(3)],
        "cold": [_cache_root_record("a" * 64, phase="cold") for _ in range(3)],
        "hit": [_cache_root_record("a" * 64, phase="hit") for _ in range(3)],
    }
    groups[phase][1]["persistent_cache_events"] = dict(zip(runner._CACHE_EVENT_NAMES, events, strict=True))

    with pytest.raises(ValueError, match="invalid public cache events"):
        runner.validated_cache_protocol_identity(
            groups["compile"],
            groups["cold"],
            groups["hit"],
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )


def test_cache_protocol_failure_reports_closed_nine_root_equality_partition() -> None:
    executable_a = "a" * 64
    executable_b = "d" * 64
    compile_records = (
        _cache_root_record(executable_a, final_hlo="compile HLO", file_count=2, total_bytes=100),
        _cache_root_record(executable_b, final_hlo="changed HLO", file_count=3, total_bytes=200),
        _cache_root_record(executable_a, final_hlo="compile HLO", file_count=2, total_bytes=100),
    )
    cold_records = (
        _cache_root_record(executable_a, phase="cold", final_hlo="compile HLO"),
        _cache_root_record(executable_b, phase="cold", final_hlo="changed cached HLO"),
        _cache_root_record(executable_a, phase="cold", final_hlo="compile HLO"),
    )
    hit_records = tuple(_cache_root_record(executable_a, phase="hit", final_hlo="compile HLO") for _ in range(3))

    with pytest.raises(ValueError) as raised:
        runner.validated_cache_protocol_identity(
            compile_records,
            cold_records,
            hit_records,
            case_id="contract_map_9836cdbed389db24",
            backend="ordinary_xla",
            required_processes=3,
        )

    diagnostic = json.loads(str(raised.value).split("diagnostic=", 1)[1])
    assert diagnostic["case_id"] == "contract_map_9836cdbed389db24"
    assert diagnostic["backend"] == "ordinary_xla"
    assert diagnostic["schema_version"] == 3
    assert diagnostic["canonical_equality_partition"] == "class_0"
    assert diagnostic["expected_cached_equality_partitions"] == 1
    assert diagnostic["fresh_compile_equality_partitions"] == 2
    assert diagnostic["observed_equality_partitions"] == 2
    root_fields = diagnostic["root_fields"]
    roots = tuple(dict(zip(root_fields, root, strict=True)) for root in diagnostic["roots"])
    classes = tuple(dict(zip(diagnostic["class_fields"], record, strict=True)) for record in diagnostic["classes"])
    assert tuple(f"{root['phase']}[{root['index']}]" for root in roots) == (
        "compile[0]",
        "compile[1]",
        "compile[2]",
        "cold[0]",
        "cold[1]",
        "cold[2]",
        "hit[0]",
        "hit[1]",
        "hit[2]",
    )
    assert tuple(root["equality_partition"] for root in roots) == (
        "class_0",
        "class_1",
        "class_0",
        "class_0",
        "class_1",
        "class_0",
        "class_0",
        "class_0",
        "class_0",
    )
    assert roots[4] == {
        "equality_partition": "class_1",
        "final_hlo_sha256": hashlib.sha256(b"changed cached HLO").hexdigest(),
        "index": 1,
        "persistent_cache_file_count": 2,
        "persistent_cache_total_bytes": 4096,
        "phase": "cold",
    }
    assert classes[1] == {
        "equality_partition": "class_1",
        "cache_key_digest": "a" * 64,
        "serialized_executable_sha256": executable_b,
    }
    assert "compile HLO" not in str(raised.value)


def test_run_cache_protocol_propagates_bounded_root_diagnostic_from_production_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    phase_counts = {"compile": 0, "cold": 0, "hit": 0}

    def worker_command(*args, cache_kind: str, **kwargs) -> tuple[str, str]:
        return ("worker", cache_kind)

    def worker_result(command, **kwargs):
        phase = command[1]
        index = phase_counts[phase]
        phase_counts[phase] += 1
        record = _materialized_cache_record(
            kwargs["environment"], b"canonical", phase=phase, final_hlo=f"private HLO {phase} {index}"
        )
        if phase == "cold" and index == 1:
            record["persistent_cache_serialized_executable_sha256"] = "d" * 64
            key = str(record["persistent_cache_key"])
            record["persistent_cache_identity"] = hashlib.sha256(key.encode() + bytes.fromhex("d" * 64)).hexdigest()
        return {
            **record,
            "private_cache_path": f"/secret/cache/{phase}/{index}",
        }

    monkeypatch.setattr(runner, "_worker_base_command", worker_command)
    monkeypatch.setattr(runner, "run_timed_compile_worker_command", worker_result)
    canonical_root, canonical_record = _canonical_cache_record(tmp_path)

    with pytest.raises(ValueError) as raised:
        runner._run_cache_protocol(
            config,
            "contract_map_9836cdbed389db24",
            "ordinary_xla",
            tmp_path / "generated.json",
            tmp_path / "cache_protocol",
            canonical_root,
            canonical_record,
        )

    message = str(raised.value)
    diagnostic = json.loads(message.split("diagnostic=", 1)[1])
    assert len(message) <= runner._MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS
    assert diagnostic["case_id"] == "contract_map_9836cdbed389db24"
    assert diagnostic["backend"] == "ordinary_xla"
    assert len(diagnostic["roots"]) == 9
    assert "private HLO" not in message
    assert "/secret/cache" not in message


def test_canonical_cache_preparation_is_unmeasured_and_seals_exact_worker_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    monkeypatch.setattr(runner, "_worker_base_command", lambda *args, **kwargs: ("worker", "canonical"))
    timed_calls = []
    monkeypatch.setattr(
        runner,
        "run_timed_compile_worker_command",
        lambda *args, **kwargs: timed_calls.append((args, kwargs)),
    )

    def run_worker(command, *, environment, json_output):
        assert command == ("worker", "canonical")
        record = _materialized_cache_record(environment, b"canonical", phase="compile", final_hlo="canonical HLO")
        json_output.write_text(json.dumps(record))
        return record

    monkeypatch.setattr(runner, "_run_worker_command", run_worker)
    prepared = runner._prepare_canonical_cache(
        config,
        "contract_map_case",
        BackendVariant.ORDINARY_XLA.value,
        tmp_path / "generated.json",
        tmp_path / "canonical-preparation",
    )

    assert timed_calls == []
    sealed_root = Path(prepared["canonical_cache_root"])
    snapshot = runner._persistent_cache_snapshot(runner._persistent_cache_files(sealed_root))
    assert snapshot.root_identity == prepared["canonical_cache_root_identity"]
    assert prepared["record"]["persistent_cache_events"] == dict(zip(runner._CACHE_EVENT_NAMES, (1, 0, 1), strict=True))


def test_coordinator_starts_no_timed_cache_protocol_before_case_numerical_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    monkeypatch.setattr(
        runner,
        "require_clean_h100_preflight",
        lambda config: runner.PreflightEvidence(
            source_sha=config.source_sha,
            gpu_name="H100",
            compute_capability="9.0",
            architecture="sm_90a",
            tools=(),
            source_tree=None,
            source_capsule_manifest_sha256=None,
        ),
    )
    monkeypatch.setattr(runner, "compile_generated_candidates", lambda config: ())
    monkeypatch.setattr(runner, "audit_imported_local_modules", lambda config: None)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: SimpleNamespace(generated_contract_map_candidates=lambda: ()),
    )
    order = []

    def prepare(config, case_id, backend, generated_manifest, directory):
        order.append(f"canonical:{backend}")
        root = directory / "sealed_root"
        root.mkdir(parents=True)
        key = f"jit_step-{len(order):x}" + "a" * 63
        executable = backend.encode()
        entry = _compressed_cache_entry(executable)
        (root / f"{key}-cache").write_bytes(entry)
        target, files = runner._persistent_cache_target(root)
        snapshot = runner._persistent_cache_snapshot(files)
        return {
            "canonical_cache_root": str(root),
            "canonical_cache_root_identity": snapshot.root_identity,
            "record": {
                **_cache_root_record(
                    target.serialized_executable_sha256,
                    cache_key_digest=key.removeprefix("jit_step-"),
                    phase="compile",
                    final_hlo=f"hlo-{backend}",
                    file_count=snapshot.file_count,
                    total_bytes=snapshot.total_bytes,
                    root_identity=snapshot.root_identity,
                ),
                "persistent_cache_entry_sha256": target.compressed_entry_sha256,
            },
        }

    monkeypatch.setattr(runner, "_prepare_canonical_cache", prepare)
    monkeypatch.setattr(
        runner,
        "_run_cache_protocol",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("timed cache protocol ran before case")),
    )

    def stop_at_case(*args, **kwargs):
        order.append("case")
        raise RuntimeError("case boundary reached")

    monkeypatch.setattr(runner, "_run_profiled_case", stop_at_case)
    with pytest.raises(RuntimeError, match="case boundary reached"):
        runner.run_coordinator(config)

    assert order[-1] == "case"
    assert len(order) == len(BackendVariant) + 1


def test_cache_protocol_nine_distinct_roots_emit_complete_diagnostic_under_bound() -> None:
    records = {
        phase: tuple(
            _cache_root_record(
                f"{phase_index * 3 + index + 1:x}" * 64,
                cache_key_digest=f"{phase_index * 3 + index + 10:x}"[-1] * 64,
                phase=phase,
                final_hlo="private\x00HLO\U0001f680/path/" + ("x" * 10_000) + str(index),
                file_count=2**63 - 1,
                total_bytes=2**63 - 1,
            )
            for index in range(3)
        )
        for phase_index, phase in enumerate(("compile", "cold", "hit"))
    }

    with pytest.raises(ValueError) as raised:
        runner.validated_cache_protocol_identity(
            records["compile"],
            records["cold"],
            records["hit"],
            case_id="c" * 128,
            backend="b" * 64,
            required_processes=3,
        )

    message = str(raised.value)
    diagnostic = json.loads(message.split("diagnostic=", 1)[1])
    assert 3_000 < len(message) <= runner._MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS
    assert diagnostic["case_id"] == "c" * 128
    assert diagnostic["backend"] == "b" * 64
    assert diagnostic["observed_equality_partitions"] == 9
    assert len(diagnostic["classes"]) == 9
    assert len(diagnostic["roots"]) == 9
    assert "private" not in message
    assert "\\u0000" not in message
    assert "\\ud83d" not in message


def test_run_cache_protocol_maximal_reachable_partition_emits_bounded_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    phase_counts = {"compile": 0, "cold": 0, "hit": 0}

    def worker_command(*args, cache_kind: str, **kwargs) -> tuple[str, str]:
        return "worker", cache_kind

    def worker_result(command, **kwargs):
        phase = command[1]
        index = phase_counts[phase]
        phase_counts[phase] += 1
        class_index = index + (0 if phase == "compile" else 3)
        record = _materialized_cache_record(
            kwargs["environment"],
            f"fresh-{index}".encode() if phase == "compile" else b"fresh-0",
            phase=phase,
            final_hlo="private\x00HLO\U0001f680/" + ("x" * 10_000),
        )
        if phase != "compile":
            executable = f"{class_index + 1:x}" * 64
            record["persistent_cache_serialized_executable_sha256"] = executable
            key = str(record["persistent_cache_key"])
            record["persistent_cache_identity"] = hashlib.sha256(key.encode() + bytes.fromhex(executable)).hexdigest()
        if phase != "compile":
            record["persistent_cache_file_count"] = 2**63 - 1
            record["persistent_cache_total_bytes"] = 2**63 - 1
        return record

    monkeypatch.setattr(runner, "_worker_base_command", worker_command)
    monkeypatch.setattr(runner, "run_timed_compile_worker_command", worker_result)
    canonical_root, canonical_record = _canonical_cache_record(tmp_path)

    with pytest.raises(ValueError) as raised:
        runner._run_cache_protocol(
            config,
            "c" * 128,
            "b" * 64,
            tmp_path / "generated.json",
            tmp_path / "cache_protocol",
            canonical_root,
            canonical_record,
        )

    message = str(raised.value)
    diagnostic = json.loads(message.split("diagnostic=", 1)[1])
    assert len(message) <= runner._MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS
    assert diagnostic["observed_equality_partitions"] == 7
    assert len(diagnostic["roots"]) == 9
    assert "private" not in message


def test_run_cache_protocol_rejects_cold_hit_root_byte_change_before_semantic_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    phase_counts = {"compile": 0, "cold": 0, "hit": 0}

    def worker_command(*args, cache_kind: str, **kwargs) -> tuple[str, str]:
        return "worker", cache_kind

    def worker_result(command, **kwargs):
        phase = command[1]
        index = phase_counts[phase]
        phase_counts[phase] += 1
        record = _materialized_cache_record(kwargs["environment"], b"canonical", phase=phase)
        if phase == "hit" and index == 0:
            record["persistent_cache_root_identity"] = "d" * 64
        return record

    monkeypatch.setattr(runner, "_worker_base_command", worker_command)
    monkeypatch.setattr(runner, "run_timed_compile_worker_command", worker_result)
    canonical_root, canonical_record = _canonical_cache_record(tmp_path)

    with pytest.raises(ValueError, match="root changed between cold and hit"):
        runner._run_cache_protocol(
            config,
            "contract_map_9836cdbed389db24",
            "ordinary_xla",
            tmp_path / "generated.json",
            tmp_path / "cache_protocol",
            canonical_root,
            canonical_record,
        )


def test_cache_protocol_diagnostic_fails_closed_at_serialized_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_records = tuple(_cache_root_record("a" * 64, phase="compile") for _ in range(3))
    cold_records = tuple(_cache_root_record("a" * 64, phase="cold") for _ in range(3))
    hit_records = tuple(_cache_root_record("a" * 64, phase="hit") for _ in range(3))
    mismatched = (*cold_records[:2], _cache_root_record("d" * 64, phase="cold"))
    monkeypatch.setattr(runner, "_MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS", 1)

    with pytest.raises(AssertionError, match="exceeds its reviewed bound"):
        runner.validated_cache_protocol_identity(
            compile_records,
            mismatched,
            hit_records,
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("persistent_cache_compression", "zstandard", "invalid compression contract"),
        ("persistent_cache_key", "jit_other-" + "a" * 64, "invalid target cache key"),
        ("persistent_cache_serialized_executable_sha256", "not-a-digest", "invalid serialized executable"),
        ("persistent_cache_identity", "not-a-digest", "invalid content identity"),
        ("persistent_cache_identity", "d" * 64, "inconsistent content identity"),
        ("persistent_cache_entry_sha256", "not-a-digest", "invalid compressed entry identity"),
        ("persistent_cache_root_identity", "not-a-digest", "invalid root identity"),
        ("persistent_cache_compile_time", True, "invalid cached compile time"),
        ("persistent_cache_compile_time", -1, "invalid cached compile time"),
        ("persistent_cache_events", {}, "invalid public cache events"),
        ("persistent_cache_file_count", True, "invalid file count"),
        ("persistent_cache_file_count", 0, "invalid file count"),
        ("persistent_cache_total_bytes", True, "invalid byte total"),
        ("persistent_cache_total_bytes", 0, "invalid byte total"),
        ("final_hlo", "", "invalid final HLO"),
    ),
)
def test_cache_protocol_rejects_malformed_root_diagnostic_fields(field: str, value: object, message: str) -> None:
    record = _cache_root_record("a" * 64)
    record[field] = value
    records = (record, _cache_root_record("a" * 64), _cache_root_record("a" * 64))

    with pytest.raises(ValueError, match=message):
        runner.validated_cache_protocol_identity(
            records,
            tuple(_cache_root_record("a" * 64) for _ in range(3)),
            tuple(_cache_root_record("a" * 64) for _ in range(3)),
            case_id="contract_map_case",
            backend="ordinary_xla",
            required_processes=3,
        )


def test_executable_hlo_rejects_mismatch_between_timing_cache_and_profile_workers() -> None:
    records = tuple({"final_hlo": "same"} for _ in range(3))
    protocol = {
        "compile": (records[0], {"final_hlo": "fresh nondeterministic HLO"}, {"final_hlo": "other fresh HLO"}),
        "cold": records,
        "hit": records,
    }

    assert (
        runner.validated_executable_hlo(
            "ordinary_xla",
            case_worker_hlo="same",
            cache_protocol=protocol,
            profile_worker_hlo="same",
        )
        == "same"
    )
    with pytest.raises(ValueError, match="differs across"):
        runner.validated_executable_hlo(
            "ordinary_xla",
            case_worker_hlo="borrowed",
            cache_protocol=protocol,
            profile_worker_hlo="same",
        )


def test_measurement_workers_require_canonical_key_executable_root_and_public_hit() -> None:
    backends = tuple(backend.value for backend in BackendVariant)
    protocols = {
        backend: {
            "canonical": _cache_root_record(f"{index + 1:x}" * 64, phase="compile"),
            "compile": (_cache_root_record(f"{index + 4:x}" * 64, phase="compile"),),
        }
        for index, backend in enumerate(backends)
    }
    root = "f" * 64
    events = dict(zip(runner._CACHE_EVENT_NAMES, (1, 1, 0), strict=True))

    def evidence(backend: str) -> dict[str, object]:
        canonical = protocols[backend]["canonical"]
        return {
            "persistent_cache_entry_sha256": canonical["persistent_cache_entry_sha256"],
            "persistent_cache_events": events,
            "persistent_cache_key": canonical["persistent_cache_key"],
            "persistent_cache_root_identity": root,
            "persistent_cache_serialized_executable_sha256": canonical["persistent_cache_serialized_executable_sha256"],
        }

    case = {"persistent_cache": {backend: evidence(backend) for backend in sorted(backends)}}
    profiles = {
        backend: runner.NcuProfileEvidence(
            metrics=(),
            report_path="report",
            report_sha256="a" * 64,
            sass_source_path="sass",
            sass_source_sha256="b" * 64,
            final_hlo="hlo",
            persistent_cache=evidence(backend),
        )
        for backend in backends
    }
    runner.validate_measurement_cache_consumers(protocols, case, profiles, canonical_root_identity=root)

    case["persistent_cache"][backends[1]]["persistent_cache_events"] = dict(
        zip(runner._CACHE_EVENT_NAMES, (1, 0, 1), strict=True)
    )
    with pytest.raises(ValueError, match="did not execute the canonical cached executable"):
        runner.validate_measurement_cache_consumers(protocols, case, profiles, canonical_root_identity=root)


def test_ordinary_xla_boundary_comes_from_numbered_final_hlo_entry_abi() -> None:
    evidence = runner.derive_ordinary_xla_executable_evidence(
        _ordinary_hlo_fixture(),
        rows=3,
        reduction=5,
        features=7,
        profiled_launches=("ordinary_kernel",),
    )

    assert evidence.logical_training_step_boundary["input_layouts"] == [
        "bf16[3,5]{1,0}",
        "bf16[5,7]{1,0}",
        "bf16[7,5]{1,0}",
        "bf16[3,5]{1,0}",
    ]
    assert evidence.logical_training_step_boundary["saved_state_names_and_bytes"] == {}
    assert evidence.logical_training_step_boundary["recompute_operations"] == []
    assert [value["role"] for value in evidence.manifest["logical_inputs"]] == ["x", "w0", "w1", "do"]
    assert [value["role"] for value in evidence.manifest["logical_outputs"]] == ["y", "dx", "dw0", "dw1"]
    assert [fusion["name"] for fusion in evidence.manifest["fusions"]] == ["y.fused"]
    assert evidence.manifest["entry_copies"] == []
    assert evidence.manifest["entry_transposes"] == []
    assert evidence.manifest["entry_bitcasts"] == []


def test_ordinary_xla_boundary_rejects_generated_or_mutated_layout_claim() -> None:
    mutated = _ordinary_hlo_fixture().replace("%x = bf16[3,5]{1,0}", "%x = bf16[3,5]{0,1}")

    with pytest.raises(ValueError, match="parameter layouts changed"):
        runner.derive_ordinary_xla_executable_evidence(
            mutated,
            rows=3,
            reduction=5,
            features=7,
            profiled_launches=("ordinary_kernel",),
        )


def test_ordinary_xla_boundary_ignores_dead_wrappers_and_rejects_reachable_unproven_adapters() -> None:
    dead_wrapper = _ordinary_hlo_fixture().replace(
        "  %y.fused =",
        "  %dead = bf16[3,5]{1,0} copy(%x)\n  %y.fused =",
    )

    evidence = runner.derive_ordinary_xla_executable_evidence(
        dead_wrapper,
        rows=3,
        reduction=5,
        features=7,
        profiled_launches=("ordinary_kernel",),
    )
    assert evidence.manifest["entry_copies"] == []

    reachable_wrapper = _ordinary_hlo_fixture().replace(
        "%y.fused = bf16[3,5]{1,0} fusion(%x), kind=kLoop, calls=%identity",
        "%y.fused = bf16[3,5]{1,0} copy(%x)",
    )
    with pytest.raises(ValueError, match="require materialization evidence"):
        runner.derive_ordinary_xla_executable_evidence(
            reachable_wrapper,
            rows=3,
            reduction=5,
            features=7,
            profiled_launches=("ordinary_kernel",),
        )


def test_generated_kernel_identity_is_exact_and_sass_binds_loaded_shared_object(tmp_path: Path) -> None:
    names = ("generated_first", "generated_second")
    shared_library = tmp_path / "loaded.so"
    shared_library.write_bytes(b"authoritative-loaded-image")
    loaded_sass = tmp_path / "loaded.sass"
    loaded_sass.write_text(
        "Function : generated_second\n"
        '.headerflags @"EF_CUDA_SM90A"\n'
        "/*0000*/ @!P0 MOV R1, R2 ; /* 0x0000df00ff017b82 */\n"
        "/* 0x000fe20000000800 */\n"
        "Function : generated_first\n"
        "/*0000*/ EXIT ;\n"
    )
    cubin_sass = tmp_path / "cubin.sass"
    cubin_sass.write_text("Function : unrelated_cubin_body\n")
    artifact = runner.GeneratedArtifact(
        case_id="case",
        backend="shuttle_fast",
        physical_digest="a" * 64,
        source_path="source.cu",
        source_sha256="1" * 64,
        shared_library_path=str(shared_library),
        shared_library_sha256=hashlib.sha256(shared_library.read_bytes()).hexdigest(),
        ptx_path="generated.ptx",
        ptx_sha256="3" * 64,
        cubin_path="separate.cubin",
        cubin_sha256="4" * 64,
        cubin_sass_path=str(cubin_sass),
        cubin_sass_sha256="5" * 64,
        loaded_image_sass_path=str(loaded_sass),
        loaded_image_sass_sha256=hashlib.sha256(loaded_sass.read_bytes()).hexdigest(),
        compiler_flags=("nvcc",),
        ptxas_resources=tuple(
            {
                "kernel_name": name,
                "registers_per_thread": 32,
                "spill_load_bytes": 0,
                "spill_store_bytes": 0,
                "stack_frame_bytes": 0,
                "static_shared_bytes": 0,
            }
            for name in names
        ),
    )
    candidate = SimpleNamespace(generated=SimpleNamespace(kernel_names=names))
    metrics = tuple(_ncu_metric(name) for name in names)

    records = runner.generated_kernel_records(candidate, artifact, metrics)

    assert [record["name"] for record in records] == list(names)
    assert all(record["sass_path"] == str(loaded_sass) for record in records)
    assert all(record["sass_sha256"] == hashlib.sha256(loaded_sass.read_bytes()).hexdigest() for record in records)
    assert all(record["cubin"]["path"] == "separate.cubin" for record in records)
    lookalike = (_ncu_metric("generated_first_suffix"), metrics[1])
    with pytest.raises(ValueError, match="exactly once"):
        runner.generated_kernel_records(candidate, artifact, lookalike)


def test_generated_compile_disassembles_authoritative_loaded_shared_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runner_config(tmp_path)
    kernel_name = "generated_kernel"
    generated = SimpleNamespace(
        source='extern "C" __global__ void generated_kernel() {}',
        kernel_names=(kernel_name,),
        physical_digest="a" * 64,
    )
    candidate = SimpleNamespace(
        case=SimpleNamespace(case_id="case"),
        backend=SimpleNamespace(value="shuttle_fast"),
        generated=generated,
    )
    training = SimpleNamespace(generated_contract_map_candidates=lambda: (candidate,))
    original_import_module = runner.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "lib.tile_lifetime.benchmarks.h100_contract_map_backend_training":
            return training
        return original_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", import_module)

    def compile_plan(
        generated: Any, *, artifact_directory: Path, nvcc: Path, include_directory: Path
    ) -> ContractMapCompilePlan:
        del generated, nvcc, include_directory
        source = artifact_directory / "candidate.cu"
        shared = artifact_directory / "candidate.so"
        ptx = artifact_directory / "candidate.ptx"
        cubin = artifact_directory / "candidate.cubin"
        cubin_sass = artifact_directory / "candidate.sass"
        return ContractMapCompilePlan(
            source_path=source,
            shared_library_path=shared,
            ptx_path=ptx,
            cubin_path=cubin,
            sass_path=cubin_sass,
            shared_library_command=("nvcc", str(source), "-o", str(shared)),
            ptx_command=("nvcc", str(source), "-o", str(ptx)),
            cubin_command=("nvcc", str(source), "-o", str(cubin)),
            sass_command=(str(config.tools.cuobjdump), "--dump-sass", str(cubin)),
        )

    monkeypatch.setattr(resources, "contract_map_compile_plan", compile_plan)
    monkeypatch.setattr(
        resources,
        "parse_ptxas_kernel_resources",
        lambda output, *, expected_kernel_names: (
            PtxasKernelResources(
                kernel_name=expected_kernel_names[0],
                registers_per_thread=32,
                spill_load_bytes=0,
                spill_store_bytes=0,
                stack_frame_bytes=0,
                static_shared_bytes=0,
            ),
        ),
    )
    monkeypatch.setattr(cuda_toolchain, "cuda_toolkit_link_flags", lambda nvcc, *, runtime_search_path: ())
    monkeypatch.setattr(cuda_toolchain, "cuda_toolkit_shared_library_link_flags", lambda nvcc, names: ())

    disassembled_paths: list[Path] = []

    def run_retained(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if len(command) == 3 and command[1] == "--dump-sass":
            target = Path(command[2])
            disassembled_paths.append(target)
            name = kernel_name if target.suffix == ".so" else "separate_cubin_kernel"
            return subprocess.CompletedProcess(command, 0, f"Function : {name}\n/*0000*/ EXIT ;\n", "")
        output = Path(command[-1])
        output.write_bytes(f"compiled:{output.suffix}".encode())
        return subprocess.CompletedProcess(command, 0, "ptxas output", "")

    monkeypatch.setattr(runner, "_run_retained", run_retained)

    (artifact,) = runner.compile_generated_candidates(config)

    assert disassembled_paths == [Path(artifact.cubin_path), Path(artifact.shared_library_path)]
    assert Path(artifact.loaded_image_sass_path).read_text() == f"Function : {kernel_name}\n/*0000*/ EXIT ;\n"


@pytest.mark.parametrize(
    ("sass", "message"),
    (
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n",
            "coverage differs",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n" "Function : generated_second_suffix\n/*0000*/ EXIT ;\n",
            "coverage differs",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n"
            "Function : generated_extra\n/*0000*/ EXIT ;\n",
            "coverage differs",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n"
            "Function : generated_first\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "repeats a function identity",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n" "Function - generated_second\n/*0000*/ EXIT ;\n",
            "malformed function identity",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n" "Function : generated_second\n/*0010*/ this is not SASS\n",
            "malformed address-bearing instruction",
        ),
        (
            'Function : generated_first\n/*0000*/ EXIT ;\nFunction : generated_second\n.headerflags @"SM90A"\n',
            "contains no valid instructions",
        ),
        ("", "is empty"),
        (
            "warning: disassembly truncated\n"
            "Function : generated_first\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "warning, error, or unavailable",
        ),
        (
            "Function : generated_first\n/*0000*/ EXIT ;\n" "Function : generated_second\n/*0000*/ EXIT ;\n\0",
            "contains NUL",
        ),
        (
            "/*0000*/ EXIT ;\n"
            "Function : generated_first\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "outside a function section",
        ),
        (
            "Function : generated_first\n/* 0x000fe20000000800 */\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "standalone instruction encoding",
        ),
        (
            "Function : generated_first\n/*0000*/ MOV R1, R2 ;\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "repeated or reordered addresses",
        ),
        (
            "Function : generated_first\n/*0010*/ MOV R1, R2 ;\n/*0000*/ EXIT ;\n"
            "Function : generated_second\n/*0000*/ EXIT ;\n",
            "repeated or reordered addresses",
        ),
    ),
    ids=(
        "missing",
        "lookalike",
        "extra",
        "duplicate",
        "bad-anchor",
        "bad-instruction",
        "empty-section",
        "empty-output",
        "diagnostic",
        "nul",
        "instruction-before-function",
        "standalone-encoding",
        "duplicate-address",
        "reordered-address",
    ),
)
def test_loaded_shared_object_sass_topology_rejects_inexact_or_malformed_sections(sass: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        runner.validate_cuda_sass_kernel_topology(sass, ("generated_first", "generated_second"))


def test_generated_kernel_records_reject_changed_loaded_image_or_sass(tmp_path: Path) -> None:
    shared_library = tmp_path / "loaded.so"
    shared_library.write_bytes(b"loaded-image")
    loaded_sass = tmp_path / "loaded.sass"
    loaded_sass.write_text("Function : generated\n/*0000*/ EXIT ;\n")
    artifact = runner.GeneratedArtifact(
        case_id="case",
        backend="shuttle_fast",
        physical_digest="a" * 64,
        source_path="source.cu",
        source_sha256="1" * 64,
        shared_library_path=str(shared_library),
        shared_library_sha256=hashlib.sha256(shared_library.read_bytes()).hexdigest(),
        ptx_path="generated.ptx",
        ptx_sha256="3" * 64,
        cubin_path="separate.cubin",
        cubin_sha256="4" * 64,
        cubin_sass_path="separate.sass",
        cubin_sass_sha256="5" * 64,
        loaded_image_sass_path=str(loaded_sass),
        loaded_image_sass_sha256=hashlib.sha256(loaded_sass.read_bytes()).hexdigest(),
        compiler_flags=("nvcc",),
        ptxas_resources=(
            {
                "kernel_name": "generated",
                "registers_per_thread": 32,
                "spill_load_bytes": 0,
                "spill_store_bytes": 0,
                "stack_frame_bytes": 0,
                "static_shared_bytes": 0,
            },
        ),
    )
    candidate = SimpleNamespace(generated=SimpleNamespace(kernel_names=("generated",)))

    shared_library.write_bytes(b"changed-loaded-image")
    with pytest.raises(ValueError, match="shared-library content changed"):
        runner.generated_kernel_records(candidate, artifact, (_ncu_metric("generated"),))

    shared_library.write_bytes(b"loaded-image")
    loaded_sass.write_text("Function : generated\n/*0000*/ NOP ;\n")
    with pytest.raises(ValueError, match="SASS content changed"):
        runner.generated_kernel_records(candidate, artifact, (_ncu_metric("generated"),))


def test_runner_never_publishes_an_invalid_or_conforming_bundle(tmp_path: Path) -> None:
    output = tmp_path / "accepted_bundle.json"
    with pytest.raises(ValueError, match="architecture-nonconforming"):
        runner.publish_validated_bundle(output, {"architecture_status": "conforming", "records": ()})
    assert not output.exists()
    with pytest.raises(ValueError, match="all 24"):
        runner.publish_validated_bundle(
            output,
            {"architecture_status": ArchitectureStatus.NONCONFORMING.value, "records": ()},
        )
    assert not output.exists()


def _ordinary_hlo_fixture() -> str:
    return """HloModule fixture

%identity (p: bf16[3,5]) -> bf16[3,5] {
  %p = bf16[3,5]{1,0} parameter(0)
  ROOT %out = bf16[3,5]{1,0} bitcast(%p)
}

ENTRY %main (x: bf16[3,5], w0: bf16[5,7], w1: bf16[7,5], do: bf16[3,5]) -> (bf16[3,5], bf16[3,5], bf16[5,7], bf16[7,5]) {
  %do = bf16[3,5]{1,0} parameter(3)
  %w1 = bf16[7,5]{1,0} parameter(2)
  %x = bf16[3,5]{1,0} parameter(0)
  %w0 = bf16[5,7]{1,0} parameter(1)
  %y.fused = bf16[3,5]{1,0} fusion(%x), kind=kLoop, calls=%identity
  ROOT %result = (bf16[3,5]{1,0}, bf16[3,5]{1,0}, bf16[5,7]{1,0}, bf16[7,5]{1,0}) tuple(%y.fused, %do, %w0, %w1)
}
"""


def _ncu_metric(name: str) -> Any:
    return runner.NcuKernelMetrics(
        name=name,
        block_size=(256, 1, 1),
        registers_per_thread=32,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        active_blocks_per_sm=2,
        limiting_occupancy_resource="registers",
        achieved_occupancy=0.5,
    )


def _capsule_runner_config(tmp_path: Path) -> Any:
    config = _runner_config(tmp_path)
    local = config.source_root / "lib/tile_lifetime/local.py"
    local.parent.mkdir(parents=True)
    local.write_text("VALUE = 1\n")
    local.chmod(0o644)
    contents = local.read_bytes()
    manifest = {
        "archive": {"filename": "h100-evidence-source-capsule.zip", "sha256": "3" * 64},
        "members": [
            {
                "mode": "100644",
                "path": "lib/tile_lifetime/local.py",
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size": len(contents),
                "type": "file",
            }
        ],
        "schema_version": 1,
        "source": {"commit": config.source_sha, "tree": "2" * 40},
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_bytes(raw)
    return replace(
        config,
        source_tree="2" * 40,
        source_capsule_manifest=manifest_path,
        source_capsule_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _runner_config(tmp_path: Path) -> Any:
    source_root = tmp_path / "source"
    source_root.mkdir()
    toolkit = tmp_path / "cuda" / "bin"
    toolkit.mkdir(parents=True)
    tools = tmp_path / "tools"
    tools.mkdir()

    def executable(path: Path) -> Path:
        path.write_text(path.name)
        path.chmod(0o755)
        return path.resolve()

    return runner.RunnerConfig(
        source_root=source_root,
        source_sha="1" * 40,
        artifact_directory=tmp_path / "artifacts",
        tools=runner.ToolPaths(
            git=executable(tools / "git"),
            nvidia_smi=executable(tools / "nvidia-smi"),
            nvcc=executable(toolkit / "nvcc"),
            ptxas=executable(toolkit / "ptxas"),
            cuobjdump=executable(toolkit / "cuobjdump"),
            ncu=executable(tools / "ncu"),
            nsys=executable(tools / "nsys"),
        ),
        require_jax_version="1.2.3",
    )


def _worker_cache_fixture(tmp_path: Path, backends: tuple[str, ...]) -> tuple[Path, Path]:
    cache = tmp_path / "sealed-cache"
    cache.mkdir()
    records = {}
    for index, backend in enumerate(backends):
        key = f"jit_step-{index + 1:x}" + "a" * 63
        executable = f"serialized-{backend}".encode()
        entry = _compressed_cache_entry(executable)
        (cache / f"{key}-cache").write_bytes(entry)
        records[backend] = {
            "persistent_cache_entry_sha256": hashlib.sha256(entry).hexdigest(),
            "persistent_cache_key": key,
            "persistent_cache_serialized_executable_sha256": hashlib.sha256(executable).hexdigest(),
            "final_hlo": f"hlo-{backend}",
        }
    contract = runner._write_worker_cache_contract(tmp_path / "cache-contract.json", cache, records)
    return cache, contract


_NCU_RAW_FIXTURE = Path(__file__).parent / "fixtures/h100_contract_map_ncu_raw.csv"


def _read_ncu_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_ncu_rows(path: Path, fieldnames: list[str], rows: Any) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_ncu_csv(path: Path) -> None:
    fieldnames, rows = _read_ncu_csv(_NCU_RAW_FIXTURE)
    _write_ncu_rows(path, fieldnames, rows)
