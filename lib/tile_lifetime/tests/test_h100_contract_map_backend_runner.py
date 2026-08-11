# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import tile_lifetime.contract_map_backend_resources as resources
import tile_lifetime.cuda_toolchain as cuda_toolchain
from tile_lifetime.contract_map_backend_resources import ContractMapCompilePlan, PtxasKernelResources
from tile_lifetime.h100_contract_map_benchmark import (
    ArchitectureStatus,
    BackendVariant,
    default_h100_contract_map_benchmark_plan,
)

runner = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner")


def _ncu_sass_export(*sections: tuple[str, tuple[str, ...]]) -> str:
    lines = []
    for name, instructions in sections:
        lines.extend((f"Kernel Name: {name}", "Address Source", "----------------"))
        lines.extend(instructions)
    return "\n".join(lines) + "\n"


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
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line for line in lines if "launch__registers_per_thread" not in line) + "\n")
    with pytest.raises(ValueError, match="omits metrics"):
        runner.parse_ncu_metrics(output)


def test_runner_ncu_sass_parser_requires_valid_instruction_rows_for_exact_kernel_sections() -> None:
    records = runner.parse_ncu_sass(
        _ncu_sass_export(
            ("KernelA", ("0000000000000000 MOV R1, R2", "0000000000000010 EXIT")),
            ("KernelB", ("/*0020*/ FFMA R3, R4, R5, R6",)),
        ),
        ("KernelA", "KernelB"),
    )

    assert tuple(record.name for record in records) == ("KernelA", "KernelB")
    assert tuple(instruction.mnemonic for instruction in records[0].instructions) == ("MOV", "EXIT")
    assert records[1].instructions == (runner.NcuSassInstruction(address=0x20, mnemonic="FFMA"),)


def test_runner_ncu_sass_spills_are_read_only_from_validated_instruction_rows(tmp_path: Path) -> None:
    sass_path = tmp_path / "ordinary.sass"
    sass_path.write_text(
        _ncu_sass_export(
            ("KernelA", ("0000000000000000 LDL.64 R2, [R4]", "0000000000000010 STL [R4], R2")),
        )
    )
    profile = runner.NcuProfileEvidence(
        metrics=(_ncu_metric("KernelA"),),
        report_path="profile.ncu-rep",
        report_sha256="1" * 64,
        sass_source_path=str(sass_path),
        sass_source_sha256="2" * 64,
        final_hlo="hlo",
    )

    parsed = runner.parse_ncu_sass(sass_path.read_text(), ("KernelA",))
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
            _ncu_sass_export(("KernelA", ("0000000000000000 MOV R1, R2",))),
            ("KernelA", "KernelB"),
            "coverage differs",
        ),
        (
            _ncu_sass_export(
                ("KernelA", ("0000000000000000 MOV R1, R2",)),
                ("KernelA", ("0000000000000010 EXIT",)),
            ),
            ("KernelA",),
            "repeats a kernel section",
        ),
        (
            _ncu_sass_export(("KernelA_suffix", ("0000000000000000 MOV R1, R2",))),
            ("KernelA",),
            "coverage differs",
        ),
        (
            "Kernel Name: KernelA\nAddress Source\nnot-an-address MOV R1, R2\n",
            ("KernelA",),
            "unrecognized.*line",
        ),
        (
            "Kernel Name: KernelA\nAddress Source\n0000000000000000 NOTREAL R1, R2\n",
            ("KernelA",),
            "unrecognized SASS instruction mnemonic",
        ),
        (
            "==WARNING== Profiler output was truncated\n",
            ("KernelA",),
            "warning, error, or unavailable source",
        ),
        (
            "==ERROR== Profiler report cannot be imported\n",
            ("KernelA",),
            "warning, error, or unavailable source",
        ),
        (
            "SASS is unavailable for KernelA\n",
            ("KernelA",),
            "warning, error, or unavailable source",
        ),
        (
            "ordinary plain text with no SASS structure\n",
            ("KernelA",),
            "unrecognized.*line",
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


def test_runner_nsys_parser_attributes_kernel_and_copy_activity_to_exact_ranges(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT)")
        database.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        database.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER)"
        )
        database.execute("INSERT INTO NVTX_EVENTS VALUES (100, 400, 'contract_map.steady.0.ordinary_xla')")
        database.execute("INSERT INTO StringIds VALUES (7, 'fusion_kernel')")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (150, 250, 7)")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (260, 280, 64, 8)")

    records = runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))

    assert records == (
        runner.TraceRange(
            name="contract_map.steady.0.ordinary_xla",
            ordered_kernel_names=("fusion_kernel",),
            kernel_duration_ns=100,
            device_to_device_count=1,
            device_to_device_bytes=64,
            host_to_device_count=0,
            host_to_device_bytes=0,
            unexpected_copy_count=0,
        ),
    )


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
    copied = replace(traces[0], device_to_device_count=1, device_to_device_bytes=64)
    with pytest.raises(ValueError, match="unexpected copies"):
        runner.merge_trace_timing(plan, {"raw_samples": worker_rows}, (copied, *traces[1:]))


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
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "99")

    uncached = runner._worker_environment(tmp_path / "uncached_dump")
    isolated = runner._worker_environment(tmp_path / "cached_dump", tmp_path / "isolated_cache")

    assert "JAX_COMPILATION_CACHE_DIR" not in uncached
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in uncached
    assert isolated["JAX_COMPILATION_CACHE_DIR"] == str((tmp_path / "isolated_cache").resolve())
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


def test_cache_protocol_rejects_any_compile_cold_or_hit_root_identity_mismatch() -> None:
    identity = "a" * 64
    matching = tuple({"persistent_cache_identity": identity} for _ in range(3))

    assert runner.validated_cache_protocol_identity(matching, matching, matching, required_processes=3) == identity
    mismatched = (*matching[:2], {"persistent_cache_identity": "b" * 64})
    with pytest.raises(ValueError, match="all compile, cold, and hit roots"):
        runner.validated_cache_protocol_identity(mismatched, matching, matching, required_processes=3)


def test_executable_hlo_rejects_mismatch_between_timing_cache_and_profile_workers() -> None:
    records = tuple({"final_hlo": "same"} for _ in range(3))
    protocol = {"compile": records, "cold": records, "hit": records}

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


def _write_ncu_csv(path: Path) -> None:
    values = {
        "launch__block_size": "256",
        "launch__registers_per_thread": "48",
        "launch__shared_mem_per_block_static": "128",
        "launch__shared_mem_per_block_dynamic": "0",
        "launch__occupancy_limit_blocks": "4",
        "launch__occupancy_limit_registers": "2",
        "launch__occupancy_limit_shared_mem": "8",
        "launch__occupancy_limit_warps": "2",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "62.5",
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "Kernel Name", "Metric Name", "Metric Value"))
        writer.writeheader()
        for metric, value in values.items():
            writer.writerow({"ID": "1", "Kernel Name": "KernelA", "Metric Name": metric, "Metric Value": value})
