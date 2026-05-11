#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract compact SonicMoE/Pallas IR and PTX snippets from local dump dirs."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

GLOBAL_MEM_RE = re.compile(r"\b(?:ld|st)\.global")


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def selected_upstream_prefix(upstream_root: Path) -> str:
    selected = None
    for path in sorted(upstream_root.glob("*.asm.ttir")):
        contents = read_text(path)
        if "tensor<1x2048x" in contents or "tensor<2048xf32" in contents:
            selected = path
            break

    if selected is None:
        matches = sorted(upstream_root.glob("*.asm.ttir"))
        if not matches:
            raise FileNotFoundError(f"no upstream TTIR files under {upstream_root}")
        selected = matches[0]

    return str(selected)[: -len(".asm.ttir")]


def interesting_ttir(path: Path) -> str:
    lines = []
    for line_number, line in enumerate(read_text(path).splitlines(), start=1):
        if any(token in line for token in ("scf.for", "tt.load", "tt.reduce", "tt.store", "tt.reduce.return")):
            lines.append(f"{line_number}: {line.rstrip()}")
    return "\n".join(lines)


def ptx_entry_and_global_memory(path: Path, *, max_mem: int | None = None) -> str:
    raw_lines = read_text(path).splitlines()
    entry_lines: list[str] = []
    in_entry = False
    paren_balance = 0

    for line_number, line in enumerate(raw_lines, start=1):
        if ".visible .entry" in line:
            in_entry = True
        if not in_entry:
            continue

        entry_lines.append(f"{line_number}: {line.rstrip()}")
        paren_balance += line.count("(") - line.count(")")
        if paren_balance <= 0 and line.strip().endswith(")"):
            break
        if len(entry_lines) >= 40:
            break

    mem_lines = [
        f"{line_number}: {line.rstrip()}"
        for line_number, line in enumerate(raw_lines, start=1)
        if GLOBAL_MEM_RE.search(line)
    ]

    if max_mem is not None:
        omitted = max(0, len(mem_lines) - max_mem)
        mem_lines = mem_lines[:max_mem]
        if omitted:
            mem_lines.append(f"... {omitted} global memory instructions omitted ...")

    return "\n".join(["# PTX entry", *entry_lines, "", "# Global memory instructions", *mem_lines])


def selected_xla_ptx(root: Path) -> Path:
    candidates = [path for path in root.rglob("*.ptx") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no PTX files under {root}")

    return max(
        candidates,
        key=lambda path: len(GLOBAL_MEM_RE.findall(read_text(path))) + (1000 if "jit__lambda" in path.name else 0),
    )


def stablehlo_custom_call_summary(path: Path) -> str:
    contents = read_text(path)
    custom_call = ""
    for line in contents.splitlines():
        if "stablehlo.custom_call" in line or "custom-call" in line:
            custom_call = line.strip()
            break

    ir_start = custom_call.find('ir = "')
    if ir_start >= 0:
        ir_end = custom_call.find('",', ir_start + len('ir = "'))
        if ir_end >= 0:
            custom_call = (
                custom_call[:ir_start] + 'ir = "<embedded Triton bytecode omitted>"' + custom_call[ir_end + 1 :]
            )

    return f"{custom_call[:1800]}\n\ncontains_xla_gpu_triton={'xla.gpu.triton' in contents}"


def emit_log_record(name: str, source_path: Path, text: str) -> None:
    print(
        json.dumps(
            {
                "event": "ptx_side_by_side_snippet",
                "name": name,
                "source_path": str(source_path),
                "text_b64": base64.b64encode(text.encode()).decode(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def write_snippet(output_dir: Path, name: str, source_path: Path, text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix if source_path.suffix else ".txt"
    path = output_dir / f"{name}{suffix}"
    path.write_text(text)
    print(json.dumps({"event": "snippet_file", "name": name, "path": str(path), "source_path": str(source_path)}))


def collect_snippets(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    upstream_prefix = selected_upstream_prefix(args.ir_dir / "upstream_weighted")
    sonic_ttir = Path(upstream_prefix + ".asm.ttir")
    sonic_ptx = Path(upstream_prefix + ".asm.ptx")

    pallas_token_ptx = selected_xla_ptx(args.xla_token_loop_dir)
    pallas_tiled_ptx = selected_xla_ptx(args.xla_tiled_dir)
    pallas_stablehlo = args.ir_dir / "pallas_token_loop_weighted" / "pallas_triton_token_loop_weighted.stablehlo.mlir"

    return [
        ("sonic_selected_ttir_ops", sonic_ttir, interesting_ttir(sonic_ttir)),
        ("sonic_selected_ptx_mem", sonic_ptx, ptx_entry_and_global_memory(sonic_ptx)),
        ("pallas_token_loop_ptx_mem", pallas_token_ptx, ptx_entry_and_global_memory(pallas_token_ptx, max_mem=96)),
        ("pallas_tiled_ptx_mem", pallas_tiled_ptx, ptx_entry_and_global_memory(pallas_tiled_ptx, max_mem=104)),
        ("pallas_token_loop_stablehlo_custom_call", pallas_stablehlo, stablehlo_custom_call_summary(pallas_stablehlo)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", type=Path, default=Path("/tmp/sonicmoe_ir_compare"))
    parser.add_argument("--xla-tiled-dir", type=Path, default=Path("/tmp/xla_pallas_tiled"))
    parser.add_argument("--xla-token-loop-dir", type=Path, default=Path("/tmp/xla_pallas_token_loop"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--emit-log-records", action="store_true")
    args = parser.parse_args()

    for name, source_path, text in collect_snippets(args):
        if args.output_dir is not None:
            write_snippet(args.output_dir, name, source_path, text)
        if args.emit_log_records:
            emit_log_record(name, source_path, text)


if __name__ == "__main__":
    main()
