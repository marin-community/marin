#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IR_DIR="${IR_DIR:-/tmp/sonicmoe_ir_compare}"
rm -rf "$IR_DIR" /tmp/xla_pallas_tiled /tmp/xla_pallas_token_loop
mkdir -p "$IR_DIR"

uv run --package marin --extra gpu --group dev python .agents/scripts/sonicmoe_upstream_token_gather_bench.py \
  --install-deps --weighted --tokens 8192 --hidden 2048 --experts 8 --topk 2 --dtype bf16 \
  --kernel-repeat 16 --replicate-input --warmup 3 --steps 10 --write-ir-dir "$IR_DIR/upstream_weighted"

XLA_FLAGS="--xla_dump_to=/tmp/xla_pallas_tiled --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_gpu_dump_llvmir" \
  uv run --package marin --extra gpu --group dev python .agents/scripts/sonicmoe_jax_ffi_token_gather_bench.py \
  --skip-ffi --weighted --tokens 8192 --hidden 2048 --experts 8 --topk 2 \
  --kernel-repeat 16 --replicate-input --also-pallas --pallas-variant tiled \
  --pallas-token-block 8 --pallas-hidden-block 256 --pallas-warps 4 \
  --warmup 3 --steps 10 --write-ir-dir "$IR_DIR/pallas_tiled_weighted"

XLA_FLAGS="--xla_dump_to=/tmp/xla_pallas_token_loop --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_gpu_dump_llvmir" \
  uv run --package marin --extra gpu --group dev python .agents/scripts/sonicmoe_jax_ffi_token_gather_bench.py \
  --skip-ffi --weighted --tokens 8192 --hidden 2048 --experts 8 --topk 2 \
  --kernel-repeat 16 --replicate-input --also-pallas --pallas-variant token_loop \
  --pallas-token-block 1 --pallas-hidden-block 2048 --pallas-warps 4 \
  --warmup 3 --steps 10 --write-ir-dir "$IR_DIR/pallas_token_loop_weighted"

python - <<'PY'
from __future__ import annotations

import json
import re
from pathlib import Path


def text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception as exc:
        return f"<read failed: {exc}>"


def summarize(name: str, path: Path) -> None:
    contents = text(path)
    counts = {
        "bytes": path.stat().st_size,
        "lines": contents.count("\n") + 1,
        "tt_load": contents.count("tt.load") + contents.count("tl.load"),
        "tt_store": contents.count("tt.store") + contents.count("tl.store"),
        "tt_reduce": contents.count("tt.reduce"),
        "tt_dot": contents.count("tt.dot") + contents.count("ttng.dot"),
        "scf_for": contents.count("scf.for"),
        "llvm_load": contents.count("llvm.load"),
        "llvm_store": contents.count("llvm.store"),
        "ptx_ld_global": len(re.findall(r"\bld\.global", contents)),
        "ptx_st_global": len(re.findall(r"\bst\.global", contents)),
        "ptx_bra": len(re.findall(r"\bbra", contents)),
        "ptx_bar": len(re.findall(r"\bbar\.", contents)),
    }
    print(json.dumps({"event": "ir_summary", "name": name, "path": str(path), **counts}, sort_keys=True), flush=True)
    if path.suffix in {".hlo", ".txt", ".mlir"} and path.stat().st_size > 20_000:
        return

    interesting = []
    for line in contents.splitlines():
        if any(
            token in line
            for token in (
                "tt.load",
                "tt.store",
                "tt.reduce",
                "scf.for",
                "custom-call",
                "xla.gpu.triton",
                "triton_gpu",
                "ld.global",
                "st.global",
            )
        ):
            interesting.append(line.strip()[:500])
        if len(interesting) >= 20:
            break
    print(json.dumps({"event": "ir_excerpt", "name": name, "lines": interesting}, sort_keys=True), flush=True)


def hlo_custom_call_summary(name: str, path: Path) -> None:
    contents = text(path)
    custom_line = ""
    for line in contents.splitlines():
        if "custom-call" in line or "stablehlo.custom_call" in line:
            custom_line = line.strip()
            break
    print(
        json.dumps(
            {
                "event": "hlo_custom_call",
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "custom_line_prefix": custom_line[:1200],
                "contains_xla_gpu_triton": "xla.gpu.triton" in contents,
                "embedded_triton_mentions": contents.count("triton"),
                "embedded_tt_load_mentions": contents.count("tt.load"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


roots = [
    Path("/tmp/sonicmoe_ir_compare"),
    Path("/tmp/xla_pallas_tiled"),
    Path("/tmp/xla_pallas_token_loop"),
]
print(json.dumps({"event": "ir_file_index_start"}), flush=True)
for root in roots:
    if not root.exists():
        print(json.dumps({"root": str(root), "missing": True}), flush=True)
        continue
    files = sorted(path for path in root.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "root": str(root),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    suffix_counts = {}
    for path in files:
        suffix_counts[path.suffix or "<none>"] = suffix_counts.get(path.suffix or "<none>", 0) + 1
    print(json.dumps({"event": "ir_suffix_counts", "root": str(root), "suffix_counts": suffix_counts}, sort_keys=True), flush=True)

upstream_root = Path("/tmp/sonicmoe_ir_compare/upstream_weighted")
selected_ttir = None
for path in sorted(upstream_root.glob("*.asm.ttir")):
    contents = text(path)
    if "tensor<1x2048x" in contents or "tensor<2048xf32" in contents:
        selected_ttir = path
        break
if selected_ttir is None:
    matches = sorted(upstream_root.glob("*.asm.ttir"))
    selected_ttir = matches[0] if matches else None

if selected_ttir is not None:
    selected_prefix = str(selected_ttir)[: -len(".asm.ttir")]
    print(json.dumps({"event": "upstream_selected_ir_prefix", "prefix": selected_prefix}, sort_keys=True), flush=True)
    for key in ("source", "ttir", "ttgir", "llir", "ptx"):
        path = Path(f"{selected_prefix}.asm.{key}")
        if path.exists():
            summarize(f"upstream_selected_{key}", path)

for root, label in [
    (Path("/tmp/sonicmoe_ir_compare/pallas_tiled_weighted"), "pallas_tiled"),
    (Path("/tmp/sonicmoe_ir_compare/pallas_token_loop_weighted"), "pallas_token_loop"),
]:
    for path in sorted(root.glob("*.hlo.txt")):
        summarize(f"{label}_hlo", path)
        hlo_custom_call_summary(f"{label}_hlo", path)
    for path in sorted(root.glob("*.stablehlo.mlir")):
        summarize(f"{label}_stablehlo", path)
        hlo_custom_call_summary(f"{label}_stablehlo", path)

for root, label in [
    (Path("/tmp/xla_pallas_tiled"), "xla_tiled"),
    (Path("/tmp/xla_pallas_token_loop"), "xla_token_loop"),
]:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".ll", ".ptx", ".mlir"}
    ]
    scored = []
    for path in candidates:
        contents = text(path)
        score = sum(
            contents.count(token)
            for token in ("__gpu$xla.gpu.triton", "xla.gpu.triton", "triton", "tt.load", "ld.global")
        )
        if score:
            ptx_score = len(re.findall(r"\bld\.global", contents)) + len(re.findall(r"\bst\.global", contents))
            scored.append((ptx_score, score, path))
    for ptx_score, score, path in sorted(scored, reverse=True)[:6]:
        summarize(f"{label}_dump_score_{score}", path)
        print(
            json.dumps(
                {
                    "event": "xla_dump_candidate",
                    "name": label,
                    "path": str(path),
                    "score": score,
                    "ptx_score": ptx_score,
                    "bytes": path.stat().st_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
PY
