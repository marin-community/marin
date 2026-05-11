# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run real SonicMoE and the JAX/Pallas port with the same shape flags."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import add_common_arguments, emit_record


def _append_common_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--tokens",
        str(args.tokens),
        "--hidden",
        str(args.hidden),
        "--experts",
        str(args.experts),
        "--topk",
        str(args.topk),
        "--dtype",
        args.dtype,
        "--kernel-repeat",
        str(args.kernel_repeat),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
    ]
    if args.weighted:
        result.append("--weighted")
    if args.replicate_input:
        result.append("--replicate-input")
    return result


def _run(case: str, command: list[str]) -> None:
    emit_record({"event": "case_start", "case": case, "command": command})
    subprocess.run(command, check=True)
    emit_record({"event": "case_end", "case": case})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--skip-real-sonic", action="store_true")
    parser.add_argument("--skip-pallas", action="store_true")
    parser.add_argument("--pallas-backends", default="pallas_triton_faithful,pallas_triton_token_kblock,pallas_triton")
    parser.add_argument("--token-block", type=int, default=16)
    parser.add_argument("--hidden-block", type=int, default=64)
    parser.add_argument("--k-block", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--write-ir-dir", type=Path)
    add_common_arguments(parser)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    common_args = _append_common_args(args)

    if not args.skip_real_sonic:
        command = [sys.executable, str(script_dir / "real_sonic_token_gather_sum.py"), *common_args]
        if args.install_deps:
            command.append("--install-deps")
        if args.write_ir_dir is not None:
            command.extend(["--write-ir-dir", str(args.write_ir_dir)])
        _run("real_sonic_token_gather_sum", command)

    if not args.skip_pallas:
        for backend in [backend.strip() for backend in args.pallas_backends.split(",") if backend.strip()]:
            command = [
                sys.executable,
                str(script_dir / "pallas_token_gather_sum_port.py"),
                *common_args,
                "--backend",
                backend,
                "--token-block",
                str(args.token_block),
                "--hidden-block",
                str(args.hidden_block),
                "--k-block",
                str(args.k_block),
                "--num-warps",
                str(args.num_warps),
            ]
            if args.write_ir_dir is not None:
                command.extend(["--write-ir-dir", str(args.write_ir_dir)])
            _run(f"ported_pallas_{backend}", command)


if __name__ == "__main__":
    main()
