"""Run the released Open-MOPD rollout with one vLLM port range per worker."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Sequence
from functools import partial
from typing import Callable

VLLM_PORT_MIN = 20_000
VLLM_PORT_SPAN = 40_000
VLLM_RANK_PORT_STRIDE = 997


def evaluation_port_seed(output_uri: str) -> int:
    """Derive a stable per-evaluation port seed from its durable output identity."""
    digest = hashlib.sha256(output_uri.encode()).digest()
    return VLLM_PORT_MIN + int.from_bytes(digest[:8], "big") % VLLM_PORT_SPAN


def worker_port(port_seed: int, global_rank: int) -> int:
    """Return a distinct valid vLLM port starting point for one rollout worker."""
    return VLLM_PORT_MIN + (port_seed - VLLM_PORT_MIN + global_rank * VLLM_RANK_PORT_STRIDE) % VLLM_PORT_SPAN


def _worker_main(
    args: argparse.Namespace,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    tp_size: int,
    *,
    port_seed: int,
    released_worker: Callable[..., None],
) -> None:
    os.environ["VLLM_PORT"] = str(worker_port(port_seed, global_dp_rank))
    released_worker(args, dp_size, local_dp_rank, global_dp_rank, tp_size)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vllm-port-seed", type=int, required=True)
    port_args, released_argv = parser.parse_known_args(argv)
    return port_args, released_argv


def run(argv: Sequence[str] | None = None) -> int:
    """Delegate fan-out to the released runner after isolating each worker's port."""
    port_args, released_argv = _parse_args(argv)

    # The pinned authors' package only exists after the evaluation task stages its checkout.
    from evals.rollout_engine import vllm_rollout  # noqa: PLC0415

    released_worker = vllm_rollout.worker_main
    vllm_rollout.worker_main = partial(
        _worker_main,
        port_seed=port_args.vllm_port_seed,
        released_worker=released_worker,
    )
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *released_argv]
        vllm_rollout.main()
    finally:
        sys.argv = original_argv
        vllm_rollout.worker_main = released_worker
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
