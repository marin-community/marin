# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Joint-decode completion algorithm for downstream-scaling evals.

Decodes from two models that share a tokenizer. At each step:
  1. Get top-k tokens from model A (decoder) and model B (advisor).
  2. Among A's top-k, pick the token with highest rank in B's top-k.
  3. If A's top-k and B's top-k don't overlap, fall back to A's top-1.

Structurally mirrors iid.py. The two engines run as subprocesses on distinct
chips of the local TPU host (because TPU_VISIBLE_CHIPS is process-level). A
JointDecoder helper encapsulates subprocess management, the HTTP token-decision
server, and the selection rule, exposing .generate(prompts) shaped like
vllm.LLM.generate.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, MirroredValue
from marin.execution.remote import remote
from marin.execution.types import this_output_path, versioned
from zephyr import Dataset, ShardInfo, ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, localize_mirror_path, version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    # Required at `uv sync` time so vllm's setup.py skips CUDA-version
    # detection (which asserts CUDA_HOME). Propagated to the container build
    # via remote(env_vars=...).
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}

# Subprocess workers prefix every IPC line on stdout so coordinator can ignore
# vLLM/JAX/runtime log noise that may leak to stdout.
IPC_PREFIX = "__JD__:"
DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
DEFAULT_POLL_BACKOFF = 10.0


@dataclass(frozen=True)
class JointDecodeSamplingConfig:
    n_samples: int
    max_tokens: int
    top_k_a: int
    top_k_b: int
    seed: int
    # Retained for executor cache-key stability; no longer consumed.
    temperature: float = 1.0
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if self.n_samples != 1:
            raise ValueError("joint_decode is deterministic per prompt; n_samples must be 1 " f"(got {self.n_samples})")


@dataclass(frozen=True)
class JointDecodeModelConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = False
    # Halve the RPA-kernel KV-page block size. Required for delphi-shaped
    # models (otherwise vmem error); harms perf on standard models like llama,
    # so default off.
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class JointDecodeExecutionConfig:
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    # In-flight requests per generate() IPC round-trip. None → whole chunk in
    # one round-trip (no microbatching). Smaller values keep A's and B's
    # vLLM schedulers in lockstep at the cost of engine batch parallelism.
    microbatch_size: int | None = None
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF
    barrier_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.microbatch_size is not None and self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1 or None (got {self.microbatch_size})")


@dataclass(frozen=True)
class JointDecodeConfig:
    sampling: JointDecodeSamplingConfig
    advisor_model_path: str | InputName | MirroredValue
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    execution: JointDecodeExecutionConfig


@dataclass(frozen=True)
class JointDecodeCompletionStepConfig:
    output_path: str
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    sampling: JointDecodeSamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str
    chunk_size: int
    microbatch_size: int
    heartbeat_timeout: float
    poll_backoff: float
    barrier_timeout_s: float


@dataclass(frozen=True)
class JointDecodeChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str
    success_path: str


@dataclass(frozen=True)
class JointDecodeLocalWorkerConfig:
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    sampling: JointDecodeSamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    ledger_path: str
    poll_backoff: float
    microbatch_size: int
    barrier_timeout_s: float
    owner: str
    chip_pair: tuple[int, int]


@dataclass(frozen=True)
class JointDecodeCompletionAlgorithm:
    config: JointDecodeConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_joint_decode_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=prompts_path,
            config=self.config,
        )


def make_joint_decode_completion_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    config: JointDecodeConfig,
) -> ExecutorStep:
    microbatch_size = (
        config.execution.chunk_size if config.execution.microbatch_size is None else config.execution.microbatch_size
    )
    return ExecutorStep(
        name=name,
        fn=remote(
            run_joint_decode_completion_chunks,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm", "tpu"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=JointDecodeCompletionStepConfig(
            output_path=this_output_path(),
            decoder_model_path=version_path(model_path),  # type: ignore[arg-type]
            advisor_model_path=version_path(config.advisor_model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            sampling=versioned(config.sampling),  # type: ignore[arg-type]
            decoder_model=versioned(config.decoder_model),  # type: ignore[arg-type]
            advisor_model=versioned(config.advisor_model),  # type: ignore[arg-type]
            worker_pools=config.execution.worker_pools,
            ledger_prefix=config.execution.ledger_prefix,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            microbatch_size=microbatch_size,
            heartbeat_timeout=config.execution.heartbeat_timeout,
            poll_backoff=config.execution.poll_backoff,
            barrier_timeout_s=config.execution.barrier_timeout_s,
        ),
    )


def _chunk_specs(chunks_dir: str, num_prompts: int, n_samples: int, chunk_size: int) -> list[JointDecodeChunkSpec]:
    total_requests = num_prompts * n_samples
    return [
        JointDecodeChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, total_requests),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
            success_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS"),
        )
        for chunk_id, start in enumerate(range(0, total_requests, chunk_size))
    ]


# ---- JointDecoder: replaces vllm.LLM from the chunk loop's perspective ----


@dataclass
class _GenerateOutput:
    """Mirrors the bits of vllm.RequestOutput that the chunk loop reads."""

    text: str
    finish_reason: str | None


def _select_token(a_topk: list[dict[str, Any]], b_topk: list[dict[str, Any]]) -> int:
    """Pick token from A's top-k with highest rank in B's top-k; fall back to A's top-1."""
    a_ids = [int(t["token_id"]) for t in a_topk]
    if not a_ids:
        raise ValueError("Empty top-k from model A; ensure top_k_a >= 1")
    b_rank = {int(t["token_id"]): i for i, t in enumerate(b_topk)}
    overlap = [(b_rank[tid], tid) for tid in a_ids if tid in b_rank]
    if not overlap:
        return a_ids[0]
    overlap.sort()
    return overlap[0][1]


class _Coordinator:
    """Per-step barrier matching A and B's top-k POSTs from the runner."""

    def __init__(self, timeout_s: float):
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._barriers: dict[bytes, dict[str, Any]] = {}

    def handle(self, side: str, payload: dict[str, Any]) -> dict[str, Any]:
        req_ids = list(payload["request_ids"])
        step_indices = payload["step_indices"]
        topk = payload.get("topk") or {}

        # Canonical key on (req_id, step) pairs — matches both sides at the
        # same lockstep step regardless of dict ordering.
        key = json.dumps(sorted((rid, step_indices[rid]) for rid in req_ids)).encode()

        with self._lock:
            entry: dict[str, Any] | None = self._barriers.get(key)
            if entry is None:
                entry = {
                    "a": None,
                    "b": None,
                    "ready": threading.Event(),
                    "result": None,
                    "req_ids": req_ids,
                }
                self._barriers[key] = entry
            entry[side] = topk

            if entry["a"] is not None and entry["b"] is not None:
                tokens: dict[str, int] = {}
                for rid in entry["req_ids"]:
                    a_topk = entry["a"].get(rid, [])
                    b_topk = entry["b"].get(rid, [])
                    tokens[rid] = _select_token(a_topk, b_topk)
                entry["result"] = {"tokens": tokens}
                entry["ready"].set()

        if not entry["ready"].wait(timeout=self._timeout_s):
            raise TimeoutError(f"Joint-decode barrier timed out for req_ids={req_ids}")

        with self._lock:
            self._barriers.pop(key, None)

        assert entry["result"] is not None
        return entry["result"]


class _DecisionHandler(BaseHTTPRequestHandler):
    coordinator: _Coordinator | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # silence access logs
        return

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            payload = json.loads(body)
            side = self.path.lstrip("/")
            if side not in ("a", "b"):
                self.send_error(404, f"unknown path {self.path!r}")
                return
            assert self.coordinator is not None
            response = self.coordinator.handle(side, payload)
            response_bytes = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as exc:
            logger.exception("Error handling token-decision POST")
            self.send_error(500, str(exc))


class JointDecoder:
    """Two-engine joint decoder. Used as a context manager."""

    def __init__(
        self,
        *,
        decoder_model_path: str,
        advisor_model_path: str,
        sampling: JointDecodeSamplingConfig,
        decoder_model: JointDecodeModelConfig,
        advisor_model: JointDecodeModelConfig,
        chip_a: int = 0,
        chip_b: int = 1,
        server_port: int = 0,
        barrier_timeout_s: float = 60.0,
        microbatch_size: int,
    ) -> None:
        self.decoder_model_path = decoder_model_path
        self.advisor_model_path = advisor_model_path
        self.sampling = sampling
        self.decoder_model = decoder_model
        self.advisor_model = advisor_model
        self.chip_a = chip_a
        self.chip_b = chip_b
        self.server_port = server_port
        self.barrier_timeout_s = barrier_timeout_s
        self.microbatch_size = microbatch_size

        self._coordinator: _Coordinator | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._proc_a: subprocess.Popen | None = None
        self._proc_b: subprocess.Popen | None = None
        self._chunk_seq = 0

    def __enter__(self) -> JointDecoder:
        self._coordinator = _Coordinator(self.barrier_timeout_s)
        _DecisionHandler.coordinator = self._coordinator
        self._http_server = ThreadingHTTPServer(("127.0.0.1", self.server_port), _DecisionHandler)
        actual_port = self._http_server.server_address[1]
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
        logger.info("Joint-decode HTTP coordinator listening on 127.0.0.1:%d", actual_port)

        try:
            self._proc_a = self._spawn_worker(
                chip=self.chip_a,
                model_path=self.decoder_model_path,
                model_cfg=self.decoder_model,
                top_k=self.sampling.top_k_a,
                decision_url=f"http://127.0.0.1:{actual_port}/a",
            )
            self._proc_b = self._spawn_worker(
                chip=self.chip_b,
                model_path=self.advisor_model_path,
                model_cfg=self.advisor_model,
                top_k=self.sampling.top_k_b,
                decision_url=f"http://127.0.0.1:{actual_port}/b",
            )
            handshake_a = self._read_ipc(self._proc_a, expect_kind="handshake")
            handshake_b = self._read_ipc(self._proc_b, expect_kind="handshake")
            self._validate_handshake(handshake_a, handshake_b)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _spawn_worker(
        self,
        *,
        chip: int,
        model_path: str,
        model_cfg: JointDecodeModelConfig,
        top_k: int,
        decision_url: str,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["TPU_VISIBLE_CHIPS"] = str(chip)
        env["TPU_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        env["RERANK_TOKEN_DECISION_URL"] = decision_url
        env["RERANK_TOKEN_DECISION_TOP_K"] = str(top_k)
        # Worker's own HTTP timeout must be > server-side barrier timeout, so
        # the server gets a chance to time out and report rather than the
        # client tearing the connection down first.
        env["RERANK_TOKEN_DECISION_TIMEOUT"] = str(self.barrier_timeout_s + 10.0)
        for key, value in VLLM_TPU_ENV_VARS.items():
            env.setdefault(key, value)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.downstream_scaling.evals.algorithms.joint_decode_xregion",
            "--mode",
            "worker",
            "--chip",
            str(chip),
            "--model-path",
            model_path,
            "--max-tokens",
            str(self.sampling.max_tokens),
            "--max-model-len",
            str(model_cfg.max_model_len),
            "--seed",
            str(self.sampling.seed),
        ]
        if model_cfg.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(model_cfg.gpu_memory_utilization)]
        if model_cfg.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if model_cfg.apply_rpa_block_size_patch:
            cmd.append("--apply-rpa-block-size-patch")
        if self.sampling.stop:
            cmd += ["--stop", json.dumps(list(self.sampling.stop))]

        return subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

    def _read_ipc(self, proc: subprocess.Popen, *, expect_kind: str) -> dict[str, Any]:
        """Block until proc emits an IPC-prefixed line; non-IPC lines are forwarded to logger."""
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                rc = proc.poll()
                raise RuntimeError(f"Joint-decode worker (pid={proc.pid}) exited with rc={rc} before sending IPC")
            line = line.rstrip("\n")
            if line.startswith(IPC_PREFIX):
                payload = json.loads(line[len(IPC_PREFIX) :])
                kind = payload.get("kind")
                if kind != expect_kind:
                    raise RuntimeError(f"Expected IPC kind={expect_kind!r}, got {kind!r}")
                return payload
            if line:
                logger.debug("[worker pid=%d] %s", proc.pid, line)

    def _validate_handshake(self, h_a: dict[str, Any], h_b: dict[str, Any]) -> None:
        if h_a["vocab_size"] != h_b["vocab_size"]:
            raise RuntimeError(
                f"Tokenizer vocab size mismatch: A={h_a['vocab_size']} B={h_b['vocab_size']}; "
                "joint decode requires shared tokenizer."
            )
        if h_a["eos_token_id"] != h_b["eos_token_id"]:
            raise RuntimeError(
                f"EOS token id mismatch: A={h_a['eos_token_id']} B={h_b['eos_token_id']}; "
                "joint decode requires both engines to stop on the same token id."
            )

    def generate(self, prompts: list[str]) -> list[_GenerateOutput]:
        outputs: list[_GenerateOutput] = []
        for start in range(0, len(prompts), self.microbatch_size):
            outputs.extend(self._generate_microbatch(prompts[start : start + self.microbatch_size]))
        return outputs

    def _generate_microbatch(self, prompts: list[str]) -> list[_GenerateOutput]:
        request_ids = [f"jd-c{self._chunk_seq}-r{i:06d}" for i in range(len(prompts))]
        self._chunk_seq += 1
        request = {
            "command": "process_chunk",
            "request_ids": request_ids,
            "prompts": prompts,
        }
        line = json.dumps(request) + "\n"
        for proc in (self._proc_a, self._proc_b):
            assert proc is not None and proc.stdin is not None
            proc.stdin.write(line)
            proc.stdin.flush()

        # Read both subprocesses' results concurrently. Pipe buffers fill at
        # ~64 KB so serial reads can deadlock on large chunks.
        results: dict[str, Any] = {}

        def reader(name: str, proc: subprocess.Popen) -> None:
            try:
                results[name] = self._read_ipc(proc, expect_kind="result")
            except Exception as exc:
                results[name] = exc

        threads = [
            threading.Thread(target=reader, args=("a", self._proc_a), daemon=True),
            threading.Thread(target=reader, args=("b", self._proc_b), daemon=True),
        ]
        for t in threads:
            t.start()
        # Generous deadline: chunk processing is bounded by max_tokens x steps
        # plus the barrier timeout per step. We don't enforce a wall-clock
        # cap here — let the barrier be the source of truth for hangs.
        for t in threads:
            t.join()

        for name in ("a", "b"):
            if isinstance(results.get(name), Exception):
                raise results[name]

        result_a = results["a"]
        text_results: dict[str, str] = result_a["results"]
        finish_reasons: dict[str, str] = result_a.get("finish_reasons", {})

        outputs: list[_GenerateOutput] = []
        for rid in request_ids:
            outputs.append(
                _GenerateOutput(
                    text=text_results.get(rid, ""),
                    finish_reason=finish_reasons.get(rid),
                )
            )
        return outputs

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for name, proc in (("a", self._proc_a), ("b", self._proc_b)):
            if proc is None:
                continue
            try:
                if proc.poll() is None and proc.stdin is not None and not proc.stdin.closed:
                    try:
                        proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        pass
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception:
                logger.exception("Error shutting down joint-decode worker %s", name)
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None


# ---- Subprocess worker (--mode worker) ----


def _patch_rpa_kernel_block_sizes() -> None:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel  # noqa: PLC0415

    original = rpa_kernel.get_default_block_sizes
    if getattr(original, "_marin_joint_decode_patched", False):
        return

    def patched_get_default_block_sizes(*args: Any, **kwargs: Any) -> dict[str, int]:
        sizes = dict(original(*args, **kwargs))
        case = kwargs.get("case")
        if case is not rpa_kernel.RpaCase.DECODE:
            page_size = args[5]
            sizes["bq_sz"] = max(1, sizes["bq_sz"] // 2)
            sizes["bq_csz"] = max(1, sizes["bq_csz"] // 2)
            sizes["bkv_sz"] = max(page_size, sizes["bkv_sz"] // 2)
            sizes["bkv_csz"] = max(page_size, sizes["bkv_csz"] // 2)
        return sizes

    patched_get_default_block_sizes._marin_joint_decode_patched = True  # type: ignore[attr-defined]
    rpa_kernel.get_default_block_sizes = patched_get_default_block_sizes


def _emit_ipc(payload: dict[str, Any]) -> None:
    sys.stdout.write(IPC_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _run_worker(args: argparse.Namespace) -> None:
    # vLLM/JAX logging goes to stderr by default; ensure our logger does too
    # so stdout is reserved for IPC.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [worker chip=%(process)d] %(message)s",
    )

    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)
    if args.apply_rpa_block_size_patch:
        _patch_rpa_kernel_block_sizes()

    from vllm import LLM, SamplingParams  # noqa: PLC0415  # imported after TPU_VISIBLE_CHIPS is set

    resolved_path = discover_hf_checkpoints(args.model_path)[-1]
    resolved_path = localize_mirror_path(resolved_path)
    logger.info("Joint-decode worker chip=%d resolved %s -> %s", args.chip, args.model_path, resolved_path)

    kwargs: dict[str, Any] = {
        "model": resolved_path,
        "trust_remote_code": True,
        "load_format": "runai_streamer",
        "seed": args.seed,
        "tensor_parallel_size": 1,  # subprocess sees one chip
        "data_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        ignore_eos=False,
        stop_token_ids=[eos_id] if eos_id is not None else None,
        stop=json.loads(args.stop) if args.stop else None,
    )

    _emit_ipc({"kind": "handshake", "vocab_size": len(tokenizer), "eos_token_id": eos_id})

    engine = llm.llm_engine
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        msg = json.loads(line)
        cmd = msg.get("command")
        if cmd == "shutdown":
            break
        if cmd != "process_chunk":
            raise RuntimeError(f"Unknown joint-decode worker command: {cmd!r}")

        request_ids: list[str] = msg["request_ids"]
        prompts: list[str] = msg["prompts"]
        for rid, prompt in zip(request_ids, prompts, strict=True):
            engine.add_request(request_id=rid, prompt=prompt, params=sampling_params)

        live = set(request_ids)
        text_results: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}
        while live:
            for output in engine.step():
                if not output.finished:
                    continue
                rid = output.request_id
                if rid not in live:
                    continue
                completion = cast(Any, output).outputs[0]
                text_results[rid] = completion.text
                finish_reasons[rid] = completion.finish_reason or "unknown"
                live.discard(rid)

        _emit_ipc(
            {
                "kind": "result",
                "results": text_results,
                "finish_reasons": finish_reasons,
            }
        )


# ---- Xregion local workers and top-level run ----


def _run_joint_decode_chunk(
    chunk: JointDecodeChunkSpec,
    *,
    decoder: JointDecoder,
    prompt_ids: list[str],
    prompts: list[str],
    n_samples: int,
) -> None:
    request_indices = range(chunk.chunk_start, chunk.chunk_end)
    chunk_prompt_ids = [prompt_ids[i // n_samples] for i in request_indices]
    chunk_completion_indices = [i % n_samples for i in request_indices]
    chunk_prompts = [prompts[i // n_samples] for i in request_indices]

    outputs = decoder.generate(chunk_prompts)

    records = []
    for prompt_id, completion_index, output in zip(
        chunk_prompt_ids,
        chunk_completion_indices,
        outputs,
        strict=True,
    ):
        records.append(
            {
                "id": prompt_id,
                "completion_index": completion_index,
                "completion": {
                    "text": output.text,
                    "metadata": {"finish_reason": output.finish_reason},
                },
            }
        )

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _num_prompts(prompts_path: str) -> int:
    return sum(1 for _ in read_prompt_rows(prompts_path))


def _child_config_from_file(path: str) -> JointDecodeLocalWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    return JointDecodeLocalWorkerConfig(
        decoder_model_path=data["decoder_model_path"],
        advisor_model_path=data["advisor_model_path"],
        prompts_path=data["prompts_path"],
        sampling=JointDecodeSamplingConfig(**data["sampling"]),
        decoder_model=JointDecodeModelConfig(**data["decoder_model"]),
        advisor_model=JointDecodeModelConfig(**data["advisor_model"]),
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        microbatch_size=data["microbatch_size"],
        barrier_timeout_s=data["barrier_timeout_s"],
        owner=data["owner"],
        chip_pair=tuple(data["chip_pair"]),
    )


def _run_joint_decode_local_worker(config: JointDecodeLocalWorkerConfig) -> None:
    expected_visible_chips = ",".join(str(chip) for chip in config.chip_pair)
    actual_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    if actual_visible_chips != expected_visible_chips:
        raise ValueError(f"TPU_VISIBLE_CHIPS={actual_visible_chips!r}, expected {expected_visible_chips!r}")

    prompt_rows = list(read_prompt_rows(config.prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]
    n_samples = config.sampling.n_samples

    with JointDecoder(
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        sampling=config.sampling,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        chip_a=config.chip_pair[0],
        chip_b=config.chip_pair[1],
        server_port=0,
        barrier_timeout_s=config.barrier_timeout_s,
        microbatch_size=config.microbatch_size,
    ) as decoder:
        while True:
            with ledger.claim_next_chunk(config.ledger_path, config.owner) as claim:
                if claim is None:
                    summary = ledger.summarize(config.ledger_path)
                    if summary.done == summary.total:
                        return
                    time.sleep(config.poll_backoff)
                    continue

                _run_joint_decode_chunk(
                    JointDecodeChunkSpec(**claim.chunk),
                    decoder=decoder,
                    prompt_ids=prompt_ids,
                    prompts=prompts,
                    n_samples=n_samples,
                )
                ledger.mark_done(claim)


def _chip_pairs(chips_per_vm: int) -> list[tuple[int, int]]:
    if chips_per_vm % 2 != 0:
        raise ValueError(f"joint decode needs an even number of chips per VM, got {chips_per_vm}")
    return [(start, start + 1) for start in range(0, chips_per_vm, 2)]


def _child_owner(pool_id: str, shard_idx: int, chip_pair: tuple[int, int]) -> str:
    return f"{pool_id}/shard-{shard_idx}/chips-{chip_pair[0]},{chip_pair[1]}"


def _write_child_config(tmpdir: Path, config: JointDecodeLocalWorkerConfig) -> Path:
    chips = "-".join(str(chip) for chip in config.chip_pair)
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("joint-decode local worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: JointDecodeCompletionStepConfig,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    chip_pair: tuple[int, int],
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = JointDecodeLocalWorkerConfig(
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        prompts_path=config.prompts_path,
        sampling=config.sampling,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        microbatch_size=config.microbatch_size,
        barrier_timeout_s=config.barrier_timeout_s,
        owner=_child_owner(pool_id, shard_idx, chip_pair),
        chip_pair=chip_pair,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in chip_pair)

    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label
    env["TPU_PROCESS_BOUNDS"] = "1,1,1"
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,1,1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmpdir / f"jax_cache_{chip_label.replace(',', '_')}")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.downstream_scaling.evals.algorithms.joint_decode_xregion",
        "--xregion-worker-child-config",
        str(config_path),
    ]
    logger.info("Launching joint-decode local worker shard=%d chips=%s", shard_idx, chip_label)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return proc, _stream_child_output(proc, label=chip_label)


def _terminate_children(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _wait_for_children(procs: list[subprocess.Popen[str]], threads: list[threading.Thread], ledger_path: str) -> None:
    while True:
        summary = ledger.summarize(ledger_path)
        ledger_complete = summary.done == summary.total
        all_done = True

        for proc in procs:
            return_code = proc.poll()
            if return_code is None:
                all_done = False
                continue
            if return_code != 0:
                if ledger_complete:
                    logger.warning("joint-decode local worker exited after ledger completion with rc=%d", return_code)
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"joint-decode local worker failed with rc={return_code}; "
                    f"ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(
                f"joint-decode local workers exited before completion: {summary.done}/{summary.total} chunks done"
            )

        time.sleep(1.0)

    for thread in threads:
        thread.join(timeout=5)


def _supervise_joint_decode_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: JointDecodeCompletionStepConfig,
    ledger_path: str,
    pool: WorkerPoolConfig,
) -> Iterator[dict[str, object]]:
    if pool.vm_count != 1:
        raise ValueError(f"joint decode xregion supports only single-VM TPU pools, got vm_count={pool.vm_count}")
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError("joint decode supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set")

    chip_pairs = _chip_pairs(pool.chips_per_vm)
    logger.info(
        "Starting joint-decode supervisor pool=%s shard=%d chips_per_vm=%d chip_pairs=%s",
        pool.pool_id,
        shard_info.shard_idx,
        pool.chips_per_vm,
        chip_pairs,
    )

    with tempfile.TemporaryDirectory(prefix="joint_decode_xregion_local_workers_") as tmp:
        tmpdir = Path(tmp)
        procs: list[subprocess.Popen[str]] = []
        threads: list[threading.Thread] = []
        try:
            for chip_pair in chip_pairs:
                proc, proc_threads = _spawn_child(
                    tmpdir=tmpdir,
                    config=config,
                    ledger_path=ledger_path,
                    pool_id=pool.pool_id,
                    shard_idx=shard_info.shard_idx,
                    chip_pair=chip_pair,
                )
                procs.append(proc)
                threads.extend(proc_threads)
            _wait_for_children(procs, threads, ledger_path)
        except Exception:
            _terminate_children(procs)
            raise

    yield {"status": "done", "pool_id": pool.pool_id, "shard_idx": shard_info.shard_idx}


def run_joint_decode_completion_chunks(config: JointDecodeCompletionStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("joint decode xregion requires at least one worker pool")

    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(chunks_dir, _num_prompts(config.prompts_path), config.sampling.n_samples, config.chunk_size)
    ledger_path = ledger.convert_mirror_path(
        ledger_prefix=config.ledger_prefix,
        output_path=config.output_path,
    )
    ledger.ensure_manifest(ledger_path, chunks)

    def make_process_shard(pool: WorkerPoolConfig):
        return functools.partial(
            _supervise_joint_decode_worker,
            config=config,
            ledger_path=ledger_path,
            pool=pool,
        )

    xregion_pool.run_worker_pools(
        worker_pools=config.worker_pools,
        ledger_path=ledger_path,
        make_process_shard=make_process_shard,
        poll_backoff=config.poll_backoff,
        heartbeat_timeout=config.heartbeat_timeout,
    )

    summary = ledger.summarize(ledger_path)
    if summary.done != summary.total:
        raise RuntimeError(f"joint decode xregion incomplete: {summary.done}/{summary.total} chunks done")

    path = completions_file(config.output_path)
    done_ids = set(ledger.done_chunk_ids(ledger_path))
    chunk_paths = [chunk.output_path for chunk in chunks if chunk.chunk_id in done_ids]
    aggregate_pipeline = (
        Dataset.from_list(chunk_paths)
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "completions": [item["completion"] for item in items],
                "metadata": {
                    "completion_algorithm": "joint_decode_xregion",
                    "decoder_model_path": config.decoder_model_path,
                    "advisor_model_path": config.advisor_model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    aggregate_workers = max(pool.num_workers for pool in config.worker_pools)
    ZephyrContext(
        name="joint-decode-xregion-completions-aggregate",
        max_workers=aggregate_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote joint-decode xregion completion rows to %s", path)


# ---- CLI entry for subprocess workers ----


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker"], default=None)
    parser.add_argument("--xregion-worker-child-config", default=None)
    parser.add_argument("--chip", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--apply-rpa-block-size-patch", action="store_true")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    if args.xregion_worker_child_config is not None:
        _run_joint_decode_local_worker(_child_config_from_file(args.xregion_worker_child_config))
        return
    if args.mode == "worker":
        required = {
            "chip": args.chip,
            "model_path": args.model_path,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing worker arguments: {missing}")
        _run_worker(args)
        return
    raise ValueError("Expected --mode worker or --xregion-worker-child-config")


if __name__ == "__main__":
    _main()
