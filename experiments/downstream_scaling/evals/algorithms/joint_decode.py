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
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import ExecutorStep, InputName, MirroredValue, this_output_path, versioned
from marin.execution.remote import remote
from marin.utils import fsspec_exists
from zephyr import Dataset, ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

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


@dataclass(frozen=True)
class JointDecodeSamplingConfig:
    n_samples: int
    max_tokens: int
    top_k_a: int
    top_k_b: int
    seed: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if self.n_samples != 1:
            raise ValueError(
                "joint_decode is deterministic per prompt; n_samples must be 1 "
                f"(got {self.n_samples})"
            )


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
    num_workers: int
    worker_resources: ResourceConfig
    chunk_size: int = 512
    chip_a: int = 0
    chip_b: int = 1
    barrier_timeout_s: float = 60.0
    server_port: int = 0


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
    num_workers: int
    chunk_size: int
    worker_resources: ResourceConfig
    chip_a: int
    chip_b: int
    barrier_timeout_s: float
    server_port: int


@dataclass(frozen=True)
class JointDecodeChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str
    success_path: str


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
    return ExecutorStep(
        name=name,
        fn=remote(
            run_joint_decode_completion_chunks,
            resources=config.execution.worker_resources,
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
            num_workers=config.execution.num_workers,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            worker_resources=config.execution.worker_resources,
            chip_a=config.execution.chip_a,
            chip_b=config.execution.chip_b,
            barrier_timeout_s=config.execution.barrier_timeout_s,
            server_port=config.execution.server_port,
        ),
    )


def _chunk_specs(
    chunks_dir: str, num_prompts: int, n_samples: int, chunk_size: int
) -> list[JointDecodeChunkSpec]:
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
            entry = self._barriers.get(key)
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

    def log_message(self, format: str, *args: Any) -> None:  # silence access logs
        return

    def do_POST(self) -> None:  # noqa: N802
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
        except Exception as exc:  # noqa: BLE001
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

        self._coordinator: _Coordinator | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._proc_a: subprocess.Popen | None = None
        self._proc_b: subprocess.Popen | None = None
        self._chunk_seq = 0

    def __enter__(self) -> "JointDecoder":
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
            "experiments.downstream_scaling.evals.algorithms.joint_decode",
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
            "--temperature",
            str(self.sampling.temperature),
            "--top-p",
            str(self.sampling.top_p),
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
                raise RuntimeError(
                    f"Joint-decode worker (pid={proc.pid}) exited with rc={rc} before sending IPC"
                )
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
            except Exception as exc:  # noqa: BLE001
                results[name] = exc

        threads = [
            threading.Thread(target=reader, args=("a", self._proc_a), daemon=True),
            threading.Thread(target=reader, args=("b", self._proc_b), daemon=True),
        ]
        for t in threads:
            t.start()
        # Generous deadline: chunk processing is bounded by max_tokens × steps
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
                    except Exception:  # noqa: BLE001
                        pass
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception:  # noqa: BLE001
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
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    orig_get_tuned = rpa_kernel.get_tuned_block_sizes

    def patched_get_tuned(*args: Any, **kwargs: Any) -> Any:
        bkv_p, bq_sz = orig_get_tuned(*args, **kwargs)
        return (max(1, bkv_p // 2), bq_sz)

    rpa_kernel.get_tuned_block_sizes = patched_get_tuned


def _add_request(engine: Any, request_id: str, prompt: Any, sampling_params: Any) -> None:
    """vLLM has shifted between keyword and positional add_request signatures."""
    try:
        engine.add_request(request_id=request_id, prompt=prompt, params=sampling_params)
    except TypeError:
        engine.add_request(request_id, prompt, sampling_params)


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

    from vllm import LLM, SamplingParams  # imported after TPU_VISIBLE_CHIPS is set

    resolved_path = discover_hf_checkpoints(args.model_path)[-1]
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
        temperature=args.temperature,
        top_p=args.top_p,
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
            _add_request(engine, rid, prompt, sampling_params)

        live = set(request_ids)
        text_results: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}
        while live:
            for output in engine.step():
                if not getattr(output, "finished", False):
                    continue
                rid = output.request_id
                if rid not in live:
                    continue
                completion = output.outputs[0]
                text_results[rid] = getattr(completion, "text", "") or ""
                finish_reasons[rid] = getattr(completion, "finish_reason", None) or "unknown"
                live.discard(rid)

        _emit_ipc(
            {
                "kind": "result",
                "results": text_results,
                "finish_reasons": finish_reasons,
            }
        )


# ---- Per-shard process and top-level run (mirrors iid.py) ----


def _process_joint_decode_shard(
    chunks: list[JointDecodeChunkSpec],
    *,
    config: JointDecodeCompletionStepConfig,
    prompt_ids: list[str],
    prompts: list[str],
):
    n_samples = config.sampling.n_samples
    decoder_kwargs = dict(
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        sampling=config.sampling,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        chip_a=config.chip_a,
        chip_b=config.chip_b,
        server_port=config.server_port,
        barrier_timeout_s=config.barrier_timeout_s,
    )

    with JointDecoder(**decoder_kwargs) as decoder:
        for chunk in chunks:
            if fsspec_exists(chunk.success_path):
                yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": True}
                continue

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
            with fsspec.open(chunk.success_path, "wt") as f:
                f.write("ok\n")
            yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": False}


def run_joint_decode_completion_chunks(config: JointDecodeCompletionStepConfig) -> None:
    prompt_rows = list(read_prompt_rows(config.prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]
    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(chunks_dir, len(prompt_rows), config.sampling.n_samples, config.chunk_size)

    for _ in _process_joint_decode_shard(chunks, config=config, prompt_ids=prompt_ids, prompts=prompts):
        pass

    path = completions_file(config.output_path)
    aggregate_pipeline = (
        Dataset.from_files(os.path.join(chunks_dir, "chunk-*.jsonl.gz"))
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "completions": [item["completion"] for item in items],
                "metadata": {
                    "completion_algorithm": "joint_decode",
                    "decoder_model_path": config.decoder_model_path,
                    "advisor_model_path": config.advisor_model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="joint-decode-completions-aggregate",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote joint-decode completion rows to %s", path)


# ---- CLI entry for subprocess workers ----


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker"], required=True)
    parser.add_argument("--chip", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--apply-rpa-block-size-patch", action="store_true")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    if args.mode == "worker":
        _run_worker(args)


if __name__ == "__main__":
    _main()
