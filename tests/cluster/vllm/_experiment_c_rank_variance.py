#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment C: G1 of the rank-variance plan — controlled baseline + in-server microreproducer.

Implements plan G1 (.agents/projects/20260721_vllm_rank_variance_root_cause_plan.md): a
probe-instrumented vLLM serve of the Snowball export with prefix caching OFF, measuring —
in pre-registered order — the collective microreproducer, fresh-recompute determinism
(the load-bearing measurement), concurrent/isolated/staggered/wave-realistic modes, rank
permutation, a length ladder, three extra long prompts, an A-A'-A'' within-node launch
band (three server restarts), and a gate-exact-history session (caching ON, gate wave
order). A second job supplies the cross-job band. Worker instrumentation is injected via
vLLM's supported --worker-extension-cls (see _marin_rank_probe.py); bulk data travels
through a node-local side-channel directory, never through logs.

Emitted blocks (between EXPERIMENT_C_JSON_BEGIN/END markers, one block per line to
survive the ~49k-char Iris log truncation):
  {"experiment": "c", "schema": 1, "session": "S1", "kind": "probe"|"observations",
   "mode": ..., "length": ..., "round": ..., ...}
Observation shape matches experiment B: {rank, greedy_token_id, logprobs, request_id,
elapsed}. Analysis happens offline against these blocks plus the local goldens.

    uv run python tests/cluster/vllm/_experiment_c_rank_variance.py --dry-run
    uv run python tests/cluster/vllm/_experiment_c_rank_variance.py --arm both
"""

import argparse
import json
import logging
import pathlib
import sys
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

_repo_root = pathlib.Path(__file__).resolve().parents[3]
# draccus ships a top-level regular `tests` package that shadows the repo's namespace
# package; shim it for the *local* fixture read only (remote entrypoints import no tests.*).
_tests_pkg = types.ModuleType("tests")
_tests_pkg.__path__ = [str(_repo_root / "tests")]
sys.modules["tests"] = _tests_pkg

from fray.iris_backend import FrayIrisClient  # noqa: E402
from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment  # noqa: E402
from iris.cluster.setup_scripts import default_setup_script  # noqa: E402
from iris.cluster.types import JobName  # noqa: E402
from iris.rpc import job_pb2  # noqa: E402
from iris.test_util import wait_for_condition  # noqa: E402
from rigging.timing import Duration  # noqa: E402

from tests.cluster.conftest import MARIN_GPU_CLUSTER, open_cluster_client  # noqa: E402

logger = logging.getLogger(__name__)

SENTINEL_CASE_ID = "knowledge-longbench-02"
GPU_COUNT = 8
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 512
RETURNED_LOGPROBS = 50
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"
LADDER_LENGTHS = (128, 2048, 8192)  # full length runs via the isolated mode
# G2 trace: 128 tokens is one prefill chunk (26 MoE layers => 26 combine calls), the
# smallest length that still shows the rank spread (S = 0.042 measured in G1).
TRACE_LENGTHS = (128,)
TRACE_MAX_CALLS = 130
# Trace entries run ~200 bytes; keep each emitted block well under the ~49k-char
# limit at which the Iris log truncates a line into unparseable JSON.
TRACE_EMIT_CHUNK = 60
# G3: screen at 128, confirm at the two lengths with the largest measured spreads.
FIXED_COMBINE_LENGTHS = (128, 8192, 15025)
STAGGER_OFFSET = 0.2
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 10 * 60.0
RPC_TIMEOUT = 10 * 60.0
PENDING_TIMEOUT = 30 * 60.0
MAIN_RUNTIME_TIMEOUT = 150 * 60.0
CROSSJOB_RUNTIME_TIMEOUT = 60 * 60.0
PERMUTATION_SEED = 20260721
SIZES_LOG_HEAD, SIZES_LOG_TAIL = 200, 50

# Mirrors tests/cluster/vllm/snowball.py; inlined so remote entrypoints import no tests.*.
MODEL_NAME = "snowball-step-42150-bf16"
EXPORT_URI = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/"
PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8.json"
)
PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
BATCH_SIZE = 8

JSON_BEGIN = "EXPERIMENT_C_JSON_BEGIN"
JSON_END = "EXPERIMENT_C_JSON_END"

PROBE_SOURCE = (pathlib.Path(__file__).parent / "_marin_rank_probe.py").read_text()


def _fetch_fixture_cases() -> dict[str, tuple[int, ...]]:
    """Download the public prompt fixture; return case_id -> prompt token ids."""
    import urllib.request  # noqa: PLC0415 -- runs in the remote job's interpreter

    with urllib.request.urlopen(PROMPT_FIXTURE_URL, timeout=120) as response:
        payload = json.loads(response.read())
    return {case["id"]: tuple(case["prompt_token_ids"]) for case in payload["cases"]}


def _gate_batches(cases: dict[str, tuple[int, ...]]) -> list[list[str]]:
    """Reproduce snowball._prompt_batches bucket order (case ids only)."""
    batches: list[list[str]] = []
    remaining = dict(cases)
    for max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = sorted(case_id for case_id, ids in remaining.items() if len(ids) <= max_tokens)
        for case_id in bucket:
            del remaining[case_id]
        assert len(bucket) % BATCH_SIZE == 0, (max_tokens, len(bucket))
        batches.extend(bucket[start : start + BATCH_SIZE] for start in range(0, len(bucket), BATCH_SIZE))
    assert not remaining
    return batches


def _companion_case_ids(cases: dict[str, tuple[int, ...]]) -> list[str]:
    """Deterministic pick of long non-sentinel cases: first 7 by id serve as wave
    partners, first 3 as the extra-prompt generality check."""
    return sorted(case_id for case_id, ids in cases.items() if len(ids) > 2048 and case_id != SENTINEL_CASE_ID)[:7]


def run_experiment_c(probe_source: str, session_plan: str) -> None:
    """Remote entrypoint. session_plan: "main" (S1+A-band+S4) or "crossjob" (S1-lite)."""
    import glob  # noqa: PLC0415
    import json  # noqa: PLC0415
    import logging  # noqa: PLC0415 -- entrypoint runs in the remote job's interpreter
    import os  # noqa: PLC0415
    import random  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import requests  # noqa: PLC0415
    from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec  # noqa: PLC0415
    from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource  # noqa: PLC0415
    from marin.inference.vllm_backend import VllmBackend  # noqa: PLC0415

    log = logging.getLogger("experiment_c")

    def emit(payload: dict) -> None:
        print(JSON_BEGIN)
        print(json.dumps(payload))
        print(JSON_END)

    # --- probe injection: write the extension module, point PYTHONPATH at it ---
    probe_dir = tempfile.mkdtemp(prefix="marin_probe_mod_")
    with open(os.path.join(probe_dir, "marin_rank_probe.py"), "w") as handle:
        handle.write(probe_source)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = probe_dir + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"

    with urllib.request.urlopen(PROMPT_FIXTURE_URL, timeout=120) as response:
        fixture_payload = json.loads(response.read())
    cases = {case["id"]: tuple(case["prompt_token_ids"]) for case in fixture_payload["cases"]}
    sentinel = cases[SENTINEL_CASE_ID]
    full_length = len(sentinel)
    companions = _companion_case_ids(cases)
    wave_partners, extra_cases = companions[:7], companions[:3]
    truncated = {length: sentinel[-length:] for length in LADDER_LENGTHS}
    truncated[full_length] = sentinel

    spec = ModelSpec(
        model=MODEL_NAME,
        model_path=EXPORT_URI,
        num_chips=GPU_COUNT,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    gate_args = (
        "--data-parallel-size",
        str(GPU_COUNT),
        "--enable-expert-parallel",
        "--model-loader-extra-config",
        '{"distributed":true}',
        "--max-num-seqs",
        "1",
        "--max-logprobs",
        str(RETURNED_LOGPROBS),
        "--attention-backend",
        VLLM_ATTENTION_BACKEND,
        "--worker-extension-cls",
        "marin_rank_probe.MarinRankProbe",
    )

    def engine_config(*, prefix_caching: bool) -> VllmEngineConfig:
        extra = gate_args if prefix_caching else (*gate_args, "--no-enable-prefix-caching")
        return VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            extra_args=extra,
        )

    def rpc(session: str, base_url: str, method: str, **kwargs) -> dict | None:
        """Call a probe method and record its ack. Probe failures are reported and
        stepped over: instrumentation must not abort a battery that still has valid
        measurements to take (the probe itself never raises — see _reported)."""
        try:
            response = requests.post(
                f"{base_url}/collective_rpc",
                json={"method": method, "kwargs": kwargs},
                timeout=(HTTP_CONNECT_TIMEOUT, RPC_TIMEOUT),
            )
            response.raise_for_status()
            payload = response.json() if response.content else None
        except Exception as error:
            payload = {"transport_error": repr(error)}
        emit({"experiment": "c", "schema": 1, "session": session, "kind": "probe_ack", "method": method, "ack": payload})
        log.info("session=%s probe=%s ack=%s", session, method, str(payload)[:400])
        return payload

    def collect_side_files(session: str, out_dir: str, pattern: str, kind: str) -> None:
        for path in sorted(glob.glob(os.path.join(out_dir, pattern))):
            with open(path) as handle:
                emit(
                    {
                        "experiment": "c",
                        "schema": 1,
                        "session": session,
                        "kind": kind,
                        "file": os.path.basename(path),
                        "data": json.load(handle),
                    }
                )

    def collect_sizes_logs(session: str, out_dir: str) -> None:
        for path in sorted(glob.glob(os.path.join(out_dir, "sizes_rank*_*.jsonl"))):
            with open(path) as handle:
                lines = [json.loads(line) for line in handle]
            trimmed = (
                lines[:SIZES_LOG_HEAD] + lines[-SIZES_LOG_TAIL:]
                if len(lines) > SIZES_LOG_HEAD + SIZES_LOG_TAIL
                else lines
            )
            emit(
                {
                    "experiment": "c",
                    "schema": 1,
                    "session": session,
                    "kind": "sizes_log",
                    "file": os.path.basename(path),
                    "total_lines": len(lines),
                    "data": trimmed,
                }
            )

    def request_case(
        completions_url: str, model_id: str, token_ids: tuple[int, ...], rank: int, request_id: str, delay: float = 0.0
    ) -> dict:
        if delay:
            time.sleep(delay)
        start = time.monotonic()
        response = requests.post(
            completions_url,
            headers={"X-data-parallel-rank": str(rank), "X-Request-Id": request_id},
            json={
                "model": model_id,
                "prompt": list(token_ids),
                "add_special_tokens": False,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": RETURNED_LOGPROBS,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            },
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        )
        response.raise_for_status()
        (choice,) = response.json()["choices"]
        assert choice["prompt_token_ids"] == list(token_ids), request_id
        (greedy_token_id,) = choice["token_ids"]
        (returned,) = choice["logprobs"]["top_logprobs"]
        logprobs = {int(token.removeprefix("token_id:")): float(lp) for token, lp in returned.items()}
        return {
            "rank": rank,
            "greedy_token_id": int(greedy_token_id),
            "logprobs": {str(t): logprobs[t] for t in sorted(logprobs)},
            "request_id": request_id,
            "elapsed": round(time.monotonic() - start, 3),
        }

    def measured_battery(session: str, served, out_dir: str, *, lite: bool) -> None:
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        model_id = served.model_id

        def one(token_ids, rank, request_id):
            return request_case(completions_url, model_id, token_ids, rank, f"{session}-{request_id}")

        def rotate_sizes(tag: str) -> None:
            """Give each mode its own sizes-log budget: a single 15k prefill spends
            ~780 combine calls, so a shared budget only ever covers the warmup."""
            rpc(session, served.base_url, "marin_probe_rotate_sizes_log", tag=tag)

        def emit_mode(mode: str, length: int, observations: list[dict], round_index: int = 0) -> None:
            emit(
                {
                    "experiment": "c",
                    "schema": 1,
                    "session": session,
                    "kind": "observations",
                    "mode": mode,
                    "length": length,
                    "round": round_index,
                    "observations": observations,
                }
            )
            log.info(
                "session=%s mode=%s length=%d round=%d done (%d obs)",
                session,
                mode,
                length,
                round_index,
                len(observations),
            )

        def concurrent_wave(
            token_ids_by_rank: dict[int, tuple[int, ...]], mode: str, request_prefix: str, stagger: float = 0.0
        ) -> list[dict]:
            with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
                futures = [
                    executor.submit(
                        request_case,
                        completions_url,
                        model_id,
                        token_ids,
                        rank,
                        f"{session}-{request_prefix}-rank{rank}",
                        index * stagger,
                    )
                    for index, (rank, token_ids) in enumerate(sorted(token_ids_by_rank.items()))
                ]
                return [future.result() for future in futures]

        # 1) injection smoke + effective-config record
        rpc(session, served.base_url, "marin_probe_env", out_dir=out_dir)
        collect_side_files(session, out_dir, "env_rank*.json", "probe_env")
        # 2) per-call combine-variant log for everything that follows
        rpc(session, served.base_url, "marin_probe_install_sizes_log", out_dir=out_dir)
        # 3) collective microreproducer (H1 primitive, both variants), before any inference
        rpc(session, served.base_url, "marin_probe_microreproducer", out_dir=out_dir)
        collect_side_files(session, out_dir, "micro_rank*.json", "probe_micro")
        # 4) one discarded warmup request (JIT/lazy-init settle), recorded but flagged
        rotate_sizes("warmup")
        emit_mode("warmup", full_length, [one(sentinel, 0, "warmup")])
        # 5) fresh-recompute determinism — the load-bearing measurement
        repeats = 4 if lite else 8
        rotate_sizes("fresh")
        emit_mode("fresh_determinism", full_length, [one(sentinel, 0, f"fresh-r0-{index}") for index in range(repeats)])
        if not lite:
            emit_mode(
                "fresh_determinism", 8192, [one(truncated[8192], 4, f"fresh-r4-{index}") for index in range(repeats)]
            )
        # 6) C1: all ranks concurrent, same prompt
        for round_index in range(1 if lite else 2):
            rotate_sizes(f"c1-{round_index}")
            emit_mode(
                "c1_concurrent",
                full_length,
                concurrent_wave({rank: sentinel for rank in range(GPU_COUNT)}, "c1", f"c1-{round_index}"),
                round_index,
            )
        # 7) isolated: sequential per-rank replay (experiment-B protocol)
        for length in (full_length, 8192):
            rotate_sizes(f"iso-{length}")
            emit_mode(
                "isolated",
                length,
                [one(truncated[length], rank, f"iso-{length}-rank{rank}") for rank in range(GPU_COUNT)],
            )
        if lite:
            collect_sizes_logs(session, out_dir)
            return
        # 8) staggered wave
        for length in (full_length, 8192):
            rotate_sizes(f"stag-{length}")
            emit_mode(
                "staggered",
                length,
                concurrent_wave(
                    {rank: truncated[length] for rank in range(GPU_COUNT)},
                    "staggered",
                    f"stag-{length}",
                    stagger=STAGGER_OFFSET,
                ),
            )
        # 9) wave-realistic: sentinel on one rank, distinct long cases on the others
        for round_index, sentinel_rank in enumerate((0, 4, 0, 4)):
            rotate_sizes(f"wave-{round_index}")
            wave = {}
            partner_iter = iter(wave_partners)
            for rank in range(GPU_COUNT):
                wave[rank] = sentinel if rank == sentinel_rank else cases[next(partner_iter)]
            observations = concurrent_wave(wave, "wave_realistic", f"wave-{round_index}")
            emit_mode("wave_realistic", full_length, observations, round_index)
        # 10) target-rank permutation: same replay, different service order
        for round_index, order in enumerate(
            (
                tuple(reversed(range(GPU_COUNT))),
                tuple(random.Random(PERMUTATION_SEED).sample(range(GPU_COUNT), GPU_COUNT)),
            )
        ):
            emit_mode(
                "permutation",
                full_length,
                [one(sentinel, rank, f"perm-{round_index}-rank{rank}") for rank in order],
                round_index,
            )
        # 11) length ladder
        for length in LADDER_LENGTHS:
            emit_mode(
                "ladder",
                length,
                [one(truncated[length], rank, f"ladder-{length}-rank{rank}") for rank in range(GPU_COUNT)],
            )
        # 12) generality: three additional long prompts, every rank
        for case_id in extra_cases:
            emit_mode(
                f"extra:{case_id}",
                len(cases[case_id]),
                [one(cases[case_id], rank, f"extra-{case_id}-rank{rank}") for rank in range(GPU_COUNT)],
            )
        collect_sizes_logs(session, out_dir)

    def gate_exact_battery(session: str, served, out_dir: str) -> None:
        """Caching ON, exact gate history: bucket waves in fixture order, then the
        concurrent sentinel wave. Probe RPCs only after all measurements."""
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        batches = _gate_batches(cases)
        with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
            for wave_index, batch in enumerate(batches):
                futures = [
                    executor.submit(
                        request_case,
                        completions_url,
                        served.model_id,
                        cases[case_id],
                        rank,
                        f"{session}-wave{wave_index}-{case_id}-rank{rank}",
                    )
                    for rank, case_id in enumerate(batch)
                ]
                observations = [future.result() for future in futures]
                emit(
                    {
                        "experiment": "c",
                        "schema": 1,
                        "session": session,
                        "kind": "observations",
                        "mode": "gate_wave",
                        "length": 0,
                        "round": wave_index,
                        "observations": observations,
                    }
                )
            futures = [
                executor.submit(
                    request_case, completions_url, served.model_id, sentinel, rank, f"{session}-sentinel-rank{rank}"
                )
                for rank in range(GPU_COUNT)
            ]
            observations = [future.result() for future in futures]
            emit(
                {
                    "experiment": "c",
                    "schema": 1,
                    "session": session,
                    "kind": "observations",
                    "mode": "gate_sentinel",
                    "length": full_length,
                    "round": 0,
                    "observations": observations,
                }
            )
        rpc(session, served.base_url, "marin_probe_env", out_dir=out_dir)
        collect_side_files(session, out_dir, "env_rank*.json", "probe_env")

    def collect_trace(session: str, out_dir: str, tag: str) -> None:
        for path in sorted(glob.glob(os.path.join(out_dir, f"trace_rank*_{tag}.jsonl"))):
            with open(path) as handle:
                entries = [json.loads(line) for line in handle]
            # One block per rank exceeds the ~49k-char log line limit and is truncated
            # into unparseable JSON (sharp edge #17), so emit in chunks.
            for start in range(0, len(entries), TRACE_EMIT_CHUNK):
                emit(
                    {
                        "experiment": "c",
                        "schema": 1,
                        "session": session,
                        "kind": "trace",
                        "tag": tag,
                        "file": os.path.basename(path),
                        "chunk": start // TRACE_EMIT_CHUNK,
                        "data": entries[start : start + TRACE_EMIT_CHUNK],
                    }
                )
        log.info("session=%s trace tag=%s done", session, tag)

    def fixed_combine_battery(session: str, served, out_dir: str) -> None:
        """G3 combine branch: does a destination-independent combine collapse the spread?

        B-A-A-B bracketed within one server so the launch state is held fixed — the one
        variable that G1 showed moves per-rank values on its own. Pre-registered: the
        treatment collapses the rank spread at every length; the bracketing baselines
        reproduce each other.
        """
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        rpc(session, served.base_url, "marin_probe_env", out_dir=out_dir)
        collect_side_files(session, out_dir, "env_rank*.json", "probe_env")
        request_case(completions_url, served.model_id, truncated[128], 0, f"{session}-warmup")

        def sweep(phase: str) -> None:
            for length in FIXED_COMBINE_LENGTHS:
                observations = [
                    request_case(
                        completions_url, served.model_id, truncated[length], rank, f"{session}-{phase}-{length}-r{rank}"
                    )
                    for rank in range(GPU_COUNT)
                ]
                emit(
                    {
                        "experiment": "c",
                        "schema": 1,
                        "session": session,
                        "kind": "observations",
                        "mode": f"fixedcombine:{phase}",
                        "length": length,
                        "round": 0,
                        "observations": observations,
                    }
                )
                log.info("session=%s phase=%s length=%d done", session, phase, length)

        sweep("baseline_pre")
        rpc(session, served.base_url, "marin_probe_install_fixed_combine")
        sweep("treatment_a")
        sweep("treatment_b")
        rpc(session, served.base_url, "marin_probe_uninstall_fixed_combine")
        sweep("baseline_post")

    def trace_battery(session: str, served, out_dir: str) -> None:
        """G2: capture the MoE boundaries on every rank, once per serving rank.

        Comparing rank r's own prefill across r is the like-for-like measurement: it
        asks whether the same computation, on the same weights and tokens, gives the
        same answer regardless of which rank runs it.
        """
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        rpc(session, served.base_url, "marin_probe_env", out_dir=out_dir)
        collect_side_files(session, out_dir, "env_rank*.json", "probe_env")

        # Warm up outside the trace so JIT/lazy-init work never lands in the capture.
        request_case(completions_url, served.model_id, truncated[128], 0, f"{session}-warmup")
        # One capture per serving rank, not one per wave. The engines serialize these
        # requests -- the DP size vector during a "concurrent" wave is [1,...,128,...,1],
        # one real chunk and seven dummy tokens -- so a shared capture would compare a
        # rank's real tokens against other ranks' dummies. Arming per serving rank makes
        # capture r hold exactly r's own prefill, which is the like-for-like comparison.
        for length in TRACE_LENGTHS:
            for serving_rank in range(GPU_COUNT):
                tag = f"len{length}r{serving_rank}"
                rpc(
                    session,
                    served.base_url,
                    "marin_probe_arm_trace",
                    out_dir=out_dir,
                    tag=tag,
                    max_calls=TRACE_MAX_CALLS,
                )
                observation = request_case(
                    completions_url, served.model_id, truncated[length], serving_rank, f"{session}-{tag}"
                )
                rpc(session, served.base_url, "marin_probe_disarm_trace")
                emit(
                    {
                        "experiment": "c",
                        "schema": 1,
                        "session": session,
                        "kind": "observations",
                        "mode": f"trace:len{length}",
                        "length": length,
                        "round": serving_rank,
                        "observations": [observation],
                    }
                )
                collect_trace(session, out_dir, tag)

    backend_off = VllmBackend(engine_config(prefix_caching=False))
    if session_plan == "crossjob":
        out_dir = tempfile.mkdtemp(prefix="marin_probe_out_XJ_")
        with backend_off.serve(spec) as served:
            measured_battery("XJ", served, out_dir, lite=True)
        return

    if session_plan == "trace":
        for session in ("T1", "T2"):
            out_dir = tempfile.mkdtemp(prefix=f"marin_probe_out_{session}_")
            with backend_off.serve(spec) as served:
                trace_battery(session, served, out_dir)
        return

    if session_plan == "tracefixed":
        # Same per-rank capture, with the destination-independent combine installed.
        # If ranks still differ at the first MoE output once the collective can no
        # longer be the cause, the remaining difference is in the partials fed to it.
        out_dir = tempfile.mkdtemp(prefix="marin_probe_out_TF_")
        with backend_off.serve(spec) as served:
            rpc("TF", served.base_url, "marin_probe_install_fixed_combine")
            trace_battery("TF", served, out_dir)
        return

    if session_plan == "fixedcombine":
        for session in ("F1", "F2"):
            out_dir = tempfile.mkdtemp(prefix=f"marin_probe_out_{session}_")
            with backend_off.serve(spec) as served:
                fixed_combine_battery(session, served, out_dir)
        return

    assert session_plan == "main", session_plan
    for session in ("S1", "S2", "S3"):
        out_dir = tempfile.mkdtemp(prefix=f"marin_probe_out_{session}_")
        with backend_off.serve(spec) as served:
            measured_battery(session, served, out_dir, lite=session != "S1")
    out_dir = tempfile.mkdtemp(prefix="marin_probe_out_S4_")
    with VllmBackend(engine_config(prefix_caching=True)).serve(spec) as served:
        gate_exact_battery("S4", served, out_dir)


def _job_request(session_plan: str) -> JobRequest:
    return JobRequest(
        name=f"snowball-experiment-c-{session_plan}-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(run_experiment_c, args=[PROBE_SOURCE, session_plan]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g"),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "NCCL_DEBUG": "INFO",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def submit(session_plan: str) -> int:
    request = _job_request(session_plan)
    timeout = MAIN_RUNTIME_TIMEOUT if session_plan == "main" else CROSSJOB_RUNTIME_TIMEOUT
    with open_cluster_client(MARIN_GPU_CLUSTER) as client:
        job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
        logger.info("submitted %s plan=%s", job.job_id, session_plan)
        try:
            task_id = JobName.from_string(job.job_id).task(0)
            wait_for_condition(
                lambda: client.task_status(task_id).state
                not in (job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING),
                timeout=Duration.from_seconds(PENDING_TIMEOUT),
                poll_interval=5,
            )
            job.wait(timeout=timeout, stream_logs=True)
        finally:
            if not JobStatus.finished(job.status()):
                logger.warning("terminating unfinished job %s", job.job_id)
                job.terminate()
    return 0


def dry_run() -> int:
    cases = _fetch_fixture_cases()
    sentinel = cases[SENTINEL_CASE_ID]
    companions = _companion_case_ids(cases)
    batches = _gate_batches(cases)
    summary = {
        "sentinel_length": len(sentinel),
        "ladder": [*LADDER_LENGTHS, len(sentinel)],
        "wave_partners": companions[:7],
        "extra_cases": companions[:3],
        "gate_waves": len(batches),
        "gate_wave_first": batches[0],
        "probe_source_bytes": len(PROBE_SOURCE),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["main", "crossjob", "trace", "tracefixed", "fixedcombine", "both"])
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if arguments.dry_run:
        return dry_run()
    if arguments.arm == "both":
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(submit, plan) for plan in ("main", "crossjob")]
            return max(future.result() for future in futures)
    assert arguments.arm, "--arm or --dry-run required"
    return submit(arguments.arm)


if __name__ == "__main__":
    sys.exit(main())
