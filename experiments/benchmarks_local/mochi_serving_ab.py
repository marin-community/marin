# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mochi (535B-A23B) serving settings on one data/expert-parallel vLLM engine: decode and prefill per setting.

``stage-metadata`` runs on every node before Ray starts: it copies the export's config and weight index and the
pinned tokenizer into a node-local directory. ``run`` runs on the Ray head: it starts one engine (TP 1, DP = EP =
serving GPUs) through the vendored SkyRL engine factory, streams the export's real weights (or uses dummy weights),
and submits fixed-length request waves. Each wave sends ``per_gpu x GPUs`` requests at once with one prompt length
and exactly ``--decode-length`` generated tokens. Engine counters are read every ``--sample-seconds``; the prefill
rate is the wave's prompt tokens over the time until the prompt counter reaches them, and the decode step is the
mean running sequences per GPU over the generated tokens per second per GPU in the middle 60% of the decode-only
phase.
"""

import argparse
import asyncio
import dataclasses
import glob
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

NCCL_DIR = "/tmp/mochi-nccl"
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"


def stage_metadata(args) -> None:
    import fsspec
    from transformers import AutoTokenizer

    started = time.monotonic()
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(args.source)
    for name in ("config.json", "model.safetensors.index.json", "export-manifest.json"):
        fs.get(f"{root.rstrip('/')}/{name}", str(destination / name))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(destination))
    config = json.loads((destination / "config.json").read_text())
    assert len(tokenizer) == config["vocab_size"], (len(tokenizer), config["vocab_size"])
    Path(NCCL_DIR).mkdir(parents=True, exist_ok=True)
    print(
        "MOCHI_AB_STAGE "
        + json.dumps(
            {
                "metadata_seconds": round(time.monotonic() - started, 1),
                "files": sorted(p.name for p in destination.iterdir()),
            }
        ),
        flush=True,
    )


def build_config(args):
    from omegaconf import open_dict
    from skyrl_train.config.utils import get_default_config

    gpus = args.serving_nodes * 8
    cfg = get_default_config()
    model = cfg.trainer.policy.model
    model.path = args.metadata_dir
    model.tokenizer_path = args.metadata_dir
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.logger = "console"
    cfg.trainer.run_name = args.name
    cfg.trainer.algorithm.batch_invariant = bool(args.batch_invariant)
    cfg.trainer.distributed.placement_group_timeout_seconds = 1800
    generator = cfg.generator
    generator.backend = "vllm"
    generator.run_engines_locally = True
    generator.enable_http_endpoint = False
    generator.weight_sync_backend = "nccl"
    sampling = generator.sampling_params
    sampling.max_generate_length = args.decode_length
    sampling.temperature = 1.0
    sampling.top_p = 1.0
    sampling.top_k = -1
    sampling.logprobs = 0
    sampling.stop = None
    with open_dict(sampling):
        # Every request generates exactly decode_length tokens.
        sampling.min_tokens = args.decode_length
        sampling.stop_token_ids = []
    generator.inference_engine_tensor_parallel_size = 1
    generator.inference_engine_data_parallel_size = gpus
    generator.inference_engine_expert_parallel_size = gpus
    generator.inference_engine_pipeline_parallel_size = 1
    generator.num_inference_engines = 1
    generator.max_num_batched_tokens = args.max_num_batched_tokens
    generator.max_num_seqs = args.max_num_seqs
    generator.gpu_memory_utilization = args.gpu_memory_utilization
    generator.enable_prefix_caching = False
    generator.enforce_eager = False
    generator.engine_init_timeout_seconds = 4200
    engine_init_kwargs = {
        "enable_return_routed_experts": args.route_capture,
        "max_model_len": args.max_model_len,
        "enable_flashinfer_autotune": False,
        "kernel_config": {"moe_backend": "triton"},
        "hf_overrides": {"max_seq_len": args.max_model_len, "max_position_embeddings": args.max_model_len},
    }
    if args.weights == "dummy":
        engine_init_kwargs["load_format"] = "dummy"
    else:
        with open_dict(model):
            model.source_uri = args.source
        generator.model_loading = "stream"
        # Bound each process's CPU staging buffer: unbounded, eight streaming processes held about 120 GB each on a
        # node while loading the 1.07 TB export.
        engine_init_kwargs["model_loader_extra_config"] = {
            "distributed": bool(args.runai_distributed),
            "memory_limit": int(args.runai_memory_limit_gib * 2**30),
        }
    generator.engine_init_kwargs = engine_init_kwargs
    return cfg


def put(output: str, name: str, data: bytes) -> None:
    import fsspec

    with fsspec.open(f"{output.rstrip('/')}/{name}", "wb") as handle:
        handle.write(data)


def save_report(output: str, report: dict) -> None:
    put(output, "report.json", json.dumps(report, default=str).encode())


def engine_rows(snapshot, histograms=False):
    """Per-engine current and cumulative counters; histogram sums only when asked."""
    rows = []
    for engine in snapshot.engines:
        row = {
            "engine": engine.engine_id,
            "t": engine.timestamp,
            "current": dataclasses.asdict(engine.current),
            "cumulative": dataclasses.asdict(engine.cumulative),
        }
        if histograms:
            row["histograms"] = [
                {"name": h.name, "attributes": dict(h.attributes), "count": h.count, "total": h.total}
                for h in engine.histograms
            ]
        rows.append(row)
    return rows


def totals(rows):
    return {
        "prompt_tokens": sum(r["cumulative"]["prompt_tokens"] for r in rows),
        "generation_tokens": sum(r["cumulative"]["generation_tokens"] for r in rows),
        "preemptions": sum(r["cumulative"]["preemptions"] for r in rows),
        "running": sum(r["current"]["running_requests"] for r in rows),
        "waiting": sum(r["current"]["waiting_capacity"] + r["current"]["waiting_deferred"] for r in rows),
        "kv_usage_max": max((r["current"]["kv_cache_usage"] for r in rows), default=0.0),
        "kv_usage_mean": sum(r["current"]["kv_cache_usage"] for r in rows) / max(len(rows), 1),
        "engines": len(rows),
    }


def summarize(samples, start, gpus, requests, prompt_length, decode_length):
    """Prefill phase ends when the prompt counter reaches the wave's prompt tokens; decode is measured after it."""
    base = samples[0]["totals"]
    wanted = requests * prompt_length
    prefill_end = next(
        (s["t"] for s in samples if s["totals"]["prompt_tokens"] - base["prompt_tokens"] >= 0.999 * wanted), None
    )
    out = {
        "prompt_tokens_counted": samples[-1]["totals"]["prompt_tokens"] - base["prompt_tokens"],
        "generation_tokens_counted": samples[-1]["totals"]["generation_tokens"] - base["generation_tokens"],
        "preemptions": samples[-1]["totals"]["preemptions"] - base["preemptions"],
        "kv_usage_max": max(s["totals"]["kv_usage_max"] for s in samples),
        "running_max": max(s["totals"]["running"] for s in samples),
        "prefill_end_seconds": None if prefill_end is None else prefill_end - start,
    }
    if prefill_end is None:
        return out
    out["prefill_tokens_per_second_per_gpu"] = wanted / max(prefill_end - start, 1e-9) / gpus
    decode = [s for s in samples if s["t"] >= prefill_end and s["totals"]["running"] > 0]
    if len(decode) >= 5:
        span = decode[-1]["t"] - decode[0]["t"]
        window = [s for s in decode if decode[0]["t"] + 0.2 * span <= s["t"] <= decode[0]["t"] + 0.8 * span]
        if len(window) >= 2:
            dt = window[-1]["t"] - window[0]["t"]
            dg = window[-1]["totals"]["generation_tokens"] - window[0]["totals"]["generation_tokens"]
            running = sum(s["totals"]["running"] for s in window) / len(window)
            out.update(
                decode_window_seconds=dt,
                decode_tokens_per_second_per_gpu=dg / dt / gpus,
                decode_running_per_gpu=running / gpus,
                decode_step_ms=1000.0 * (running / gpus) / max(dg / dt / gpus, 1e-9),
                decode_mean_context=prompt_length + decode_length / 2,
            )
    return out


def engine_core_pids():
    """EngineCore processes on this node (the DP ranks whose Ray actors run here)."""
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        text = " ".join(process.info.get("cmdline") or []) + " " + (process.info.get("name") or "")
        if "EngineCore" in text:
            found.append((process.info["pid"], text.strip()[:200]))
    return found


def py_spy_dumps(count=2, interval=5.0):
    """Python stacks of one EngineCore process on the head node, taken mid-decode."""
    pids = engine_core_pids()
    result = {"engine_core_processes": pids[:16], "dumps": []}
    if not pids:
        result["error"] = "no EngineCore process found on the head node"
        return result
    binary = os.path.join(os.path.dirname(sys.executable), "py-spy")
    if not os.path.exists(binary):
        binary = "py-spy"
    pid = pids[0][0]
    for index in range(count):
        try:
            completed = subprocess.run([binary, "dump", "--pid", str(pid)], capture_output=True, text=True, timeout=60)
            result["dumps"].append(
                {
                    "pid": pid,
                    "t": time.time(),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-20000:],
                    "stderr": completed.stderr[-4000:],
                }
            )
        except Exception as error:
            result["dumps"].append({"pid": pid, "error": repr(error)})
        if index + 1 < count:
            time.sleep(interval)
    return result


NCCL_PATTERNS = {
    "coll_channels": re.compile(
        r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer"
    ),
    "comm_init": re.compile(r"comm 0x[0-9a-f]+ rank (\d+) nRanks (\d+) nNodes (\d+) localRanks (\d+)"),
    "env_set": re.compile(r"(NCCL_[A-Z_]+) set by environment to (\S+)"),
    "channel_line": re.compile(r"Channel (\d+)/(\d+) :"),
}


def nccl_summary(directory=NCCL_DIR):
    """Channel counts, communicator sizes and environment overrides from this node's NCCL INFO logs."""
    files = sorted(glob.glob(os.path.join(directory, "*")))
    channels = Counter()
    comms = Counter()
    env = Counter()
    channel_totals = Counter()
    lines = 0
    for path in files:
        try:
            with open(path, errors="replace") as handle:
                for line in handle:
                    lines += 1
                    if match := NCCL_PATTERNS["coll_channels"].search(line):
                        channels[match.group(0)] += 1
                    elif match := NCCL_PATTERNS["comm_init"].search(line):
                        comms[f"nRanks {match.group(2)} nNodes {match.group(3)} localRanks {match.group(4)}"] += 1
                    elif match := NCCL_PATTERNS["env_set"].search(line):
                        env[f"{match.group(1)}={match.group(2)}"] += 1
                    elif match := NCCL_PATTERNS["channel_line"].search(line):
                        channel_totals[f"of {match.group(2)}"] += 1
        except OSError:
            continue
    return {
        "files": len(files),
        "lines": lines,
        "coll_channels": dict(channels),
        "communicators": dict(comms),
        "environment": dict(env),
        "channel_lines_by_total": dict(channel_totals),
    }


def nccl_archive(directory=NCCL_DIR) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            archive.add(path, arcname=os.path.basename(path))
    return buffer.getvalue()


async def run_point(
    client, sampling, gpus, prompt_length, per_gpu, decode_length, sample_seconds, timeout, seed, on_decode=None
):
    from skyrl_train.inference_engines.vllm.stats import IntervalReadMode

    requests = per_gpu * gpus
    rng = random.Random(seed)
    # Token ids from the ordinary vocabulary range; the content does not matter for timing.
    prompts = [[rng.randrange(1000, 120000) for _ in range(prompt_length)] for _ in range(requests)]
    samples = []
    first = await client.get_stats(IntervalReadMode.PEEK)
    start = time.time()
    base = totals(engine_rows(first))
    samples.append({"t": start, "totals": base})
    task = asyncio.create_task(client.generate({"prompt_token_ids": prompts, "sampling_params": sampling}))
    triggered = False
    while not task.done():
        if time.time() - start > timeout:
            task.cancel()
            raise TimeoutError(f"point exceeded {timeout} s")
        await asyncio.wait([task], timeout=sample_seconds)
        rows = engine_rows(await client.get_stats(IntervalReadMode.PEEK))
        samples.append({"t": time.time(), "totals": totals(rows)})
        current = samples[-1]["totals"]
        print(
            "MOCHI_AB_SAMPLE "
            + json.dumps({"L": prompt_length, "c": per_gpu, "s": round(samples[-1]["t"] - start, 1), **current}),
            flush=True,
        )
        prefilled = current["prompt_tokens"] - base["prompt_tokens"] >= 0.999 * requests * prompt_length
        if on_decode is not None and not triggered and prefilled and current["running"] > 0:
            triggered = True
            on_decode()
    output = task.result()
    finished = time.time()
    last = await client.get_stats(IntervalReadMode.PEEK)
    lengths = [len(r) for r in output["response_ids"]]
    point = {
        "prompt_length": prompt_length,
        "per_gpu": per_gpu,
        "requests": requests,
        "decode_length": decode_length,
        "wall_seconds": finished - start,
        "response_lengths": {"min": min(lengths), "max": max(lengths)},
        "stop_reasons": sorted(set(output["stop_reasons"])),
        "end_engines": engine_rows(last, histograms=True),
        "start_engines": engine_rows(first, histograms=True),
        "samples": samples,
    }
    point["summary"] = summarize(samples, start, gpus, requests, prompt_length, decode_length)
    return point


def parse_plan(text):
    plan = []
    for item in text.split(";"):
        length, concurrencies = item.split(":")
        plan.append((int(length), [int(c) for c in concurrencies.split(",")]))
    return plan


def run(args) -> int:
    import ray
    import skyrl_train.utils.utils as skyrl_utils
    from omegaconf import OmegaConf

    nccl_env = {
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH",
        "NCCL_DEBUG_FILE": f"{NCCL_DIR}/nccl.%h.%p.log",
    }
    original_prepare = skyrl_utils.prepare_runtime_environment

    def prepare_with_nccl_info(cfg):
        env_vars = original_prepare(cfg)
        env_vars.update(nccl_env)
        return env_vars

    skyrl_utils.prepare_runtime_environment = prepare_with_nccl_info
    cfg = build_config(args)
    gpus = args.serving_nodes * 8
    report = {
        "status": "starting",
        "args": vars(args),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "points": [],
        "errors": [],
        "started_at": time.time(),
    }
    save_report(args.output, report)
    skyrl_utils.initialize_ray(cfg)
    # Engine actors take managed names from this process's environment; keep NCCL INFO there too.
    os.environ.update(nccl_env)
    # The RunAI S3 low-throughput interval; engine actors take it from this process's environment.
    os.environ["RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS"] = str(args.runai_request_timeout_ms)
    report["driver_environment"] = {
        key: os.environ.get(key)
        for key in (
            "VLLM_BATCH_INVARIANT",
            "NCCL_DEBUG",
            "NCCL_MAX_NCHANNELS",
            "NCCL_PROTO",
            "NCCL_CUMEM_ENABLE",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
        )
    }

    from skyrl_train.config.rollout_validation import EntrypointOperation
    from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
    from skyrl_train.tokenizer import tokenizer_from_config

    tokenizer = tokenizer_from_config(cfg)
    started = time.monotonic()
    try:
        engines = create_ray_wrapped_inference_engines_from_config(
            cfg, None, tokenizer, operation=EntrypointOperation.GENERATE
        )
    except Exception:
        report.update(
            status="engine_init_failed",
            engine_init_seconds=time.monotonic() - started,
            engine_init_error=traceback.format_exc(),
        )
        print(
            "MOCHI_AB_ENGINE_FAILED "
            + json.dumps({"seconds": report["engine_init_seconds"], "error": report["engine_init_error"][-6000:]}),
            flush=True,
        )
        report["nccl_head_node"] = nccl_summary()
        save_report(args.output, report)
        return 1
    client = InferenceEngineClient(engines, tokenizer, cfg)
    report.update(status="running", engine_init_seconds=time.monotonic() - started, engine_count=len(engines))
    try:
        report["placement"] = ray.get(engines[0].inference_engine_actor.report_engine_placement.remote())
    except Exception:
        report["placement_error"] = traceback.format_exc()[-2000:]
    report["nccl_after_init"] = nccl_summary()
    save_report(args.output, report)
    print(
        "MOCHI_AB_ENGINE "
        + json.dumps(
            {"init_seconds": report["engine_init_seconds"], "engines": len(engines), "nccl": report["nccl_after_init"]}
        )[:6000],
        flush=True,
    )
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    report["sampling"] = sampling

    async def main():
        warm = await run_point(
            client,
            {**sampling, "max_tokens": 32, "min_tokens": 32},
            gpus,
            1024,
            1,
            32,
            args.sample_seconds,
            args.point_timeout,
            seed=0,
        )
        report["warmup_seconds"] = warm["wall_seconds"]
        report["warmup_summary"] = warm["summary"]
        for prompt_length, concurrencies in parse_plan(args.plan):
            cap = None
            for per_gpu in concurrencies:
                if per_gpu == 0:
                    if cap is None:
                        continue
                    per_gpu = cap
                if cap is not None and per_gpu > cap:
                    report["errors"].append(
                        {
                            "prompt_length": prompt_length,
                            "per_gpu": per_gpu,
                            "skipped": f"above the per-GPU limit {cap}",
                        }
                    )
                    continue
                if any(p["prompt_length"] == prompt_length and p["per_gpu"] == per_gpu for p in report["points"]):
                    continue
                try:
                    point = await run_point(
                        client,
                        sampling,
                        gpus,
                        prompt_length,
                        per_gpu,
                        args.decode_length,
                        args.sample_seconds,
                        args.point_timeout,
                        seed=prompt_length * 1000 + per_gpu,
                    )
                except Exception:
                    report["errors"].append(
                        {"prompt_length": prompt_length, "per_gpu": per_gpu, "error": traceback.format_exc()}
                    )
                    save_report(args.output, report)
                    print("MOCHI_AB_ERROR " + json.dumps(report["errors"][-1])[:3000], flush=True)
                    continue
                report["points"].append(point)
                save_report(args.output, report)
                print(
                    "MOCHI_AB_POINT "
                    + json.dumps({"L": prompt_length, "c": per_gpu, "wall": point["wall_seconds"], **point["summary"]}),
                    flush=True,
                )
                if per_gpu == 1 and cap is None:
                    usage = point["summary"]["kv_usage_max"]
                    # One sequence per GPU at full length uses `usage` of the cache; keep 10% headroom.
                    cap = min(args.max_num_seqs, int(0.9 / usage)) if usage > 0 else 8
                    report.setdefault("per_gpu_limits", {})[str(prompt_length)] = {
                        "limit": cap,
                        "kv_usage_one_per_gpu": usage,
                    }
                    save_report(args.output, report)
        if args.profile_point:
            length, per_gpu = (int(x) for x in args.profile_point.split(":"))
            holder = {}

            def dump_later():
                def target():
                    time.sleep(args.profile_delay)
                    holder["py_spy"] = py_spy_dumps()

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                holder["thread"] = thread

            point = await run_point(
                client,
                sampling,
                gpus,
                length,
                per_gpu,
                args.decode_length,
                args.sample_seconds,
                args.point_timeout,
                seed=7,
                on_decode=dump_later,
            )
            if "thread" in holder:
                holder["thread"].join(timeout=180)
            report["profile_point"] = {"summary": point["summary"], "wall_seconds": point["wall_seconds"]}
            report["py_spy"] = holder.get("py_spy", {"error": "decode phase not reached"})
            print("MOCHI_AB_PYSPY " + json.dumps(report["py_spy"])[:12000], flush=True)

    asyncio.run(main())
    report["nccl_head_node"] = nccl_summary()
    print("MOCHI_AB_NCCL " + json.dumps(report["nccl_head_node"])[:6000], flush=True)
    try:
        put(args.output, "nccl-head-node.tar.gz", nccl_archive())
    except Exception:
        report["nccl_archive_error"] = traceback.format_exc()[-2000:]
    report["status"] = "passed" if report["points"] else "failed"
    report["finished_at"] = time.time()
    save_report(args.output, report)
    print("MOCHI_AB_STATUS " + report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-metadata")
    stage.add_argument("--source", required=True)
    stage.add_argument("--dest", default="/tmp/mochi-model")
    runner = sub.add_parser("run")
    runner.add_argument("--output", required=True)
    runner.add_argument("--name", required=True)
    runner.add_argument("--source", required=True)
    runner.add_argument("--metadata-dir", default="/tmp/mochi-model")
    runner.add_argument("--weights", choices=("real", "dummy"), default="real")
    runner.add_argument("--runai-distributed", type=int, default=1)
    runner.add_argument("--runai-memory-limit-gib", type=float, default=16)
    runner.add_argument("--runai-request-timeout-ms", type=int, default=60000)
    runner.add_argument("--serving-nodes", type=int, required=True)
    runner.add_argument("--batch-invariant", type=int, required=True)
    runner.add_argument("--max-num-batched-tokens", type=int, required=True)
    runner.add_argument("--max-num-seqs", type=int, required=True)
    runner.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    runner.add_argument("--max-model-len", type=int, default=9216)
    runner.add_argument("--route-capture", action="store_true")
    # prompt length:per-GPU concurrencies; 0 is the largest that fits, measured from the one-per-GPU point.
    runner.add_argument("--plan", default="8192:1,8,0;512:1,8,32,0")
    runner.add_argument("--decode-length", type=int, default=512)
    runner.add_argument("--profile-point", default="8192:4")
    runner.add_argument("--profile-delay", type=float, default=20.0)
    runner.add_argument("--sample-seconds", type=float, default=2.0)
    runner.add_argument("--point-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "stage-metadata":
        stage_metadata(args)
        return
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
