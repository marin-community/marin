# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Snowball vLLM engine profiles on one 8xH100 node.

Holds the quick loop's engine geometry fixed (DP8/EP8, Triton MoE, 24,576-token window,
8,192-token prefill chunks, 0.75 GPU memory) and varies the determinism-related settings:
batch-invariant kernels, the NCCL channel pins that batch invariance applies, the attention
backend, and prefix caching. Each variant gets a concurrency sweep with ``vllm bench serve``
and a parity probe that checks whether greedy outputs change with batch composition.

Runs inside an Iris task after SkyRL's ``bootstrap_runtime.sh`` installed the pinned vLLM.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

SERVED_NAME = "snowball"
PORT = 8011
HEALTH_TIMEOUT = 25 * 60
NCCL_GUARD_ENV = "MARIN_BI_KEEP_NCCL_DEFAULTS"
PROBE_MAX_TOKENS = 128
FILLER_COUNT = 120
FILLER_MAX_TOKENS = 256

# The NCCL block the pinned vLLM's override_envs_for_invariance() applies, made conditional.
NCCL_BLOCK = """    # NCCL determinism settings
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"
"""
NCCL_BLOCK_GUARDED = f"""    # NCCL determinism settings (Marin benchmark guard: {NCCL_GUARD_ENV}=1 keeps NCCL defaults)
    if os.environ.get("{NCCL_GUARD_ENV}") != "1":
        os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
        os.environ["NCCL_COLLNET_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["NCCL_P2P_NET_DISABLE"] = "1"
        os.environ["NCCL_MIN_NCHANNELS"] = "1"
        os.environ["NCCL_MAX_NCHANNELS"] = "1"
        os.environ["NCCL_PROTO"] = "Simple"
        os.environ["NCCL_ALGO"] = "allreduce:tree"
        os.environ["NCCL_NTHREADS"] = "1"
        os.environ["NCCL_SOCKET_NTHREADS"] = "1"
"""


@dataclass(frozen=True)
class Variant:
    name: str
    batch_invariant: bool
    attention_backend: str
    prefix_caching: bool
    keep_nccl_defaults: bool


VARIANTS = {
    # The quick loop's arm D engine profile.
    "armD": Variant("armD", True, "TRITON_ATTN", False, False),
    # Arm D with the engines' NCCL collectives left at NCCL defaults.
    "armD_nccl": Variant("armD_nccl", True, "TRITON_ATTN", False, True),
    # Arm D with prefix caching re-enabled.
    "armD_prefix": Variant("armD_prefix", True, "TRITON_ATTN", True, False),
    # The main recipe's profile, as a control.
    "main": Variant("main", False, "FLASH_ATTN", True, False),
}

PROBE_PROMPTS = [
    "Explain why the harmonic series diverges, step by step.",
    "Write a Python function that merges two sorted lists and prove it runs in linear time.",
    "A train leaves at 3:40pm traveling 72 km/h. How far has it gone by 6:10pm? Show the work.",
    "Summarize the causes of the 1929 stock market crash in five sentences.",
    "What is the derivative of x^x? Derive it carefully.",
    "Describe how a hash map handles collisions and compare two strategies.",
    "Translate to French and explain each grammatical choice: 'The report was due yesterday.'",
    "List three sorting algorithms, their worst-case complexities, and when to prefer each.",
]


def fmt(value: float | None, width: int, digits: int, scale: float = 1.0) -> str:
    return f"{'-':>{width}}" if value is None else f"{value / scale:{width}.{digits}f}"


def log(message: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def stage_model(model_uri: str, stage_dir: Path) -> Path | None:
    """Copy the HF export to node-local disk once; return None when staging is unavailable."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    fsutil = shutil.which("fsutil")
    cmd = [fsutil] if fsutil else ["uv", "run", "fsutil"]
    started = time.time()
    try:
        run([*cmd, "cp", "-r", "--no-clobber", model_uri.rstrip("/") + "/", str(stage_dir) + "/"])
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        log(f"staging failed ({error}); serving from {model_uri} with the streaming loader")
        return None
    # fsutil may place the prefix under a nested directory; locate the export by its config.
    configs = sorted(stage_dir.rglob("config.json"))
    model_dir = configs[0].parent if configs else stage_dir
    index = model_dir / "model.safetensors.index.json"
    if not configs or not index.exists():
        log(f"staging incomplete under {model_dir}: configs={configs}, index present={index.exists()}")
        return None
    expected = set(json.loads(index.read_text())["weight_map"].values())
    missing = sorted(name for name in expected if not (model_dir / name).exists())
    if missing:
        log(f"staging incomplete under {model_dir}: {len(missing)} of {len(expected)} shards missing: {missing[:3]}")
        return None
    staged_bytes = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file())
    log(f"staged {staged_bytes / 1e9:.1f} GB to {model_dir} in {time.time() - started:.0f}s")
    return model_dir


def guard_nccl_pins(venv: Path) -> Path:
    """Make the pinned vLLM's NCCL pins conditional on an environment variable."""
    matches = list(venv.glob("lib/python3.*/site-packages/vllm/model_executor/determinism/batch_invariant.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one batch_invariant.py under {venv}, found {matches}")
    path = matches[0]
    source = path.read_text()
    if NCCL_BLOCK_GUARDED in source:
        return path
    if source.count(NCCL_BLOCK) != 1:
        raise RuntimeError(f"{path} does not contain the expected NCCL block exactly once")
    path.write_text(source.replace(NCCL_BLOCK, NCCL_BLOCK_GUARDED))
    log(f"guarded NCCL pins in {path}")
    return path


def server_command(venv: Path, model: str, variant: Variant, remote: bool) -> list[str]:
    cmd = [
        str(venv / "bin" / "vllm"),
        "serve",
        model,
        "--trust-remote-code",
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--moe-backend",
        "triton",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "24576",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1024",
        "--gpu-memory-utilization",
        "0.75",
        "--attention-backend",
        variant.attention_backend,
        "--seed",
        "0",
    ]
    if not variant.prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if remote:
        cmd += ["--load-format", "runai_streamer", "--model-loader-extra-config", '{"distributed":true}']
    return cmd


def server_environment(variant: Variant) -> dict[str, str]:
    env = dict(os.environ)
    env["VLLM_BATCH_INVARIANT"] = "1" if variant.batch_invariant else "0"
    env[NCCL_GUARD_ENV] = "1" if variant.keep_nccl_defaults else "0"
    env["VLLM_USE_V1"] = "1"
    return env


def wait_for_health(timeout: float) -> float:
    started = time.time()
    while time.time() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                if response.status == 200:
                    return time.time() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not healthy after {timeout:.0f}s")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=60)
    deadline = time.time() + 120
    while time.time() < deadline:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.split()
        if used and all(int(value) < 2048 for value in used):
            return
        time.sleep(5)
    log("warning: GPU memory not released within 120s after server stop")


def bench(venv: Path, tokenizer: str, concurrency: int, input_len: int, output_len: int, out_dir: Path) -> dict:
    num_prompts = max(32, 2 * concurrency)
    filename = f"bench_c{concurrency}.json"
    run(
        [
            str(venv / "bin" / "vllm"),
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--model",
            SERVED_NAME,
            "--tokenizer",
            tokenizer,
            "--trust-remote-code",
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(concurrency),
            "--ignore-eos",
            "--seed",
            "0",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,90,99",
            "--save-result",
            "--result-dir",
            str(out_dir),
            "--result-filename",
            filename,
            "--disable-tqdm",
        ]
    )
    result = json.loads((out_dir / filename).read_text())
    keys = (
        "duration",
        "completed",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "p90_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "mean_e2el_ms",
        "p90_e2el_ms",
    )
    summary = {key: result.get(key) for key in keys}
    summary["concurrency"] = concurrency
    summary["num_prompts"] = num_prompts
    return summary


def completion(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": SERVED_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1234,
            "logprobs": 1,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        choice = json.load(response)["choices"][0]
    return {"tokens": choice["logprobs"]["tokens"], "logprobs": choice["logprobs"]["token_logprobs"]}


def filler_prompt(index: int) -> str:
    sentences = [
        f"Observation {index * 7 + k}: the sensor reported {((index + 1) * (k + 3)) % 97} units at hour {k}."
        for k in range(60)
    ]
    return " ".join(sentences) + " Write a detailed report analyzing these observations."


def parity_probe() -> dict:
    """Compare greedy outputs alone, under concurrent load, and on a repeated solo pass."""
    solo = [completion(prompt, PROBE_MAX_TOKENS) for prompt in PROBE_PROMPTS]
    with ThreadPoolExecutor(max_workers=FILLER_COUNT + len(PROBE_PROMPTS)) as pool:
        fillers = [pool.submit(completion, filler_prompt(index), FILLER_MAX_TOKENS) for index in range(FILLER_COUNT)]
        time.sleep(3)
        probes = [pool.submit(completion, prompt, PROBE_MAX_TOKENS) for prompt in PROBE_PROMPTS]
        loaded = [future.result() for future in probes]
        for future in fillers:
            future.result()
    repeat = [completion(prompt, PROBE_MAX_TOKENS) for prompt in PROBE_PROMPTS]

    def compare(reference: list[dict], other: list[dict]) -> dict:
        token_matches = sum(a["tokens"] == b["tokens"] for a, b in zip(reference, other, strict=True))
        max_diff = 0.0
        for a, b in zip(reference, other, strict=True):
            for x, y in zip(a["logprobs"], b["logprobs"], strict=False):
                if x is not None and y is not None:
                    max_diff = max(max_diff, abs(x - y))
        return {"token_matches": token_matches, "prompts": len(reference), "max_abs_logprob_diff": max_diff}

    return {"solo_vs_loaded": compare(solo, loaded), "solo_vs_repeat": compare(solo, repeat)}


def tail(path: Path, lines: int) -> str:
    text = path.read_text(errors="replace").splitlines()
    return "\n".join(text[-lines:])


def run_variant(
    variant: Variant,
    venv: Path,
    model: str,
    remote: bool,
    tokenizer: str,
    concurrencies: list[int],
    input_len: int,
    output_len: int,
    out_root: Path,
) -> dict:
    out_dir = out_root / variant.name
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"
    cmd = server_command(venv, model, variant, remote)
    log(f"=== variant {variant.name}: {asdict(variant)}")
    with server_log.open("w") as handle:
        process = subprocess.Popen(cmd, env=server_environment(variant), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            startup = wait_for_health(HEALTH_TIMEOUT)
            log(f"server healthy after {startup:.0f}s")
            sweep = [bench(venv, tokenizer, c, input_len, output_len, out_dir) for c in concurrencies]
            for row in sweep:
                log(
                    f"c={row['concurrency']:4d} out_tok/s={fmt(row['output_throughput'], 8, 1)} "
                    f"tpot_ms mean={fmt(row['mean_tpot_ms'], 7, 2)} p90={fmt(row['p90_tpot_ms'], 7, 2)} "
                    f"ttft_ms mean={fmt(row['mean_ttft_ms'], 8, 1)} e2e_s p90={fmt(row['p90_e2el_ms'], 6, 1, 1000)}"
                )
            parity = parity_probe()
            log(f"parity: {json.dumps(parity)}")
        finally:
            stop_server(process)
    engine_lines = [
        line
        for line in server_log.read_text(errors="replace").splitlines()
        if "init engine" in line or "KV cache size" in line or "Model loading took" in line
    ]
    result = {
        "variant": asdict(variant),
        "startup_seconds": startup,
        "sweep": sweep,
        "parity": parity,
        "engine_lines": engine_lines[:12],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "server.tail.log").write_text(tail(server_log, 400))
    server_log.unlink()
    return result


def print_summary(results: list[dict]) -> None:
    print("\n=== summary (output tokens/s across 8 engines; TPOT = ms per output token per stream) ===")
    header = f"{'variant':12} {'conc':>5} {'out_tok/s':>10} {'tpot_mean':>10} {'tpot_p90':>9} {'ttft_mean_ms':>12} {'e2e_p90_s':>9}"
    print(header)
    for result in results:
        for row in result["sweep"]:
            print(
                f"{result['variant']['name']:12} {row['concurrency']:5d} {fmt(row['output_throughput'], 10, 1)} "
                f"{fmt(row['mean_tpot_ms'], 10, 2)} {fmt(row['p90_tpot_ms'], 9, 2)} {fmt(row['mean_ttft_ms'], 12, 1)} "
                f"{fmt(row['p90_e2el_ms'], 9, 1, 1000)}"
            )
        parity = result["parity"]
        print(
            f"{result['variant']['name']:12} parity solo_vs_loaded={parity['solo_vs_loaded']['token_matches']}/"
            f"{parity['solo_vs_loaded']['prompts']} maxdlp={parity['solo_vs_loaded']['max_abs_logprob_diff']:.3e} "
            f"solo_vs_repeat={parity['solo_vs_repeat']['token_matches']}/{parity['solo_vs_repeat']['prompts']} "
            f"maxdlp={parity['solo_vs_repeat']['max_abs_logprob_diff']:.3e} startup={result['startup_seconds']:.0f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--tokenizer-fallback", required=True, help="Tokenizer when staging fails, e.g. an HF repo id.")
    parser.add_argument("--venv", required=True, type=Path, help="SkyRL runtime venv with the pinned vLLM.")
    parser.add_argument("--stage-dir", default="/tmp/snowball-sft", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variants", default="armD,armD_nccl,armD_prefix,main")
    parser.add_argument("--concurrency", default="8,64,256,512,768")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--plan", action="store_true", help="Print the server commands and exit.")
    args = parser.parse_args()

    variants = [VARIANTS[name] for name in args.variants.split(",")]
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    if args.plan:
        for variant in variants:
            print(" ".join(server_command(args.venv, args.model_uri, variant, remote=True)))
            print({k: v for k, v in server_environment(variant).items() if k.startswith(("VLLM_", "MARIN_"))})
        return

    args.out.mkdir(parents=True, exist_ok=True)
    run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    staged = stage_model(args.model_uri, args.stage_dir)
    model = str(staged) if staged else args.model_uri
    tokenizer = str(staged) if staged else args.tokenizer_fallback
    if any(variant.keep_nccl_defaults for variant in variants):
        guard_nccl_pins(args.venv)

    results = []
    for variant in variants:
        try:
            results.append(
                run_variant(variant, args.venv, model, staged is None, tokenizer, concurrencies, args.input_len, args.output_len, args.out)
            )
        except Exception as error:  # noqa: BLE001 - keep benchmarking the remaining variants
            log(f"variant {variant.name} failed: {error!r}")
            results.append({"variant": asdict(variant), "error": repr(error), "sweep": [], "parity": None, "startup_seconds": 0})
        (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    print_summary([result for result in results if result.get("sweep")])
    failures = [result for result in results if result.get("error")]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
