# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness probes and the throughput sweep run against one live vLLM server.

Every request sends token ids, so every arm scores identical histories. Requests that must land on
a fixed data-parallel engine carry vLLM's ``X-data-parallel-rank`` header.
"""

import asyncio
import itertools
import json
import logging
import time
from dataclasses import dataclass, field

import aiohttp
import psutil

logger = logging.getLogger(__name__)

_RANK_HEADER = "X-data-parallel-rank"
_TOP_K = 5
_TIMEOUT = aiohttp.ClientTimeout(total=3 * 3600)
# Attempts per load request when the connection drops before the stream completes.
_STREAM_ATTEMPTS = 3


@dataclass(frozen=True)
class LoadConfig:
    concurrencies: tuple[int, ...]
    max_tokens: int
    waves: int
    temperature: float = 1.0


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    load: LoadConfig
    long_lengths: tuple[int, ...]
    greedy_tokens: int = 256
    long_greedy_tokens: int = 64
    invariance_cases: int = 8
    filler_per_rank: int = 8
    filler_tokens: int = 2048
    data_parallel_size: int = 16
    profile_per_engine: tuple[int, ...] = (1, 32, 128)


def _top(entry: dict | None) -> list[list]:
    """vLLM logprob dict -> [[token_id, logprob], ...] sorted by logprob."""
    if entry is None:
        return []
    items = []
    for token, value in entry.items():
        token_id = int(token.split(":")[-1]) if isinstance(token, str) else int(token)
        logprob = value["logprob"] if isinstance(value, dict) else value
        items.append([token_id, float(logprob)])
    items.sort(key=lambda item: -item[1])
    return items


@dataclass
class Client:
    base_url: str
    model: str
    session: aiohttp.ClientSession = field(init=False)

    async def __aenter__(self) -> "Client":
        # A fresh connection per request: reusing a pooled connection that the server's keep-alive
        # timer is closing raised ServerDisconnectedError mid-sweep on long (4K-token) streams.
        self.session = aiohttp.ClientSession(timeout=_TIMEOUT, connector=aiohttp.TCPConnector(limit=0, force_close=True))
        return self

    async def __aexit__(self, *exc) -> None:
        await self.session.close()

    async def completion(self, body: dict, rank: int | None = None) -> dict:
        headers = {_RANK_HEADER: str(rank)} if rank is not None else {}
        async with self.session.post(
            f"{self.base_url}/completions", json={"model": self.model, **body}, headers=headers
        ) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"completion failed {response.status}: {text[:2000]}")
            return json.loads(text)

    async def score(self, tokens: list[int], rank: int | None = None) -> dict:
        """Prompt log-probs at every position plus the next-token distribution."""
        result = await self.completion(
            {
                "prompt": tokens,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": _TOP_K,
                "logprobs": _TOP_K,
                "return_tokens_as_token_ids": True,
            },
            rank,
        )
        choice = result["choices"][0]
        positions = [_top(entry) for entry in choice["prompt_logprobs"][1:]]
        positions.append(_top(choice["logprobs"]["top_logprobs"][0]))
        return {"positions": positions}

    async def greedy(self, tokens: list[int], max_tokens: int, rank: int | None = None) -> dict:
        result = await self.completion(
            {
                "prompt": tokens,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "logprobs": 2,
                "return_tokens_as_token_ids": True,
            },
            rank,
        )
        logprobs = result["choices"][0]["logprobs"]
        generated = [int(token.split(":")[-1]) for token in logprobs["tokens"]]
        return {"tokens": generated, "top2": [_top(entry) for entry in logprobs["top_logprobs"]]}

    async def metrics_text(self) -> str:
        root = self.base_url.removesuffix("/v1")
        async with self.session.get(f"{root}/metrics") as response:
            return await response.text()


def _running_by_engine(metrics: str) -> dict[str, float]:
    running = {}
    for line in metrics.splitlines():
        if line.startswith("vllm:num_requests_running{"):
            labels, value = line.rsplit(" ", 1)
            engine = labels.split('engine="', 1)[1].split('"', 1)[0]
            running[engine] = float(value)
    return running


async def _filler(client: Client, prompts: list[list[int]], ranks: list[int], per_rank: int, tokens: int) -> list:
    jobs = []
    for rank in ranks:
        for index in range(per_rank):
            body = {
                "prompt": prompts[(rank * per_rank + index) % len(prompts)],
                "max_tokens": tokens,
                "temperature": 1.0,
                "seed": rank * 1000 + index,
                "ignore_eos": True,
            }
            jobs.append(asyncio.create_task(client.completion(body, rank)))
    return jobs


async def _wait_engines_busy(client: Client, ranks: list[int], timeout: float = 600.0) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    while True:
        running = _running_by_engine(await client.metrics_text())
        if all(running.get(str(rank), 0) > 0 for rank in ranks):
            return running
        if time.monotonic() > deadline:
            raise RuntimeError(f"filler did not occupy engines {ranks}: {running}")
        await asyncio.sleep(1.0)


async def gate_probes(client: Client, cases: dict, config: ProbeConfig) -> dict:
    """G1/G2/G4 histories; G2 peer and same-rank perturbation on engine 0."""
    gate = cases["gate"]
    results: dict = {"score": {}, "greedy": {}, "invariance": {}}

    for case in gate:
        results["score"][case["case_id"]] = await client.score(case["tokens"], rank=0)
    logger.info("scored %d gate histories alone on engine 0", len(gate))
    for case in gate:
        results["greedy"][case["case_id"]] = await client.greedy(case["tokens"], config.greedy_tokens, rank=0)
    logger.info("greedy decoded %d prompts alone on engine 0", len(gate))

    probe_cases = gate[: config.invariance_cases // 2] + gate[-config.invariance_cases // 2 :]
    filler_prompts = [case["tokens"] for case in cases["load"][:64]]

    async def perturbed(label: str, ranks: list[int]) -> None:
        jobs = await _filler(client, filler_prompts, ranks, config.filler_per_rank, config.filler_tokens)
        busy = await _wait_engines_busy(client, ranks)
        scored = {}
        decoded = {}
        for case in probe_cases:
            scored[case["case_id"]] = await client.score(case["tokens"], rank=0)
            decoded[case["case_id"]] = await client.greedy(case["tokens"], 64, rank=0)
        after = _running_by_engine(await client.metrics_text())
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        results["invariance"][label] = {
            "score": scored,
            "greedy": decoded,
            "busy_before": busy,
            "busy_after": after,
        }
        logger.info("invariance %s done; busy=%s", label, busy)

    peers = list(range(1, config.data_parallel_size))
    await perturbed("peer_load", peers)
    await perturbed("same_rank_load", [0])
    # Reference with no load, recorded after the perturbations to expose drift over time.
    results["invariance"]["alone_after"] = {
        "score": {case["case_id"]: await client.score(case["tokens"], rank=0) for case in probe_cases},
        "greedy": {case["case_id"]: await client.greedy(case["tokens"], 64, rank=0) for case in probe_cases},
    }
    return results


async def _decode_and_rescore(client: Client, tokens: list[int], max_tokens: int, rank: int) -> dict:
    """Greedy-decode, then score prompt + output by prefill; return both chosen-token log-probs."""
    decoded = await client.greedy(tokens, max_tokens, rank=rank)
    generated = decoded["tokens"]
    decode_lp = [
        dict((int(t), lp) for t, lp in top)[token] for top, token in zip(decoded["top2"], generated, strict=True)
    ]
    scored = await client.score(tokens + generated, rank=rank)
    # score() positions predict tokens[1:] then the next token; generated token i sits at len(tokens) + i.
    prefill_lp = []
    for i, token in enumerate(generated):
        top = dict((int(t), lp) for t, lp in scored["positions"][len(tokens) - 1 + i])
        prefill_lp.append(top.get(token))
    return {"tokens": generated, "decode_lp": decode_lp, "prefill_lp": prefill_lp}


async def decode_consistency(client: Client, cases: dict, config: ProbeConfig) -> dict:
    """Decode-path check: decode-time log-probs against prefill scoring of the same tokens.

    Prefill scoring is gated against JAX (gate b), but decode can take a different execution path
    (e.g. full CUDA graphs). Greedy continuations are decoded alone on engine 0 and then 32 at a
    time on every engine (a larger decode graph), and each continuation is re-scored by prefill
    on the same engine. A decode-path bug shows up as decode/prefill disagreement far beyond the
    BF16 noise floor that the control shows.
    """
    gate = cases["gate"]
    alone = {}
    for case in gate:
        alone[case["case_id"]] = await _decode_and_rescore(client, case["tokens"], config.greedy_tokens, 0)
    logger.info("decode consistency: %d cases alone on engine 0", len(alone))
    batched_cases = [(rank, index, case) for rank in range(config.data_parallel_size) for index, case in enumerate(gate)]
    results = await asyncio.gather(
        *(_decode_and_rescore(client, case["tokens"], config.greedy_tokens, rank) for rank, _, case in batched_cases)
    )
    batched = {
        f"{rank}:{case['case_id']}": result for (rank, _, case), result in zip(batched_cases, results, strict=True)
    }
    logger.info("decode consistency: %d cases batched on all engines", len(batched))
    return {"alone": alone, "batched": batched}


async def long_probes(client: Client, cases: dict, config: ProbeConfig) -> dict:
    results = {}
    for case in cases["long"]:
        length = len(case["tokens"])
        if length not in config.long_lengths:
            continue
        start = time.monotonic()
        scored = await client.score(case["tokens"], rank=0)
        score_seconds = time.monotonic() - start
        start = time.monotonic()
        decoded = await client.greedy(case["tokens"], config.long_greedy_tokens, rank=0)
        results[case["case_id"]] = {
            "score": scored,
            "greedy": decoded,
            "score_seconds": score_seconds,
            "greedy_seconds": time.monotonic() - start,
        }
        logger.info("long case %s scored in %.1fs", case["case_id"], score_seconds)
    return results


async def _stream_request(client: Client, body: dict, record: dict, rank: int | None = None) -> None:
    record["submit"] = time.monotonic()
    stamps: list = []
    # Callers that cancel the stream still see the stamps collected so far.
    record["stamps"] = stamps
    async with client.session.post(
        f"{client.base_url}/completions",
        json={
            "model": client.model,
            **body,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        },
        headers={_RANK_HEADER: str(rank)} if rank is not None else {},
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"stream failed {response.status}: {(await response.text())[:2000]}")
        completed = 0
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens", 0) > completed:
                now = time.monotonic()
                stamps.append((now, usage["completion_tokens"] - completed))
                completed = usage["completion_tokens"]
    record["end"] = time.monotonic()
    record["tokens"] = completed
    record["stamps"] = stamps


async def load_sweep(client: Client, cases: dict, config: ProbeConfig, on_progress=None) -> dict:
    """Closed-loop RL-rollout-like load: short chat prompts, fixed-length sampled outputs."""
    prompts = [case["tokens"] for case in cases["load"]]
    results = {}
    # Batch-1 latency point.
    single = []
    for index in range(8):
        record: dict = {}
        await _stream_request(
            client,
            {"prompt": prompts[index], "max_tokens": 256, "temperature": 1.0, "seed": index, "ignore_eos": True},
            record,
        )
        single.append(record)
    results["c1"] = _summarize(single, concurrency=1, window=None)
    if on_progress is not None:
        on_progress(results)
    logger.info("C1: %s", {k: v for k, v in results["c1"].items() if k != "requests"})

    for concurrency in config.load.concurrencies:
        total = concurrency * config.load.waves
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(total):
            queue.put_nowait(index)
        records: list[dict] = [dict() for _ in range(total)]
        last_submit = [0.0]

        async def worker(queue=queue, records=records, last_submit=last_submit) -> None:
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                last_submit[0] = time.monotonic()
                body = {
                    "prompt": prompts[index % len(prompts)],
                    "max_tokens": config.load.max_tokens,
                    "temperature": config.load.temperature,
                    "seed": index,
                    "ignore_eos": True,
                }
                for attempt in range(_STREAM_ATTEMPTS):
                    try:
                        await _stream_request(client, body, records[index])
                        break
                    except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                        records[index] = {"retries": attempt + 1}
                        if attempt + 1 == _STREAM_ATTEMPTS:
                            raise

        start = time.monotonic()
        cpu: dict = {}
        sampler = asyncio.create_task(_sample_cpu(cpu))
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        sampler.cancel()
        firsts = sorted(r["stamps"][0][0] for r in records if r.get("stamps"))
        window = (firsts[min(concurrency, len(firsts)) - 1], last_submit[0])
        results[f"c{concurrency}"] = _summarize(records, concurrency=concurrency, window=window)
        results[f"c{concurrency}"]["makespan"] = time.monotonic() - start
        results[f"c{concurrency}"]["cpu"] = _cpu_summary(cpu)
        if on_progress is not None:
            on_progress(results)
        logger.info("C%d: %s", concurrency, {k: v for k, v in results[f"c{concurrency}"].items() if k != "requests"})
    return results


async def _post_root(client: Client, path: str) -> float:
    """POST an admin endpoint at the server root; return the call's wall time."""
    started = time.monotonic()
    async with client.session.post(f"{client.base_url.removesuffix('/v1')}/{path}") as response:
        if response.status != 200:
            raise RuntimeError(f"{path} failed {response.status}: {(await response.text())[:2000]}")
    return time.monotonic() - started


def _gaps_in(records: list[dict], start: float, end: float) -> list[float]:
    """Per-request inter-token gaps (seconds per token) whose both ends fall inside [start, end]."""
    gaps = []
    for record in records:
        stamps = record.get("stamps") or []
        for (t0, _), (t1, n1) in itertools.pairwise(stamps):
            if start <= t0 and t1 <= end:
                gaps.append((t1 - t0) / n1)
    return gaps


def _gap_summary(gaps: list[float]) -> dict:
    return {
        "n": len(gaps),
        "itl_p50": _quantile(gaps, 0.5),
        "itl_p10": _quantile(gaps, 0.1),
        "itl_p90": _quantile(gaps, 0.9),
        "itl_mean": sum(gaps) / len(gaps) if gaps else None,
    }


async def profile_points(client: Client, cases: dict, config: ProbeConfig) -> dict:
    """Pure-decode steps at a fixed number of sequences per engine, with a torch-profiler window.

    For each point, every engine gets exactly ``per_engine`` streaming requests (pinned by the
    rank header, all admitted at once, so the window has no prefill). Inter-token latency is
    measured in three windows on the same requests: before, during and after /start_profile ..
    /stop_profile, which shows how much the profiler perturbs the step. The server must run with
    --profiler-config (torch, max_iterations bounds the traced steps).
    """
    prompts = [case["tokens"] for case in cases["load"]]
    ranks = list(range(config.data_parallel_size))
    results = {}
    for per_engine in config.profile_per_engine:
        records: list[dict] = []
        tasks = []
        for rank in ranks:
            for index in range(per_engine):
                record: dict = {"rank": rank}
                records.append(record)
                body = {
                    "prompt": prompts[(rank * per_engine + index) % len(prompts)],
                    "max_tokens": config.load.max_tokens,
                    "temperature": config.load.temperature,
                    "seed": rank * 1000 + index,
                    "ignore_eos": True,
                }
                tasks.append(asyncio.create_task(_stream_request(client, body, record, rank)))
        running = await _wait_engines_busy(client, ranks)
        deadline = time.monotonic() + 120.0
        while any(running.get(str(rank), 0) < per_engine for rank in ranks):
            if time.monotonic() > deadline:
                raise RuntimeError(f"engines did not reach {per_engine} running: {running}")
            await asyncio.sleep(1.0)
            running = _running_by_engine(await client.metrics_text())
        await asyncio.sleep(10.0)
        before_start = time.monotonic()
        await asyncio.sleep(10.0)
        profile_start = time.monotonic()
        start_call = await _post_root(client, "start_profile")
        await asyncio.sleep(8.0)
        profile_end = time.monotonic()
        stop_call = await _post_root(client, "stop_profile")
        after_start = time.monotonic()
        await asyncio.sleep(10.0)
        after_end = time.monotonic()
        running_end = _running_by_engine(await client.metrics_text())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        finished = sum(1 for record in records if "end" in record)
        results[f"e{per_engine}"] = {
            "per_engine": per_engine,
            "requests": len(records),
            "finished_before_cancel": finished,
            "running_at_start": running,
            "running_at_end": running_end,
            "before": _gap_summary(_gaps_in(records, before_start, profile_start)),
            "profiled": _gap_summary(_gaps_in(records, profile_start, profile_end)),
            "after": _gap_summary(_gaps_in(records, after_start, after_end)),
            "start_profile_seconds": start_call,
            "stop_profile_seconds": stop_call,
            "windows": [before_start, profile_start, profile_end, after_start, after_end],
        }
        logger.info(
            "profile e%d: %s", per_engine, {k: results[f"e{per_engine}"][k] for k in ("before", "profiled", "after")}
        )
        # Let the aborted requests drain before the next point.
        await asyncio.sleep(10.0)
    return results


def _api_server_processes() -> list[psutil.Process]:
    """vLLM API server processes on this host (vLLM sets their titles to APIServer / ApiServer_N)."""
    found = []
    for process in psutil.process_iter(["name", "cmdline"]):
        title = " ".join([process.info.get("name") or "", *(process.info.get("cmdline") or [])[:1]])
        if "APIServer" in title or "ApiServer" in title:
            found.append(process)
    return found


async def _sample_cpu(samples: dict, period: float = 5.0) -> None:
    """Record CPU percent (100 = one core) for this client process and the API server processes."""
    me = psutil.Process()
    me.cpu_percent()
    servers = _api_server_processes()
    for server in servers:
        server.cpu_percent()
    while True:
        await asyncio.sleep(period)
        samples.setdefault("client", []).append(me.cpu_percent())
        total = 0.0
        peak = 0.0
        for server in servers:
            try:
                value = server.cpu_percent()
            except psutil.Error:
                continue
            total += value
            peak = max(peak, value)
        samples.setdefault("api_server_total", []).append(total)
        samples.setdefault("api_server_max_process", []).append(peak)
        samples["api_server_processes"] = len(servers)


def _cpu_summary(samples: dict) -> dict:
    out = {"api_server_processes": samples.get("api_server_processes")}
    for key in ("client", "api_server_total", "api_server_max_process"):
        values = samples.get(key) or [0.0]
        out[f"{key}_cpu_mean"] = sum(values) / len(values)
        out[f"{key}_cpu_max"] = max(values)
    return out


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def _summarize(records: list[dict], *, concurrency: int, window: tuple[float, float] | None) -> dict:
    ttft = [r["stamps"][0][0] - r["submit"] for r in records if r.get("stamps")]
    itl = [(r["end"] - r["stamps"][0][0]) / (r["tokens"] - 1) for r in records if r.get("stamps") and r["tokens"] > 1]
    summary = {
        "concurrency": concurrency,
        "requests": len(records),
        "retried_requests": sum(1 for r in records if r.get("retries")),
        "output_tokens": sum(r.get("tokens", 0) for r in records),
        "ttft_p50": _quantile(ttft, 0.5),
        "ttft_p90": _quantile(ttft, 0.9),
        "ttft_mean": sum(ttft) / len(ttft) if ttft else None,
        "itl_p50": _quantile(itl, 0.5),
        "itl_p90": _quantile(itl, 0.9),
        "itl_mean": sum(itl) / len(itl) if itl else None,
    }
    if window is not None and window[1] > window[0]:
        tokens = sum(n for r in records for t, n in r.get("stamps", ()) if window[0] <= t < window[1])
        summary["window_seconds"] = window[1] - window[0]
        summary["window_output_tokens_per_second"] = tokens / (window[1] - window[0])
    else:
        start = min(r["submit"] for r in records)
        end = max(r["end"] for r in records)
        summary["window_seconds"] = end - start
        summary["window_output_tokens_per_second"] = summary["output_tokens"] / (end - start)
    return summary
