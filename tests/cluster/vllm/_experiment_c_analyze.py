#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline analysis for Experiment C output streams (plan G1 endpoints).

Reads one or more client-stream log files containing EXPERIMENT_C_JSON blocks and
computes the pre-registered endpoints: primary S (span of p(423) per condition/length),
secondary D (max pairwise |dp| over the union of top-50 token sets, with a coverage
flag), bitwise fresh-recompute determinism, per-rank golden error at full length, the
A-A'-A'' within-node band, cross-session/cross-job micro checksums, and the
combine-variant profile from the sizes logs.

    uv run python tests/cluster/vllm/_experiment_c_analyze.py <log-file> [<log-file> ...]
"""

import json
import math
import pathlib
import sys
from collections import defaultdict

BEGIN = "EXPERIMENT_C_JSON_BEGIN"
GOLDEN_PATH = pathlib.Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
SENTINEL_CASE_ID = "knowledge-longbench-02"
FULL_LENGTH = 15025


def read_blocks(paths: list[pathlib.Path]) -> list[dict]:
    """Parse emitted blocks, skipping any the log truncated (~49k chars per line).

    A dropped block is reported rather than silently ignored: a partial trace would
    otherwise look like a complete one with fewer calls.
    """
    blocks, dropped = [], 0
    for path in paths:
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if BEGIN in line and index + 1 < len(lines):
                payload = lines[index + 1]
                brace = payload.find("{")
                if brace < 0:
                    continue
                try:
                    block = json.loads(payload[brace:])
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                block["_source"] = path.name
                blocks.append(block)
    if dropped:
        print(f"WARNING: {dropped} block(s) unparseable (log truncation) — coverage is incomplete")
    return blocks


def sentinel_golden() -> dict[int, float]:
    payload = json.loads(GOLDEN_PATH.read_bytes())
    (case,) = [case for case in payload["cases"] if case["id"] == SENTINEL_CASE_ID]
    return {score["token_id"]: math.exp(score["logprob"]) for score in case["top_logprobs"]}


def p423(observation: dict) -> float | None:
    logprob = observation["logprobs"].get("423")
    return math.exp(logprob) if logprob is not None else None


def pairwise_top50_d(observations: list[dict]) -> tuple[float, bool]:
    """Max pairwise |dp| over tokens in both ranks' returned top-50; flags coverage
    when the top-50 token sets differ."""
    worst, coverage_gap = 0.0, False
    for index_a in range(len(observations)):
        for index_b in range(index_a + 1, len(observations)):
            lp_a, lp_b = observations[index_a]["logprobs"], observations[index_b]["logprobs"]
            shared = lp_a.keys() & lp_b.keys()
            if lp_a.keys() != lp_b.keys():
                coverage_gap = True
            for token in shared:
                worst = max(worst, abs(math.exp(lp_a[token]) - math.exp(lp_b[token])))
    return worst, coverage_gap


def analyze_observations(blocks: list[dict], golden: dict[int, float]) -> None:
    """Cross-rank endpoints. Only modes where every rank scores the *same* prompt are
    comparable this way: wave_realistic deliberately gives seven ranks different
    prompts, so it is reported separately by analyze_wave_realistic."""
    print("\n== observation endpoints (S = span of p(423); D = max pairwise |dp| top-50) ==")
    print(
        f"{'session':8} {'mode':22} {'len':>6} {'rnd':>3} {'n':>2}  {'S':>8}  {'D':>8} cov  "
        f"{'greedy':14} {'worst_vs_golden':>15}"
    )
    for block in blocks:
        if block.get("kind") != "observations" or block["mode"] == "wave_realistic":
            continue
        observations = block["observations"]
        values = [value for value in (p423(observation) for observation in observations) if value is not None]
        span = max(values) - min(values) if len(values) > 1 else 0.0
        d_value, coverage_gap = pairwise_top50_d(observations) if len(observations) > 1 else (0.0, False)
        greedies = sorted({observation["greedy_token_id"] for observation in observations})
        worst_golden = 0.0
        if block["length"] == FULL_LENGTH:
            for observation in observations:
                for token, golden_probability in golden.items():
                    actual_logprob = observation["logprobs"].get(str(token))
                    actual = math.exp(actual_logprob) if actual_logprob is not None else 0.0
                    worst_golden = max(worst_golden, abs(golden_probability - actual))
        print(
            f"{block['session']:8} {block['mode']:22} {block['length']:>6} {block.get('round', 0):>3} "
            f"{len(observations):>2}  {span:8.6f}  {d_value:8.6f} {'*' if coverage_gap else ' '}   "
            f"{greedies!s:14} {worst_golden:>15.6f}"
        )


def analyze_determinism(blocks: list[dict]) -> None:
    print("\n== fresh-recompute determinism (bitwise identity of repeated requests) ==")
    for block in blocks:
        if block.get("kind") != "observations" or block["mode"] not in ("fresh_determinism", "warmup"):
            continue
        signatures = [json.dumps(observation["logprobs"], sort_keys=True) for observation in block["observations"]]
        distinct = len(set(signatures))
        values = [p423(observation) for observation in block["observations"]]
        spread = max(values) - min(values) if len(values) > 1 else 0.0
        print(
            f"  {block['session']:4} len={block['length']:>6} n={len(signatures)} distinct_bit_patterns={distinct} "
            f"p423_spread={spread:.6f} {'BIT-STABLE' if distinct == 1 else 'UNSTABLE'}"
        )


def analyze_wave_realistic(blocks: list[dict]) -> None:
    """Does the sentinel's own answer move when its seven peers run different prompts?

    Only the sentinel-carrying rank is comparable; the peer ranks hold other prompts.
    The sentinel rank alternates by round, and its value is checked against the same
    rank's isolated/concurrent value in the same session.
    """
    reference: dict[tuple[str, int], float] = {}
    for block in blocks:
        if block.get("kind") != "observations" or block["mode"] != "isolated" or block["length"] != FULL_LENGTH:
            continue
        for observation in block["observations"]:
            value = p423(observation)
            if value is not None:
                reference[(block["session"], observation["rank"])] = value

    print("\n== wave-realistic: sentinel rank only, vs its own isolated value ==")
    for block in blocks:
        if block.get("kind") != "observations" or block["mode"] != "wave_realistic":
            continue
        # The sentinel rank is the one whose distribution matches the sentinel golden
        # token set; identify it as the rank whose top token is the sentinel's.
        candidates = [
            (observation["rank"], p423(observation))
            for observation in block["observations"]
            if observation["logprobs"].get("423") is not None
        ]
        for rank, value in candidates:
            baseline = reference.get((block["session"], rank))
            delta = abs(value - baseline) if baseline is not None else float("nan")
            print(
                f"  {block['session']:4} round={block.get('round', 0)} rank={rank} p423={value:.6f} "
                f"isolated={baseline if baseline is None else f'{baseline:.6f}'} |delta|={delta:.6f}"
            )


def analyze_aband(blocks: list[dict]) -> None:
    """Per-rank |dp423| across repeated sessions of the same config, per mode/length."""
    per_key: dict[tuple, dict[str, dict[int, float]]] = defaultdict(dict)
    for block in blocks:
        if block.get("kind") != "observations" or block["mode"] not in ("isolated", "c1_concurrent"):
            continue
        rank_map = {}
        for observation in block["observations"]:
            value = p423(observation)
            if value is not None:
                rank_map[observation["rank"]] = value
        per_key[(block["mode"], block["length"], block.get("round", 0))][block["session"]] = rank_map
    print("\n== launch bands: per-rank |dp423| across sessions (within-node S1/S2/S3; cross-job XJ) ==")
    for key, sessions in sorted(per_key.items()):
        if len(sessions) < 2:
            continue
        names = sorted(sessions)
        for index_a in range(len(names)):
            for index_b in range(index_a + 1, len(names)):
                map_a, map_b = sessions[names[index_a]], sessions[names[index_b]]
                shared = map_a.keys() & map_b.keys()
                if not shared:
                    continue
                deltas = [abs(map_a[rank] - map_b[rank]) for rank in sorted(shared)]
                print(
                    f"  {key[0]:14} len={key[1]:>6} rnd={key[2]} {names[index_a]}->{names[index_b]}: "
                    f"max={max(deltas):.6f} mean={sum(deltas) / len(deltas):.6f} n={len(deltas)}"
                )


def analyze_micro(blocks: list[dict]) -> None:
    print("\n== microreproducer (per-op combine deltas; checksums compare across ranks/sessions) ==")
    rows: dict[tuple, dict] = {}
    for block in blocks:
        if block.get("kind") != "probe_micro":
            continue
        data = block["data"]
        for result in data["results"]:
            key = (block["session"], data["dp_rank"], result["mode"], result["scale"])
            rows[key] = result
    by_mode: dict[tuple, list] = defaultdict(list)
    for (session, rank, mode, scale), result in sorted(rows.items()):
        by_mode[(session, mode, scale)].append((rank, result))
    for (session, mode, scale), entries in sorted(by_mode.items()):
        checksums = {result["output_checksum"] for _, result in entries}
        stable = all(result["bitwise_stable_across_iters"] for _, result in entries)
        worst_vs_ref = max(result["vs_reference_max_abs"]["rank_order"] for _, result in entries)
        worst_mismatch = max(result["vs_reference_mismatch_elements"]["rank_order"] for _, result in entries)
        order_spread = max(result["reference_order_spread_elements"] for _, result in entries)
        print(
            f"  {session:4} {mode:14} scale={scale:>5} ranks={len(entries)} distinct_checksums={len(checksums)} "
            f"iter_stable={stable} max|out-ref|={worst_vs_ref:.3e} mismatch_elems={worst_mismatch} "
            f"ref_order_spread_elems={order_spread}"
        )


def analyze_trace(blocks: list[dict]) -> None:
    """G2: locate the first boundary at which ranks disagree, and whether the router's
    selected experts differ.

    Reports, per (session, tag, call): how many distinct checksums the eight ranks
    produced at each boundary, and how many distinct top-4 expert selections the final
    token got. The first call where b1_predispatch_hidden is already multi-valued marks
    divergence entering the MoE; b4 identical with b5 multi-valued isolates the combine.
    """
    boundaries = ("b1_predispatch_hidden", "b2_gathered_hidden", "b4_precombine_partials", "b5_postcombine")
    # Captures are tagged len<L>r<serving rank>; within a capture only the serving rank
    # holds real tokens (the other seven contribute one dummy each), so the comparable
    # series is "the serving rank's own entries", collected across captures.
    own: dict[tuple, dict[int, dict[str, str]]] = defaultdict(dict)
    experts: dict[tuple, dict[int, str]] = defaultdict(dict)
    for block in blocks:
        if block.get("kind") != "trace" or "r" not in block["tag"].split("len")[-1]:
            continue
        length_text, serving_text = block["tag"].removeprefix("len").split("r")
        length, serving_rank = int(length_text), int(serving_text)
        file_rank = int(block["file"].split("rank")[1].split("_")[0])
        if file_rank != serving_rank:
            continue
        for entry in block["data"]:
            # Keep the serving rank's real rows. shape[0] equals the length only for a
            # single-chunk prefill; a multi-chunk prefill has shape[0] = the chunk size,
            # so filter on "not a lone dummy token" instead of on the length.
            if entry.get("shape", [0])[0] <= 1:
                continue
            key = (block["session"], length, entry["call"])
            if entry["tag"] == "b1_topk_ids":
                experts[key][serving_rank] = json.dumps(entry.get("last_token_expert_ids"))
            else:
                own[key].setdefault(serving_rank, {})[entry["tag"]] = entry["checksum"]

    if not own:
        return
    headers = ("b1_hidden", "b2_gathered", "b4_partials", "b5_combined", "top4_sets")
    print("\n== boundary trace: distinct values across serving ranks (1 = all ranks agree) ==")
    print(f"{'session':8} {'len':>6} {'call':>5} {'ranks':>5}  " + " ".join(f"{header:>11}" for header in headers))
    for key in sorted(own):
        per_rank = own[key]
        counts = [len({values.get(boundary) for values in per_rank.values()} - {None}) for boundary in boundaries]
        counts.append(len(set(experts.get(key, {}).values())))
        cells = " ".join(f"{count if count else '-':>11}" for count in counts)
        print(f"{key[0]:8} {key[1]:>6} {key[2]:>5} {len(per_rank):>5}  {cells}")


def analyze_env(blocks: list[dict]) -> None:
    print("\n== effective config records ==")
    for block in blocks:
        if block.get("kind") != "probe_env":
            continue
        data = block["data"]
        env = data.get("env", {})
        print(
            f"  {block['session']:4} rank={data['dp_rank']} gpu={data['gpu_uuid'][:13]} pci={data.get('pci_bus_id')} "
            f"cache={data['enable_prefix_caching']} eager={data['enforce_eager']} cg={data['cudagraph_mode']} "
            f"algo={env.get('NCCL_ALGO')} proto={env.get('NCCL_PROTO')} moe={list(data['moe_kernel_by_layer_group'])}"
        )


def analyze_sizes(blocks: list[dict]) -> None:
    print("\n== combine sizes profile (equal -> single ncclReduceScatter; uneven -> grouped ncclReduce) ==")
    for block in blocks:
        if block.get("kind") != "sizes_log":
            continue
        entries = block["data"]
        equal = sum(1 for entry in entries if entry["equal"])
        patterns: dict[str, int] = defaultdict(int)
        for entry in entries:
            patterns[str(entry["sizes"])] += 1
        top = sorted(patterns.items(), key=lambda item: -item[1])[:4]
        print(
            f"  {block['session']:4} {block['file']:18} total_calls={block['total_lines']} sampled={len(entries)} "
            f"equal={equal}/{len(entries)} top_patterns={top}"
        )


def main() -> int:
    paths = [pathlib.Path(argument) for argument in sys.argv[1:]]
    assert paths, "usage: _experiment_c_analyze.py <log-file> [...]"
    blocks = read_blocks(paths)
    print(f"parsed {len(blocks)} blocks from {len(paths)} files")
    golden = sentinel_golden()
    analyze_env(blocks)
    analyze_micro(blocks)
    analyze_determinism(blocks)
    analyze_observations(blocks, golden)
    analyze_wave_realistic(blocks)
    analyze_aband(blocks)
    analyze_trace(blocks)
    analyze_sizes(blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
