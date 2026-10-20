# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build every arm of one shootout round from one trace prefix, then verify them: the v7 build in one command.

Each arm is one build.py run under <output>/<arm>/, in order: validation, plain, RS-multi, relay, recovery. Every arm
takes the same trace prefix, pool v5, replay continuation v2 and the clean-completion rule; plain takes the first clean
pass per task, RS-multi resamples among clean passes on plain's task slots, relay holds each family to plain's B for
it, and recovery injects into plain's rows (default SWE mode, no handoff). If a relay family ends below plain's B for
it, the flagged fallback also builds plain-matched-relay/ (plain cut to relay's total in that family) and the summary
says so; which plain the relay comparison uses is a decision this tool does not make. verify_arms then checks plain,
RS-multi, relay and recovery together, and the matched plain on its own.

--relay-trace-prefix reads relay rounds from a different prefix than plain (e.g. traces-v7-v0/ for v0-student relay);
--parent names the base every arm fine-tunes and --relay-student the student the relay rounds must have served
(default: both Snowball v0.1). Plain, RS-multi, recovery and validation rows do not depend on either.

A production chain first runs pool_gate (plain shards all finished, each relay family's STOP marker present and no
round uploaded without _SUCCESS, no capability token in any finished shard) and builds nothing unless it passes.
--dry-run skips the gate and passes --dry-run to every build and to verify_arms: same pins and checks,
DRY_RUN_NOT_FOR_TRAINING markers, no _SUCCESS.

    python -m experiments.post_training.sft_shootout_rows.chain --builder-commit <sha> --output <fresh prefix> \\
        [--trace-prefix <s3 .../traces-v7/>] [--relay-trace-prefix <s3 .../traces-v7-v0/>] [--parent v0.1|v0] \\
        [--relay-student v0.1|v0] [--exclude-trials <jsonl>] [--dry-run]
"""

import argparse
import hashlib
import json
import subprocess
import sys

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import BASES, REPLAY_URI, TRACE_PREFIX, family_supervised

POOL = "s3://marin-us-east-02a/marin/experiments/sft-shootout/pool/tasks.jsonl"
POOL_MANIFEST = "s3://marin-us-east-02a/marin/experiments/sft-shootout/pool/manifest.json"
MODULE = "experiments.post_training.sft_shootout_rows"


def run(module: str, *arguments: str) -> None:
    print("CHAIN_RUN " + " ".join([module, *arguments]), flush=True)
    subprocess.run([sys.executable, "-m", f"{MODULE}.{module}", *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-prefix", default=TRACE_PREFIX)
    parser.add_argument("--relay-trace-prefix", help="relay rounds' prefix (default: --trace-prefix)")
    parser.add_argument("--parent", choices=sorted(BASES), default="v0.1")
    parser.add_argument("--relay-student", choices=sorted(BASES), help="default: --parent")
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--pool-manifest", default=POOL_MANIFEST)
    parser.add_argument("--replay", default=REPLAY_URI)
    parser.add_argument("--builder-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--exclude-trials", help="attempts every arm drops before selection (build.py exclude_trials)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.relay_trace_prefix = args.relay_trace_prefix or args.trace_prefix
    args.relay_student = args.relay_student or args.parent
    root = StoragePath(args.output)
    if root.exists():
        raise ValueError("Use a fresh output prefix")
    prefix = {arm: str(root / arm) for arm in ("validation", "plain", "rs-multi", "relay", "recovery")}
    common = [
        "--parent",
        args.parent,
        "--pool",
        args.pool,
        "--pool-manifest",
        args.pool_manifest,
        "--replay",
        args.replay,
        "--builder-commit",
        args.builder_commit,
        "--require-clean-completion",
    ]
    common += ["--tokenizer-dir", args.tokenizer_dir] if args.tokenizer_dir else []
    common += ["--exclude-trials", args.exclude_trials] if args.exclude_trials else []
    common += ["--dry-run"] if args.dry_run else []

    if not args.dry_run:
        run("pool_gate", "--trace-prefix", args.trace_prefix, "--relay-trace-prefix", args.relay_trace_prefix)

    def build(arm: str, output: str, *extra: str) -> dict:
        traces = (
            ["--trace-prefix", args.relay_trace_prefix, "--relay-student", args.relay_student]
            if arm == "relay"
            else ["--trace-prefix", args.trace_prefix]
        )
        run("build", "--arm", arm, *traces, *common, *extra, "--output", output)
        return json.loads((StoragePath(output) / "manifest.json").read_text())

    manifests = {"validation": build("validation", prefix["validation"])}
    manifests["plain"] = build("plain", prefix["plain"], "--plain-selection", "first")
    for arm in ("rs-multi", "relay", "recovery"):
        manifests[arm] = build(arm, prefix[arm], "--plain", prefix["plain"])
    short = {f: v for f, v in manifests["relay"]["relay_families"].items() if v["delta"] < 0}
    if short:
        prefix["plain-matched-relay"] = str(root / "plain-matched-relay")
        manifests["plain-matched-relay"] = build(
            "plain", prefix["plain-matched-relay"], "--plain", prefix["plain"], "--match-relay", prefix["relay"]
        )
    flag = ["--dry-run"] if args.dry_run else []
    run("verify_arms", *flag, *(prefix[arm] for arm in ("plain", "rs-multi", "relay", "recovery")))
    if short:
        run("verify_arms", *flag, prefix["plain-matched-relay"])
    summary = {
        "dry_run": args.dry_run,
        "trace_prefix": args.trace_prefix,
        "relay_trace_prefix": args.relay_trace_prefix,
        "parent": args.parent,
        "relay_student": args.relay_student,
        "builder_commit": args.builder_commit,
        "arms": {
            arm: {
                "prefix": prefix[arm],
                "manifest_sha256": hashlib.sha256((StoragePath(prefix[arm]) / "manifest.json").read_bytes()).hexdigest(),
                "trace_files": len(manifest["traces"]),
                "B_supervised_tokens": manifest["B_supervised_tokens"],
                "family_supervised": None if arm == "validation" else family_supervised(manifest),
                "eligibility": manifest["eligibility"],
            }
            for arm, manifest in manifests.items()
        },
        "relay_families": manifests["relay"]["relay_families"],
        "relay_pool": manifests["relay"]["relay_pool"],
        "fallback": (
            {"triggered": True, "short_families": sorted(short), "matched_plain": prefix["plain-matched-relay"]}
            if short
            else {"triggered": False}
        ),
    }
    print("CHAIN_BUILT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
