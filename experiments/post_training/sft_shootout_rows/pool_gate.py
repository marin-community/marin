# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that a round's generation pools are complete before the production build reads them.

Plain: every shard listed in <trace prefix>/shards/<family>-shard-NN.json has <family>/shard-NN/ with _SUCCESS and
rebuilt.jsonl. Relay, per family under the relay prefix: the family STOP marker <family>-relay/STOP exists (run_lane.sh
starts no round after it), no round was uploaded without _SUCCESS (failed rounds moved to round-N.failed-<id>/ are
ignored), and every round with _SUCCESS has its rebuilt.jsonl. Tokens: the lanes' scrub
writes no marker, so every finished shard's rebuilt*.jsonl and requests*.jsonl are scanned again with check_tokens
(the builder's own test); any hit fails the gate.

A running round is invisible here: harbor_glm uploads a round's directory only when it finishes. So the gate cannot
tell a lane that is still mid-round (or read no STOP just before it was written) from a finished one; confirm every
lane's job has exited before the build. Prints one POOL_GATE line and exits 1 unless every check passes.

    python -m experiments.post_training.sft_shootout_rows.pool_gate [--trace-prefix <s3 .../traces-v7/>] \\
        [--relay-trace-prefix <s3 .../traces-v7-v0/>]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import TRACE_FAMILIES, TRACE_PREFIX
from experiments.post_training.sft_shootout_rows.check_tokens import expand, scan

ROUND = re.compile(r"/round-[0-9]+$")


def finished(directory: StoragePath) -> bool:
    return (directory / "_SUCCESS").exists() and (directory / "rebuilt.jsonl").exists()


def plain_report(prefix: str, family: str) -> tuple[dict, list[str]]:
    root = StoragePath(prefix)
    listed = sorted(
        str(path).rsplit("-shard-", 1)[1].removesuffix(".json")
        for path in (root / f"shards/{family}-shard-*.json").glob()
    )
    done = [shard for shard in listed if finished(root / f"{family}/shard-{shard}")]
    missing = sorted(set(listed) - set(done))
    report = {"shards_listed": len(listed), "finished": len(done), "missing": missing}
    return report, [str(root / f"{family}/shard-{shard}") for shard in done]


def relay_report(prefix: str, family: str) -> tuple[dict, list[str]]:
    root = StoragePath(prefix) / f"{family}-relay"
    rounds = sorted(
        str(path).rstrip("/") for path in (root / "shard-*/round-*").glob() if ROUND.search(str(path).rstrip("/"))
    )
    done, in_progress, without_rebuilt = [], [], []
    for path in rounds:
        directory = StoragePath(path)
        if not (directory / "_SUCCESS").exists():
            in_progress.append(path)
        elif not (directory / "rebuilt.jsonl").exists():
            without_rebuilt.append(path)
        else:
            done.append(path)
    report = {
        "family_stop": (root / "STOP").exists(),
        "shards_with_rounds": len({path.rsplit("/round-", 1)[0] for path in rounds}),
        "rounds_finished": len(done),
        "without_success": [p.removeprefix(str(root) + "/") for p in in_progress],
        "success_without_rebuilt": [p.removeprefix(str(root) + "/") for p in without_rebuilt],
    }
    return report, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-prefix", default=TRACE_PREFIX)
    parser.add_argument("--relay-trace-prefix", help="default: --trace-prefix")
    args = parser.parse_args()
    relay_prefix = args.relay_trace_prefix or args.trace_prefix
    gate: dict = {"trace_prefix": args.trace_prefix, "relay_trace_prefix": relay_prefix, "plain": {}, "relay": {}}
    prefixes: list[str] = []
    for family in TRACE_FAMILIES:
        gate["plain"][family], done = plain_report(args.trace_prefix, family)
        prefixes += done
        gate["relay"][family], done = relay_report(relay_prefix, family)
        prefixes += done
    with ThreadPoolExecutor(16) as pool:
        files = [path for paths in pool.map(expand, prefixes) for path in paths]
        scans = list(pool.map(scan, files))
    gate["tokens"] = {
        "prefixes": len(prefixes),
        "files": len(files),
        "lines": sum(s["lines"] for s in scans),
        "files_with_hits": [s["file"] for s in scans if s["hits"]],
    }
    failures = [
        f"plain {family}: shards not finished {report['missing']}"
        for family, report in gate["plain"].items()
        if report["missing"] or not report["shards_listed"]
    ]
    for family, report in gate["relay"].items():
        if not report["family_stop"]:
            failures.append(f"relay {family}: no family STOP marker")
        if report["without_success"] or report["success_without_rebuilt"]:
            failures.append(f"relay {family}: rounds not finished")
    if gate["tokens"]["files_with_hits"]:
        failures.append("capability tokens in finished shards")
    gate["passed"], gate["failures"] = not failures, failures
    print("POOL_GATE " + json.dumps(gate), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
