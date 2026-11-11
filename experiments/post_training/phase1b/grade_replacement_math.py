# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade pinned replacement-teacher continuations on fixed math probes."""

import argparse
import json

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS
from experiments.post_training.probe_sets.buckets import decode_token_bytes
from experiments.post_training.probe_sets.continuations import local_tokenizer, read_jsonl
from experiments.post_training.probe_sets.grade_continuations import grade_reply, set_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuations", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    with fsspec.open(f"{args.continuations.rstrip('/')}/manifest.json") as handle:
        manifest = json.load(handle)
    tokenizer = local_tokenizer(manifest["checkpoint"])
    tokens = FAMILY_TOKEN_MAPS[manifest["family"]]
    stop_ids = set(manifest["stop_token_ids"])
    report = {"continuations": args.continuations, "probe_dir": args.probe_dir, "sets": {}}
    for name in args.sets:
        probes = read_jsonl(f"{args.probe_dir.rstrip('/')}/{name}.jsonl")
        raw = read_jsonl(f"{args.continuations.rstrip('/')}/{name}.raw.jsonl")
        if len(probes) != len(raw):
            raise ValueError(f"{name}: probe and continuation counts differ")
        verdicts = []
        for probe, row in zip(probes, raw, strict=True):
            if row["probe_id"] != probe["probe_id"]:
                raise ValueError(f"{name}: probe and continuation IDs differ")
            ids = row["response_ids"]
            reply = decode_token_bytes(tokenizer, ids[:-1] if ids and ids[-1] in stop_ids else ids)
            verdicts.append({"probe_id": row["probe_id"], **grade_reply("R", probe, reply, tokens.think_close)})
        report["sets"][name] = set_report(verdicts)
        print(name, json.dumps(report["sets"][name], sort_keys=True))
    with fsspec.open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
