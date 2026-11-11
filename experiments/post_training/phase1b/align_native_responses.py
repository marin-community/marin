# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render fixed sampled responses after each checkpoint's native prompt."""

import argparse

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.native_math_continuations import model_stop_ids, native_prompt_ids, scored_row
from experiments.post_training.probe_sets.continuations import local_tokenizer, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    configure_coreweave_s3()
    tokenizer = local_tokenizer(args.checkpoint)
    stop_ids = model_stop_ids(args.checkpoint)
    for name in args.sets:
        probes = read_jsonl(f"{args.probe_dir.rstrip('/')}/{name}.jsonl")
        responses = {
            row["probe_id"]: row["response_ids"]
            for row in read_jsonl(f"{args.response_dir.rstrip('/')}/{name}.raw.jsonl")
        }
        if len(probes) != len(responses):
            raise ValueError(f"{name}: missing or duplicate responses")
        rows = [
            scored_row(
                probe, name, native_prompt_ids(tokenizer, probe["messages"]), responses[probe["probe_id"]], stop_ids
            )
            for probe in probes
        ]
        digest = write_jsonl(f"{args.out.rstrip('/')}/{name}.jsonl", rows)
        print(f"{name}: {len(rows)} rows, sha256={digest}")


if __name__ == "__main__":
    main()
