# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select short, ended native samples for the vLLM/Megatron parity check."""

import argparse

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.probe_sets.continuations import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    configure_coreweave_s3()

    rows = read_jsonl(args.input)
    eligible = [row for row in rows if row["turn_ended"]]
    selected = sorted(eligible, key=lambda row: (len(row["token_ids"]), row["sample_id"]))[: args.count]
    if len(selected) != args.count:
        raise ValueError(f"need {args.count} ended samples, found {len(selected)}")
    digest = write_jsonl(args.out, selected)
    print(
        f"selected {len(selected)}/{len(rows)} ended samples, "
        f"{sum(row['completion_tokens'] for row in selected)} response tokens, sha256={digest}"
    )


if __name__ == "__main__":
    main()
