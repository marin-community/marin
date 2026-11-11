# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract T1's fixed P_J responses for reciprocal math-teacher scoring."""

import argparse

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.probe_sets.continuations import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    probes = read_jsonl(f"{args.probe_dir.rstrip('/')}/P_J.jsonl")
    scored = read_jsonl(f"{args.probe_dir.rstrip('/')}/P_J-T1.jsonl")
    if len(probes) != len(scored):
        raise ValueError("P_J probes and T1 responses have different row counts")
    raw = []
    for probe, row in zip(probes, scored, strict=True):
        if row["sample_id"] != probe["probe_id"] or row["source"] != probe["source"]:
            raise ValueError(f"P_J and T1 response misaligned: {probe['probe_id']}")
        prompt_tokens = row["prompt_tokens"]
        response_ids = row["token_ids"][prompt_tokens:]
        if not response_ids or len(response_ids) != row["completion_tokens"]:
            raise ValueError(f"empty or truncated T1 response: {probe['probe_id']}")
        raw.append({"probe_id": probe["probe_id"], "response_ids": response_ids})

    digest = write_jsonl(f"{args.out.rstrip('/')}/P_J.raw.jsonl", raw)
    print(f"P_J: {len(raw)} fixed T1 responses, sha256={digest}")


if __name__ == "__main__":
    main()
