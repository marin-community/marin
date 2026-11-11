# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a common response panel within every candidate's context window."""

import argparse
import json

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.probe_sets.continuations import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-input", nargs=2, action="append", required=True, metavar=("NAME", "URI"))
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--require-ended", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    panels = {name: read_jsonl(uri) for name, uri in args.named_input}
    if len(panels) != len(args.named_input):
        raise ValueError("duplicate panel names")
    reference = next(iter(panels.values()))
    for name, panel in panels.items():
        if len(panel) != len(reference):
            raise ValueError(f"{name}: row count differs from reference")
        for first, row in zip(reference, panel, strict=True):
            if first["sample_id"] != row["sample_id"]:
                raise ValueError(f"{name}: sample ID mismatch at {first['sample_id']}")
            if first["token_ids"][first["prompt_tokens"] :] != row["token_ids"][row["prompt_tokens"] :]:
                raise ValueError(f"{name}: response token mismatch at {first['sample_id']}")

    ended = [
        index
        for index in range(len(reference))
        if not args.require_ended or all(panel[index]["turn_ended"] for panel in panels.values())
    ]
    selected = [
        index for index in ended if all(len(panel[index]["token_ids"]) <= args.max_tokens for panel in panels.values())
    ]
    if not selected:
        raise ValueError("common context panel is empty")
    digests = {}
    for name, panel in panels.items():
        digests[name] = write_jsonl(f"{args.out.rstrip('/')}/{name}.jsonl", [panel[index] for index in selected])
    summary = {
        "source_rows": len(reference),
        "selected_rows": len(selected),
        "dropped_rows": len(reference) - len(selected),
        "non_ended_rows": len(reference) - len(ended),
        "context_dropped_rows": len(ended) - len(selected),
        "max_tokens": args.max_tokens,
        "sha256": digests,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
