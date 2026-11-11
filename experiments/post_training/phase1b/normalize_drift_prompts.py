# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mark frozen B probe prefixes as context for response-only drift scoring."""

import argparse

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.probe_sets.continuations import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    for name in ("P_chat", "P_think"):
        rows = read_jsonl(f"{args.input_dir.rstrip('/')}/{name}.jsonl")
        normalized = []
        for row in rows:
            prompt_tokens = row["prompt_tokens"]
            token_ids = row["token_ids"]
            tags = row["token_type_tags"]
            if not 0 < prompt_tokens < len(token_ids) or len(tags) != len(token_ids):
                raise ValueError(f"{row['sample_id']}: invalid token/tag lengths")
            normalized.append({**row, "token_type_tags": ["context"] * prompt_tokens + tags[prompt_tokens:]})
        digest = write_jsonl(f"{args.out.rstrip('/')}/{name}.jsonl", normalized)
        print(f"{name}: {len(normalized)} rows, sha256={digest}")


if __name__ == "__main__":
    main()
