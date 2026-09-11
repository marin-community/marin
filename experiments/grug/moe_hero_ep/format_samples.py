# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the saved sample JSONs from S3 and emit markdown grouped by prompt, one
block per checkpoint, for pasting into the checkpoint-completions issue."""

import json
import logging

import click
import fsspec

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = "s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/analysis"
# multi-prompt files: (checkpoint label, s3 key)
MULTI = [
    ("6k", "sample_multi_step6000_t0.2_s0.json"),
    ("24k", "sample_multi_step24000_t0.2_s0.json"),
    ("42k", "sample_multi_step42000_t0.2_s0.json"),
]
LIST = [
    ("6k", "sample_multi_step6000_t0.2_s1.json"),
    ("24k", "sample_multi_step24000_t0.2_s1.json"),
    ("42k", "sample_multi_step42000_t0.2_s1.json"),
]
SINGLE = [("42k temp=1.0", "sample_gen_step42000.json"), ("42k temp=0 (greedy)", "sample_gen_step42000_t0_s0.json")]


def _load(key):
    with fsspec.open(f"{BASE}/{key}") as fh:
        return json.load(fh)


def _emit_group(files, header, temp_note):
    logger.info("\n\n===GROUP=== %s", header)
    loaded = [(lbl, _load(k)) for lbl, k in files]
    prompts = [r["prompt"] for r in loaded[0][1]["results"]]
    for i, prompt in enumerate(prompts):
        logger.info("\n### `%s`\n_(%s)_", prompt, temp_note)
        for lbl, data in loaded:
            comp = data["results"][i]["completion"]
            eos = " ⟨EOS⟩" if data["results"][i].get("hit_eos") else " ⟨cut at max_new_tokens⟩"
            logger.info("\n**%s**%s\n```\n%s%s\n```", lbl, eos, prompt, comp)


@click.command()
def main() -> None:
    _emit_group(MULTI, "Code / open-ended prompts (temp 0.2, seed 0)", "temp 0.2")
    _emit_group(LIST, "Factual list prompts (temp 0.2, seed 1)", "temp 0.2")


if __name__ == "__main__":
    main()
