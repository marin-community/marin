# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pod-side audit: is every assistant turn's ``<|im_end|>`` a trained label? (CPU only).

The stage-1 masking check compares the trained labels with the shared template's own
``{% generation %}`` mask, so a template mask that left the turn-close token out would pass. This
audit counts the turn-close id directly instead: for each CL12 row it tokenizes with the exact
stage payload's strategy (pinned fork, pinned tokenizer, the payload's own inline template) and
records assistant turns against labeled turn-close tokens. It then packs the same tokenized rows
with Axolotl's first-fit-decreasing binner into ``sequence_len`` bins and runs the training
collator over each bin, so a boundary that dropped or masked a label would show up as a
per-bin shortfall.

Run inside the pinned Axolotl task image on a CPU node; the first pass installs the pinned fork
(no Cut Cross Entropy) and re-executes the script under the image's Axolotl virtualenv::

    python experiments/post_training/axolotl_sft/turn_label_check.py \\
        --config-uri s3://BUCKET/.../s1-wildchat-rewild/axolotl_config.yaml \\
        --manifest-uri s3://BUCKET/.../s1-wildchat-rewild/axolotl_run_manifest.json \\
        --rows 2500 --out-uri s3://BUCKET/.../stage0-gate/nemotron_h/turn-label-check-s1.json

Exits non-zero when any trained assistant turn lacks the turn-close label, when the packing
collator drops a label, or when ``</s>`` (the bases' document-end id) appears in a label.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("turn_label_check")

WORK_ROOT = Path("/tmp/turn-label-check")
IMAGE_PYTHON = "/workspace/axolotl-venv/bin/python"
FORK_REPO = "https://github.com/marin-community/axolotl"
IGNORE = -100
# The pinned fork is installed by the first pass, which then re-executes under the image's
# virtualenv; the analysis imports the fork, so it can only run after that install.
REEXEC_ENV = "TURN_LABEL_CHECK_REEXEC"


def _run(command: list[str], **kwargs) -> None:
    logger.info("+ %s", " ".join(command))
    subprocess.run(command, check=True, **kwargs)


def _aws_environment() -> dict[str, str]:
    # CoreWeave object storage requires virtual-hosted addressing.
    config = WORK_ROOT / "aws-config"
    config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
    return os.environ | {"AWS_CONFIG_FILE": str(config), "AWS_MAX_ATTEMPTS": "20", "AWS_RETRY_MODE": "adaptive"}


def _s3(arguments: list[str]) -> None:
    _run(["aws", "s3", *arguments, "--only-show-errors"], env=_aws_environment())


def _download(uri: str, name: str) -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    local = WORK_ROOT / name
    if not local.exists():
        _s3(["cp", uri, str(local)])
    return local


def _install_fork(revision: str) -> None:
    """The pinned fork over the image's Axolotl; the tokenizer path is the fork's strategy."""
    fork = WORK_ROOT / "axolotl-fork"
    if not fork.exists():
        _run(["git", "clone", "--quiet", FORK_REPO, str(fork)])
    _run(["git", "-C", str(fork), "checkout", "--quiet", revision])
    _run(["uv", "pip", "install", "--python", IMAGE_PYTHON, "-e", str(fork)])
    installed = subprocess.run(
        [IMAGE_PYTHON, "-c", "import transformers; print(transformers.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    logger.info("fork installed with transformers %s", installed)


def _assistant_close_positions(input_ids: list[int], header: list[int], turn_close: int) -> list[int | None]:
    """For each rendered assistant header, the index of the ``<|im_end|>`` that closes that turn.

    Independent of the template's ``{% generation %}`` mask and of the strategy's turn search:
    the header is encoded separately, the close is the first turn-close token after it, and
    ``None`` marks a turn that never closes.
    """
    positions: list[int | None] = []
    for index in range(len(input_ids) - len(header) + 1):
        if input_ids[index : index + len(header)] != header:
            continue
        try:
            positions.append(input_ids.index(turn_close, index + len(header)))
        except ValueError:
            positions.append(None)
    return positions


def _row_label_stats(feature: dict, assistant_turns: int, turn_close: int, header: list[int]) -> dict:
    input_ids, labels = feature["input_ids"], feature["labels"]
    labeled_positions = [index for index, label in enumerate(labels) if label != IGNORE]
    labeled_close = [index for index in labeled_positions if labels[index] == turn_close]
    present_close = [index for index, token in enumerate(input_ids) if token == turn_close]
    after = collections.Counter()
    for index in labeled_close:
        if index + 1 >= len(labels):
            after["end_of_row"] += 1
        else:
            after["labeled" if labels[index + 1] != IGNORE else "unlabeled"] += 1
    # Every rendered user/system/tool turn also ends with the turn-close token, so the count of
    # closes alone cannot attribute a label to an assistant turn; the header scan does that.
    turns = _assistant_close_positions(input_ids, header, turn_close)
    turns_without_close = sum(1 for position in turns if position is None)
    turns_with_unlabeled_close = sum(1 for position in turns if position is not None and labels[position] == IGNORE)
    last_labeled_token = labels[labeled_positions[-1]] if labeled_positions else None
    return {
        "assistant_turns": assistant_turns,
        "assistant_headers_rendered": len(turns),
        "assistant_turns_without_close_token": turns_without_close,
        "assistant_turns_with_unlabeled_close": turns_with_unlabeled_close,
        "labeled_turn_close": len(labeled_close),
        "turn_close_in_input_ids": len(present_close),
        "unlabeled_turn_close": len(present_close) - len(labeled_close),
        "labeled_positions": len(labeled_positions),
        "after_labeled_turn_close": dict(after),
        # A trained turn must end on the turn-close token: anything else trained after it would
        # teach the model to keep writing past the end of a turn.
        "last_labeled_token": last_labeled_token,
        "eos_in_input_ids": sum(1 for token in input_ids if token == 2),
        "eos_in_labels": sum(1 for token in labels if token == 2),
        "length": len(input_ids),
    }


def _pack_stats(features: list[dict], tokenizer, strlen: int, bin_size: int, pad_multiple: int | None) -> dict:
    """Pack with the training binner, then check the labels the training collator returns."""
    import numpy as np  # noqa: PLC0415 - image environment
    from axolotl.utils.samplers.multipack import pack_group  # noqa: PLC0415 - image environment
    from packed_attention import SegmentPositionCollator  # noqa: PLC0415 - image environment

    turn_close = tokenizer.convert_tokens_to_ids("<|im_end|>")
    lengths = np.array([len(feature["input_ids"]) for feature in features], dtype=np.int32)
    bins = pack_group(lengths, 0, strlen, max_bins=len(features), bin_size=bin_size, safe_mode=True)
    collator = SegmentPositionCollator(
        tokenizer=tokenizer, return_tensors="pt", padding=True, pad_to_multiple_of=pad_multiple
    )
    segments_close = sum(feature["labels"].count(turn_close) for feature in features)
    stats = {
        "bins": len(bins),
        "segments": len(features),
        "segments_placed": sum(len(bin_indices) for bin_indices in bins),
        "bin_size": bin_size,
        "strlen": strlen,
        "pad_multiple": pad_multiple,
        "labels_identical_to_segments": True,
        "close_labels_before": segments_close,
        "close_labels_after": 0,
        # A pack's last position is never a target (labels are shifted by one), so a label there
        # contributes no gradient.
        "close_labels_at_pack_final_position": 0,
        "padded_lengths": set(),
    }
    after = 0
    for bin_indices in bins:
        pack = [features[index] for index in bin_indices]
        packed = collator([pack])
        packed_labels = packed["labels"][0].tolist()
        packed_ids = packed["input_ids"][0].tolist()
        stats["padded_lengths"].add(len(packed_labels))
        offset = 0
        for feature in pack:
            length = len(feature["input_ids"])
            if packed_ids[offset : offset + length] != feature["input_ids"]:
                stats["labels_identical_to_segments"] = False
            if packed_labels[offset : offset + length] != feature["labels"]:
                stats["labels_identical_to_segments"] = False
            offset += length
        after += sum(1 for label in packed_labels[:offset] if label == turn_close)
        if packed_labels[offset - 1] == turn_close:
            stats["close_labels_at_pack_final_position"] += 1
    stats["close_labels_after"] = after
    stats["padded_lengths"] = sorted(stats["padded_lengths"])
    return stats


def _audit(config: dict, data_path: Path, rows: int) -> dict:
    from axolotl.prompt_strategies.chat_template import StrategyLoader  # noqa: PLC0415 - image environment
    from axolotl.utils.dict import DictDefault  # noqa: PLC0415 - image environment
    from transformers import AutoTokenizer  # noqa: PLC0415 - image environment

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision_of_model"])
    tokenizer.add_special_tokens({"eos_token": config["special_tokens"]["eos_token"]})
    # Axolotl's tokenizer loader falls back to the eos token when no pad token is set; the
    # training collator therefore pads input ids with the turn-close id and labels with -100.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    turn_close = tokenizer.convert_tokens_to_ids("<|im_end|>")
    document_end = tokenizer.convert_tokens_to_ids("</s>")
    if tokenizer.convert_tokens_to_ids(config["special_tokens"]["eos_token"]) != turn_close:
        raise ValueError("the payload's eos token is not the turn-close token")
    strategy = StrategyLoader()(tokenizer, DictDefault(config), dict(config["datasets"][0]))
    strlen = int(config["sequence_len"])
    # The rendered assistant header, encoded on its own: the anchor for the per-turn scan.
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    features: list[dict] = []
    stats: dict = {
        "turn_close_id": turn_close,
        "document_end_id": document_end,
        "sequence_len": strlen,
        "assistant_header_ids": header,
        "rows_requested": rows,
        "rows_read": 0,
        "rows_tokenized": 0,
        "rows_over_sequence_len": 0,
        "errors": [],
        "turns_per_row": collections.Counter(),
        "labeled_minus_turns": collections.Counter(),
        "assistant_turns": 0,
        "assistant_turns_without_close_token": 0,
        "assistant_turns_with_unlabeled_close": 0,
        "labeled_turn_close": 0,
        "rows_with_missing_turn_close": [],
        "rows_with_unclosed_turn": [],
        "rows_with_extra_turn_close": [],
        "rows_whose_last_label_is_not_turn_close": [],
        "after_labeled_turn_close": collections.Counter(),
        "turn_close_in_input_ids": 0,
        "unlabeled_turn_close": 0,
        "rows_with_document_end_in_input_ids": 0,
        "rows_with_document_end_in_labels": 0,
    }
    with gzip.open(data_path, "rt") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            row = json.loads(line)
            stats["rows_read"] += 1
            if stats["rows_read"] % 500 == 0:
                logger.info("tokenized %d rows", stats["rows_read"])
            assistant_turns = sum(1 for message in row.get("messages", []) if message.get("role") == "assistant")
            try:
                tokenized = strategy.tokenize_prompt(row)
            except Exception as error:  # A single bad row must not hide the audit.
                stats["errors"].append({"row_id": row.get("row_id"), "error": f"{type(error).__name__}: {error}"[:300]})
                continue
            length = len(tokenized["input_ids"])
            if length > strlen:
                # Axolotl drops these before training; they are not part of the trained data.
                stats["rows_over_sequence_len"] += 1
                continue
            feature = {
                "input_ids": list(tokenized["input_ids"]),
                "labels": list(tokenized["labels"]),
                "position_ids": list(range(length)),
                "attention_mask": [1] * length,
            }
            row_stats = _row_label_stats(feature, assistant_turns, turn_close, header)
            features.append(feature)
            stats["rows_tokenized"] += 1
            stats["turns_per_row"][assistant_turns] += 1
            stats["labeled_minus_turns"][row_stats["labeled_turn_close"] - assistant_turns] += 1
            stats["assistant_turns"] += assistant_turns
            stats["assistant_turns_without_close_token"] += row_stats["assistant_turns_without_close_token"]
            stats["assistant_turns_with_unlabeled_close"] += row_stats["assistant_turns_with_unlabeled_close"]
            stats["labeled_turn_close"] += row_stats["labeled_turn_close"]
            stats["turn_close_in_input_ids"] += row_stats["turn_close_in_input_ids"]
            stats["unlabeled_turn_close"] += row_stats["unlabeled_turn_close"]
            stats["after_labeled_turn_close"].update(row_stats["after_labeled_turn_close"])
            stats["rows_with_document_end_in_input_ids"] += bool(row_stats["eos_in_input_ids"])
            stats["rows_with_document_end_in_labels"] += bool(row_stats["eos_in_labels"])
            if row_stats["assistant_turns_without_close_token"]:
                stats["rows_with_unclosed_turn"].append(row.get("row_id"))
            if row_stats["assistant_turns_with_unlabeled_close"]:
                stats["rows_with_missing_turn_close"].append(row.get("row_id"))
            if row_stats["labeled_turn_close"] > assistant_turns:
                stats["rows_with_extra_turn_close"].append(row.get("row_id"))
            if row_stats["last_labeled_token"] not in (None, turn_close):
                stats["rows_whose_last_label_is_not_turn_close"].append(row.get("row_id"))

    pad_multiple = None
    if config.get("pad_to_sequence_len"):
        multiple = 64
        pad_multiple = multiple * -(-strlen // multiple)
    stats["packing"] = _pack_stats(
        features, tokenizer, strlen, int(config.get("sample_packing_bin_size") or 200), pad_multiple
    )
    stats["passed"] = (
        not stats["rows_with_missing_turn_close"]
        and not stats["rows_with_extra_turn_close"]
        and not stats["rows_whose_last_label_is_not_turn_close"]
        and not stats["rows_with_document_end_in_labels"]
        and stats["packing"]["labels_identical_to_segments"]
        and stats["packing"]["close_labels_before"] == stats["packing"]["close_labels_after"]
    )
    for key in ("turns_per_row", "labeled_minus_turns", "after_labeled_turn_close"):
        stats[key] = dict(sorted(stats[key].items()))
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-uri", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--data-uri", default=None, help="override the manifest's data file")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--out-uri", required=True)
    args = parser.parse_args()

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(_download(args.manifest_uri, "axolotl_run_manifest.json").read_text())
    config_path = _download(args.config_uri, "axolotl_config.yaml")

    if os.environ.get(REEXEC_ENV) != "1":
        _install_fork(manifest["fork_revision"])
        os.environ[REEXEC_ENV] = "1"
        os.execv(IMAGE_PYTHON, [IMAGE_PYTHON, *sys.argv])

    import yaml  # noqa: PLC0415 - image environment

    config = yaml.safe_load(config_path.read_text())
    data_uri = args.data_uri or manifest["data_uri"]
    data_path = _download(data_uri, data_uri.rsplit("/", 1)[-1])
    config["datasets"] = [dict(config["datasets"][0], path=str(data_path))]

    result = _audit(config, data_path, args.rows)
    result["data_uri"] = data_uri
    result["config_uri"] = args.config_uri
    result["base"] = manifest["base"]
    result["template_sha256"] = manifest["template_sha256"]
    out_path = WORK_ROOT / "turn-label-check.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    logger.info("result: %s", json.dumps({key: result[key] for key in ("passed", "rows_tokenized", "assistant_turns")}))
    _s3(["cp", str(out_path), args.out_uri])
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
