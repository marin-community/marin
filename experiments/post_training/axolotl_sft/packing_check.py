# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packed-attention isolation check for the fork's multipack path (pod-side, Axolotl image).

Tokenizes a few real CL12 conversations through the payload's own prompt strategy, packs them with the
collator Axolotl uses for the configured attention backend, and compares each segment's
logits and loss with the same conversation run alone. Also checks that position ids reset
per segment and that the packed labels keep the template's ``{% generation %}`` spans, and
reruns the pack with the first segment's tokens reversed: the later segments must not change.

For Qwen3.5 in float32, run with ``TRITON_F32_DEFAULT=ieee``: fla's Gated DeltaNet kernels
otherwise take TF32 dots, and packed and alone logits then differ by ~5e-2 relative from
rounding alone (segment 0 included); with IEEE dots they agree to ~1e-5.

    python experiments/post_training/axolotl_sft/packing_check.py PAYLOAD.json DATA.jsonl.gz \\
        --attn flex_attention [--model BASE_OR_PATH --device cuda] [--rows 3]

Without ``--model`` a tiny random model of the base's architecture runs on CPU, which checks
the masking path; with the real base on GPU it checks the trained configuration. Exits
non-zero when any segment differs beyond tolerance.
"""

import argparse
import gzip
import json
import sys

import torch
from axolotl.prompt_strategies import load as load_strategy
from axolotl.utils.collators.batching import V2BatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.dict import DictDefault
from packed_attention import SegmentPositionCollator, apply_packing_patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

IGNORE = -100
# bf16 logits differ at about 1e-2 relative between packed and unpacked kernels.
RELATIVE_TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 3e-2}
# The swap test reruns the pack with the first segment's tokens reversed: same layout, same
# kernels, so a later segment's logits may change only through leakage.
SWAP_TOLERANCE = 1e-5


def _features(strategy, rows: list[dict]) -> list[dict]:
    features = []
    for row in rows:
        tokenized = strategy.tokenize_prompt(row)
        length = len(tokenized["input_ids"])
        features.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": [1] * length,
                "labels": tokenized["labels"],
                "position_ids": list(range(length)),
            }
        )
    return features


def _tiny_model(base: str, revision: str, attn: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(base, revision=revision)
    for name, value in {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
    }.items():
        if hasattr(config, name):
            setattr(config, name, value)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, attn_implementation=attn, torch_dtype=torch.float32)


def _segment_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted = labels[1:]
    keep = shifted != IGNORE
    return torch.nn.functional.cross_entropy(logits[:-1][keep].float(), shifted[keep])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    parser.add_argument("data")
    parser.add_argument("--attn", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="real-base dtype; float32 separates leakage from bf16 kernel noise",
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument(
        "--collator",
        choices=("training", "axolotl-default"),
        default="training",
        help="the training run's SegmentPositionCollator, or Axolotl's own multipack collator (leaks)",
    )
    args = parser.parse_args()

    with open(args.payload) as handle:
        config = json.load(handle)["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision_of_model"])
    tokenizer.add_special_tokens({"eos_token": config["special_tokens"]["eos_token"]})
    dataset_config = dict(config["datasets"][0])
    strategy = load_strategy(dataset_config["type"], tokenizer, DictDefault(config), dataset_config)
    with gzip.open(args.data, "rt") as handle:
        rows = [json.loads(line) for _, line in zip(range(args.rows), handle, strict=False)]
    features = _features(strategy, rows)

    model_type = AutoConfig.from_pretrained(config["base_model"], revision=config["revision_of_model"]).model_type
    apply_packing_patch(model_type)
    if args.model is None:
        model = _tiny_model(config["base_model"], config["revision_of_model"], args.attn)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=config["revision_of_model"] if args.model == config["base_model"] else None,
            attn_implementation=args.attn,
            torch_dtype=getattr(torch, args.dtype),
            device_map="auto" if args.device == "cuda" else args.device,
        )
    model.eval()
    dtype = next(model.parameters()).dtype
    collator_cls = SegmentPositionCollator if args.collator == "training" else V2BatchSamplerDataCollatorForSeq2Seq
    collator = collator_cls(tokenizer=tokenizer, return_tensors="pt", padding=False)
    packed = collator([features])
    device = next(model.parameters()).device

    failures = 0
    with torch.no_grad():
        attention_mask = packed["attention_mask"].to(device) if "attention_mask" in packed else None
        packed_logits = model(
            input_ids=packed["input_ids"].to(device),
            attention_mask=attention_mask,
            position_ids=packed["position_ids"].to(device),
            # As in training: a KV cache turns off transformers' packed-segment detection.
            use_cache=False,
        ).logits[0]
        swapped_ids = packed["input_ids"].clone()
        first = len(features[0]["input_ids"])
        swapped_ids[0, :first] = swapped_ids[0, :first].flip(0)
        swapped_logits = model(
            input_ids=swapped_ids.to(device),
            attention_mask=attention_mask,
            position_ids=packed["position_ids"].to(device),
            use_cache=False,
        ).logits[0]
        tolerance = RELATIVE_TOLERANCE[dtype]
        offset = 0
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            ids = torch.tensor([feature["input_ids"]], device=device)
            alone = model(input_ids=ids, position_ids=torch.arange(length, device=device)[None], use_cache=False).logits[
                0
            ]
            segment = packed_logits[offset : offset + length]
            scale = alone.float().abs().max().item()
            per_position = (segment.float() - alone.float()).abs().amax(-1) / scale
            difference = per_position.max().item()
            offending = (per_position > tolerance).nonzero().flatten().tolist()
            swap_change = (
                None
                if index == 0
                else (swapped_logits[offset : offset + length].float() - segment.float()).abs().max().item() / scale
            )
            labels = torch.tensor(feature["labels"], device=device)
            packed_labels = packed["labels"][0, offset : offset + length].to(device)
            losses = (_segment_loss(segment, packed_labels).item(), _segment_loss(alone, labels).item())
            positions_reset = packed["position_ids"][0, offset : offset + length].tolist() == list(range(length))
            labels_kept = packed_labels.tolist() == feature["labels"]
            ids_kept = packed["input_ids"][0, offset : offset + length].tolist() == feature["input_ids"]
            ok = (
                difference <= tolerance
                and (swap_change is None or swap_change <= SWAP_TOLERANCE)
                and positions_reset
                and labels_kept
            )
            failures += not ok
            print(
                json.dumps(
                    {
                        "segment": index,
                        "tokens": length,
                        "max_relative_logit_difference": difference,
                        "positions_over_tolerance": len(offending),
                        "max_relative_change_when_first_segment_reversed": swap_change,
                        "first_offending_positions": offending[:8],
                        "loss_packed": losses[0],
                        "loss_alone": losses[1],
                        "positions_reset": positions_reset,
                        "labels_kept": labels_kept,
                        "ids_kept": ids_kept,
                        "packed_shape": list(packed["input_ids"].shape),
                        "ok": ok,
                    }
                ),
                flush=True,
            )
            offset += length
    print(
        json.dumps(
            {
                "attn": args.attn,
                "collator": args.collator,
                "dtype": str(dtype),
                "segments": len(features),
                "failures": failures,
            }
        )
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
