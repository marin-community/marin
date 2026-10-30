# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer expert loads on ``T`` for an HF MoE checkpoint (pod-side, Axolotl image, one GPU node).

The Stage 0 routing gate for the Axolotl bases. ``T`` documents are packed into 4,096-token
windows with position ids restarting per document. Qwen3.5 also needs Axolotl's Gated DeltaNet
packing patch to isolate recurrent state (see ``packed_attention.py``). Each layer's router
logits select its top-k experts, and the loads are summed. The report has the same
``loads``/``tokens``/``t_file`` fields as
:mod:`experiments.post_training.stage0.routing_health`, whose ``--compare`` applies the gate.
Top-k on the raw router logits is the selection rule for softmax routers (Qwen3-MoE,
Qwen3.5-MoE); a router whose selection adds a bias (NemotronH) needs its own rule.

    python experiments/post_training/axolotl_sft/routing_loads.py MODEL T.jsonl OUT.json [--revision R]
"""

import argparse
import json

import torch
from packed_attention import apply_packing_patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

WINDOW_TOKENS = 4096


def packed_windows(documents, window):
    """``(ids, position_ids)`` rows of ``window`` tokens; the partial tail is dropped.

    Positions restart at every document and at every window start, so a document cut by a
    window boundary continues as a new segment in the next window.
    """
    ids: list[int] = []
    owners: list[int] = []
    for index, document in enumerate(documents):
        ids.extend(document)
        owners.extend([index] * len(document))
        while len(ids) >= window:
            row_owners = owners[:window]
            positions = []
            for i, owner in enumerate(row_owners):
                positions.append(0 if i == 0 or owner != row_owners[i - 1] else positions[-1] + 1)
            yield ids[:window], positions
            ids, owners = ids[window:], owners[window:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("t_file")
    parser.add_argument("out")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--label", required=True, help="the t_file name recorded in the report")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    apply_packing_patch(config.model_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    config = model.config.get_text_config()
    top_k, experts = config.num_experts_per_tok, config.num_experts
    with open(args.t_file) as handle:
        documents = [tokenizer.encode(json.loads(line)["text"], add_special_tokens=False) for line in handle]

    loads = None
    tokens = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for ids, positions in packed_windows(documents, WINDOW_TOKENS):
            output = model(
                input_ids=torch.tensor([ids], device=device),
                position_ids=torch.tensor([positions], device=device),
                output_router_logits=True,
                use_cache=False,
            )
            layer_loads = []
            for logits in output.router_logits:
                if logits is None:
                    continue
                selected = logits.float().topk(top_k, dim=-1).indices.flatten()
                layer_loads.append(torch.bincount(selected, minlength=experts).cpu())
            step = torch.stack(layer_loads)
            loads = step if loads is None else loads + step
            tokens += len(ids)
    report = {"checkpoint": args.model, "t_file": args.label, "tokens": tokens, "loads": loads.tolist()}
    with open(args.out, "w") as handle:
        json.dump(report, handle)
    print(json.dumps({"tokens": tokens, "layers": len(report["loads"]), "experts": experts}))


if __name__ == "__main__":
    main()
