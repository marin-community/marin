# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer expert loads on ``T`` for an HF MoE checkpoint (pod-side, Axolotl image, one GPU node).

The Stage 0 routing gate for the Axolotl bases. ``T`` documents are packed into 4,096-token
windows with position ids restarting per document. Qwen3.5 and NemotronH also need Axolotl's
packing patches to isolate recurrent state (Gated DeltaNet, Mamba-2; see ``packed_attention.py``).
Each layer's selected top-k experts are counted, and the loads are summed. The report has the
same ``loads``/``tokens``/``t_file`` fields as
:mod:`experiments.post_training.stage0.routing_health`, whose ``--compare`` applies the gate.
Top-k on the raw router logits is the selection rule for softmax routers (Qwen3-MoE,
Qwen3.5-MoE). NemotronH selects on sigmoid scores plus ``e_score_correction_bias`` with group
limiting, so its selections are recorded from the model's own ``route_tokens_to_experts``;
only routed experts are counted (the shared expert is always on).

    python experiments/post_training/axolotl_sft/routing_loads.py MODEL T.jsonl OUT.json [--revision R]
"""

import argparse
import json

import torch
from packed_attention import apply_packing_patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.nemotron_h import modeling_nemotron_h

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


def record_nemotron_h_selections(model) -> list[list[torch.Tensor]]:
    """Record every NemotronH MoE layer's selected expert indices, in layer order.

    Each forward appends one ``[tokens, top_k]`` tensor per MoE layer to that layer's list.
    """
    if not modeling_nemotron_h.is_fast_path_available:
        # The torch fallback scan ignores seq_idx, so Mamba state would leak across documents.
        raise RuntimeError("NemotronH Mamba kernels did not load; packed documents would not be isolated")
    moes = [module for module in model.modules() if isinstance(module, modeling_nemotron_h.NemotronHMoE)]
    selections: list[list[torch.Tensor]] = [[] for _ in moes]
    for moe, sink in zip(moes, selections, strict=True):
        route = moe.route_tokens_to_experts

        def recording_route(router_logits, route=route, sink=sink):
            indices, weights = route(router_logits)
            sink.append(indices.detach())
            return indices, weights

        moe.route_tokens_to_experts = recording_route
    return selections


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
    nemotron_h = config.model_type == "nemotron_h"
    top_k = config.num_experts_per_tok
    experts = config.n_routed_experts if nemotron_h else config.num_experts
    selections = record_nemotron_h_selections(model) if nemotron_h else None
    with open(args.t_file) as handle:
        documents = [tokenizer.encode(json.loads(line)["text"], add_special_tokens=False) for line in handle]

    loads = None
    tokens = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for ids, positions in packed_windows(documents, WINDOW_TOKENS):
            inputs = {
                "input_ids": torch.tensor([ids], device=device),
                "position_ids": torch.tensor([positions], device=device),
                "use_cache": False,
            }
            if nemotron_h:
                model(**inputs)
                selected_per_layer = [sink.pop().flatten() for sink in selections]
            else:
                output = model(**inputs, output_router_logits=True)
                selected_per_layer = [
                    logits.float().topk(top_k, dim=-1).indices.flatten()
                    for logits in output.router_logits
                    if logits is not None
                ]
            layer_loads = []
            for selected in selected_per_layer:
                if selected.numel() != len(ids) * top_k:
                    raise RuntimeError(f"expected {len(ids) * top_k} selections, got {selected.numel()}")
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
