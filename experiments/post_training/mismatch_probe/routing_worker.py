# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoring-only router capture on the existing Megatron probe forward."""

import ray
import torch
from skyrl_train.models.router_instrumentation import RouterObservation, observe_router_forwards
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.megatron.router_telemetry import build_router_layer_map, layer_index


class RoutingPolicyWorker(MegatronPolicyWorkerBase):
    def probe_score_forward(self, data):
        """Return routes at input-token positions, retaining the padded batch layout."""
        if data.metadata["probe_micro_batch_size"] != 1:
            raise ValueError("Router capture requires one sequence per micro-batch")
        layer_map = {}
        for chunk in self.actor_module:
            layer_map.update(build_router_layer_map(chunk))
        layers = sorted(set(layer_map.values()))
        if not layers:
            raise ValueError("Router capture found no instrumented layers")
        captures = {layer: [] for layer in layers}

        def observe(observation: RouterObservation) -> None:
            selected = observation.selected_experts.detach()
            scores = observation.selection_logits.detach().float()
            top_scores = scores.topk(selected.shape[-1] + 1, dim=-1).values
            margin = top_scores[:, -2] - top_scores[:, -1]
            captures[layer_index(observation.router, layer_map)].append(
                (selected.cpu(), observation.combine_weights.detach().float().cpu(), margin.cpu())
            )

        with observe_router_forwards(observe):
            output = super().probe_score_forward(data)
        batch, width = data["sequences"].shape
        lengths = data["attention_mask"].sum(-1).tolist()
        top_k = captures[layers[0]][0][0].shape[-1]
        selected = torch.full((batch, width, len(layers), top_k), -1, dtype=torch.int64)
        combine = torch.zeros_like(selected, dtype=torch.float32)
        margins = torch.zeros((batch, width, len(layers)), dtype=torch.float32)
        for layer_column, layer in enumerate(layers):
            if len(captures[layer]) != batch:
                raise ValueError(f"Layer {layer}: expected {batch} forwards, got {len(captures[layer])}")
            for row, ((ids, weights, margin), length) in enumerate(zip(captures[layer], lengths, strict=True)):
                if ids.shape != (length, top_k):
                    raise ValueError(f"Layer {layer}, row {row}: routes {ids.shape} do not match {length} tokens")
                selected[row, -length:, layer_column] = ids
                combine[row, -length:, layer_column] = weights
                margins[row, -length:, layer_column] = margin
        output["router_selected"] = selected
        output["router_combine"] = combine
        output["router_margin"] = margins
        output["router_layers"] = torch.tensor(layers).expand(batch, -1)
        return output


RoutingWorker = ray.remote(num_gpus=1)(RoutingPolicyWorker)
