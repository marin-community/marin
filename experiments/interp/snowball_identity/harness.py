# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Activation capture and interventions for Snowball v0.1 on GPUs (task #296).

Loads the checkpoint in BF16 into the torch GrugMoE (``skyrl_train.models.grug_moe``) with the 26 layers
split evenly over the visible GPUs. Every forward is a full forward (the torch GrugMoE has no KV cache).

Subcommands (each writes under ``--output``, an s3:// or local directory):

``positions``
    Teacher-forced log-probs at every position of the logit-check sequences (``sequences.py``): the actual
    token's log-prob and the top 50, saved as ``positions.pt``. ``logit_check.py`` compares them with
    ``vllm_reference.py``.
``capture``
    For each prompt, at the chosen positions: the residual stream after each layer; the attention, routed-MLP
    and shared-expert outputs per layer; the router's top-4 expert ids and combine weights; the post-activation
    neurons of the 4 selected routed experts and of the shared expert; the top-50 next-token log-probs; and the
    log-probs of each listed target continuation. One ``capture/<id>.pt`` per prompt plus ``capture.jsonl``.
``generate``
    Sample continuations, with the same interventions active. Writes ``generate.jsonl``.

Interventions (``capture`` and ``generate``; each flag may repeat):

``--add-vector PATH,LAYER,SCALE[,assistant]``
    Add ``SCALE * v`` to the residual stream after LAYER, at all positions or at assistant positions only.
    PATH is a ``.pt`` file holding a 1-D tensor of size 2560.
``--neuron LAYER,EXPERT,INDEX,SCALE``
    Multiply one post-activation neuron by SCALE (0 zeroes it). EXPERT is a routed expert id or ``shared``.
``--patch SOURCE_ID,LAYER,POS[,SOURCE_POS]``
    Replace the residual stream after LAYER at POS with the value from prompt SOURCE_ID's run at SOURCE_POS
    (default POS). Positions may be negative (counted from the end). The source runs without interventions.

Positions (``--positions``): ``last``, ``assistant`` (the assistant header onward), ``all``, or a comma list.

To run many jobs on one loaded model, use ``worker.py`` (an S3 job queue).
"""

import argparse
import io
import json
import logging
import sys
import time
import types
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity.prompts import (
    SNOWBALL_V01,
    Prompt,
    hf_repo_revision,
    load_tokenizer,
    read_prompts,
)
from experiments.interp.snowball_identity.sequences import Sequence, build_sequences

logger = logging.getLogger(__name__)

TOP_K = 50
SKYRL_TRAIN = Path(__file__).resolve().parents[3] / "vendor" / "skyrl" / "skyrl-train"


def import_grug():
    """Import the vendored torch GrugMoE without ``skyrl_train.utils/__init__``, which pulls in ray and the trainer.

    The model module needs only ``skyrl_train.utils.flash_attention``.
    """
    if str(SKYRL_TRAIN) not in sys.path:
        sys.path.insert(0, str(SKYRL_TRAIN))
    try:
        import_module("loguru")
    except ModuleNotFoundError:
        # The GPU extra lacks loguru; the model module only calls logger.info/warning.
        sys.modules["loguru"] = types.SimpleNamespace(logger=logging.getLogger("skyrl_train.models.grug_moe"))
    if "skyrl_train.utils" not in sys.modules:
        utils = types.ModuleType("skyrl_train.utils")
        utils.__path__ = [str(SKYRL_TRAIN / "skyrl_train" / "utils")]
        sys.modules["skyrl_train.utils"] = utils
    return import_module("skyrl_train.models.grug_moe")


COMPONENTS = ("resid_pre", "attn", "routed", "shared", "resid", "head")
"""Patchable activations of a layer. ``resid_pre`` enters the layer and ``resid`` leaves it; ``attn``, ``routed``
and ``shared`` are the sublayer outputs before the residual add; ``head`` is one attention head's slice of
``o_proj``'s input (``head_dim`` values)."""


@dataclass(frozen=True)
class AddVector:
    layer: int
    vector: torch.Tensor
    scale: float
    assistant_only: bool


@dataclass(frozen=True)
class NeuronEdit:
    layer: int
    expert: int | None
    """Routed expert id, or None for the shared expert."""
    index: int
    scale: float


@dataclass(frozen=True)
class Replace:
    """Overwrite one activation at ``positions`` with ``values`` ([len(positions), width])."""

    layer: int
    component: str
    positions: torch.Tensor
    values: torch.Tensor
    head: int | None = None


@dataclass(frozen=True)
class PatchSpec:
    source: str
    layer: int
    position: int
    source_position: int


@dataclass
class Interventions:
    add: list[AddVector] = field(default_factory=list)
    neurons: list[NeuronEdit] = field(default_factory=list)
    replace: list[Replace] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "add": [{"layer": a.layer, "scale": a.scale, "assistant_only": a.assistant_only} for a in self.add],
            "neurons": [
                {
                    "layer": n.layer,
                    "expert": "shared" if n.expert is None else n.expert,
                    "index": n.index,
                    "scale": n.scale,
                }
                for n in self.neurons
            ],
            "replace": [
                {"layer": r.layer, "component": r.component, "head": r.head, "positions": r.positions.tolist()}
                for r in self.replace
            ],
        }


KINDS = ("resid", "attn", "routed", "shared", "head_in", "router_ids", "router_weights", "expert_act", "shared_act")


class Recorder:
    """Activations at the capture positions of one batched forward: each entry is [batch, positions, ...]."""

    def __init__(self, positions: torch.Tensor, layer_count: int, batch: int, length: int):
        self.positions = positions
        self.batch, self.length = batch, length
        self.slot = torch.full((length,), -1, dtype=torch.long)
        self.slot[positions] = torch.arange(len(positions))
        self.embed: torch.Tensor | None = None
        for kind in KINDS:
            setattr(self, kind, [None] * layer_count)

    def take(self, tensor: torch.Tensor, flat: bool = False) -> torch.Tensor:
        """[batch, positions, ...] float32 CPU rows of a [batch, length, ...] tensor ([batch * length, ...] if flat)."""
        if flat:
            tensor = tensor.reshape(self.batch, self.length, *tensor.shape[1:])
        return tensor[:, self.positions.to(tensor.device)].float().cpu()

    def resid_pre(self, layer: int) -> torch.Tensor:
        return self.embed if layer == 0 else self.resid[layer - 1]

    def select(self, row: int) -> "Recorder":
        """The same record for one batch row, without the batch dimension."""
        single = Recorder(self.positions, len(self.resid), 1, self.length)
        single.embed = None if self.embed is None else self.embed[row]
        for kind in KINDS:
            setattr(single, kind, [None if value is None else value[row] for value in getattr(self, kind)])
        return single

    def stacked(self) -> dict[str, torch.Tensor]:
        """Each kind stacked over layers on the dimension after positions."""
        stack_dim = 1 if self.embed is not None and self.embed.dim() == 2 else 2
        return {
            "resid": torch.stack(self.resid, stack_dim),
            "attn_out": torch.stack(self.attn, stack_dim),
            "routed_out": torch.stack(self.routed, stack_dim),
            "shared_out": torch.stack(self.shared, stack_dim),
            "router_ids": torch.stack(self.router_ids, stack_dim),
            "router_weights": torch.stack(self.router_weights, stack_dim),
            "expert_act": torch.stack(self.expert_act, stack_dim),
            "shared_act": torch.stack(self.shared_act, stack_dim),
        }


class _ForwardState:
    """Per-forward lookups shared with the hooks: batch shape, recorder and each row's interventions."""

    def __init__(self):
        self.reset()

    def reset(self, batch: int = 1, length: int = 0, recorder: Recorder | None = None, variants=()):
        self.batch, self.length, self.recorder = batch, length, recorder
        self.neurons: dict[tuple[int, int | None], list[tuple[int, int, float]]] = {}
        self.replace: dict[tuple[int, str], list[tuple[int, Replace]]] = {}
        for row, variant in enumerate(variants):
            for edit in variant.neurons:
                self.neurons.setdefault((edit.layer, edit.expert), []).append((row, edit.index, edit.scale))
            for replace in variant.replace:
                if replace.component not in COMPONENTS:
                    raise ValueError(f"Unknown component {replace.component!r}")
                self.replace.setdefault((replace.layer, replace.component), []).append((row, replace))


def _apply_replacements(tensor: torch.Tensor, replacements: list[tuple[int, Replace]], head_dim: int = 0):
    tensor = tensor.clone()
    for row, replace in replacements:
        positions = replace.positions.to(tensor.device)
        values = replace.values.to(device=tensor.device, dtype=tensor.dtype)
        if replace.head is None:
            tensor[row, positions] = values
        else:
            tensor[row, positions, replace.head * head_dim : (replace.head + 1) * head_dim] = values
    return tensor


class _HarnessExpertExecution:
    """Eager routed-expert execution that exposes and edits the post-activation neurons.

    Same arithmetic as ``GrugMoeExperts.forward_eager``, but only over the selected experts.
    """

    def __init__(self, layer: int, state: _ForwardState):
        self.layer = layer
        self.state = state

    def run(self, experts, reorderer, hidden_states, selected_experts, combine_weights):
        del reorderer
        state, recorder = self.state, self.state.recorder
        if recorder is not None:
            captured = torch.zeros(
                state.batch, len(recorder.positions), selected_experts.shape[1], experts.gate_proj.weight.shape[1]
            )
            slot = recorder.slot.to(selected_experts.device)
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert in torch.unique(selected_experts).tolist():
            token_idx, slot_idx = torch.where(selected_experts == expert)
            expert_input = hidden_states.index_select(0, token_idx)
            gate = F.linear(expert_input, experts.gate_proj.weight[expert])
            activation = F.silu(gate) * F.linear(expert_input, experts.up_proj.weight[expert])
            rows = token_idx // state.length
            for row, index, scale in state.neurons.get((self.layer, expert), ()):
                activation[rows == row, index] *= scale
            if recorder is not None:
                kept = slot[token_idx % state.length]
                mask = kept >= 0
                if mask.any():
                    captured[rows[mask].cpu(), kept[mask].cpu(), slot_idx[mask].cpu()] = activation[mask].float().cpu()
            expert_output = F.linear(activation, experts.down_proj.weight[expert])
            weight = combine_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, (expert_output * weight).float())
        if recorder is not None:
            recorder.expert_act[self.layer] = captured
        return output.to(hidden_states.dtype)


class SnowballHarness:
    """A loaded GrugMoE with capture hooks; ``devices`` holds the layer split (one device is fine)."""

    def __init__(self, model, devices: tuple[torch.device, ...]):
        grug = import_grug()
        self.model = model
        self.layers = model.model.layers
        self.layer_count = len(self.layers)
        self.head_dim = model.config.head_dim
        self.long_layers = grug.grug_long_layer_flags(self.layer_count)
        self.devices = devices
        self.layer_devices = [devices[layer * len(devices) // self.layer_count] for layer in range(self.layer_count)]
        self.state = _ForwardState()
        for layer, block in enumerate(self.layers):
            block.mlp._expert_execution = _HarnessExpertExecution(layer, self.state)
            block.self_attn.register_forward_hook(self._output_hook(layer, "attn"))
            block.self_attn.o_proj.register_forward_pre_hook(self._head_hook(layer))
            block.mlp.register_forward_hook(self._output_hook(layer, "routed"))
            block.shared_expert.register_forward_hook(self._output_hook(layer, "shared"))
            block.mlp.router.register_forward_hook(self._router_hook(layer))
            block.shared_expert.down_proj.register_forward_pre_hook(self._shared_activation_hook(layer))

    @classmethod
    def load(cls, weights: str, revision: str, devices: tuple[torch.device, ...]) -> "SnowballHarness":
        """Build the model on the meta device and assign the BF16 checkpoint tensors on their layer's GPU."""
        grug = import_grug()
        fs, path = filesystem_for(weights)
        config = grug.GrugMoeConfig(**json.loads(fs.cat_file(prefix_join(path, "config.json"))))
        config._attn_implementation = "eager"
        with torch.device("meta"):
            model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
        layer_count = config.num_hidden_layers
        reader = CheckpointReader(CheckpointSource(path=weights, revision=revision))
        expected = model.state_dict()
        if set(reader.weight_map) != set(expected):
            raise ValueError(f"Grug implementation and checkpoint keys differ: {weights}")

        def device_for(name: str) -> torch.device:
            if name.startswith("model.layers."):
                return devices[int(name.split(".")[2]) * len(devices) // layer_count]
            if name.startswith(("model.embed_tokens", "model.embed_norm", "model.embed_gated_norm")):
                return devices[0]
            return devices[-1]

        started = time.time()
        loaded = {}
        for name in sorted(expected, key=lambda key: (reader.weight_map[key], key)):
            tensor = reader.tensor(name)
            if tensor.shape != expected[name].shape:
                raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs model {tuple(expected[name].shape)}")
            loaded[name] = tensor.to(device_for(name))
        model.load_state_dict(loaded, strict=True, assign=True)
        logger.info("Loaded %d tensors in %.0fs", len(loaded), time.time() - started)
        return cls(model, devices)

    def _output_hook(self, layer: int, kind: str):
        def hook(module, args, output):
            replacements = self.state.replace.get((layer, kind))
            if replacements:
                output = _apply_replacements(output, replacements)
            if self.state.recorder is not None:
                getattr(self.state.recorder, kind)[layer] = self.state.recorder.take(output)
            return output if replacements else None

        return hook

    def _head_hook(self, layer: int):
        def hook(module, args):
            (heads,) = args
            replacements = self.state.replace.get((layer, "head"))
            if replacements:
                heads = _apply_replacements(heads, replacements, self.head_dim)
            if self.state.recorder is not None:
                self.state.recorder.head_in[layer] = self.state.recorder.take(heads)
            return (heads,) if replacements else None

        return hook

    def _router_hook(self, layer: int):
        def hook(module, args, output):
            if self.state.recorder is not None:
                self.state.recorder.router_ids[layer] = self.state.recorder.take(
                    output.selected_experts, flat=True
                ).long()
                self.state.recorder.router_weights[layer] = self.state.recorder.take(output.combine_weights, flat=True)

        return hook

    def _shared_activation_hook(self, layer: int):
        def hook(module, args):
            (activation,) = args
            edits = self.state.neurons.get((layer, None), ())
            if edits:
                activation = activation.clone()
                for row, index, scale in edits:
                    activation[row, ..., index] *= scale
            if self.state.recorder is not None:
                self.state.recorder.shared_act[layer] = self.state.recorder.take(activation)
            return (activation,) if edits else None

        return hook

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        model = self.model.model
        return model.final_gated_norm(model.norm(hidden.to(self.devices[-1])))

    def run(self, *args, grad: bool = False, **kwargs) -> tuple[torch.Tensor, Recorder | None]:
        """One batched forward over equal-length sequences, row ``b`` with ``variants[b]``.

        Returns float32 CPU log-probs [batch, len(logit_positions), vocab] (default: the last position), or only
        the ``gather`` token ids' log-probs [batch, positions, len(gather)]. With ``grad`` the log-probs stay on
        the last GPU and carry autograd (for optimizing an edit through the frozen model).
        """
        with torch.set_grad_enabled(grad):
            logprobs, recorder = self._run(*args, **kwargs)
        return (logprobs if grad else logprobs.cpu()), recorder

    def _run(
        self,
        tokens: list[list[int]] | list[tuple[int, ...]],
        *,
        assistant_start: int,
        variants: list[Interventions] | None = None,
        capture: torch.Tensor | None = None,
        logit_positions: torch.Tensor | None = None,
        gather: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Recorder | None]:
        batch, length = len(tokens), len(tokens[0])
        if any(len(row) != length for row in tokens):
            raise ValueError("Batched sequences must have equal lengths")
        variants = variants or [Interventions() for _ in range(batch)]
        if len(variants) != batch:
            raise ValueError(f"{len(variants)} variants for {batch} sequences")
        recorder = None if capture is None else Recorder(capture, self.layer_count, batch, length)
        self.state.reset(batch, length, recorder, variants)
        try:
            model = self.model.model
            hidden = model.embed_gated_norm(
                model.embed_norm(model.embed_tokens(torch.tensor([list(row) for row in tokens], device=self.devices[0])))
            )
            if recorder is not None:
                recorder.embed = recorder.take(hidden)
            positions = torch.arange(length).unsqueeze(0)
            assistant = torch.arange(length) >= assistant_start
            for layer, block in enumerate(self.layers):
                device = self.layer_devices[layer]
                hidden = hidden.to(device)
                replacements = self.state.replace.get((layer, "resid_pre"))
                if replacements:
                    hidden = _apply_replacements(hidden, replacements)
                hidden = block(hidden, None, positions.to(device), self.long_layers[layer])
                for row, variant in enumerate(variants):
                    for add in variant.add:
                        if add.layer == layer:
                            delta = (add.scale * add.vector).to(device=device, dtype=hidden.dtype)
                            mask = assistant if add.assistant_only else torch.ones(length, dtype=torch.bool)
                            hidden[row, mask.to(device)] += delta
                replacements = self.state.replace.get((layer, "resid"))
                if replacements:
                    hidden = _apply_replacements(hidden, replacements)
                if recorder is not None:
                    recorder.resid[layer] = recorder.take(hidden)
            if logit_positions is None:
                logit_positions = torch.tensor([length - 1])
            selected = self.final_norm(hidden[:, logit_positions.to(hidden.device)])
            logprobs = torch.log_softmax(self.model.lm_head(selected).float(), dim=-1)
            if gather is not None:
                logprobs = logprobs[..., gather.to(logprobs.device)]
            return logprobs, recorder
        finally:
            self.state.reset()

    def forward(
        self,
        tokens: list[int] | tuple[int, ...],
        *,
        assistant_start: int,
        interventions: Interventions | None = None,
        capture: torch.Tensor | None = None,
        logit_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Recorder | None]:
        """One sequence: log-probs [len(logit_positions), vocab] and the record without the batch dimension."""
        logprobs, recorder = self.run(
            [list(tokens)],
            assistant_start=assistant_start,
            variants=[interventions or Interventions()],
            capture=capture,
            logit_positions=logit_positions,
        )
        return logprobs[0], None if recorder is None else recorder.select(0)

    def target_logprobs(self, prompt: Prompt, target_tokens: list[int], interventions: Interventions) -> list[float]:
        tokens = list(prompt.tokens) + target_tokens
        positions = torch.arange(len(prompt.tokens) - 1, len(tokens) - 1)
        logprobs, _ = self.forward(
            tokens, assistant_start=prompt.assistant_start, interventions=interventions, logit_positions=positions
        )
        return logprobs[torch.arange(len(target_tokens)), torch.tensor(target_tokens)].tolist()


def capture_positions(spec: str, prompt: Prompt) -> torch.Tensor:
    length = len(prompt.tokens)
    if spec == "last":
        return torch.tensor([length - 1])
    if spec == "all":
        return torch.arange(length)
    if spec == "assistant":
        return torch.arange(prompt.assistant_start, length)
    return torch.tensor(sorted({int(item) % length for item in spec.split(",")}))


def read_bytes(path: str) -> bytes:
    fs, local = filesystem_for(path)
    return fs.cat_file(local)


class Output:
    def __init__(self, root: str):
        self.fs, self.root = filesystem_for(root)
        self.url = root

    def write(self, name: str, payload: bytes) -> None:
        destination = prefix_join(self.root, name)
        self.fs.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
        self.fs.pipe_file(destination, payload)

    def write_json(self, name: str, value) -> None:
        self.write(name, json.dumps(value).encode())

    def save(self, name: str, value) -> None:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        self.write(name, buffer.getvalue())


def parse_interventions(args, prompts: dict[str, Prompt]) -> tuple[Interventions, list[PatchSpec]]:
    interventions = Interventions()
    for item in args.add_vector:
        path, layer, scale, *rest = item.split(",")
        vector = torch.load(io.BytesIO(read_bytes(path)), map_location="cpu").flatten().float()
        if vector.numel() != 2560:
            raise ValueError(f"{path}: expected 2560 values, got {vector.numel()}")
        interventions.add.append(AddVector(int(layer), vector, float(scale), rest == ["assistant"]))
    for item in args.neuron:
        layer, expert, index, scale = item.split(",")
        interventions.neurons.append(
            NeuronEdit(int(layer), None if expert == "shared" else int(expert), int(index), float(scale))
        )
    patches = []
    for item in args.patch:
        source, layer, position, *rest = item.split(",")
        if source not in prompts:
            raise ValueError(f"--patch source {source!r} is not in the prompt file")
        patches.append(PatchSpec(source, int(layer), int(position), int(rest[0]) if rest else int(position)))
    return interventions, patches


def resolve_patches(
    harness: SnowballHarness, base: Interventions, patches: list[PatchSpec], prompts: dict[str, Prompt], target: Prompt
) -> Interventions:
    if not patches:
        return base
    resolved = []
    for patch in patches:
        source = prompts[patch.source]
        source_position = patch.source_position % len(source.tokens)
        _, recorder = harness.forward(
            source.tokens, assistant_start=source.assistant_start, capture=torch.tensor([source_position])
        )
        resolved.append(
            Replace(
                patch.layer,
                "resid",
                torch.tensor([patch.position % len(target.tokens)]),
                recorder.resid[patch.layer][:1],
            )
        )
    return Interventions(base.add, base.neurons, resolved)


def top(logprobs: torch.Tensor, tokenizer, k: int = 5) -> list[dict]:
    values, ids = logprobs.topk(k)
    return [
        {"id": int(i), "token": tokenizer.decode([int(i)]), "logprob": float(v)}
        for v, i in zip(values.tolist(), ids.tolist(), strict=True)
    ]


def run_positions(harness: SnowballHarness, sequences: list[Sequence], output: Output) -> None:
    """Entry ``i`` of each tensor is the distribution over token ``i`` given tokens ``0..i-1`` (vLLM's layout)."""
    results = {}
    started = time.time()
    for number, sequence in enumerate(sequences):
        tokens = torch.tensor(sequence.tokens)
        logprobs, _ = harness.forward(sequence.tokens, assistant_start=0, logit_positions=torch.arange(len(tokens) - 1))
        top_lp, top_ids = logprobs.topk(TOP_K, dim=-1)
        actual = torch.full((len(tokens),), float("nan"))
        actual[1:] = logprobs[torch.arange(len(tokens) - 1), tokens[1:]]
        results[sequence.id] = {
            "tokens": list(sequence.tokens),
            "actual": actual,
            "top_ids": torch.cat([torch.full((1, TOP_K), -1), top_ids]).int(),
            "top_lp": torch.cat([torch.full((1, TOP_K), float("nan")), top_lp]),
        }
        if number % 50 == 0:
            logger.info("positions %d/%d %s (%.0fs)", number, len(sequences), sequence.id, time.time() - started)
    output.save("positions.pt", results)


def run_capture(harness, prompts, by_id, base, patches, args, output: Output, tokenizer) -> None:
    rows = []
    for prompt in prompts:
        interventions = resolve_patches(harness, base, patches, by_id, prompt)
        positions = capture_positions(args.positions, prompt)
        logprobs, recorder = harness.forward(
            prompt.tokens,
            assistant_start=prompt.assistant_start,
            interventions=interventions,
            capture=positions,
            logit_positions=positions,
        )
        values, ids = logprobs.topk(TOP_K)
        targets = {}
        for target in prompt.targets:
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            per_token = harness.target_logprobs(prompt, target_tokens, interventions)
            targets[target] = {"tokens": target_tokens, "logprobs": per_token, "total": sum(per_token)}
        record = {
            "id": prompt.id,
            "tokens": list(prompt.tokens),
            "assistant_start": prompt.assistant_start,
            "positions": positions,
            **recorder.stacked(),
            "top_ids": ids,
            "top_logprobs": values,
            "targets": targets,
            "interventions": interventions.describe(),
        }
        if args.full_logits:
            record["logprobs"] = logprobs
        output.save(f"capture/{prompt.id}.pt", record)
        rows.append(
            {
                "id": prompt.id,
                "length": len(prompt.tokens),
                "positions": positions.tolist(),
                "next_token_at_last": top(logprobs[-1], tokenizer),
                "targets": {name: value["total"] for name, value in targets.items()},
                "interventions": interventions.describe(),
            }
        )
        logger.info("capture %s: %s", prompt.id, json.dumps(rows[-1]["targets"]))
    output.write("capture.jsonl", "".join(json.dumps(row) + "\n" for row in rows).encode())


def run_generate(harness, prompts, by_id, base, patches, args, output: Output, tokenizer, stop_ids: set[int]) -> None:
    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    for prompt in prompts:
        interventions = resolve_patches(harness, base, patches, by_id, prompt)
        for sample in range(args.samples):
            tokens = list(prompt.tokens)
            for _ in range(args.max_new_tokens):
                logprobs, _ = harness.forward(
                    tokens, assistant_start=prompt.assistant_start, interventions=interventions
                )
                if args.temperature == 0:
                    token = int(logprobs[0].argmax())
                else:
                    token = int(torch.multinomial((logprobs[0] / args.temperature).softmax(-1), 1, generator=generator))
                tokens.append(token)
                if token in stop_ids:
                    break
            completion = tokens[len(prompt.tokens) :]
            rows.append(
                {
                    "id": prompt.id,
                    "sample": sample,
                    "completion": tokenizer.decode(completion),
                    "completion_tokens": completion,
                    "interventions": interventions.describe(),
                }
            )
            logger.info("generate %s/%d: %r", prompt.id, sample, rows[-1]["completion"][:200])
    output.write("generate.jsonl", "".join(json.dumps(row) + "\n" for row in rows).encode())


COMMANDS = ("positions", "capture", "generate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--weights", default=SNOWBALL_V01)
    parser.add_argument("--prompts", default="check", help='JSONL path, or "check" for the built-in set')
    parser.add_argument("--output", required=True, help="output directory (s3:// or local)")
    parser.add_argument("--identity", help="positions: identity.jsonl with the contrastive rows (s3:// or local)")
    parser.add_argument("--positions", default="last")
    parser.add_argument("--full-logits", action="store_true", help="capture: also save full log-probs")
    parser.add_argument("--add-vector", action="append", default=[])
    parser.add_argument("--neuron", action="append", default=[])
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def stop_tokens(weights: str) -> set[int]:
    eos = json.loads(read_bytes(prefix_join(weights, "generation_config.json")))["eos_token_id"]
    return set(eos if isinstance(eos, list) else [eos])


def load_harness(weights: str) -> SnowballHarness:
    devices = tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
    if not devices:
        raise RuntimeError("No CUDA devices visible")
    hf = hf_repo_revision(weights)
    return SnowballHarness.load(weights, (hf[1] if hf else None) or "", devices)


def run_command(harness: SnowballHarness, tokenizer, args, output: Output, stop_ids: set[int]) -> None:
    prompts = read_prompts(args.prompts, tokenizer)
    by_id = {prompt.id: prompt for prompt in prompts}
    base, patches = parse_interventions(args, by_id)
    output.write_json(f"{args.command}-args.json", {**vars(args), "torch": torch.__version__})
    if args.command == "positions":
        run_positions(harness, build_sequences(tokenizer, args.identity), output)
    elif args.command == "capture":
        run_capture(harness, prompts, by_id, base, patches, args, output, tokenizer)
    else:
        run_generate(harness, prompts, by_id, base, patches, args, output, tokenizer, stop_ids)


def main() -> None:
    args = build_parser().parse_args()
    harness = load_harness(args.weights)
    run_command(harness, load_tokenizer(args.weights), args, Output(args.output), stop_tokens(args.weights))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
