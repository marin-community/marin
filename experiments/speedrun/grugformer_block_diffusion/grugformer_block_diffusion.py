"""Grugformer block diffusion speedrun entrypoint (learning experiment).

This is the main place you'll tweak hyperparameters and wire the objective into
Marin's speedrun harness.

Run:
  python marin/run/ray_run.py -- \
    python -m experiments.speedrun.grugformer_block_diffusion.grugformer_block_diffusion
"""

# nodryrun

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from levanter.grug.model import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.grugformer_block_diffusion.objective import BlockDiffusionObjectiveConfig, GrugBlockDiffusionWrapper

logger = logging.getLogger("ray")


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        return {
            "130m": ResourceConfig.with_tpu("v5p-8"),
        }
    return {
        "130m": ResourceConfig.with_gpu("A100-80G", count=1),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128}


@LmConfig.register_subclass("grugformer_block_diffusion")
@dataclass(frozen=True)
class GrugformerBlockDiffusionConfig(LmConfig[GrugBlockDiffusionWrapper]):
    """LmConfig wrapper around grug core hyperparameters + diffusion objective settings."""

    # LmConfig field
    max_seq_len: int = 2048

    # Grug core hyperparams (keep small while iterating)
    hidden_dim: int = 512
    intermediate_dim: int = 1792
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int | None = None

    # Diffusion objective hyperparams
    block_size: int = 128
    num_denoise_steps: int = 8
    mask_token_id: int = 0

    @property
    def model_type(self) -> type[GrugBlockDiffusionWrapper]:
        return GrugBlockDiffusionWrapper

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugBlockDiffusionWrapper:
        grug_cfg = GrugModelConfig(
            vocab_size=Vocab.size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        grug = GrugWrapper.init(Vocab, grug_cfg, key=key)
        obj_cfg = BlockDiffusionObjectiveConfig(
            block_size=self.block_size,
            num_denoise_steps=self.num_denoise_steps,
            mask_token_id=self.mask_token_id,
        )
        return GrugBlockDiffusionWrapper(grug=grug, obj_cfg=obj_cfg)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        return int(transformer + 2 * token_embedding)


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    if size != "130m":
        raise ValueError(f"Unknown size: {size}")

    model_cfg = GrugformerBlockDiffusionConfig()
    batch = _batch_sizes()[size]
    steps = _get_num_train_steps(int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size)), batch, model_cfg.max_seq_len)
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources,
        train_seq_len=model_cfg.max_seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    )

    run_name = f"grugformer_block_diffusion_{size}"
    desc = f"Grugformer block diffusion ({size})."
    cfg = SpeedrunConfig(
        author=Author(
            name="__YOUR_NAME__",
            affiliation="__YOUR_AFFILIATION__",
            url="__YOUR_URL__",
        ),
        description=desc,
        model_config=model_cfg,
        train_config=train,
    )
    return run_name, cfg


def main() -> None:
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))

    steps = []
    name, cfg = build_run("130m", use_tpu=use_tpu)
    if cfg.vocab_size != llama3_tokenizer_vocab_size:
        raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
    cfg.print_run_info()
    steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Grugformer block diffusion (learning experiment).")


if __name__ == "__main__":
    main()
