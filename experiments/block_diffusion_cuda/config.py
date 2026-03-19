from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Variant = Literal["baseline", "bigdn"]
DatasetKind = Literal["toy", "text", "hf"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    mask_token_id: int = 50_257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    block_size: int = 128
    window_blocks: int = 8
    diffusion_steps: int = 8
    variant: Variant = "baseline"
    attention_period: int = 4
    gdn_heads: int = 12
    timestep_embed_dim: int = 256

    @property
    def max_seq_len(self) -> int:
        return self.block_size * self.window_blocks

    @property
    def max_context_tokens(self) -> int:
        return self.max_seq_len - self.block_size


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetKind = "toy"
    tokenizer: str = "gpt2"
    text_file: str | None = None
    hf_dataset: str | None = None
    hf_dataset_config: str | None = None
    hf_split: str = "train"
    hf_text_field: str = "text"
    max_examples: int | None = None
    toy_size: int = 4096
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    out_dir: Path = Path("runs/block_diffusion_cuda")
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    steps: int = 1_000
    log_every: int = 10
    eval_every: int = 0
    save_every: int = 200
    sample_every: int = 0
    num_workers: int = 2
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    compile: bool = True
    bf16: bool = True
    ddp: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
