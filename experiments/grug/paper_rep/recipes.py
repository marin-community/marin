# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Base-tuned recipes for the arXiv 2609.19107 replication (paper Table 5).

Each recipe is split into the model-side knobs (WTE, UIS, RM, OM) and the
optimizer-side knobs (GLR, ELRM, HLRM, WD, WU, WDR, Adam betas/eps), all
selected by the paper's two rounds of chain tuning at the base size on 1B
tokens. The transfer probe (paper Table 6) trains the Operator-1 model under
the Vanilla recipe.
"""

from dataclasses import dataclass

from experiments.grug.paper_rep.model import PAPER_VOCAB_SIZE, GrugModelConfig, split_prelude_core_coda
from experiments.grug.paper_rep.optimizer import PaperMuonConfig


@dataclass(frozen=True)
class PaperRecipe:
    """Paper Table 5 recipe: model-side and optimizer-side knobs."""

    # Model side.
    rm: float
    om: float
    wte: float
    uis: float

    # Optimizer side.
    glr: float
    elrm: float
    hlrm: float
    wd: float
    wu: int
    wdr: float
    beta1: float
    beta2: float
    beta2_eps: float = 1e-10


VANILLA_RECIPE = PaperRecipe(
    rm=0.25,
    om=0.5,
    wte=0.007,
    uis=0.063,
    glr=0.04,
    elrm=0.453,
    hlrm=0.113,
    wd=0.071,
    wu=40,
    wdr=0.6,
    beta1=0.8,
    beta2=0.95,
)

OPERATOR1_RECIPE = PaperRecipe(
    rm=0.5,
    om=1.0,
    wte=0.113,
    uis=0.354,
    glr=0.04,
    elrm=0.905,
    hlrm=0.08,
    wd=0.05,
    wu=0,
    wdr=0.8,
    beta1=0.8,
    beta2=0.98,
)


def model_config(
    recipe: PaperRecipe,
    *,
    num_layers: int,
    vocab_size: int = PAPER_VOCAB_SIZE,
    max_seq_len: int = 2048,
    boundary_operator: bool = False,
    injection_scale: float = 1.0,
    width: int | None = None,
) -> GrugModelConfig:
    """Build the model config for a recipe at a given depth.

    Paper A.1: width = 128 * depth (d8 -> 1024), d_ff = 3 * width (param
    matching: ~210M params at d8, ~123M at d6, with a 50,304 vocab). The
    prelude/core/coda split follows paper Table 2 when the boundary operator
    is on. ``injection_scale`` is the paper's alpha_emb (1 for Operator-1).
    """
    hidden_dim = width if width is not None else 128 * num_layers
    return GrugModelConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=3 * hidden_dim,
        num_layers=num_layers,
        num_heads=hidden_dim // 64,
        num_kv_heads=hidden_dim // 64,
        max_seq_len=max_seq_len,
        embed_init_std=recipe.wte,
        input_init_scale=recipe.uis,
        residual_multiplier=recipe.rm,
        output_multiplier=recipe.om,
        boundary_operator=boundary_operator,
        **(
            dict(
                prelude_len=split_prelude_core_coda(num_layers).prelude,
                coda_len=split_prelude_core_coda(num_layers).coda,
                injection_scale=injection_scale,
            )
            if boundary_operator
            else {}
        ),
    )


def optimizer_config(recipe: PaperRecipe) -> PaperMuonConfig:
    """Build the optimizer config for a recipe.

    The LR schedule is the paper's: warmup for WU steps, then a linear warmdown
    over the last WDR fraction of training, decaying to zero.
    """
    return PaperMuonConfig(
        learning_rate=recipe.glr,
        embed_lr_multiplier=recipe.elrm,
        head_lr_multiplier=recipe.hlrm,
        weight_decay=recipe.wd,
        beta1=recipe.beta1,
        beta2=recipe.beta2,
        epsilon=recipe.beta2_eps,
        lr_schedule="linear",
        warmup=recipe.wu,
        decay=recipe.wdr,
        min_lr_ratio=0.0,
    )


__all__ = [
    "OPERATOR1_RECIPE",
    "VANILLA_RECIPE",
    "PaperRecipe",
    "model_config",
    "optimizer_config",
]
