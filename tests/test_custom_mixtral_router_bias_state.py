import jax

import haliax as hax
from haliax.quantization import apply_updates

from experiments.speedrun import custom_mixtral as custom_mixtral_module
from experiments.speedrun.custom_mixtral import CustomMixtralConfig, MixtralLMHeadModel, RouterBiasState


def test_router_bias_overwrite_tree_only_updates_router_bias():
    config = CustomMixtralConfig(
        seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        equilibrium_lb_loss_scale=1.0,
        equilibrium_lb_objective="article_original",
    )
    vocab = hax.Axis("vocab", 32)
    model = MixtralLMHeadModel.init(vocab, config, key=jax.random.PRNGKey(0))
    router_bias_overwrite = hax.zeros((config.Layers, config.Experts))

    updates = jax.tree_util.tree_map(
        lambda _: None,
        model,
        is_leaf=lambda x: isinstance(x, (hax.NamedArray, custom_mixtral_module.hq.OverwriteWithGradient)),
    )
    overwrites = model._router_bias_overwrite_tree(router_bias_overwrite)
    updated = apply_updates(model, updates, overwrites)

    assert updated.embeddings.token_embeddings.weight.array is not None
    assert hax.all(updated.embeddings.token_embeddings.weight == model.embeddings.token_embeddings.weight).scalar()
    assert isinstance(updated.transformer.layers.stacked.block_sparse_moe.router_bias, RouterBiasState)
    assert updated.transformer.layers.stacked.block_sparse_moe.router_bias.bias.axes == (
        config.Layers,
        config.Experts,
    )
    assert (updated.transformer.layers.stacked.block_sparse_moe.router_bias.bias.array == 0).all()
