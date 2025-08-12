import equinox as eqx
import jax.random as jrandom

import haliax as hax

from levanter.models.gpt2 import Gpt2Config
from levanter.optim import AdamConfig


def test_weight_decay_masking():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    string_list_config = AdamConfig(
        weight_decay_modules=[
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
            "token_embeddings.weight",
            "position_embeddings.weight",
        ]
    )
    regex_config = AdamConfig(
        weight_decay_modules=r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings",
    )
    # masking using `equinox.tree_at`
    true_mask = _tree_at_mask(model)
    # masking using list of module path
    list_string_mask = string_list_config.build_weight_decay_mask()(model)

    regex_mask = regex_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(list_string_mask, true_mask)
    assert eqx.tree_equal(regex_mask, true_mask)


def test_weight_decay_masking_with_class_names():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    string_list_config = AdamConfig(
        weight_decay_modules=[
            "Linear.weight",
            "Embedding.weight",
        ]
    )

    # masking using `equinox.tree_at`
    true_mask = _tree_at_mask(model)
    # masking using list of module path
    list_string_mask = string_list_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(list_string_mask, true_mask)


def test_default_weight_decay_masking():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    default_config = AdamConfig(default_weight_decay_mask=True)
    no_default_config = AdamConfig(default_weight_decay_mask=False)
    equivalent_default_config = AdamConfig(weight_decay_modules=None, default_weight_decay_mask=None)

    no_default_mask = no_default_config.build_weight_decay_mask()
    assert no_default_mask is None

    # masking using `equinox.tree_at`
    tree_at_mask = _tree_at_mask(model, decay_embeddings=False)
    # masking using list of module path
    default_mask = default_config.build_weight_decay_mask()(model)
    equivalent_default_tree = equivalent_default_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(default_mask, equivalent_default_tree)
    assert eqx.tree_equal(default_mask, tree_at_mask)


def _tree_at_mask(params, decay_embeddings=True):
    # let's mask all leaves as False
    params = hax.tree_util.tree_map(lambda _: False, params)

    def apply_weight_decay(tree):
        # there is no weight decay performed in LayerNorms and bias
        nodes = []

        # apply on embedding
        if decay_embeddings:
            nodes.append(tree.embeddings.token_embeddings.weight)
            nodes.append(tree.embeddings.position_embeddings.weight)

        # apply on attention
        nodes.append(tree.transformer.blocks.stacked.attn.c_attn.weight)
        nodes.append(tree.transformer.blocks.stacked.attn.c_proj.weight)

        # apply on MLP
        nodes.append(tree.transformer.blocks.stacked.mlp.c_fc.weight)
        nodes.append(tree.transformer.blocks.stacked.mlp.c_proj.weight)

        return nodes

    # apply weight decay when necessary
    params = eqx.tree_at(
        where=apply_weight_decay,
        pytree=params,
        replace_fn=lambda _: True,
    )

    return params
