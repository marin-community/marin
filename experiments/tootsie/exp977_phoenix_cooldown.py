"""
https://github.com/stanford-crfm/marin/issues/977

Codename: sensible-starling

This experiment is a cooldown run for the tootsie-8b model starting from adept-phoenix. It is trained on the
same mix as exp934_hq_vs_pt's best mix's full mix

We also add z-loss, since in spoonbill we found that to be very helpful
"""
from experiments.defaults import default_train
from .exp600_tootsie import (
    llama_8b_tootsie_adept_phoenix,
    llama_8b_train_config_phase4,
)

tootsie_8b_sensible_starling = dataclasses.replace(
    default_train(
        name="tootsie-8b-sensible-starling",
        tokenized=llama_8b_tootsie_adept_phoenix.tokenized,

