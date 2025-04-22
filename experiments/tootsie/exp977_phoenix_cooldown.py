"""
https://github.com/stanford-crfm/marin/issues/977

Codename: sensible-starling

This experiment is a cooldown run for the tootsie-8b model starting from adept-phoenix. It is trained on the
same mix as exp934
"""
from experiments.defaults import default_train
from .exp600_tootsie import (
    PHASE_3_END,
    llama_8b_tootsie_phase3,
    llama_8b_train_config_phase3,
    phase_3_data_mixture,
    llama_8b_tootsie_adept_phoenix
)

# this is another major cooldown akin to monumental-jellyfish
