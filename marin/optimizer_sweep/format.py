from marin.optimizer_sweep.Adam.format import adam_train_config
from marin.optimizer_sweep.AdamMini.format import adam_mini_train_config
from marin.optimizer_sweep.Cautious.format import cautious_train_config
from marin.optimizer_sweep.Kron.format import kron_train_config
from marin.optimizer_sweep.Lion.format import lion_train_config
from marin.optimizer_sweep.Mars.format import mars_train_config
from marin.optimizer_sweep.Muon.format import muon_train_config
from marin.optimizer_sweep.Nadam.format import nadam_train_config
from marin.optimizer_sweep.Scion.format import scion_train_config
from marin.optimizer_sweep.Soap.format import soap_train_config
from marin.optimizer_sweep.Sophia.format import sophia_train_config

map_tag_to_format = {
    "adamw": adam_train_config,
    "nadamw": nadam_train_config,
    "cautious": cautious_train_config,
    "kron": kron_train_config,
    "lion": lion_train_config,
    "mars": mars_train_config,
    "muon": muon_train_config,
    "scion": scion_train_config,
    "soap": soap_train_config,
    "soape": soap_train_config,
    "sophia": sophia_train_config,
    "mini": adam_mini_train_config,
}
