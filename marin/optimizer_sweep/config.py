from marin.optimizer_sweep.Adam.adam import adam_config
from marin.optimizer_sweep.AdamMini.adam_mini import adam_mini_config
from marin.optimizer_sweep.Cautious.cautious import cautious_config
from marin.optimizer_sweep.Kron.kron import kron_config
from marin.optimizer_sweep.Lion.lion import lion_config
from marin.optimizer_sweep.Mars.mars import mars_config
from marin.optimizer_sweep.Muon.muon import muon_config
from marin.optimizer_sweep.Nadam.nadam import nadam_config
from marin.optimizer_sweep.Scion.scion import scion_config
from marin.optimizer_sweep.Soap.soap import soap_config
from marin.optimizer_sweep.Sophia.sophia import sophia_config

map_tag_to_config = {
    "adamw": adam_config,
    "nadamw": nadam_config,
    "cautious": cautious_config,
    "kron": kron_config,
    "lion": lion_config,
    "mars": mars_config,
    "muon": muon_config,
    "scion": scion_config,
    "soap": soap_config,
    "soape": soap_config,
    "sophia": sophia_config,
    "mini": adam_mini_config,
}
