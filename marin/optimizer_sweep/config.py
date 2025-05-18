from experiments.simple_train_config import SimpleTrainConfig
from marin.optimizer_sweep.AdamMini.adam_mini_config import AdamMiniTrainConfig
from marin.optimizer_sweep.Kron.kron_config import KronTrainConfig
from marin.optimizer_sweep.Mars.mars_config import MarsTrainConfig
from marin.optimizer_sweep.Muon.muon_config import MuonTrainConfig
from marin.optimizer_sweep.Scion.scion_config import ScionTrainConfig
from marin.optimizer_sweep.Soap.soap_config import SoapTrainConfig
from marin.optimizer_sweep.Sophia.sophia_config import SophiaTrainConfig

map_tag_to_config = {
    "adamw": SimpleTrainConfig,
    "nadamw": SimpleTrainConfig,
    "cautious": SimpleTrainConfig,
    "kron": KronTrainConfig,
    "lion": SimpleTrainConfig,
    "mars": MarsTrainConfig,
    "muon": MuonTrainConfig,
    "scion": ScionTrainConfig,
    "soap": SoapTrainConfig,
    "soape": SoapTrainConfig,
    "sophia": SophiaTrainConfig,
    "mini": AdamMiniTrainConfig,
}
