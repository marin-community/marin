from experiments.simple_train_config import SimpleTrainConfig
from optimizer_sweep.Kron.kron_config import KronTrainConfig
from optimizer_sweep.Mars.mars_config import MarsTrainConfig
from optimizer_sweep.Muon.muon_config import MuonTrainConfig
from optimizer_sweep.Scion.scion_config import ScionTrainConfig
from optimizer_sweep.Soap.soap_config import SoapTrainConfig
from optimizer_sweep.Sophia.sophia_config import SophiaTrainConfig
from optimizer_sweep.AdamMini.adam_mini_config import AdamMiniTrainConfig
map_tag_to_config = {
    'adamw': SimpleTrainConfig,
    'nadamw': SimpleTrainConfig,
    'cautious': SimpleTrainConfig,
    'kron': KronTrainConfig,
    'lion': SimpleTrainConfig,
    'mars': MarsTrainConfig,
    'muon': MuonTrainConfig,
    'scion': ScionTrainConfig,
    'soap': SoapTrainConfig,
    'soape': SoapTrainConfig,
    'sophia': SophiaTrainConfig,
    'mini': AdamMiniTrainConfig
}