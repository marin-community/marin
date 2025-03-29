from experiments.defaults import default_train
from optimizer_sweep.Cautious.cautious import cautious_train
from optimizer_sweep.Kron.kron import kron_train
from optimizer_sweep.Lion.lion import lion_train
from optimizer_sweep.Mars.mars import mars_train
from optimizer_sweep.Muon.muon import muon_train
from optimizer_sweep.Scion.scion import scion_train
from optimizer_sweep.Soap.soap import soap_train
from optimizer_sweep.Sophia.sophia import sophia_train
from optimizer_sweep.AdamMini.adam_mini import adam_mini_train
map_tag_to_train = {
    'adamw': default_train,
    'nadamw': default_train,
    'cautious': cautious_train,
    'kron': kron_train,
    'lion': lion_train,
    'mars': mars_train,
    'muon': muon_train,
    'scion': scion_train,
    'soap': soap_train,
    'soape': soap_train,
    'sophia': sophia_train,
    'mini': adam_mini_train
}