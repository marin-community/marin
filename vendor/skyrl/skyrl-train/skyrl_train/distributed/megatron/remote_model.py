"""Megatron-Bridge adapter for range-backed Hugging Face weights."""

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.state import StateDict, StateSource
from megatron.bridge.utils.common_utils import extract_expert_number_from_param

from skyrl_train.distributed.megatron.remote_weight_tasks import prefetch_items_from_tasks
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore, lazy_first_dim_patterns_for_bridge


class RemoteSafetensorsStateSource(StateSource):
    """Expose an object-store tensor store through Megatron-Bridge's lazy API."""

    def __init__(
        self,
        source_uri: str,
        metadata_dir: str,
        *,
        lazy_first_dim_patterns: tuple[str, ...] = (),
    ) -> None:
        self.store = RemoteSafetensorsTensorStore(
            source_uri,
            metadata_dir,
            lazy_first_dim_patterns=lazy_first_dim_patterns,
        )

    def get_all_keys(self) -> list[str]:
        return self.store.get_all_keys()

    def load_tensors(self, keys: list[str]):
        return self.store.load_tensors(keys)


def install_remote_hf_state(bridge, source_uri: str, metadata_dir: str) -> RemoteSafetensorsStateSource:
    """Replace Bridge's local safetensors accessor before its model hook is built."""
    source = RemoteSafetensorsStateSource(
        source_uri,
        metadata_dir,
        lazy_first_dim_patterns=lazy_first_dim_patterns_for_bridge(bridge),
    )
    bridge.hf_pretrained._state_dict_accessor = StateDict(source)
    return source


def make_prefetch_hook(bridge: AutoBridge, store: RemoteSafetensorsTensorStore):
    """Plan rank-owned weight ranges before Bridge imports the model's weights."""

    def prefetch(models):
        store.plan_prefetch(
            prefetch_items_from_tasks(
                bridge.get_conversion_tasks(models),
                store.lazy_first_dim_keys,
                extract_expert_number_from_param,
            )
        )
        return models

    return prefetch
