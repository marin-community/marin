import pytest

from skyrl_train.models.grug_context_parallel import validate_cp_attention


def test_snowball_cp4_rejects_incompatible_attention_communication():
    shape = dict(query_heads=20, kv_heads=5, tensor_parallel=1, context_parallel=4)
    with pytest.raises(ValueError):
        validate_cp_attention(**shape, communication="a2a", sliding_window=True)
    for communication in ("p2p", "a2a+p2p"):
        with pytest.raises(ValueError):
            validate_cp_attention(**shape, communication=communication, sliding_window=True)
    modes = ["all_gather", "all_gather", "p2p"]
    for is_long, communication in zip((False, False, True), modes, strict=True):
        validate_cp_attention(**shape, communication=communication, sliding_window=not is_long)
