from levanter.distributed import _square_brace_expand


def test_square_brace_expand():
    custom_sequence = "node[001-004,007]suffix"
    expanded_nodes = _square_brace_expand(custom_sequence)
    assert expanded_nodes == ["node001suffix", "node002suffix", "node003suffix", "node004suffix", "node007suffix"]

    custom_sequence_2 = "prefix[001-002]node[005-006]suffix"
    expanded_nodes_2 = _square_brace_expand(custom_sequence_2)
    assert expanded_nodes_2 == [
        "prefix001node005suffix",
        "prefix001node006suffix",
        "prefix002node005suffix",
        "prefix002node006suffix",
    ]

    custom_sequence_3 = "node[1-11]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)]

    custom_sequence_3 = "node[1-11,21]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)] + ["node21suffix"]
