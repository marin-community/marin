from levanter.utils.py_utils import actual_sizeof


def test_actual_sizeof():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": "this is a string", "b": "this is another string"}

    assert actual_sizeof(d1) < actual_sizeof(d2)
