# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the reproduced inventory and calculator test contracts."""

import hashlib

INVENTORY_TEST_SHA = "2a94ef5bcc42f81330ae1cd2f26c759b333880df34df977560c8cb07daa48115"
CALCULATOR_TEST_SHA = "698f27b8cce36580a6c8108e3af3d773d6d4b4e1d5fc07d6128ef3c76bee90a6"
INVENTORY_UNDERFLOW = "    item.decrease_quantity(5)  # Attempt to decrease below zero"
INVENTORY_ALLOWED_UNDERFLOW = """    try:
        item.decrease_quantity(5)
    except ValueError:
        pass  # Rejecting an underflow is allowed if the quantity stays unchanged.
""".rstrip()
INVENTORY_TESTS = """

@pytest.mark.parametrize('amount', [0, -1, 1.5, '2'])
def test_invalid_inventory_quantities_preserve_state(amount):
    item = Item(name='Apple', quantity=10, price=0.5)
    inventory = Inventory()
    inventory.add_item(item)
    with pytest.raises((ValueError, TypeError)):
        item.increase_quantity(amount)
    assert item.quantity == 10
    with pytest.raises((ValueError, TypeError)):
        inventory.remove_item('Apple', amount)
    assert inventory.get_inventory()[0]['quantity'] == 10


def test_item_underflow_preserves_positive_quantity():
    item = Item(name='Apple', quantity=3, price=0.5)
    try:
        item.decrease_quantity(4)
    except ValueError:
        pass
    assert item.quantity == 3
"""
CALCULATOR_INTERFACE = "\n\nPlace the `Calculator` class in `/app/calculator.py`.\n"
CALCULATOR_TESTS = """

@pytest.mark.parametrize('method,a,b,expected', [
    ('add', -1.25, 0.5, -0.75),
    ('subtract', 0.5, 1.25, -0.75),
    ('multiply', -1.5, 2.5, -3.75),
    ('multiply', 4, 0, 0.0),
    ('divide', 7.5, 2.5, 3.0),
    ('divide', 1, 4, 0.25),
])
def test_calculator_operations_return_floats(method, a, b, expected):
    result = getattr(Calculator(), method)(a, b)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


def test_calculator_methods_have_documentation():
    for name in ['add', 'subtract', 'multiply', 'divide']:
        assert getattr(Calculator, name).__doc__


def test_calculator_division_error_message():
    with pytest.raises(ValueError) as error:
        Calculator().divide(1.0, 0.0)
    assert str(error.value) == 'Cannot divide by zero.'
"""


def repair_contract(test: bytes, instruction: str) -> tuple[bytes, str]:
    """Repair only the exact test fixtures with their corresponding task instructions."""
    digest = hashlib.sha256(test).hexdigest()
    if digest == INVENTORY_TEST_SHA and "inventory.py" in instruction:
        original = test.decode()
        assert INVENTORY_UNDERFLOW in original
        return (
            (original.replace(INVENTORY_UNDERFLOW, INVENTORY_ALLOWED_UNDERFLOW) + INVENTORY_TESTS).encode(),
            instruction,
        )
    if digest == CALCULATOR_TEST_SHA and "class called `Calculator`" in instruction:
        return test + CALCULATOR_TESTS.encode(), instruction + CALCULATOR_INTERFACE
    return test, instruction
