# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


class Item:
    def __init__(self, name, quantity, price):
        self.name, self.quantity, self.price = name, quantity, price

    def increase_quantity(self, amount):
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("positive integer required")
        self.quantity += amount

    def decrease_quantity(self, amount):
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("positive integer required")
        if amount > self.quantity:
            raise ValueError("underflow")
        self.quantity -= amount


class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, item):
        if item.name in self.items:
            self.items[item.name].increase_quantity(item.quantity)
        else:
            self.items[item.name] = item

    def remove_item(self, item_name, quantity):
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("positive integer required")
        if item_name in self.items and quantity <= self.items[item_name].quantity:
            self.items[item_name].decrease_quantity(quantity)

    def get_inventory(self):
        return [dict(name=i.name, quantity=i.quantity, price=i.price) for i in self.items.values()]
