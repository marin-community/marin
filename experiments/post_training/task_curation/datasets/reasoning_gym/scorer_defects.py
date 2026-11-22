# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym 0.1.25 tasks whose scorer gives full reward to a constant reply.

Reject these rows at conversion so generated source indexes stay stable and each rejection is recorded.
"""

CONSTANT_REPLY_TASKS = {
    "coin_flip": "The reply 0 earns the full reward on every entry whose answer is below 1",
    "complex_arithmetic": "The reply NaN earns the full reward on every entry",
    "game_of_life_halting": "Every nonempty reply earns the full reward",
    "propositional_logic": "A disjunction that names a premise earns the full reward on most entries",
    "puzzle24": "The scorer does not read the entry's numbers: 1*2*3*4 earns the full reward on every entry",
}
