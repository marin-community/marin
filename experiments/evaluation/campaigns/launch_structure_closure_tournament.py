# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the tournament runner by its stable module name before Iris pickles it."""

import importlib


def main() -> None:
    runner = importlib.import_module("experiments.evaluation.campaigns.eval-policy.structure_closure_tournament")
    runner.main()


if __name__ == "__main__":
    main()
