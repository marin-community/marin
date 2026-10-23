# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task data and expert traces (WP7): the X and R task families and the scripts that
turn strong-teacher traces into expert training data.

- ``code_x`` synthesizes CruxEval-style Python prediction tasks, graded by sandboxed
  execution. No public CruxEval prompt may enter the set.
- ``math_r`` draws long-thinking competition-math problems with checkable final
  answers, excluding the AIME/HMMT/OlympiadBench pools reserved for ``P_think``.
- ``expert_traces`` collects GLM traces through the relay for building ``E_X``/``E_R``
  (R traces keep their thinking and drop — never truncate — over-cap rows).
- ``expert_outputs`` samples graded expert completions and writes them as tokenized
  fixed-completion rows for the M1 datasets.
- ``build`` is the command line over all four.
"""
