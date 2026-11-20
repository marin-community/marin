# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Let latex2sympy2_extended 1.11.0 convert LaTeX percentages under antlr 4.9.3 and 4.11.

latex2sympy2_extended ships a parser generated for each of antlr 4.9.3, 4.11 and 4.13.2 and loads
the one that matches the installed runtime. Its 1.11.0 converter checks ``atom.FUNC_GAMMA()``
before ``atom.PERCENT_NUMBER()``, but only the 4.13.2 parser was regenerated with that accessor.
Under antlr 4.11, which sympy's ``parse_latex`` requires, every number followed by ``\%`` raises
``AttributeError``, and math-verify falls back to the raw text without a warning.

Importing this module gives the loaded parser the accessor antlr 4.13.2 generates. The older
grammar has no ``FUNC_GAMMA`` alternative in ``atom``, so the accessor returns ``None`` and the
converter reaches its percent branch. The module changes nothing when the parser already has the
accessor; delete it and its imports once a latex2sympy2_extended release regenerates the older
parsers. Import it in every module that parses LaTeX with math-verify, so subprocesses that import
only that module are covered too.
"""

from latex2sympy2_extended.antlr_parser import PSParser

if not hasattr(PSParser.AtomContext, "FUNC_GAMMA"):
    PSParser.AtomContext.FUNC_GAMMA = lambda self: self.getToken(PSParser.FUNC_GAMMA, 0)
