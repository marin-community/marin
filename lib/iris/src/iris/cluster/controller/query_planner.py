# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite query-planner hints for controller queries."""

from sqlalchemy import func, literal_column
from sqlalchemy.sql.elements import ColumnElement

# SQLite planner hint: `tasks.state IN (<active states>)` selects ~0.5% of rows
# on a populated controller DB (~3.7k of ~773k). `sqlite_stat1` only stores the
# average rows-per-distinct-value of an index's leading column, so the planner
# estimates an active-state predicate as ~14% of the table and full-scans
# instead of driving off the small active set. Wrap such predicates with
# `hint_rare_state(...)` so the planner picks the state-driven plan.
#
# The probability argument to SQLite's `likelihood()` must be a literal constant
# in the SQL text, not a bound parameter — `literal_column` inlines it.
# See https://www.sqlite.org/lang_corefunc.html#likelihood.
_RARE_STATE_PROBABILITY = literal_column("0.005")


def hint_rare_state(predicate: ColumnElement[bool]) -> ColumnElement[bool]:
    """Wrap a `state IN (<rare states>)` predicate in SQLite's `likelihood()` hint.

    Used by scheduling-loop queries (per-tick budget spend, per-minute timeout
    enforcement) so the planner drives off the active-state index instead of
    full-scanning the tasks table.
    """
    return func.likelihood(predicate, _RARE_STATE_PROBABILITY)
