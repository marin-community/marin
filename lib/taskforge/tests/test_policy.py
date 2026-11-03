# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A policy is its prompt files and settings by dotted key; overlays replace entries by key.

Invariants: none (pins which entries an overlay replaces and which digests it moves).
"""

import shutil

import pytest

from taskforge.defaults import DEFAULT_POLICY, DEFAULTS_DIR
from taskforge.llm.policy import LLMPolicy
from taskforge.policy import PROMPTS_DIR, load_policy, resolution_lines
from taskforge.stage50_validate.adversary import AdversaryMode, adversary_roster


def test_an_overlay_replaces_only_the_entries_it_names_and_the_digests_under_their_prefix():
    policy = DEFAULT_POLICY.overlaid({"author.nudge": "Call a tool; submit `{program}`."}, {"band.low": 0})

    assert policy.prompt("author.nudge").render(program="program.py") == "Call a tool; submit `program.py`."
    assert policy.prompt("author.system") == DEFAULT_POLICY.prompt("author.system")
    assert policy.setting("band.low", float) == 0.0
    assert policy.stage_digest("author") != DEFAULT_POLICY.stage_digest("author")
    assert policy.stage_digest("proposer") == DEFAULT_POLICY.stage_digest("proposer")
    assert policy.setting("adversary.roster", tuple) == DEFAULT_POLICY.setting("adversary.roster", tuple) == ()
    lines = {line.split()[1]: line for line in resolution_lines(DEFAULT_POLICY, policy)}
    assert lines.keys() == DEFAULT_POLICY.entries.keys()
    assert lines["band.low"] == "POLICY band.low declared=0.25 resolved=0.0"


def test_a_prompt_file_added_under_prompts_is_a_policy_entry(tmp_path):
    shutil.copytree(DEFAULTS_DIR, tmp_path, dirs_exist_ok=True)
    (tmp_path / PROMPTS_DIR / "adversary" / "off_by_one.md").write_text("Find an answer one unit off that passes.\n")

    policy = load_policy(tmp_path).overlaid({"adversary.roster": ["off_by_one"]})

    (adversary,) = adversary_roster(policy, LLMPolicy())
    assert (adversary.spec.name, adversary.spec.brief) == ("off_by_one", "Find an answer one unit off that passes.")
    assert (adversary.k, adversary.mode, adversary.prompts) == (1, AdversaryMode.BLACKBOX, policy)


@pytest.mark.parametrize(
    ("overlay", "refusal"),
    [({"author.turns": "Take {turns} turns."}, "no entry author.turns"), ({"solver.k": "8"}, "solver.k is a int")],
    ids=["unknown key", "wrong type"],
)
def test_an_overlay_naming_an_unknown_key_or_a_value_of_another_type_is_refused(overlay, refusal):
    with pytest.raises(ValueError, match=refusal):
        DEFAULT_POLICY.overlaid(overlay)
