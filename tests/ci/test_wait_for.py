# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The comment-noise filter, exercised on bodies taken verbatim from marin PR history.

The bodies below are real: the review bots' in-progress placeholders, their verdicts, their
findings, and the human comments that must never be filtered. Trimmed only for length.
"""

import pytest

from scripts.ci import wait_for
from scripts.ci.wait_for import CLAUDE_BOT, CODEX_BOT, CommentFilter, GhRecord, Significance

HUMAN = "rjpower"
CODEQL_BOT = "github-advanced-security[bot]"

SPINNER = (
    '<img src="https://github.com/user-attachments/assets/5ac382c7-e004-429b-8e35-7feb3e8f9c6f" '
    'width="14px" height="14px" style="vertical-align: middle; margin-left: 4px;" />'
)

# The placeholder claude[bot] posts the instant a PR opens, then edits in place into the review.
# Its heading tracks whatever prompt the bot is running, so the wording varies between runs.
CLAUDE_WORKING = f"""Claude Code is working… {SPINNER}

I'll analyze this and get back to you.

[View job run](https://github.com/marin-community/marin/actions/runs/29212065256)"""

# A heading, unchecked boxes, and a job link, carrying no findings of its own.
CLAUDE_PROGRESS_CHECKLIST = f"""### Code review in progress {SPINNER}

- [ ] Gather PR context and diff
- [ ] Check PR description against body rules
- [ ] Run multi-agent correctness review (bugs + CLAUDE.md)
- [ ] Validate findings
- [ ] Post review

[View job run](https://github.com/marin-community/marin/actions/runs/29212065256)"""

CLAUDE_PROGRESS_RUNNING = f"""### PR review: cross-cluster log forwarding from finelog

Running a multi-agent correctness review.

- [ ] Gather context (diff, AGENTS.md/CLAUDE.md, PR description check)
- [ ] Parallel compliance + bug review
- [ ] Validate findings
- [ ] Post review

{SPINNER}

[View job run](https://github.com/marin-community/marin/actions/runs/29104990192)"""

CLAUDE_CLEAN = """**Claude finished @rjpower's task in 2m 37s** —— [View job](https://github.com/marin-community/marin/actions/runs/29212065256)

---
### Code review

No issues found. Checked for bugs and CLAUDE.md compliance.

I traced the reconstruction change end to end: all call sites are updated for the new required
`workdir_files` kwarg, and `container_profile` round-trips safely."""

# The bots qualify the noun they cleared, so the verdict is not the bare phrase "no issues".
CLAUDE_CLEAN_QUALIFIED = """### Code review

- [x] Gather context (diff, AGENTS.md/TESTING.md, PR description)
- [x] Multi-agent correctness + compliance review

**No code issues found.** Checked for bugs and CLAUDE.md/AGENTS.md/TESTING.md compliance
across all 4 files. The gate exemption is placed safely and the tests assert observable
behavior rather than internals."""

# A qualified verdict clears one axis only. Here correctness is clear but the compliance pass
# reports dead code, so the comment is actionable despite leading with "No correctness bugs".
CLAUDE_SUBSET_CLEAN_WITH_FINDING = """### Code review

- [x] Multi-agent correctness + compliance review

**No correctness bugs found.** The logit-mixing math is sound (verified `_mix_top_logprobs`
against the `pytest.approx` values in `test_logit_mixing.py`).

#### Findings (2 — both minor dead code, AGENTS.md *"Delete dead code: unused parameters"*)

1. **`logit_mixing.py:513`** — `_unused_scratch` is never read.
2. **`worker.py:88`** — the `legacy_mode` parameter is dead."""

# A completed checklist that carries a real bug: checkbox scaffolding must not suppress it.
CLAUDE_FINDING = """**Claude finished @rjpower's task in 3m 39s** —— [View job](https://github.com/marin-community/marin/actions/runs/28803285934)

---
### Code Review — PR #6982

- [x] Gather PR context & diff
- [x] Multi-agent correctness + CLAUDE.md review
- [x] Post review

**One finding** (posted inline):

- **`_read_targets_from_stdin` crashes on an empty first CSV field** — `job.py:119`.
  `line.split(",", 1)[0].split()[0]` raises `IndexError` when the first column is empty."""

CLAUDE_ERROR = """**Claude encountered an error after 0s** —— [View job](https://github.com/marin-community/marin/actions/runs/28903845207)

---
I'll analyze this and get back to you."""

CODEX_WRAPPER = """
### 💡 Codex Review

Here are some automated review suggestions for this pull request.

**Reviewed commit:** `0c4d105124`

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>
Codex has been enabled on this repository.
</details>"""

CODEX_FINDING = """**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Replace PR-head vLLM SHA before landing**

This change pins the vLLM fork to the SHA from `marin-community/vllm#7`, but the protocol above
requires PR-head SHAs to stay temporary and be replaced by the landed fork `main` SHA."""

CODEQL_FINDING = """## CodeQL / Clear-text logging of sensitive information

This expression logs sensitive data (secret) as clear text."""

# A human enumerating required fixes as an all-unchecked task list: structurally a checklist,
# but every box is work the agent must do.
HUMAN_FIX_LIST = """Before this can land:

- [ ] Drop the `hasattr` compat shim
- [ ] Move the constant to the top of the module
- [ ] Add a regression test for the empty-CSV case"""

CATALOG: list[tuple[str, str, str, Significance]] = [
    # name, author, body, expected
    ("claude working placeholder", CLAUDE_BOT, CLAUDE_WORKING, Significance.PROGRESS),
    ("claude progress checklist", CLAUDE_BOT, CLAUDE_PROGRESS_CHECKLIST, Significance.PROGRESS),
    ("claude progress prose line", CLAUDE_BOT, CLAUDE_PROGRESS_RUNNING, Significance.PROGRESS),
    ("claude clean verdict", CLAUDE_BOT, CLAUDE_CLEAN, Significance.CLEAN),
    ("claude qualified clean verdict", CLAUDE_BOT, CLAUDE_CLEAN_QUALIFIED, Significance.CLEAN),
    ("claude subset-clean with a finding", CLAUDE_BOT, CLAUDE_SUBSET_CLEAN_WITH_FINDING, Significance.CONCERN),
    ("claude review with a finding", CLAUDE_BOT, CLAUDE_FINDING, Significance.CONCERN),
    ("claude job failure", CLAUDE_BOT, CLAUDE_ERROR, Significance.CONCERN),
    ("codex review wrapper", CODEX_BOT, CODEX_WRAPPER, Significance.WRAPPER),
    ("codex inline finding", CODEX_BOT, CODEX_FINDING, Significance.CONCERN),
    ("codeql alert", CODEQL_BOT, CODEQL_FINDING, Significance.CONCERN),
    ("human fix list", HUMAN, HUMAN_FIX_LIST, Significance.CONCERN),
    ("human approval", HUMAN, "lgtm, ship it", Significance.CONCERN),
    ("human one-liner", HUMAN, "this drops the retry on 429s", Significance.CONCERN),
    ("empty body", HUMAN, "", Significance.WRAPPER),
]


@pytest.mark.parametrize(
    ("author", "body", "expected"), [(a, b, e) for _, a, b, e in CATALOG], ids=[c[0] for c in CATALOG]
)
def test_catalog_bodies_classify_as_expected(author: str, body: str, expected: Significance) -> None:
    assert wait_for.classify_significance(body, author) is expected


@pytest.mark.parametrize("body", [CLAUDE_PROGRESS_CHECKLIST, CLAUDE_WORKING, CLAUDE_CLEAN])
@pytest.mark.parametrize("author", [HUMAN, "some-new-bot[bot]"])
def test_mundane_bot_shapes_still_wake_when_anyone_else_posts_them(author: str, body: str) -> None:
    """Rules apply only to the automation they name: an uncatalogued author always wakes the agent."""
    assert wait_for.classify_significance(body, author) is Significance.CONCERN


def _comment(comment_id: int, author: str, body: str) -> GhRecord:
    return GhRecord(id=comment_id, author=author, body=body, url="u", state=None, kind="issue_comment")


class _ScriptedComments:
    """Stands in for `gh_api_list`: one list of issue comments per polling round."""

    def __init__(self, rounds: list[list[GhRecord]]):
        self.rounds = rounds
        self.round = -1

    def __call__(self, repo: str, path: str, *, kind: str) -> list[GhRecord]:
        if kind != "issue_comment":  # the review-comment endpoint stays empty
            return []
        self.round = min(self.round + 1, len(self.rounds) - 1)
        return self.rounds[self.round]


def _comment_source(
    monkeypatch: pytest.MonkeyPatch,
    rounds: list[list[GhRecord]],
    comment_filter: CommentFilter = CommentFilter.SIGNIFICANT,
) -> wait_for.CommentSource:
    monkeypatch.setattr(wait_for, "gh_api_list", _ScriptedComments(rounds))
    spec = wait_for.parse_spec("github.pr_comment 7138")
    return wait_for.CommentSource(spec, "marin-community/marin", set(), comment_filter)


def test_placeholder_does_not_fire_but_the_review_it_becomes_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claude[bot] comment is edited in place from a placeholder into the review.

    Suppressing the placeholder must not cost us the review: the arm keeps polling, and fires
    once the same comment id carries real content.
    """
    placeholder = _comment(1, CLAUDE_BOT, CLAUDE_PROGRESS_CHECKLIST)
    review = _comment(1, CLAUDE_BOT, CLAUDE_FINDING)
    source = _comment_source(monkeypatch, [[], [placeholder], [placeholder], [review]])

    assert source.check() is None  # baseline snapshot
    assert source.check() is None  # placeholder posted
    assert source.check() is None  # unchanged placeholder

    fired = source.check()
    assert fired is not None
    assert fired["comments"] == [
        {
            "author": CLAUDE_BOT,
            "body": CLAUDE_FINDING,
            "url": "u",
            "kind": "issue_comment",
            "significance": "concern",
        }
    ]


def test_a_suppressed_comment_never_fires_once_a_later_one_wakes_the_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comment ruled mundane is absorbed, so it does not ride along when a real one arrives."""
    placeholder = _comment(1, CLAUDE_BOT, CLAUDE_PROGRESS_CHECKLIST)
    human = _comment(2, HUMAN, "this drops the retry on 429s")
    source = _comment_source(monkeypatch, [[], [placeholder], [placeholder, human]])

    assert source.check() is None
    assert source.check() is None
    fired = source.check()
    assert fired is not None
    assert [c["author"] for c in fired["comments"]] == [HUMAN]


def test_comment_filter_all_fires_on_a_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    placeholder = _comment(1, CLAUDE_BOT, CLAUDE_PROGRESS_CHECKLIST)
    source = _comment_source(monkeypatch, [[], [placeholder]], CommentFilter.ALL)

    assert source.check() is None
    fired = source.check()
    assert fired is not None
    assert fired["comments"][0]["significance"] == "progress"


def _review(review_id: int, author: str, body: str, state: str) -> GhRecord:
    return GhRecord(id=review_id, author=author, body=body, url="u", state=state, kind="review")


def _review_source(monkeypatch: pytest.MonkeyPatch, rounds: list[list[GhRecord]]) -> wait_for.ReviewSource:
    scripted = _ScriptedComments(rounds)
    monkeypatch.setattr(wait_for, "gh_api_list", lambda repo, path, *, kind: scripted(repo, path, kind="issue_comment"))
    spec = wait_for.parse_spec("github.review 7138")
    return wait_for.ReviewSource(spec, "marin-community/marin", set(), CommentFilter.SIGNIFICANT)


def test_changes_requested_fires_even_with_a_body_that_reads_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The merge-deciding state is the signal; the body never overrides it."""
    review = _review(1, HUMAN, "lgtm apart from one thing", "CHANGES_REQUESTED")
    source = _review_source(monkeypatch, [[], [review]])

    assert source.check() is None
    fired = source.check()
    assert fired is not None
    assert fired["reviews"] == [{"author": HUMAN, "state": "CHANGES_REQUESTED", "url": "u"}]


def test_codex_review_wrapper_does_not_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex's top-level body is a container; its findings arrive as inline comments that fire."""
    source = _review_source(monkeypatch, [[], [_review(1, CODEX_BOT, CODEX_WRAPPER, "COMMENTED")]])

    assert source.check() is None
    assert source.check() is None
