# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ci.belay_intake import (
    TRANSFER_BOT,
    IntakeComment,
    Reaction,
    pending_transfers,
    run,
    transfer_body,
    transferred_ids_in,
)

LARRY_ID = 42926649
APPROVERS = frozenset({LARRY_ID})
EDITED_AT = "2026-09-28T10:00:00Z"
BEFORE_EDIT = "2026-09-28T09:00:00Z"
AFTER_EDIT = "2026-09-28T11:00:00Z"


def _comment(comment_id: int, *reactions: tuple[int, str, str, str], body: str = "idea") -> IntakeComment:
    return IntakeComment(
        comment_id=comment_id,
        author="someone",
        body=body,
        url=f"https://github.com/o/r/issues/1#issuecomment-{comment_id}",
        updated_at=EDITED_AT,
        reactions=tuple(
            Reaction(user_id=uid, login=login, content=content, created_at=at) for uid, login, content, at in reactions
        ),
    )


def _approved_ids(comments: list[IntakeComment], transferred: set[int] = frozenset()) -> list[int]:
    return [c.comment_id for c, _ in pending_transfers(comments, APPROVERS, set(transferred))]


def _bot_comment(body: str) -> dict:
    return {"user": {"login": TRANSFER_BOT}, "body": body}


def test_only_an_approver_thumbs_up_approves():
    comments = [
        _comment(1, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT)),
        _comment(2, (111, "randomuser", "+1", AFTER_EDIT)),
        _comment(3, (LARRY_ID, "ClassicLarry", "heart", AFTER_EDIT)),
        _comment(4),
    ]
    assert _approved_ids(comments) == [1]


def test_approval_is_by_user_id_not_login():
    # Someone who claimed an approver's old login after a rename has a different id and does not approve.
    impostor = _comment(1, (999, "ClassicLarry", "+1", AFTER_EDIT))
    renamed_approver = _comment(2, (LARRY_ID, "NewLarryName", "+1", AFTER_EDIT))
    assert _approved_ids([impostor, renamed_approver]) == [2]


def test_an_edit_after_approval_needs_a_new_approval():
    stale = _comment(1, (LARRY_ID, "ClassicLarry", "+1", BEFORE_EDIT))
    reapproved = _comment(2, (LARRY_ID, "ClassicLarry", "+1", BEFORE_EDIT), (9633, "dlwh", "+1", AFTER_EDIT))
    assert _approved_ids([stale, reapproved], transferred=set()) == []
    assert [c.comment_id for c, _ in pending_transfers([reapproved], frozenset({LARRY_ID, 9633}), set())] == [2]


def test_markers_count_only_from_bot_transfer_comments():
    approved = _comment(1, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT))
    target = [
        _bot_comment(transfer_body(approved, "ClassicLarry")),
        {"user": {"login": "someuser"}, "body": "<!-- belay-intake:5 -->"},
        _bot_comment("status note <!-- belay-intake:6 --> then more text"),
    ]
    assert transferred_ids_in(target) == {1}


def test_marker_text_inside_a_suggestion_cannot_mark_other_comments():
    injected = _comment(1, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="please see <!-- belay-intake:2 -->")
    assert transferred_ids_in([_bot_comment(transfer_body(injected, "ClassicLarry"))]) == {1}


def test_copied_text_cannot_ping_anyone():
    body = transfer_body(_comment(7, body="Try X @bob and @marin-community/team"), "Larry")
    assert "@bob" not in body and "@marin-community" not in body


class _FakeGitHub:
    def __init__(self, intake: list[dict], reactions: dict[int, list[dict]], target: list[dict]):
        self.intake, self.reactions, self.target = intake, reactions, target
        self.posted: list[str] = []
        self.reacted: list[tuple[int, str]] = []
        self.reaction_fetches: list[int] = []

    def issue_comments(self, issue: int) -> list[dict]:
        if issue == 1:
            return self.intake
        return self.target + [_bot_comment(b) for b in self.posted]

    def comment_reactions(self, comment_id: int) -> list[dict]:
        self.reaction_fetches.append(comment_id)
        return self.reactions.get(comment_id, [])

    def post_comment(self, issue: int, body: str) -> None:
        del issue
        self.posted.append(body)

    def react(self, comment_id: int, content: str) -> None:
        self.reacted.append((comment_id, content))


def _raw(comment_id: int, body: str, **reaction_counts: int) -> dict:
    return {
        "id": comment_id,
        "user": {"login": "alice"},
        "body": body,
        "html_url": f"u{comment_id}",
        "updated_at": EDITED_AT,
        "reactions": {"+1": reaction_counts.get("thumbs", 0), "rocket": reaction_counts.get("rocket", 0)},
    }


def test_run_copies_once_marks_with_rocket_and_skips_unapproved_reaction_fetches():
    intake = [_raw(10, "idea A", thumbs=1), _raw(11, "idea B")]
    reactions = {10: [{"user": {"id": LARRY_ID, "login": "ClassicLarry"}, "content": "+1", "created_at": AFTER_EDIT}]}
    github = _FakeGitHub(intake, reactions, target=[])
    assert run(github, intake_issue=1, target_issue=2, approvers=APPROVERS, dry_run=False) == 1
    assert github.reacted == [(10, "rocket")]
    assert "> idea A" in github.posted[0]
    assert github.reaction_fetches == [10]
    # A second pass finds the bot's marker for comment 10 and copies nothing.
    intake[0]["reactions"]["rocket"] = 1
    assert run(github, intake_issue=1, target_issue=2, approvers=APPROVERS, dry_run=False) == 0
    assert len(github.posted) == 1


def test_run_adds_a_missing_rocket_to_an_already_copied_comment():
    approved = _comment(10, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="idea A")
    target = [_bot_comment(transfer_body(approved, "ClassicLarry"))]
    github = _FakeGitHub([_raw(10, "idea A", thumbs=1)], reactions={}, target=target)
    assert run(github, intake_issue=1, target_issue=2, approvers=APPROVERS, dry_run=False) == 0
    assert github.reacted == [(10, "rocket")]
    assert github.posted == []
