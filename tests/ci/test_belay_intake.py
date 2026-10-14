# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from scripts.ci.belay_intake import (
    INTAKE_INDEX,
    LEGACY_TARGET_ISSUE,
    REGISTRY_DIR,
    TRANSFER_BOT,
    IntakeComment,
    Reaction,
    next_experiment_id,
    pending_transfers,
    register,
    reply_body,
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
        _bot_comment(transfer_body(approved, "ClassicLarry", "X0001")),
        {"user": {"login": "someuser"}, "body": "<!-- belay-intake:5 -->"},
        _bot_comment("status note <!-- belay-intake:6 --> then more text"),
    ]
    assert transferred_ids_in(target) == {1}


def test_marker_text_inside_a_suggestion_cannot_mark_other_comments():
    injected = _comment(1, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="please see <!-- belay-intake:2 -->")
    assert transferred_ids_in([_bot_comment(transfer_body(injected, "ClassicLarry", "X0001"))]) == {1}


def test_copied_text_cannot_ping_anyone():
    body = transfer_body(_comment(7, body="Try X @bob and @marin-community/team"), "Larry", "X0001")
    assert "@bob" not in body and "@marin-community" not in body


WEEK_ISSUE = 2
INTAKE_ISSUE = 1


class _FakeGitHub:
    """Issues by number (the intake issue, this week's issue, the legacy issue); posts land on the right one."""

    repo = "o/r"

    def __init__(self, intake: list[dict], reactions: dict[int, list[dict]], target: list[dict], legacy=()):
        self.reactions = reactions
        self.issues = {INTAKE_ISSUE: list(intake), WEEK_ISSUE: list(target), LEGACY_TARGET_ISSUE: list(legacy)}
        self.posted: dict[int, list[str]] = {INTAKE_ISSUE: [], WEEK_ISSUE: []}
        self.reacted: list[tuple[int, str]] = []
        self.reaction_fetches: list[int] = []
        self.weekly = [WEEK_ISSUE]
        self.files: dict[str, str] = {
            f"{REGISTRY_DIR}/X0007-older-idea.yaml": "id: X0007\n",
            f"{REGISTRY_DIR}/categories.json": json.dumps(["MoE and routing", "Unsorted"]),
        }

    def open_issues_labeled(self, label: str) -> list[int]:
        assert label == "belay-weekly"
        return self.weekly

    def issue_comments(self, issue: int) -> list[dict]:
        return self.issues[issue] + [_bot_comment(b) for b in self.posted.get(issue, [])]

    def comment_reactions(self, comment_id: int) -> list[dict]:
        self.reaction_fetches.append(comment_id)
        return self.reactions.get(comment_id, [])

    def post_comment(self, issue: int, body: str) -> None:
        self.posted[issue].append(body)

    def react(self, comment_id: int, content: str) -> None:
        self.reacted.append((comment_id, content))

    def get_file(self, path: str, ref: str) -> tuple[str, str] | None:
        del ref
        return (self.files[path], f"sha-{len(self.files[path])}") if path in self.files else None

    def list_dir(self, path: str, ref: str) -> list[str]:
        del ref
        return [p.rsplit("/", 1)[1] for p in self.files if p.startswith(path + "/")]

    def put_file(self, path: str, ref: str, text: str, message: str, sha: str | None = None) -> None:
        del ref, message
        assert (sha is None) == (path not in self.files), "updates must pass the current sha, creates must not"
        self.files[path] = text


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
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 1
    assert github.reacted == [(10, "rocket")]
    assert "> idea A" in github.posted[WEEK_ISSUE][0]
    assert github.posted[INTAKE_ISSUE][0].startswith("🤖 Registered as **X0008**")
    assert github.reaction_fetches == [10]
    # A second pass finds the bot's marker for comment 10 and copies nothing.
    intake[0]["reactions"]["rocket"] = 1
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 0
    assert len(github.posted[WEEK_ISSUE]) == 1 and len(github.posted[INTAKE_ISSUE]) == 1


def test_run_adds_a_missing_rocket_to_an_already_copied_comment():
    approved = _comment(10, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="idea A")
    target = [_bot_comment(transfer_body(approved, "ClassicLarry", "X0001"))]
    github = _FakeGitHub([_raw(10, "idea A", thumbs=1)], reactions={}, target=target)
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 0
    assert github.reacted == [(10, "rocket")]
    assert github.posted == {INTAKE_ISSUE: [], WEEK_ISSUE: []}  # copied before the registry: no id, no reply


def test_run_registers_the_next_experiment_id_once():
    intake = [
        _raw(
            10,
            "**Title:** Sqrt-softplus router\n**Category:** MoE and routing\n**Idea:** combine with sqrt(softplus)",
            thumbs=1,
        )
    ]
    reactions = {10: [{"user": {"id": LARRY_ID, "login": "ClassicLarry"}, "content": "+1", "created_at": AFTER_EDIT}]}
    github = _FakeGitHub(intake, reactions, target=[])
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 1
    assert f"{REGISTRY_DIR}/X0008-sqrt-softplus-router.yaml" in github.files
    assert github.posted[WEEK_ISSUE][0].startswith("📥 **Intake suggestion X0008**")
    assert json.loads(github.files[INTAKE_INDEX]) == {"10": "X0008"}


def test_a_retried_transfer_reuses_its_experiment_id():
    approved = _comment(10, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="idea A")
    github = _FakeGitHub([], reactions={}, target=[])
    github.files[INTAKE_INDEX] = json.dumps({"10": "X0003"})
    assert register(github, approved, "ClassicLarry") == "X0003"
    assert not any(p.endswith("idea-a.yaml") for p in github.files)


def test_next_experiment_id_ignores_other_files():
    assert next_experiment_id(["X0004-a.yaml", "X0012-b.yaml", "intake_ids.json", "README.md"]) == "X0013"
    assert next_experiment_id([]) == "X0001"


def test_a_comment_copied_last_week_or_to_the_legacy_issue_is_not_copied_again():
    approved = _comment(10, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="idea A")
    reply = _bot_comment(reply_body(11, "X0009", 5, "o/r"))
    intake = [_raw(10, "idea A", thumbs=1, rocket=1), _raw(11, "idea B", thumbs=1, rocket=1), reply]
    github = _FakeGitHub(intake, reactions={}, target=[], legacy=[_bot_comment(transfer_body(approved, "L", "X0001"))])
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 0
    assert github.posted == {INTAKE_ISSUE: [], WEEK_ISSUE: []} and github.reaction_fetches == []


def test_a_copy_without_its_reply_gets_the_reply():
    approved = _comment(10, (LARRY_ID, "ClassicLarry", "+1", AFTER_EDIT), body="idea A")
    target = [_bot_comment(transfer_body(approved, "ClassicLarry", "X0004"))]
    github = _FakeGitHub([_raw(10, "idea A", thumbs=1, rocket=1)], reactions={}, target=target)
    github.files[INTAKE_INDEX] = json.dumps({"10": "X0004"})
    assert run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=False) == 0
    assert github.posted[INTAKE_ISSUE] == [reply_body(10, "X0004", WEEK_ISSUE, "o/r")]


def test_run_needs_exactly_one_open_weekly_issue():
    github = _FakeGitHub([], reactions={}, target=[])
    github.weekly = [2, 3]
    with pytest.raises(RuntimeError):
        run(github, intake_issue=INTAKE_ISSUE, approvers=APPROVERS, dry_run=True)
