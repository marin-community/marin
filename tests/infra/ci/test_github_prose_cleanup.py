import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.ci.github_prose_cleanup import (
    CLEANUP_FOOTER_MARKER,
    GithubItemKind,
    cleanup_github_body,
    cleanup_result_from_event,
)


def test_cleanup_github_body_removes_hype_without_dropping_facts():
    body = """## **Oh my god!** 🚀

Importantly, this is not just faster, but 20% cheaper.
<h1>This changes everything</h1>
Latency fell from 10 seconds to 8 seconds.
"""

    result = cleanup_github_body(body)

    assert result.changed
    assert result.cleaned_body == ("This is faster and 20% cheaper.\nLatency fell from 10 seconds to 8 seconds.")


def test_cleanup_github_body_preserves_content_inside_code():
    body = """Wow — use the parser output below.

```markdown
**Oh my god**
<h1>Raw HTML example</h1>
This is not just X, but Y.
```

Keep `not just X, but Y` unchanged in inline code.
"""

    result = cleanup_github_body(body)

    assert (
        result.cleaned_body
        == """Use the parser output below.

```markdown
**Oh my god**
<h1>Raw HTML example</h1>
This is not just X, but Y.
```

Keep `not just X, but Y` unchanged in inline code."""
    )


def test_cleanup_github_body_keeps_structural_bold_labels():
    body = """**Reproduction**

1. Run the tokenizer.

- **Root cause:** the mask starts one token late.
- The token IDs are **BYTE-EXACT 60/60**.
"""

    result = cleanup_github_body(body)

    assert (
        result.cleaned_body
        == """**Reproduction**

1. Run the tokenizer.

- **Root cause:** the mask starts one token late.
- The token IDs are BYTE-EXACT 60/60."""
    )


def test_cleanup_github_body_strips_presentational_html_but_keeps_text():
    body = """<h1>Compatibility</h1>

<p>The serializer emits the existing wire format.</p>
"""

    result = cleanup_github_body(body)

    assert result.cleaned_body == ("Compatibility\n\nThe serializer emits the existing wire format.")


def test_cleanup_github_body_splits_this_not_that_rhetoric():
    body = "This is not a rewrite, but a formatting pass. The API remains unchanged."

    result = cleanup_github_body(body)

    assert result.cleaned_body == ("This is a formatting pass. It is not a rewrite. The API remains unchanged.")


def test_cleanup_github_body_is_noop_for_clean_body_with_archive_footer():
    body = (
        "Latency fell from 10 seconds to 8 seconds.\n\n"
        "---\n"
        "[Original description](https://github.com/marin-community/marin/issues/1#issuecomment-1) "
        f"{CLEANUP_FOOTER_MARKER}"
    )

    result = cleanup_github_body(body)

    assert not result.changed
    assert result.cleaned_body == body


def test_cleanup_github_body_replaces_stale_archive_footer_after_new_cleanup():
    body = (
        "Wow — latency fell from 10 seconds to 8 seconds.\n\n"
        "---\n"
        "[Original description](https://github.com/marin-community/marin/issues/1#issuecomment-1) "
        f"{CLEANUP_FOOTER_MARKER}"
    )

    result = cleanup_github_body(body)

    assert result.changed
    assert result.cleaned_body == "Latency fell from 10 seconds to 8 seconds."


def test_cleanup_result_from_event_archives_original_pr_body():
    original = "Wow — the tokenizer is not just faster, but easier to inspect.\n\n```\n`code`\n```"
    event = {"pull_request": {"number": 7455, "body": original}}

    result = cleanup_result_from_event(event)

    assert result.kind is GithubItemKind.PULL_REQUEST
    assert result.number == 7455
    assert result.changed
    assert result.original_body_hash == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert original in result.archive_comment_body
    assert result.archive_marker in result.archive_comment_body
    assert result.archive_comment_body.startswith("🤖 Archived the description before automated prose cleanup.")


def test_cleanup_result_from_event_skips_when_exact_archive_would_be_too_large():
    original = f"Wow — {'x' * 65_400}"
    event = {"issue": {"number": 1234, "body": original}}

    result = cleanup_result_from_event(event)

    assert result.kind is GithubItemKind.ISSUE
    assert not result.changed
    assert result.cleaned_body == original
    assert result.skip_reason == "archive_too_large"


def test_cli_writes_cleanup_and_github_action_outputs(tmp_path: Path):
    event_path = tmp_path / "event.json"
    result_path = tmp_path / "result.json"
    github_output_path = tmp_path / "github-output.txt"
    event_path.write_text(
        json.dumps({"issue": {"number": 1234, "body": "Wow — latency fell by 20%."}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/ci/github_prose_cleanup.py",
            "--event",
            str(event_path),
            "--output",
            str(result_path),
            "--github-output",
            str(github_output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["changed"] is True
    assert result["cleaned_body"] == "Latency fell by 20%."
    assert github_output_path.read_text(encoding="utf-8").splitlines() == [
        "changed=true",
        f"result={result_path}",
    ]
    assert completed.stdout == "issue #1234: cleanup required\n"
