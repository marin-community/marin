# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
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


def test_cleanup_github_body_removes_standalone_bold_heading_but_keeps_list_label():
    body = """**Reproduction**

1. Run the tokenizer.

- **Root cause:** the mask starts one token late.
- The token IDs are **BYTE-EXACT 60/60**.
"""

    result = cleanup_github_body(body)

    assert (
        result.cleaned_body
        == """1. Run the tokenizer.

- **Root cause:** the mask starts one token late.
- The token IDs are BYTE-EXACT 60/60."""
    )


def test_cleanup_github_body_removes_heading_scaffolding_but_keeps_body():
    body = """<h1>Compatibility</h1>

<p>The serializer emits the existing wire format.</p>
"""

    result = cleanup_github_body(body)

    assert result.cleaned_body == "The serializer emits the existing wire format."


def test_cleanup_github_body_removes_markdown_headings_and_checkboxes():
    body = """## Summary

The serializer emits the existing wire format.

## Completion criteria

- [x] Preserve query strings.
- [ ] Preserve response headers.
"""

    result = cleanup_github_body(body)

    assert result.cleaned_body == (
        "The serializer emits the existing wire format.\n\n- Preserve query strings.\n- Preserve response headers."
    )


def test_cleanup_github_body_turns_factual_heading_into_plain_text():
    body = "## Map CPU time falls 58.9%\n\nThe comparison covers 64 tasks."

    result = cleanup_github_body(body)

    assert result.cleaned_body == "Map CPU time falls 58.9%\n\nThe comparison covers 64 tasks."


def test_cleanup_github_body_keeps_benchmark_table_without_section_heading():
    body = """## Performance

| stage | main | branch | delta |
| --- | ---: | ---: | ---: |
| Map | 293.6s | 120.6s | -58.9% |

The comparison covers 64 tasks.
"""

    result = cleanup_github_body(body)

    assert result.cleaned_body == (
        "| stage | main | branch | delta |\n"
        "| --- | ---: | ---: | ---: |\n"
        "| Map | 293.6s | 120.6s | -58.9% |\n\n"
        "The comparison covers 64 tasks."
    )


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

    rewritten = "The tokenizer is faster and easier to inspect.\n\n```\n`code`\n```"
    result = cleanup_result_from_event(event, rewritten)

    assert result.kind is GithubItemKind.PULL_REQUEST
    assert result.number == 7455
    assert result.changed
    assert result.original_body_hash == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert original in result.archive_comment_body
    assert result.archive_marker in result.archive_comment_body
    assert result.archive_comment_body.startswith("🤖 Archived the description before automated prose cleanup.")


def test_cleanup_result_from_event_is_noop_when_rewrite_preserves_body():
    original = "Latency fell from 10 seconds to 8 seconds."
    event = {"pull_request": {"number": 7452, "body": original}}

    result = cleanup_result_from_event(event, original)

    assert not result.changed
    assert result.cleaned_body == original
    assert not result.archive_comment_body


def test_cleanup_result_from_event_rejects_empty_rewrite():
    original = "The serializer drops response headers."
    event = {"issue": {"number": 1234, "body": original}}

    result = cleanup_result_from_event(event, "\n")

    assert not result.changed
    assert result.cleaned_body == original
    assert result.skip_reason == "rewrite_empty"


def test_cleanup_result_from_event_does_not_apply_word_limit():
    original = "The benchmark compares worker CPU time."
    rewritten = " ".join(f"measurement-{index}" for index in range(250))
    event = {"pull_request": {"number": 7200, "body": original}}

    result = cleanup_result_from_event(event, rewritten)

    assert result.changed
    assert result.cleaned_body == rewritten


def test_cleanup_result_from_event_skips_when_exact_archive_would_be_too_large():
    original = f"Wow — {'x' * 65_400}"
    event = {"issue": {"number": 1234, "body": original}}

    result = cleanup_result_from_event(event, "x" * 65_400)

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
            "--rewritten-body-env",
            "REWRITTEN_BODY",
            "--github-output",
            str(github_output_path),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "REWRITTEN_BODY": "Latency fell by 20%."},
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
