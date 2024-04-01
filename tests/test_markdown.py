from markdown import minimal_markdown_escape


def test_always_escaped():
    """Tests characters that are always escaped."""
    test_cases = [
        ("[brackets]", r"\[brackets\]"),
        (r"Backslash "'\\', r"Backslash "'\\'),
        ("`code block`", r"\`code block\`"),
    ]

    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_no_escaping():
    """Tests scenarios where characters should not be escaped."""
    test_cases = [
        ("word * in the middle *", "word * in the middle *"),
        ("_ surrounded by spaces _", "_ surrounded by spaces _"),
        ("-normal hyphen-", "-normal hyphen-"),
        ("+normal plus+", "+normal plus+"),
        ("Normal line with #", "Normal line with #"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_conditional_escaping():
    """Tests *, _ with and without surrounding spaces."""
    test_cases = [
        ("word*bold*", r"word\*bold\*"),
        ("* this is not bold *", "* this is not bold *"),
        ("normal *bold*", "normal \\*bold\\*"),
        ("_italic_", "\_italic\_"),
        ("normal _italic_", "normal \_italic\_"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_line_start_escaping():
    """Tests -, #, + at the beginning of lines."""
    test_cases = [
        ("- hyphen", "\- hyphen"),
        ("+ plus", "\+ plus"),
        ("# header", "\# header"),
        ("Normal line - not escaped", "Normal line - not escaped"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected



def test_combined_scenarios():
    """Tests with multiple special characters."""
    input_text = \
    r"""This is *bold* and _italic_.
    # Header 1
    - List item 1
    + List item 2
    This has [brackets] and \backslashes\.
    """
    expected_text = \
    r"""This is \*bold\* and \_italic\_.
    \# Header 1
    \- List item 1
    \+ List item 2
    This has \[brackets\] and \backslashes\\.
    """
    assert minimal_markdown_escape(input_text) == expected_text
