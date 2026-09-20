# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.mit_ocw import ocw_html_to_markdown


def test_ocw_html_to_markdown_keeps_course_content_and_drops_navigation():
    payload = b"""
    <html><head><title>Course title</title><style>.hidden {}</style></head>
    <body><nav>Course menu</nav><main><h1>Lecture 1</h1><p>Newton's second law is F = ma.</p>
    <script>tracking()</script></main><footer>MIT footer</footer></body></html>
    """

    title, text = ocw_html_to_markdown(payload)

    assert title == "Lecture 1"
    assert "Newton's second law" in text
    assert "Course menu" not in text
    assert "tracking" not in text
    assert "MIT footer" not in text
