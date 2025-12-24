# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from textwrap import fill

import markdownify
import six
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString
from markdownify import MarkdownConverter
from regex import regex

from marin.schemas.web.convert import HtmlToMarkdownConfig

logger = logging.getLogger(__name__)


def to_markdown(html, config: HtmlToMarkdownConfig = HtmlToMarkdownConfig()):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    text = MyMarkdownConverter(config).convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ")
    return text


whitespace_re = re.compile(r"[\t ]+")
spaces_re = re.compile(r"^[ ]+$")


always_escape_pattern = re.compile(r"([\[\]<>`])")  # square brackets, backticks, angle brackets
line_start_escape_pattern = re.compile(r"^(\s*)([-+#]\s)", flags=re.MULTILINE)
# only escape backslashes before ascii punctuation. in gfm, other backslackes are literal
backslash_before_ascii_punct_pattern = re.compile(r'(\\[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~])', flags=re.ASCII)


# reference: https://github.github.com/gfm/

delimiter_run_pattern = regex.compile(
    r"""(?:
    (?<!\\[*])  # Assert not preceded by a backslash '\*'
    [*]+       # Match one or more '*
    (?!\\[*])   # Assert not followed by a backslash '\*'
    |
    (?<!\\[_])  # Assert not preceded by a backslash '\_'
    [_]+       # Match one or more '_'
    (?!\\[_])   # Assert not followed by a backslash '\_'
    )""",
    re.VERBOSE,
)


# A left-flanking delimiter run is a delimiter run that is (1) not followed by Unicode whitespace, and either
# (2a) not followed by a punctuation character, or
# (2b) followed by a punctuation character and preceded by Unicode whitespace or a punctuation character.
# For purposes of this definition, the beginning and the end of the line count as Unicode whitespace.

# Restated:
# followed by a non-space-non-punctuation character OR
# preceded by a space, punctuation character, or bol and followed by a punctuation character
left_flanking_pattern = regex.compile(
    rf"""(?:(?:{delimiter_run_pattern.pattern}(?=[^\s\p{{P}}]))|(?:(?<=[\s\p{{P}}]|^){delimiter_run_pattern.pattern}(?:\p{{P}})))""",
    re.VERBOSE,
)

# A right-flanking delimiter run is a delimiter run that is (1) not preceded by Unicode whitespace, and either
# (2a) not preceded by a punctuation character, or
# (2b) preceded by a punctuation character and followed by Unicode whitespace or a punctuation character.
# For purposes of this definition, the beginning and the end of the line count as Unicode whitespace.

# Restated:
# preceded by a non-space-non-punctuation character OR
# followed by a space, punctuation character, or eol and preceded by a punctuation character
right_flanking_pattern = regex.compile(
    rf"""(?:(?:(?<=[^\s\p{{P}}]){delimiter_run_pattern.pattern})|(?:(?<=\p{{P}}){delimiter_run_pattern.pattern}(?=\p{{P}}|\s|$)))"""
)

flanking_pattern = regex.compile(rf"({left_flanking_pattern.pattern}|{right_flanking_pattern.pattern})", re.VERBOSE)


def minimal_markdown_escape(text):
    """
    Escapes special characters in Markdown text to avoid formatting from HTML

    Args:
        text: The input text string.

    Returns:
        The escaped Markdown text.
    """
    # tries to escape as little as possible.
    # the rules we follow are:
    # '*' is not escaped if there is a space on both sides
    # '_' is not escaped if there is a space on both sides
    # '-', '#', and '+' are escaped if they are at the beginning of a line and followed by a space
    # '[' and ']' are always escaped
    # '`' is always escaped
    # '\' is escaped before ascii punctuation
    # '!' doesn't need to be escaped because we escape the following '['
    # '<' must be escaped in html, and since Markdown uses '<' for html, we should escape it

    # this has to come first because it will escape the other characters
    text = backslash_before_ascii_punct_pattern.sub(r"\\\1", text)
    text = line_start_escape_pattern.sub(r"\1\\\2", text)
    text = flanking_pattern.sub(r"\\\1", text)
    text = always_escape_pattern.sub(r"\\\1", text)

    return text


def _try_convert_int(val, default):
    # handles a few cases we see in the wild: the number, "number", 'number', number;
    # punting on percent and fraction
    try:
        return int(val)
    except ValueError:
        val = val.strip().replace('"', "").replace("'", "").replace(";", "").replace(",", "")
        try:
            return int(val)
        except ValueError:
            return default


class MyMarkdownConverter(MarkdownConverter):
    def __init__(self, config: HtmlToMarkdownConfig, **kwargs):
        self.include_links = config.include_links
        self.include_images = config.include_images

        kwargs = config.markdownify_kwargs
        super().__init__(**kwargs)

    def convert_hn(self, n, el, text, convert_as_inline):
        if convert_as_inline:
            return text

        style = self.options["heading_style"].lower()
        text = text.strip()
        if style == markdownify.UNDERLINED and n <= 2:
            line = "=" if n == 1 else "-"
            return self.underline(text, line)
        hashes = "#" * n
        if style == markdownify.ATX_CLOSED:
            return f"{hashes} {text} {hashes}\n\n"

        return f"\n\n{hashes} {text}\n\n"

    def convert_a(self, el, text, convert_as_inline):
        prefix, suffix, text = markdownify.chomp(text)

        if not self.include_links:
            return text if len(text) > 1 else ""

        if not text:
            return ""
        href = el.get("href")
        # ignore base64 images
        if href and "data" in href and ";base64" in href:
            return ""
        title = el.get("title")
        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return f"<{href}>"
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "{title}"'.format(title=title.replace('"', r"\"") if title else "")
        return f"{prefix}[{text}]({href}{title_part}){suffix}" if href else text

    # markdownify doesn't allow difference pre- and post- text for converting sub and sup
    def convert_sub(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"<sub>{text}</sub>"

    def convert_sup(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"<sup>{text}</sup>"

    def convert_br(self, el, text, convert_as_inline):
        if convert_as_inline:
            return "<br>"

        if self.options["newline_style"].lower() == markdownify.BACKSLASH:
            return "\\\n"
        else:
            return "  \n"

    def convert_img(self, el, text, convert_as_inline):
        # mostly copied from the parent class
        # the gfm spec says that the alt text is markdown, so we need to escape it

        if not self.include_images:
            return ""

        alt = el.attrs.get("alt", None) or ""
        alt = alt.replace("\n", " ")
        alt = self.escape(alt)
        src = el.attrs.get("src", None) or el.attrs.get("data-src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "{title}"'.format(title=title.replace('"', r"\"") if title else "")
        if convert_as_inline and el.parent.name not in self.options["keep_inline_images_in"]:
            return alt

        return f"![{alt}]({src}{title_part})"

    def escape(self, text):
        return minimal_markdown_escape(text)

    def convert_figure(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text

        # the super doesn't handle this specifically. we basically want to be sure there's a newline
        if not text.endswith("\n\n"):
            if not text.endswith("\n"):
                text += "\n\n"
            else:
                text += "\n"
        return text

    def convert_pre(self, el, text, convert_as_inline):
        if not text:
            return ""
        code_language = self.options["code_language"]

        if self.options["code_language_callback"]:
            code_language = self.options["code_language_callback"](el) or code_language

        if "```" in text:  # have to use <pre>
            return f"\n<pre><code>{text}</code></pre>\n"
        else:
            return f"\n```{code_language}\n{text}\n```\n"

    def _compute_num_cols(self, el):
        length_of_cells = 0
        prev_td = el.find_all("td", recursive=False)
        prev_th = el.find_all("th", recursive=False)
        length = len(prev_td) + len(prev_th)
        for td in prev_td:
            if "colspan" in td.attrs:
                length += _try_convert_int(td["colspan"], 1) - 1
        for th in prev_th:
            if "colspan" in th.attrs:
                length += _try_convert_int(th["colspan"], 1) - 1
        length_of_cells = max(length_of_cells, length)
        if el.previous_sibling:
            prev = el.previous_sibling
            _count = 1
            while prev:
                if prev.name == "tr":
                    prev_td = prev.find_all("td", recursive=False)
                    prev_th = prev.find_all("th", recursive=False)
                    length = len(prev_td) + len(prev_th)
                    for td in prev_td:
                        if "colspan" in td.attrs:
                            length += _try_convert_int(td["colspan"], 1) - 1
                    for th in prev_th:
                        if "colspan" in th.attrs:
                            length += _try_convert_int(th["colspan"], 1) - 1
                    length_of_cells = max(length_of_cells, length)
                prev = prev.previous_sibling
        return length_of_cells

    def _adjust_for_rowspan(self, el, text, length_of_cells):
        rowspan = [0 for _ in range(length_of_cells)]
        if el.previous_sibling:
            prev = el.previous_sibling
            count = 1
            while prev:
                if prev.name == "tr":
                    prev_td = prev.find_all("td", recursive=False)
                    i = 0
                    for td in prev_td:
                        if "rowspan" in td.attrs and _try_convert_int(td["rowspan"], 1) > count:
                            rowspan[i] = 1
                        if "colspan" in td.attrs:
                            i += _try_convert_int(td["colspan"], 1)
                        else:
                            i += 1
                    prev_th = prev.find_all("th", recursive=False)
                    i = 0
                    for th in prev_th:
                        if "rowspan" in th.attrs and _try_convert_int(th["rowspan"], 1) > count:
                            rowspan[i] = 1
                        if "colspan" in th.attrs:
                            i += _try_convert_int(th["colspan"], 1)
                        else:
                            i += 1
                prev = prev.previous_sibling
                count += 1
        # modify text for rowspan
        text = text.split("|")
        for i, row in enumerate(rowspan):
            if row:
                text.insert(i, " ")
        text = "|".join(text)
        return text

    def convert_tr(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text + "\n"
        # this is also mostly copied from the parent class
        # but the logic for guessing a th isn't quite right
        cells = el.find_all(["td", "th"])
        is_headrow = all([cell.name == "th" for cell in cells])

        # we can be a headrow if we are the first row in the table or if all our cells are th
        # find table parent
        if not is_headrow:
            parent = el.parent
            while parent and parent.name != "table":
                parent = parent.parent

            if parent:
                first_row = parent.find("tr")
                if first_row is el:
                    is_headrow = True

        # rowspan check
        length_of_cells = self._compute_num_cols(el)
        text = self._adjust_for_rowspan(el, text, length_of_cells)

        overline = ""
        underline = ""
        if is_headrow and not el.previous_sibling:
            # first row and is headline: print headline underline
            underline += "| " + " | ".join(["---"] * length_of_cells) + " |" + "\n"
        elif not el.previous_sibling and (
            el.parent.name == "table" or (el.parent.name == "tbody" and not el.parent.previous_sibling)
        ):
            # first row, not headline, and:
            # - the parent is table or
            # - the parent is tbody at the beginning of a table.
            # print empty headline above this row
            overline += "| " + " | ".join([""] * length_of_cells) + " |" + "\n"
            overline += "| " + " | ".join(["---"] * length_of_cells) + " |" + "\n"
        return overline + "|" + text + "\n" + underline

    def indent(self, text, level):
        return markdownify.line_beginning_re.sub("    " * level, text) if text else ""

    def process_tag(self, node, convert_as_inline, children_only=False, parent_tags=None):
        # skip aria-hidden elements
        if node.get("aria-hidden") == "true":
            return ""
        text = ""

        # some sites use tables for layout, and so we need to 'inline' them. Our heuristic is that if the table
        # has block elements in it.
        if self._is_layout_table(node):
            return self._process_layout_table(node, convert_as_inline)

        # markdown headings or cells can't include
        # block elements (elements w/newlines)
        isHeading = markdownify.html_heading_re.match(node.name) is not None
        isEmphasisLike = node.name in ["em", "strong", "b", "i", "u", "s", "del", "ins"]
        isCell = node.name in ["td", "th"]
        convert_children_as_inline = convert_as_inline

        if not children_only and (isHeading or isCell or isEmphasisLike):
            convert_children_as_inline = True

        # Remove whitespace-only textnodes in purely nested nodes
        def is_nested_node(el):
            return el and el.name in ["ol", "ul", "li", "table", "thead", "tbody", "tfoot", "tr", "td", "th"]

        def is_in_preformatted(el):
            return el.name == "pre" or el.find_parent("pre")

        if is_nested_node(node):
            for el in node.children:
                # Only extract (remove) whitespace-only text node if any of the
                # conditions is true:
                # - el is the first element in its parent
                # - el is the last element in its parent
                # - el is adjacent to an nested node
                can_extract = (
                    not el.previous_sibling
                    or not el.next_sibling
                    or is_nested_node(el.previous_sibling)
                    or is_nested_node(el.next_sibling)
                )
                if isinstance(el, NavigableString) and six.text_type(el).strip() == "" and can_extract:
                    el.extract()

        # Convert the children first
        is_in_pre = is_in_preformatted(node)

        for el in node.children:
            if isinstance(el, Comment) or isinstance(el, Doctype):
                continue
            elif isinstance(el, NavigableString):
                next_text = self.process_text(el)
                text = self.join_text(text, next_text, is_in_pre)
            else:
                text = self.join_text(text, self.process_tag(el, convert_children_as_inline), is_in_pre)

        if not children_only:
            convert_fn = getattr(self, f"convert_{node.name}", None)
            if convert_fn and self.should_convert_tag(node.name):
                text = convert_fn(node, text, convert_as_inline)

        return text

    def _is_layout_table(self, table):
        # heuristic to determine if a table is for layout
        # for some reason readability-lxml will have trs inside of a div with no table
        if table.name not in ["table", "tbody", "tr"]:
            return False

        # don't reprocess elements like tbody or tr if they are immediate children of a table
        if table.name != "table" and (table.parent and table.parent.name in ["table", "tbody"]):
            return False

        # if the table has th, caption, thead, or summary, it's probably not for layout
        if table.select_one("th, caption, thead, summary"):
            return False

        # unlikely, but if it's aria role=presentation, it's for layout
        if table.get("role") == "presentation":
            return True

        # if the table has exactly 1 td element (which happens after readability), then it's probably for layout
        if len(table.select("td")) == 1:
            return True

        # if there are block elements anywhere in the table, it's probably for layout
        if table.select_one(
            "div, h1, h2, h3, h4, h5, h6, blockquote, pre, ul, ol, dl, table, video, address, hr, section, main, nav, aside"  # noqa: E501
        ):
            return True

        # now we want to understand paragraphs. We want to look at each cell and see how many paragraphs are in it
        # if it's more than 1, it's probably for layout
        # Actually, we're fine with <p>'s. we'll convert them to <br>'s.
        # for td in table.select('td'):
        #     if len(td.select('p')) > 1:
        #         return True

        return False

    def _process_layout_table(self, table, convert_as_inline):
        # if the table is for layout, we want to inline it
        text = ""
        if table.name == "table" or table.name == "tbody":
            for row in table.find_all("tr", recursive=False):
                for cell in row.find_all(["td", "th"], recursive=False):
                    text += self.process_tag(cell, convert_as_inline, children_only=True)
                text += "\n\n"
        elif table.name == "tr":
            for cell in table.find_all(["td", "th"], recursive=False):
                text += self.process_tag(cell, convert_as_inline, children_only=True)

        return text

    def convert_li(self, el, text, convert_as_inline):
        parent = el.parent
        if parent is not None and parent.name == "ol":
            # TODO: upstream this
            # in theory this should always be an int, but in practice it might not be
            try:
                start = int(parent.get("start", 1))
            except (KeyError, ValueError):
                start = 1
            bullet = "%s." % (start + parent.index(el))
        else:
            depth = -1
            while el:
                if el.name == "ul":
                    depth += 1
                el = el.parent
            bullets = self.options["bullets"]
            bullet = bullets[depth % len(bullets)]
        return f"{bullet} {(text or '').strip()}\n"

    def convert_td(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text + " "
        colspan = 1
        if "colspan" in el.attrs:
            colspan = _try_convert_int(el["colspan"], 1)

        return " " + text.strip().replace("\n", " ") + " |" * colspan

    def convert_svg(self, el, text, convert_as_inline):
        # ignore svg elements
        return ""

    def convert_th(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text + " "
        colspan = 1
        if "colspan" in el.attrs:
            colspan = _try_convert_int(el["colspan"], 1)
        return " " + text.strip().replace("\n", " ") + " |" * colspan

    def join_text(self, text1, text2, is_in_pre):
        if not is_in_pre:
            is_paragraph = text1.endswith("\n\n")
            is_br = text1.endswith("<br>") or text1.endswith("  \n")
            # if text1 is a paragraph or br, we can trim any leading spaces
            # however, we nede to
            if is_paragraph or is_br:
                text2 = text2.lstrip()

            # mostly want to remove extra newlines
            # in MD, two newlines is a paragraph break, which is the most we want
            # so if text1 already has two newlines, we don't want to add another

            # more specifically we want to get the tail from text1 which is \n\s*$ and the head from text2 which is ^\n*
            # and replace it with at most two newlines
            tail1 = re.search(r"\n\s*$", text1)
            head2 = re.search(r"^([\n ]+|\n*)", text2)

            if tail1 and head2:
                text1 = text1[: tail1.start()]
                text2 = text2[head2.end() :]
                newline_count = tail1.group().count("\n") + head2.group().count("\n")
                if newline_count > 2:
                    newline_count = 2
                if newline_count:
                    text2 = "\n" * newline_count + text2
            # elif rhs_is_string:
            #     # if instead we are joining spaces, only join if there's not already a space
            #     tail1 = re.search(r' +$', text1)
            #     head2 = re.search(r'^ +', text2)
            #     if tail1 and head2:
            #         text1 = text1[:tail1.start()]
            #         text2 = text2[head2.end():]
            #         text2 = ' ' + text2

        return text1 + text2

    def convert_math(self, el, text, convert_as_inline):
        try:
            x = mathml_to_markdown(el)
            return x
        except Exception as e:
            logger.exception(f"Error converting math: {e}")
            return text

    def convert_p(self, el, text, convert_as_inline):
        # no reason for leading whitespace in a paragraph
        if text:
            text = text.lstrip()

        if convert_as_inline:
            # if el has a sibling, add a <br> at the end
            if el.next_sibling and text:
                return text + "<br>"
            return text

        if self.options["wrap"]:
            text = fill(text, width=self.options["wrap_width"], break_long_words=False, break_on_hyphens=False)
        return f"\n\n{text}\n\n" if text else ""

    def process_text(self, el):
        text = six.text_type(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = whitespace_re.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp"]):
            text = self.escape(text)

            # text = text.replace(' *\n', ' ')
            text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)

        # remove trailing whitespaces if any of the following condition is true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent.name == "li" and (not el.next_sibling or el.next_sibling.name in ["ul", "ol"]):
            text = text.rstrip()

        return text


# Mappings for unicode characters to LaTeX
_GREEK_LETTERS = {
    # Lowercase
    "α": "\\alpha",  # noqa: RUF001
    "β": "\\beta",
    "γ": "\\gamma",  # noqa: RUF001
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ϵ": "\\varepsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ϑ": "\\vartheta",
    "ι": "\\iota",  # noqa: RUF001
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",  # noqa: RUF001
    "ξ": "\\xi",
    "ο": "o",  # noqa: RUF001
    "π": "\\pi",
    "ϖ": "\\varpi",
    "ρ": "\\rho",  # noqa: RUF001
    "ϱ": "\\varrho",  # noqa: RUF001
    "σ": "\\sigma",  # noqa: RUF001
    "ς": "\\varsigma",
    "τ": "\\tau",
    "υ": "\\upsilon",  # noqa: RUF001
    "φ": "\\phi",
    "ϕ": "\\varphi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
    # Uppercase
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Θ": "\\Theta",
    "Λ": "\\Lambda",
    "Ξ": "\\Xi",
    "Π": "\\Pi",
    "Σ": "\\Sigma",
    "Υ": "\\Upsilon",  # noqa: RUF001
    "Φ": "\\Phi",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
    "Χ": "{\\rm X}",  # noqa: RUF001
    "Ι": "I",  # noqa: RUF001
    "Κ": "K",  # noqa: RUF001
    "Ο": "O",  # noqa: RUF001
    "ℏ": "\\hslash",
}


class LatexNode:
    """Base class for LaTeX nodes with smart string conversion and spacing logic."""

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def needs_space_before(self, other: "LatexNode") -> bool:
        """Determine if a space is needed before the other node."""
        return False


class MacroNode(LatexNode):
    """LaTeX command like \\alpha, \\langle."""

    def needs_space_before(self, other: "LatexNode") -> bool:
        # Macro needs space before text, another macro, or braced content
        return isinstance(other, (MacroNode, TextNode, BracedNode))


class TextNode(LatexNode):
    """Plain alphanumeric content like x, 1."""

    pass


class BracedNode(LatexNode):
    """Content wrapped in braces like {a}."""

    pass


class OperatorNode(LatexNode):
    """Operators and symbols like +, =, (, )."""

    pass


_MATH_OPERATORS = {
    "×": "\\times",  # noqa: RUF001
    "·": "\\cdot",
    "÷": "\\div",
    "±": "\\pm",
    "∓": "\\mp",
    "≤": "\\le",
    "≥": "\\ge",
    "≠": "\\neq",
    "≈": "\\approx",
    "≡": "\\equiv",
    "≃": "\\simeq",
    "∞": "\\infty",
    "∂": "\\partial",
    "∇": "\\nabla",
    "√": "\\surd",
    "∫": "\\int",
    "∬": "\\iint",
    "∭": "\\iiint",
    "∮": "\\oint",
    "∑": "\\sum",
    "∏": "\\prod",
    "∐": "\\coprod",
    "∈": "\\in",
    "∉": "\\notin",
    "∋": "\\ni",
    "⊂": "\\subset",
    "⊃": "\\supset",
    "⊆": "\\subseteq",
    "⊇": "\\supseteq",
    "∪": "\\cup",  # noqa: RUF001
    "∩": "\\cap",
    "→": "\\to",
    "←": "\\gets",
    "↔": "\\leftrightarrow",
    "⇒": "\\Rightarrow",
    "⇐": "\\Leftarrow",
    "⇔": "\\iff",
    "↦": "\\mapsto",
    "…": "\\dots",
    "∀": "\\forall",
    "∃": "\\exists",
    "∄": "\\nexists",
    "¬": "\\neg",
    "∧": "\\land",
    "∨": "\\lor",  # noqa: RUF001
    "⟨": "\\langle",
    "⟩": "\\rangle",
    "∥": "\\parallel",
    "∣": "\\mid",  # noqa: RUF001
    "ℏ": "\\hslash",
}


class MathMLToLatex:
    def convert(self, element):
        if not element:
            return ""

        if isinstance(element, str):
            try:
                soup = BeautifulSoup(element, "html.parser")
                element = soup.find("math")
                if not element:
                    # Fallback if no math tag found, though unlikely for valid mathml string
                    return element
            except Exception:
                return element

        node = self._visit(element)
        latex = str(node)

        # Determine wrapping
        tag = element.name
        if tag == "math":
            display = element.get("display", "inline")
            if display == "block":
                return f"\n$${latex}$$\n"
            else:
                return f"$`{latex}`$"
        return latex

    def _visit(self, element):
        if isinstance(element, NavigableString):
            text = element.strip()
            return TextNode(text) if text else TextNode("")

        if not element.name:
            return TextNode("")

        method_name = f"visit_{element.name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(element)

        # Default recursive visit
        return self._visit_children(element)

    def _visit_children(self, element):
        nodes = []
        for child in element.children:
            node = self._visit(child)
            if node and str(node):
                nodes.append(node)

        if not nodes:
            return TextNode("")

        # Join nodes with smart spacing
        result = [str(nodes[0])]
        for i in range(1, len(nodes)):
            if nodes[i - 1].needs_space_before(nodes[i]):
                result.append(" ")
            result.append(str(nodes[i]))

        return TextNode("".join(result))

    def _get_clean_children(self, element):
        return [c for c in element.children if not (isinstance(c, NavigableString) and not c.strip())]

    def visit_math(self, element):
        return self._visit_children(element)

    def visit_mrow(self, element):
        return self._visit_children(element)

    def visit_mi(self, element):
        text = element.get_text().strip()
        if text in _GREEK_LETTERS:
            return MacroNode(_GREEK_LETTERS[text])

        variant = element.get("mathvariant")
        if variant == "bold":
            return BracedNode(f"\\mathbf{{{text}}}")
        elif variant == "italic":
            return TextNode(text)
        elif variant == "double-struck":
            return BracedNode(f"\\mathbb{{{text}}}")
        elif variant == "script":
            return BracedNode(f"\\mathcal{{{text}}}")
        elif variant == "fraktur":
            return BracedNode(f"\\mathfrak{{{text}}}")
        elif variant == "sans-serif":
            return BracedNode(f"\\mathsf{{{text}}}")
        elif variant == "monospace":
            return BracedNode(f"\\mathtt{{{text}}}")

        # Basic heuristic: if length > 1 and not greek, it might be a function name
        if len(text) > 1 and text.isalpha():
            # Check for standard functions
            funcs = {
                "sin",
                "cos",
                "tan",
                "csc",
                "sec",
                "cot",
                "sinh",
                "cosh",
                "tanh",
                "log",
                "ln",
                "exp",
                "lim",
                "det",
                "dim",
                "min",
                "max",
            }
            if text in funcs:
                return MacroNode(f"\\{text}")
            return BracedNode(f"\\mathrm{{{text}}}")

        return TextNode(text)

    def visit_mn(self, element):
        return TextNode(element.get_text().strip())

    def visit_mo(self, element):
        text = element.get_text().strip()
        if text in _MATH_OPERATORS:
            return MacroNode(_MATH_OPERATORS[text])
        return OperatorNode(text)

    def visit_mtext(self, element):
        text = element.get_text()
        return BracedNode(f"\\text{{{text}}}")

    def visit_mfrac(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            num = self._visit(children[0])
            den = self._visit(children[1])
            return BracedNode(f"\\frac{{{num}}}{{{den}}}")
        return TextNode("")

    def visit_msqrt(self, element):
        content = self._visit_children(element)
        return BracedNode(f"\\sqrt{{{content}}}")

    def visit_mroot(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            base = self._visit(children[0])
            index = self._visit(children[1])
            return BracedNode(f"\\sqrt[{index}]{{{base}}}")
        return TextNode("")

    def visit_msup(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            base = self._visit(children[0])
            sup = self._visit(children[1])
            return BracedNode(f"{{{base}}}^{{{sup}}}")
        return TextNode("")

    def visit_msub(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            base = self._visit(children[0])
            sub = self._visit(children[1])
            return BracedNode(f"{{{base}}}_{{{sub}}}")
        return TextNode("")

    def visit_msubsup(self, element):
        children = self._get_clean_children(element)
        if len(children) == 3:
            base = self._visit(children[0])
            sub = self._visit(children[1])
            sup = self._visit(children[2])
            return BracedNode(f"{{{base}}}_{{{sub}}}^{{{sup}}}")
        return TextNode("")

    def visit_munder(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            base = self._visit(children[0])
            under = self._visit(children[1])
            # Check if base is an operator usually taking limits
            if "\\sum" in str(base) or "\\prod" in str(base) or "\\lim" in str(base):
                return BracedNode(f"{base}_{{{under}}}")
            return BracedNode(f"\\underset{{{under}}}{{{base}}}")
        return TextNode("")

    def visit_mover(self, element):
        children = self._get_clean_children(element)
        if len(children) == 2:
            base = self._visit(children[0])
            over = self._visit(children[1])
            return BracedNode(f"\\overset{{{over}}}{{{base}}}")
        return TextNode("")

    def visit_munderover(self, element):
        children = self._get_clean_children(element)
        if len(children) == 3:
            base = self._visit(children[0])
            under = self._visit(children[1])
            over = self._visit(children[2])
            return BracedNode(f"{{{base}}}_{{{under}}}^{{{over}}}")
        return TextNode("")

    def visit_mtable(self, element):
        rows = []
        for child in element.children:
            if child.name == "mtr":
                cells = []
                for cell in child.children:
                    if cell.name == "mtd":
                        cells.append(str(self._visit(cell)))
                rows.append(" & ".join(cells))
        content = " \\\\ ".join(rows)
        return BracedNode(f"\\begin{{matrix}} {content} \\end{{matrix}}")

    def visit_mfenced(self, element):
        open_char = element.get("open", "(")
        close_char = element.get("close", ")")
        separators = element.get("separators", ",")

        args = [str(self._visit(c)) for c in self._get_clean_children(element)]

        seps = list(separators)
        result = []

        for i, arg in enumerate(args):
            result.append(arg)
            if i < len(args) - 1:
                sep = seps[i] if i < len(seps) else seps[-1]
                if separators:
                    result.append(sep + " ")

        content = "".join(result)
        return BracedNode(f"\\left{open_char} {content} \\right{close_char}")

    def visit_mphantom(self, element):
        content = self._visit_children(element)
        return BracedNode(f"\\phantom{{{content}}}")


def mathml_to_markdown(mathml_node):
    converter = MathMLToLatex()
    return converter.convert(mathml_node)
