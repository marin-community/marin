# Convert's Trafilatura's XML back to HTML
# The idea is the Traf has canonincalized the doc, so we can convert it back to HTML
# and then convert it to markdown. This is slow, but...
import logging
import os
import re
import sys

import fsspec
import markdownify
from regex import regex

import html2text
from markdownify import MarkdownConverter
import lxml.etree as ET
import trafilatura
from trafilatura.htmlprocessing import REND_TAG_MAPPING


REVERSE_REND_TAG_MAPPING = {v: k for k, v in REND_TAG_MAPPING.items()}

logger = logging.getLogger(__name__)


def traf_xml_to_html(node):
    # gross mutation, but I think it's fine.
    for elem in node.iter():
        match elem.tag:
            case "body":
                pass
            case "div":
                pass
            case "head":
                pass
            case "header":
                elem.tag = "h2"
            case "lb":
                elem.tag = "br"
            case "h1" | "h2" | "h3" | "h4" | "h5" | "h6":
                pass
            case "ab":
                # ab can be header
                rend = elem.get("rend")
                match rend:
                    case "h1" | "h2" | "h3" | "h4" | "h5" | "h6":
                        elem.tag = rend
                    case _:
                        elem_s = ET.tostring(elem, strip_text=True).decode()
                        raise ValueError(f"Unknown rend value {rend} for {elem_s}")
            case "ref": # links
                elem.tag = "a"
                elem.set("href", elem.get("target"))
                elem.attrib.pop("target")
            case "hi": # bold etc
                rend = elem.get("rend")
                tag = REVERSE_REND_TAG_MAPPING[rend]
                elem.tag = tag
                elem.pop("rend")
            case "list":
                rend = elem.get("rend")
                elem.tag = rend
            case "item":
                elem.tag = "li"
            case "graphic": # images
                elem.tag = "img"
            case "code":
                pass
            case "quote":
                elem.tag = "blockquote"
            case "del":
                elem.tag = "s"
            case "table":
                pass
            case "row":
                elem.tag = "tr"
            case "p":
                pass
            case "cell":
                role = elem.get("role")
                if role == "head":
                    elem.tag = "th"
                else:
                    elem.tag = "td"
            case _ :
                logger.warning(f"Unknown tag {elem.tag}")




_global_html2text = html2text.HTML2Text()
_global_html2text.ignore_links = False
_global_html2text.body_width = 0
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
# Could also use some kind of PL lang-id to highlight code blocks, but probably not super necessary
_global_html2text.mark_code = True  # Optionally convert code blocks to markdown
_global_html2text.include_sup_sub = True  # Optionally include <sup> and <sub> tags
_global_html2text.pad_tables = False


def html_to_markdown(html):
    html = str(html)
    return _global_html2text.handle(html)


# Pattern matching


import re

# Pre-compile the regular expressions
always_escape_pattern = re.compile(r"([\[\]`])")  # square brackets, backticks
line_start_escape_pattern = re.compile(r"^(\s*)([-+#]\s)", flags=re.MULTILINE)
backslash_before_ascii_punct_pattern = re.compile(r'(\\[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~])', flags=re.ASCII)


# this complicated crap is for **emphasis** and _emphasis_ etc.

# A delimiter run is either a sequence of one or more * characters that is not preceded or followed by a non-backslash-escaped * character, or a sequence of one or more _ characters that is not preceded or followed by a non-backslash-escaped _ character.
delimiter_run_pattern = regex.compile(r"""(?:
    (?<!\\[*])  # Assert not preceded by a backslash '\*'
    [*]+       # Match one or more '*
    (?!\\[*])   # Assert not followed by a backslash '\*'
    |
    (?<!\\[_])  # Assert not preceded by a backslash '\_'
    [_]+       # Match one or more '_'
    (?!\\[_])   # Assert not followed by a backslash '\_'
    )""", re.VERBOSE)

    

# A left-flanking delimiter run is a delimiter run that is (1) not followed by Unicode whitespace, and either
# (2a) not followed by a punctuation character, or
# (2b) followed by a punctuation character and preceded by Unicode whitespace or a punctuation character.
# For purposes of this definition, the beginning and the end of the line count as Unicode whitespace.

# Restated:
# followed by a non-space-non-punctuation character OR
# preceded by a space, punctuation character, or bol and followed by a punctuation character

left_flanking_pattern = regex.compile(r"""(?:(?:{}(?=[^\s\p{{P}}]))|(?:(?<=[\s\p{{P}}]|^){}(?:\p{{P}})))""".format(delimiter_run_pattern.pattern, delimiter_run_pattern.pattern), re.VERBOSE)

# A right-flanking delimiter run is a delimiter run that is (1) not preceded by Unicode whitespace, and either
# (2a) not preceded by a punctuation character, or
# (2b) preceded by a punctuation character and followed by Unicode whitespace or a punctuation character.
# For purposes of this definition, the beginning and the end of the line count as Unicode whitespace.

# Restated:
# preceded by a non-space-non-punctuation character OR
# followed by a space, punctuation character, or eol and preceded by a punctuation character
right_flanking_pattern = regex.compile(r"""(?:(?:(?<=[^\s\p{{P}}]){})|(?:(?<=\p{{P}}){}(?=\p{{P}}|\s|$)))""".format(delimiter_run_pattern.pattern, delimiter_run_pattern.pattern))

flanking_pattern = regex.compile(r"({}|{})".format(left_flanking_pattern.pattern, right_flanking_pattern.pattern), re.VERBOSE)



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

    # this has to come first because it will escape the other characters
    text = backslash_before_ascii_punct_pattern.sub(r"\\\1", text)
    text = line_start_escape_pattern.sub(r"\1\\\2", text)
    text = flanking_pattern.sub(r"\\\1", text)
    text = always_escape_pattern.sub(r"\\\1", text)


    return text



def to_markdown(tree):
    return MyMarkdownConverter().convert(tree)


# reference: https://github.github.com/gfm/
class MyMarkdownConverter(MarkdownConverter):

    def __init__(self, **kwargs):
        kwargs = {
            "heading_style": "ATX",
            **kwargs
        }
        super().__init__(**kwargs)

    # markdownify doesn't allow difference pre- and post- text for converting sub and sup
    def convert_sub(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"<sub>{text}</sub>"

    def convert_sup(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"<sup>{text}</sup>"

    def convert_img(self, el, text, convert_as_inline):
        # mostly copied from the parent class
        # the gfm spec says that the alt text is markdown, so we need to escape it
        alt = el.attrs.get('alt', None) or ''
        alt = self.escape(alt)
        src = el.attrs.get('src', None) or ''
        title = el.attrs.get('title', None) or ''
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        if (convert_as_inline
                and el.parent.name not in self.options['keep_inline_images_in']):
            return alt

        return '![%s](%s%s)' % (alt, src, title_part)

    def escape(self, text):
        return minimal_markdown_escape(text)


    def convert_figure(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text

        # the super doesn't handle this specifically. we basically want to be sure there's a newline
        if not text.endswith('\n\n'):
            if not text.endswith('\n'):
                text += '\n\n'
            else:
                text += '\n'
        return text



    def convert_tr(self, el, text, convert_as_inline):
        # this is also mostly copied from the parent class
        # but the logic for guessing a th isn't quite right
        cells = el.find_all(['td', 'th'])
        is_headrow = all([cell.name == 'th' for cell in cells])

        # we can be a headrow if we are the first row in the table or if all our cells are th
        # find table parent
        if not is_headrow:
            parent = el.parent
            while parent and parent.name != 'table':
                parent = parent.parent

            if parent:
                first_row = parent.find('tr')
                if first_row is el:
                    is_headrow = True

        overline = ''
        underline = ''
        if is_headrow and not el.previous_sibling:
            # first row and is headline: print headline underline
            underline += '| ' + ' | '.join(['---'] * len(cells)) + ' |' + '\n'
        elif (not el.previous_sibling
              and (el.parent.name == 'table'
                   or (el.parent.name == 'tbody'
                       and not el.parent.previous_sibling))):
            # first row, not headline, and:
            # - the parent is table or
            # - the parent is tbody at the beginning of a table.
            # print empty headline above this row
            overline += '| ' + ' | '.join([''] * len(cells)) + ' |' + '\n'
            overline += '| ' + ' | '.join(['---'] * len(cells)) + ' |' + '\n'
        return overline + '|' + text + '\n' + underline


if __name__ == '__main__':
    for orig_html_path in sys.argv[1:]:
        with fsspec.open(orig_html_path, "r") as f:
            html = f.read()

        out = trafilatura.bare_extraction(html, output_format="python",
                                          include_links=True, include_comments=True,
                                          include_images=True, include_tables=True,
                                          as_dict=False)

        if orig_html_path.endswith("/"):
            orig_html_path = orig_html_path[:-1]

        base_name = os.path.basename(orig_html_path)
        base_name = os.path.splitext(base_name)[0]

        node = out.body
        if not node:
            logger.error("Trafilatura did not return a body")
        else:

            ET.indent(node, space="  ")
            orig_node_string = ET.tostring(node, pretty_print=True).decode()

            with open(f"{base_name}.xml", "w") as f:
                f.write(orig_node_string)

            traf_xml_to_html(node)
            with open(f"{base_name}.traf.html", "w") as f:
                f.write(ET.tostring(node, pretty_print=True).decode())

            markdown = html_to_markdown(ET.tostring(node, pretty_print=True).decode())

            with open(f"{base_name}.traf.md", "w") as f:
                print(markdown, file=f)

        import readabilipy
        reabilitied = readabilipy.simple_json_from_html_string(html, use_readability=True)
        # readabilipy is beautifulsoup-ba
        # tree_str = ET.tostring(tree, pretty_print=True).decode()
        tree = reabilitied["content"]
        tree_str = str(tree)
        markdown2 = html_to_markdown(tree_str)

        title = reabilitied["title"]

        with open(f"{base_name}.readability.html", "w") as f:
            f.write(tree_str)

        with open(f"{base_name}.readability.md", "w") as f:
            print(f"# {title}\n", file=f)
            print(markdown2, file=f)

        md = to_markdown(tree)
        with open(f"{base_name}.readability.markdownify.md", "w") as f:
            print(f"# {title}\n", file=f)
            print(md, file=f)


