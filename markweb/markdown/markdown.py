import logging
import os
import re

import markdownify
import six
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString
from markdownify import MarkdownConverter
import lxml.etree as ET

from regex import regex

import html2text

logger = logging.getLogger(__name__)

# TODOs:
# - [x] add more tests of core functionality (tables, lists, etc)
# - [x] add code block lang id
# - [x] add latex math support

def to_markdown(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    text = MyMarkdownConverter().convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ")
    return text


_global_html2text = html2text.HTML2Text()

_global_html2text.ignore_links = False
_global_html2text.body_width = 0
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
# Could also use some kind of PL lang-id to highlight code blocks, but probably not super necessary
_global_html2text.mark_code = True  # Optionally convert code blocks to markdown
_global_html2text.include_sup_sub = True  # Optionally include <sup> and <sub> tags
_global_html2text.pad_tables = False


def html2text_markdown(html):
    html = str(html)
    return _global_html2text.handle(html)


always_escape_pattern = re.compile(r"([\[\]`])")  # square brackets, backticks
line_start_escape_pattern = re.compile(r"^(\s*)([-+#]\s)", flags=re.MULTILINE)
# only escape backslashes before ascii punctuation. in gfm, other backslackes are literal
backslash_before_ascii_punct_pattern = re.compile(r'(\\[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~])', flags=re.ASCII)


# reference: https://github.github.com/gfm/

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



class MyMarkdownConverter(MarkdownConverter):

    def __init__(self, **kwargs):
        kwargs = {
            "heading_style": "ATX",
            "keep_inline_images_in": ["li", "p", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "a"],
            # "code_language_callback": self._infer_code_language,
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
        src = el.attrs.get('src', None) or el.attrs.get("data-src", None) or ''
        title = el.attrs.get('title', None) or ''
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        if (convert_as_inline
                and el.parent.name not in self.options['keep_inline_images_in']):
            return alt

        return '![%s](%s%s)' % (alt, src, title_part)

    def escape(self, text):
        return minimal_markdown_escape(text)

    def _infer_code_language(self, el):
        text = el.get_text()
        if not text:
            return None
        from .guess_code import predict
        lang = predict(text)[0][0]
        return lang.lower()



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

    def process_tag(self, node, convert_as_inline, children_only=False):
        # skip aria-hidden elements
        if node.get('aria-hidden') == 'true':
            return ''
        text = ''

        # markdown headings or cells can't include
        # block elements (elements w/newlines)
        isHeading = markdownify.html_heading_re.match(node.name) is not None
        isCell = node.name in ['td', 'th']
        convert_children_as_inline = convert_as_inline

        if not children_only and (isHeading or isCell):
            convert_children_as_inline = True

        # Remove whitespace-only textnodes in purely nested nodes
        def is_nested_node(el):
            return el and el.name in ['ol', 'ul', 'li',
                                      'table', 'thead', 'tbody', 'tfoot',
                                      'tr', 'td', 'th']

        if is_nested_node(node):
            for el in node.children:
                # Only extract (remove) whitespace-only text node if any of the
                # conditions is true:
                # - el is the first element in its parent
                # - el is the last element in its parent
                # - el is adjacent to an nested node
                can_extract = (not el.previous_sibling
                               or not el.next_sibling
                               or is_nested_node(el.previous_sibling)
                               or is_nested_node(el.next_sibling))
                if (isinstance(el, NavigableString)
                        and six.text_type(el).strip() == ''
                        and can_extract):
                    el.extract()

        # Convert the children first
        for el in node.children:
            if isinstance(el, Comment) or isinstance(el, Doctype):
                continue
            elif isinstance(el, NavigableString):
                next_text = self.process_text(el)
                text = self.join_text(text, next_text)
            else:
                text = self.join_text(text, self.process_tag(el, convert_children_as_inline))

        if not children_only:
            convert_fn = getattr(self, 'convert_%s' % node.name, None)
            if convert_fn and self.should_convert_tag(node.name):
                text = convert_fn(node, text, convert_as_inline)

        return text


    def join_text(self, text1, text2):
        # mostly want to remove extra newlines
        # in MD, two newlines is a paragraph break, which is the most we want
        # so if text1 already has two newlines, we don't want to add another
        if text1.endswith('\n\n'):
            if text2.startswith('\n\n'):
                text2 = text2[2:]
            elif text2.startswith('\n'):
                text2 = text2[1:]
        elif text1.endswith('\n'):
            if text2.startswith('\n\n'):
                text2 = text2[1:]
            elif text2.startswith('\n'):
                text2 = text2[0:]
        return text1 + text2

    def convert_math(self, el, text, convert_as_inline):
        try:
            x = mathml_to_markdown(el)
            return x
        except Exception as e:
            logger.exception(f"Error converting math: {e}")
            return text



_xslt_mml = None

_xslt_mml_path = os.path.join(os.path.dirname(__file__), "xsl_yarosh/mmltex.xsl")



# cf https://github.com/oerpub/mathconverter/blob/master/converter.py#L14
# (we've modified the xslt to output simpler markdown when possible
def mathml_to_markdown(mathml_node):
    global _xslt_mml
    if _xslt_mml is None:
        _xslt_mml = ET.parse(_xslt_mml_path)

    # mathml_node is a bs4 element. we need to convert it to an lxml element
    mml_str = str(mathml_node)
    # often times, the mathml doesn't have the xmlns, which lxml needs
    # need to also handle display being present or not
    # this is hacky but probably enough
    if 'xmlns' not in mml_str:
        if 'display' in mml_str:
            mml_str = mml_str.replace('<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"')
        else:
            # default is inline
            mml_str = mml_str.replace('<math', '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"')

    mml_dom = ET.fromstring(mml_str)

    transform = ET.XSLT(_xslt_mml)
    try:
        mml_dom = transform(mml_dom)
    except Exception as e:
        print(transform.error_log)
        raise e
    return str(mml_dom)
