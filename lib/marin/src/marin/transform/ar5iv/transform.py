# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BeautifulSoup cleanup helpers for ar5iv HTML.

These transforms are composed by :func:`marin.transform.ar5iv.transform_ar5iv.clean_html`
to strip academic-paper boilerplate before markdown extraction.
"""

from html import escape, unescape

from bs4 import BeautifulSoup


def transform_abstract(html: BeautifulSoup):
    # Transform the abstract from h6 to h2
    abstract = html.find_all("h6", {"class": "ltx_title_abstract"})
    for ab in abstract:
        ab.name = "h2"
    return html


def remove_authors(html: BeautifulSoup):
    # Remove authors since we only care about information after first section
    authors = html.find_all("div", {"class": "ltx_authors"})
    for author in authors:
        author.decompose()
        section = author.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.decompose()
            section = new_section
    return html


def remove_title_page(html: BeautifulSoup):
    # Remove title page since we only care about information after first section
    title_page = html.find_all("div", {"class": "ltx_titlepage"})
    for tp in title_page:
        tp.decompose()


def clean_li(html: BeautifulSoup):
    # Remove the li tags since they repeat the same information (eg 1. 1.)
    tags = html.find_all("span", {"class": "ltx_tag_item"})
    for tag in tags:
        tag.decompose()
    tags = html.find_all("span", {"class": "ltx_tag_listingline"})
    for tag in tags:
        tag.decompose()


def remove_biblio(html: BeautifulSoup):
    # Remove the biblio since there is a lot of noise
    biblio = html.find_all("section", {"id": "bib"})
    for bib in biblio:
        bib.decompose()


def remove_footnotes(html: BeautifulSoup):
    # Remove footnotes since they are plopped in the middle of the text
    footnotes = html.find_all("div", {"class": "ltx_role_footnote"})
    for fn in footnotes:
        fn.decompose()


def remove_biblinks(html: BeautifulSoup):
    # Remove the biblinks since we are removing the biblio
    biblinks = html.find_all("a", {"class": "ltx_ref"})
    for biblink in biblinks:
        # Removes reference links
        # biblink.decompose()
        # Removes linking but keeps text
        biblink.unwrap()


def remove_references(html: BeautifulSoup):
    # Remove the reference section
    references = html.find_all("section", {"id": "ltx_bibliography"})
    for ref in references:
        ref.decompose()

    # Remove the references section
    references = html.find_all("ul", {"class": "ltx_biblist"})
    for ref in references:
        ref.decompose()


def linelisting_to_newline(html: BeautifulSoup):
    # Turn new line listings into new lines
    linelisting = html.find_all("div", {"class": "ltx_listingline"})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))


def unwrap_eqn(page: BeautifulSoup) -> BeautifulSoup:
    """
    Extract alttext from math element and convert to LaTeX format.
    Returns BeautifulSoup object with the formatted equation.
    """
    math_elements = page.find_all("math")

    for math_elem in math_elements:
        if not math_elem or "alttext" not in math_elem.attrs:
            continue

        equation = str(math_elem["alttext"])
        equation = unescape(equation)
        equation = equation.replace("\\", "\\\\")
        equation = equation.replace("<", r"\<")
        equation = equation.replace(">", r"\>")

        # HTML-escape the equation to prevent < and > from being interpreted as tags
        # This is critical for equations like $T_{0}\<T\<6$ where \< would otherwise
        # be parsed as an opening tag by HTML parsers like lxml/resiliparse
        equation = escape(equation)

        is_display = math_elem.get("display") == "block"

        if is_display:
            formatted_eq = BeautifulSoup(f"<p><br><br>$${equation}$$<br><br></p>", "html.parser")
        else:
            formatted_eq = BeautifulSoup(f"${equation}$", "html.parser")

        math_elem.replace_with(formatted_eq)

    return page


def deconstruct_eqn(html: BeautifulSoup):
    # Unwrap equation tables to ensure math mode is not in a table
    eqntables = html.find_all("table", {"class": "ltx_eqn_table"})
    for eqn in eqntables:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqnrows = html.find_all("tr", {"class": "ltx_eqn_row"})
    for eqn in eqnrows:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()

    eqncell = html.find_all("td", {"class": "ltx_eqn_cell"})
    for eqn in eqncell:
        eqn.unwrap()


def remove_ar5iv_footer(html: BeautifulSoup):
    # This is the ar5iv footer generated on xyz date
    footer = html.find_all("footer")
    for fn in footer:
        fn.decompose()


def remove_before_section(html: BeautifulSoup):
    # We only care about information after the first section
    section = html.find("section")
    if section:
        section = section.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.extract()
            section = new_section


def remove_figure_captions(html: BeautifulSoup):
    # Remove the figure captions since they are not needed
    captions = html.find_all("figcaption", {"class": "ltx_caption"})
    for caption in captions:
        caption.decompose()
