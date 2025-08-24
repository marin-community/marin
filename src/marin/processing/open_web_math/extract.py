#!/usr/bin/env python3
"""
This file was taken from https://github.com/keirp/OpenWebMath
"""
import os
import re

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree

from .latex_processing import extract_delimited_math, extract_math, get_math_config, replace_math_tags_with_dollar_signs
from .line_processing import remove_boilerplate, remove_chinese_characters, remove_edit_buttons, remove_empty_headers
from .tree_processing import (
    add_se_separators,
    extract_code,
    extract_headings,
    extract_tables,
    main_content_preprocess,
    post_process_headings,
    remove_buttons,
    remove_dense_links,
    remove_display_none,
    remove_image_figures,
    wikipedia_preprocess,
)
from .utils import ReplacementManager

selectors_path = os.path.join(os.path.dirname(__file__), "banned_selectors.txt")
with open(selectors_path, "r") as f:
    selectors = [line.replace("\n", "").strip() for line in f]
    # Remove empty lines
    selectors = [line for line in selectors if line]


def filter_tree(tree, replacement_manager, config):
    """Filters the HTML tree to remove unwanted elements."""

    # Remove display none elements
    remove_display_none(tree)

    # Remove the wikipedia footer
    wikipedia_preprocess(tree)

    if config["remove_buttons"]:
        # Remove any bootstrap buttons
        remove_buttons(tree)

    if config["remove_image_figures"]:
        # Remove any figures that only contain images
        remove_image_figures(tree)

    if config["markdown_code"]:
        # Wrap the code in markdown code blocks
        extract_code(tree, replacement_manager)

    # Record the location of headings and format them
    extract_headings(tree, replacement_manager, config["markdown_headings"])

    # Remove link lists
    remove_dense_links(tree)

    # Format tables
    extract_tables(tree.document, replacement_manager, config["table_config"])

    # Process stack exchange separators
    add_se_separators(tree)

    # Preprocess main content
    main_content_preprocess(tree)

    return tree


def html_preprocessing(html):
    html = html.replace("&lt;math&gt;", "[itex]")
    html = html.replace("&lt;/math&gt;", "[/itex]")
    return html


def replace_tags(html, old, new):
    pattern = re.compile(old, re.IGNORECASE)
    return pattern.sub(new, html)


def extract_text(html, config, fast=False):
    """Extracts plain text from an HTML string."""
    html = replace_tags(html, "<template", "<div")
    html = replace_tags(html, "</template", "</div")
    html = replace_tags(html, "<frameset", "<div")
    html = replace_tags(html, "</frameset>", "</div>")
    html = html_preprocessing(html)
    tree = HTMLTree.parse(html)
    replacement_manager = ReplacementManager()

    if fast:
        links = tree.document.query_selector_all("a")
        span_links = tree.document.query_selector_all("span a")
        if len(links) > 3000 or len(span_links) > 3000:
            print("Too many links, skipping")
            return None

    if config["extract_latex"]:
        math_config = get_math_config(tree.document.html)
        tree, info = extract_math(tree, replacement_manager)
    else:
        info = {}
    tree = filter_tree(tree, replacement_manager, config)

    # Disable their filters because we use our own.
    text = extract_plain_text(tree, main_content=True, alt_texts=False, skip_elements=selectors)

    if config["extract_latex"]:
        text = extract_delimited_math(text, math_config, info, replacement_manager)

    text = post_process_headings(text)

    lines = text.split("\n")

    if config["remove_chinese"]:
        # Remove Chinese characters
        lines = remove_chinese_characters(lines)

    if config["boilerplate_config"]["enable"]:
        # Remove boilerplate
        lines = remove_boilerplate(lines, config["boilerplate_config"], replacement_manager)

    # Remove headings with nothing (or only other headings) after
    lines = remove_empty_headers(lines, replacement_manager)

    # Strip lines
    lines = [line.strip() for line in lines]

    # Create the final string
    text = "\n".join(lines)

    # Escape any dollar signs in the text
    text = text.replace("$", "\\$")

    # Now, add the dollar signs for math
    text = replace_math_tags_with_dollar_signs(text)

    if config["remove_edit_buttons"]:
        # Remove edit buttons
        lines = text.split("\n")
        lines = remove_edit_buttons(lines)
        text = "\n".join(lines)

    # If there are over two newlines in a row, replace with two
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = replacement_manager.remove_tags(text)

    text = text.strip()

    return text, info
