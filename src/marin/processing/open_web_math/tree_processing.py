#!/usr/bin/env python3
"""
This file was taken from https://github.com/keirp/OpenWebMath
"""
from resiliparse.parse.html import DOMCollection
from tabulate import tabulate

from .utils import has_style


def remove_buttons(tree):
    btns = tree.document.query_selector_all(".btn")
    for btn in btns:
        parent = btn.parent
        parent.remove_child(btn)
    # Remove any button tags
    btns = tree.document.query_selector_all("button")
    for btn in btns:
        parent = btn.parent
        if parent:
            parent.remove_child(btn)


def remove_links(tree):
    """Replace links with spans so that resiliparse doesn't try to remove them."""
    links = tree.document.query_selector_all("a")
    for link in links:
        parent = link.parent
        if parent is None:
            continue
        new_span = tree.create_element("span")
        new_span.text = link.text
        parent.replace_child(new_span, link)


def flatten(node):
    """Remove any divs or spans that only have one child and replace them with their child."""
    divs = node.query_selector_all("div")
    spans = node.query_selector_all("span")
    for div in divs:
        if len(div.child_nodes) == 1:
            parent = div.parent
            if parent is None:
                continue
            parent.replace_child(div.child_nodes[0], div)
    for span in spans:
        if len(span.child_nodes) == 1:
            parent = span.parent
            if parent is None:
                continue
            parent.replace_child(span.child_nodes[0], span)

    return node


def remove_dense_links(tree):
    """Remove lists that only have links."""
    # First, remove any nav elements to be safe.
    navs = tree.document.query_selector_all("nav")
    for nav in navs:
        parent = nav.parent
        if parent is None:
            continue
        parent.remove_child(nav)

    lists = tree.document.query_selector_all("ul, ol, div, span, nav, table, p")
    to_remove = []
    for _list in lists:
        if len(_list.child_nodes) == 0 or len(_list.child_nodes) == 1:
            continue
        children = _list.child_nodes
        links = _list.query_selector_all("a")
        total_children_text = "".join([x.text.strip() for x in children if type(x) is not DOMCollection])
        total_links_text = "".join([x.text.strip() for x in links])
        if len(total_children_text) == 0 or len(total_links_text) == 0:
            continue
        ratio = len(total_links_text) / len(total_children_text)
        if ratio > 0.8:
            parent = _list.parent
            if parent is None:
                continue
            to_remove.append(_list)

    for _list in to_remove:
        parent = _list.parent
        if parent is None:
            continue
        parent.remove_child(_list)


def remove_image_figures(tree):
    to_remove = []
    imgs = tree.document.query_selector_all("img")
    for img in imgs:
        cur_node = img
        while cur_node is not None:
            if cur_node.class_name == "figure":
                parent = cur_node.parent
                if parent:
                    to_remove.append(cur_node)
                break
            cur_node = cur_node.parent

    for node in to_remove:
        parent = node.parent
        if parent is None:
            continue
        parent.remove_child(node)


def remove_link_clusters(tree):
    # First, find all links that are in span blocks. If they have no siblings, delete the span.
    to_remove = []

    span_links = tree.document.query_selector_all("span a")
    for link in span_links:
        parent = link.parent
        if parent is None:
            continue
        n_siblings = 0
        for sibling in parent.child_nodes:
            if sibling.type == 1:
                n_siblings += 1
                break
        if n_siblings == 1:
            grandparent = parent.parent
            if grandparent is None:
                continue
            # grandparent.remove_child(parent)
            to_remove.append(parent)

    links = list(tree.document.query_selector_all("a"))

    i = 0
    while len(links) > 0:
        link = links[0]
        del links[0]
        parent = link.parent
        i += 1
        if parent is None or parent.parent is None:
            continue
        n_links = 0
        n_children = len(parent.child_nodes)
        child_links = parent.query_selector_all("a")
        if len(child_links) == n_children:
            for child_link in child_links:
                # Check if it's visible and not empty.
                empty = child_link.text is None or child_link.text.strip() == ""
                styles = child_link.getattr("style")
                visible = styles is None or not (
                    has_style("display: none", styles) or has_style("visibility: hidden", styles)
                )
                if visible and not empty:
                    n_links += 1
            multilink = n_links > 1 and n_children == n_links
            if multilink:
                grandparent = parent.parent
                if grandparent is None:
                    continue
                # grandparent.remove_child(parent)
                to_remove.append(parent)

    for node in to_remove:
        parent = node.parent
        if parent is None:
            continue
        parent.remove_child(node)


def extract_code(tree, replacement_manager):
    wp_syntax = tree.document.query_selector_all(".wp_syntax")
    codes = tree.document.query_selector_all("code")
    code_responsive = tree.document.query_selector_all(".code_responsive")
    pre_tags = tree.document.query_selector_all("pre")
    for code in [*wp_syntax, *codes, *code_responsive, *pre_tags]:
        multiline = code.text.count("\n") > 0
        if len(code.text) > 0:
            if multiline:
                code.text = replacement_manager.add_replacement(f"```{code.text}```", tag="code")
            else:
                code.text = replacement_manager.add_replacement(f"`{code.text}`", tag="code")


def extract_tables(node, replacement_manager, table_config):
    if table_config["format"] == "none":
        return
    # Don't worry about tables that have tables in them or have headers
    # tables = node.query_selector_all('table:not(:has(table *))')
    tables = node.query_selector_all("table:not(:has(table, h1, h2, h3, h4, h5, h6))")
    for table in tables:
        table_data = []
        headers = []
        # Find all headers
        ths = table.query_selector_all("th")
        for th in ths:
            headers.append(th.text)
        trs = table.query_selector_all("tr")
        for tr in trs:
            row_data = []
            tds = tr.query_selector_all("td")
            for td in tds:
                # Remove any scripts
                scripts = td.query_selector_all("script")
                for script in scripts:
                    script.parent.remove_child(script)
                # Get the text of each td element
                row_data.append(td.text)
                col_span = td.getattr("colspan")
                if col_span:
                    try:
                        col_span = int(col_span)
                        if col_span > 100:
                            continue
                    except ValueError:
                        continue
                    # Add empty cells for colspans
                    for _ in range(col_span - 1):
                        row_data.append("")
            table_data.append(row_data)
        if len(table_data) == 0 or len(table_data[0]) == 0:
            continue
        # Post processing
        # Make sure all rows have the same number of columns
        max_cols = max([len(row) for row in table_data])
        for row in table_data:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))
        # Strip all cells
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                table_data[i][j] = table_data[i][j].strip()
        # If any columns or rows are consistently empty, remove them
        # Remove empty columns
        empty_columns = []
        for i in range(len(table_data[0])):
            if all([len(row[i]) == 0 for row in table_data]):
                empty_columns.append(i)

        for i in reversed(empty_columns):
            for row in table_data:
                del row[i]
        # Remove empty rows
        table_data = [row for row in table_data if len(row) > 0]

        # Remove any newlines from the table
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                table_data[i][j] = table_data[i][j].replace("\n", " ")
        # Check that the table has at least one row and one column
        if len(table_data) >= table_config["min_rows"] and len(table_data[0]) >= table_config["min_cols"]:
            # Replace the table with a markdown
            parent = table.parent
            if parent:
                if len(headers) == 0:
                    headers = [""] * len(table_data[0])
                rendered_table = tabulate(table_data, tablefmt=table_config["format"], headers=headers)
                table.html = replacement_manager.add_replacement(rendered_table, tag="table")
        elif len(table_data) > 0 and len(table_data[0]) > 0:
            # Do the same but use a plain format
            # Replace the table with a markdown
            parent = table.parent
            if parent:
                if len(headers) == 0:
                    headers = [""] * len(table_data[0])
                rendered_table = tabulate(table_data, tablefmt="plain", headers=headers)
                table.html = replacement_manager.add_replacement(rendered_table, tag="table")
        else:
            # Remove empty tables
            if table.parent:
                table.parent.remove_child(table)

    return node


def extract_headings(tree, replacement_manager, markdown_formatting):
    header_to_format = {f"h{i}": f"[heading_{i}]" for i in range(1, 7)}
    to_remove = []
    for heading_tag in header_to_format:
        hs = tree.document.query_selector_all(heading_tag)
        for heading in hs:
            text = ""
            for child in heading.child_nodes:
                if child.text.strip() != "" and child.type != 8:
                    text += child.text
                    child.text = ""
            text = text.strip()
            if len(text) == 0:
                # remove the heading
                if heading.parent:
                    to_remove.append(heading)
                continue
            if markdown_formatting:
                heading.text = replacement_manager.add_replacement(
                    header_to_format[heading_tag] + " " + text + "\n\n", tag=heading_tag
                )
            else:
                heading.text = replacement_manager.add_replacement(text + "\n\n", tag=heading_tag)

    for heading in to_remove:
        parent = heading.parent
        if parent:
            parent.remove_child(heading)


def post_process_headings(text):
    """Replace [heading_i] with '#' * i"""
    for i in range(6, 0, -1):
        text = text.replace(f"[heading_{i}]", "#" * i)
    return text


def add_se_separators(tree):
    user_infos = tree.document.query_selector_all("table.fw")
    # Replace all of these with spans <span>-</span>
    for user_info in user_infos:
        new_span = tree.create_element("span")
        new_span.text = "-"
        parent = user_info.parent
        # Remove the table
        parent.remove_child(user_info)
        # Add the span
        parent.append_child(new_span)


def wikipedia_preprocess(tree):
    external_links = tree.document.query_selector("#External_links")
    if external_links:
        # Remove all next until nothing left
        node = external_links.parent.next
        while node:
            next_node = node.next
            node.parent.remove_child(node)
            node = next_node
        external_links.parent.remove_child(external_links)

    edit_buttons = tree.document.query_selector_all(".mw-editsection")
    for edit_button in edit_buttons:
        if edit_button.parent:
            edit_button.parent.remove_child(edit_button)


def remove_display_none(tree):
    # Remove all elements with display none
    elements = tree.document.query_selector_all('[style*="display:none"]')
    for element in elements:
        element.parent.remove_child(element)


def preserve_question_headers(tree):
    elements = tree.document.query_selector_all("#question-header")
    for element in elements:
        inner_h1 = element.query_selector("h1")
        if inner_h1:
            new_h1 = tree.create_element("h1")
            new_h1.text = inner_h1.text
            element.parent.replace_child(new_h1, element)


def main_content_preprocess(tree):
    """Make any changes that are necessary to maximize the performance
    of the resiliparse main_content=True option."""

    # Look for qa-main class
    qa_main = tree.document.query_selector(".qa-main")
    if qa_main:
        qa_main.setattr("class", "article-body")

    # If there is a role=main and a question-header class, add the question-header to the top of the role=main
    role_main = tree.document.query_selector('[role="main"]')
    if role_main:
        question_header = tree.document.query_selector("#question-header")
        if question_header:
            first_child = role_main.first_child
            if first_child:
                role_main.insert_before(question_header, first_child)

    post_content = tree.document.query_selector(".postcontent")
    if post_content:
        post_body = tree.document.query_selector(".postbody")
        if post_body:
            # Set the class of postbody to postcontent and remove the postcontent class
            post_body.setattr("class", "postcontent")
            post_content.setattr("class", "")

    # Find .postbit
    postbit = tree.document.query_selector(".postbit")
    if postbit:
        # Change the class to article-body
        postbit.setattr("class", "")

    # Find all ul and add a few wrapping divs to move them farther from the root node
    uls = tree.document.query_selector_all("ul")
    for ul in uls:
        # Create 4 nested divs and set the html of the last one to the html of the ul.
        # Then replace the ul with the last div
        div1 = tree.create_element("div")
        div2 = tree.create_element("div")
        div3 = tree.create_element("div")
        div4 = tree.create_element("div")
        div4.html = ul.html
        div3.append_child(div4)
        div2.append_child(div3)
        div1.append_child(div2)
        if ul.parent:
            ul.parent.replace_child(div1, ul)
