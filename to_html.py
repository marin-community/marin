# Convert's Trafilatura's XML back to HTML
# The idea is the Traf has canonincalized the doc, so we can convert it back to HTML
# and then convert it to markdown. This is slow, but...
import logging
import os
import sys

import fsspec
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



import html2text

_global_markdown_converter = html2text.HTML2Text()
_global_markdown_converter.ignore_links = False
_global_markdown_converter.body_width = 0
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
# Could also use some kind of PL lang-id to highlight code blocks, but probably not super necessary
_global_markdown_converter.mark_code = True  # Optionally convert code blocks to markdown
_global_markdown_converter.include_sup_sub = True  # Optionally include <sup> and <sub> tags
_global_markdown_converter.pad_tables = False


def html_to_markdown(html):
    html = str(html)
    return _global_markdown_converter.handle(html)



if __name__ == '__main__':
    orig_html_path = sys.argv[1]

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


