# Convert's Trafilatura's XML back to HTML
# The idea is the Traf has canonincalized the doc, so we can convert it back to HTML
# and then convert it to markdown. This is slow, but...
import os
import sys

import fsspec

from markweb.markdown import html2text_markdown, to_markdown


if __name__ == '__main__':
    for orig_html_path in sys.argv[1:]:
        # being a little sneaky. not really doing crawling.
        with fsspec.open(orig_html_path, "r",
                         client_kwargs={
                             "headers": {
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                                }
                         }) as f:
            html = f.read()

        # out = trafilatura.bare_extraction(html, output_format="python",
        #                                   include_links=True, include_comments=True,
        #                                   include_images=True, include_tables=True,
        #                                   as_dict=False)

        if orig_html_path.endswith("/"):
            orig_html_path = orig_html_path[:-1]

        base_name = os.path.basename(orig_html_path)
        base_name = os.path.splitext(base_name)[0]

        # node = out.body
        # if not node:
        #     logger.error("Trafilatura did not return a body")
        # else:
        #
        #     ET.indent(node, space="  ")
        #     orig_node_string = ET.tostring(node, pretty_print=True).decode()
        #
        #     with open(f"{base_name}.xml", "w") as f:
        #         f.write(orig_node_string)
        #
        #     traf_xml_to_html(node)
        #     with open(f"{base_name}.traf.html", "w") as f:
        #         f.write(ET.tostring(node, pretty_print=True).decode())
        #
            # markdown = html_to_markdown(ET.tostring(node, pretty_print=True).decode())
            #
            # with open(f"{base_name}.traf.md", "w") as f:
            #     print(markdown, file=f)

        import readabilipy
        reabilitied = readabilipy.simple_json_from_html_string(html, use_readability=True)
        # readabilipy is beautifulsoup-ba
        # tree_str = ET.tostring(tree, pretty_print=True).decode()
        tree = reabilitied["content"]
        tree_str = str(tree)
        markdown2 = html2text_markdown(tree_str)

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


