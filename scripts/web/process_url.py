# Convert's Trafilatura's XML back to HTML
# The idea is the Traf has canonincalized the doc, so we can convert it back to HTML
# and then convert it to markdown. This is slow, but...
import os
import sys

import fsspec

import marin
import marin.web as web
from marin.markdown import to_markdown


if __name__ == "__main__":
    out_path = "output"
    os.makedirs(out_path, exist_ok=True)
    for url in sys.argv[1:]:
        # being a little sneaky. not really doing crawling.
        with fsspec.open(
            url,
            "r",
            client_kwargs={
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                }
            },
        ) as f:
            html = f.read()

        # out = trafilatura.bare_extraction(html, output_format="python",
        #                                   include_links=True, include_comments=True,
        #                                   include_images=True, include_tables=True,
        #                                   as_dict=False)

        html_path = url
        if html_path.endswith("/"):
            html_path = html_path[:-1]
        if "?" in url:
            maybe_html_path = url.split("?")[0]
            if maybe_html_path:
                html_path = maybe_html_path

        base_name = os.path.basename(html_path)
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

        out = web.convert_page(html, url=url)
        title = out["title"]
        md = out["content"]

        with open(f"{out_path}/{base_name}.orig.html", "w") as f:
            print(html, file=f)

        with open(f"{out_path}/{base_name}.readability.html", "w") as f:
            print(out["html"], file=f)

        with open(f"{out_path}/{base_name}.md", "w") as f:
            print(f"# {title}\n", file=f)
            print(md, file=f)
