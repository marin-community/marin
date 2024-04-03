from .markdown import to_markdown

def convert_page(html: str):
    from readabilipy import simple_json_from_html_string

    reabilitied = simple_json_from_html_string(html, use_readability=True)
    tree = reabilitied["content"]
    markdown = to_markdown(tree)

    return {
        "title": reabilitied["title"],
        "content": markdown,
        "date": reabilitied["date"],
        "byline": reabilitied["byline"],
    }