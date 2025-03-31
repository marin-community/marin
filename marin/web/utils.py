from bs4 import BeautifulSoup

from marin.markdown.markdown import to_markdown
from marin.schemas.web.convert import HtmlToMarkdownConfig


def extract_content_from_dom(
    html: str,
    kwargs: dict,
    markdownify_config: HtmlToMarkdownConfig,
) -> str:
    """
    This function extracts the main content DOM from the HTML content. We have a custom fork of Resiliparse at
    https://github.com/krypticmouse/chatnoir-resiliparse/tree/develop/resiliparse which modifies the `extract_plain_text`
    method to return the main content DOM instead of plain text.

    This method then converts the main content DOM to markdown using the `to_markdown` method via markdownify.

    Parameters:
        html (str): HTML content to extract.
        kwargs (dict): Keyword arguments to pass to the `extract_plain_text` method.
        markdownify_config (HtmlToMarkdownConfig): Configuration for markdownify.

    Returns:
        str: Markdown content of the main content DOM.

    NOTE: This method is using as custom fork of Resiliparse that is not meant to be merged into the main repository of
    Resiliparse. This is a custom modification for the purpose of this experiment. So, this method will not work with
    the main Resiliparse package. No plans to merge this into the main Resiliparse package yet.
    """

    from resiliparse_dom.extract.html2text import extract_simplified_dom

    tree = extract_simplified_dom(html, **kwargs)
    tree = BeautifulSoup(str(tree), "html.parser")

    # convert to markdown
    markdown = to_markdown(tree, markdownify_config)
    return markdown.replace("\x00", "").strip()
