import pubmed_parser
import re


def xml2md(xml_str):
    article_info = pubmed_parser.parse_pubmed_xml(xml_str)
    paragraphs = pubmed_parser.parse_pubmed_paragraph(xml_str)
    text = ""
    text += "# " + article_info["full_title"] + "\n\n"
    text += "## Abstract" + "\n\n"
    text += article_info["abstract"]
    current_section_title = ""
    if not paragraphs:
        return None
    for paragraph in paragraphs:
        section_title = paragraph["section"]
        if section_title != current_section_title and section_title.strip():
            text += "\n\n## " + section_title
            current_section_title = section_title
       text += "\n\n" + paragraph["text"].strip()
    return text
