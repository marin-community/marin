import pubmed_parser
import re


def xml2json(xml_str):
    article_info = pubmed_parser.parse_pubmed_xml(xml_str)
    paragraphs = pubmed_parser.parse_pubmed_paragraph(xml_str)
    article_json = {"text": "", "id": f"pmc-{article_info['pmid']}", "source": "pubmed_central", "metadata": {}}
    article_json["text"] += "# " + article_info["full_title"] + "\n\n"
    article_json["text"] += "## Abstract" + "\n\n"
    article_json["text"] += article_info["abstract"]
    current_section_title = ""
    if not paragraphs:
        return None
    for paragraph in paragraphs:
        section_title = paragraph["section"]
        if section_title != current_section_title and section_title.strip():
            article_json["text"] += "\n\n## " + section_title
            current_section_title = section_title
        article_json["text"] += "\n\n" + paragraph["text"].strip()
    return article_json
