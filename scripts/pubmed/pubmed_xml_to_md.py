"""
file: pubmed_xml_to_md.py
--------------------------
Converts PubMed XML files to Markdown format.
"""

import os
import re
from collections import defaultdict
from tqdm import tqdm

from util import parse_pubmed_xml, parse_pubmed_paragraph


def xml_to_md(input_file, output_dir, remove_links=False):
    meta = parse_pubmed_xml(input_file)
    paragraphs = parse_pubmed_paragraph(input_file)

    meta_dict = {xml["pmid"]: xml for xml in meta}
    # join paragraphs by pmid
    pmid_to_para_jsons = defaultdict(list)
    for para in paragraphs:
        pmid_to_para_jsons[para["pmid"]].append(para)

    print("Number of unique pmmids: ", len(meta_dict))
    print("Number of unique pmids from paragraphs: ", len(pmid_to_para_jsons))

    for pmid in tqdm(pmid_to_para_jsons.keys()):
        assoc_meta = meta_dict[pmid]
        assoc_paragraphs = pmid_to_para_jsons[pmid]

        with open(os.path.join(output_dir, pmid + ".md"), "w") as f:
            f.write("# " + assoc_meta["full_title"] + "\n\n")
            f.write("## Abstract" + "\n\n" + assoc_meta["abstract"])

            # write section texts
            current_section_title = ""
            if not assoc_paragraphs:
                return None

            for paragraph in assoc_paragraphs:
                section_title = paragraph["section"]
                if section_title != current_section_title and section_title.strip():
                    f.write("\n\n## " + section_title)
                    current_section_title = section_title
                f.write("\n\n" + process_xml_paragraph(paragraph["text"].strip(), remove_links))


def process_xml_paragraph(xml_paragraph, remove_links=False):
    """
    Cleans and filters a paragraph from PubMed XML to Markdown format.
    """
    # remove empty paragraphs
    if not xml_paragraph:
        return None

    if remove_links:
        xml_paragraph = delete_links(xml_paragraph)

    return xml_paragraph


def delete_links(md_text):
    """
    Deletes links from the text of the following formats:
    - <sup xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML"><xref rid="R10" ref-type="bibr">10</xref></sup>
    - <italic xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML">P &lt;</italic>
    - <xref rid="R10" ref-type="bibr">10</xref>
    """
    # <sup xmlns:xlink...>
    clean_text = re.sub(r"<sup[^>]*xmlns:xlink[^>]*>.*?</sup>", "", md_text, flags=re.DOTALL)
    # <italic xmlns:xlink...>
    clean_text = re.sub(
        r"<italic[^>]*xmlns:xlink[^>]*>.*?</italic>", "", clean_text, flags=re.DOTALL
    )
    # <xref>
    clean_text = re.sub(r"<xref[^>]*>.*?</xref>", "", clean_text, flags=re.DOTALL)
    return clean_text


if __name__ == "__main__":
    DATA_DIRS = "data"
    xml_to_md(
        input_file=os.path.join(DATA_DIRS, "europe_pmc", "PMC7240001_PMC7250000.xml.gz"),
        output_dir=os.path.join(DATA_DIRS, "processed"),
        remove_links=True,
    )
