# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pubmed_parser


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
