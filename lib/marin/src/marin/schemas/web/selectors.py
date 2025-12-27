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

"""
CSS selectors for removing boilerplate elements from HTML during extraction.
"""

# Selectors to remove from Wikipedia DOM tree. These mostly contain footers, headers,
# navigation elements, reference sections, link clusters, filler sections, etc.
WIKI_BLACKLISTED_SELECTORS = [
    "div.navbox",
    "span.portal-bar",
    "div#catlinks",
    "h2#References",
    "h2#External_links",
    "h2#See_also",
    "div#p-navigation",
    "span.mw-editsection",
    "h2.Further_reading",
    "header",
    "a.mw-jump-link",
    "div.printfooter",
    "div.vector-header-container",
    ".noprint",
    "span.mw-cite-backlink",
    "sup.reference",
    "div#mw-indicators",
    "span.portal-barion",
    "h2#Notes",
    "h3#Sources",
    "ol.references",
    "div#mw-indicator-coordinates",
]

# Selectors to remove from arXiv DOM tree. These mostly contain reference sections,
# authors, and title sections (prepended manually to avoid duplication).
ARXIV_BLACKLISTED_SELECTORS = [
    "h2.ltx_title_bibliography",
    "div.ltx_classification",
    "span.ltx_role_author",
    "h1.ltx_title",
]
