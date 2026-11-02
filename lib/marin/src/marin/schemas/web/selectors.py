# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
