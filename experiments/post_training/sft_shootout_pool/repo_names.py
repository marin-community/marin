# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import names of the SWE-smith repos in DCAgent's packages, for clean-copy probes and trace filters.

A clean copy of a repo outside /testbed shows up under these names (a site-packages or _vendor directory, a module
file, or a Go module cache entry). Go repos have no Python import name; their copies live under
/go/pkg/mod/github.com/<owner>/<name>@<version>.
"""

IMPORT_NAMES = {
    "Cog-Creators/Red-DiscordBot": ["redbot"],
    "Knio/dominate": ["dominate"],
    "PyCQA/flake8": ["flake8"],
    "Suor/funcy": ["funcy"],
    "agronholm/exceptiongroup": ["exceptiongroup"],
    "agronholm/typeguard": ["typeguard"],
    "andialbrecht/sqlparse": ["sqlparse"],
    "ariga/atlas": [],
    "bottlepy/bottle": ["bottle"],
    "c-bata/go-prompt": [],
    "caarlos0/env": [],
    "cantools/cantools": ["cantools"],
    "chardet/chardet": ["chardet"],
    "cknd/stackprinter": ["stackprinter"],
    "google/textfsm": ["textfsm"],
    "hjson/hjson-go": [],
    "hukkin/tomli": ["tomli"],
    "jaraco/inflect": ["inflect"],
    "jd/tenacity": ["tenacity"],
    "joke2k/faker": ["faker"],
    "keleshev/schema": ["schema"],
    "lepture/mistune": ["mistune"],
    "luozhouyang/python-string-similarity": ["strsimilarity"],
    "madzak/python-json-logger": ["pythonjsonlogger"],
    "mahmoud/boltons": ["boltons"],
    "marshmallow-code/webargs": ["webargs"],
    "matthewwithanm/python-markdownify": ["markdownify"],
    "pandas-dev/pandas": ["pandas"],
    "paramiko/paramiko": ["paramiko"],
    "pdfminer/pdfminer.six": ["pdfminer"],
    "pudo/dataset": ["dataset"],
    "pygments/pygments": ["pygments"],
    "pytest-dev/iniconfig": ["iniconfig"],
    "python-openxml/python-docx": ["docx"],
    "pyupio/safety": ["safety"],
    "rsalmei/alive-progress": ["alive_progress"],
    "seatgeek/thefuzz": ["thefuzz", "fuzzywuzzy"],
    "oauthlib/oauthlib": ["oauthlib"],
}
