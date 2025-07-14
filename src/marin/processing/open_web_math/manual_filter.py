#!/usr/bin/env python3
"""
This file is a lightly-modified version of:
https://github.com/keirp/OpenWebMath/blob/main/filtering/filter_dataset.py
"""
import re

BAD_URLS = [
    "worldwidescience",
    "science.gov",
    "archive.org",
    "scribd.com",
    "unz.com",
    "/profile/",
    "/researcher",
    "noobstarter.com",
    "philpapers.org",
    "thesa.com",
    "beyondhighbrow.com",
    "careyoukeep.com",
    "eevblog.com",
    "happyslide.net",
    "issuu.com",
    "zh-cn.unz.com",
    "vixra.org",
    "medcraveonline.com",
    "sciendo.com",
    "open.library.ubc.ca",
    "eurotrib.com",
    "postthreads.org",
    "jim.bmj.com",
    "wanweibaike.com",
    "hzdr.de",
    "/joursearch/",
    "docplayer.net",
    "bookofmormonblog.org",
    "bradford-delong.com",
    "profiles.stanford.edu",
    "vo.astronet.ru",
    "homainstallationen.at",
    "/author/",
    "/authors/" "/serials/" "read.dukeupress.edu",
    "thewikipost.org",
    "is.tuebingen.mpg.de",
    "discourse.darkjedibrotherhood.com",
    "springermedizin.de",
    "materials-chain.com",
    "www.unzmag.net",
    "is.mpg.de",
    "hobby8.5ch.net",
    "forums.penny-arcade.com",
    "wowwiki.com",
    "8chan.moe",
    "plosone.org",
    "www.is.mpg.de",
    "feeds.churchturing.org",
    "learn.gcs.edu",
    "mobinuke.com",
    "judithcurry.com",
    "tek-tips.com",
    "skepticforum.com",
    "all_publications",
    ".de/publications",
    "nih.gov",
    "lastfm.it",
    "/commit",
    "vitaminstore",
    "studylib.net",
    "dokumen.pub",
    "manualzz.com",
    "fraser.stlouisfed.org",
]

libretext_good = [
    "math",
    "phys",
    "stats",
]

accented_chars = set(["ü", "ï", "ö", "ê", "ä", "â", "ê", "î", "û", "ô", "è", "é", "à"])


def has_accented_char(text):
    num_accent = sum([c in accented_chars for c in text.lower()])
    if len(text) == 0:
        return False
    return num_accent / len(text) > 0.015


def count_latex_formulas(text):
    # Remove unwanted patterns
    cleaned_text = re.sub(r"\$\$\\PageIndex\{[^\}]*\}\$\$", "", text)
    cleaned_text = re.sub(r"\$\\PageIndex\{[^\}]*\}\$", "", cleaned_text)

    # Pattern for inline and display math
    pattern = r"\$\$[^\$]*\$\$|\$[^\$]*\$"

    matches = re.findall(pattern, cleaned_text)

    return len(matches)


def is_good_url(url: str):
    if "arxiv-vanity" in url:
        return False
    # Check if /search is in the path
    if "/search" in url and "//search" not in url:
        return False
    if "proceedings" in url:
        return False
    if "bibbase" in url:
        return False
    if "nrsworld.com" in url:
        return False
    if "bibtex" in url:
        return False
    if "issn" in url:
        return False
    if "arxiv-export" in url:
        return False
    if "bmjopen" in url:
        return False
    if "stackexchange.com/users" in url:
        return False
    if "mathoverflow.net/users" in url:
        return False
    return True


def manual_url_filter(url: str, original_text: str) -> tuple[bool, str]:
    """
    Given the URL and text of an example, run the open-web-math manual filters and rules.

    Returns a tuple of (bool, str). The first tuple item denotes whether or not
    the example passes the filter (i.e., True if the example should be kept) and
    the second tuple item returns the modified text of the example.

    Parameters:
    url (str): URL of the example.
    original_text (str): Original text of the example.
    """
    should_filter = not is_good_url(url) or any(bad_url in url for bad_url in BAD_URLS)

    # Remove any line that has more than one "newcommand"
    lines = original_text.split("\n")
    new_lines = []
    for line in lines:
        if line.count("newcommand") > 1:
            continue
        new_lines.append(line)
    new_text = "\n".join(new_lines)

    # Filter less than 100 characters
    if len(new_text) < 100:
        should_filter = True

    if "libretexts" in url:
        # Check if the url is part of the whitelist
        is_whitelist = False
        for good in libretext_good:
            if good in url:
                is_whitelist = True
                break
        if not is_whitelist:
            # Throw out if 0 math formulas
            if count_latex_formulas(new_text) == 0:
                should_filter = True

    # Filter out accents
    if has_accented_char(new_text):
        should_filter = True

    passes_filters = not should_filter
    return (passes_filters, new_text)
