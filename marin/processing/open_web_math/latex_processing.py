#!/usr/bin/env python3
"""
This file was taken from https://github.com/keirp/OpenWebMath
"""
import html
import json
import logging
import os
import re
from urllib.parse import unquote

from lxml import etree as ET
from py_asciimath.translator.translator import ASCIIMath2Tex
from resiliparse.parse.html import traverse_dom

logging.getLogger().setLevel(logging.INFO)

color_regex = re.compile(r"\\textcolor\[.*?\]\{.*?\}")

asciimath2tex = ASCIIMath2Tex(log=False)

PARAGRAPH_TAGS = frozenset(
    {
        "body",
        "blockquote",
        "caption",
        "center",
        "col",
        "colgroup",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "form",
        "legend",
        "optgroup",
        "option",
        "p",
        "pre",
        "table",
        "td",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

latex_math_commands = [
    "\\end",
    "\\begin",
    "\\ref",
    "\\frac",
    "\\label",
    "\\bf",
    "\\right",
    "\\left",
    "\\rm",
    "\\alpha",
    "\\mu",
    "\\def",
    "\\it",
    "\\pi",
    "\\sigma",
    "\\sum",
    "\\lambda",
    "\\beta",
    "\\nu",
    "\\partial",
    "\\int",
    "\\delta",
    "\\rho",
    "\\phi",
    "\\gamma",
    "\\omega",
    "\\over",
    "\\nonumber",
    "\\bar",
    "\\sqrt",
    "\\theta",
    "\\tau",
    "\\em",
    "\\rangle",
    "\\hat",
    "\\tilde",
    "\\cal",
    "\\hline",
    "\\item",
    "\\psi",
    "\\vec",
    "\\langle",
    "\\epsilon",
    "\\eta",
    "\\cdot",
    "\\in",
    "\\xi",
    "\\infty",
    "\\quad",
    "\\mathcal",
    "\\times",
    "\\emph",
    "\\mathbf",
    "\\prime",
    "\\be",
    "\\mathrm",
    "\\ee",
    "\\vspace",
    "\\pm",
    "\\chi",
    "\\ell",
    "\\text",
    "\\qquad",
    "\\noindent",
    "\\to",
    "\\varphi",
    "\\hspace",
    "\\leq",
    "\\cos",
    "\\eqref",
    "\\overline",
    "\\sin",
    "\\kappa",
    "\\hbox",
    "\\rightarrow",
    "\\varepsilon",
    "\\textit",
    "\\dagger",
    "\\big",
    "\\otimes",
    "\\equiv",
    "\\zeta",
    "\\dot",
    "\\ln",
]

latex_image_class_names = [
    "latexcenter",
    "latex",
    "tex",
    "latexdisplay",
    "latexblock",
    "latexblockcenter",
]


latex_math_commands = [re.escape(term) for term in latex_math_commands]
latex_math_commands = [x + "(?![a-zA-Z])" for x in latex_math_commands]
latex_regex = re.compile("|".join(latex_math_commands))


def extract_asciimath(s):
    parsed = asciimath2tex.translate(s)
    return parsed


cur_file = os.path.abspath(__file__)
xsl_path = os.path.join(os.path.dirname(cur_file), "mmltex/mmltex.xsl")

xslt = ET.parse(xsl_path)
transform = ET.XSLT(xslt)


def mml_to_latex(mml_code):
    # Remove any attibutes from the math tag
    mml_code = re.sub(r"(<math.*?>)", r"\1", mml_code)
    mml_ns = mml_code.replace("<math>", '<math xmlns="http://www.w3.org/1998/Math/MathML">')  # Required.
    mml_dom = ET.fromstring(mml_ns)
    mmldom = transform(mml_dom)
    latex_code = str(mmldom)
    return latex_code


def wrap_math(s, display=False):
    s = re.sub(r"\s+", " ", s)
    s = color_regex.sub("", s)
    s = s.replace("$", "")
    s = s.replace("\n", " ")
    s = s.strip()
    if len(s) == 0:
        return s
    # Don't wrap if it's already in \align
    if "align" in s:
        return s
    if display:
        return "[extract_tex]" + s + "[/extract_tex]"
    return "[extract_itex]" + s + "[/extract_itex]"


def get_math_config(html):
    has_mathjax = re.search(r"mathjax", html.lower())
    has_katex = re.search(r"katex", html.lower())
    has_latex_math_command = latex_regex.search(html)
    if not has_mathjax and not has_katex and not has_latex_math_command:
        return None
    # Get LaTeX config for MathJax
    regex = r"tex2jax: {[^}]*}"
    latex_config = {
        "inlineMath": [
            ["$", "$"],
            [r"\[", r"\]"],
            ["[itex]", "[/itex]"],
            ["[math]", "[/math]"],
            ["[latex]", "[/latex]"],
            ["[texx]", "[/texx]"],
        ],
        "displayMath": [[r"\(", r"\)"], ["$$", "$$"], ["[tex]", "[/tex]"]],
        "skipTags": ["script", "noscript", "style", "textarea", "pre", "code"],
        "ignoreClass": "tex2jax_ignore",
    }
    try:
        match = re.search(regex, html)
        if match:
            config = match.group(0)
            # Make it a valid json object by adding quotes around the keys
            config = re.sub(r"(\w+):", r'"\1":', config)
            config = "{" + config + "}"
            # config = re.sub(r"\\", r"\\\\", config)
            config = re.sub(r"'", r'"', config)
            config = re.sub(r",\s*}", "}", config)
            extracted_latex_config = json.loads(config)["tex2jax"]
            # latex_config.update(extracted_latex_config)
            # Update this in a smart way: if the key is already there, append the values
            # if the key is not there, add it

            for key in extracted_latex_config:
                if key in latex_config and key != "ignoreClass":
                    latex_config[key] += extracted_latex_config[key]
                else:
                    latex_config[key] = extracted_latex_config[key]
    except Exception:
        pass

    # Get LaTeX config for KaTeX
    """      delimiters: [
          {left: '$$', right: '$$', display: true}
      ],
    """
    regex = r"delimiters: \[[^\]]*\]"
    try:
        match = re.search(regex, html)
        if match:
            config = match.group(0)
            # Make it a valid json object by adding quotes around the keys
            config = re.sub(r"(\w+):", r'"\1":', config)
            # The match is a list without the [] around it. Wrap with {"delimiters": ...}
            config = "{" + config + "}"
            config = re.sub(r"'", r'"', config)
            config = re.sub(r",\s*}", "}", config)
            extracted_latex_config = json.loads(config)["delimiters"]
            for delimiter in extracted_latex_config:
                if delimiter["display"]:
                    latex_config["displayMath"].append([delimiter["left"], delimiter["right"]])
                else:
                    latex_config["inlineMath"].append([delimiter["left"], delimiter["right"]])
    except Exception:
        pass

    # Get AsciiMath config
    regex = r"asciimath2jax: {[^}]*}"
    asciimath_config = {
        "delimiters": [["`", "`"]],
        "skipTags": ["script", "noscript", "style", "textarea", "pre", "code"],
        "ignoreClass": "asciimath2jax_ignore",
    }
    try:
        match = re.search(regex, html)
        if match:
            config = match.group(0)
            # Make it a valid json object by adding quotes around the keys
            config = re.sub(r"(\w+):", r'"\1":', config)
            config = "{" + config + "}"
            # config = re.sub(r"\\", r"\\\\", config)
            config = re.sub(r"'", r'"', config)
            config = re.sub(r",\s*}", "}", config)
            extracted_asciimath_config = json.loads(config)["asciimath2jax"]
            asciimath_config.update(extracted_asciimath_config)
    except Exception:
        pass
    return {"latex": latex_config, "asciimath": asciimath_config}


def html_unescape(s):
    return html.unescape(s)


def replace_math_tags_with_dollar_signs(text):
    # Replace each of these in the proper way
    # itex -> $...$
    # tex -> $$...$$
    # asciimath -> ...

    # Instead of this, simply replace extract_itex with $ and extract_tex with $$.
    text = re.sub(r"\[extract_itex\]", "$", text)
    text = re.sub(r"\[/extract_itex\]", "$", text)
    text = re.sub(r"\[extract_tex\]", "$$", text)
    text = re.sub(r"\[/extract_tex\]", "$$", text)

    return text


def update_text_with_delimiters(text, delimiters, replacement_manager, info):

    def replace_itex(match):
        wrapped = wrap_math(match.group(1))
        tagged = replacement_manager.add_replacement(wrapped, tag="math")
        return tagged

    def replace_tex(match):
        wrapped = wrap_math(match.group(1), display=True)
        tagged = replacement_manager.add_replacement(wrapped, tag="math")
        return tagged

    def replace_asciimath(match):
        wrapped = match.group(1)
        tagged = replacement_manager.add_replacement(wrapped, tag="math")
        return tagged

    for delimiter, delimiter_type in delimiters:
        start_delimiter = re.escape(delimiter[0])
        end_delimiter = re.escape(delimiter[1])
        regex = f"{start_delimiter}(.*?){end_delimiter}"
        if delimiter_type == "INLINE_LATEX":
            # Simply replace the delimiters with [itex] and [/itex]
            updated_text = re.sub(regex, replace_itex, text, flags=re.DOTALL)
            if updated_text != text:
                info["found_math"] = True
                info["mathjax_inline_tex"] += 1
            text = updated_text
        elif delimiter_type == "DISPLAY_LATEX":
            updated_text = re.sub(regex, replace_tex, text, flags=re.DOTALL)
            if updated_text != text:
                info["found_math"] = True
                info["mathjax_display_tex"] += 1
            text = updated_text
        elif delimiter_type == "ASCIIMATH":
            updated_text = re.sub(regex, replace_asciimath, text, flags=re.DOTALL)
            if updated_text != text:
                info["found_math"] = True
                info["mathjax_asciimath"] += 1
            text = updated_text

    return text


def extract_delimited_math(text, mathjax_config, info, replacement_manager):
    """This operates on plain text and extracts LaTeX and AsciiMath"""
    # import pdb; pdb.set_trace()
    if mathjax_config is None:
        return text
    delimiters = []
    for delimiter in mathjax_config["latex"]["inlineMath"]:
        delimiters.append((delimiter, "INLINE_LATEX"))
    for delimiter in mathjax_config["latex"]["displayMath"]:
        delimiters.append((delimiter, "DISPLAY_LATEX"))
    for delimiter in mathjax_config["asciimath"]["delimiters"]:
        delimiters.append((delimiter, "ASCIIMATH"))

    delimiters = sorted(delimiters, key=lambda x: len(x[0][0]), reverse=True)
    text = update_text_with_delimiters(text, delimiters, replacement_manager, info)
    return text


def extract_math(tree, replacement_manager):
    """Webpages often contain LaTeX or AsciiMath equations that are
    hidden within the HTML. This function extracts the LaTeX and
    AsciiMath equations from the HTML.
    """

    info = {
        "found_math": False,
        "script_math_tex": 0,
        "script_math_asciimath": 0,
        "math_annotations": 0,
        "math_alttext": 0,
        "mathml": 0,
        "mathjax_tag": 0,
        "mathjax_inline_tex": 0,
        "mathjax_display_tex": 0,
        "mathjax_asciimath": 0,
        "img_math": 0,
        "codecogs_latex": 0,
        "wp_latex": 0,
        "mimetex.cgi": 0,
        "/images/math/codecogs": 0,
        "mathtex.cgi": 0,
        "katex": 0,
        "math-container": 0,
        "wp-katex-eq": 0,
        "align": 0,
        "equation": 0,
        "x-ck12": 0,
        "texerror": 0,
    }

    # Find and tag any \align environments
    def start_callback(element):
        regex = r"\\begin{align}(.*?)\\end{align}"
        if element.node.type == 3:
            text = element.node.text
            matches = re.findall(regex, text, re.DOTALL)
            for match in matches:
                info["align"] += 1
                info["found_math"] = True
                match = replacement_manager.add_replacement(match, tag="math")
                text.replace(match, match)
            element.node.text = text

    def end_callback(element):
        pass

    body = tree.document.query_selector("body")
    traverse_dom(body, start_callback, end_callback)

    # Find any \equation environments
    def start_callback(element):
        regex = r"\\begin{equation}(.*?)\\end{equation}"
        if element.node.type == 3:
            text = element.node.text
            matches = re.findall(regex, text, re.DOTALL)
            for match in matches:
                info["equation"] += 1
                info["found_math"] = True
                match = match.replace("\\begin{equation}", "")
                match = match.replace("\\end{equation}", "")
                wrapped_text = wrap_math(match, display=True)
                wrapped_text = replacement_manager.add_replacement(wrapped_text, tag="math")
                text = text.replace(match, wrapped_text)
            # Remove the \begin{equation} and \end{equation} tags
            text = text.replace("\\begin{equation}", "")
            text = text.replace("\\end{equation}", "")
            element.node.text = text

    def end_callback(element):
        pass

    body = tree.document.query_selector("body")
    traverse_dom(body, start_callback, end_callback)

    # Find all .texerror
    texerrors = tree.document.query_selector_all(".texerror")
    for texerror in texerrors:

        # Find the text between {} (maximum length) and replace the texerror with that text
        match = re.search(r"\{(.{1,})\}", texerror.text)
        if match:
            info["found_math"] = True
            info["texerror"] += 1
            wrapped_match = wrap_math(match.group(1))
            texerror.html = replacement_manager.add_replacement(wrapped_match, tag="math")

    # This has a ton of repeated code, but it's nice to have fine control over
    # how each source is handled.
    imgs = tree.document.query_selector_all("img")
    for img in imgs:

        class_attr = img.getattr("class")
        if class_attr is not None:
            class_list = class_attr.split(" ")
            if any([img_class in class_list for img_class in latex_image_class_names]):
                alt = img.getattr("alt")
                if alt is None:
                    continue
                new_span = tree.create_element("span")
                wrapped_alt = wrap_math(alt)
                new_span.html = replacement_manager.add_replacement(wrapped_alt, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_alt.strip()) > 0:
                    info["found_math"] = True
                info["img_math"] += 1

        src = img.getattr("src")
        if src is None:
            continue
        if "codecogs.com" in src:
            try:
                latex = src.split("?")[1:]
                latex = "?".join(latex)  # In case there are multiple ? in the latex
                latex = unquote(latex)
                new_span = tree.create_element("span")
                wrapped_latex = wrap_math(latex)
                new_span.html = replacement_manager.add_replacement(wrapped_latex, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_latex.strip()) > 0:
                    info["found_math"] = True
                info["codecogs_latex"] += 1
            except Exception:
                pass
        if "latex.php" in src:
            try:
                # they usually have "alt='-i u_t + &#92;Delta u = |u|^2 u'"
                alt = img.getattr("alt")
                if alt is None:
                    continue
                # Unescape the latex
                alt = unquote(alt)
                # Get the latex
                wrapped_alt = wrap_math(alt)
                new_span = tree.create_element("span")
                new_span.html = replacement_manager.add_replacement(wrapped_alt, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_alt.strip()) > 0:
                    info["found_math"] = True
                info["wp_latex"] += 1
            except Exception:
                pass
        if "/images/math/codecogs" in src:
            try:
                # they usually have "alt='-i u_t + &#92;Delta u = |u|^2 u'"
                alt = img.getattr("alt")
                if alt is None:
                    continue
                # Unescape the latex
                alt = unquote(alt)
                # Get the latex
                wrapped_alt = wrap_math(alt)
                new_span = tree.create_element("span")
                new_span.html = replacement_manager.add_replacement(wrapped_alt, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_alt.strip()) > 0:
                    info["found_math"] = True
                info["/images/math/codecogs"] += 1
            except Exception:
                pass
        if "mimetex.cgi" in src:
            try:
                latex = src.split("?")[1:]
                latex = "?".join(latex)  # In case there are multiple ? in the latex
                latex = unquote(latex)
                new_span = tree.create_element("span")
                wrapped_latex = wrap_math(latex)
                new_span.html = replacement_manager.add_replacement(wrapped_latex, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_latex.strip()) > 0:
                    info["found_math"] = True
                info["mimetex.cgi"] += 1
            except Exception:
                pass
        if "mathtex.cgi" in src:
            try:
                latex = src.split("?")[1:]
                latex = "?".join(latex)  # In case there are multiple ? in the latex
                latex = unquote(latex)
                new_span = tree.create_element("span")
                wrapped_latex = wrap_math(latex)
                new_span.html = replacement_manager.add_replacement(wrapped_latex, tag="math")
                parent = img.parent
                parent.replace_child(new_span, img)
                if len(wrapped_latex.strip()) > 0:
                    info["found_math"] = True
                info["mathtex.cgi"] += 1
            except Exception:
                pass
        class_attr = img.getattr("class")
        if class_attr is not None:
            if "x-ck12" in class_attr:
                try:
                    latex = img.getattr("alt")
                    latex = unquote(latex)
                    new_span = tree.create_element("span")
                    wrapped_latex = wrap_math(latex)
                    new_span.html = replacement_manager.add_replacement(wrapped_latex, tag="math")
                    parent = img.parent
                    parent.replace_child(new_span, img)
                    if len(wrapped_latex.strip()) > 0:
                        info["found_math"] = True
                    info["x-ck12"] += 1
                except Exception:
                    pass

    # Find any blocks with class math-container and replace them with spans
    math_containers = tree.document.query_selector_all(".math-container")
    for math_container in math_containers:
        text = math_container.text
        new_span = tree.create_element("span")
        wrapped_math = wrap_math(text, display=True)
        new_span.html = replacement_manager.add_replacement(wrapped_math, tag="math")
        parent = math_container.parent
        parent.replace_child(new_span, math_container)
        if len(wrapped_math.strip()) > 0:
            info["found_math"] = True
        info["math-container"] += 1

    katex_inline_wp = tree.document.query_selector_all(".wp-katex-eq")
    for katex in katex_inline_wp:
        text = katex.text
        new_span = tree.create_element("span")
        display_attr = katex.getattr("data-display")
        if display_attr is not None:
            display = display_attr == "true"
        else:
            display = False
        wrapped_math = wrap_math(text, display=display)
        new_span.html = replacement_manager.add_replacement(wrapped_math, tag="math")
        parent = katex.parent
        parent.replace_child(new_span, katex)
        if len(wrapped_math.strip()) > 0:
            info["found_math"] = True
        info["wp-katex-eq"] += 1

    # Find all script[type="math/tex"] tags and replace them with spans
    latex_script_tags = tree.document.query_selector_all('script[type="math/tex"]')
    for script_tag in latex_script_tags:
        text = script_tag.text
        new_span = tree.create_element("span")
        wrapped_text = wrap_math(text)
        new_span.html = replacement_manager.add_replacement(wrapped_text, tag="math")
        parent = script_tag.parent
        parent.replace_child(new_span, script_tag)
        if len(wrapped_text.strip()) > 0:
            info["found_math"] = True
        info["script_math_tex"] += 1

    asciimath_script_tags = tree.document.query_selector_all('script[type="math/asciimath"]')
    for script_tag in asciimath_script_tags:
        try:
            text = script_tag.text
            new_span = tree.create_element("span")
            wrapped_asciimath = wrap_math(extract_asciimath(text))
            new_span.html = replacement_manager.add_replacement(wrapped_asciimath, tag="math")
            parent = script_tag.parent
            parent.replace_child(new_span, script_tag)
            if len(wrapped_asciimath.strip()) > 0:
                info["found_math"] = True
            info["script_math_asciimath"] += 1
        except Exception:
            # Delete this script tag
            parent = script_tag.parent
            parent.remove_child(script_tag)

    # For katex, find all elements with class = tex
    katex_spans = tree.document.query_selector_all(".tex")
    for katex_span in katex_spans:
        try:
            # Check if they have data-expr attr
            expr = katex_span.getattr("data-expr")
            if expr is None:
                continue
            # Replace with a span
            new_span = tree.create_element("span")
            wrapped_expr = wrap_math(expr)
            new_span.html = replacement_manager.add_replacement(wrapped_expr, tag="math")
            parent = katex_span.parent
            parent.replace_child(new_span, katex_span)
            if len(wrapped_expr.strip()) > 0:
                info["found_math"] = True
            info["katex"] += 1
        except Exception:
            pass

    # Find any spans with class "katex"
    katex_spans = tree.document.query_selector_all("span.katex")
    for katex_span in katex_spans:
        # Find any spans with class "katex-html" and remove them
        katex_html_spans = katex_span.query_selector_all("span.katex-html")
        for katex_html_span in katex_html_spans:
            parent = katex_html_span.parent
            parent.remove_child(katex_html_span)

    # Remove any .MathJax_Preview spans
    mathjax_preview_spans = tree.document.query_selector_all("span.MathJax_Preview")
    for mathjax_preview_span in mathjax_preview_spans:
        parent = mathjax_preview_span.parent
        parent.remove_child(mathjax_preview_span)

    # Find any math tags
    math_tags = tree.document.query_selector_all("math")
    # For each math tag, see if there is an annotation tag with
    # encoding="application/x-tex" inside it
    for math_tag in math_tags:
        annotation_tag = math_tag.query_selector('annotation[encoding="application/x-tex"]')
        if annotation_tag is not None:
            # Get the text content of the annotation tag
            text = annotation_tag.text
            # Set the content of the math tag to the text
            # replace this math tag with a span tag with the text
            # To do this, we need to get the parent of the math tag
            parent = math_tag.parent
            # Then, we need to create a new span tag
            new_span = tree.create_element("span")
            # Set the html of the new span tag to the text
            wrapped_text = wrap_math(text)
            new_span.html = replacement_manager.add_replacement(wrapped_text, tag="math")
            # Then, we need to replace the math tag with the new span tag
            parent.replace_child(new_span, math_tag)
            # If the parent has style="display:none", then we need to
            # remove the style attribute
            style_value = parent.getattr("style")
            if style_value is not None:
                normalized_style_value = style_value.lower().strip().replace(" ", "").replace(";", "")
                if "display:none" in normalized_style_value:
                    parent.delattr("style")
            if len(wrapped_text.strip()) > 0:
                info["found_math"] = True
            info["math_annotations"] += 1
        # Check if the math tag has an alttext attribute
        elif math_tag.getattr("alttext") is not None:
            # Get the alttext attribute
            alttext = math_tag.getattr("alttext")
            new_span = tree.create_element("span")
            # Set the html of the new span tag to the text
            wrapped_alttext = wrap_math(alttext)
            new_span.html = replacement_manager.add_replacement(wrapped_alttext, tag="math")
            # Then, we need to replace the math tag with the new span tag
            parent = math_tag.parent
            parent.replace_child(new_span, math_tag)
            if len(wrapped_alttext.strip()) > 0:
                info["found_math"] = True
            info["math_alttext"] += 1
        # Otherwise, translate the math tag to LaTeX
        else:
            try:
                # Try translating to LaTeX
                mathml = math_tag.html
                # If this includes xmlns:mml, then we need to replace all
                # instances of mml: with nothing
                if "xmlns:mml" in mathml:
                    mathml = mathml.replace("mml:", "")
                    # replace xmlns:mml="..." with nothing
                    mathml = re.sub(r'xmlns:mml=".*?"', "", mathml)
                latex = mml_to_latex(mathml)
                # Make a new span tag
                new_span = tree.create_element("span")
                # Set the html of the new span tag to the text
                wrapped_latex = wrap_math(latex)
                new_span.html = replacement_manager.add_replacement(wrapped_latex, tag="math")
                # Then, we need to replace the math tag with the new span tag
                parent = math_tag.parent
                parent.replace_child(new_span, math_tag)
                if len(wrapped_latex.strip()) > 0:
                    info["found_math"] = True
                info["mathml"] += 1
            except Exception:
                parent = math_tag.parent
                parent.remove_child(math_tag)

    mathjax_tags = tree.document.query_selector_all("mathjax")
    for mathjax_tag in mathjax_tags:
        # Get the inner text of the mathjax tag
        text = mathjax_tag.text
        text = html.unescape(text)
        # Use regex to find text wrapped in hashes
        matches = re.findall(r"#(.+?)#", text)
        # For each match, replace the match with the LaTeX
        for match in matches:
            try:
                latex = extract_asciimath(match)
                # Replace the match with the LaTeX
                text = text.replace(f"#{match}#", latex)
            except Exception:
                pass

        # Create a new span tag
        new_span = tree.create_element("span")
        # Set the html of the new span tag to the text
        new_span.html = replacement_manager.add_replacement(text, tag="math")
        # Then, we need to replace the mathjax tag with the new span tag
        parent = mathjax_tag.parent
        parent.replace_child(new_span, mathjax_tag)
        if len(text.strip()) > 0:
            info["found_math"] = True
        info["mathjax_tag"] += 1

    return tree, info


def remove_color(text):
    return re.sub(color_regex, "", text)
