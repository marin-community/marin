from py_mini_racer import py_mini_racer

# Load Readability.js (you need to have the Readability.js source code as a string)
# ctx.eval(readability_js_source_code)

_ctx = None

def _lazy_load_readability():
    global _ctx
    if _ctx is not None:
        return _ctx

    with open("min_readability.min.js", "r") as f:
        src = f.read()

    _ctx = py_mini_racer.MiniRacer()
    _ctx.eval(src)
    print(_ctx.eval("ContentExtractor"))
    print(_ctx.eval("ContentExtractor.extractContent"))
    # add a simple function to call it
    # Readabilty is a module that exports a class also called Readability
    # _ctx.eval("function extract_content(html) { return Readability(html).parse(); }")
    # _ctx.eval("function extract_content(html) { return new Readability.Readability(html).parse; }")
    # needs to be a dom object

    return _ctx


def extract_content(your_html_content):
    result = _lazy_load_readability().call("ContentExtractor.extractContent", your_html_content)

    return result



if __name__ == '__main__':
    html_content = """
    <html>
    <head>
    <title>Some title</title>
    </head>
    <body>
    <h1>Some heading</h1>
    <p>Some paragraph.</p>
    </body>
    </html>
    """
    print(extract_content(html_content))
