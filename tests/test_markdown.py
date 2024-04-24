import pytest

from markweb.markdown import mathml_to_markdown, minimal_markdown_escape


def test_always_escaped():
    """Tests characters that are always escaped."""
    test_cases = [
        ("[brackets]", r"\[brackets\]"),
        (r"Backslash "'\\', r"Backslash "'\\'),
        ("`code block`", r"\`code block\`"),
    ]

    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_no_escaping():
    """Tests scenarios where characters should not be escaped."""
    test_cases = [
        ("word * in the middle *", "word * in the middle *"),
        ("_ surrounded by spaces _", "_ surrounded by spaces _"),
        ("-normal hyphen-", "-normal hyphen-"),
        ("+normal plus+", "+normal plus+"),
        ("Normal line with #", "Normal line with #"),
        # don't escape @, even in links
        ("[@user](https://example.com)", r"\[@user\](https://example.com)"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_conditional_escaping():
    """Tests *, _ with and without surrounding spaces."""
    test_cases = [
        ("word*bold*", r"word\*bold\*"),
        ("* this is not bold *", "* this is not bold *"),
        ("normal *bold*", "normal \\*bold\\*"),
        ("_italic_", "\_italic\_"),
        ("normal _italic_", "normal \_italic\_"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_line_start_escaping():
    """Tests -, #, + at the beginning of lines."""
    test_cases = [
        ("- hyphen", "\- hyphen"),
        ("+ plus", "\+ plus"),
        ("# header", "\# header"),
        ("Normal line - not escaped", "Normal line - not escaped"),
    ]
    for text, expected in test_cases:
        assert minimal_markdown_escape(text) == expected


def test_combined_scenarios():
    """Tests with multiple special characters."""
    input_text = \
        r"""This is *bold* and _italic_.
    # Header 1
    - List item 1
    + List item 2
    This has [brackets] and \backslashes\.
    """
    expected_text = \
        r"""This is \*bold\* and \_italic\_.
    \# Header 1
    \- List item 1
    \+ List item 2
    This has \[brackets\] and \backslashes\\.
    """
    assert expected_text == minimal_markdown_escape(input_text)


def test_mathml_to_markdown():
    """Tests conversion of MathML to markdown."""
    test_cases = [
        ("<math><mi>x</mi><mo>+</mo><mn>1</mn></math>", "$`x+1`$"),
        # integration
        ("<math><mo>&#x222B;</mo><mi>x</mi><mi>d</mi><mi>x</mi></math>", r"$`\int xdx`$"),
        # some fancy information theory: KL divergence
        ("<math><mi>D</mi><mo stretchy=\"false\">(</mo><mi>P</mi><mo>&#x2225;</mo><mi>Q</mi><mo stretchy=\"false\">)</mo></math>", r"$`D(P\parallel Q)`$"),
        # chain rule of probability
        ("<math><mi>P</mi><mo stretchy=\"false\">(</mo><mi>A</mi><mo stretchy=\"false\">)</mo><mo>=</mo><mi>P</mi><mo stretchy=\"false\">(</mo><mi>A</mi><mo stretchy=\"false\">|</mo><mi>B</mi><mo stretchy=\"false\">)</mo><mi>P</mi><mo stretchy=\"false\">(</mo><mi>B</mi><mo stretchy=\"false\">)</mo></math>",
         r"$`P(A)=P(A|B)P(B)`$"),
        # a fraction
        ("""
            <math xmlns='http://www.w3.org/1998/Math/MathML' display='inline'>  
      <mfrac>  
        <mfrac>  
          <mn>1</mn>  
          <mi>x</mi>  
        </mfrac>  
        <mrow>  
          <mi>y</mi>  
          <mo>-</mo>  
          <mn>2</mn>  
        </mrow>  
      </mfrac>  
    </math>""", r"$`\frac{\frac{1}{x}}{y-2}`$"),
        # pythagorean theorem
        ("""<math xmlns='http://www.w3.org/1998/Math/MathML' display="inline"> <mrow> <msup> <mi>a</mi> <mn>2</mn> </msup> <mo>+</mo> <msup> <mi>b</mi> <mn>2</mn> </msup> <mo>=</mo> <msup> <mi>c</mi> <mn>2</mn> </msup> </mrow> </math>""",
         r"$`a^2+b^2=c^2`$"),
        # More complicated exponentiation gets brackets
        ('<math display="inline"><msup><mi>x</mi><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow></msup></math>', "$`x^{y+1}`$"),
        # subscripts
        ('<math display="inline"><msub><mi>x</mi><mi>i</mi></msub></math>', "$`x_i`$"),
        # subscripts and superscripts
        ('<math display="inline"><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup></math>', "$`x_i^2`$"),
        # complex subscripts and superscripts
        ('<math display="inline"><msubsup><mi>x</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow><mrow><mi>z</mi><mo>-</mo><mn>1</mn></mrow></msubsup></math>', "$`x_{i+1}^{z-1}`$"),
        # physics stuff: quantum
        ('<math display="inline"><mrow><mo>&#x27E8;</mo><mi>&#x03A8;</mi><mo>&#x2223;</mo><mi>&#x03A6;</mi><mo>&#x27E9;</mo></mrow></math>', "$`\\langle \\Psi \\mid \\Phi \\rangle `$"),
        ('<math display="inline"><mrow><mo>&#x27E8;</mo><mi>&#x03A8;</mi><mo>&#x2223;</mo><mi>&#x03A6;</mi><mo>&#x27E9;</mo><mo>&#x2223;</mo><mi>&#x03A7;</mi></mrow></math>', "$`\\langle \\Psi \\mid \\Phi \\rangle \\mid {\\rm X}`$"),
        ('<math display="inline"><mrow><mo>&#x27E8;</mo><msub><mi>&#x03A8;</mi><mi>i</mi></msub><mo>&#x2223;</mo><msub><mi>&#x03A6;</mi><mi>j</mi></msub><mo>&#x27E9;</mo></mrow></math>', "$`\\langle \\Psi _i\\mid \\Phi _j\\rangle `$"),
        # physics stuff: relativity
        ('<math display="inline"><mrow><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></mrow></math>', "$`E=mc^2`$"),
        # physics stuff: schrodinger
        ('<math display="inline"><mrow><mi>i</mi><mi>&#x210F;</mi><mi>&#x03B6;</mi><mo>=</mo><mi>H</mi><mi>&#x03B6;</mi></mrow></math>', "$`i\\hslash \\zeta =H\\zeta `$"),



    ]
    for mathml, expected in test_cases:
        assert mathml_to_markdown(mathml) == expected


def test_mathml_block():
    """Tests conversion of MathML to markdown."""
    test_cases = [
        ("<math display='block'><mi>x</mi><mo>+</mo><mn>1</mn></math>", "\n$$x+1$$\n"),
        # integration
        ("<math display='block'><mo>&#x222B;</mo><mi>x</mi><mi>d</mi><mi>x</mi></math>", "\n$$\int xdx$$\n"),
        # cauchy-schwarz inequality
        ("<math display='block'><mrow><mo>(</mo><mrow><mfrac><mrow><mo>|</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>|</mo></mrow><mrow><mo>(</mo><mrow><msup><mi>a</mi><mn>2</mn></msup><mo>+</mo><msup><mi>b</mi><mn>2</mn></msup></mrow><mo>)</mo></mrow></mfrac></mrow><mo>)</mo><mo>^</mo><mn>2</mn></mrow></math>",
         "\n$$\\left(\\frac{|a,b|}{\\left(a^2+b^2\\right)}\\right)^2$$\n"),
    ]

    for mathml, expected in test_cases:
        assert mathml_to_markdown(mathml) == expected
if __name__ == "__main__":
    pytest.main(["-v", "test_markdown.py"])



