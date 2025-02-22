------------------------------------------------------------------------

------------------------------------------------------------------------

\<mphantom\> element renders invisibly keeping same size and other
dimensions, including baseline position, as its contents would have if
they were rendered normally. It is used to align parts of an expression
by invisibly duplicating sub-expressions.

Syntax
------

Here is the simple syntax to use this tag −

    <mphantom> expression </mphantom>

Parameters
----------

Here is the description of all the parameters of this tag −

Attributes
----------

Here is the description of all the attributes of this tag −

Examples
--------

    <math xmlns = "http://www.w3.org/1998/Math/MathML">
       <mfrac>
          <mrow>
             <mi> x </mi>
             <mo> + </mo>
             <mi> y </mi>
             <mo> + </mo>
             <mi> z </mi>
          </mrow>
          
          <mrow>
             <mi> x </mi>
             <mphantom>
                <mo> + </mo>
             </mphantom>
             
             <mphantom>
                <mi> y </mi>
             </mphantom>
             <mo> + </mo>
             <mi> z </mi>
          </mrow>
       </mfrac>
    </math>  

Output
------

$`\frac{x+y+z}{x+z}`$
