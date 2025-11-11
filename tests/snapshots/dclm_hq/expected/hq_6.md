# WWW-HTML Apr-Jun 1995: Thinking about style sheets

# Thinking about style sheets

Kevin Hughes (*kevinh@eit.COM*)  
*Tue, 2 May 1995 04:07:10 +0500* * **Messages sorted by:** \[ date \]\[ thread \]\[ subject \]\[ author \]
* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"

---

This post is a long one; I would encourage replies to be  
broken up into manageable subjects such as fonts, etc.  
The following thoughts revolve around creating a simple  
yet powerful style sheet language for the World-Wide Web. I have  
incorporated comments here and there from a previous email  
discussion involving Hakon Lie, Dave Raggett, Dan Connolly, and  
Bill Perry. Hakon should feel free to put this post under the  
style sheet area at the W3O site.  
A W3 style-sheet mailing list will likely be created,  
but Hakon says this will take some days. There should be pointers  
at http://www.w3.org/ in time. In the meantime you can refer to:  
http://gummo.stanford.edu/html/hypermail/archives.html  

If you're not familiar with style sheets, please take  
a look at:  

http://www.w3.org/hypertext/WWW/Style/  

...and take a look at the current proposals and ideas out  
there.  
The format I discuss below is a bit different and somewhat  
improved over Hakon's format as described at the Chicago conference:  

http://www.w3.org/hypertext/WWW/People/howcome/p/cascade.html  

...but from the examples below it should be fairly easy  
to figure things out.  
Finally, a few seconds of silence for the NSFNET backbone  
service, which was decommissioned yesterday, April 30, 1995.  
May it rest in peace,  

-- Kevin  

Thinking About Style Sheets  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Hakon Lie described a simple yet powerful style sheet format in  
Darmstadt, one which he has been working on for some time and had  
been greatly improved since its first rollout in Chicago. Despite  
that fact that it has not been formally described anywhere, I feel  
that the format is superior to the other proposed formats for a number  
of reasons:  

1) It's simple yet flexible.  

Many proposed formats are much too complex for their own  
good - in order to read the simplest formatting commands,  
browsers have to parse ornate structures and operations  
which add little to the overall utility of the format.  
Hakon's format is broken into manageable, logical chunks  
that allow for extensibility and ease of parsing. The  
format is also well-suited to be written in a compact format  
that keeps the style sheet size to a minimum.  

2) It's human readable and writeable.  

The Web no longer belongs to the average Internet-savvy  
technician who writes daily in Perl and C++. In order to  
urge authors and designers to make use of style sheets  
the format must be understandable so that one can "look  
under the hood" when needed. Certainly tools and editors  
will be created that makes style sheet creation easy,  
but the Web will be rewarded a hundredfold by using a  
format for which it is easy to create tools for.  

Below I propose a number of changes and additions to Hakon's format,  
using the time-tested lessons learned by the most popular DTP  
software packages on the market today as well as a little Web design  
experience. I feel that style sheets have the potential to revolutionize  
the way people use and think about the Web and hope that the standards  
process does the right thing both for users and authors.  

Note that many of the examples below will replace the need for using  
tag attributes and nonstandard HTML extensions to create layout. This  
is a good thing, as the layout/style will finally be separated more  
cleanly from the document structure.  

The idea, if I understand correctly, is:  

\<HTML\>  
\<STYLE NOTATION="experimental"\>  

\<!-- The style sheet (or perhaps a pointer to it) goes here. --\>  
\<!-- Here I can define a particular style, or rather, a CLASS. --\>  

P\[mystyle\] : font.size = 12pt  

\</STYLE\>  
\<BODY\>  
Regular structured markup goes here.  

\<P CLASS="mystyle"\>This is a paragraph in my style,  
with a font size of 12 points.  
\</BODY\>  
\</HTML\>  

If you find this pretty easy to understand, well, that's the idea.  
Browsers may very well support DTDs as well as a readable style sheet  
format like Hakon's.  

Linked style sheets will announce themselves to browsers with a MIME  
type such as:  

application/x-perry-stylesheet  
application/x-arena-stylesheet  
application/x-w3c-experimental  
application/x-dsssl-lite  

New Things  
~~~~~~~~~~  

I would say that authors should have the capability to include  
style sheets within the document as well as point to particular  
style sheets (that is, link to them). This makes it easy for a site  
to have a directory of "house" style sheets that apply to all documents  
in a particular web.  

First, there should be a way to specify data that applies to all tags,  
or data that applies to entire style sheet or class of objects. This  
may be done with wildcards:  

BODY : \*.color = red  
doc : \*.color = red  
* : font.size = 12pt  

"doc" is something representing the entire document. Most likely  
this should be replaced with HTML, as one may want different  
properties for HEAD and BODY.  

Apparently Hakon's notation allows for pattern matching on  
sequences of tags, hierarchies of tags, and attribute values.  

HTML : \*.color = red  

In the second example, the font size of 12 points applies to the data in  
all tags. In the first example, the color of all elements in the BODY  
tag are red.  

In regards to sizes and such, there should be a way to specify a  
base size and then allow other styles to be sized relative to the  
base size.  

H1 : font.size += 3pt  
H1 : font.size = P\[mystyle\]:font.size + 10pt  
H1 : font.size = 1.5 * P:font.size  

There should be a full listing and study of the proposed format  
name hierarchy, so that attributes are named in a logical, structured  
manner. More general objects should be on the left, with values becoming  
more specific to the right. So "color.background" should instead  
be "background.color", because background is a general object, while  
color is a specific property.  

Style sheets should support the most common units in DTP; these may be:  

inches (in)  
pixels (px)  
centimeters/millimeters (cm/mm)  
ems (em)  
points (pt)  
characters(ch)  
pica(pi) - "pi" is equal to 12pt  

The characters unit is needed because a designer may wish to specify  
an indent or tab stop by the number of characters in the current font,  
for instance, in a PRE tag.  

All elements that support values should support a value of "default".  
Browsers would come out of the box with a default style sheet preselected  
that the user never need change if they don't want to.  

One possible way to specify tab stops might be:  

PRE : tabs = 8ch, 16ch, 24ch  

Dave Raggett notes that this is discouraged, as HTML 3.0 includes  
a TAB element that does many nice things.  

It would be useful to allow a number of format "shortcuts" so that people  
writing sheets manually don't have to type so much and so that the size  
of the style sheet stays at a minimum. One possible thing to do is to  
use the semicolon (is there a more appropriate character?) for allowing  
the designer to specify multiple values for multiple tags:  

P H1 : font.color = red; blue  
H1 H2 LI : font.size = 1.0; 0.8; 0.2  

In the first example, data in P tags would be red; data in H1 tags  
would be blue.  

In the second, the font size in H1 is the normal default size, H2  
has a font size of 0.8 times that, and LI has a font size of 0.2  
times that.  

Bill Perry suggests another kind of shortcut:  

H1 : font.size=12pt; background.color = red;  
foreground.color = blue  

This is very nice, as authors can deal with a whole tag at a time on  
one line or split things up over multiple lines so the organization  
makes more sense. Again, this is for human usability.  

Another kind of shortcut is to provide shorter versions of  
words:  

"img" for "image"  
"bg" for "background"  

H1 : bg.color = blue  

In specifying color, the designer should be able to enter the  
decimal/percentage values using the RGB color space, since it is more  
understandable and most image processing programs use these values:  

P : font.color = 191, 191, 191  
P : font.color = 0.75, 0.75, 0.75  

The percentages are percents of the range from 0 to 255. In the  
3D world, the range would be from 0 to 1, but we're not dealing with  
that context (yet).  

It would also be nice if one could define style-sheet specific color  
labels so that colors could be referred to by name rather than by number:  

define "My Gray" "191, 191, 191"  
P : font.color = "My Gray"  

The use of such pre-processor-like definitions could be considered a  
good thing, since it again reduces the style sheet size and makes  
readability easier. For colors, as an example, one may be able to  
specify the color name as given in the X11 rgb database.  

It may be useful to allow the designer to provide hints about the  
document dimensions, but it is unclear as to where and how such metadata  
might be specified. Perhaps:  

HTML : width = 600px  
HTML : height = 800px  

Some browsers have a widget that when selected sizes the window to  
the widest non-textual element on the screen. I think the browser  
should resize itself automatically according to these hints if the user  
wants.  

Alignment and Justification  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Alignment should refer to vertical movement, as in the IMG tag, while  
justification should be used to specify horizontal movement. Thus:  

align may be top, middle, bottom, basetop, basemiddle, basebottom  
justify may be left, right, center, full  

IMG : align = basetop  
P : justify = center  

Top, middle, and bottom align elements relative to the size of the  
highest character on the current line, measured from the baseline.  

Dave Raggett comments that HTML has got it wrong in this respect, with  
ALIGN being used for vertical and horizontal alignment. I say that although  
it's probably too late to change things in HTML, let's get it right in  
style sheets.  

Dave also says that one needs bleedleft, bleedright, and bleedboth  
additionally for figures and tables.  

--- ---  
--- | | | |  
--- --- | | | | --- | | \<- font height  
ABC | | | | | | | | | |  
--- | | | | --- --- | | --- \<- baseline  
*| | \___ | | | |*  
--- | | ---  
*| |*  
---  

top mid bot bas bas bas  
top mid bot  

Containers and Spacing  
~~~~~~~~~~~~~~~~~~~~~~  

If browser layout is parsed and understood internally in a container-like  
manner, it should not be difficult to give every element a margin on each  
side, which specifies the amount of space above, below, and to the sides  
of an element. In this way, margins for an entire document as well as  
margins for a particular element can be specified.  

margin values can be top, bottom, left, right  

BODY : margin.top = 10px  
BODY : margin.bottom = 10px  
P : margin.left = 10px  

Regarding "container" properties, one can think of borders and styles,  
backgrounds and foregrounds, and (tiled) images.  

BODY : background.color = red  
BODY : foreground.color = blue  
BODY : background.image = "http://somewhere.com/tile.gif"  
P : border.width = 3px  

border.style may be none, default, line, bevel  

P : border.style = line  
P : border.color = black  
P : border.image = "picture.frame.tile.gif"  

The border may go inside, centered with, or outside the margin boundaries.  
A property name should be suggested for this - somehow "justify" and  
"align" don't seem appropriate.  

Line size could be specified. When applied to a textual element, this  
would indicate leading, the space below a line's baseline. When applied  
to an element like HR, this refers to the height of the element.  

HR : line.size = 3px  
P : line.size = 12pt  

Line indentation can be specified for elements - most designers find  
general indents and indents for the first line to be desireable. So:  

P : line.indent.first = 1in  
P : line.indent = 0.5in  

So the general spacing model for an element would look like:  

------------------ \<- margin.top  
*| blah blah... |*  
*| | \<- line.indent.first*  
*| | \<- line.indent*  
*| | \<- line.size (leading)*  
*| |*  
------------------ \<- margin.bottom  
^ ^  
margin.left margin.right  

In this model, no negative values are allowed.  

Fonts  
~~~~~  

Technically a font has a few basic properties:  

font.family (Helvetica, Lucida)  
font.weight (Plain, Bold, Roman)  
font.angle (Italic, Regular)  
font.variation (Regular, Compressed)  

Although one should be able to completely specify a font with these  
attributes, things can and should be simplified:  

P : font.name = "Helvetica Compressed Bold Italic"  

font.name would allow one to specify a font by its full name as a  
shortcut. However, using the prior four variables may be desired since  
they could provide hints to the browser on how to generate the font  
on the fly.  

Hakon suggested that the order of the words in font.name should not  
matter - the set should be descriptive enough to find a particular  
font.  

Besides font.size and font.color, I would propose font.style, which  
would provide hints to the browser as how to display the given font.  
This style would be generated dynamically (mathematically perhaps),  
based on the given font name or family/weight/angle/variation set.  

font.style may be default, plain, bold, italic, underline,  
double underline, strikethrough, overline, change bar,  
inverse, superscript, subscript, outline, shadow,  
small caps, big caps, lowercase, uppercase  

The case styles would display text as follows:  

plain: "John Smith studies at MIT."  
small caps: "john smith studies at mIT."  
big caps: "John Smith Studies At MIT."  
lowercase: "john smith studies at mit."  
uppercase: "JOHN SMITH STUDIES AT MIT."  

Of course you could specify multiple values where it makes sense:  

P : font.style = subscript & bold  

One desired style is using an all-capitals string with larger  
capitals at the beginning of words. One way to do this may be:  

P : font.style = uppercase & bigcaps  

Much in the same way that one can specify different spacing for the  
first line of an element, one should be able to specify different  
styles for the first letter of a word. This may enable things  
like drop caps.  

P : font.family.dropcap = Times  
P : font.family.firstletter = Times  

The first example changes the first letter in the entire paragraph;  
the second changes the first letter of every word.  

For elements such as drop caps and images there should be a way  
to specify how many lines high the element is compared to the  
normal text, and how that element should be aligned in respect  
to the first line of the element. This is so things like this  
can be made:  

---- blah blah blah  
*| | blah blah blah*  
---- blah blah blah  
blah blah blah blah  

P : font.size.dropcap += 3pt  

The dropcap is 3 points larger than the element's normal  
font size.  

P : font.size.dropcap \*= 4  

The size of the dropcap is four times bigger than the  
size of the normal font size.  

P : align.dropcap = top  
IMG\[dropcap\] : align.dropcap = top  

When a dropcap has an alignment of "top" its top is  
aligned with the first line of the element and drops below  
that line, allowing subsequent lines to wrap around it.  

Perhaps there should be "align.topwrap" and "align.bottomwrap"  
instead?  

Links  
~~~~~  

The designer should be able to specify how links are displayed, but  
users should be able to easily override these specifications, as  
they may be dependent on a particular navigation style. "imglink"  
would refer to a link on an image, while "txtlink" would refer to  
a textual link.  

txtlink.color = blue  
txtlink.color = blue \< AGE / 7d \< "Background Color"  
imglink.color = default  
imglink.size = 3px  
imglink.style may be line, bevel, default, none  
txtlink.style may be underline, double underline,  
inverse, default, none  

Note that in conjunction with AGE links may be colored dynamically.  
If there is a predefined value for the background color, links can  
disappear with time and yet still be active, as in the example above.  

There should be a way to specify new versus followed links, but I  
have no good ideas as how to do so.  

Based on something Hakon said:  

HTML : txtlink.color = red \< FETCHED \< blue  

Style Sheet Permissions  
~~~~~~~~~~~~~~~~~~~~~~~  

There should be an extra layer in the current permissions model:  

\<- author insist | legal  
user insist  
author important  
user important  
author normal  
user normal  
application default  

"author insist" would first display the page as the author intended  
but allow the user to change the style (for example, via a pop-up  
"styles" menu in the browser) after the initial download. The "legal"  
specification would display as the author intended and not allow the  
user to make changes. If the particular style in a browser was not  
supported for a legal document, then the document would not display  
at all.  

P : font.size = 12pt ! legal  

Authors writing legal documents may first display a readable  
disclaimer which tells the user what they must to do read the legal  
document (get a particular browser, enter a password, use a particular  
style, etc.), why it is legal (display copyrights, licensing restrictions,  
etc.) and then link to the document in question.  

Bill Perry suggests that it is more reasonable is for a dialog box to appear:  

"Your settings will override something flagged as 'legal'.  
Should I continue?"  

\[ OK \] \[ Cancel \]  

All of this was a heavily debated issue at Darmstadt and will continue  
to be for some time. One thing to consider is that in the real world,  
given enough time and resources, you can copy anything man-made reasonably  
well and change it to suit your worldview. People will always want  
to do this to things. Now that it can be very easy for them to do so,  
why not let them?  

Users should be able to specify particular styles they do not want  
to see, for instance, a color blind user may never want to see any  
elements colored red or blue and instead would wish them to be colored  
black. Dave Raggett notes that authors, not users, should be the ones to  
anticipate this type of need in the style sheet. Forcing the browser  
to deal with it would lead to crude results.  

Note that the issues of filtering out particular styles and filtering  
out particular media/textual content are somewhat different and discussions  
regarding them should keep this in mind. The style area of a document  
should hold information related to layout and "look and feel". Within  
the content area, things that more closely affect the textual/media  
content should be put there, such as ways to represent different languages  
and multiple versions of the same content (animations, ISMAP things, etc.).  
Care should be taken not to confuse the two.  

Dave Raggett suggests that the style sheet should have a way to insert  
particular elements into the document so that one could create  
templates, page headers and footers, etc. easily.  

Perhaps:  

HEADER : insert "page.header.html"  
FOOTER : insert "page.footer.html"  

This inserts HTML within tags, while:  

H1 : insert.before "\<IMG SRC=\\"rule.gif\\"\>"  
H1 : insert.after "some text"  

adds text/HTML before or after elements.  

Industrial-Strength Stuff  
~~~~~~~~~~~~~~~~~~~~~~~~~  

Here are some features which may also be added, in no particular order:  

Kerning  
Tracking (spaces between words, letters, spacing limits)  
The ability to specify particular color models  
Columns (auto flow, positioning)  
Ways to allow elements to float over others  
Different ways of word wrapping  
Ways to handle breaks in text  
Page numbering  
Numbering of list elements  
Transitions (a la a presentation program)  
Control over text flow  

Finally...  
~~~~~~~~~~  

We should be seeing a full description of the experimental notation,  
with proposals, etc. and a full listing of properties, values,  
meta-variables, etc. on the Web pretty soon.  

The above examples and features cover a good deal of the general  
functionality of the most common DTP programs and remove the need  
for many of the nonstandard HTML extensions that are seen today.  
With a powerful style sheet language, the Web will certainly take on  
a new level of sophistication and many of the problems authors have  
had in the past regarding online design and layout will be solved.  
Dave Raggett notes that notation that is meant to be read and used by  
humans must have the right level of abstraction so that users and authors  
can easily control style.  

The web's incorporation of style sheets will certainly raise the  
level of expectation higher: users will not just expect to see a few  
blocks of text but will soon expect the features previously only available  
in "real" desktop publishing programs. Because of this it's quite  
possible that developers will force themselves to look at parsing  
and browser architecture in a more object-oriented manner, giving  
way to a more modular, container-based approach. Arena and Bert Bos's  
work are starting points, but once you add in OpenDoc or OLE,  
COBRA, HotJava, etc. you can begin to see how powerful that  
approach can be.  

People will finally think more about the structure of information  
and more about making it universal and interchangable. And hopefully  
we can finally see some serious content on the Internet.  

Arena is currently implementing Hakon's style sheet format, as is  
Bill Perry - what is needed is a lot more discussion and real code  
out there before a proposal can be presented to an IETF working group.  
Hakon says that a W3C workshop on stylesheets is planned for September;  
hopefully we'll see some good things happen before then.  

```
--
Kevin Hughes * kevinh@eit.com
Enterprise Integration Technologies Webmaster (http://www.eit.com/)
Hypermedia Industrial Designer * Duty now for the future!

```

---

* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"

WWW-HTML Apr-Jun 1995: Thinking about style sheets 

# Thinking about style sheets

Kevin Hughes (*kevinh@eit.COM*)  
*Tue, 2 May 1995 04:07:10 +0500* * **Messages sorted by:** \[ date \]\[ thread \]\[ subject \]\[ author \]
* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"

---

This post is a long one; I would encourage replies to be  
broken up into manageable subjects such as fonts, etc.  
The following thoughts revolve around creating a simple  
yet powerful style sheet language for the World-Wide Web. I have  
incorporated comments here and there from a previous email  
discussion involving Hakon Lie, Dave Raggett, Dan Connolly, and  
Bill Perry. Hakon should feel free to put this post under the  
style sheet area at the W3O site.  
A W3 style-sheet mailing list will likely be created,  
but Hakon says this will take some days. There should be pointers  
at http://www.w3.org/ in time. In the meantime you can refer to:  
http://gummo.stanford.edu/html/hypermail/archives.html  

If you're not familiar with style sheets, please take  
a look at:  

http://www.w3.org/hypertext/WWW/Style/  

...and take a look at the current proposals and ideas out  
there.  
The format I discuss below is a bit different and somewhat  
improved over Hakon's format as described at the Chicago conference:  

http://www.w3.org/hypertext/WWW/People/howcome/p/cascade.html  

...but from the examples below it should be fairly easy  
to figure things out.  
Finally, a few seconds of silence for the NSFNET backbone  
service, which was decommissioned yesterday, April 30, 1995.  
May it rest in peace,  

-- Kevin  

Thinking About Style Sheets  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Hakon Lie described a simple yet powerful style sheet format in  
Darmstadt, one which he has been working on for some time and had  
been greatly improved since its first rollout in Chicago. Despite  
that fact that it has not been formally described anywhere, I feel  
that the format is superior to the other proposed formats for a number  
of reasons:  

1) It's simple yet flexible.  

Many proposed formats are much too complex for their own  
good - in order to read the simplest formatting commands,  
browsers have to parse ornate structures and operations  
which add little to the overall utility of the format.  
Hakon's format is broken into manageable, logical chunks  
that allow for extensibility and ease of parsing. The  
format is also well-suited to be written in a compact format  
that keeps the style sheet size to a minimum.  

2) It's human readable and writeable.  

The Web no longer belongs to the average Internet-savvy  
technician who writes daily in Perl and C++. In order to  
urge authors and designers to make use of style sheets  
the format must be understandable so that one can "look  
under the hood" when needed. Certainly tools and editors  
will be created that makes style sheet creation easy,  
but the Web will be rewarded a hundredfold by using a  
format for which it is easy to create tools for.  

Below I propose a number of changes and additions to Hakon's format,  
using the time-tested lessons learned by the most popular DTP  
software packages on the market today as well as a little Web design  
experience. I feel that style sheets have the potential to revolutionize  
the way people use and think about the Web and hope that the standards  
process does the right thing both for users and authors.  

Note that many of the examples below will replace the need for using  
tag attributes and nonstandard HTML extensions to create layout. This  
is a good thing, as the layout/style will finally be separated more  
cleanly from the document structure.  

The idea, if I understand correctly, is:  

\<HTML\>  
\<STYLE NOTATION="experimental"\>  

\<!-- The style sheet (or perhaps a pointer to it) goes here. --\>  
\<!-- Here I can define a particular style, or rather, a CLASS. --\>  

P\[mystyle\] : font.size = 12pt  

\</STYLE\>  
\<BODY\>  
Regular structured markup goes here.  

\<P CLASS="mystyle"\>This is a paragraph in my style,  
with a font size of 12 points.  
\</BODY\>  
\</HTML\>  

If you find this pretty easy to understand, well, that's the idea.  
Browsers may very well support DTDs as well as a readable style sheet  
format like Hakon's.  

Linked style sheets will announce themselves to browsers with a MIME  
type such as:  

application/x-perry-stylesheet  
application/x-arena-stylesheet  
application/x-w3c-experimental  
application/x-dsssl-lite  

New Things  
~~~~~~~~~~  

I would say that authors should have the capability to include  
style sheets within the document as well as point to particular  
style sheets (that is, link to them). This makes it easy for a site  
to have a directory of "house" style sheets that apply to all documents  
in a particular web.  

First, there should be a way to specify data that applies to all tags,  
or data that applies to entire style sheet or class of objects. This  
may be done with wildcards:  

BODY : \*.color = red  
doc : \*.color = red  
* : font.size = 12pt  

"doc" is something representing the entire document. Most likely  
this should be replaced with HTML, as one may want different  
properties for HEAD and BODY.  

Apparently Hakon's notation allows for pattern matching on  
sequences of tags, hierarchies of tags, and attribute values.  

HTML : \*.color = red  

In the second example, the font size of 12 points applies to the data in  
all tags. In the first example, the color of all elements in the BODY  
tag are red.  

In regards to sizes and such, there should be a way to specify a  
base size and then allow other styles to be sized relative to the  
base size.  

H1 : font.size += 3pt  
H1 : font.size = P\[mystyle\]:font.size + 10pt  
H1 : font.size = 1.5 * P:font.size  

There should be a full listing and study of the proposed format  
name hierarchy, so that attributes are named in a logical, structured  
manner. More general objects should be on the left, with values becoming  
more specific to the right. So "color.background" should instead  
be "background.color", because background is a general object, while  
color is a specific property.  

Style sheets should support the most common units in DTP; these may be:  

inches (in)  
pixels (px)  
centimeters/millimeters (cm/mm)  
ems (em)  
points (pt)  
characters(ch)  
pica(pi) - "pi" is equal to 12pt  

The characters unit is needed because a designer may wish to specify  
an indent or tab stop by the number of characters in the current font,  
for instance, in a PRE tag.  

All elements that support values should support a value of "default".  
Browsers would come out of the box with a default style sheet preselected  
that the user never need change if they don't want to.  

One possible way to specify tab stops might be:  

PRE : tabs = 8ch, 16ch, 24ch  

Dave Raggett notes that this is discouraged, as HTML 3.0 includes  
a TAB element that does many nice things.  

It would be useful to allow a number of format "shortcuts" so that people  
writing sheets manually don't have to type so much and so that the size  
of the style sheet stays at a minimum. One possible thing to do is to  
use the semicolon (is there a more appropriate character?) for allowing  
the designer to specify multiple values for multiple tags:  

P H1 : font.color = red; blue  
H1 H2 LI : font.size = 1.0; 0.8; 0.2  

In the first example, data in P tags would be red; data in H1 tags  
would be blue.  

In the second, the font size in H1 is the normal default size, H2  
has a font size of 0.8 times that, and LI has a font size of 0.2  
times that.  

Bill Perry suggests another kind of shortcut:  

H1 : font.size=12pt; background.color = red;  
foreground.color = blue  

This is very nice, as authors can deal with a whole tag at a time on  
one line or split things up over multiple lines so the organization  
makes more sense. Again, this is for human usability.  

Another kind of shortcut is to provide shorter versions of  
words:  

"img" for "image"  
"bg" for "background"  

H1 : bg.color = blue  

In specifying color, the designer should be able to enter the  
decimal/percentage values using the RGB color space, since it is more  
understandable and most image processing programs use these values:  

P : font.color = 191, 191, 191  
P : font.color = 0.75, 0.75, 0.75  

The percentages are percents of the range from 0 to 255. In the  
3D world, the range would be from 0 to 1, but we're not dealing with  
that context (yet).  

It would also be nice if one could define style-sheet specific color  
labels so that colors could be referred to by name rather than by number:  

define "My Gray" "191, 191, 191"  
P : font.color = "My Gray"  

The use of such pre-processor-like definitions could be considered a  
good thing, since it again reduces the style sheet size and makes  
readability easier. For colors, as an example, one may be able to  
specify the color name as given in the X11 rgb database.  

It may be useful to allow the designer to provide hints about the  
document dimensions, but it is unclear as to where and how such metadata  
might be specified. Perhaps:  

HTML : width = 600px  
HTML : height = 800px  

Some browsers have a widget that when selected sizes the window to  
the widest non-textual element on the screen. I think the browser  
should resize itself automatically according to these hints if the user  
wants.  

Alignment and Justification  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Alignment should refer to vertical movement, as in the IMG tag, while  
justification should be used to specify horizontal movement. Thus:  

align may be top, middle, bottom, basetop, basemiddle, basebottom  
justify may be left, right, center, full  

IMG : align = basetop  
P : justify = center  

Top, middle, and bottom align elements relative to the size of the  
highest character on the current line, measured from the baseline.  

Dave Raggett comments that HTML has got it wrong in this respect, with  
ALIGN being used for vertical and horizontal alignment. I say that although  
it's probably too late to change things in HTML, let's get it right in  
style sheets.  

Dave also says that one needs bleedleft, bleedright, and bleedboth  
additionally for figures and tables.  

--- ---  
--- | | | |  
--- --- | | | | --- | | \<- font height  
ABC | | | | | | | | | |  
--- | | | | --- --- | | --- \<- baseline  
*| | \___ | | | |*  
--- | | ---  
*| |*  
---  

top mid bot bas bas bas  
top mid bot  

Containers and Spacing  
~~~~~~~~~~~~~~~~~~~~~~  

If browser layout is parsed and understood internally in a container-like  
manner, it should not be difficult to give every element a margin on each  
side, which specifies the amount of space above, below, and to the sides  
of an element. In this way, margins for an entire document as well as  
margins for a particular element can be specified.  

margin values can be top, bottom, left, right  

BODY : margin.top = 10px  
BODY : margin.bottom = 10px  
P : margin.left = 10px  

Regarding "container" properties, one can think of borders and styles,  
backgrounds and foregrounds, and (tiled) images.  

BODY : background.color = red  
BODY : foreground.color = blue  
BODY : background.image = "http://somewhere.com/tile.gif"  
P : border.width = 3px  

border.style may be none, default, line, bevel  

P : border.style = line  
P : border.color = black  
P : border.image = "picture.frame.tile.gif"  

The border may go inside, centered with, or outside the margin boundaries.  
A property name should be suggested for this - somehow "justify" and  
"align" don't seem appropriate.  

Line size could be specified. When applied to a textual element, this  
would indicate leading, the space below a line's baseline. When applied  
to an element like HR, this refers to the height of the element.  

HR : line.size = 3px  
P : line.size = 12pt  

Line indentation can be specified for elements - most designers find  
general indents and indents for the first line to be desireable. So:  

P : line.indent.first = 1in  
P : line.indent = 0.5in  

So the general spacing model for an element would look like:  

------------------ \<- margin.top  
*| blah blah... |*  
*| | \<- line.indent.first*  
*| | \<- line.indent*  
*| | \<- line.size (leading)*  
*| |*  
------------------ \<- margin.bottom  
^ ^  
margin.left margin.right  

In this model, no negative values are allowed.  

Fonts  
~~~~~  

Technically a font has a few basic properties:  

font.family (Helvetica, Lucida)  
font.weight (Plain, Bold, Roman)  
font.angle (Italic, Regular)  
font.variation (Regular, Compressed)  

Although one should be able to completely specify a font with these  
attributes, things can and should be simplified:  

P : font.name = "Helvetica Compressed Bold Italic"  

font.name would allow one to specify a font by its full name as a  
shortcut. However, using the prior four variables may be desired since  
they could provide hints to the browser on how to generate the font  
on the fly.  

Hakon suggested that the order of the words in font.name should not  
matter - the set should be descriptive enough to find a particular  
font.  

Besides font.size and font.color, I would propose font.style, which  
would provide hints to the browser as how to display the given font.  
This style would be generated dynamically (mathematically perhaps),  
based on the given font name or family/weight/angle/variation set.  

font.style may be default, plain, bold, italic, underline,  
double underline, strikethrough, overline, change bar,  
inverse, superscript, subscript, outline, shadow,  
small caps, big caps, lowercase, uppercase  

The case styles would display text as follows:  

plain: "John Smith studies at MIT."  
small caps: "john smith studies at mIT."  
big caps: "John Smith Studies At MIT."  
lowercase: "john smith studies at mit."  
uppercase: "JOHN SMITH STUDIES AT MIT."  

Of course you could specify multiple values where it makes sense:  

P : font.style = subscript & bold  

One desired style is using an all-capitals string with larger  
capitals at the beginning of words. One way to do this may be:  

P : font.style = uppercase & bigcaps  

Much in the same way that one can specify different spacing for the  
first line of an element, one should be able to specify different  
styles for the first letter of a word. This may enable things  
like drop caps.  

P : font.family.dropcap = Times  
P : font.family.firstletter = Times  

The first example changes the first letter in the entire paragraph;  
the second changes the first letter of every word.  

For elements such as drop caps and images there should be a way  
to specify how many lines high the element is compared to the  
normal text, and how that element should be aligned in respect  
to the first line of the element. This is so things like this  
can be made:  

---- blah blah blah  
*| | blah blah blah*  
---- blah blah blah  
blah blah blah blah  

P : font.size.dropcap += 3pt  

The dropcap is 3 points larger than the element's normal  
font size.  

P : font.size.dropcap \*= 4  

The size of the dropcap is four times bigger than the  
size of the normal font size.  

P : align.dropcap = top  
IMG\[dropcap\] : align.dropcap = top  

When a dropcap has an alignment of "top" its top is  
aligned with the first line of the element and drops below  
that line, allowing subsequent lines to wrap around it.  

Perhaps there should be "align.topwrap" and "align.bottomwrap"  
instead?  

Links  
~~~~~  

The designer should be able to specify how links are displayed, but  
users should be able to easily override these specifications, as  
they may be dependent on a particular navigation style. "imglink"  
would refer to a link on an image, while "txtlink" would refer to  
a textual link.  

txtlink.color = blue  
txtlink.color = blue \< AGE / 7d \< "Background Color"  
imglink.color = default  
imglink.size = 3px  
imglink.style may be line, bevel, default, none  
txtlink.style may be underline, double underline,  
inverse, default, none  

Note that in conjunction with AGE links may be colored dynamically.  
If there is a predefined value for the background color, links can  
disappear with time and yet still be active, as in the example above.  

There should be a way to specify new versus followed links, but I  
have no good ideas as how to do so.  

Based on something Hakon said:  

HTML : txtlink.color = red \< FETCHED \< blue  

Style Sheet Permissions  
~~~~~~~~~~~~~~~~~~~~~~~  

There should be an extra layer in the current permissions model:  

\<- author insist | legal  
user insist  
author important  
user important  
author normal  
user normal  
application default  

"author insist" would first display the page as the author intended  
but allow the user to change the style (for example, via a pop-up  
"styles" menu in the browser) after the initial download. The "legal"  
specification would display as the author intended and not allow the  
user to make changes. If the particular style in a browser was not  
supported for a legal document, then the document would not display  
at all.  

P : font.size = 12pt ! legal  

Authors writing legal documents may first display a readable  
disclaimer which tells the user what they must to do read the legal  
document (get a particular browser, enter a password, use a particular  
style, etc.), why it is legal (display copyrights, licensing restrictions,  
etc.) and then link to the document in question.  

Bill Perry suggests that it is more reasonable is for a dialog box to appear:  

"Your settings will override something flagged as 'legal'.  
Should I continue?"  

\[ OK \] \[ Cancel \]  

All of this was a heavily debated issue at Darmstadt and will continue  
to be for some time. One thing to consider is that in the real world,  
given enough time and resources, you can copy anything man-made reasonably  
well and change it to suit your worldview. People will always want  
to do this to things. Now that it can be very easy for them to do so,  
why not let them?  

Users should be able to specify particular styles they do not want  
to see, for instance, a color blind user may never want to see any  
elements colored red or blue and instead would wish them to be colored  
black. Dave Raggett notes that authors, not users, should be the ones to  
anticipate this type of need in the style sheet. Forcing the browser  
to deal with it would lead to crude results.  

Note that the issues of filtering out particular styles and filtering  
out particular media/textual content are somewhat different and discussions  
regarding them should keep this in mind. The style area of a document  
should hold information related to layout and "look and feel". Within  
the content area, things that more closely affect the textual/media  
content should be put there, such as ways to represent different languages  
and multiple versions of the same content (animations, ISMAP things, etc.).  
Care should be taken not to confuse the two.  

Dave Raggett suggests that the style sheet should have a way to insert  
particular elements into the document so that one could create  
templates, page headers and footers, etc. easily.  

Perhaps:  

HEADER : insert "page.header.html"  
FOOTER : insert "page.footer.html"  

This inserts HTML within tags, while:  

H1 : insert.before "\<IMG SRC=\\"rule.gif\\"\>"  
H1 : insert.after "some text"  

adds text/HTML before or after elements.  

Industrial-Strength Stuff  
~~~~~~~~~~~~~~~~~~~~~~~~~  

Here are some features which may also be added, in no particular order:  

Kerning  
Tracking (spaces between words, letters, spacing limits)  
The ability to specify particular color models  
Columns (auto flow, positioning)  
Ways to allow elements to float over others  
Different ways of word wrapping  
Ways to handle breaks in text  
Page numbering  
Numbering of list elements  
Transitions (a la a presentation program)  
Control over text flow  

Finally...  
~~~~~~~~~~  

We should be seeing a full description of the experimental notation,  
with proposals, etc. and a full listing of properties, values,  
meta-variables, etc. on the Web pretty soon.  

The above examples and features cover a good deal of the general  
functionality of the most common DTP programs and remove the need  
for many of the nonstandard HTML extensions that are seen today.  
With a powerful style sheet language, the Web will certainly take on  
a new level of sophistication and many of the problems authors have  
had in the past regarding online design and layout will be solved.  
Dave Raggett notes that notation that is meant to be read and used by  
humans must have the right level of abstraction so that users and authors  
can easily control style.  

The web's incorporation of style sheets will certainly raise the  
level of expectation higher: users will not just expect to see a few  
blocks of text but will soon expect the features previously only available  
in "real" desktop publishing programs. Because of this it's quite  
possible that developers will force themselves to look at parsing  
and browser architecture in a more object-oriented manner, giving  
way to a more modular, container-based approach. Arena and Bert Bos's  
work are starting points, but once you add in OpenDoc or OLE,  
COBRA, HotJava, etc. you can begin to see how powerful that  
approach can be.  

People will finally think more about the structure of information  
and more about making it universal and interchangable. And hopefully  
we can finally see some serious content on the Internet.  

Arena is currently implementing Hakon's style sheet format, as is  
Bill Perry - what is needed is a lot more discussion and real code  
out there before a proposal can be presented to an IETF working group.  
Hakon says that a W3C workshop on stylesheets is planned for September;  
hopefully we'll see some good things happen before then.  

```
--
Kevin Hughes * kevinh@eit.com
Enterprise Integration Technologies Webmaster (http://www.eit.com/)
Hypermedia Industrial Designer * Duty now for the future!

```

---

* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"

WWW-HTML Apr-Jun 1995: Thinking about style sheets 

# Thinking about style sheets

Kevin Hughes (*kevinh@eit.COM*)  
*Tue, 2 May 1995 04:07:10 +0500* * **Messages sorted by:** \[ date \]\[ thread \]\[ subject \]\[ author \]
* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"

---

This post is a long one; I would encourage replies to be  
broken up into manageable subjects such as fonts, etc.  
The following thoughts revolve around creating a simple  
yet powerful style sheet language for the World-Wide Web. I have  
incorporated comments here and there from a previous email  
discussion involving Hakon Lie, Dave Raggett, Dan Connolly, and  
Bill Perry. Hakon should feel free to put this post under the  
style sheet area at the W3O site.  
A W3 style-sheet mailing list will likely be created,  
but Hakon says this will take some days. There should be pointers  
at http://www.w3.org/ in time. In the meantime you can refer to:  
http://gummo.stanford.edu/html/hypermail/archives.html  

If you're not familiar with style sheets, please take  
a look at:  

http://www.w3.org/hypertext/WWW/Style/  

...and take a look at the current proposals and ideas out  
there.  
The format I discuss below is a bit different and somewhat  
improved over Hakon's format as described at the Chicago conference:  

http://www.w3.org/hypertext/WWW/People/howcome/p/cascade.html  

...but from the examples below it should be fairly easy  
to figure things out.  
Finally, a few seconds of silence for the NSFNET backbone  
service, which was decommissioned yesterday, April 30, 1995.  
May it rest in peace,  

-- Kevin  

Thinking About Style Sheets  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Hakon Lie described a simple yet powerful style sheet format in  
Darmstadt, one which he has been working on for some time and had  
been greatly improved since its first rollout in Chicago. Despite  
that fact that it has not been formally described anywhere, I feel  
that the format is superior to the other proposed formats for a number  
of reasons:  

1) It's simple yet flexible.  

Many proposed formats are much too complex for their own  
good - in order to read the simplest formatting commands,  
browsers have to parse ornate structures and operations  
which add little to the overall utility of the format.  
Hakon's format is broken into manageable, logical chunks  
that allow for extensibility and ease of parsing. The  
format is also well-suited to be written in a compact format  
that keeps the style sheet size to a minimum.  

2) It's human readable and writeable.  

The Web no longer belongs to the average Internet-savvy  
technician who writes daily in Perl and C++. In order to  
urge authors and designers to make use of style sheets  
the format must be understandable so that one can "look  
under the hood" when needed. Certainly tools and editors  
will be created that makes style sheet creation easy,  
but the Web will be rewarded a hundredfold by using a  
format for which it is easy to create tools for.  

Below I propose a number of changes and additions to Hakon's format,  
using the time-tested lessons learned by the most popular DTP  
software packages on the market today as well as a little Web design  
experience. I feel that style sheets have the potential to revolutionize  
the way people use and think about the Web and hope that the standards  
process does the right thing both for users and authors.  

Note that many of the examples below will replace the need for using  
tag attributes and nonstandard HTML extensions to create layout. This  
is a good thing, as the layout/style will finally be separated more  
cleanly from the document structure.  

The idea, if I understand correctly, is:  

\<HTML\>  
\<STYLE NOTATION="experimental"\>  

\<!-- The style sheet (or perhaps a pointer to it) goes here. --\>  
\<!-- Here I can define a particular style, or rather, a CLASS. --\>  

P\[mystyle\] : font.size = 12pt  

\</STYLE\>  
\<BODY\>  
Regular structured markup goes here.  

\<P CLASS="mystyle"\>This is a paragraph in my style,  
with a font size of 12 points.  
\</BODY\>  
\</HTML\>  

If you find this pretty easy to understand, well, that's the idea.  
Browsers may very well support DTDs as well as a readable style sheet  
format like Hakon's.  

Linked style sheets will announce themselves to browsers with a MIME  
type such as:  

application/x-perry-stylesheet  
application/x-arena-stylesheet  
application/x-w3c-experimental  
application/x-dsssl-lite  

New Things  
~~~~~~~~~~  

I would say that authors should have the capability to include  
style sheets within the document as well as point to particular  
style sheets (that is, link to them). This makes it easy for a site  
to have a directory of "house" style sheets that apply to all documents  
in a particular web.  

First, there should be a way to specify data that applies to all tags,  
or data that applies to entire style sheet or class of objects. This  
may be done with wildcards:  

BODY : \*.color = red  
doc : \*.color = red  
* : font.size = 12pt  

"doc" is something representing the entire document. Most likely  
this should be replaced with HTML, as one may want different  
properties for HEAD and BODY.  

Apparently Hakon's notation allows for pattern matching on  
sequences of tags, hierarchies of tags, and attribute values.  

HTML : \*.color = red  

In the second example, the font size of 12 points applies to the data in  
all tags. In the first example, the color of all elements in the BODY  
tag are red.  

In regards to sizes and such, there should be a way to specify a  
base size and then allow other styles to be sized relative to the  
base size.  

H1 : font.size += 3pt  
H1 : font.size = P\[mystyle\]:font.size + 10pt  
H1 : font.size = 1.5 * P:font.size  

There should be a full listing and study of the proposed format  
name hierarchy, so that attributes are named in a logical, structured  
manner. More general objects should be on the left, with values becoming  
more specific to the right. So "color.background" should instead  
be "background.color", because background is a general object, while  
color is a specific property.  

Style sheets should support the most common units in DTP; these may be:  

inches (in)  
pixels (px)  
centimeters/millimeters (cm/mm)  
ems (em)  
points (pt)  
characters(ch)  
pica(pi) - "pi" is equal to 12pt  

The characters unit is needed because a designer may wish to specify  
an indent or tab stop by the number of characters in the current font,  
for instance, in a PRE tag.  

All elements that support values should support a value of "default".  
Browsers would come out of the box with a default style sheet preselected  
that the user never need change if they don't want to.  

One possible way to specify tab stops might be:  

PRE : tabs = 8ch, 16ch, 24ch  

Dave Raggett notes that this is discouraged, as HTML 3.0 includes  
a TAB element that does many nice things.  

It would be useful to allow a number of format "shortcuts" so that people  
writing sheets manually don't have to type so much and so that the size  
of the style sheet stays at a minimum. One possible thing to do is to  
use the semicolon (is there a more appropriate character?) for allowing  
the designer to specify multiple values for multiple tags:  

P H1 : font.color = red; blue  
H1 H2 LI : font.size = 1.0; 0.8; 0.2  

In the first example, data in P tags would be red; data in H1 tags  
would be blue.  

In the second, the font size in H1 is the normal default size, H2  
has a font size of 0.8 times that, and LI has a font size of 0.2  
times that.  

Bill Perry suggests another kind of shortcut:  

H1 : font.size=12pt; background.color = red;  
foreground.color = blue  

This is very nice, as authors can deal with a whole tag at a time on  
one line or split things up over multiple lines so the organization  
makes more sense. Again, this is for human usability.  

Another kind of shortcut is to provide shorter versions of  
words:  

"img" for "image"  
"bg" for "background"  

H1 : bg.color = blue  

In specifying color, the designer should be able to enter the  
decimal/percentage values using the RGB color space, since it is more  
understandable and most image processing programs use these values:  

P : font.color = 191, 191, 191  
P : font.color = 0.75, 0.75, 0.75  

The percentages are percents of the range from 0 to 255. In the  
3D world, the range would be from 0 to 1, but we're not dealing with  
that context (yet).  

It would also be nice if one could define style-sheet specific color  
labels so that colors could be referred to by name rather than by number:  

define "My Gray" "191, 191, 191"  
P : font.color = "My Gray"  

The use of such pre-processor-like definitions could be considered a  
good thing, since it again reduces the style sheet size and makes  
readability easier. For colors, as an example, one may be able to  
specify the color name as given in the X11 rgb database.  

It may be useful to allow the designer to provide hints about the  
document dimensions, but it is unclear as to where and how such metadata  
might be specified. Perhaps:  

HTML : width = 600px  
HTML : height = 800px  

Some browsers have a widget that when selected sizes the window to  
the widest non-textual element on the screen. I think the browser  
should resize itself automatically according to these hints if the user  
wants.  

Alignment and Justification  
~~~~~~~~~~~~~~~~~~~~~~~~~~~  

Alignment should refer to vertical movement, as in the IMG tag, while  
justification should be used to specify horizontal movement. Thus:  

align may be top, middle, bottom, basetop, basemiddle, basebottom  
justify may be left, right, center, full  

IMG : align = basetop  
P : justify = center  

Top, middle, and bottom align elements relative to the size of the  
highest character on the current line, measured from the baseline.  

Dave Raggett comments that HTML has got it wrong in this respect, with  
ALIGN being used for vertical and horizontal alignment. I say that although  
it's probably too late to change things in HTML, let's get it right in  
style sheets.  

Dave also says that one needs bleedleft, bleedright, and bleedboth  
additionally for figures and tables.  

--- ---  
--- | | | |  
--- --- | | | | --- | | \<- font height  
ABC | | | | | | | | | |  
--- | | | | --- --- | | --- \<- baseline  
*| | \___ | | | |*  
--- | | ---  
*| |*  
---  

top mid bot bas bas bas  
top mid bot  

Containers and Spacing  
~~~~~~~~~~~~~~~~~~~~~~  

If browser layout is parsed and understood internally in a container-like  
manner, it should not be difficult to give every element a margin on each  
side, which specifies the amount of space above, below, and to the sides  
of an element. In this way, margins for an entire document as well as  
margins for a particular element can be specified.  

margin values can be top, bottom, left, right  

BODY : margin.top = 10px  
BODY : margin.bottom = 10px  
P : margin.left = 10px  

Regarding "container" properties, one can think of borders and styles,  
backgrounds and foregrounds, and (tiled) images.  

BODY : background.color = red  
BODY : foreground.color = blue  
BODY : background.image = "http://somewhere.com/tile.gif"  
P : border.width = 3px  

border.style may be none, default, line, bevel  

P : border.style = line  
P : border.color = black  
P : border.image = "picture.frame.tile.gif"  

The border may go inside, centered with, or outside the margin boundaries.  
A property name should be suggested for this - somehow "justify" and  
"align" don't seem appropriate.  

Line size could be specified. When applied to a textual element, this  
would indicate leading, the space below a line's baseline. When applied  
to an element like HR, this refers to the height of the element.  

HR : line.size = 3px  
P : line.size = 12pt  

Line indentation can be specified for elements - most designers find  
general indents and indents for the first line to be desireable. So:  

P : line.indent.first = 1in  
P : line.indent = 0.5in  

So the general spacing model for an element would look like:  

------------------ \<- margin.top  
*| blah blah... |*  
*| | \<- line.indent.first*  
*| | \<- line.indent*  
*| | \<- line.size (leading)*  
*| |*  
------------------ \<- margin.bottom  
^ ^  
margin.left margin.right  

In this model, no negative values are allowed.  

Fonts  
~~~~~  

Technically a font has a few basic properties:  

font.family (Helvetica, Lucida)  
font.weight (Plain, Bold, Roman)  
font.angle (Italic, Regular)  
font.variation (Regular, Compressed)  

Although one should be able to completely specify a font with these  
attributes, things can and should be simplified:  

P : font.name = "Helvetica Compressed Bold Italic"  

font.name would allow one to specify a font by its full name as a  
shortcut. However, using the prior four variables may be desired since  
they could provide hints to the browser on how to generate the font  
on the fly.  

Hakon suggested that the order of the words in font.name should not  
matter - the set should be descriptive enough to find a particular  
font.  

Besides font.size and font.color, I would propose font.style, which  
would provide hints to the browser as how to display the given font.  
This style would be generated dynamically (mathematically perhaps),  
based on the given font name or family/weight/angle/variation set.  

font.style may be default, plain, bold, italic, underline,  
double underline, strikethrough, overline, change bar,  
inverse, superscript, subscript, outline, shadow,  
small caps, big caps, lowercase, uppercase  

The case styles would display text as follows:  

plain: "John Smith studies at MIT."  
small caps: "john smith studies at mIT."  
big caps: "John Smith Studies At MIT."  
lowercase: "john smith studies at mit."  
uppercase: "JOHN SMITH STUDIES AT MIT."  

Of course you could specify multiple values where it makes sense:  

P : font.style = subscript & bold  

One desired style is using an all-capitals string with larger  
capitals at the beginning of words. One way to do this may be:  

P : font.style = uppercase & bigcaps  

Much in the same way that one can specify different spacing for the  
first line of an element, one should be able to specify different  
styles for the first letter of a word. This may enable things  
like drop caps.  

P : font.family.dropcap = Times  
P : font.family.firstletter = Times  

The first example changes the first letter in the entire paragraph;  
the second changes the first letter of every word.  

For elements such as drop caps and images there should be a way  
to specify how many lines high the element is compared to the  
normal text, and how that element should be aligned in respect  
to the first line of the element. This is so things like this  
can be made:  

---- blah blah blah  
*| | blah blah blah*  
---- blah blah blah  
blah blah blah blah  

P : font.size.dropcap += 3pt  

The dropcap is 3 points larger than the element's normal  
font size.  

P : font.size.dropcap \*= 4  

The size of the dropcap is four times bigger than the  
size of the normal font size.  

P : align.dropcap = top  
IMG\[dropcap\] : align.dropcap = top  

When a dropcap has an alignment of "top" its top is  
aligned with the first line of the element and drops below  
that line, allowing subsequent lines to wrap around it.  

Perhaps there should be "align.topwrap" and "align.bottomwrap"  
instead?  

Links  
~~~~~  

The designer should be able to specify how links are displayed, but  
users should be able to easily override these specifications, as  
they may be dependent on a particular navigation style. "imglink"  
would refer to a link on an image, while "txtlink" would refer to  
a textual link.  

txtlink.color = blue  
txtlink.color = blue \< AGE / 7d \< "Background Color"  
imglink.color = default  
imglink.size = 3px  
imglink.style may be line, bevel, default, none  
txtlink.style may be underline, double underline,  
inverse, default, none  

Note that in conjunction with AGE links may be colored dynamically.  
If there is a predefined value for the background color, links can  
disappear with time and yet still be active, as in the example above.  

There should be a way to specify new versus followed links, but I  
have no good ideas as how to do so.  

Based on something Hakon said:  

HTML : txtlink.color = red \< FETCHED \< blue  

Style Sheet Permissions  
~~~~~~~~~~~~~~~~~~~~~~~  

There should be an extra layer in the current permissions model:  

\<- author insist | legal  
user insist  
author important  
user important  
author normal  
user normal  
application default  

"author insist" would first display the page as the author intended  
but allow the user to change the style (for example, via a pop-up  
"styles" menu in the browser) after the initial download. The "legal"  
specification would display as the author intended and not allow the  
user to make changes. If the particular style in a browser was not  
supported for a legal document, then the document would not display  
at all.  

P : font.size = 12pt ! legal  

Authors writing legal documents may first display a readable  
disclaimer which tells the user what they must to do read the legal  
document (get a particular browser, enter a password, use a particular  
style, etc.), why it is legal (display copyrights, licensing restrictions,  
etc.) and then link to the document in question.  

Bill Perry suggests that it is more reasonable is for a dialog box to appear:  

"Your settings will override something flagged as 'legal'.  
Should I continue?"  

\[ OK \] \[ Cancel \]  

All of this was a heavily debated issue at Darmstadt and will continue  
to be for some time. One thing to consider is that in the real world,  
given enough time and resources, you can copy anything man-made reasonably  
well and change it to suit your worldview. People will always want  
to do this to things. Now that it can be very easy for them to do so,  
why not let them?  

Users should be able to specify particular styles they do not want  
to see, for instance, a color blind user may never want to see any  
elements colored red or blue and instead would wish them to be colored  
black. Dave Raggett notes that authors, not users, should be the ones to  
anticipate this type of need in the style sheet. Forcing the browser  
to deal with it would lead to crude results.  

Note that the issues of filtering out particular styles and filtering  
out particular media/textual content are somewhat different and discussions  
regarding them should keep this in mind. The style area of a document  
should hold information related to layout and "look and feel". Within  
the content area, things that more closely affect the textual/media  
content should be put there, such as ways to represent different languages  
and multiple versions of the same content (animations, ISMAP things, etc.).  
Care should be taken not to confuse the two.  

Dave Raggett suggests that the style sheet should have a way to insert  
particular elements into the document so that one could create  
templates, page headers and footers, etc. easily.  

Perhaps:  

HEADER : insert "page.header.html"  
FOOTER : insert "page.footer.html"  

This inserts HTML within tags, while:  

H1 : insert.before "\<IMG SRC=\\"rule.gif\\"\>"  
H1 : insert.after "some text"  

adds text/HTML before or after elements.  

Industrial-Strength Stuff  
~~~~~~~~~~~~~~~~~~~~~~~~~  

Here are some features which may also be added, in no particular order:  

Kerning  
Tracking (spaces between words, letters, spacing limits)  
The ability to specify particular color models  
Columns (auto flow, positioning)  
Ways to allow elements to float over others  
Different ways of word wrapping  
Ways to handle breaks in text  
Page numbering  
Numbering of list elements  
Transitions (a la a presentation program)  
Control over text flow  

Finally...  
~~~~~~~~~~  

We should be seeing a full description of the experimental notation,  
with proposals, etc. and a full listing of properties, values,  
meta-variables, etc. on the Web pretty soon.  

The above examples and features cover a good deal of the general  
functionality of the most common DTP programs and remove the need  
for many of the nonstandard HTML extensions that are seen today.  
With a powerful style sheet language, the Web will certainly take on  
a new level of sophistication and many of the problems authors have  
had in the past regarding online design and layout will be solved.  
Dave Raggett notes that notation that is meant to be read and used by  
humans must have the right level of abstraction so that users and authors  
can easily control style.  

The web's incorporation of style sheets will certainly raise the  
level of expectation higher: users will not just expect to see a few  
blocks of text but will soon expect the features previously only available  
in "real" desktop publishing programs. Because of this it's quite  
possible that developers will force themselves to look at parsing  
and browser architecture in a more object-oriented manner, giving  
way to a more modular, container-based approach. Arena and Bert Bos's  
work are starting points, but once you add in OpenDoc or OLE,  
COBRA, HotJava, etc. you can begin to see how powerful that  
approach can be.  

People will finally think more about the structure of information  
and more about making it universal and interchangable. And hopefully  
we can finally see some serious content on the Internet.  

Arena is currently implementing Hakon's style sheet format, as is  
Bill Perry - what is needed is a lot more discussion and real code  
out there before a proposal can be presented to an IETF working group.  
Hakon says that a W3C workshop on stylesheets is planned for September;  
hopefully we'll see some good things happen before then.  

```
--
Kevin Hughes * kevinh@eit.com
Enterprise Integration Technologies Webmaster (http://www.eit.com/)
Hypermedia Industrial Designer * Duty now for the future!

```

---

* **Next message:** Robert A. Mesa: "Print Form Code"
* **Previous message:** H&kon W Lie: "Parsability (re: Thinking about style sheets)"
