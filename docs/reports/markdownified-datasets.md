# Dataset Card: Markdownified Datasets

This dataset collection consists of several large-scale text corpora that have been processed and tokenized through a custom fork of Resiliparse. The datasets are primarily intended for language model training and research purposes.

## Dataset Statistics

| Dataset | Token Count | Approximate Size | Source |
|---------|------------|------------------|--------|
| Wiki | 8587224558 | 8.59B tokens | [Wikipedia Dump](https://dumps.wikimedia.org/other/enterprise_html/runs/20241201/enwiki-NS0-20241201-ENTERPRISE-HTML.json.tar.gz) |
| Ar5iv No Problem | 2742463924 | 2.74B tokens | [Ar5iv Dump](https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/) |
| Ar5iv Warning | 19552307274 | 19.6B tokens | [Ar5iv Dump](https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/) |
| Stack Exchange | 20413785853 | 20.4B tokens | [Stack Exchange Dump](https://archive.org/details/stackexchange) |

## Processing Methodology

These datasets were processed using our [custom fork of Resiliparse](https://github.com/stanford-crfm/chatnoir-resiliparse) which simplified the raw HTML DOM, the simplified DOM was then processed by our [custom implementation](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/markdown/markdown.py#L145-L650) of [Markdownify](https://github.com/matthewwithanm/python-markdownify). The exact modifications and enhancements made to the [original Resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse) are documented in the next section, the processing pipeline appears to have:

1. Extracted text from various sources (Wikipedia, Ar5iv, Stack Exchange)
2. Simplified the raw HTML DOM through our custom fork of Resiliparse
3. Process the simplified DOM with our custom Markdownify implementation to covert DOM to Markdown.

## Heuristic Filters

The markdownification pipeline applies specific heuristic filters to each dataset to clean and improve the quality of the markdown content. These filters are designed to remove noise, preserve valuable content, and ensure consistent formatting.

### Wikipedia

The Wikipedia preprocessing pipeline includes several heuristic filters:

- **Blacklisted Selectors Removal**: Removes specific DOM elements like navigation bars, footer elements, reference sections, and edit buttons using CSS selectors (e.g., `div.navbox`, `span.mw-editsection`, `div#catlinks`).
- **External Links Removal**: Automatically removes "External Links" sections and their content.
- **Dense Link Removal**: Identifies and removes link clusters which are sections that are primarily composed of links (>80% links).
- **Numerical and Character Thresholds**: Filters out content with excessive digit percentages (>50%), insufficient word counts (<70 words), or excessive special character percentages (>50%).
- **Reference Section Removal**: Removes references sections to reduce noise while preserving informative content.
- **Table Formatting**: Converts HTML tables to markdown tables while preserving structure and removing empty rows/columns.
- **Main Content Extraction**: Leverages Resiliparse to extract the main content and avoid boilerplate elements.

### Ar5iv

The Ar5iv dataset uses specialized filters for academic papers:

- **Abstract Transformation**: Converts abstract sections into proper headings for better structure.
- **Metadata Removal**: Removes author information, title pages, and article metadata to reduce noise.
- **List Cleaning**: Removes duplicate numbering patterns that occur when LaTeX numbering combines with HTML list markers (e.g., "1. 1.").
- **Academic Elements Removal**: Removes bibliography sections, footnotes, and citation links that don't add value to the main content.
- **Equation Formatting**: Transforms equation tables into inline elements for better markdown conversion and preserves LaTeX notation.
- **Footer Removal**: Removes the ar5iv-specific footer information.
- **Figure Caption Removal**: Removes figure captions to reduce noise.
- **Code Preservation**: Converts code listing lines to proper newlines to preserve code formatting.
- **Whitespace Normalization**: Standardizes whitespace and newlines.

### Stack Exchange

The Stack Exchange dataset applies several specific filters:

- **Q&A Structure Preservation**: Special processing to maintain the question-answer structure of posts.
- **Markup Formatting**: Proper conversion of Stack Exchange markup to markdown, preserving code blocks, lists, and other formatting.
- **Separator Addition**: Adds separators between questions and answers for better readability.
- **Main Content Preprocessing**: Special handling for Stack Exchange specific elements like "qa-main" class, question headers, and post content.


## Usage Notes

When using these datasets, people should be aware of:
- The potential differences in quality between the "No Problem" and "Warning" subsets of Ar5iv.
- These are Markdownified version of the raw dataset and have not been filtered for quality.

## Resiliparse Custom Fork

This fork extends the original **Resiliparse** HTML-to-text extractor with a new helper, `extract_simplified_dom`, that yields a _cleaned & simplified_ HTML snippet instead of plain text.  The goal is to preserve **minimal HTML structure** (e.g. headings, paragraphs, links, lists) while still removing boiler-plate, scripts, tracking pixels, etc.

### Behaviour at a Glance

| Mode                | Original `extract_plain_text` | New `extract_simplified_dom` |
|---------------------|--------------------------------|------------------------------|
|Output               |Plain text only                |Simplified HTML               |
|Preserves `<p><h1>`… |Optional via `minimal_html`     |Always (unless filtered)      |
|Whitespace handling  |Collapses to single space/`\n`  |Follows DOM tree indentation  |
|Link handling        |`href` rendered as text         |Anchor kept as `<a>`          |
|List handling        |Bullets/indices converted to •  |`<ul>/<ol>` retained          |
|Boiler-plate removal |Yes                             |Yes                           |

### Implementation Details

The core enhancement is the implementation of `extract_simplified_dom` which leverages [Lexbor's](https://github.com/lexbor/lexbor) DOM serialization capabilities to maintain HTML structure while still applying the filtering and content extraction logic from the original `extract_plain_text` function.

1. **DOM Serialization**:
      - Added `serialize_node` function that converts DOM nodes to their HTML string representation
      - Utilizes Lexbor's `lxb_html_serialize_tree_str` to create a faithful representation of the DOM structure

2. **Modified Extraction Logic**:
      - Preserves important semantic HTML tags rather than stripping all markup
      - Follows the same element filtering rules as `extract_plain_text` (skipping script, style, etc.)
      - Maintains structural relationships between elements

3. **Configuration Options**:
      - Maintains the same parameter interface as `extract_plain_text` for consistency
      - Allows for the same customization of content extraction (links, alt texts, form fields, etc.)

This enhancement maintains full backward compatibility with the existing Resiliparse API while extending its capabilities for applications that need more structure than plain text extraction provides.

## Examples

We have several "snapshot tests" as quality control. You can see some of them in our GitHub repo:

- [Ar5iv](https://github.com/marin-community/marin/blob/main/tests/snapshots/ar5iv/expected/arxiv_4.md)
- [Stack Exchange](https://github.com/marin-community/marin/blob/main/tests/snapshots/stackexchange/expected/stackoverflow.md)
- [Wikipedia](https://github.com/marin-community/marin/blob/main/tests/snapshots/wiki/expected/aquila.md)

While the conversion is by no means perfect, we believe the datasets are of high quality and a useful resource for the community.

### ### Acknowledgements

We would like to express our sincere gratitude to:

- **Janek Bevendorff** for creating the original Resiliparse project
- The **Arxiv Labs** and **KWARC** teams for their meticulous work in curating the Ar5iv dataset
- **Matthew Dapena-Tretter** for developing the original Markdownify Project

Their contributions have been really important in making this work possible.
