# This folder contains script for the example server and how to use it

1. Make a jsonl file with the examples you want to serve. The format of jsonl file is discussed below.
2. Save the jsonl file at location `gcs://marin-data/examples/{filename}.jsonl`.
3. You can see the example in ith line by visiting `http://34.66.160.95/content/{filename}/{i}`.
4. For example see: http://34.66.160.95/content/fineweb/0, gcs://marin-data/examples/fineweb.jsonl

## Note
For each request I read through the lines of the jsonl file, which means a query for 1000th line would be slower than that of 1st line.

## JSONL format
Each line of the jsonl file should be a json object with the following structure:

```json
{
    "id": "unique_id",
    "content": [
        {
            "title": "Title of the content",
            "type": "text",
            "text": "This is plain text content."
        },
        {
            "title": "Title of the content",
            "type": "html",
            "text": "<i>This is italic HTML content.</i>"
        },
        {
            "title": "Title of the content",
            "type": "md",
            "text": "## This is a markdown subheader"
        }
    ]
}
```
1. `title` is the title that shows on the dropdown menu
2. `type` determines how to render the content. Currently only `text`, `html` and `md` are supported.
3. `text` is content to be rendered. For `text` it's plain text, for `html` and `md` it's rendered html and markdown.