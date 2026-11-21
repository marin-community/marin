You are a financial agent. Today is April 07, 2025. You are given a question and you need to answer it using the tools provided.
You can only use the built-in and provided tools. The provided tools are:
1. google_web_search

Search the web for information using Google Search. Results are limited to content before April 7, 2025.

**Usage:**
```bash
python3 /app/tools/google_search.py --search-query "your query here"
```

**Required Arguments:**
- `--search-query`: The query to search for

**Note:** Returns up to 10 results by default. All results are from dates before April 7, 2025.

---

2. edgar_search

Search the EDGAR Database through the SEC API. You should provide a query, a list of form types, a list of CIKs, a start date, an end date, a page number, and a top N results. The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.

**Usage:**
```bash
python3 /app/tools/edgar_search.py   --query "search keywords"   --form-types '["10-K", "10-Q"]'   --ciks '["0000320193"]'   --start-date "2024-01-01"   --end-date "2024-12-31"   --page "1"   --top-n-results 10
```

**Required Arguments (ALL 7 are required):**
- `--query`: The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'
- `--form-types`: Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types). Use `'[]'` for all form types.
- `--ciks`: Filters results to filings by specified CIKs, type list of strings. Default is None (all filers). Use `'[]'` for all filers.
- `--start-date`: Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago.
- `--end-date`: End date for the search range, in the same format as startDate. Default is today. Cannot exceed April 7, 2025.
- `--page`: Pagination for results. Default is '1'
- `--top-n-results`: The top N results to return after the query. Useful if you are not sure the result you are looking for is ranked first after your query.

**Important:** The results contain only metadata (filing information), not the full text of the filings. To get the full text, you need to use `parse_html_page` on the filing URLs.

---

3. parse_html_page

Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues. You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure. The data structure is a dictionary.

**Usage:**
```bash
python3 /app/tools/parse_html.py   --url "https://example.com/page.html"   --key "my_key_name"
```

**Required Arguments (BOTH required):**
- `--url`: The URL of the HTML page to parse
- `--key`: The key to use when saving the result in the conversation's data structure (dict)

**Example:**
```bash
python3 /app/tools/parse_html.py   --url "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193"   --key "apple_sec_page"
```

**Note:**
- Content is saved to `/app/data_storage.json` and can be retrieved using the `retrieve_information` tool.
- The tool removes script and style elements from the HTML before extracting text.
- Beware of rate limiting. If a URL is blocked or the server takes too long, it will timeout.

---

4. retrieve_information

Retrieve information from the conversation's data structure (dict) and allow character range extraction.

**IMPORTANT:** Your prompt MUST include at least one key from the data storage using the exact format: {key_name}

For example, if you want to analyze data stored under the key "financial_report", your prompt should look like:
"Analyze the following financial report and extract the revenue figures: {financial_report}"

The {key_name} will be replaced with the actual content stored under that key before being sent to the LLM.
If you don't use this exact format with double braces, the tool will fail to retrieve the information.

You can optionally specify character ranges for each document key to extract only portions of documents. That can be useful to avoid token limit errors or improve efficiency by selecting only part of the document.
For example, if "financial_report" contains "Annual Report 2023" and you specify a range [1, 5] for that key,
only "nnual" will be inserted into the prompt.

The output is the result from the LLM that receives the prompt with the inserted data.

**Usage:**
```bash
python3 /app/tools/retrieve_information.py   --prompt "Analyze this content: {key_name}"   --input-character-ranges '{}'
```

**Required Arguments:**
- `--prompt`: The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {key_name} - for example: 'Summarize this 10-K filing: {company_10k}'. The content stored under each key will replace the {key_name} placeholder.

**Optional Arguments:**
- `--input-character-ranges`: A dictionary mapping document keys to their character ranges. Each range should be an array where the first element is the start index and the second element is the end index. Can be used to only read portions of documents. By default, the full document is used. To use the full document, set the range to an empty list []. Example: `'{"key1": [0, 1000]}'` or `'{}'` for full content.

---

## Data Storage

- Parsed HTML content is stored in `/app/data_storage.json`
- You can view stored keys: `cat /app/data_storage.json | jq 'keys'`
- You can access stored content: `cat /app/data_storage.json | jq '.key_name' -r`


```bash
echo "Your answer here" > /app/answer.txt
```


You may not interract with the user.
When you have the answer, you should respond with 'FINAL ANSWER:' followed by your answer.
At the end of your answer, you should provide your sources in a dictionary with the following format:
{
    "sources": [
        {
            "url": "https://example.com",
            "name": "Name of the source"
        },
        ...
    ]
}

Question:
Calculate the 3 year revenue CAGR for Palantir Technologies from 2021 to 2024.
## Expected Output

**CRITICAL:** Write your final answer to `/app/answer.txt`

